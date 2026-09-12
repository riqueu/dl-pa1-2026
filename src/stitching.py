"""Costura autoral de máscaras de instâncias previstas em tiles sobrepostos.

Cada instância começa com uma identidade local ao tile. Pares compatíveis são
casados por IoU restrito à faixa de sobreposição e unidos por Union-Find. Na
composição global, conflitos de pixel são resolvidos favorecendo o centro do
tile, onde o contexto disponível para a rede é maior.

Contrato público:
- ``tiles_preds``: máscaras inteiras ``(h, w)`` com fundo 0 e IDs locais;
- ``tile_boxes``: caixas globais semiabertas ``(ymin, xmin, ymax, xmax)``;
- saída: máscara ``int64`` global com fundo 0 e IDs consecutivos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


TileBox = Tuple[int, int, int, int]
InstanceNode = Tuple[int, int]


@dataclass
class _DisjointSet:
    """Union-Find com compressão de caminho e união por tamanho."""

    parent: List[int]
    size: List[int]

    @classmethod
    def create(cls, count: int) -> "_DisjointSet":
        return cls(parent=list(range(count)), size=[1] * count)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.size[root_first] < self.size[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        self.size[root_first] += self.size[root_second]


def _validate_inputs(
    tiles_preds: Sequence[np.ndarray],
    tile_boxes: Sequence[TileBox],
    full_shape: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[TileBox]]:
    if len(tiles_preds) != len(tile_boxes):
        raise ValueError("tiles_preds e tile_boxes devem ter o mesmo tamanho.")
    if len(full_shape) != 2 or any(int(value) <= 0 for value in full_shape):
        raise ValueError("full_shape deve conter altura e largura positivas.")

    full_height, full_width = (int(full_shape[0]), int(full_shape[1]))
    masks: List[np.ndarray] = []
    boxes: List[TileBox] = []
    for index, (prediction, box) in enumerate(zip(tiles_preds, tile_boxes)):
        mask = np.asarray(prediction)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Tile {index}: a máscara deve ser um array inteiro 2D.")
        if mask.size and int(mask.min()) < 0:
            raise ValueError(f"Tile {index}: IDs de instância não podem ser negativos.")
        if len(box) != 4:
            raise ValueError(f"Tile {index}: caixa deve ser (ymin, xmin, ymax, xmax).")

        y_min, x_min, y_max, x_max = (int(value) for value in box)
        if not (0 <= y_min < y_max <= full_height and 0 <= x_min < x_max <= full_width):
            raise ValueError(f"Tile {index}: caixa {box} fora do canvas {full_shape}.")
        if mask.shape != (y_max - y_min, x_max - x_min):
            raise ValueError(
                f"Tile {index}: shape {mask.shape} incompatível com a caixa "
                f"{(y_min, x_min, y_max, x_max)}."
            )

        masks.append(mask.astype(np.int64, copy=False))
        boxes.append((y_min, x_min, y_max, x_max))

    return masks, boxes


def _intersection_box(first: TileBox, second: TileBox) -> TileBox | None:
    y_min = max(first[0], second[0])
    x_min = max(first[1], second[1])
    y_max = min(first[2], second[2])
    x_max = min(first[3], second[3])
    if y_min >= y_max or x_min >= x_max:
        return None
    return y_min, x_min, y_max, x_max


def _crop_global_overlap(mask: np.ndarray, box: TileBox, overlap: TileBox) -> np.ndarray:
    y_min, x_min, y_max, x_max = overlap
    return mask[
        y_min - box[0] : y_max - box[0],
        x_min - box[1] : x_max - box[1],
    ]


def _greedy_overlap_matches(
    first: np.ndarray,
    second: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Faz matching um-para-um por IoU decrescente dentro do overlap."""
    first_ids, first_counts = np.unique(first[first > 0], return_counts=True)
    second_ids, second_counts = np.unique(second[second > 0], return_counts=True)
    if first_ids.size == 0 or second_ids.size == 0:
        return []

    area_first = dict(zip(first_ids.tolist(), first_counts.tolist()))
    area_second = dict(zip(second_ids.tolist(), second_counts.tolist()))
    both_foreground = (first > 0) & (second > 0)
    if not both_foreground.any():
        return []

    pairs, intersections = np.unique(
        np.stack([first[both_foreground], second[both_foreground]], axis=1),
        axis=0,
        return_counts=True,
    )
    candidates: List[Tuple[float, int, int]] = []
    for (first_id, second_id), intersection in zip(pairs, intersections):
        union = area_first[int(first_id)] + area_second[int(second_id)] - int(intersection)
        iou = float(intersection / union)
        if iou >= threshold:
            candidates.append((iou, int(first_id), int(second_id)))

    matched_first = set()
    matched_second = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, first_id, second_id in sorted(candidates, reverse=True):
        if first_id in matched_first or second_id in matched_second:
            continue
        matched_first.add(first_id)
        matched_second.add(second_id)
        matches.append((first_id, second_id, iou))
    return matches


def _enumerate_nodes(masks: Sequence[np.ndarray]) -> Tuple[List[InstanceNode], Dict[InstanceNode, int]]:
    nodes = [
        (tile_index, int(local_id))
        for tile_index, mask in enumerate(masks)
        for local_id in np.unique(mask)
        if local_id != 0
    ]
    return nodes, {node: index for index, node in enumerate(nodes)}


def _center_confidence(box: TileBox) -> np.ndarray:
    """Pontua pixels pela proximidade normalizada ao centro do tile."""
    height = box[2] - box[0]
    width = box[3] - box[1]
    y = (np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / max(height, 1)
    x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / max(width, 1)
    return -(y[:, None] ** 2 + x[None, :] ** 2)


def _compose_global_mask(
    masks: Sequence[np.ndarray],
    boxes: Sequence[TileBox],
    full_shape: Tuple[int, int],
    node_indices: Dict[InstanceNode, int],
    disjoint_set: _DisjointSet,
) -> np.ndarray:
    canvas = np.zeros(full_shape, dtype=np.int64)
    confidence = np.full(full_shape, -np.inf, dtype=np.float32)

    for tile_index, (mask, box) in enumerate(zip(masks, boxes)):
        mapped = np.zeros(mask.shape, dtype=np.int64)
        for local_id in np.unique(mask):
            if local_id == 0:
                continue
            node_index = node_indices[(tile_index, int(local_id))]
            # +1 reserva zero para o fundo durante a composição.
            mapped[mask == local_id] = disjoint_set.find(node_index) + 1

        tile_confidence = _center_confidence(box)
        y_min, x_min, y_max, x_max = box
        target = canvas[y_min:y_max, x_min:x_max]
        target_confidence = confidence[y_min:y_max, x_min:x_max]
        update = (mapped > 0) & (tile_confidence > target_confidence)
        target[update] = mapped[update]
        target_confidence[update] = tile_confidence[update]

    present = np.unique(canvas)
    present = present[present > 0]
    if present.size == 0:
        return canvas
    relabeled = np.zeros_like(canvas)
    for new_id, old_id in enumerate(present, start=1):
        relabeled[canvas == old_id] = new_id
    return relabeled


def stitch_tiles_with_fusion(
    tiles_preds: Sequence[np.ndarray],
    tile_boxes: Sequence[TileBox],
    full_shape: Tuple[int, int],
    iou_overlap_threshold: float = 0.20,
) -> np.ndarray:
    """Funde instâncias correspondentes e devolve uma máscara global contínua."""
    if not 0.0 <= iou_overlap_threshold <= 1.0:
        raise ValueError("iou_overlap_threshold deve estar entre 0 e 1.")
    masks, boxes = _validate_inputs(tiles_preds, tile_boxes, full_shape)
    nodes, node_indices = _enumerate_nodes(masks)
    disjoint_set = _DisjointSet.create(len(nodes))

    for first_index in range(len(masks)):
        for second_index in range(first_index + 1, len(masks)):
            overlap = _intersection_box(boxes[first_index], boxes[second_index])
            if overlap is None:
                continue
            first_crop = _crop_global_overlap(masks[first_index], boxes[first_index], overlap)
            second_crop = _crop_global_overlap(masks[second_index], boxes[second_index], overlap)
            for first_id, second_id, _ in _greedy_overlap_matches(
                first_crop,
                second_crop,
                iou_overlap_threshold,
            ):
                disjoint_set.union(
                    node_indices[(first_index, first_id)],
                    node_indices[(second_index, second_id)],
                )

    return _compose_global_mask(masks, boxes, full_shape, node_indices, disjoint_set)


def compose_tiles_without_fusion(
    tiles_preds: Sequence[np.ndarray],
    tile_boxes: Sequence[TileBox],
    full_shape: Tuple[int, int],
) -> np.ndarray:
    """Compõe tiles mantendo IDs independentes, para a medição antes da correção."""
    masks, boxes = _validate_inputs(tiles_preds, tile_boxes, full_shape)
    nodes, node_indices = _enumerate_nodes(masks)
    return _compose_global_mask(
        masks,
        boxes,
        full_shape,
        node_indices,
        _DisjointSet.create(len(nodes)),
    )
