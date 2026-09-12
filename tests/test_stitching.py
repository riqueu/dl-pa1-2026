"""Testes determinísticos da fusão de instâncias entre tiles."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.metrics import evaluate_instances
from src.mosaic import predict_tiled_naive
from src.stitching import compose_tiles_without_fusion, stitch_tiles_with_fusion


class _FullForegroundModel(nn.Module):
    """Mock que produz uma única instância cobrindo cada tile."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        logits = torch.full((batch, 3, height, width), -8.0, device=x.device)
        logits[:, 1] = 8.0 + self.anchor
        return logits


class StitchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boxes = [(0, 0, 6, 6), (0, 4, 6, 10)]

    def test_funde_mesmo_objeto_na_sobreposicao(self) -> None:
        first = np.zeros((6, 6), dtype=np.int64)
        second = np.zeros((6, 6), dtype=np.int64)
        first[1:5, 3:6] = 1
        second[1:5, 0:3] = 7

        before = compose_tiles_without_fusion([first, second], self.boxes, (6, 10))
        after = stitch_tiles_with_fusion([first, second], self.boxes, (6, 10))

        self.assertEqual(len(np.unique(before)) - 1, 2)
        self.assertEqual(len(np.unique(after)) - 1, 1)
        self.assertTrue(np.all(after[1:5, 3:7] == 1))

    def test_nao_funde_objetos_sem_intersecao(self) -> None:
        first = np.zeros((6, 6), dtype=np.int32)
        second = np.zeros((6, 6), dtype=np.int32)
        first[0:2, 4:6] = 2
        second[4:6, 0:2] = 9
        result = stitch_tiles_with_fusion([first, second], self.boxes, (6, 10))
        self.assertEqual(len(np.unique(result)) - 1, 2)

    def test_fusao_transitiva_em_tres_tiles(self) -> None:
        boxes = [(0, 0, 5, 6), (0, 4, 5, 10), (0, 8, 5, 14)]
        masks = []
        for local_id in (2, 5, 11):
            mask = np.zeros((5, 6), dtype=np.int64)
            mask[1:4, :] = local_id
            masks.append(mask)
        result = stitch_tiles_with_fusion(masks, boxes, (5, 14), 0.5)
        self.assertEqual(len(np.unique(result)) - 1, 1)

    def test_metrica_melhora_apos_fusao(self) -> None:
        first = np.zeros((6, 6), dtype=np.int64)
        second = np.zeros((6, 6), dtype=np.int64)
        first[1:5, 3:6] = 1
        second[1:5, 0:3] = 1
        ground_truth = np.zeros((6, 10), dtype=np.int64)
        ground_truth[1:5, 3:7] = 1

        before = compose_tiles_without_fusion([first, second], self.boxes, ground_truth.shape)
        after = stitch_tiles_with_fusion([first, second], self.boxes, ground_truth.shape)
        before_metrics = evaluate_instances(before, ground_truth)
        after_metrics = evaluate_instances(after, ground_truth)

        self.assertGreater(after_metrics["mAP"], before_metrics["mAP"])
        self.assertEqual(after_metrics["count_error"], 0)

    def test_tile_vazio_e_ids_nao_consecutivos(self) -> None:
        first = np.zeros((6, 6), dtype=np.int64)
        second = np.zeros((6, 6), dtype=np.int64)
        second[2:4, 2:4] = 42
        result = stitch_tiles_with_fusion([first, second], self.boxes, (6, 10))
        self.assertEqual(result.dtype, np.int64)
        self.assertEqual(set(np.unique(result)), {0, 1})

    def test_integracao_com_pipeline_de_mosaico(self) -> None:
        image = np.zeros((192, 192, 3), dtype=np.float32)
        tiled = predict_tiled_naive(
            image,
            model=_FullForegroundModel().eval(),
            tile_size=128,
            stride=64,
            mode="center_crop",
            min_area=0,
        )
        stitched = stitch_tiles_with_fusion(
            tiled["tiles_preds"],
            tiled["tile_boxes"],
            image.shape[:2],
            iou_overlap_threshold=0.5,
        )
        self.assertEqual(stitched.shape, image.shape[:2])
        self.assertEqual(set(np.unique(stitched)), {1})

    def test_valida_shapes_e_tipos(self) -> None:
        with self.assertRaises(ValueError):
            stitch_tiles_with_fusion([], [(0, 0, 1, 1)], (2, 2))
        with self.assertRaises(ValueError):
            stitch_tiles_with_fusion(
                [np.zeros((2, 2), dtype=np.float32)],
                [(0, 0, 2, 2)],
                (2, 2),
            )
        with self.assertRaises(ValueError):
            stitch_tiles_with_fusion(
                [np.zeros((2, 2), dtype=np.int64)],
                [(0, 0, 1, 2)],
                (2, 2),
            )


if __name__ == "__main__":
    unittest.main()
