"""Módulo de montagem de mosaicos, extração de tiles e inferência ingênua (Parte 4).

Este módulo implementa o pipeline da Parte 4 (Membro A: Henrique):
1. Montagem de mosaicos grandes a partir de imagens do DSB2018 com GT unificado e contínuo.
2. Extração de janelas deslizantes (tiling com sobreposição arbitrária, ex.: 50%).
3. Inferência ingênua por tile (sem fusão de instâncias na borda), evidenciando a falha
   conceitual descrita no slide 83 da aula (fatiamento de núcleos e duplicação de IDs).
4. Diagnóstico geométrico e visual da fragmentação de instâncias na fronteira.
5. Exportação de estruturas desacopladas (tiles_preds, tile_boxes) para o algoritmo de
   costura (stitching) do Membro B.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch

try:
    from src.postprocess import watershed_instance_segmentation
    from src.utils import colorize_instances, overlay_mask_on_image
except ModuleNotFoundError:
    from postprocess import watershed_instance_segmentation
    from utils import colorize_instances, overlay_mask_on_image


def load_dsb2018_sample(
    image_id: str,
    data_dir: str = "data/raw/stage1_train",
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega uma imagem individual do DSB2018 e funde suas máscaras em um array de instâncias.

    Args:
        image_id: Identificador da imagem no dataset.
        data_dir: Diretório raiz contendo as pastas de imagens do DSB2018.
        target_size: Opcional (H, W) para redimensionamento.

    Returns:
        Tupla (image_rgb, instance_mask):
            - image_rgb: Array float32 (H, W, 3) em [0.0, 1.0].
            - instance_mask: Array int64 (H, W) onde 0 = fundo e 1..K = instâncias.
    """
    image_dir = os.path.join(data_dir, image_id)
    img_files = glob.glob(os.path.join(image_dir, "images", "*.png"))
    if not img_files:
        raise FileNotFoundError(f"Imagem não encontrada para ID: {image_id} em {image_dir}")

    with Image.open(img_files[0]) as pil_img:
        img_rgb = pil_img.convert("RGB")
        orig_w, orig_h = img_rgb.size
        if target_size is not None:
            img_rgb = img_rgb.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        img_np = np.array(img_rgb, dtype=np.float32) / 255.0

    mask_files = sorted(glob.glob(os.path.join(image_dir, "masks", "*.png")))
    curr_h = target_size[0] if target_size is not None else orig_h
    curr_w = target_size[1] if target_size is not None else orig_w

    instance_mask = np.zeros((curr_h, curr_w), dtype=np.int64)
    for inst_id, m_file in enumerate(mask_files, start=1):
        with Image.open(m_file) as pil_mask:
            if target_size is not None:
                pil_mask = pil_mask.resize(
                    (target_size[1], target_size[0]),
                    Image.Resampling.NEAREST
                )
            m_np = np.array(pil_mask) > 0
            instance_mask[m_np] = inst_id

    return img_np, instance_mask


def create_mosaic_sample(
    image_ids: Sequence[str],
    data_dir: str = "data/raw/stage1_train",
    grid: Tuple[int, int] = (2, 2),
    tile_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Monta um mosaico em grade a partir de imagens do DSB2018 com GT contínuo.

    Cada quadrante recebe uma imagem redimensionada para `tile_size`. As máscaras
    de instâncias são combinadas de forma que os identificadores sejam únicos e
    consecutivos globalmente: $1 \\dots K_{\\text{total}}$, sem colisões.

    Args:
        image_ids: Lista de identificadores de imagens (deve conter grid[0]*grid[1] IDs).
        data_dir: Diretório raiz das imagens.
        grid: Formato da grade (linhas, colunas), padrão (2, 2).
        tile_size: Resolução (H, W) de cada sub-imagem na grade, padrão (256, 256).

    Returns:
        Tupla (mosaic_image, mosaic_gt_mask):
            - mosaic_image: Array float32 (H_total, W_total, 3) em [0.0, 1.0].
            - mosaic_gt_mask: Array int64 (H_total, W_total) com IDs únicos de instâncias.
    """
    n_required = grid[0] * grid[1]
    if len(image_ids) < n_required:
        raise ValueError(f"Requer {n_required} image_ids para grade {grid}; recebido {len(image_ids)}.")

    total_h = grid[0] * tile_size[0]
    total_w = grid[1] * tile_size[1]

    mosaic_image = np.zeros((total_h, total_w, 3), dtype=np.float32)
    mosaic_gt = np.zeros((total_h, total_w), dtype=np.int64)

    current_id_offset = 0

    idx = 0
    for r in range(grid[0]):
        for c in range(grid[1]):
            img_id = image_ids[idx]
            img, inst = load_dsb2018_sample(img_id, data_dir=data_dir, target_size=tile_size)

            y_start = r * tile_size[0]
            y_end = y_start + tile_size[0]
            x_start = c * tile_size[1]
            x_end = x_start + tile_size[1]

            mosaic_image[y_start:y_end, x_start:x_end] = img

            # Reindexação contínua para evitar colisão de IDs entre sub-imagens
            inst_reindexed = np.zeros_like(inst)
            mask_fg = (inst > 0)
            if mask_fg.any():
                inst_reindexed[mask_fg] = inst[mask_fg] + current_id_offset
                n_inst_in_tile = int(inst.max())
                current_id_offset += n_inst_in_tile

            mosaic_gt[y_start:y_end, x_start:x_end] = inst_reindexed
            idx += 1

    return mosaic_image, mosaic_gt


def create_mosaic_from_large_image(
    image_id: str,
    data_dir: str = "data/raw/stage1_train",
    crop_size: Tuple[int, int] = (512, 512),
    origin: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrai um campo contínuo de alta resolução a partir de uma imagem grande do DSB2018.

    Permite testar o mosaico sobre lâminas biológicas contínuas (ex.: ISBI TissueBW
    1024x1024 ou fluorescência 696x520), onde células cruzam naturalmente as linhas
    de coordenadas das janelas deslizantes.

    Args:
        image_id: ID da imagem no dataset.
        data_dir: Diretório raiz do dataset.
        crop_size: (H, W) da região a ser extraída (padrão 512x512).
        origin: (ymin, xmin) de onde iniciar o recorte contínuo.

    Returns:
        Tupla (crop_image, crop_gt_mask):
            - crop_image: Array float32 (crop_h, crop_w, 3) em [0.0, 1.0].
            - crop_gt_mask: Array int64 (crop_h, crop_w) com rótulos consecutivos 1..K.
    """
    img_full, gt_full = load_dsb2018_sample(image_id, data_dir=data_dir, target_size=None)
    h_orig, w_orig = gt_full.shape

    y0, x0 = origin
    y1 = min(y0 + crop_size[0], h_orig)
    x1 = min(x0 + crop_size[1], w_orig)

    crop_image = img_full[y0:y1, x0:x1]
    crop_gt = gt_full[y0:y1, x0:x1].copy()

    # Reindexa rótulos consecutivos preservando apenas os que caíram dentro do recorte
    present = np.unique(crop_gt)
    present = present[present != 0]

    reindexed_gt = np.zeros_like(crop_gt)
    for new_id, old_id in enumerate(present, start=1):
        reindexed_gt[crop_gt == old_id] = new_id

    # Se o recorte for menor que crop_size, faz zero-pad
    if crop_image.shape[0] < crop_size[0] or crop_image.shape[1] < crop_size[1]:
        pad_img = np.zeros((crop_size[0], crop_size[1], 3), dtype=np.float32)
        pad_gt = np.zeros((crop_size[0], crop_size[1]), dtype=np.int64)
        pad_img[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
        pad_gt[:crop_gt.shape[0], :crop_gt.shape[1]] = reindexed_gt
        return pad_img, pad_gt

    return crop_image, reindexed_gt


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
) -> List[Dict[str, Any]]:
    """Gera recortes sobrepostos com coordenadas espaciais em janela deslizante.

    Garante cobertura completa do canvas: se a última janela exceder a dimensão,
    ancora na borda final da imagem.

    Args:
        image: Array (H, W) ou (H, W, C).
        tile_size: Tamanho do patch quadrado (padrão 256).
        stride: Passo de deslocamento da janela (padrão 128 para 50% de sobreposição).

    Returns:
        Lista de dicionários com:
            - 'tile': Sub-array recortado (tile_size, tile_size, C) ou (tile_size, tile_size).
            - 'box': Tupla (ymin, xmin, ymax, xmax) das coordenadas globais.
            - 'index': Índice inteiro do tile (0..N-1).
            - 'grid_pos': Posição na grade de amostragem (row_idx, col_idx).
    """
    h, w = image.shape[:2]

    # Pontos de amostragem em Y
    y_coords = list(range(0, h - tile_size + 1, stride))
    if len(y_coords) == 0 or y_coords[-1] + tile_size < h:
        y_coords.append(max(0, h - tile_size))
    y_coords = sorted(list(set(y_coords)))

    # Pontos de amostragem em X
    x_coords = list(range(0, w - tile_size + 1, stride))
    if len(x_coords) == 0 or x_coords[-1] + tile_size < w:
        x_coords.append(max(0, w - tile_size))
    x_coords = sorted(list(set(x_coords)))

    tiles = []
    tile_idx = 0
    for r_idx, y in enumerate(y_coords):
        for c_idx, x in enumerate(x_coords):
            ymin, ymax = y, y + tile_size
            xmin, xmax = x, x + tile_size
            patch = image[ymin:ymax, xmin:xmax]

            tiles.append({
                "tile": patch,
                "box": (ymin, xmin, ymax, xmax),
                "index": tile_idx,
                "grid_pos": (r_idx, c_idx),
            })
            tile_idx += 1

    return tiles


def predict_tile(
    patch: np.ndarray,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    interior_threshold: float = 0.50,
    foreground_threshold: float = 0.50,
    min_area: int = 10,
) -> np.ndarray:
    """Executa a predição da U-Net Watershed sobre um único patch 256x256.

    Args:
        patch: Array float32 (H, W, 3) em [0.0, 1.0].
        model: Modelo U-Net com 3 canais de saída (Trilha A).
        device: Dispositivo de execução (CUDA/CPU).
        interior_threshold: Limiar de interior para formação de marcadores.
        foreground_threshold: Limiar de foreground.
        min_area: Área mínima em pixels para descartar ruídos.

    Returns:
        Array int64 (H, W) com instâncias locais decodificadas por Watershed.
    """
    if device is None:
        device = next(model.parameters()).device

    # Transpõe para PyTorch (1, 3, H, W)
    t_in = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(t_in)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (3, H, W)

    pred_inst = watershed_instance_segmentation(
        probs,
        interior_threshold=interior_threshold,
        foreground_threshold=foreground_threshold,
        min_area=min_area,
        connectivity=1,
    )
    return pred_inst


def predict_tiled_naive(
    mosaic_image: np.ndarray,
    model: torch.nn.Module,
    tile_size: int = 256,
    stride: int = 128,
    device: Optional[torch.device] = None,
    mode: str = "center_crop",
    interior_threshold: float = 0.50,
    foreground_threshold: float = 0.50,
    min_area: int = 10,
) -> Dict[str, Any]:
    """Executa o pipeline de Tiling Ingênuo sobre a imagem mosaico.

    Demonstra o problema fundamental do slide 83 para segmentação de instâncias:
    - Cada tile é inferido isoladamente.
    - Na reconstrução ingênua sem fusão:
      * 'center_crop': Cada tile apenas preenche seu recorte central de confiança
        (prática de segmentação semântica). Objetos que cruzam as divisões são
        fatiados ao meio, ganhando metades com IDs diferentes.
      * 'direct_stamp': Cada tile estampa suas instâncias com novos IDs sequenciais.
        Em zonas de sobreposição, os mesmos objetos são preditos múltiplas vezes
        com rótulos diferentes, inflacionando a contagem.

    Args:
        mosaic_image: Imagem RGB (H, W, 3) em [0.0, 1.0].
        model: Modelo U-Net pré-treinado.
        tile_size: Tamanho do patch (padrão 256).
        stride: Passo do deslocamento (padrão 128).
        device: Dispositivo CUDA/CPU.
        mode: 'center_crop' ou 'direct_stamp'.
        interior_threshold: Limiar do interior para watershed.
        foreground_threshold: Limiar de foreground.
        min_area: Filtro de tamanho mínimo.

    Returns:
        Dicionário contendo:
            - 'naive_instance_mask': Array int64 (H, W) montado ingenuamente.
            - 'tiles_preds': Lista com as predições individuais (tile_size, tile_size) por patch.
            - 'tile_boxes': Lista com as caixas globais (ymin, xmin, ymax, xmax).
            - 'tiles_raw': Lista com os patches RGB extraídos.
            - 'num_instances': Contagem de instâncias no mosaico ingênuo.
            - 'mode': Modo ingênuo empregado.
    """
    h_full, w_full = mosaic_image.shape[:2]
    tiles_info = extract_tiles(mosaic_image, tile_size=tile_size, stride=stride)

    tiles_preds: List[np.ndarray] = []
    tile_boxes: List[Tuple[int, int, int, int]] = []
    tiles_raw: List[np.ndarray] = []

    for item in tiles_info:
        patch = item["tile"]
        box = item["box"]
        pred_local = predict_tile(
            patch,
            model=model,
            device=device,
            interior_threshold=interior_threshold,
            foreground_threshold=foreground_threshold,
            min_area=min_area,
        )
        tiles_preds.append(pred_local)
        tile_boxes.append(box)
        tiles_raw.append(patch)

    naive_mask = np.zeros((h_full, w_full), dtype=np.int64)

    if mode == "center_crop":
        # Prática do Slide 83: Cada tile contribui apenas com sua região central (Voronoi)
        # Cria matriz de distâncias aos centros dos tiles para definir as fronteiras de corte
        centers = []
        for (ymin, xmin, ymax, xmax) in tile_boxes:
            cy = (ymin + ymax) / 2.0
            cx = (xmin + xmax) / 2.0
            centers.append((cy, cx))

        # Grade espacial
        y_grid, x_grid = np.mgrid[0:h_full, 0:w_full]
        tile_ownership = np.zeros((h_full, w_full), dtype=np.int64)
        min_dist_sq = np.full((h_full, w_full), np.inf, dtype=np.float32)

        for t_idx, (cy, cx) in enumerate(centers):
            dist_sq = (y_grid - cy) ** 2 + (x_grid - cx) ** 2
            closer = dist_sq < min_dist_sq
            tile_ownership[closer] = t_idx
            min_dist_sq[closer] = dist_sq[closer]

        # Cada tile carimba apenas os pixels sob seu domínio Voronoi
        global_id_counter = 0
        for t_idx, (pred_tile, (ymin, xmin, ymax, xmax)) in enumerate(zip(tiles_preds, tile_boxes)):
            owned_region = (tile_ownership[ymin:ymax, xmin:xmax] == t_idx)
            sub_pred = pred_tile.copy()
            sub_pred[~owned_region] = 0

            u_ids = [int(u) for u in np.unique(sub_pred) if u != 0]
            for u in u_ids:
                global_id_counter += 1
                naive_mask[ymin:ymax, xmin:xmax][sub_pred == u] = global_id_counter

    elif mode == "direct_stamp":
        # Estampa direta: novos IDs para cada detecção, sem fusão na sobreposição
        global_id_counter = 0
        for pred_tile, (ymin, xmin, ymax, xmax) in zip(tiles_preds, tile_boxes):
            u_ids = [int(u) for u in np.unique(pred_tile) if u != 0]
            for u in u_ids:
                global_id_counter += 1
                m = (pred_tile == u)
                naive_mask[ymin:ymax, xmin:xmax][m] = global_id_counter

    else:
        raise ValueError(f"Modo ingênuo desconhecido: '{mode}'. Use 'center_crop' ou 'direct_stamp'.")

    n_instances = len(np.unique(naive_mask)) - (1 if 0 in naive_mask else 0)

    return {
        "naive_instance_mask": naive_mask,
        "tiles_preds": tiles_preds,
        "tile_boxes": tile_boxes,
        "tiles_raw": tiles_raw,
        "num_instances": n_instances,
        "mode": mode,
    }


def detect_split_nuclei(
    gt_mask: np.ndarray,
    tile_boxes: Sequence[Tuple[int, int, int, int]],
    naive_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Identifica núcleos de Ground Truth que cruzam bordas de tiles e analisa seu fatiamento.

    Args:
        gt_mask: Array (H, W) de instâncias Ground Truth (0 = fundo).
        tile_boxes: Lista de caixas (ymin, xmin, ymax, xmax) das janelas de inferência.
        naive_pred: Opcional array (H, W) da predição ingênua para verificar duplicação.

    Returns:
        Dicionário com estatísticas:
            - 'total_gt_nuclei': Total de objetos no GT.
            - 'boundary_gt_ids': Lista de IDs de GT que cruzam pelo menos uma linha de borda interna.
            - 'split_gt_details': Lista com detalhes de cada núcleo cortado.
            - 'num_boundary_nuclei': Quantidade de núcleos na fronteira.
            - 'split_ratio': Proporção de núcleos que caem em linhas de corte.
    """
    h, w = gt_mask.shape

    # Linhas de corte internas
    cut_y = set()
    cut_x = set()
    for (ymin, xmin, ymax, xmax) in tile_boxes:
        if 0 < ymin < h:
            cut_y.add(ymin)
        if 0 < ymax < h:
            cut_y.add(ymax)
        if 0 < xmin < w:
            cut_x.add(xmin)
        if 0 < xmax < w:
            cut_x.add(xmax)

    gt_ids = [int(x) for x in np.unique(gt_mask) if x != 0]
    total_gt = len(gt_ids)

    boundary_gt_ids = []
    split_details = []

    for gid in gt_ids:
        ys, xs = np.where(gt_mask == gid)
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        # Verifica se o núcleo cruza alguma linha de corte horizontal ou vertical
        crosses_y = any(min_y < cy <= max_y for cy in cut_y)
        crosses_x = any(min_x < cx <= max_x for cx in cut_x)

        if crosses_y or crosses_x:
            boundary_gt_ids.append(gid)

            detail: Dict[str, Any] = {
                "gt_id": gid,
                "bbox": (int(min_y), int(min_x), int(max_y), int(max_x)),
                "area": int(len(ys)),
                "crosses_y": bool(crosses_y),
                "crosses_x": bool(crosses_x),
            }

            if naive_pred is not None:
                # Quantos IDs preditos cobrem este núcleo no naive_pred?
                overlapping_preds = naive_pred[gt_mask == gid]
                pred_ids_hit = [int(p) for p in np.unique(overlapping_preds) if p != 0]
                detail["overlapping_pred_ids"] = pred_ids_hit
                detail["num_fragments"] = len(pred_ids_hit)

            split_details.append(detail)

    split_ratio = float(len(boundary_gt_ids) / total_gt) if total_gt > 0 else 0.0

    return {
        "total_gt_nuclei": total_gt,
        "boundary_gt_ids": boundary_gt_ids,
        "num_boundary_nuclei": len(boundary_gt_ids),
        "split_gt_details": split_details,
        "split_ratio": split_ratio,
        "cut_lines": {"y": sorted(list(cut_y)), "x": sorted(list(cut_x))},
    }


def plot_tiling_failure_analysis(
    mosaic_image: np.ndarray,
    gt_mask: np.ndarray,
    naive_pred: np.ndarray,
    tile_boxes: Sequence[Tuple[int, int, int, int]],
    save_path: str,
    zoom_crop: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Gera figura de diagnóstico evidenciando a falha do Tiling Ingênuo.

    Mostra:
    1. Mosaico RGB com as fronteiras das janelas deslizantes (50% overlap).
    2. Ground Truth contínuo (colorizado).
    3. Predição ingênua (evidenciando fatiamento de núcleos e descontinuidade de IDs).
    4. Zoom in com destaque sobre um núcleo fatiado pela borda do tile.

    Args:
        mosaic_image: Array RGB (H, W, 3) em [0, 1].
        gt_mask: Array int64 (H, W) do Ground Truth.
        naive_pred: Array int64 (H, W) da predição ingênua.
        tile_boxes: Caixas delimitadoras das janelas deslizantes.
        save_path: Caminho para salvar a imagem PNG resultante.
        zoom_crop: Opcional (ymin, xmin, ymax, xmax) para a área de zoom. Se None,
            seleciona automaticamente um núcleo de fronteira fatiado.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    h, w = gt_mask.shape

    # 1. Colorização de instâncias
    gt_colored = colorize_instances(gt_mask, seed=42)
    pred_colored = colorize_instances(naive_pred, seed=99)

    # 2. Localiza um bom recorte para zoom se não fornecido
    if zoom_crop is None:
        analysis = detect_split_nuclei(gt_mask, tile_boxes, naive_pred=naive_pred)
        # Procura um núcleo com 2 ou mais fragmentos
        candidates = [d for d in analysis["split_gt_details"] if d.get("num_fragments", 0) >= 2]
        if candidates:
            cand = candidates[0]
            cy = (cand["bbox"][0] + cand["bbox"][2]) // 2
            cx = (cand["bbox"][1] + cand["bbox"][3]) // 2
            r = 36
            zoom_crop = (
                max(0, cy - r),
                max(0, cx - r),
                min(h, cy + r),
                min(w, cx + r),
            )
        else:
            # Fallback para o centro do mosaico (interseção central)
            zoom_crop = (h // 2 - 40, w // 2 - 40, h // 2 + 40, w // 2 + 40)

    zy0, zx0, zy1, zx1 = zoom_crop

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=150)

    # Painel 1: Imagem Mosaico com linhas de janelas deslizantes
    axes[0].imshow(mosaic_image)
    axes[0].set_title(f"Mosaico Original ({h}x{w})\n+ Grade de Janelas Deslizantes", fontsize=11, fontweight="bold")
    axes[0].set_axis_off()

    # Desenha caixas de tiles com estilos alternados
    colors_grid = ["cyan", "magenta", "yellow", "lime", "orange"]
    for i, (ymin, xmin, ymax, xmax) in enumerate(tile_boxes):
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.2,
            edgecolor=colors_grid[i % len(colors_grid)],
            facecolor="none",
            linestyle="--",
            alpha=0.85,
        )
        axes[0].add_patch(rect)

    # Retângulo indicador da região de zoom
    rect_zoom = patches.Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        linewidth=2.0,
        edgecolor="red",
        facecolor="none",
        linestyle="-",
    )
    axes[0].add_patch(rect_zoom)
    axes[0].text(zx0, zy0 - 4, "Área de Zoom", color="red", fontsize=9, fontweight="bold")

    # Painel 2: Ground Truth Contínuo
    gt_overlay = overlay_mask_on_image(mosaic_image, gt_colored, alpha=0.55)
    axes[1].imshow(gt_overlay)
    n_gt = len(np.unique(gt_mask)) - (1 if 0 in gt_mask else 0)
    axes[1].set_title(f"Ground Truth Contínuo\n({n_gt} núcleos biológicos unificados)", fontsize=11, fontweight="bold")
    axes[1].set_axis_off()

    # Painel 3: Predição Ingênua (Fatiada / Duplicada)
    pred_overlay = overlay_mask_on_image(mosaic_image, pred_colored, alpha=0.55)
    axes[2].imshow(pred_overlay)
    n_pred = len(np.unique(naive_pred)) - (1 if 0 in naive_pred else 0)
    axes[2].set_title(
        f"Tiling Ingênuo (Slide 83)\n({n_pred} instâncias — contagem inflada)",
        fontsize=11,
        fontweight="bold",
        color="darkred",
    )
    axes[2].set_axis_off()

    # Painel 4: Zoom na Falha da Borda
    zoom_pred = pred_colored[zy0:zy1, zx0:zx1]
    zoom_img = mosaic_image[zy0:zy1, zx0:zx1]
    zoom_overlay = overlay_mask_on_image(zoom_img, zoom_pred, alpha=0.65)
    axes[3].imshow(zoom_overlay)

    # Adiciona linhas de corte no zoom
    analysis = detect_split_nuclei(gt_mask, tile_boxes)
    for cy in analysis["cut_lines"]["y"]:
        if zy0 < cy < zy1:
            axes[3].axhline(cy - zy0, color="red", linestyle=":", linewidth=2.0, alpha=0.9)
    for cx in analysis["cut_lines"]["x"]:
        if zx0 < cx < zx1:
            axes[3].axvline(cx - zx0, color="red", linestyle=":", linewidth=2.0, alpha=0.9)

    axes[3].set_title(
        "Zoom: Núcleo Fatiado na Borda\n(Cores distintas = IDs diferentes)",
        fontsize=11,
        fontweight="bold",
        color="darkred",
    )
    axes[3].set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"[Visualização Salva] Diagnóstico de falha gerado em: {save_path}")


if __name__ == "__main__":
    # Teste de fumaça sintético das funções do módulo
    print("=== Executando Smoke Test de src/mosaic.py ===")
    mock_canvas = np.random.rand(512, 512, 3).astype(np.float32)
    tiles = extract_tiles(mock_canvas, tile_size=256, stride=128)
    print(f"Número de tiles gerados (512x512, patch 256, stride 128): {len(tiles)}")
    assert len(tiles) == 9, f"Esperado 9 tiles, obtido {len(tiles)}"

    # Verifica caixas
    for i, t in enumerate(tiles):
        ymin, xmin, ymax, xmax = t["box"]
        assert ymax - ymin == 256
        assert xmax - xmin == 256
    print("Caixas dos tiles validadas com sucesso!")
    print("=== Smoke Test concluído com sucesso! ===")
