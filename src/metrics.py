"""Módulo de métricas para avaliação semântica e de instâncias.

Este módulo implementa do zero os algoritmos de matching (Hungarian e Greedy),
o cálculo de mAP@[.50:.95] para segmentação de instâncias e o erro absoluto de contagem,
bem como as métricas semânticas clássicas (IoU e Dice).

Regras do PA1:
- Sem dependências de bibliotecas prontas de métricas de detecção/instâncias (ex: mmdet/torchvision.detection).
- Matching explícito e documentado via Hungarian ou Greedy por IoU decrescente.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# Métricas de Segmentação Semântica (Binária: Fundo vs. Objeto)

def compute_semantic_iou(
    pred_bin: Union[np.ndarray, torch.Tensor],
    gt_bin: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> float:
    """Calcula a Interseção sobre União (IoU / Jaccard Index) binária.

    Args:
        pred_bin: Array ou Tensor binário (predição limiarizada) [0, 1].
        gt_bin: Array ou Tensor binário (ground truth) [0, 1].
        eps: Valor epsilon para estabilidade numérica.

    Returns:
        Score de IoU como float no intervalo [0.0, 1.0].
    """
    if isinstance(pred_bin, torch.Tensor):
        pred_bin = pred_bin.detach().cpu().numpy()
    if isinstance(gt_bin, torch.Tensor):
        gt_bin = gt_bin.detach().cpu().numpy()

    pred_bool = (pred_bin > 0.5).astype(bool)
    gt_bool = (gt_bin > 0.5).astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((intersection + eps) / (union + eps))


def compute_semantic_dice(
    pred_bin: Union[np.ndarray, torch.Tensor],
    gt_bin: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> float:
    """Calcula o Coeficiente Dice (F1-score) binário.

    Fórmula: Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred_bin: Array ou Tensor binário (predição limiarizada) [0, 1].
        gt_bin: Array ou Tensor binário (ground truth) [0, 1].
        eps: Valor epsilon para estabilidade numérica.

    Returns:
        Score Dice como float no intervalo [0.0, 1.0].
    """
    if isinstance(pred_bin, torch.Tensor):
        pred_bin = pred_bin.detach().cpu().numpy()
    if isinstance(gt_bin, torch.Tensor):
        gt_bin = gt_bin.detach().cpu().numpy()

    pred_bool = (pred_bin > 0.5).astype(bool)
    gt_bool = (gt_bin > 0.5).astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total_elements = pred_bool.sum() + gt_bool.sum()

    if total_elements == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((2.0 * intersection + eps) / (total_elements + eps))


# Métricas de Segmentação de Instâncias (Matching e mAP@[.50:.95])

def compute_pairwise_iou_matrix(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_ids: Sequence[int],
    gt_ids: Sequence[int],
) -> np.ndarray:
    """Calcula a matriz de IoU pareado entre instâncias preditas e ground truth.

    Implementação vetorizada em passo único via histograma 2D (np.bincount),
    evitando loops quadráticos lentos em imagens densas com centenas de núcleos.

    Args:
        pred_mask: Matriz 2D (H, W) com IDs de instâncias preditas (0 = fundo).
        gt_mask: Matriz 2D (H, W) com IDs de instâncias reais (0 = fundo).
        pred_ids: Lista com os rótulos de instâncias presentes em pred_mask.
        gt_ids: Lista com os rótulos de instâncias presentes em gt_mask.

    Returns:
        Matriz numpy (N_pred, N_gt) com os valores de IoU pareado em [0.0, 1.0].
    """
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)

    p_flat = pred_mask.ravel().astype(np.int64)
    g_flat = gt_mask.ravel().astype(np.int64)
    max_g = int(g_flat.max()) + 1

    fg = (p_flat > 0) & (g_flat > 0)
    if not fg.any():
        return iou_matrix

    p_areas = np.bincount(p_flat)
    g_areas = np.bincount(g_flat)

    pair_ids = p_flat[fg] * max_g + g_flat[fg]
    counts = np.bincount(pair_ids)

    p_idx_map = {pid: i for i, pid in enumerate(pred_ids)}
    g_idx_map = {gid: j for j, gid in enumerate(gt_ids)}

    nonzeros = np.nonzero(counts)[0]
    for idx in nonzeros:
        pid = idx // max_g
        gid = idx % max_g
        if pid in p_idx_map and gid in g_idx_map:
            inter = counts[idx]
            union = p_areas[pid] + g_areas[gid] - inter
            if union > 0:
                iou_matrix[p_idx_map[pid], g_idx_map[gid]] = float(inter / union)

    return iou_matrix


def match_instances(
    iou_matrix: np.ndarray,
    iou_threshold: float,
    method: str = "hungarian",
) -> Tuple[int, int, int]:
    """Realiza o matching 1-para-1 entre instâncias preditas e reais dado um limiar de IoU.

    Suporta duas regras de matching:
    1. 'hungarian': Otimização global (Maximum Weight Bipartite Matching) via
       linear_sum_assignment sobre custo negativo de IoU.
    2. 'greedy': Guloso por IoU decrescente (ordena todos os pares pred-gt por IoU
       e casa recursivamente os maiores pares que atendem ao limiar).

    Args:
        iou_matrix: Matriz (N_pred, N_gt) com os IoUs calculados.
        iou_threshold: Limiar de IoU mínimo para considerar casamento válido (ex: 0.50).
        method: Estratégia de matching ('hungarian' ou 'greedy').

    Returns:
        Tupla (tp, fp, fn):
            - tp: Verdadeiros Positivos (predições casadas com IoU >= limiar).
            - fp: Falsos Positivos (predições não casadas ou abaixo do limiar).
            - fn: Falsos Negativos (instâncias reais não detectadas).
    """
    n_pred, n_gt = iou_matrix.shape

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    if method == "hungarian":
        # Maximizar soma de IoU equivale a minimizar custo negativo
        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        tp = 0
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                tp += 1

        fp = n_pred - tp
        fn = n_gt - tp
        return tp, fp, fn

    elif method == "greedy":
        # Pares ordenados por IoU decrescente
        pairs = []
        for r in range(n_pred):
            for c in range(n_gt):
                iou = iou_matrix[r, c]
                if iou >= iou_threshold:
                    pairs.append((iou, r, c))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_pred = set()
        matched_gt = set()
        tp = 0

        for _, r, c in pairs:
            if r not in matched_pred and c not in matched_gt:
                matched_pred.add(r)
                matched_gt.add(c)
                tp += 1

        fp = n_pred - tp
        fn = n_gt - tp
        return tp, fp, fn

    else:
        raise ValueError(f"Método de matching desconhecido: '{method}'. Use 'hungarian' ou 'greedy'.")


def evaluate_instances(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_thresholds: Optional[Sequence[float]] = None,
    method: str = "hungarian",
) -> Dict[str, Any]:
    """Avalia predição de instâncias contra o ground truth para uma imagem.

    Calcula:
    - AP para cada limiar de IoU (padrão: 0.50 a 0.95 com passo 0.05).
    - mAP: Média dos APs sobre todos os limiares.
    - count_error: Erro absoluto de contagem de objetos (|N_pred - N_gt|).
    - Métricas detalhadas (TP, FP, FN por limiar).

    Conforme especificado no DSB2018 e no PA1:
    AP(t) = TP(t) / (TP(t) + FP(t) + FN(t))

    Args:
        pred_mask: Array 2D (H, W) de inteiros com os rótulos preditos (0 = fundo).
        gt_mask: Array 2D (H, W) de inteiros com os rótulos reais (0 = fundo).
        iou_thresholds: Sequência de limiares (default: 0.50, 0.55, ..., 0.95).
        method: Estratégia de matching ('hungarian' ou 'greedy').

    Returns:
        Dicionário com:
            - 'mAP': float
            - 'AP_per_iou': dict {threshold: ap_value}
            - 'count_error': int
            - 'n_pred': int
            - 'n_gt': int
            - 'details_per_iou': dict {threshold: {'tp': int, 'fp': int, 'fn': int}}
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05).round(2).tolist()

    # Extrai IDs únicos de instâncias (excluindo 0 = fundo)
    pred_ids = [int(x) for x in np.unique(pred_mask) if x != 0]
    gt_ids = [int(x) for x in np.unique(gt_mask) if x != 0]

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)
    count_error = abs(n_pred - n_gt)

    # Pré-calcula matriz de IoUs pareados
    iou_matrix = compute_pairwise_iou_matrix(pred_mask, gt_mask, pred_ids, gt_ids)

    ap_per_iou: Dict[float, float] = {}
    details_per_iou: Dict[float, Dict[str, int]] = {}

    for t in iou_thresholds:
        t_key = float(round(t, 2))
        tp, fp, fn = match_instances(iou_matrix, iou_threshold=t_key, method=method)

        # Se não há nem predições nem anotações na imagem, pontuação perfeita
        total_denom = tp + fp + fn
        if total_denom == 0:
            ap = 1.0
        else:
            ap = float(tp / total_denom)

        ap_per_iou[t_key] = ap
        details_per_iou[t_key] = {"tp": tp, "fp": fp, "fn": fn}

    mAP = float(np.mean(list(ap_per_iou.values())))

    return {
        "mAP": mAP,
        "AP_per_iou": ap_per_iou,
        "count_error": count_error,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "details_per_iou": details_per_iou,
    }


def evaluate_instance_batch(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    iou_thresholds: Optional[Sequence[float]] = None,
    method: str = "hungarian",
) -> Dict[str, Any]:
    """Avalia um conjunto/lote de imagens agregando as métricas de instâncias.

    Args:
        pred_masks: Lista de arrays 2D (H, W) de instâncias preditas.
        gt_masks: Lista de arrays 2D (H, W) de instâncias reais.
        iou_thresholds: Lista de limiares de IoU.
        method: Estratégia de matching ('hungarian' ou 'greedy').

    Returns:
        Dicionário com médias e estatísticas:
            - 'mean_mAP': float
            - 'mean_count_error': float
            - 'mean_AP_per_iou': dict {threshold: mean_ap}
            - 'total_images': int
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError("Quantidade de predições e máscaras GT deve ser idêntica.")

    if len(pred_masks) == 0:
        return {"mean_mAP": 0.0, "mean_count_error": 0.0, "mean_AP_per_iou": {}, "total_images": 0}

    results = [
        evaluate_instances(p, g, iou_thresholds=iou_thresholds, method=method)
        for p, g in zip(pred_masks, gt_masks)
    ]

    mean_mAP = float(np.mean([r["mAP"] for r in results]))
    mean_count_error = float(np.mean([r["count_error"] for r in results]))

    # Média por limiar
    all_thresholds = results[0]["AP_per_iou"].keys()
    mean_ap_per_iou = {
        t: float(np.mean([r["AP_per_iou"][t] for r in results]))
        for t in all_thresholds
    }

    return {
        "mean_mAP": mean_mAP,
        "mean_count_error": mean_count_error,
        "mean_AP_per_iou": mean_ap_per_iou,
        "total_images": len(results),
    }