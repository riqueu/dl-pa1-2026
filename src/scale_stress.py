"""Primitivas do teste de estresse por mudanca de escala da Parte 6.

O modulo mantem separadas tres operacoes para permitir testes deterministas:
redimensionar a entrada, decodificar as probabilidades na escala de inferencia e
agregar as metricas de instancia. O modelo e o watershed nao sao alterados.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.postprocess import watershed_instance_segmentation


def parse_scale_factors(spec: str) -> Tuple[float, ...]:
    """Converte ``"0.5,1.0,2.0"`` em fatores positivos sem repeticao."""
    try:
        factors = tuple(float(item.strip()) for item in spec.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("As escalas devem ser numeros separados por virgula.") from exc
    if not factors or any(not np.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("Informe pelo menos uma escala finita e maior que zero.")
    if len(set(factors)) != len(factors):
        raise ValueError("As escalas nao podem se repetir.")
    return factors


def scaled_spatial_size(native_size: Sequence[int], scale: float) -> Tuple[int, int]:
    """Calcula ``round(H*s), round(W*s)`` garantindo pelo menos um pixel."""
    if len(native_size) != 2:
        raise ValueError("native_size deve conter (H, W).")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale deve ser finito e maior que zero.")
    height, width = (int(value) for value in native_size)
    if height <= 0 or width <= 0:
        raise ValueError("As dimensoes espaciais devem ser positivas.")
    return max(1, round(height * scale)), max(1, round(width * scale))


def resolve_min_area(base_min_area: int, scale: float, protocol: str = "fixed") -> int:
    """Resolve a area minima em pixels para o protocolo experimental.

    ``fixed`` congela integralmente o pos-processamento de producao. ``scaled``
    preserva aproximadamente a mesma area fisica, pois areas variam com ``s^2``.
    """
    if base_min_area < 0:
        raise ValueError("base_min_area deve ser maior ou igual a zero.")
    if protocol == "fixed":
        return int(base_min_area)
    if protocol == "scaled":
        return 0 if base_min_area == 0 else max(1, round(base_min_area * scale**2))
    raise ValueError("protocol deve ser 'fixed' ou 'scaled'.")


def resize_image_batch(images: torch.Tensor, scale: float) -> torch.Tensor:
    """Redimensiona um lote BCHW com bilinear e antialiasing."""
    if images.ndim != 4:
        raise ValueError(f"Esperado lote (B, C, H, W); recebido {tuple(images.shape)}.")
    target_size = scaled_spatial_size(images.shape[-2:], scale)
    if target_size == tuple(images.shape[-2:]):
        return images
    return F.interpolate(
        images,
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def restore_instance_mask(mask: np.ndarray, native_size: Sequence[int]) -> np.ndarray:
    """Restaura uma mascara categorica por vizinho mais proximo e preserva IDs."""
    labels = np.asarray(mask)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("mask deve ser uma matriz 2D de inteiros.")
    target_size = tuple(int(value) for value in native_size)
    if labels.shape == target_size:
        return labels.astype(np.int64, copy=True)
    tensor = torch.from_numpy(labels.astype(np.float64, copy=False))[None, None]
    restored = F.interpolate(tensor, size=target_size, mode="nearest-exact")
    return restored[0, 0].to(torch.int64).numpy()


def predict_scaled_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    scale: float,
) -> np.ndarray:
    """Executa o modelo na escala solicitada e devolve softmax em NumPy."""
    scaled_images = resize_image_batch(images, scale)
    logits = model(scaled_images)
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("A Parte 6 requer checkpoint com saida de 3 classes.")
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def decode_scaled_probabilities(
    probabilities: np.ndarray,
    native_size: Sequence[int],
    scale: float,
    interior_threshold: float = 0.5,
    foreground_threshold: float = 0.5,
    base_min_area: int = 10,
    area_protocol: str = "fixed",
    connectivity: int = 1,
) -> List[np.ndarray]:
    """Aplica watershed na escala de inferencia e retorna IDs na resolucao 1x."""
    batch = np.asarray(probabilities)
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Esperado shape (B, 3, H, W); recebido {batch.shape}.")
    min_area = resolve_min_area(base_min_area, scale, area_protocol)
    predictions: List[np.ndarray] = []
    for probability_map in batch:
        scaled_mask = watershed_instance_segmentation(
            probability_map,
            interior_threshold=interior_threshold,
            foreground_threshold=foreground_threshold,
            min_area=min_area,
            connectivity=connectivity,
        )
        predictions.append(restore_instance_mask(scaled_mask, native_size))
    return predictions


def aggregate_instance_metrics(results: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Agrega a metrica DSB por imagem, incluindo AP50 e AP75."""
    if not results:
        raise ValueError("results nao pode ser vazio.")

    def values(field: str) -> np.ndarray:
        return np.asarray([float(item[field]) for item in results], dtype=np.float64)

    maps = values("mAP")
    count_errors = values("count_error")
    ap50 = np.asarray([float(item["AP_per_iou"][0.5]) for item in results])
    ap75 = np.asarray([float(item["AP_per_iou"][0.75]) for item in results])

    def sample_std(array: np.ndarray) -> float:
        return float(np.std(array, ddof=1)) if array.size > 1 else 0.0

    return {
        "n_images": int(len(results)),
        "mean_mAP": float(np.mean(maps)),
        "std_mAP": sample_std(maps),
        "mean_AP50": float(np.mean(ap50)),
        "std_AP50": sample_std(ap50),
        "mean_AP75": float(np.mean(ap75)),
        "std_AP75": sample_std(ap75),
        "mean_count_error": float(np.mean(count_errors)),
        "std_count_error": sample_std(count_errors),
    }


def add_native_relative_metrics(
    summaries: Mapping[float, Mapping[str, float]],
    native_scale: float = 1.0,
) -> Dict[float, Dict[str, Any]]:
    """Acrescenta delta, retencao e degradacao percentual relativos a 1x."""
    if native_scale not in summaries:
        raise ValueError(f"A escala de referencia {native_scale} nao esta nos resultados.")
    native_map = float(summaries[native_scale]["mean_mAP"])
    enriched: Dict[float, Dict[str, Any]] = {}
    for scale, summary in summaries.items():
        item = dict(summary)
        current = float(summary["mean_mAP"])
        item["mAP_delta_vs_1x"] = current - native_map
        if native_map > 0:
            item["mAP_retention_vs_1x"] = current / native_map
            item["mAP_degradation_percent_vs_1x"] = 100.0 * (native_map - current) / native_map
        else:
            item["mAP_retention_vs_1x"] = None
            item["mAP_degradation_percent_vs_1x"] = None
        enriched[float(scale)] = item
    return enriched
