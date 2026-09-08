"""Módulo de utilitários: visualização, análise estatística e gráficos de diagnóstico.

Este módulo implementa:
1. Colorização de instâncias e sobreposição em imagens originais.
2. Painéis comparativos lado a lado (Imagem, Ground Truth, Predição, Sobreposição).
3. Gráfico de quantificação de fracasso da Parte 1 (mAP / Erro de Contagem vs. Densidade de Núcleos).
4. Análise morfológica e distribuição de diâmetros/tamanhos de objetos (para a Parte 1 e Parte 5).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Colorização e Visualização de Instâncias

def generate_distinct_colors(n_colors: int, seed: int = 42) -> np.ndarray:
    """Gera uma paleta de N cores RGB distintas e vibrantes.

    Args:
        n_colors: Quantidade de cores necessárias.
        seed: Semente para reprodutibilidade das cores.

    Returns:
        Array numpy (n_colors + 1, 3) em [0.0, 1.0], onde o índice 0 é preto (fundo).
    """
    if n_colors <= 0:
        return np.zeros((1, 3), dtype=np.float32)

    rng = np.random.RandomState(seed)
    # HSV para RGB garante saturação e brilho altos para destacar sobre a microscopia
    hues = np.linspace(0.0, 1.0, n_colors, endpoint=False)
    rng.shuffle(hues)

    colors = np.zeros((n_colors + 1, 3), dtype=np.float32)  # index 0 = background
    for i, h in enumerate(hues, start=1):
        # Conversão HSV -> RGB manual simples com saturação 0.85 e valor 0.95
        s = rng.uniform(0.75, 0.95)
        v = rng.uniform(0.85, 1.0)
        c = v * s
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        m = v - c

        if h < 1.0 / 6.0:
            rgb = (c, x, 0.0)
        elif h < 2.0 / 6.0:
            rgb = (x, c, 0.0)
        elif h < 3.0 / 6.0:
            rgb = (0.0, c, x)
        elif h < 4.0 / 6.0:
            rgb = (0.0, x, c)
        elif h < 5.0 / 6.0:
            rgb = (x, 0.0, c)
        else:
            rgb = (c, 0.0, x)

        colors[i] = [rgb[0] + m, rgb[1] + m, rgb[2] + m]

    return colors


def colorize_instances(instance_mask: Union[np.ndarray, torch.Tensor], seed: int = 42) -> np.ndarray:
    """Mapeia uma máscara de instâncias 2D de inteiros para uma imagem RGB colorida.

    Args:
        instance_mask: Matriz 2D (H, W) com IDs de instâncias (0 = fundo).
        seed: Semente para manter consistência nas cores dos IDs.

    Returns:
        Array numpy float32 (H, W, 3) com valores em [0.0, 1.0].
    """
    if isinstance(instance_mask, torch.Tensor):
        instance_mask = instance_mask.detach().cpu().numpy()

    instance_mask = instance_mask.astype(np.int64)
    h, w = instance_mask.shape
    unique_ids = [int(x) for x in np.unique(instance_mask) if x != 0]

    if len(unique_ids) == 0:
        return np.zeros((h, w, 3), dtype=np.float32)

    palette = generate_distinct_colors(len(unique_ids), seed=seed)
    rgb_image = np.zeros((h, w, 3), dtype=np.float32)

    for idx, inst_id in enumerate(unique_ids, start=1):
        mask = (instance_mask == inst_id)
        rgb_image[mask] = palette[idx]

    return rgb_image


def overlay_mask_on_image(
    image: Union[np.ndarray, torch.Tensor],
    colored_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Sobrepõe uma máscara RGB colorida sobre a imagem original com transparência.

    Args:
        image: Imagem de entrada (H, W), (H, W, 3) ou (3, H, W) em [0.0, 1.0].
        colored_mask: Máscara colorida (H, W, 3) gerada por colorize_instances.
        alpha: Fator de transparência da máscara sobre a imagem (padrão 0.45).

    Returns:
        Array numpy float32 (H, W, 3) em [0.0, 1.0].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Se estiver no formato PyTorch (3, H, W), transpõe para (H, W, 3)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        if image.shape[0] == 1:
            image = np.repeat(image[0, :, :, None], 3, axis=2)
        else:
            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    foreground = (colored_mask.sum(axis=-1) > 0)[:, :, None]

    blended = np.where(foreground, (1.0 - alpha) * image + alpha * colored_mask, image)
    return np.clip(blended, 0.0, 1.0)


# 2. Painéis Comparativos e Plotagem Lado a Lado

def plot_sample_comparison(
    image: Union[np.ndarray, torch.Tensor],
    gt_instance_mask: Union[np.ndarray, torch.Tensor],
    pred_instance_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    title_prefix: str = "Amostra",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plota um painel comparativo de uma amostra: Imagem | Ground Truth | Predição | Sobreposição.

    Args:
        image: Imagem original (3, H, W) ou (H, W, 3).
        gt_instance_mask: Máscara de instâncias ground truth (H, W).
        pred_instance_mask: Máscara de instâncias predita opcional (H, W).
        metrics: Dicionário opcional de métricas para compor o título (ex: mAP, erro contagem).
        title_prefix: Título superior da figura.
        figsize: Dimensões da figura matplotlib.
        save_path: Caminho opcional para salvar a figura em disco.

    Returns:
        Objeto plt.Figure gerado.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(gt_instance_mask, torch.Tensor):
        gt_instance_mask = gt_instance_mask.detach().cpu().numpy()
    if pred_instance_mask is not None and isinstance(pred_instance_mask, torch.Tensor):
        pred_instance_mask = pred_instance_mask.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    gt_colored = colorize_instances(gt_instance_mask, seed=42)
    n_gt = len(np.unique(gt_instance_mask[gt_instance_mask != 0]))

    n_cols = 4 if pred_instance_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 2:
        axes = list(axes)

    # 1. Imagem Original
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    # 2. Ground Truth Colorido
    axes[1].imshow(gt_colored)
    axes[1].set_title(f"Ground Truth ({n_gt} instâncias)")
    axes[1].axis("off")

    if pred_instance_mask is not None:
        pred_colored = colorize_instances(pred_instance_mask, seed=42)
        n_pred = len(np.unique(pred_instance_mask[pred_instance_mask != 0]))

        # 3. Predição Colorida
        axes[2].imshow(pred_colored)
        title_pred = f"Predição ({n_pred} instâncias)"
        if metrics and "mAP" in metrics:
            title_pred += f"\nmAP: {metrics['mAP']:.3f}"
        axes[2].set_title(title_pred)
        axes[2].axis("off")

        # 4. Sobreposição (Overlay)
        overlay = overlay_mask_on_image(image, pred_colored, alpha=0.50)
        axes[3].imshow(overlay)
        axes[3].set_title("Sobreposição (Overlay)")
        axes[3].axis("off")

    full_title = title_prefix
    if metrics:
        extra_info = []
        if "count_error" in metrics:
            extra_info.append(f"Erro Contagem: {metrics['count_error']}")
        if "iou" in metrics:
            extra_info.append(f"IoU: {metrics['iou']:.3f}")
        if "dice" in metrics:
            extra_info.append(f"Dice: {metrics['dice']:.3f}")
        if extra_info:
            full_title += f" — ({', '.join(extra_info)})"

    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# Quantificação de Fracasso da Parte 1 (mAP / Erro vs. Densidade de Núcleos)

def plot_failure_vs_density(
    nuclei_counts: Sequence[int],
    map_scores: Sequence[float],
    count_errors: Optional[Sequence[int]] = None,
    save_path: Optional[str] = "failure_vs_density.png",
    title_suffix: str = "Baseline Semântica (Componentes Conexos)",
) -> plt.Figure:
    """Gera o gráfico obrigatório do Item 5 da Parte 1 do PA1.

    Enunciado do PA1:
    '5. Quantifiquem o fracasso: gráfico do mAP (ou do erro de contagem) em função da
        densidade de objetos na imagem. A tendência tem que ficar visível.'

    Args:
        nuclei_counts: Quantidade real de núcleos por imagem (densidade de objetos).
        map_scores: mAP@[.50:.95] obtido em cada imagem.
        count_errors: Erro absoluto de contagem |N_pred - N_gt| opcional.
        save_path: Caminho para salvar a figura gerada.
        title_suffix: Texto adicional para o título do gráfico.

    Returns:
        Objeto plt.Figure com a curva de dispersão e linha de tendência.
    """
    counts = np.array(nuclei_counts)
    maps = np.array(map_scores)

    has_error = count_errors is not None and len(count_errors) == len(counts)
    n_plots = 2 if has_error else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    # Ordenação para plotar tendência
    sort_idx = np.argsort(counts)
    sorted_counts = counts[sort_idx]
    sorted_maps = maps[sort_idx]

    # 1. Subplot: mAP vs Densidade de Objetos
    ax1 = axes[0]
    ax1.scatter(counts, maps, color="#1f77b4", alpha=0.65, edgecolors="none", s=35, label="Imagens")

    # Linha de tendência polinomial (grau 2) para evidenciar queda de mAP com a densidade
    if len(counts) > 3:
        z = np.polyfit(counts, maps, deg=2)
        p = np.poly1d(z)
        x_trend = np.linspace(counts.min(), counts.max(), 100)
        ax1.plot(x_trend, p(x_trend), color="#d62728", linewidth=2.5, linestyle="--", label="Tendência")

    ax1.set_xlabel("Densidade de Objetos (Número de núcleos na imagem)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("mAP@[.50:.95]", fontsize=11, fontweight="bold")
    ax1.set_title(f"mAP vs. Densidade de Objetos\n{title_suffix}", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Subplot: Erro de Contagem vs Densidade de Objetos (se fornecido)
    if has_error:
        errors = np.array(count_errors)[sort_idx]
        ax2 = axes[1]
        ax2.scatter(counts, count_errors, color="#ff7f0e", alpha=0.65, edgecolors="none", s=35, label="Imagens")

        if len(counts) > 3:
            z_err = np.polyfit(counts, count_errors, deg=1)
            p_err = np.poly1d(z_err)
            x_trend_err = np.linspace(counts.min(), counts.max(), 100)
            ax2.plot(x_trend_err, p_err(x_trend_err), color="#2ca02c", linewidth=2.5, linestyle="--", label="Tendência")

        ax2.set_xlabel("Densidade de Objetos (Número de núcleos na imagem)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Erro Absoluto de Contagem (|N_pred - N_gt|)", fontsize=11, fontweight="bold")
        ax2.set_title(f"Erro de Contagem vs. Densidade\n{title_suffix}", fontsize=12)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# Estatísticas Morfológicas (Distribuição de Diâmetros e Áreas)

def compute_nuclei_size_statistics(
    instance_masks: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Calcula estatísticas de tamanho (área e diâmetro equivalente) de todos os núcleos.

    O diâmetro equivalente é calculado como: d = 2 * sqrt(area / pi).
    Essencial para a inspeção exploratória e para a comparação do diâmetro dos objetos
    com o campo receptivo teórico do encoder na Parte 5.

    Args:
        instance_masks: Sequência de matrizes 2D (H, W) com IDs de instâncias.

    Returns:
        Dicionário com arrays:
            - 'areas': Array com a área em pixels de cada núcleo encontrado.
            - 'diameters': Array com o diâmetro equivalente em pixels de cada núcleo.
            - 'counts_per_image': Array com a quantidade de núcleos por imagem.
    """
    areas: List[int] = []
    counts_per_image: List[int] = []

    for mask in instance_masks:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        unique_ids = [int(x) for x in np.unique(mask) if x != 0]
        counts_per_image.append(len(unique_ids))

        for inst_id in unique_ids:
            area = int((mask == inst_id).sum())
            if area > 0:
                areas.append(area)

    areas_np = np.array(areas, dtype=np.float32)
    diameters_np = 2.0 * np.sqrt(areas_np / np.pi)

    return {
        "areas": areas_np,
        "diameters": diameters_np,
        "counts_per_image": np.array(counts_per_image, dtype=np.int64),
    }


def plot_dataset_statistics(
    stats: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Gera gráficos da distribuição de tamanho de núcleos e contagem por imagem.

    Args:
        stats: Dicionário retornado por compute_nuclei_size_statistics.
        save_path: Caminho para salvar a figura gerada.

    Returns:
        Objeto plt.Figure com os histogramas e estatísticas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Distribuição de Diâmetros Equivalentes
    diameters = stats["diameters"]
    ax1 = axes[0]
    ax1.hist(diameters, bins=40, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax1.axvline(np.mean(diameters), color="red", linestyle="--", linewidth=2, label=f"Média: {np.mean(diameters):.1f} px")
    ax1.axvline(np.median(diameters), color="blue", linestyle=":", linewidth=2, label=f"Mediana: {np.median(diameters):.1f} px")
    ax1.set_xlabel("Diâmetro Equivalente (pixels)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequência (Quantidade de Núcleos)", fontsize=11, fontweight="bold")
    ax1.set_title("Distribuição do Tamanho dos Núcleos (Diâmetro)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend()

    # 2. Distribuição da Contagem de Núcleos por Imagem
    counts = stats["counts_per_image"]
    ax2 = axes[1]
    ax2.hist(counts, bins=35, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax2.axvline(np.mean(counts), color="red", linestyle="--", linewidth=2, label=f"Média: {np.mean(counts):.1f} núcleos")
    ax2.axvline(np.median(counts), color="blue", linestyle=":", linewidth=2, label=f"Mediana: {np.median(counts):.1f} núcleos")
    ax2.set_xlabel("Número de Núcleos por Imagem", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Quantidade de Imagens", fontsize=11, fontweight="bold")
    ax2.set_title("Densidade de Objetos por Imagem", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig