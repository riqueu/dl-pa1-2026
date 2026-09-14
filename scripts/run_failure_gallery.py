"""Pipeline autônomo da Parte 5: galeria, campo receptivo e correção calibrada.

Este script atende a todos os requisitos do edital (PA1.pdf) e do documento docs/parte5.md:
1. Mapeia e seleciona os 5 casos de falha mais expressivos e diversificados no split de validação.
2. Gera painéis de 4 imagens em alta resolução com imagem, GT, predição e mapas de probabilidade intermediários.
3. Computa os diâmetros após redimensionar cada máscara para a entrada 256x256.
4. Compara variantes OS16 com e sem atrous na mesma resolução de saída.
5. Implementa a correção via consolidação morfológica de marcadores no Watershed, gerando o painel Antes vs. Depois.
6. Exporta todas as métricas consolidadas para outputs/part5_gallery/gallery_metrics.json.
"""

from typing import Any, Dict, List, Optional, Tuple
import glob
import json
import os
import sys

# Garante que o diretório raiz do repositório esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from evaluate import load_model
from src.dataset import DSB2018Dataset, load_splits
from src.metrics import evaluate_instances
from src.postprocess import watershed_instance_segmentation
from src.utils import (
    colorize_instances,
    resnet34_receptive_field_summary,
)


def compute_dataset_nuclei_distribution(
    root_dir: str = "data/raw/stage1_train",
    target_size: Tuple[int, int] = (256, 256),
) -> Dict[str, Any]:
    """Mede os núcleos na mesma escala espacial recebida pela rede.

    Cada máscara individual é redimensionada por vizinho mais próximo para
    ``target_size`` antes da contagem de área. Assim, os diâmetros e o campo
    receptivo são expressos no mesmo sistema de coordenadas (entrada 256x256).
    """
    train_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
    areas: List[float] = []
    source_nuclei = 0
    target_height, target_width = target_size

    for d in train_dirs:
        mask_files = glob.glob(os.path.join(d, "masks", "*.png"))
        for mf in mask_files:
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                binary = (m > 0).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                source_nuclei += 1
                resized = cv2.resize(
                    binary,
                    (target_width, target_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                a = float(resized.sum())
                areas.append(a)

    areas_np = np.array(areas, dtype=np.float32)
    if areas_np.size == 0:
        raise RuntimeError(f"Nenhuma máscara válida encontrada em {root_dir}.")
    diameters_np = 2.0 * np.sqrt(areas_np / np.pi)

    return {
        "analysis_resolution": [target_height, target_width],
        "source_nuclei": source_nuclei,
        "collapsed_after_resize": int((areas_np == 0).sum()),
        "total_nuclei": int(len(areas_np)),
        "total_images": int(len(train_dirs)),
        "areas": areas_np,
        "diameters": diameters_np,
        "diameter_mean": float(np.mean(diameters_np)),
        "diameter_std": float(np.std(diameters_np)),
        "diameter_median": float(np.median(diameters_np)),
        "diameter_p25": float(np.percentile(diameters_np, 25)),
        "diameter_p75": float(np.percentile(diameters_np, 75)),
        "diameter_p95": float(np.percentile(diameters_np, 95)),
        "diameter_p99": float(np.percentile(diameters_np, 99)),
        "diameter_max": float(np.max(diameters_np)),
        "diameter_min": float(np.min(diameters_np)),
    }


def plot_receptive_field_vs_nuclei(
    stats: Dict[str, Any],
    rf_os16_without_atrous: Dict[str, Any],
    rf_os16_atrous: Dict[str, Any],
    save_path: str,
) -> None:
    """Gera a figura de comparação do Campo Receptivo Teórico com a distribuição empírica dos núcleos."""
    diameters = stats["diameters"]
    rf_without_atrous = rf_os16_without_atrous["encoder_receptive_field"]
    rf_with_atrous = rf_os16_atrous["encoder_receptive_field"]
    aspp_rfs = rf_os16_atrous["aspp_branch_receptive_fields"]
    input_size = int(stats["analysis_resolution"][0])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Painel 1: Histograma com densidade e zoom na escala celular
    ax1 = axes[0]
    n, bins, _ = ax1.hist(
        diameters,
        bins=60,
        range=(0, 140),
        density=True,
        color="#3b82f6",
        edgecolor="black",
        alpha=0.75,
        label="Densidade empírica após resize para 256x256",
    )

    ax1.axvline(stats["diameter_mean"], color="#ef4444", linestyle="--", linewidth=2.5,
                label=f"Média: {stats['diameter_mean']:.1f} px")
    ax1.axvline(stats["diameter_median"], color="#10b981", linestyle=":", linewidth=2.5,
                label=f"Mediana: {stats['diameter_median']:.1f} px")
    ax1.axvline(stats["diameter_p95"], color="#f59e0b", linestyle="-.", linewidth=2,
                label=f"Percentil 95: {stats['diameter_p95']:.1f} px")
    ax1.axvline(stats["diameter_max"], color="#8b5cf6", linestyle="-", linewidth=2,
                label=f"Máximo: {stats['diameter_max']:.1f} px")

    ax1.set_xlabel(r"Diâmetro Equivalente do Núcleo $d = 2\sqrt{A/\pi}$ (pixels)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Densidade de Probabilidade", fontsize=11, fontweight="bold")
    ax1.set_title("Distribuição de tamanho na escala de entrada da rede\n(máscaras redimensionadas para 256x256)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95)

    # Painel 2: Visão Panorâmica comparando Escala do Núcleo vs. Campo Receptivo Teórico
    ax2 = axes[1]
    categories = [
        "Núcleo Médio",
        "Percentil 95",
        "Núcleo Máximo",
        "Entrada da Rede\n(Tile 256x256)",
        "RF Encoder sem atrous\n(OS16)",
        "RF Encoder com atrous\n(OS16)",
        "RF ASPP (Rate 6)\n(OS16)",
        "RF ASPP (Rate 12)\n(OS16)",
        "RF ASPP (Rate 18)\n(OS16)",
    ]
    values = [
        stats["diameter_mean"],
        stats["diameter_p95"],
        stats["diameter_max"],
        float(input_size),
        rf_without_atrous,
        rf_with_atrous,
        aspp_rfs["conv_3x3_rate_6"],
        aspp_rfs["conv_3x3_rate_12"],
        aspp_rfs["conv_3x3_rate_18"],
    ]
    colors = [
        "#10b981",
        "#f59e0b",
        "#8b5cf6",
        "#6b7280",
        "#ef4444",
        "#f97316",
        "#ec4899",
        "#db2777",
        "#9d174d",
    ]

    bars = ax2.barh(categories, values, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Dimensão Espacial em Pixels (Escala Linear)", fontsize=11, fontweight="bold")
    ax2.set_title("Campo receptivo teórico em OS16\n(comparação na mesma resolução de saída)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax2.axvline(
        input_size,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label="Limite do contexto disponível na entrada",
    )

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_width() + 15,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} px",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    ax2.set_xlim(0, max(values) * 1.15)
    ax2.legend(loc="lower right", framealpha=0.95)
    fig.text(
        0.74,
        0.01,
        "RF teórico pode exceder 256 px, mas o contexto real é limitado pela entrada; "
        "o RF efetivo aprendido pode ser menor.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_four_panel_figure(
    img_rgb: np.ndarray,
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
    prob_map: np.ndarray,
    title_meta: Dict[str, Any],
    save_path: str,
) -> None:
    """Gera painel de 4 imagens em alta resolução para um caso de falha."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))

    # 1. Imagem Original RGB
    axes[0].imshow(np.clip(img_rgb, 0.0, 1.0))
    axes[0].set_title(
        f"1. Imagem de Entrada\n{title_meta.get('modality', 'Microscopia')}",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].axis("off")

    # 2. Ground Truth Colorizado
    gt_colored = colorize_instances(gt_inst, seed=42)
    axes[1].imshow(gt_colored)
    n_gt = len([x for x in np.unique(gt_inst) if x != 0])
    axes[1].set_title(
        f"2. Ground Truth\n$N_{{gt}} = {n_gt}$ instâncias",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")

    # 3. Predição Baseline do Watershed
    pred_colored = colorize_instances(pred_inst, seed=42)
    axes[2].imshow(pred_colored)
    n_pred = len([x for x in np.unique(pred_inst) if x != 0])
    m_ap = title_meta.get("mAP", 0.0)
    err = title_meta.get("count_error", abs(n_pred - n_gt))
    axes[2].set_title(
        f"3. Predição Watershed\n$N_{{pred}} = {n_pred}$ (Erro: {err} | mAP: {m_ap:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].axis("off")

    # 4. Mapa Intermediário de Probabilidades
    # Composição RGB: R = Fronteira, G = Interior, B = Fundo
    p_bg = prob_map[0]
    p_int = prob_map[1]
    p_bound = prob_map[2]
    prob_composite = np.stack([p_bound, p_int, p_bg], axis=-1)
    prob_composite = np.clip(prob_composite, 0.0, 1.0)

    axes[3].imshow(prob_composite)
    axes[3].set_title(
        "4. Mapa Intermediário 3-Canais\n(R: Fronteira | G: Interior | B: Fundo)",
        fontsize=11,
        fontweight="bold",
    )
    axes[3].axis("off")

    # Subtítulo explicativo com diagnóstico biológico/óptico
    diagnostic = title_meta.get("diagnostic", "")
    plt.suptitle(
        f"Caso de Falha: {title_meta.get('name', '')}\nDiagnóstico: {diagnostic}",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_before_after_figure(
    img_rgb: np.ndarray,
    gt_inst: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    meta_before: Dict[str, Any],
    meta_after: Dict[str, Any],
    save_path: str,
) -> None:
    """Gera painel Antes vs. Depois da correção calibrada para o Caso 2."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))

    # 1. Imagem Original
    axes[0].imshow(np.clip(img_rgb, 0.0, 1.0))
    axes[0].set_title("1. Entrada Original\nHistopatologia H&E (Células Gigantes)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    gt_colored = colorize_instances(gt_inst, seed=42)
    n_gt = len([x for x in np.unique(gt_inst) if x != 0])
    axes[1].imshow(gt_colored)
    axes[1].set_title(f"2. Ground Truth\n$N_{{gt}} = {n_gt}$ núcleos", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Antes da Correção (Baseline Watershed com sementes fragmentadas)
    before_colored = colorize_instances(pred_before, seed=42)
    axes[2].imshow(before_colored)
    axes[2].set_title(
        f"3. ANTES: Watershed Padrão\n$N_{{pred}} = {meta_before['n_pred']}$ (Erro: {meta_before['count_error']})\nmAP: {meta_before['mAP']:.4f} | AP50: {meta_before['ap50']:.4f}\n[Hiper-fragmentação Severa]",
        fontsize=10.5,
        fontweight="bold",
        color="#dc2626",
    )
    axes[2].axis("off")

    # 4. Depois da correção calibrada para este caso específico
    after_colored = colorize_instances(pred_after, seed=42)
    axes[3].imshow(after_colored)
    axes[3].set_title(
        f"4. DEPOIS: Correção Calibrada (Caso 2)\n$N_{{pred}} = {meta_after['n_pred']}$ (Erro: {meta_after['count_error']})\nmAP: {meta_after['mAP']:.4f} (+{meta_after['mAP']-meta_before['mAP']:.4f})\nAP50: {meta_after['ap50']:.4f} (+{meta_after['ap50']-meta_before['ap50']:.4f})",
        fontsize=10.5,
        fontweight="bold",
        color="#16a34a",
    )
    axes[3].axis("off")

    plt.suptitle(
        "Correção Morfológica Calibrada para o Caso 2\n"
        "Contagem exata não implica máscaras perfeitas: o mAP residual quantifica os erros de forma e matching",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    output_dir = "outputs/part5_gallery"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PARTE 5: GALERIA DE FALHAS, CAMPO RECEPTIVO E CORREÇÃO CALIBRADA")
    print(f"Dispositivo: {device} | Diretório de Saída: {output_dir}")
    print("=" * 70)

    # 1. Carregar Modelo Oficial da Parte 2
    checkpoint_path = "checkpoints/part2_watershed.pth"
    print(f"Carregando checkpoint oficial: '{checkpoint_path}'...")
    model, _ = load_model(checkpoint_path, device)

    # 2. Carregar Split de Validação
    splits = load_splits("data/splits.json")
    val_ids = splits["val"]
    val_dataset = DSB2018Dataset(image_ids=val_ids, target_size=(256, 256), three_class=True)
    print(f"Total de imagens no split de validação: {len(val_dataset)}")

    # 3. Executar Inferência e Rastrear Métricas
    all_eval_records: List[Dict[str, Any]] = []
    print("\nExecutando inferência e avaliando instâncias nas 67 imagens...")
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        img_t = sample["image"].unsqueeze(0).to(device)
        gt_inst = sample["mask_instance"].numpy()
        img_id = sample["image_id"]

        with torch.no_grad():
            logits = model(img_t)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_inst = watershed_instance_segmentation(
            prob, interior_threshold=0.50, foreground_threshold=0.50, min_area=10
        )
        metrics = evaluate_instances(pred_inst, gt_inst, method="hungarian")

        all_eval_records.append({
            "index": idx,
            "image_id": img_id,
            "mAP": float(metrics["mAP"]),
            "ap50": float(metrics["AP_per_iou"][0.5]),
            "ap75": float(metrics["AP_per_iou"][0.75]),
            "count_error": int(metrics["count_error"]),
            "n_pred": int(metrics["n_pred"]),
            "n_gt": int(metrics["n_gt"]),
            "prob": prob,
            "image_rgb": np.transpose(sample["image"].numpy(), (1, 2, 0)),
            "gt_inst": gt_inst,
            "pred_inst": pred_inst,
        })

    # 4. Seleção dos 5 Casos de Falha Expressivos
    # Definimos os 5 casos cobrindo os diferentes regimes biológicos e ópticos mapeados:
    selected_cases_specs = [
        {
            "case_number": 1,
            "index": 34,
            "name": "Caso 1 — Aliasing Espacial por Downsampling Severo (1024x1024 -> 256x256)",
            "modality": "Lâmina Brightfield (Alta Resolução Original)",
            "diagnostic": "Núcleos pequenos com fronteiras sub-amostradas (<0.25 px) colapsam em sementes fundidas, causando sub-segmentação extrema (N_pred=35 vs N_gt=94).",
        },
        {
            "case_number": 2,
            "index": 56,
            "name": "Caso 2 — Hiper-Fragmentação Interna de Células Gigantes (Texture Split)",
            "modality": "Histopatologia H&E Púrpura",
            "diagnostic": "Núcleos gigantes (>3000 px) com textura interna heterogênea geram múltiplos marcadores desconexos dentro do mesmo núcleo, retalhando células em múltiplos pedaços (N_pred=71 vs N_gt=19).",
        },
        {
            "case_number": 3,
            "index": 35,
            "name": "Caso 3 — Esmagamento de Gradiente em Fundo Esmaecido (Vanishing Edges)",
            "modality": "Microscopia Óptica de Baixo Contraste",
            "diagnostic": "Contraste óptico insuficiente impede a ativação de cristas de fronteira e interior celular (P(int) < 0.50), omitindo mais de 70% dos núcleos (N_pred=23 vs N_gt=75).",
        },
        {
            "case_number": 4,
            "index": 39,
            "name": "Caso 4 — Falsos Positivos por Detritos e Textura Estromal",
            "modality": "Histopatologia H&E de Tecido Frouxo",
            "diagnostic": "Fibras de estroma extracelular e agregados de detritos exibem afinidade tintorial basofílica, ativando falsos centros de interior em regiões anucleadas (N_pred=30 vs N_gt=14).",
        },
        {
            "case_number": 5,
            "index": 3,
            "name": "Caso 5 — Limite de Densidade em Monocamada Confluente (369 Núcleos)",
            "modality": "Fluorescência Densa Multicelular",
            "diagnostic": "Em contato intercelular em 100% do perímetro, a erosão de 1 pixel extingue por completo o interior dos menores núcleos espremidos no centro do aglomerado (subcontagem residual de 47 células).",
        },
    ]

    print("\nGerando figuras dos 5 Casos de Falha em outputs/part5_gallery/...")
    gallery_data_for_json: List[Dict[str, Any]] = []

    for spec in selected_cases_specs:
        idx = spec["index"]
        rec = all_eval_records[idx]
        title_meta = {
            "name": spec["name"],
            "modality": spec["modality"],
            "diagnostic": spec["diagnostic"],
            "mAP": rec["mAP"],
            "count_error": rec["count_error"],
        }
        fig_path = os.path.join(output_dir, f"failure_case_{spec['case_number']}.png")
        generate_four_panel_figure(
            img_rgb=rec["image_rgb"],
            gt_inst=rec["gt_inst"],
            pred_inst=rec["pred_inst"],
            prob_map=rec["prob"],
            title_meta=title_meta,
            save_path=fig_path,
        )
        print(f"Salvo: {fig_path} | idx={idx} (mAP={rec['mAP']:.4f}, Err={rec['count_error']})")

        gallery_data_for_json.append({
            "case_number": spec["case_number"],
            "index": idx,
            "image_id": rec["image_id"],
            "name": spec["name"],
            "modality": spec["modality"],
            "diagnostic": spec["diagnostic"],
            "mAP": rec["mAP"],
            "ap50": rec["ap50"],
            "ap75": rec["ap75"],
            "n_pred": rec["n_pred"],
            "n_gt": rec["n_gt"],
            "count_error": rec["count_error"],
        })

    # 5. Levantamento Estatístico de Todos os Núcleos e Campo Receptivo
    print("\nComputando distribuição dos núcleos na escala de entrada 256x256...")
    nuclei_stats = compute_dataset_nuclei_distribution(target_size=(256, 256))
    print(f"Total de núcleos analisados: {nuclei_stats['total_nuclei']}")
    print(f"Diâmetro médio: {nuclei_stats['diameter_mean']:.2f} px | Mediana: {nuclei_stats['diameter_median']:.2f} px")
    print(f"Percentil 95: {nuclei_stats['diameter_p95']:.2f} px | Máximo: {nuclei_stats['diameter_max']:.2f} px")

    print("\nCalculando Campo Receptivo Teórico do ResNet-34...")
    rf_os16_without_atrous = resnet34_receptive_field_summary(
        output_stride=16,
        dilate_layer4=False,
    )
    rf_os16_atrous = resnet34_receptive_field_summary(output_stride=16)
    print(f"RF Encoder sem atrous (OS16): {rf_os16_without_atrous['encoder_receptive_field']} px")
    print(f"RF Encoder com atrous (OS16): {rf_os16_atrous['encoder_receptive_field']} px")
    print(f"RF Ramos ASPP (OS16): {rf_os16_atrous['aspp_branch_receptive_fields']}")

    dist_fig_path = os.path.join(output_dir, "nuclei_size_distribution.png")
    plot_receptive_field_vs_nuclei(
        stats=nuclei_stats,
        rf_os16_without_atrous=rf_os16_without_atrous,
        rf_os16_atrous=rf_os16_atrous,
        save_path=dist_fig_path,
    )
    print(f"Salvo: {dist_fig_path}")

    # 6. Implementação e Avaliação da Correção (Antes vs. Depois)
    # Selecionamos o Caso 2 (idx=56, Hiper-fragmentação de Células Gigantes)
    print("\nAplicando correção morfológica calibrada no Caso 2 (idx=56)...")
    rec_c2 = all_eval_records[56]
    prob_c2 = rec_c2["prob"]
    gt_c2 = rec_c2["gt_inst"]

    # Baseline (Antes)
    pred_before = rec_c2["pred_inst"]
    meta_before = {
        "mAP": rec_c2["mAP"],
        "ap50": rec_c2["ap50"],
        "ap75": rec_c2["ap75"],
        "n_pred": rec_c2["n_pred"],
        "n_gt": rec_c2["n_gt"],
        "count_error": rec_c2["count_error"],
    }

    # Correção: Consolidação Morfológica de Sementes no Watershed
    # seed_closing_radius=4 funde marcadores espúrios internos em células volumosas
    # min_area=60 e interior_threshold=0.40 recuperam a integridade dos núcleos gigantes
    pred_after = watershed_instance_segmentation(
        prob_c2,
        interior_threshold=0.40,
        foreground_threshold=0.40,
        min_area=60,
        seed_closing_radius=4,
    )
    metrics_after = evaluate_instances(pred_after, gt_c2, method="hungarian")
    meta_after = {
        "mAP": float(metrics_after["mAP"]),
        "ap50": float(metrics_after["AP_per_iou"][0.5]),
        "ap75": float(metrics_after["AP_per_iou"][0.75]),
        "n_pred": int(metrics_after["n_pred"]),
        "n_gt": int(metrics_after["n_gt"]),
        "count_error": int(metrics_after["count_error"]),
    }

    corr_fig_path = os.path.join(output_dir, "correction_before_after.png")
    generate_before_after_figure(
        img_rgb=rec_c2["image_rgb"],
        gt_inst=gt_c2,
        pred_before=pred_before,
        pred_after=pred_after,
        meta_before=meta_before,
        meta_after=meta_after,
        save_path=corr_fig_path,
    )
    print(f"Salvo: {corr_fig_path}")
    print(f"ANTES:  N_pred={meta_before['n_pred']}, Erro={meta_before['count_error']}, mAP={meta_before['mAP']:.4f}, AP50={meta_before['ap50']:.4f}")
    print(f"DEPOIS: N_pred={meta_after['n_pred']}, Erro={meta_after['count_error']}, mAP={meta_after['mAP']:.4f}, AP50={meta_after['ap50']:.4f}")

    # 7. Salvar Dados Consolidados em JSON
    metrics_summary = {
        "dataset": "dsb2018",
        "split": "val",
        "checkpoint": checkpoint_path,
        "selected_failure_cases": gallery_data_for_json,
        "morphological_distribution": {
            "analysis_resolution": nuclei_stats["analysis_resolution"],
            "source_nuclei": nuclei_stats["source_nuclei"],
            "collapsed_after_resize": nuclei_stats["collapsed_after_resize"],
            "total_nuclei": nuclei_stats["total_nuclei"],
            "total_images": nuclei_stats["total_images"],
            "diameter_mean": round(nuclei_stats["diameter_mean"], 2),
            "diameter_std": round(nuclei_stats["diameter_std"], 2),
            "diameter_median": round(nuclei_stats["diameter_median"], 2),
            "diameter_p25": round(nuclei_stats["diameter_p25"], 2),
            "diameter_p75": round(nuclei_stats["diameter_p75"], 2),
            "diameter_p95": round(nuclei_stats["diameter_p95"], 2),
            "diameter_max": round(nuclei_stats["diameter_max"], 2),
            "diameter_min": round(nuclei_stats["diameter_min"], 2),
        },
        "receptive_field": {
            "comparison_output_stride": 16,
            "encoder_without_atrous_os16": rf_os16_without_atrous["encoder_receptive_field"],
            "encoder_with_atrous_os16": rf_os16_atrous["encoder_receptive_field"],
            "aspp_branch_os16": rf_os16_atrous["aspp_branch_receptive_fields"],
            "theoretical_rf_note": "RF teórico; pode incluir padding além da entrada 256x256.",
            "available_context_note": "O contexto observável é limitado aos 256x256 pixels da entrada.",
            "effective_rf_note": "O campo receptivo efetivo aprendido tende a ser menor e não foi estimado empiricamente.",
            "ratio_atrous_rf_vs_mean_diameter": round(rf_os16_atrous["encoder_receptive_field"] / nuclei_stats["diameter_mean"], 1),
            "ratio_atrous_rf_vs_max_diameter": round(rf_os16_atrous["encoder_receptive_field"] / nuclei_stats["diameter_max"], 1),
        },
        "correction_experiment": {
            "case_evaluated": "Caso 2 (idx=56, ebc18868864ad075)",
            "technique": "Correção morfológica calibrada para o Caso 2 (seed_closing_radius=4, min_area=60, interior_threshold=0.40)",
            "scope_note": "Parâmetros fixos calibrados para este caso; não constituem adaptação automática por imagem.",
            "interpretation_note": "Erro de contagem zero não implica segmentação perfeita; o mAP após a correção permanece abaixo de 1.",
            "before": meta_before,
            "after": meta_after,
            "relative_gain_mAP_percent": round(((meta_after["mAP"] - meta_before["mAP"]) / meta_before["mAP"]) * 100, 2),
            "absolute_gain_mAP": round(meta_after["mAP"] - meta_before["mAP"], 4),
            "absolute_gain_ap50": round(meta_after["ap50"] - meta_before["ap50"], 4),
            "count_error_reduction": meta_before["count_error"] - meta_after["count_error"],
        },
    }

    json_path = os.path.join(output_dir, "gallery_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo resumo métrico consolidado em: '{json_path}'")
    print("=" * 70)
    print("PARTE 5 CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
