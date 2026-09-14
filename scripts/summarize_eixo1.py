"""Consolida as duas seeds de cada arquitetura do Eixo 1 e gera gráficos comparativos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = [
    ("unet_skips", "U-Net Padrão (Skips)", "#1f77b4"),
    ("unet_noskips", "U-Net Sem Skips (Gargalo)", "#d62728"),
    (
        "deeplab_aspp",
        "Cabeça DeepLab/ASPP autoral, inspirada no DeepLabv3",
        "#2ca02c",
    ),
]
SEEDS = (42, 123)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolida resultados do Eixo 1.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/part3_eixo1"))
    return parser.parse_args()


def load_metrics(output_root: Path, config: str, seed: int) -> Dict[str, Any]:
    path = output_root / f"{config}_seed{seed}" / "metrics_dsb2018_val_hungarian.json"
    if not path.exists():
        raise FileNotFoundError(f"Métricas ausentes: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    summary: Dict[str, Any] = {
        "seeds": list(SEEDS),
        "std_definition": "desvio-padrão amostral entre seeds",
        "configurations": {},
    }
    
    table_rows = []

    for config_key, label, color in CONFIGS:
        records = [load_metrics(output_root, config_key, seed) for seed in SEEDS]
        map_summary = summarize([record["mean_mAP"] for record in records])
        cnt_summary = summarize([record["mean_count_error"] for record in records])
        iou_summary = summarize([record["mean_semantic_iou"] for record in records])
        dice_summary = summarize([record["mean_semantic_dice"] for record in records])

        summary["configurations"][config_key] = {
            "label": label,
            "per_seed": {
                str(seed): {
                    "mAP": record["mean_mAP"],
                    "count_error": record["mean_count_error"],
                    "semantic_iou": record["mean_semantic_iou"],
                    "semantic_dice": record["mean_semantic_dice"],
                }
                for seed, record in zip(SEEDS, records)
            },
            "mAP": map_summary,
            "count_error": cnt_summary,
            "semantic_iou": iou_summary,
            "semantic_dice": dice_summary,
        }

        table_rows.append({
            "key": config_key,
            "label": label,
            "color": color,
            "mAP_mean": map_summary["mean"],
            "mAP_std": map_summary["std"],
            "cnt_mean": cnt_summary["mean"],
            "cnt_std": cnt_summary["std"],
            "iou_mean": iou_summary["mean"],
            "iou_std": iou_summary["std"],
            "dice_mean": dice_summary["mean"],
            "dice_std": dice_summary["std"],
        })

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "summary.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(f"Resumo salvo em {destination}")

    # Tabela formatada no terminal
    print("\n" + "=" * 125)
    print(f"{'Arquitetura':<58} | {'mAP@[.50:.95]':<18} | {'Erro Contagem':<16} | {'IoU Semântico':<16}")
    print("-" * 125)
    for row in table_rows:
        print(f"{row['label']:<58} | {row['mAP_mean']:.4f} ± {row['mAP_std']:.4f}   | {row['cnt_mean']:.2f} ± {row['cnt_std']:.2f}     | {row['iou_mean']:.4f} ± {row['iou_std']:.4f}")
    print("=" * 125 + "\n")

    # Gráficos comparativos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    plot_label_by_key = {
        "unet_skips": "U-Net\n(com skips)",
        "unet_noskips": "U-Net\n(sem skips)",
        "deeplab_aspp": "Cabeça DeepLab/ASPP autoral\n(inspirada no DeepLabv3)",
    }
    labels = [plot_label_by_key[r["key"]] for r in table_rows]
    maps = [r["mAP_mean"] for r in table_rows]
    map_errs = [r["mAP_std"] for r in table_rows]
    colors = [r["color"] for r in table_rows]

    cnts = [r["cnt_mean"] for r in table_rows]
    cnt_errs = [r["cnt_std"] for r in table_rows]

    # Painel 1: mAP
    bars1 = ax1.bar(labels, maps, yerr=map_errs, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.set_title("mAP@[.50:.95] por Arquitetura", fontsize=12, fontweight="bold")
    ax1.set_ylabel("mAP Médio", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.set_ylim(0, max(maps) * 1.25)
    for bar, m in zip(bars1, maps):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.015, f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Painel 2: Erro de Contagem
    bars2 = ax2.bar(labels, cnts, yerr=cnt_errs, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax2.set_title("Erro Médio de Contagem por Arquitetura", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Erro de Contagem (|N_pred - N_gt|)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.set_ylim(0, max(cnts) * 1.3)
    for bar, c in zip(bars2, cnts):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{c:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = output_root / "architecture_ablation.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Gráfico comparativo salvo em: {plot_path}")


if __name__ == "__main__":
    main()
