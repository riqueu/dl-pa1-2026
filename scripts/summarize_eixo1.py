"""Consolida as duas seeds de cada arquitetura do Eixo 1."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List


CONFIGS = ("unet_skips", "unet_noskips", "deeplab_aspp")
SEEDS = (42, 123)
OUTPUT_ROOT = Path("outputs/part3_eixo1")


def load_metrics(config: str, seed: int) -> Dict[str, Any]:
    path = OUTPUT_ROOT / f"{config}_seed{seed}" / "metrics_dsb2018_val_hungarian.json"
    if not path.exists():
        raise FileNotFoundError(f"Métricas ausentes: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values),
    }


def main() -> None:
    summary: Dict[str, Any] = {
        "seeds": list(SEEDS),
        "std_definition": "desvio-padrão amostral entre seeds",
        "configurations": {},
    }
    for config in CONFIGS:
        records = [load_metrics(config, seed) for seed in SEEDS]
        summary["configurations"][config] = {
            "per_seed": {
                str(seed): {
                    "mAP": record["mean_mAP"],
                    "count_error": record["mean_count_error"],
                    "semantic_iou": record["mean_semantic_iou"],
                    "semantic_dice": record["mean_semantic_dice"],
                }
                for seed, record in zip(SEEDS, records)
            },
            "mAP": summarize([record["mean_mAP"] for record in records]),
            "count_error": summarize([record["mean_count_error"] for record in records]),
            "semantic_iou": summarize([record["mean_semantic_iou"] for record in records]),
            "semantic_dice": summarize([record["mean_semantic_dice"] for record in records]),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / "summary.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(f"Resumo salvo em {destination}")


if __name__ == "__main__":
    main()
