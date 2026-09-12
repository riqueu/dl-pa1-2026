"""Executa o teste de estresse por mudanca de escala da Parte 6.

Compara a U-Net da Parte 2 e o DeepLab/ASPP autoral da Parte 3 em 0.5x,
1.0x e 2.0x. A inferencia ocorre na resolucao escalada; a mascara categorica
retorna a 256x256 por vizinho mais proximo antes do matching Hungarian.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate import load_model
from src.dataset import DSB2018Dataset, load_splits
from src.metrics import evaluate_instances
from src.scale_stress import (
    add_native_relative_metrics,
    aggregate_instance_metrics,
    decode_scaled_probabilities,
    parse_scale_factors,
    predict_scaled_probabilities,
    resize_image_batch,
    resolve_min_area,
)
from src.utils import colorize_instances


MODEL_LABELS = {
    "unet_watershed": "U-Net ResNet-34",
    "deeplab_aspp": "DeepLab/ASPP autoral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parte 6 - teste de estresse por mudanca de escala",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--unet-checkpoint", default="checkpoints/part2_watershed.pth")
    parser.add_argument(
        "--aspp-checkpoint",
        default="checkpoints/part3_eixo1/deeplab_aspp_seed42.pth",
    )
    parser.add_argument("--data-dir", default="data/raw/stage1_train")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--scales", default="0.5,1.0,2.0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--interior-threshold", type=float, default=0.5)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--skip-area-control",
        action="store_true",
        help="Nao executa o controle adicional com min_area proporcional a escala ao quadrado.",
    )
    parser.add_argument("--output-dir", default="outputs/part6_stress")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def checkpoint_metadata(path: str) -> Dict[str, Any]:
    """Le metadados primitivos do checkpoint sem serializar os pesos."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        return {"path": path, "model_args": {}, "epoch": None, "metrics": {}}
    return {
        "path": path,
        "model_args": state.get("model_args", {}),
        "epoch": state.get("epoch"),
        "metrics": state.get("metrics", {}),
    }


def compact_result(image_id: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "mAP": float(result["mAP"]),
        "AP50": float(result["AP_per_iou"][0.5]),
        "AP75": float(result["AP_per_iou"][0.75]),
        "count_error": int(result["count_error"]),
        "n_pred": int(result["n_pred"]),
        "n_gt": int(result["n_gt"]),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    scales: Sequence[float],
    protocols: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Avalia um modelo uma unica vez por escala e reutiliza as probabilidades."""
    per_image: Dict[str, Dict[float, List[Dict[str, Any]]]] = {
        protocol: {scale: [] for scale in scales} for protocol in protocols
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Imagens", leave=False):
            images = batch["image"].to(device)
            gt_masks = batch["mask_instance"].cpu().numpy()
            image_ids = list(batch["image_id"])
            native_size = tuple(images.shape[-2:])

            for scale in scales:
                probabilities = predict_scaled_probabilities(model, images, scale)
                for protocol in protocols:
                    predictions = decode_scaled_probabilities(
                        probabilities,
                        native_size=native_size,
                        scale=scale,
                        interior_threshold=args.interior_threshold,
                        foreground_threshold=args.foreground_threshold,
                        base_min_area=args.min_area,
                        area_protocol=protocol,
                        connectivity=args.connectivity,
                    )
                    for image_id, prediction, gt_mask in zip(image_ids, predictions, gt_masks):
                        result = evaluate_instances(prediction, gt_mask, method="hungarian")
                        per_image[protocol][scale].append(compact_result(image_id, result))

    output: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for protocol in protocols:
        bare_summaries = {
            scale: aggregate_instance_metrics(
                [
                    {
                        "mAP": item["mAP"],
                        "count_error": item["count_error"],
                        "AP_per_iou": {0.5: item["AP50"], 0.75: item["AP75"]},
                    }
                    for item in per_image[protocol][scale]
                ]
            )
            for scale in scales
        }
        summaries = add_native_relative_metrics(bare_summaries)
        output[protocol] = {
            scale: {
                "resolved_min_area": resolve_min_area(args.min_area, scale, protocol),
                "summary": summaries[scale],
                "per_image": per_image[protocol][scale],
            }
            for scale in scales
        }
    return output


def plot_degradation_curve(
    model_results: Mapping[str, Mapping[str, Mapping[float, Mapping[str, Any]]]],
    scales: Sequence[float],
    save_path: str,
) -> None:
    """Gera paineis de mAP e AP50 para o protocolo de pipeline congelado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    colors = {"unet_watershed": "#1976D2", "deeplab_aspp": "#EF6C00"}
    for model_key, result in model_results.items():
        fixed = result["fixed"]
        maps = [fixed[scale]["summary"]["mean_mAP"] for scale in scales]
        ap50s = [fixed[scale]["summary"]["mean_AP50"] for scale in scales]
        label = MODEL_LABELS[model_key]
        axes[0].plot(scales, maps, marker="o", linewidth=2.2, color=colors[model_key], label=label)
        axes[1].plot(scales, ap50s, marker="o", linewidth=2.2, color=colors[model_key], label=label)
        for scale, value in zip(scales, maps):
            if scale == 1.0:
                continue
            degradation = fixed[scale]["summary"]["mAP_degradation_percent_vs_1x"]
            if degradation is not None:
                axes[0].annotate(
                    f"{degradation:+.1f}%",
                    (scale, value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=colors[model_key],
                )

    for axis, title, ylabel in zip(
        axes,
        ("mAP@[.50:.95]", "AP50"),
        ("mAP medio", "AP50 medio"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xticks(scales, [f"{scale:g}x" for scale in scales])
        axis.set_xlabel("Fator de escala na entrada")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Parte 6 - degradacao sob mudanca de escala\nWatershed com parametros congelados")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_challenge_image(
    model_results: Mapping[str, Mapping[str, Mapping[float, Mapping[str, Any]]]],
    scales: Sequence[float],
) -> str:
    """Seleciona automaticamente a imagem com maior queda conjunta fora de 1x."""
    extreme_scales = [scale for scale in scales if scale != 1.0]
    scores: Dict[str, float] = {}
    for result in model_results.values():
        fixed = result["fixed"]
        native = {item["image_id"]: item["mAP"] for item in fixed[1.0]["per_image"]}
        by_scale = {
            scale: {item["image_id"]: item["mAP"] for item in fixed[scale]["per_image"]}
            for scale in extreme_scales
        }
        for image_id, native_map in native.items():
            worst_extreme = min(by_scale[scale][image_id] for scale in extreme_scales)
            scores[image_id] = scores.get(image_id, 0.0) + max(0.0, native_map - worst_extreme)
    return max(scores, key=scores.get)


def predict_visual_sample(
    models: Mapping[str, torch.nn.Module],
    sample: Mapping[str, Any],
    scales: Sequence[float],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[float, np.ndarray]]:
    image = sample["image"].unsqueeze(0).to(device)
    native_size = tuple(image.shape[-2:])
    predictions: Dict[str, Dict[float, np.ndarray]] = {}
    with torch.no_grad():
        for model_key, model in models.items():
            predictions[model_key] = {}
            for scale in scales:
                probabilities = predict_scaled_probabilities(model, image, scale)
                predictions[model_key][scale] = decode_scaled_probabilities(
                    probabilities,
                    native_size=native_size,
                    scale=scale,
                    interior_threshold=args.interior_threshold,
                    foreground_threshold=args.foreground_threshold,
                    base_min_area=args.min_area,
                    area_protocol="fixed",
                    connectivity=args.connectivity,
                )[0]
    return predictions


def plot_visual_comparison(
    sample: Mapping[str, Any],
    predictions: Mapping[str, Mapping[float, np.ndarray]],
    scales: Sequence[float],
    save_path: str,
) -> None:
    """Gera a grade entrada/U-Net/ASPP/GT nas tres escalas."""
    image = sample["image"]
    gt = sample["mask_instance"].numpy()
    fig, axes = plt.subplots(4, len(scales), figsize=(4.2 * len(scales), 14))
    if len(scales) == 1:
        axes = np.asarray(axes).reshape(4, 1)

    for column, scale in enumerate(scales):
        scaled = resize_image_batch(image.unsqueeze(0), scale)[0]
        shown = scaled.permute(1, 2, 0).cpu().numpy()
        axes[0, column].imshow(np.clip(shown, 0, 1))
        axes[0, column].set_title(f"Entrada {scale:g}x ({shown.shape[0]}x{shown.shape[1]})")

        for row, model_key in ((1, "unet_watershed"), (2, "deeplab_aspp")):
            prediction = predictions[model_key][scale]
            metrics = evaluate_instances(prediction, gt, method="hungarian")
            axes[row, column].imshow(colorize_instances(prediction))
            axes[row, column].set_title(
                f"{MODEL_LABELS[model_key]}\nmAP={metrics['mAP']:.3f}; "
                f"erro cont.={metrics['count_error']}"
            )

        axes[3, column].imshow(colorize_instances(gt))
        axes[3, column].set_title("Ground truth 1x")
        for row in range(4):
            axes[row, column].axis("off")

    fig.suptitle(f"Caso de maior degradacao conjunta: {sample['image_id']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready_model_results(
    results: Mapping[str, Mapping[str, Mapping[float, Mapping[str, Any]]]],
) -> Dict[str, Any]:
    return {
        model_key: {
            protocol: {f"{scale:g}": payload for scale, payload in by_scale.items()}
            for protocol, by_scale in by_protocol.items()
        }
        for model_key, by_protocol in results.items()
    }


def main() -> None:
    args = parse_args()
    scales = parse_scale_factors(args.scales)
    if 1.0 not in scales:
        raise ValueError("A escala 1.0 e obrigatoria como referencia experimental.")
    if len(scales) < 2:
        raise ValueError("Informe ao menos uma escala de estresse alem de 1.0.")
    scaled_sizes = [round(args.target_size * scale) for scale in scales]
    if any(size % 32 != 0 for size in scaled_sizes):
        raise ValueError(
            "A U-Net requer que target_size * escala seja multiplo de 32; "
            f"dimensoes calculadas: {scaled_sizes}."
        )
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset DSB2018 nao encontrado em: {args.data_dir}")
    for checkpoint in (args.unet_checkpoint, args.aspp_checkpoint):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint}")

    device = torch.device(args.device)
    protocols = ["fixed"] if args.skip_area_control else ["fixed", "scaled"]
    splits = load_splits(args.splits_path)
    image_ids = splits.get(args.split, [])
    if not image_ids:
        raise ValueError(f"O split '{args.split}' esta vazio.")
    dataset = DSB2018Dataset(
        root_dir=args.data_dir,
        image_ids=image_ids,
        target_size=(args.target_size, args.target_size),
        three_class=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint_paths = {
        "unet_watershed": args.unet_checkpoint,
        "deeplab_aspp": args.aspp_checkpoint,
    }
    models: Dict[str, torch.nn.Module] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for model_key, checkpoint_path in checkpoint_paths.items():
        model, out_channels = load_model(checkpoint_path, device)
        if out_channels != 3:
            raise ValueError(f"{checkpoint_path} possui {out_channels} canal(is); esperado: 3.")
        models[model_key] = model
        metadata[model_key] = checkpoint_metadata(checkpoint_path)

    model_results: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
    for model_key, model in models.items():
        print(f"Avaliando {MODEL_LABELS[model_key]} em {scales}...")
        model_results[model_key] = evaluate_model(
            model, loader, scales, protocols, device, args
        )

    challenge_id = choose_challenge_image(model_results, scales)
    challenge_index = dataset.image_ids.index(challenge_id)
    challenge_sample = dataset[challenge_index]
    visual_predictions = predict_visual_sample(models, challenge_sample, scales, device, args)

    os.makedirs(args.output_dir, exist_ok=True)
    curve_path = os.path.join(args.output_dir, "scale_degradation_curve.png")
    panel_path = os.path.join(args.output_dir, "scale_visual_comparison.png")
    metrics_path = os.path.join(args.output_dir, "scale_metrics.json")
    plot_degradation_curve(model_results, scales, curve_path)
    plot_visual_comparison(challenge_sample, visual_predictions, scales, panel_path)

    payload = {
        "experiment": "Parte 6 - mudanca de escala",
        "reference_resolution": [args.target_size, args.target_size],
        "scales": list(scales),
        "split": args.split,
        "n_images": len(dataset),
        "matching": "hungarian",
        "postprocess": {
            "decoder": "watershed_3_classes",
            "interior_threshold": args.interior_threshold,
            "foreground_threshold": args.foreground_threshold,
            "base_min_area": args.min_area,
            "connectivity": args.connectivity,
            "protocols": protocols,
        },
        "checkpoints": metadata,
        "challenge_image_id": challenge_id,
        "models": json_ready_model_results(model_results),
    }
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"Metricas: {metrics_path}")
    print(f"Curva: {curve_path}")
    print(f"Painel: {panel_path}")


if __name__ == "__main__":
    main()
