"""Script de execução e diagnóstico da Parte 4 (Membro A: Henrique).

Demonstra:
1. Montagem do mosaico grande (512x512) com GT contínuo a partir de imagens do DSB2018.
2. Execução do Tiling Ingênuo com janelas de 256x256 e stride de 128 (50% sobreposição).
3. Avaliação quantitativa com mAP@[.50:.95], AP@0.50, AP@0.75 e erro de contagem.
4. Identificação geométrica dos núcleos que cruzam as bordas dos tiles e foram fatiados.
5. Geração de gráficos diagnósticos com zoom nos núcleos fragmentados.
6. Persistência dos dados intermediários para consumo do algoritmo de costura do Membro B.
"""

from typing import Dict, Any, List
import argparse
import json
import os
import sys
import numpy as np
import torch

from src.models import build_model
from src.metrics import evaluate_instances
from src.mosaic import (
    create_mosaic_sample,
    create_mosaic_from_large_image,
    predict_tiled_naive,
    detect_split_nuclei,
    plot_tiling_failure_analysis,
)


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Carrega o modelo U-Net treinado com pesos da Parte 2 (Watershed)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    model_args = state.get("model_args", {}) if isinstance(state, dict) else {}
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    out_channels = model_args.get("out_channels", 3)
    encoder = model_args.get("encoder", "resnet34")
    up_mode = model_args.get("up_mode", "transpose")
    use_skips = model_args.get("use_skips", True)

    model = build_model(
        encoder=encoder,
        out_channels=out_channels,
        up_mode=up_mode,
        use_skips=use_skips,
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    model.out_channels = out_channels
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração do Tiling Ingênuo (Parte 4)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/part2_watershed.pth", help="Caminho do checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/part4_mosaic", help="Diretório de saída")
    parser.add_argument("--tile_size", type=int, default=256, help="Tamanho do patch quadrado")
    parser.add_argument("--stride", type=int, default=128, help="Passo da janela deslizante (128 = 50%% overlap)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[Parte 4] Inicializando pipeline no dispositivo: {device}")

    # 1. Carrega modelo oficial
    model = load_model(args.checkpoint, device=device)
    print(f"[Parte 4] Modelo carregado: {args.checkpoint} (canais: {model.out_channels})")

    # 2. Carrega lista de validação
    with open("data/splits.json", "r") as f:
        splits = json.load(f)
    val_ids = splits.get("val", [])

    # Seleciona 4 imagens representativas e densas para o Mosaico 2x2
    # Estas 4 imagens têm alta densidade celular e cobrem os 4 quadrantes
    selected_4_ids = [
        "ed8c31b001a0c23c33402f94a5ee6b0209e0c6419eb52d5d02255513e3a672fc",
        "64eeef16fdc4e26523d27bfa71a1d38d2cb2e4fa116c0d0ea56b1322f806f0b9",
        "e50ac10d1dce6496d092d966784ed3795969128ca0bc58199a36d558ed529203",
        "243443ae303cc09cfbea85bfd22b0c4f026342f3dfc3aa1076f27867910d025b",
    ]
    # Verifica se os IDs existem no dataset local
    for img_id in selected_4_ids:
        if not os.path.exists(os.path.join("data/raw/stage1_train", img_id)):
            print(f"[Aviso] ID {img_id} não encontrado, usando primeiros IDs do val set.")
            selected_4_ids = val_ids[:4]
            break

    print(f"[Parte 4] Montando mosaico 2x2 (512x512) a partir de: {selected_4_ids}")
    mosaic_grid_img, mosaic_grid_gt = create_mosaic_sample(
        selected_4_ids,
        data_dir="data/raw/stage1_train",
        grid=(2, 2),
        tile_size=(args.tile_size, args.tile_size),
    )
    n_gt_grid = len(np.unique(mosaic_grid_gt)) - (1 if 0 in mosaic_grid_gt else 0)
    print(f"[Parte 4] Mosaico em grade montado com sucesso! Shape: {mosaic_grid_img.shape}, GT total: {n_gt_grid} núcleos.")

    # 3. Execução do Tiling Ingênuo - Modo Center-Crop (Slide 83 da aula)
    print("\n--- Executando Tiling Ingênuo: Modo 'center_crop' (Prática de Semântica - Slide 83) ---")
    naive_res_crop = predict_tiled_naive(
        mosaic_grid_img,
        model=model,
        tile_size=args.tile_size,
        stride=args.stride,
        device=device,
        mode="center_crop",
    )
    eval_crop = evaluate_instances(naive_res_crop["naive_instance_mask"], mosaic_grid_gt)
    split_info_crop = detect_split_nuclei(
        mosaic_grid_gt,
        naive_res_crop["tile_boxes"],
        naive_pred=naive_res_crop["naive_instance_mask"],
    )

    print(f"Predições ingênuas (center_crop): {naive_res_crop['num_instances']} instâncias")
    print(f"mAP@[.50:.95]: {eval_crop['mAP']:.4f} | AP@0.50: {eval_crop['AP_per_iou'][0.50]:.4f} | AP@0.75: {eval_crop['AP_per_iou'][0.75]:.4f}")
    print(f"Erro absoluto de contagem: {eval_crop['count_error']} (|{naive_res_crop['num_instances']} - {n_gt_grid}|)")
    print(f"Núcleos de GT que cruzam linhas de corte dos tiles: {split_info_crop['num_boundary_nuclei']} ({split_info_crop['split_ratio']*100:.1f}%)")
    split_count = sum(1 for d in split_info_crop["split_gt_details"] if d.get("num_fragments", 0) >= 2)
    print(f"Núcleos efetivamente fatiados em múltiplos IDs: {split_count}")

    # 4. Execução do Tiling Ingênuo - Modo Direct-Stamp
    print("\n--- Executando Tiling Ingênuo: Modo 'direct_stamp' (Estampa Direta com IDs Independentes) ---")
    naive_res_stamp = predict_tiled_naive(
        mosaic_grid_img,
        model=model,
        tile_size=args.tile_size,
        stride=args.stride,
        device=device,
        mode="direct_stamp",
    )
    eval_stamp = evaluate_instances(naive_res_stamp["naive_instance_mask"], mosaic_grid_gt)
    print(f"Predições ingênuas (direct_stamp): {naive_res_stamp['num_instances']} instâncias")
    print(f"mAP@[.50:.95]: {eval_stamp['mAP']:.4f} | AP@0.50: {eval_stamp['AP_per_iou'][0.50]:.4f} | AP@0.75: {eval_stamp['AP_per_iou'][0.75]:.4f}")
    print(f"Erro absoluto de contagem: {eval_stamp['count_error']} (|{naive_res_stamp['num_instances']} - {n_gt_grid}|)")

    # 5. Gera visualização principal de falha do mosaico em grade
    plot_path_grid = os.path.join(args.output_dir, "tiling_naive_failure.png")
    plot_tiling_failure_analysis(
        mosaic_image=mosaic_grid_img,
        gt_mask=mosaic_grid_gt,
        naive_pred=naive_res_crop["naive_instance_mask"],
        tile_boxes=naive_res_crop["tile_boxes"],
        save_path=plot_path_grid,
    )

    # 6. Teste com lâmina contínua real de alta resolução (696x520)
    # Seleciona a lâmina '0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1' do conjunto de validação
    large_val_id = "0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1"
    continuous_results: Dict[str, Any] = {}
    if os.path.exists(os.path.join("data/raw/stage1_train", large_val_id)):
        print(f"\n--- Executando Tiling Ingênuo em Lâmina Biológica Contínua ({large_val_id[:8]}...) ---")
        cont_img, cont_gt = create_mosaic_from_large_image(
            large_val_id,
            data_dir="data/raw/stage1_train",
            crop_size=(512, 512),
            origin=(0, 0),
        )
        n_gt_cont = len(np.unique(cont_gt)) - (1 if 0 in cont_gt else 0)

        naive_cont = predict_tiled_naive(
            cont_img,
            model=model,
            tile_size=args.tile_size,
            stride=args.stride,
            device=device,
            mode="center_crop",
        )
        eval_cont = evaluate_instances(naive_cont["naive_instance_mask"], cont_gt)
        split_cont = detect_split_nuclei(cont_gt, naive_cont["tile_boxes"], naive_pred=naive_cont["naive_instance_mask"])
        split_count_cont = sum(1 for d in split_cont["split_gt_details"] if d.get("num_fragments", 0) >= 2)

        print(f"GT contínuo: {n_gt_cont} núcleos | Predição ingênua: {naive_cont['num_instances']} instâncias")
        print(f"mAP@[.50:.95]: {eval_cont['mAP']:.4f} | AP@0.50: {eval_cont['AP_per_iou'][0.50]:.4f} | AP@0.75: {eval_cont['AP_per_iou'][0.75]:.4f}")
        print(f"Erro de contagem: {eval_cont['count_error']}")
        print(f"Núcleos na fronteira: {split_cont['num_boundary_nuclei']} | Fatiados: {split_count_cont}")

        plot_path_cont = os.path.join(args.output_dir, "tiling_naive_failure_continuous.png")
        plot_tiling_failure_analysis(
            mosaic_image=cont_img,
            gt_mask=cont_gt,
            naive_pred=naive_cont["naive_instance_mask"],
            tile_boxes=naive_cont["tile_boxes"],
            save_path=plot_path_cont,
        )

        continuous_results = {
            "image_id": large_val_id,
            "n_gt": n_gt_cont,
            "n_pred_naive": naive_cont["num_instances"],
            "mAP": eval_cont["mAP"],
            "ap50": eval_cont["AP_per_iou"][0.50],
            "ap75": eval_cont["AP_per_iou"][0.75],
            "count_error": eval_cont["count_error"],
            "boundary_nuclei": split_cont["num_boundary_nuclei"],
            "split_nuclei_count": split_count_cont,
        }

    # 7. Salva resumo em JSON
    metrics_summary = {
        "mosaic_grid_sample": {
            "image_ids": selected_4_ids,
            "total_gt_nuclei": n_gt_grid,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "num_tiles": len(naive_res_crop["tile_boxes"]),
            "center_crop_mode": {
                "n_pred": naive_res_crop["num_instances"],
                "mAP": eval_crop["mAP"],
                "ap50": eval_crop["AP_per_iou"][0.50],
                "ap75": eval_crop["AP_per_iou"][0.75],
                "count_error": eval_crop["count_error"],
                "boundary_nuclei": split_info_crop["num_boundary_nuclei"],
                "split_ratio": split_info_crop["split_ratio"],
                "split_nuclei_count": split_count,
            },
            "direct_stamp_mode": {
                "n_pred": naive_res_stamp["num_instances"],
                "mAP": eval_stamp["mAP"],
                "ap50": eval_stamp["AP_per_iou"][0.50],
                "ap75": eval_stamp["AP_per_iou"][0.75],
                "count_error": eval_stamp["count_error"],
            },
        },
        "continuous_large_image": continuous_results,
    }

    json_path = os.path.join(args.output_dir, "naive_tiling_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[Parte 4] Métricas salvas em: {json_path}")
    print("[Parte 4] Execução da demonstração concluída com sucesso!")


if __name__ == "__main__":
    main()
