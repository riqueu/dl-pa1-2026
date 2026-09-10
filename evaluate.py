"""Pipeline de avaliação em lote no conjunto de teste/validação (CLI).

Este script atende ao requisito do README do PA1 ('um comando que avalia'):
- Avalia predições semânticas (IoU e Dice) e predições de instâncias (mAP@[.50:.95] e contagem).
- Suporta matching Hungarian ou Greedy com limiarização configurável.
- Gera automaticamente o gráfico de quantificação de fracasso (mAP vs. densidade de núcleos)
  exigido no Item 5 da Parte 1.
"""

from typing import Any, Dict, List, Tuple
import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DSB2018Dataset, SyntheticDataset, load_splits
from src.metrics import (
    compute_semantic_iou,
    compute_semantic_dice,
    evaluate_instances,
)
from src.postprocess import naive_connected_components, watershed_instance_segmentation
from src.utils import (
    plot_failure_vs_density,
    plot_sample_comparison,
)


def parse_args() -> argparse.Namespace:
    """Configura e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="PA1 - Avaliação em lote de segmentação semântica e de instâncias",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Caminho para os pesos salvos do modelo (.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dsb2018", "synthetic"],
        default="dsb2018",
        help="Base de dados a avaliar",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "train"],
        default="val",
        help="Partição dos dados a avaliar (para dsb2018)",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default="data/splits.json",
        help="Arquivo JSON contendo as partições estratificadas",
    )
    parser.add_argument(
        "--matching",
        type=str,
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Estratégia de matching para mAP de instâncias",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "semantic", "watershed"],
        default="auto",
        help="Representação da saída; auto infere pelo número de canais do checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Limiar de probabilidade binária sobre a saída sigmoide da rede",
    )
    parser.add_argument(
        "--interior-threshold",
        type=float,
        default=0.50,
        help="Limiar dos marcadores de interior no modo watershed",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.50,
        help="Limiar de P(interior) + P(fronteira) no modo watershed",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 2],
        default=1,
        help="Conectividade usada pelo pós-processamento",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Área mínima das instâncias previstas; zero desliga o filtro",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamanho do lote na avaliação",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Número de imagens quando dataset=synthetic",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Processos usados pelo DataLoader; use zero em ambientes restritos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_results",
        help="Diretório onde tabelas e gráficos serão exportados",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gera os gráficos de densidade e painéis comparativos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo de execução (cuda ou cpu)",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    """Reconstrói o modelo do checkpoint e infere sua representação de saída."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint não encontrado em '{checkpoint_path}'. "
            "Treine o modelo antes de executar a avaliação."
        )

    from src.models import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_args = dict(checkpoint.get("model_args", {}))
    else:
        # Compatibilidade com checkpoints antigos contendo somente o state_dict.
        state_dict = checkpoint
        model_args = {}

    out_channels = int(model_args.get("out_channels", 1))
    if out_channels not in (1, 3):
        raise RuntimeError(
            f"Checkpoint com {out_channels} canais não corresponde à baseline nem ao watershed."
        )

    # Os pesos serão substituídos imediatamente; evita download ImageNet na avaliação.
    model_args["pretrained"] = False
    model = build_model(**model_args)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    task = "watershed" if out_channels == 3 else "semantic"
    return model, task


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PA1 - PIPELINE DE AVALIAÇÃO EM LOTE")
    print(f"Dataset: {args.dataset} | Split: {args.split} | Matching: {args.matching}")
    print(f"Dispositivo: {device} | Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # 1. Carregar Dataset
    if args.dataset == "dsb2018":
        if not os.path.exists(args.splits_path):
            from src.dataset import create_stratified_splits
            print(f"Splits não encontrados. Criando partições estratificadas em '{args.splits_path}'...")
            splits = create_stratified_splits(splits_path=args.splits_path)
        else:
            splits = load_splits(args.splits_path)

        image_ids = splits.get(args.split, [])
        if len(image_ids) == 0:
            print(f"Aviso: Nenhuma imagem encontrada no split '{args.split}'.")
            sys.exit(1)

        dataset = DSB2018Dataset(image_ids=image_ids, target_size=(256, 256))
    else:
        dataset = SyntheticDataset(num_samples=args.num_samples, seed=123)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Total de imagens a avaliar: {len(dataset)}")

    # 2. Carregar Modelo
    model, checkpoint_task = load_model(args.checkpoint, device)

    if args.task != "auto" and args.task != checkpoint_task:
        raise ValueError(
            f"--task={args.task} não combina com o checkpoint de tarefa {checkpoint_task}."
        )
    task = checkpoint_task if args.task == "auto" else args.task
    print(f"Representação inferida do checkpoint: {task}")

    # 3. Loop de Inferência e Coleta de Métricas
    semantic_ious: List[float] = []
    semantic_dices: List[float] = []
    instance_results: List[Dict[str, Any]] = []
    gt_counts: List[int] = []
    pred_counts: List[int] = []
    image_ids_list: List[str] = []

    # Para salvar exemplos visuais (melhor, mediano, pior)
    sample_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Avaliando imagens"):
            images = batch["image"].to(device)
            gt_semantics = batch["mask_semantic"]
            gt_instances = batch["mask_instance"].cpu().numpy()
            batch_ids = batch.get("image_id", [f"img_{i}" for i in range(len(images))])

            logits = model(images)
            if task == "watershed":
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().numpy()

            for b in range(len(images)):
                gt_sem = gt_semantics[b, 0].numpy()
                gt_inst = gt_instances[b]
                i_id = batch_ids[b]

                if task == "watershed":
                    pred_inst = watershed_instance_segmentation(
                        probs[b],
                        interior_threshold=args.interior_threshold,
                        foreground_threshold=args.foreground_threshold,
                        min_area=args.min_size,
                        connectivity=args.connectivity,
                    )
                    pred_binary = (pred_inst > 0).astype(np.float32)
                else:
                    prob_map = probs[b, 0]
                    pred_inst = naive_connected_components(
                        prob_map,
                        threshold=args.threshold,
                        from_logits=False,
                        connectivity=args.connectivity,
                        min_size=args.min_size,
                    )
                    pred_binary = (prob_map >= args.threshold).astype(np.float32)

                # Métricas semânticas sobre a união das instâncias previstas.
                s_iou = compute_semantic_iou(pred_binary, gt_sem)
                s_dice = compute_semantic_dice(pred_binary, gt_sem)
                semantic_ious.append(s_iou)
                semantic_dices.append(s_dice)

                # Métrica de Instâncias (Hungarian ou Greedy)
                inst_res = evaluate_instances(pred_inst, gt_inst, method=args.matching)
                instance_results.append(inst_res)
                gt_counts.append(inst_res["n_gt"])
                pred_counts.append(inst_res["n_pred"])
                image_ids_list.append(i_id)

                sample_records.append({
                    "image": images[b].cpu(),
                    "gt": gt_inst,
                    "pred": pred_inst,
                    "mAP": inst_res["mAP"],
                    "count_error": inst_res["count_error"],
                    "iou": s_iou,
                    "dice": s_dice,
                    "id": i_id,
                })

    # 4. Agregação e Relatório das Métricas
    mean_sem_iou = float(np.mean(semantic_ious))
    mean_sem_dice = float(np.mean(semantic_dices))
    mean_map = float(np.mean([r["mAP"] for r in instance_results]))
    mean_count_err = float(np.mean([r["count_error"] for r in instance_results]))

    print("\n" + "=" * 70)
    print("RESULTADOS DA AVALIAÇÃO")
    print("=" * 70)
    print(f"Métricas Semânticas (Item 1):")
    print(f"  - Mean IoU (Jaccard): {mean_sem_iou:.4f}")
    print(f"  - Mean Dice (F1)   : {mean_sem_dice:.4f}")
    print(f"\nMétricas de Instância ({args.matching.capitalize()} - Item 3):")
    print(f"  - mAP@[.50:.95]    : {mean_map:.4f}")
    print(f"  - Erro Médio Cont. : {mean_count_err:.2f} núcleos/imagem")

    # Tabela de AP por limiar de IoU
    thresholds = sorted(instance_results[0]["AP_per_iou"].keys())
    print("\nDetalhamento do AP por limiar de IoU:")
    print("-" * 45)
    print(f"{'Limiar de IoU':<15} | {'AP':<15}")
    print("-" * 45)
    for t in thresholds:
        t_ap = np.mean([r["AP_per_iou"][t] for r in instance_results])
        print(f"IoU >= {t:<7.2f} | {t_ap:<15.4f}")
    print("-" * 45)

    # 5. Exportação de Arquivo de Métricas JSON
    metrics_summary = {
        "dataset": args.dataset,
        "split": args.split,
        "task": task,
        "matching": args.matching,
        "mean_semantic_iou": mean_sem_iou,
        "mean_semantic_dice": mean_sem_dice,
        "mean_mAP": mean_map,
        "mean_count_error": mean_count_err,
        "AP_per_threshold": {f"{t:.2f}": float(np.mean([r["AP_per_iou"][t] for r in instance_results])) for t in thresholds},
    }
    json_path = os.path.join(
        args.output_dir,
        f"metrics_{args.dataset}_{args.split}_{task}_{args.matching}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nMétricas salvas em: {json_path}")

    # 6. Gráfico de Fracasso vs Densidade (Item 5 da Parte 1)
    if args.save_plots:
        print("\nGerando gráficos de diagnóstico e painéis de falhas...")
        failure_plot_path = os.path.join(
            args.output_dir,
            f"failure_vs_density_{task}_{args.matching}.png",
        )
        plot_failure_vs_density(
            nuclei_counts=gt_counts,
            map_scores=[r["mAP"] for r in instance_results],
            count_errors=[r["count_error"] for r in instance_results],
            save_path=failure_plot_path,
            title_suffix=f"({args.matching.capitalize()} Matching)",
        )
        print(f"Gráfico de fracasso salvo em: {failure_plot_path}")

        # Salvar as 3 amostras mais ilustrativas: Melhor, Mediana e Pior
        sample_records.sort(key=lambda x: x["mAP"])
        worst = sample_records[0]
        median = sample_records[len(sample_records) // 2]
        best = sample_records[-1]

        plot_sample_comparison(
            worst["image"], worst["gt"], worst["pred"],
            metrics={"mAP": worst["mAP"], "count_error": worst["count_error"], "iou": worst["iou"]},
            title_prefix=f"Pior Caso ({worst['id']})",
            save_path=os.path.join(args.output_dir, f"sample_{task}_worst.png"),
        )
        plot_sample_comparison(
            median["image"], median["gt"], median["pred"],
            metrics={"mAP": median["mAP"], "count_error": median["count_error"], "iou": median["iou"]},
            title_prefix=f"Caso Mediano ({median['id']})",
            save_path=os.path.join(args.output_dir, f"sample_{task}_median.png"),
        )
        plot_sample_comparison(
            best["image"], best["gt"], best["pred"],
            metrics={"mAP": best["mAP"], "count_error": best["count_error"], "iou": best["iou"]},
            title_prefix=f"Melhor Caso ({best['id']})",
            save_path=os.path.join(args.output_dir, f"sample_{task}_best.png"),
        )
        print(f"Painéis visuais (pior, mediano, melhor) salvos em: {args.output_dir}")

    print("\nAvaliação concluída com sucesso!")


if __name__ == "__main__":
    main()
