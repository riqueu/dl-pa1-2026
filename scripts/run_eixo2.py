"""Script de automação para o Eixo 2 das Ablações (Funções de Perda e Fator gamma).

Executa a grade completa de ablações com 2 seeds (42 e 123):
- CE Ponderada (pesos automáticos)
- Focal Loss com gamma=0.0 (equivalente à CE)
- Focal Loss com gamma=1.0
- Focal Loss com gamma=2.0
- Focal Loss com gamma=5.0

Para cada configuração:
1. Treina o modelo U-Net 3-canais por 15 épocas no DSB2018.
2. Avalia no conjunto de validação com Hungarian matching.
3. Coleta as métricas, calcula média ± desvio padrão e gera gráficos/tabelas consolidadas.
"""

from typing import Any, Dict, List
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt

CONFIGS = [
    {
        "name": "ce_weighted",
        "loss": "weighted_ce_3c",
        "class_weights": "auto",
        "gamma": 0.0,
        "label": "CE Ponderada",
    },
    {
        "name": "focal_gamma0",
        "loss": "multiclass_focal",
        "class_weights": "auto",
        "gamma": 0.0,
        "label": "Focal (γ=0)",
    },
    {
        "name": "focal_gamma1",
        "loss": "multiclass_focal",
        "class_weights": "auto",
        "gamma": 1.0,
        "label": "Focal (γ=1)",
    },
    {
        "name": "focal_gamma2",
        "loss": "multiclass_focal",
        "class_weights": "auto",
        "gamma": 2.0,
        "label": "Focal (γ=2)",
    },
    {
        "name": "focal_gamma5",
        "loss": "multiclass_focal",
        "class_weights": "auto",
        "gamma": 5.0,
        "label": "Focal (γ=5)",
    },
]

DEFAULT_SEEDS = [42, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automação do Eixo 2 da Parte 3 (Ablações de Perda)")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Interpretador Python")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas por treino")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Lista de seeds")
    parser.add_argument("--config", type=str, default="all", help="Nome da config específica ou 'all'")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Pula treinos já concluídos")
    parser.add_argument("--force", action="store_true", help="Força re-execução de tudo")
    return parser.parse_args()


def run_command(cmd: List[str], desc: str) -> None:
    print(f"\n[EXEC] {desc}")
    print(" ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERRO] Comando falhou com código {res.returncode}")
        sys.exit(res.returncode)


def main() -> None:
    args = parse_args()
    py_bin = args.python_bin

    selected_configs = (
        CONFIGS if args.config == "all" else [c for c in CONFIGS if c["name"] == args.config]
    )
    if not selected_configs:
        print(f"Configuração '{args.config}' não encontrada. Opções: {[c['name'] for c in CONFIGS]}")
        sys.exit(1)

    os.makedirs("runs/part3_eixo2", exist_ok=True)
    os.makedirs("checkpoints/part3_eixo2", exist_ok=True)
    os.makedirs("outputs/part3_eixo2", exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}

    for cfg in selected_configs:
        cfg_name = cfg["name"]
        results[cfg_name] = {
            "label": cfg["label"],
            "loss": cfg["loss"],
            "gamma": cfg["gamma"],
            "seeds": {},
        }

        for seed in args.seeds:
            run_id = f"{cfg_name}_seed{seed}"
            run_dir = os.path.join("runs/part3_eixo2", run_id)
            ckpt_path = os.path.join("checkpoints/part3_eixo2", f"{run_id}.pth")
            eval_dir = os.path.join("outputs/part3_eixo2", run_id)
            eval_json = os.path.join(eval_dir, "metrics_dsb2018_val_hungarian.json")

            # 1. Treinamento
            needs_train = True
            if os.path.exists(ckpt_path) and not args.force:
                print(f"[SKIP] Checkpoint já existente em {ckpt_path}")
                needs_train = False

            if needs_train:
                train_cmd = [
                    py_bin, "train.py",
                    "--dataset", "dsb2018",
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--out_channels", "3",
                    "--loss", cfg["loss"],
                    "--class_weights", cfg["class_weights"],
                    "--gamma", str(cfg["gamma"]),
                    "--seed", str(seed),
                    "--out", run_dir,
                    "--checkpoint", ckpt_path,
                ]
                run_command(train_cmd, f"Treinando {cfg['label']} (Seed {seed})")

            # 2. Avaliação
            needs_eval = True
            if os.path.exists(eval_json) and not args.force and not needs_train:
                print(f"[SKIP] Avaliação já existente em {eval_json}")
                needs_eval = False

            if needs_eval:
                eval_cmd = [
                    py_bin, "evaluate.py",
                    "--checkpoint", ckpt_path,
                    "--dataset", "dsb2018",
                    "--split", "val",
                    "--matching", "hungarian",
                    "--output-dir", eval_dir,
                    "--no-plots",
                ]
                run_command(eval_cmd, f"Avaliando {cfg['label']} (Seed {seed})")

            # 3. Ler Métricas
            with open(eval_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            results[cfg_name]["seeds"][str(seed)] = {
                "mAP": metrics["mean_mAP"],
                "AP50": metrics["AP_per_threshold"].get("0.50", 0.0),
                "AP75": metrics["AP_per_threshold"].get("0.75", 0.0),
                "count_error": metrics["mean_count_error"],
                "iou": metrics["mean_semantic_iou"],
                "dice": metrics["mean_semantic_dice"],
            }

    # 4. Consolidar Estatísticas (Média ± Desvio)
    summary_table: List[Dict[str, Any]] = []

    print("\n" + "=" * 95)
    print("CONSOLIDAÇÃO FINAL — PARTE 3: EIXO 2 (FUNÇÕES DE PERDA & FATOR γ)")
    print("=" * 95)
    header = f"{'Configuração':<18} | {'Perda':<16} | {'γ':<4} | {'mAP@[.50:.95]':<20} | {'AP@0.50':<12} | {'Erro Contagem':<16}"
    print(header)
    print("-" * 95)

    for cfg_name, data in results.items():
        seed_data = data["seeds"]
        maps = [v["mAP"] for v in seed_data.values()]
        ap50s = [v["AP50"] for v in seed_data.values()]
        ap75s = [v["AP75"] for v in seed_data.values()]
        count_errs = [v["count_error"] for v in seed_data.values()]
        ious = [v["iou"] for v in seed_data.values()]
        dices = [v["dice"] for v in seed_data.values()]

        mean_map, std_map = float(np.mean(maps)), float(np.std(maps))
        mean_ap50, std_ap50 = float(np.mean(ap50s)), float(np.std(ap50s))
        mean_cnt, std_cnt = float(np.mean(count_errs)), float(np.std(count_errs))
        mean_iou, std_iou = float(np.mean(ious)), float(np.std(ious))
        mean_dice, std_dice = float(np.mean(dices)), float(np.std(dices))

        data["summary"] = {
            "mAP_mean": mean_map,
            "mAP_std": std_map,
            "AP50_mean": mean_ap50,
            "AP50_std": std_ap50,
            "AP75_mean": float(np.mean(ap75s)),
            "AP75_std": float(np.std(ap75s)),
            "count_error_mean": mean_cnt,
            "count_error_std": std_cnt,
            "iou_mean": mean_iou,
            "iou_std": std_iou,
            "dice_mean": mean_dice,
            "dice_std": std_dice,
        }

        row = f"{data['label']:<18} | {data['loss']:<16} | {data['gamma']:<4.1f} | {mean_map:.4f} ± {std_map:.4f}     | {mean_ap50:.4f}       | {mean_cnt:.2f} ± {std_cnt:.2f} núcleos"
        print(row)

        summary_table.append({
            "Configuração": data["label"],
            "Perda": data["loss"],
            "gamma": data["gamma"],
            "mAP": f"{mean_map:.4f} ± {std_map:.4f}",
            "AP50": f"{mean_ap50:.4f} ± {std_ap50:.4f}",
            "Erro Contagem": f"{mean_cnt:.2f} ± {std_cnt:.2f}",
            "IoU": f"{mean_iou:.4f} ± {std_iou:.4f}",
            "Dice": f"{mean_dice:.4f} ± {std_dice:.4f}",
        })

    print("=" * 95)

    # 5. Salvar JSON Consolidado
    summary_path = "outputs/part3_eixo2/summary_eixo2.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResumo JSON salvo em: {summary_path}")

    # 6. Gerar Gráfico de Diagnóstico de γ
    focal_configs = [c for c in CONFIGS if "focal" in c["name"]]
    if len(focal_configs) >= 3 and all(c["name"] in results for c in focal_configs):
        gammas = [results[c["name"]]["gamma"] for c in focal_configs]
        maps = [results[c["name"]]["summary"]["mAP_mean"] for c in focal_configs]
        map_errs = [results[c["name"]]["summary"]["mAP_std"] for c in focal_configs]
        cnts = [results[c["name"]]["summary"]["count_error_mean"] for c in focal_configs]
        cnt_errs = [results[c["name"]]["summary"]["count_error_std"] for c in focal_configs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.errorbar(gammas, maps, yerr=map_errs, fmt="o-", color="#1f77b4", capsize=5, linewidth=2, markersize=7)
        ax1.set_title("mAP@[.50:.95] vs. Fator de Foco γ", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Parâmetro γ da Focal Loss", fontsize=11)
        ax1.set_ylabel("mAP Médio", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.errorbar(gammas, cnts, yerr=cnt_errs, fmt="s-", color="#d62728", capsize=5, linewidth=2, markersize=7)
        ax2.set_title("Erro Médio de Contagem vs. Fator de Foco γ", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Parâmetro γ da Focal Loss", fontsize=11)
        ax2.set_ylabel("Erro Médio (núcleos/imagem)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plot_path = "outputs/part3_eixo2/focal_gamma_ablation.png"
        plt.suptitle("Parte 3 — Ablação do Eixo 2: Efeito do Foco em Fronteiras Desbalanceadas", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Gráfico de diagnóstico salvo em: {plot_path}")

    print("\nExecução do Eixo 2 concluída com sucesso!")


if __name__ == "__main__":
    main()
