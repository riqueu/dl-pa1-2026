#!/usr/bin/env bash
set -euo pipefail

# Grade reproduzível do Eixo 1. Runs e avaliações novos vão para scratch/ por
# padrão, preservando os resultados versionados em runs/ e outputs/. Os pesos
# ficam no caminho canônico consumido pela Parte 6.
#
# PA1_CLASS_WEIGHTS="1.0,2.877,5.731" bash scripts/run_eixo1.sh

if [[ -z "${PA1_CLASS_WEIGHTS:-}" ]]; then
    echo "Defina PA1_CLASS_WEIGHTS com os pesos fixos calculados no split de treino."
    echo "Pesos usados no experimento: 1.0,2.877,5.731"
    echo "Exemplo: PA1_CLASS_WEIGHTS=\"1.0,2.877,5.731\" bash scripts/run_eixo1.sh"
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${PA1_RUN_ROOT:-scratch/part3_eixo1/runs}"
CHECKPOINT_ROOT="${PA1_CHECKPOINT_ROOT:-checkpoints/part3_eixo1}"
OUTPUT_ROOT="${PA1_OUTPUT_ROOT:-scratch/part3_eixo1/outputs}"

mkdir -p "$RUN_ROOT" "$CHECKPOINT_ROOT" "$OUTPUT_ROOT"

configs=(unet_skips unet_noskips deeplab_aspp)
seeds=(42 123)

for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        run_dir="${RUN_ROOT}/${config}_seed${seed}"
        checkpoint="${CHECKPOINT_ROOT}/${config}_seed${seed}.pth"
        output_dir="${OUTPUT_ROOT}/${config}_seed${seed}"
        metrics_json="${output_dir}/metrics_dsb2018_val_hungarian.json"

        if [[ -f "$checkpoint" ]]; then
            echo "Checkpoint existente; treino pulado: $checkpoint"
        else
            if [[ -e "$run_dir" ]]; then
                echo "Run parcial já existe sem checkpoint: $run_dir"
                echo "Remova-o manualmente ou escolha outro PA1_RUN_ROOT; nada foi sobrescrito."
                exit 3
            fi

            architecture_args=(--decoder_type unet)
            if [[ "$config" == "unet_noskips" ]]; then
                architecture_args+=(--no_skips)
            elif [[ "$config" == "deeplab_aspp" ]]; then
                architecture_args=(--decoder_type aspp --output_stride 16 --aspp_rates 6 12 18)
            fi

            "$PYTHON_BIN" train.py \
                --dataset dsb2018 \
                --epochs 15 \
                --batch_size 8 \
                --lr 1e-3 \
                --seed "$seed" \
                --encoder resnet34 \
                --out_channels 3 \
                --loss weighted_ce_3c \
                --class_weights "$PA1_CLASS_WEIGHTS" \
                --select_by mAP \
                --out "$run_dir" \
                --checkpoint "$checkpoint" \
                "${architecture_args[@]}"
        fi

        if [[ -f "$metrics_json" ]]; then
            echo "Avaliação existente; etapa pulada: $metrics_json"
            continue
        fi
        if [[ -e "$output_dir" ]]; then
            echo "Diretório de avaliação parcial já existe: $output_dir"
            echo "Remova-o manualmente ou escolha outro PA1_OUTPUT_ROOT; nada foi sobrescrito."
            exit 4
        fi

        "$PYTHON_BIN" evaluate.py \
            --checkpoint "$checkpoint" \
            --dataset dsb2018 \
            --split val \
            --matching hungarian \
            --batch-size 8 \
            --output-dir "$output_dir"
    done
done

"$PYTHON_BIN" scripts/summarize_eixo1.py --output-root "$OUTPUT_ROOT"
