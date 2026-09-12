#!/usr/bin/env bash
set -euo pipefail

# Grade reproduzível do Eixo 1. Exemplo:
# PA1_CLASS_WEIGHTS="0.5,1.5,8.0" bash scripts/run_eixo1.sh

if [[ -z "${PA1_CLASS_WEIGHTS:-}" ]]; then
    echo "Defina PA1_CLASS_WEIGHTS com os pesos fixos calculados no split de treino."
    echo "Exemplo: PA1_CLASS_WEIGHTS=\"0.5,1.5,8.0\" bash scripts/run_eixo1.sh"
    exit 2
fi

configs=(unet_skips unet_noskips deeplab_aspp)
seeds=(42 123)

for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        run_dir="runs/part3_eixo1/${config}_seed${seed}"
        checkpoint="checkpoints/part3_eixo1/${config}_seed${seed}.pth"
        output_dir="outputs/part3_eixo1/${config}_seed${seed}"

        if [[ -e "$checkpoint" || -e "$run_dir" || -e "$output_dir" ]]; then
            echo "Saída já existente para ${config}, seed ${seed}; abortando para não sobrescrever."
            exit 3
        fi

        architecture_args=(--decoder_type unet)
        if [[ "$config" == "unet_noskips" ]]; then
            architecture_args+=(--no_skips)
        elif [[ "$config" == "deeplab_aspp" ]]; then
            architecture_args=(--decoder_type aspp --output_stride 16 --aspp_rates 6 12 18)
        fi

        python train.py \
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

        python evaluate.py \
            --checkpoint "$checkpoint" \
            --dataset dsb2018 \
            --split val \
            --matching hungarian \
            --batch-size 8 \
            --output-dir "$output_dir"
    done
done

python scripts/summarize_eixo1.py
