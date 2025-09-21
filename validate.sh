#!/bin/bash

export WANDB_API_KEY='5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe' # seonghu, for debugging

# Temporary visualize config for attention map visualization

# Allow passing multiple checkpoints as comma-separated list via CKPTS env var
CKPTS="${CKPTS:-2000,4000,8000,10000,12000,16000}"

for ckpt in $(echo "$CKPTS" | tr ',' ' '); do
    CKPT_NAME="checkpoint-${ckpt}"
    RUN_NAME="EXP_val_run_${ckpt}_$(date +%Y%m%d_%H%M%S)"
    export RUN_NAME
    CUDA_VISIBLE_DEVICES=2 accelerate launch --mixed_precision="fp16" \
                        --num_processes=1 \
                        --num_machines 1 \
                        --main_process_port 29443 \
                        --config_file configs/deepspeed/acc_val.yaml validate.py \
                        --tracker_project_name "nvs-vggt-distill" \
                        --seed 0 \
                        --run_name "$RUN_NAME" \
                        --viz_config_file "configs/visualize_val.yaml" \
                        --val_path="/mnt/data2/nvs-data/diff-distill/check_points/distill_0917_point_3sharpmlp" \
                        --validation_checkpoint="$CKPT_NAME" \
                        --val_cfg=2.0 \
                        --visualize_attention_maps

    python validate_wandb_correlation.py --output_postfix "$RUN_NAME"


done