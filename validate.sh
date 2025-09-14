#!/bin/bash

export WANDB_API_KEY='5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe' # seonghu, for debugging

# Temporary visualize config for attention map visualization

CUDA_VISIBLE_DEVICES=3 accelerate launch --mixed_precision="fp16" \
                  --num_processes=1 \
                  --num_machines 1 \
                  --main_process_port 29443 \
                  --config_file configs/deepspeed/acc_val.yaml validate.py \
                  --tracker_project_name "nvs-vggt-distill" \
                  --seed 0 \
                  --val_path="check_points/cat3d" \
                  --validation_checkpoint="checkpoint-4000" \
                  --val_cfg=2.0 \
                  --visualize_attention_maps \
