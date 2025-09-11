#!/bin/bash

export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16" \
                  --num_processes=2 \
                  --num_machines 1 \
                  --main_process_port 29443 \
                  --config_file configs/deepspeed/acc_zero2.yaml train.py \
                  --tracker_project_name "nvs-vggt-distill" \
                  --config_file="configs/cat3d_distill.yaml" \
                  --output_dir="check_points/cat3d_distill" \
                  --train_log_interval=500000000000 \
                  --val_interval=200 \
                  --val_cfg=2.0 \
                  --min_decay=0.5 \
                  --log_every 1 \
                  --seed 0 \
                  --no_eval \
                #   --use_ema \
                #   --ema_decay_step=30 \
                #   --ema_decay=0.9995 \

# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16" \
#                   --num_processes=2 \
#                   --num_machines 1 \
#                   --main_process_port 29443 \
#                   --config_file configs/deepspeed/acc_zero2.yaml train.py \
#                   --tracker_project_name "cat3d_2" \
#                   --config_file="configs/cat3d.yaml" \
#                   --output_dir="check_points/cat3d_2" \
#                   --train_log_interval=250 \
#                   --val_interval=200000000000 \
#                   --val_cfg=2.0 \
#                   --use_ema \
#                   --ema_decay=0.9995 \
#                   --min_decay=0.5 \
#                   --ema_decay_step=30 \
#                   --seed 0