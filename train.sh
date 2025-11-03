#!/bin/bash

# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463' #jiho
export WANDB_API_KEY='3177b4c2c8cf009d18dc8cfc41cfa1d2fc813f67' # minkyung
CUDA_VISIBLE_DEVICES=5 accelerate launch --mixed_precision="bf16" \
                  --num_processes=1 --num_machines 1 --main_process_port 21342 \
                  --config_file configs/deepspeed/acc_zero2_bf16.yaml train.py \
                  --tracker_project_name "nvs-unet-feasibility" \
                  --output_dir="check_points/1101_naive_b10_mix_last" \
                  --train_log_interval=100000000 \
                  --val_interval=10000 \
                  --val_cfg=2.0 \
                  --min_decay=0.5 \
                  --log_every 4 \
                  --seed 0 \
                  --run_name 1101_naive_b10_mix_last \
                  --config_file="check_points/1101_naive_b10_mix_last/config.yaml" \
                  --num_workers_per_gpu 1 \
                  --checkpointing_last_steps 5000 \
                  # --val_at_first
                  # --resume_from_last
                  # --resume_from_checkpoint checkpoint-20000 \
                  # --resume_path="check_points/1019_distill_b12_crossperview_mlp1_L10_lw02_sharp" \
                  # --only_resume_weight \
                  # --val_at_first \
                  # no scheduling, just resume weight
                #   --use_ema \
                #   --ema_decay_step=30 \
                #   --ema_decay=0.9995 \
                #   --resume_from_checkpoint checkpoint-2000
                #   --resume_path dir

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