#!/bin/bash

# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
export WANDB_API_KEY='3177b4c2c8cf009d18dc8cfc41cfa1d2fc813f67'
export CUDA_VISIBLE_DEVICES=0
accelerate launch --mixed_precision="fp16" \
                  --num_processes=1 \
                  --num_machines 1 \
                  --main_process_port 29443 \
                  --config_file configs/deepspeed/acc_zero2.yaml test.py \
                  --tracker_project_name "nvs-vggt-distill-test" \
                  --seed 0 \
                  --run_name test_ho \
                  --test_dataset_config configs/datasets/test.yaml \
                  --test_dataset dtu_test \
                  --test_nframe 6 \
                  --test_cond_num 2 \
                  --test_use_vggt_camera \
                  --test_run_path check_points/lr1_cosine_noema \
                  --test_train_step 30000 \
                  --test_cfg 2.0 \
                  --test_num_inference_steps 50 \
                  --test_viz_num 15 \
                  --test_compute_fid \
                  # --use_ema


