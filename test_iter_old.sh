#!/bin/bash
# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
export WANDB_API_KEY='3177b4c2c8cf009d18dc8cfc41cfa1d2fc813f67'

# Data
DATASETS=(re10k_wds_test_4_6 re10k_wds_test_4_10)
COND_NUM=1
# Model
RUN_NAME=0919_naive_b3_lr2.5
STEPS=(200000 240000 280000 320000 ) # (120000 10000 80000 20000 40000)  100000 60000
BASE_PORT=12348
for STEP in "${STEPS[@]}"; do
    i=0
    for DATASET in "${DATASETS[@]}"; do
        PORT=$((BASE_PORT + i))
        export CUDA_VISIBLE_DEVICES=0
        accelerate launch --mixed_precision="fp16" \
                        --num_processes=1 \
                        --num_machines 1 \
                        --main_process_port ${PORT} \
                        --config_file configs/deepspeed/acc_zero2.yaml test.py \
                        --tracker_project_name "nvs-vggt-distill-test" \
                        --seed 0 \
                        --test_remote \
                        --test_remote_base_path /scratch/kaist-cvlab/jiho/nvsdiff-attn-distill/check_points \
                        --test_dataset_config configs/datasets/test.yaml \
                        --run_name "${RUN_NAME}" \
                        --test_train_step "${STEP}" \
                        --test_cfg 2.0 \
                        --test_num_inference_steps 50 \
                        --test_viz_num 2 \
                        --test_dataset "${DATASET}" \
                        --test_cond_num ${COND_NUM} \
                        --test_remote_remove_ckpt \
                        --test_save_csv_path check_points/results_1002.csv \
                        --test_batch_size 1
        #                 #   --test_use_vggt_camera \
        #                 #   --test_compute_fid \
        ((i++))
    done
done