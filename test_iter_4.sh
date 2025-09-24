#!/bin/bash
# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
export WANDB_API_KEY='3177b4c2c8cf009d18dc8cfc41cfa1d2fc813f67'

# Data
DATASETS=(re10k_wds_test_4_6 re10k_wds_test_4_10)
COND_NUM=1
# Model
RUN_NAME=0922_distill_perview_lr25_mlp1_L10_lw02
STEPS=(10000 20000 40000 80000 120000) # 120000 80000 6000
BASE_PORT=45277
for STEP in "${STEPS[@]}"; do
    i=0
    for DATASET in "${DATASETS[@]}"; do
        PORT=$((BASE_PORT + i))
        export CUDA_VISIBLE_DEVICES=5
        accelerate launch --mixed_precision="fp16" \
                        --num_processes=1 \
                        --num_machines 1 \
                        --main_process_port ${PORT} \
                        --config_file configs/deepspeed/acc_zero2.yaml test.py \
                        --tracker_project_name "nvs-vggt-distill-test" \
                        --seed 0 \
                        --test_remote_base_path /scratch/kaist-cvlab/jiho/nvsdiff-attn-distill/check_points \
                        --test_dataset_config configs/datasets/test.yaml \
                        --run_name "${RUN_NAME}" \
                        --test_train_step "${STEP}" \
                        --test_cfg 2.0 \
                        --test_num_inference_steps 50 \
                        --test_viz_num 2 \
                        --test_remote \
                        --test_dataset "${DATASET}" \
                        --test_nframe 4 \
                        --test_cond_num ${COND_NUM} \
                        --test_remote_remove_ckpt \
                        --test_save_csv_path check_points/results.csv \
                        --test_batch_size 10
        #                 #   --test_use_vggt_camera \
        #                 #   --test_compute_fid \
        ((i++))
    done
done