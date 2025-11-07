#!/bin/bash
# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
export WANDB_API_KEY='3177b4c2c8cf009d18dc8cfc41cfa1d2fc813f67'

# Data
WANDB_PROJECT_NAME="nvs-unet-feasibility-test"
DATASETS=(co3d_wds_val_subset_unseen_scene_refidx03_4_8 co3d_wds_val_subset_unseen_scene_refidx03_4_6 re10k_wds_val_refidx03_4_12 re10k_wds_val_refidx03_4_10)
COND_NUM=2
RUN_NAME=1105_naive_b8 # 1105_naive_b8, 1105_distill_b8_mlp1_L10_lw01
SAVE_CSV=check_points_gist/1106_mix.csv
STEPS=(100000) # (280000 320000 240000 200000 160000 120000 ) # (120000 10000 80000 20000 40000)  100000 60000
BASE_PORT=21424
for STEP in "${STEPS[@]}"; do
    i=0
    for DATASET in "${DATASETS[@]}"; do
        PORT=$((BASE_PORT + i))
        export CUDA_VISIBLE_DEVICES=7
        accelerate launch --mixed_precision="fp16" \
                        --num_processes=1 \
                        --num_machines 1 \
                        --main_process_port ${PORT} \
                        --config_file configs/deepspeed/acc_zero2.yaml test.py \
                        --tracker_project_name "${WANDB_PROJECT_NAME}" \
                        --seed 0 \
                        --test_dataset_config configs/datasets/test_all.yaml \
                        --run_base_path check_points_gist \
                        --run_name "${RUN_NAME}" \
                        --test_train_step "${STEP}" \
                        --test_cfg 2.0 \
                        --test_num_inference_steps 50 \
                        --test_viz_num 10 \
                        --test_dataset "${DATASET}" \
                        --test_cond_num ${COND_NUM} \
                        --test_save_csv_path ${SAVE_CSV} \
                        --test_batch_size 1 \
                        --test_eval_w_mask \
                        --test_remote \
                        --test_remote_base_path /scratch/kaist-cvlab/jiho/nvsdiff-attn-distill/check_points_gist/ 
                        # --test_remote_remove_ckpt 
                        #   --test_use_vggt_camera \
                        #   --test_compute_fid \
        ((i++))
    done
done