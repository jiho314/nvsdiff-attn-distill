#!/bin/bash
# export WANDB_API_KEY='4ab8d4a0db9aec6c80956ccf58616de15392a463'
export WANDB_API_KEY='3177b4c2c8cf009d18dc8cfc41cfa1d2fc813f67'

# Data
# DATASETS=(co3d_wds_test_4_6 co3d_wds_test_4_8)
DATASETS=(co3d_wds_val_subset_unseen_scene_3_3 co3d_wds_val_subset_unseen_category_3_3 )
COND_NUM=1
RUN_NAME=1030_distill_b10_crossperview_mlp1_L10_lw02_sharp_mix # 0919_naive_b3_lr2.5
SAVE_CSV=check_points/1031_mix_1.csv
STEPS=(55000 ) # (280000 320000 240000 200000 160000 120000 ) # (120000 10000 80000 20000 40000)  100000 60000
BASE_PORT=21424
for STEP in "${STEPS[@]}"; do
    i=0
    for DATASET in "${DATASETS[@]}"; do
        PORT=$((BASE_PORT + i))
        export CUDA_VISIBLE_DEVICES=2
        accelerate launch --mixed_precision="fp16" \
                        --num_processes=1 \
                        --num_machines 1 \
                        --main_process_port ${PORT} \
                        --config_file configs/deepspeed/acc_zero2.yaml test.py \
                        --tracker_project_name "nvs-unet-feasibility-test" \
                        --seed 0 \
                        --test_dataset_config configs/datasets/test.yaml \
                        --run_base_path check_points \
                        --run_name "${RUN_NAME}" \
                        --test_train_step "${STEP}" \
                        --test_cfg 2.0 \
                        --test_num_inference_steps 50 \
                        --test_viz_num 10 \
                        --test_dataset "${DATASET}" \
                        --test_cond_num ${COND_NUM} \
                        --test_save_csv_path ${SAVE_CSV} \
                        --test_batch_size 1 \
                        --test_eval_w_mask 
        #                 #   --test_use_vggt_camera \
        #                 #   --test_compute_fid \
                        # --test_remote \
                        # --test_remote_base_path /scratch/kaist-cvlab/jiho/nvsdiff-attn-distill/check_points \
                        # --test_remote_remove_ckpt \
        ((i++))
    done
done