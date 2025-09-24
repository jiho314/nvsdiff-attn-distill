#!/bin/bash

# This script now supports conditional logging and local saving.
# Set --use_wandb to log to Weights & Biases.
# Omit --use_wandb to save images locally to the --output_dir.

# export WANDB_API_KEY='...'
export CUDA_VISIBLE_DEVICES=2

# --- CONFIGURATION ---
USE_WANDB=false                 # Set to true or false
OUTPUT_DIR="./test_results_local_2000" # Directory for local image saving

# --- NEW: Individual Scene Testing ---
INDIVIDUAL_TEST=false            # Set to true to test one specific scene
TEST_ID="0 4 9"                      # The ID (index) of the scene you want to test

# --- BUILD ARGUMENTS ---
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
  WANDB_ARGS="--use_wandb"
else
  WANDB_ARGS="--output_dir $OUTPUT_DIR"
fi

# Add individual test arguments if enabled
INDIVIDUAL_ARGS=""
if [ "$INDIVIDUAL_TEST" = true ]; then
  INDIVIDUAL_ARGS="--individual_test --test_viz_id $TEST_ID"
fi

accelerate launch --mixed_precision="fp16" \
                  --num_processes=1 \
                  --num_machines 1 \
                  --main_process_port 29444 \
                  --config_file configs/deepspeed/acc_zero2.yaml test_final_qual.py \
                  --tracker_project_name "nvs-vggt-distill-test" \
                  --seed 0 \
                  --run_name final_qual_debugging \
                  --test_dataset_config configs/datasets/test.yaml \
                  --test_dataset re10k_wds_test \
                  --test_nframe 3 \
                  --test_cond_num 2 \
                  --test_use_vggt_camera \
                  --test_run_path /mnt/data1/jiho/vggt-nvs/nvsdiff-attn-distill/check_points/0919_naive_b3_lr2.5 \
                  --test_train_step 2000 \
                  --test_cfg 2.0 \
                  --test_num_inference_steps 50 \
                  --test_viz_num 100 \
                  $WANDB_ARGS \
                  $INDIVIDUAL_ARGS \
                  # --test_compute_fid \
                  # --use_ema

echo "Script finished."