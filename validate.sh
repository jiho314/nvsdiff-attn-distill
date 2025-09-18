#!/bin/bash

export WANDB_API_KEY='5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe' # seonghu, for debugging

# Temporary visualize config for attention map visualization

ckpt=16000
CKPT_NAME="checkpoint-${ckpt}"
RUN_NAME="naive_val_run_${ckpt}_$(date +%Y%m%d_%H%M%S)"
export RUN_NAME
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" \
                    --num_processes=1 \
                    --num_machines 1 \
                    --main_process_port 29443 \
                    --config_file configs/deepspeed/acc_val.yaml validate.py \
                    --tracker_project_name "nvs-vggt-distill" \
                    --seed 0 \
                    --run_name "$RUN_NAME" \
                    --viz_config_file "configs/visualize_val.yaml" \
                    --val_path="check_points/distill_0917_point_3sharpmlp" \
                    --validation_checkpoint="$CKPT_NAME" \
                    --val_cfg=2.0 \
                    --visualize_attention_maps

# Attempt to auto-detect the wandb run created by this validation run
echo "Attempting to auto-detect wandb run for $RUN_NAME ..."
RUN_ID=$(python - <<'PY'
import glob,os,json,sys
import wandb
run_name = os.environ.get('RUN_NAME')
# Try local wandb metadata first
cand = sorted(glob.glob('**/wandb/run-*', recursive=True), key=os.path.getmtime, reverse=True)
for rd in cand:
    meta = os.path.join(rd, 'wandb-metadata.json')
    if os.path.exists(meta):
        try:
            d = json.load(open(meta))
            entity = d.get('entity') or d.get('user') or d.get('username') or ''
            project = d.get('project') or ''
            run = d.get('run_id') or d.get('id') or ''
            if run:
                if entity and project:
                    print(entity + "/" + project + "/" + run)
                elif project:
                    print(project + "/" + run)
                else:
                    print(run)
                sys.exit(0)
        except Exception:
            continue

# Fallback: use wandb API to find run by displayName/name matching RUN_NAME
try:
    api = wandb.Api()
    project = 'nvs-vggt-distill'
    runs = api.runs(project, per_page=50)
    for r in runs:
        if getattr(r, 'displayName', None) == run_name or getattr(r, 'name', None) == run_name:
            print(r.path)
            sys.exit(0)
except Exception:
    pass

sys.exit(0)
PY
)

    if [ -z "$RUN_ID" ]; then
        echo "Could not detect RUN_ID for $RUN_NAME; skipping correlation for this checkpoint."
    else
        echo "Detected RUN_ID: $RUN_ID"
        python validate_wandb_correlation.py --run_id "$RUN_ID" --output_postfix "$RUN_NAME"
    fi
done