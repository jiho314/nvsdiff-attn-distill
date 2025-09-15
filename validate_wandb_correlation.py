import argparse
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import wandb


def fetch_run_metrics(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history(samples=100000)
    return history


def compute_and_save_correlation(history_df, out_path):
    # 기대되는 컬럼들: 'val/psnr', 'val/ssim', 'val/lpips', 그리고 새로 추가된 'val/stepNN/...' attention 로그
    # 주의: 기존 'val/attention_loss/...' 타입 컬럼은 사용하지 않고, `val/step.../unet..._..._head...` 패턴만 고려합니다.
    cols = [
        c for c in history_df.columns
        if c in ('val/psnr', 'val/ssim', 'val/lpips')
        or c.startswith('val/step')
        or c.startswith('val/correlation_')
    ]
    if not cols:
        raise RuntimeError('No validation metric columns found in run history')

    # Work on a copy of the selected raw columns (do NOT drop rows here)
    metrics_df = history_df[cols].copy()

    # Normalize / rename metrics
    if 'val/lpips' in metrics_df.columns:
        metrics_df['LPIPS'] = metrics_df['val/lpips']
    if 'val/psnr' in metrics_df.columns:
        metrics_df['PSNR_inv'] = -metrics_df['val/psnr']
    if 'val/ssim' in metrics_df.columns:
        metrics_df['SSIM_inv'] = -metrics_df['val/ssim']

    # Identify metric columns and attention columns logged per step
    # Support LPIPS, PSNR (inverted), SSIM (inverted) for correlation
    metric_cols = [c for c in ('LPIPS', 'PSNR_inv', 'SSIM_inv') if c in metrics_df.columns]
    # Keep columns that follow the step/unet pattern (headed or headless),
    # e.g. 'val/step40/unet4_vggttrack_head_head9' or 'val/step40/unet4_vggtpoint_map'
    attn_cols = [
        c for c in metrics_df.columns
        if c.startswith('val/step') and 'unet' in c
    ]

    if not metric_cols:
        raise RuntimeError('No metric columns (LPIPS/PSNR/SSIM) found in run history')
    if not attn_cols:
        raise RuntimeError('No attention loss columns found in run history')

    # Coerce to numeric where possible
    numeric = metrics_df[metric_cols + attn_cols].apply(pd.to_numeric, errors='coerce')

    # Filter columns that have at least MIN_SAMPLES numeric entries
    # Lowered to 1 to include newly-logged attention columns (many heads/steps)
    MIN_SAMPLES = 1
    good_metrics = [c for c in metric_cols if int(numeric[c].notna().sum()) >= MIN_SAMPLES]
    good_attn = [c for c in attn_cols if int(numeric[c].notna().sum()) >= MIN_SAMPLES]
    print('Numeric candidate columns and non-null counts:', {c: int(numeric[c].notna().sum()) for c in numeric.columns})
    print('Using metric columns for correlation:', good_metrics)
    print('Using attention-loss columns for correlation:', good_attn)

    if not good_metrics or not good_attn:
        raise RuntimeError(f'Need at least one metric and one attention-loss column with >= {MIN_SAMPLES} samples')

    # Keep only the good columns
    numeric = numeric[good_metrics + good_attn]

    # Parse attention column names into (layer, head, step)
    import re
    def _parse_attn_name(name: str):
        # Expected format examples:
        #  - val/step40/unet4_vggttrack_head_head99
        #  - val/step40/unet12_vggttrack_head_head3
        # Robustly extract step, unet layer, and head index.
        layer = None
        head = None
        step = None

        # step: look for 'step' followed by digits, possibly inside path-like segments
        m = re.search(r'step(?:[_/])?(\d+)', name)
        if not m:
            m = re.search(r'step(\d+)', name)
        if m:
            step = int(m.group(1))

        # unet layer: 'unet{N}' or 'unet{N}_' patterns
        m = re.search(r'unet(\d+)', name)
        if m:
            layer = int(m.group(1))

        # head: patterns like 'head_head99' or 'head99' or '_head_99'
        # find the last occurrence of 'head' followed by optional separators and digits
        m_all = list(re.finditer(r'head[_/:-]?(\d+)', name))
        if not m_all:
            # fallback: any 'head' followed by digits without separator
            m_all = list(re.finditer(r'head(\d+)', name))
        if m_all:
            # prefer the last match which often corresponds to the head index
            m = m_all[-1]
            head = int(m.group(1))

        return layer, head, step

    attn_meta = []
    for acol in good_attn:
        layer, head, step = _parse_attn_name(acol)
        attn_meta.append({
            'col': acol,
            'layer': layer if layer is not None else -1,
            'head': head if head is not None else -1,
            'step': step if step is not None else -1,
        })

    # Collect unique sorted indices
    layers = sorted({m['layer'] for m in attn_meta})
    heads = sorted({m['head'] for m in attn_meta})
    steps = sorted({m['step'] for m in attn_meta})

    if not layers or not heads or not steps:
        raise RuntimeError('Parsed attention columns do not contain layer/head/step information')

    # Build correlation mapping: (metric, step, head, layer) -> correlation
    corr_map = {}
    for mcol in good_metrics:
        for meta in attn_meta:
            acol = meta['col']
            try:
                val = float(numeric[acol].corr(numeric[mcol]))
            except Exception:
                val = np.nan
            corr_map[(mcol, meta['step'], meta['head'], meta['layer'])] = val

    # Prepare plotting: produce two figures per run:
    #  1) correlation heatmaps for each metric (rows=metrics, cols=steps), values in [-1,1] (RdBu)
    #  2) attention loss values per step (mean over history) visualized similarly, normalized per-layer to [0,1]
    metric_list = good_metrics
    n_metrics = len(metric_list)
    n_steps = len(steps)
    all_heads = list(heads)
    n_heads = len(all_heads)
    n_layers = len(layers)

    # Build matrices for each metric and step, then save each (metric, step) as a separate image file.
    corr_mats = {m: {} for m in metric_list}
    loss_mats = {}
    for step in steps:
        loss = np.full((n_heads, n_layers), np.nan, dtype=float)
        for hi, h in enumerate(all_heads):
            for li, l in enumerate(layers):
                matching = [m for m in attn_meta if m['step'] == step and m['head'] == h and m['layer'] == l]
                if matching:
                    acol = matching[0]['col']
                    loss_val = float(numeric[acol].mean()) if int(numeric[acol].notna().sum()) > 0 else np.nan
                else:
                    loss_val = np.nan
                loss[hi, li] = loss_val
        loss_mats[step] = loss
        for mcol in metric_list:
            corr = np.full((n_heads, n_layers), np.nan, dtype=float)
            for hi, h in enumerate(all_heads):
                for li, l in enumerate(layers):
                    corr_val = corr_map.get((mcol, step, h, l), np.nan)
                    corr[hi, li] = corr_val
            corr_mats[mcol][step] = corr

    # Ensure output directory exists for diagnostic CSVs and images
    base, ext = os.path.splitext(out_path)
    out_dir = base + '_per_metric'
    os.makedirs(out_dir, exist_ok=True)

    # Diagnostic: save and print loss statistics per step to help debug unexpected values
    for step in steps:
        mat = loss_mats[step]
        flat = mat.ravel()
        finite = flat[np.isfinite(flat)]
        cnt = int(np.isfinite(flat).sum())
        if finite.size > 0:
            mn = float(np.nanmin(finite))
            mx = float(np.nanmax(finite))
            avg = float(np.nanmean(finite))
        else:
            mn = mx = avg = float('nan')
        print(f"Loss stats step={step}: count={cnt}, min={mn}, max={mx}, mean={avg}")
        # save raw per-head x per-layer matrix to CSV for inspection
        df_loss = pd.DataFrame(mat, index=[f'Head_{h}' for h in all_heads], columns=[f'Unet_{l}' for l in layers])
        csv_path = os.path.join(out_dir, f"loss_step{step}.csv")
        df_loss.to_csv(csv_path)

    # out_dir already created above

    # Save one image per metric containing ALL steps as subplots (use raw correlation values, vmin/vmax fixed to [-1,1])
    for mcol in metric_list:
        fig, axes = plt.subplots(1, n_steps, figsize=(max(4, n_steps * 3), max(4, n_heads * 0.12)))
        if n_steps == 1:
            axes = np.array([axes])
        for j_s, step in enumerate(steps):
            ax = axes[j_s]
            mat = corr_mats[mcol][step]
            row_labels = [f'Head {h}' for h in all_heads]
            col_labels = [f'Unet {l}' for l in layers]
            df = pd.DataFrame(mat, index=row_labels, columns=col_labels)
            ytick = True if j_s == 0 else False
            sns.heatmap(df, annot=False, fmt='.3f', cmap='RdBu_r', vmin=-1.0, vmax=1.0,
                        cbar=(j_s == n_steps-1), ax=ax, yticklabels=ytick, xticklabels=True)
            ax.set_xlabel('Layer')
            if j_s == 0:
                ax.set_ylabel('Head')
            else:
                ax.set_ylabel('')
            # place the step title lower so annotations can be placed above without overlap
            ax.set_title(f"step {step}", y=0.88)
            ax.tick_params(axis='x', rotation=90)
            # compute per-layer mean (average over heads) and display above columns
            col_means = np.nanmean(mat, axis=0)
            formatted = []
            for v in col_means:
                if np.isnan(v):
                    formatted.append('nan')
                else:
                    # format: >=1 -> 2 decimals, <1 -> 3 decimals
                    if abs(v) >= 1:
                        formatted.append(f"{v:.2f}")
                    else:
                        formatted.append(f"{v:.3f}")
            # annotation y-position in axis fraction (above the axes top); tune to avoid overlap with title
            ann_y = 0.96
            for xi, lab in enumerate(formatted):
                x_frac = (xi + 0.5) / float(len(col_means))
                ax.annotate(lab, xy=(x_frac, ann_y), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, rotation=0, clip_on=False)
        # increase top margin so annotations are not clipped and do not overlap titles
        fig.subplots_adjust(top=0.86)
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"{os.path.basename(base)}_{mcol}.png")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    # Save one loss image containing ALL steps as subplots using RAW mean loss values (no normalization)
    # use per-step vmin/vmax so each heatmap uses its own value range
    fig, axes = plt.subplots(1, n_steps, figsize=(max(4, n_steps * 3), max(4, n_heads * 0.12)))
    if n_steps == 1:
        axes = np.array([axes])
    for j_s, step in enumerate(steps):
        ax = axes[j_s]
        mat = loss_mats[step]
        row_labels = [f'Head {h}' for h in all_heads]
        col_labels = [f'Unet {l}' for l in layers]
        df = pd.DataFrame(mat, index=row_labels, columns=col_labels)
        finite_vals = df.values[np.isfinite(df.values)]
        if finite_vals.size > 0:
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
        else:
            vmin, vmax = 0.0, 1.0
        ytick = True if j_s == 0 else False
        sns.heatmap(df, annot=False, fmt='.3f', cmap='viridis', vmin=vmin, vmax=vmax,
                    cbar=(j_s == n_steps-1), ax=ax, yticklabels=ytick, xticklabels=True)
        ax.set_xlabel('Layer')
        if j_s == 0:
            ax.set_ylabel('Head')
        else:
            ax.set_ylabel('')
        ax.set_title(f"step {step}", y=0.88)
        ax.tick_params(axis='x', rotation=90)
    fig.subplots_adjust(top=0.86)
    plt.tight_layout()
    out_file = os.path.join(out_dir, f"{os.path.basename(base)}_loss.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    # For downstream summary keep behavior compatible: return used numeric columns and numeric df
    good_cols = good_metrics + good_attn
    return good_cols, numeric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='validation_metrics_correlation_wandb.png')
    args = parser.parse_args()

    os.environ['WANDB_API_KEY'] = "5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe"
    # args.run_id = "jsh0423_/nvs-vggt-distill/kkprdt1l" # 2000
    # args.run_id = "jsh0423_/nvs-vggt-distill/96m4i9kp" # 6000
    args.run_id = "jsh0423_/nvs-vggt-distill/of5e6c4m"
    
    history = fetch_run_metrics(args.run_id)

    # Diagnostic: print available columns and non-null counts
    print('Available columns in run history:')
    for c in history.columns:
        non_null = int(history[c].notna().sum())
        print(f"  {c}: non-null={non_null}")

    # Compute and save correlation plot; receive selected good_cols and numeric df
    good_cols, numeric_df = compute_and_save_correlation(history, args.out)

    # Print summary (mean/std) only for good_cols
    # print('\nSummary for used columns:')
    # for c in good_cols:
    #     non_null = int(numeric_df[c].notna().sum())
    #     mean = float(numeric_df[c].mean())
    #     std = float(numeric_df[c].std())
    #     print(f"  {c}: count={non_null}, mean={mean:.6f}, std={std:.6f}")

    # # Upload PNG to the run via API (do not call run.log_code which is not available on Api.run)
    # api = wandb.Api()
    # run = api.run(args.run_id)
    # # upload_file is available on Run objects from the API; log_code is not
    # run.upload_file(args.out)
    # print('Saved and uploaded correlation plot to:', args.out)


if __name__ == '__main__':
    main()


