import argparse
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import wandb


def fetch_run_metrics(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history(samples=100000)
    return history


def compute_and_save_correlation(history_df, out_path):
    # 기대되는 컬럼들: 'val/psnr', 'val/ssim', 'val/lpips', 'val/attention_loss/*'
    cols = [c for c in history_df.columns if c in ('val/psnr', 'val/ssim', 'val/lpips') or c.startswith('val/attention_loss/') or c.startswith('val/correlation_')]
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

    # Build candidate columns list (include attention-loss scalar keys if present)
    use_cols = [c for c in ('LPIPS', 'PSNR_inv', 'SSIM_inv') if c in metrics_df.columns]
    attn_cols = [c for c in metrics_df.columns if c.startswith('val/attention_loss/')]
    use_cols += attn_cols

    if not use_cols:
        raise RuntimeError('No candidate metric columns found after normalization. Available: ' + ','.join(list(metrics_df.columns)))

    # Coerce to numeric where possible (artifact/image columns will become NaN)
    numeric = metrics_df[use_cols].apply(pd.to_numeric, errors='coerce')

    # Filter columns that have at least MIN_SAMPLES numeric entries
    MIN_SAMPLES = 5
    good_cols = [c for c in numeric.columns if int(numeric[c].notna().sum()) >= MIN_SAMPLES]
    print('Numeric candidate columns and non-null counts:', {c: int(numeric[c].notna().sum()) for c in numeric.columns})
    print('Using good_cols for correlation:', good_cols)

    if not good_cols:
        raise RuntimeError(f'No numeric columns with at least {MIN_SAMPLES} samples found. Cannot compute correlation.')

    numeric = numeric[good_cols]
    numeric = numeric.reindex(sorted(numeric.columns), axis=1)

    corr = numeric.corr()

    plt.figure(figsize=(max(6, corr.shape[0]*1.5), max(6, corr.shape[0]*1.5)))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Return selected numeric dataframe and columns for downstream summary
    return good_cols, numeric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='validation_metrics_correlation_wandb.png')
    args = parser.parse_args()

    os.environ['WANDB_API_KEY'] = "5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe"
    # args.run_id = "jsh0423_/nvs-vggt-distill/kkprdt1l" # 2000
    # args.run_id = "jsh0423_/nvs-vggt-distill/96m4i9kp" # 6000
    args.run_id = "jsh0423_/nvs-vggt-distill/mg01dyae" # 4000

    history = fetch_run_metrics(args.run_id)

    # Diagnostic: print available columns and non-null counts
    print('Available columns in run history:')
    for c in history.columns:
        non_null = int(history[c].notna().sum())
        print(f"  {c}: non-null={non_null}")

    # Compute and save correlation plot; receive selected good_cols and numeric df
    good_cols, numeric_df = compute_and_save_correlation(history, args.out)

    # Print summary (mean/std) only for good_cols
    print('\nSummary for used columns:')
    for c in good_cols:
        non_null = int(numeric_df[c].notna().sum())
        mean = float(numeric_df[c].mean())
        std = float(numeric_df[c].std())
        print(f"  {c}: count={non_null}, mean={mean:.6f}, std={std:.6f}")

    # Upload PNG to the run via API (do not call run.log_code which is not available on Api.run)
    api = wandb.Api()
    run = api.run(args.run_id)
    # upload_file is available on Run objects from the API; log_code is not
    run.upload_file(args.out)
    print('Saved and uploaded correlation plot to:', args.out)


if __name__ == '__main__':
    main()


