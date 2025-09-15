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
        # keep common validation metrics
        if c in ('val/psnr', 'val/ssim', 'val/lpips')
        # include any attention_loss logged under val/ or other paths
        or ('attention_loss' in c)
        # support both 'val/stepN/...' and other variants like 'attn/.../step0/unet...'
        or ('/step' in c and 'unet' in c)
        # legacy/alternate prefixes
        or c.startswith('val/step')
        or c.startswith('val/correlation_')
        or c.startswith('attn/')
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
    # Accept attention columns logged under several naming schemes:
    #  - 'val/step{N}/unet...'
    #  - 'attn/.../step{N}/unet...'
    #  - any column containing '/step' and 'unet'
    # Collect attention columns from multiple logging schemes:
    #  - consolidated per-run: 'val/attention_loss/...'
    #  - per-sample per-step: 'attn/.../step{N}/unet...'
    #  - older per-step under 'val/step.../unet...'
    attn_cols = [
        c for c in metrics_df.columns
        if ('attention_loss' in c)
        or (('/step' in c or c.startswith('val/step')) and 'unet' in c)
        or c.startswith('attn/')
    ]

    if not metric_cols:
        raise RuntimeError('No metric columns (LPIPS/PSNR/SSIM) found in run history')
    if not attn_cols:
        raise RuntimeError('No attention loss columns found in run history')

    # Robust numeric extraction: wandb can log images/dicts/lists for attention columns.
    # Try to extract a scalar from common container types (numbers, strings, lists, dicts).
    import numbers
    import re
    def _extract_scalar(x):
        if x is None:
            return np.nan
        # pandas NA check
        try:
            if isinstance(x, float) and np.isnan(x):
                return np.nan
        except Exception:
            pass
        if isinstance(x, numbers.Number):
            return float(x)
        if isinstance(x, str):
            try:
                return float(x)
            except Exception:
                return np.nan
        if isinstance(x, (list, tuple, np.ndarray)):
            try:
                arr = np.array(x, dtype=float)
                if arr.size == 0:
                    return np.nan
                return float(np.nanmean(arr))
            except Exception:
                return np.nan
        if isinstance(x, dict):
            # prefer common scalar keys
            for k in ('loss', 'value', 'mean', 'map'):
                if k in x:
                    try:
                        return float(x[k])
                    except Exception:
                        pass
            # fallback: average numeric values
            vals = []
            for v in x.values():
                if isinstance(v, numbers.Number):
                    vals.append(float(v))
            if vals:
                return float(np.mean(vals))
            return np.nan
        return np.nan

    cols_to_process = metric_cols + attn_cols
    numeric = pd.DataFrame({c: metrics_df[c].apply(_extract_scalar) for c in cols_to_process})

    # helper: parse attention column names into (layer, head, step)
    def _parse_attn_name(name: str):
        layer = None
        head = None
        step = None
        m = re.search(r'step(?:[_/])?(\d+)', name)
        if not m:
            m = re.search(r'step(\d+)', name)
        if m:
            try:
                step = int(m.group(1))
            except Exception:
                step = None
        m = re.search(r'unet(\d+)', name)
        if m:
            try:
                layer = int(m.group(1))
            except Exception:
                layer = None
        m_all = list(re.finditer(r'head[_/:-]?(\d+)', name))
        if not m_all:
            m_all = list(re.finditer(r'head(\d+)', name))
        if m_all:
            try:
                head = int(m_all[-1].group(1))
            except Exception:
                head = None
        return layer, head, step

    # If attention columns are logged per-sample under 'attn/.../stepN/...',
    # aggregate them into per-(step,unet,type) series by averaging across samples
    sample_cols = [c for c in attn_cols if c.startswith('attn/')]
    aggregated_df = {}
    if sample_cols:
        # group sample columns by parsed (step, layer, type)
        groups = {}
        for c in sample_cols:
            layer, head, step = _parse_attn_name(c)
            short = c.split('/')[-1]
            t = short
            if layer is not None and layer != -1:
                t = re.sub(rf'unet{layer}_?', '', short)
            key = (step if step is not None else -1, layer if layer is not None else -1, t)
            groups.setdefault(key, []).append(c)
        # create aggregated columns (one per group)
        for (step, layer, t), cols_grp in groups.items():
            col_name = f"attn_step{step}_unet{layer}_{t}"
            # mean across sample columns (per history row)
            aggregated_df[col_name] = numeric[cols_grp].astype(float).mean(axis=1)
    # add aggregated columns to numeric (if any) and build final attn candidate list
    if aggregated_df:
        agg_df = pd.DataFrame(aggregated_df, index=numeric.index)
        numeric = pd.concat([numeric, agg_df], axis=1)
        # final attention list: aggregated cols + any non-sample attn cols (e.g., val/attention_loss/...)
        # Also canonicalize any existing per-step/unet columns (e.g. 'val/step10/unet4_...')
        extra_cols = []
        non_sample = [c for c in attn_cols if not c.startswith('attn/')]
        for c in non_sample:
            # if this column already represented in aggregated_df, skip
            if c in aggregated_df:
                continue
            # parse name and create standardized column name
            layer, head, step = _parse_attn_name(c)
            short = c.split('/')[-1]
            t = short
            if layer is not None and layer != -1:
                t = re.sub(rf'unet{layer}_?', '', short)
            col_name = f"attn_step{step if step is not None else -1}_unet{layer if layer is not None else -1}_{t}"
            if col_name not in numeric.columns:
                try:
                    numeric[col_name] = numeric[c].astype(float)
                except Exception:
                    numeric[col_name] = numeric[c]
            extra_cols.append(col_name)
        final_attn_cols = list(aggregated_df.keys()) + extra_cols
    else:
        final_attn_cols = attn_cols

    # replace attn_cols with the final list (includes aggregated per-step columns)
    attn_cols = final_attn_cols

    # Parse attention column names into (layer, head, step)
    def _parse_attn_name(name: str):
        # Robustly extract step, unet layer, and head index from a column name
        layer = None
        head = None
        step = None

        # step: look for 'step' followed by digits, possibly separated by '/' or '_'
        m = re.search(r'step(?:[_/])?(\d+)', name)
        if not m:
            m = re.search(r'step(\d+)', name)
        if m:
            try:
                step = int(m.group(1))
            except Exception:
                step = None

        # unet layer: 'unet{N}' pattern
        m = re.search(r'unet(\d+)', name)
        if m:
            try:
                layer = int(m.group(1))
            except Exception:
                layer = None

        # head index: patterns like 'head99' or 'head_99' or 'head-head99'
        m_all = list(re.finditer(r'head[_/:-]?(\d+)', name))
        if not m_all:
            m_all = list(re.finditer(r'head(\d+)', name))
        if m_all:
            try:
                head = int(m_all[-1].group(1))
            except Exception:
                head = None

        return layer, head, step

    # Filter columns that have at least MIN_SAMPLES numeric entries
    MIN_SAMPLES = 2
    good_metrics = [c for c in metric_cols if int(numeric[c].notna().sum()) >= MIN_SAMPLES]
    good_attn = [c for c in attn_cols if int(numeric[c].notna().sum()) >= MIN_SAMPLES]
    # Print counts but hide columns that have only 1 sample (user requested)
    counts = {c: int(numeric[c].notna().sum()) for c in numeric.columns}
    counts_filtered = {c: cnt for c, cnt in counts.items() if cnt > 1}
    print('Numeric candidate columns and non-null counts (showing >1):', counts_filtered)
    print('Using metric columns for correlation:', good_metrics)
    print('Using attention-loss columns for correlation:', good_attn)

    if not good_metrics or not good_attn:
        raise RuntimeError(f'Need at least one metric and one attention-loss column with >= {MIN_SAMPLES} samples')

    # Keep only the good columns
    numeric = numeric[good_metrics + good_attn]

    # Compute pairwise Pearson correlation between each metric and each attention column.
    # Require at least 2 paired samples and non-constant arrays to compute correlation.
    corr_df = pd.DataFrame(index=good_metrics, columns=good_attn, dtype=float)
    for mcol in good_metrics:
        for acol in good_attn:
            a = numeric[acol]
            b = numeric[mcol]
            mask = a.notna() & b.notna()
            if int(mask.sum()) < 2:
                corr = np.nan
            else:
                aa = a[mask].astype(float).to_numpy()
                bb = b[mask].astype(float).to_numpy()
                if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
                    corr = np.nan
                else:
                    try:
                        corr = float(np.corrcoef(aa, bb)[0, 1])
                    except Exception:
                        corr = np.nan
            corr_df.at[mcol, acol] = corr

    # Prepare plotting: heatmap with rows=metrics, cols=attention columns
    metric_list = good_metrics
    n_metrics = len(metric_list)
    attn_list = good_attn
    n_attn = len(attn_list)

    # Also compute mean attention loss per attn column for diagnostics
    attn_means = {acol: float(numeric[acol].mean()) if int(numeric[acol].notna().sum()) > 0 else np.nan for acol in attn_list}

    # Ensure output directory exists for diagnostic CSVs and images
    base, ext = os.path.splitext(out_path)
    out_dir = base + '_per_metric'
    os.makedirs(out_dir, exist_ok=True)

    # Build metadata for attention columns (layer, step, and short name)
    attn_meta = []
    for acol in attn_list:
        layer, head, step = _parse_attn_name(acol)
        short = acol.split('/')[-1]
        # derive a type string after the 'unet{N}_' prefix if present
        t = short
        if layer is not None and layer != -1:
            t = re.sub(rf'unet{layer}_?', '', short)
        attn_meta.append({'col': acol, 'short': short, 'layer': layer if layer is not None else -1, 'step': step if step is not None else -1, 'type': t})

    # Group attention columns by step -> by (layer, type) and aggregate correlations
    steps_present = sorted({m['step'] for m in attn_meta})
    if not steps_present:
        steps_present = [-1]

    # For each step produce two image-like subplots: track_head and point_map
    metric_display = {m: m if not m.endswith('_inv') else m for m in metric_list}
    for step in steps_present:
        cols_in_step = [m['col'] for m in attn_meta if m['step'] == step]
        if not cols_in_step:
            continue
        sub = corr_df[cols_in_step]

        # map col -> meta for this step
        meta_map = {m['col']: m for m in attn_meta if m['col'] in cols_in_step}

        def build_group_df(filter_fn):
            groups = {}
            for col, meta in meta_map.items():
                if not filter_fn(meta):
                    continue
                key = f"Unet {meta['layer']} - {meta['type']}"
                groups.setdefault(key, []).append(col)
            if not groups:
                return None
            # order rows by unet layer descending when possible
            def _layer_key(k):
                m = re.search(r'Unet\s+(-?\d+)', k)
                return -int(m.group(1)) if m else 0
            ordered = sorted(groups.items(), key=lambda kv: _layer_key(kv[0]))
            rows = []
            row_labels = []
            for key, cols_for_key in ordered:
                vals = sub[cols_for_key].astype(float).mean(axis=1)
                rows.append(vals.to_numpy())
                row_labels.append(key)
            heat = np.vstack(rows) if rows else np.empty((0, len(metric_list)))
            df = pd.DataFrame(heat, index=row_labels, columns=metric_list)
            return df

        df_track = build_group_df(lambda m: 'track' in m['type'])
        df_point = build_group_df(lambda m: 'point' in m['type'] or 'map' in m['type'])

        # prepare figure with two subplots side-by-side
        nrows = max((len(df_track.index) if df_track is not None else 0), (len(df_point.index) if df_point is not None else 0))
        if nrows == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, max(3, 0.35 * nrows)), sharey=False)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        cmap = 'RdBu_r'
        vmin, vmax = -1.0, 1.0

        # left: track
        ax = axes[0]
        if df_track is None or df_track.empty:
            ax.text(0.5, 0.5, 'no track data', ha='center', va='center')
            ax.axis('off')
        else:
            sns.heatmap(df_track, annot=True, fmt='.3f', cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('track head')

        # right: pointmap
        ax = axes[1]
        if df_point is None or df_point.empty:
            ax.text(0.5, 0.5, 'no pointmap data', ha='center', va='center')
            ax.axis('off')
        else:
            sns.heatmap(df_point, annot=True, fmt='.3f', cmap=cmap, vmin=vmin, vmax=vmax, cbar=True, ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title('point map')

        step_label = f'step{step}' if step != -1 else 'global'
        fig.suptitle(f'Correlation per-layer ({step_label})')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_file = os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_attn_metric_heatmap.png")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

        # save csvs
        if df_track is not None:
            df_track.to_csv(os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_track.csv"))
        if df_point is not None:
            df_point.to_csv(os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_pointmap.csv"))

    # Save overall corr and means as diagnostics
    corr_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_metric_corr.csv")
    corr_df.to_csv(corr_csv)
    means_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_means.csv")
    pd.DataFrame.from_dict(attn_means, orient='index', columns=['mean']).to_csv(means_csv)

    # finished: return used columns and numeric dataframe
    good_cols = good_metrics + good_attn
    return good_cols, numeric

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
    args.run_id = "jsh0423_/nvs-vggt-distill/599rlnpd"
    
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


