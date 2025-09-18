import argparse
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import wandb
from scipy.stats import pearsonr, norm
from datetime import datetime


def fetch_run_metrics(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    history = run.history(samples=100000)
    return history


def compute_and_save_correlation(history_df, out_path, output_postfix=None):
    # 기대되는 컬럼들: 'val/psnr', 'val/ssim', 'val/lpips', 그리고 새로 추가된 'val/stepNN/...' attention 로그
    # 주의: 기존 'val/attention_loss/...' 타입 컬럼은 사용하지 않고, `val/step.../unet..._..._head...` 패턴만 고려합니다.
    # Only consider columns logged under the 'val/' prefix to avoid mixing per-sample 'attn/' namespaces
    cols = [c for c in history_df.columns if str(c).startswith('val/')]
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
    # Only pick attention-related columns that are logged under the 'val/' prefix
    attn_cols = [
        c for c in metrics_df.columns
        if str(c).startswith('val/') and (
            ('attention_loss' in c)
            or ('unet' in c)
            or ('cross_entropy' in c)
            or ('vggt' in c)
            or ('head' in c)
            or ('point_map' in c)
        )
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

    # Some logging schemes produce duplicate column labels (e.g. repeated 'attn_step-1_unet4_cross_entropy').
    # Pandas operations below assume unique column labels; if duplicates exist, aggregate them by taking the mean
    # across duplicate columns to produce a single representative series per label.
    try:
        if numeric.columns.duplicated().any():
            dup_cols = [c for c, dup in zip(numeric.columns, numeric.columns.duplicated()) if dup]
            # perform groupby on column labels and average duplicates
            numeric = numeric.groupby(by=numeric.columns, axis=1).mean()
            # update attn_cols/metric_cols lists to the deduplicated set
            # keep only those labels present after grouping
            cols_after = list(numeric.columns)
            metric_cols = [c for c in metric_cols if c in cols_after]
            attn_cols = [c for c in attn_cols if c in cols_after]
    except Exception:
        # if grouping fails for any reason, fall back to dropping exact duplicate columns
        try:
            numeric = numeric.loc[:, ~numeric.columns.duplicated()]
            metric_cols = [c for c in metric_cols if c in numeric.columns]
            attn_cols = [c for c in attn_cols if c in numeric.columns]
        except Exception:
            pass

    # helper: parse attention column names into (layer, head, step)
    def _parse_attn_name(name: str):
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

        # head index: patterns like 'head99' or 'head_99' or 'head-head99' or trailing '/head9'
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
        # create aggregated columns (one per group) and name them under 'val/step{N}/...'
        for (step, layer, t), cols_grp in groups.items():
            col_name = f"val/step{step}/unet{layer}_{t}"
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
            # robust parsing: support trailing '/P{fn}' suffix and varied path segments
            parts = c.split('/')
            fn = None
            base_part = parts[-1]
            # pattern like .../<base_part_prev>/P<fn>
            if base_part.startswith('P') and len(parts) >= 2:
                fn = base_part[1:]
                base_part = parts[-2]
            # trailing '/headN' path: prefer the previous segment as the base (e.g., '.../cross_entropy/head9')
            m_head_path = re.match(r'head[_/:-]?(\d+)$', base_part)
            if m_head_path and len(parts) >= 2:
                # shift base to previous segment so 'type' becomes 'cross_entropy' instead of 'head9'
                base_part = parts[-2]

            # parse layer/step from the full column name
            layer, head, step = _parse_attn_name(c)

            short = base_part
            t = short
            if layer is not None and layer != -1:
                t = re.sub(rf'unet{layer}_?', '', short)

            # append fn suffix to standardized col name if present
            fn_suffix = f"_P{fn}" if fn is not None else ""
            col_name = f"attn_step{step if step is not None else -1}_unet{layer if layer is not None else -1}_{t}{fn_suffix}"
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
    # ensure uniqueness to avoid pandas reindex errors later
    # preserve order
    def _unique_preserve(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    good_metrics = _unique_preserve(good_metrics)
    good_attn = _unique_preserve(good_attn)
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

    # Compute pairwise Pearson correlation, p-value, and 95% CI between each metric and each attention column.
    # Require at least 2 paired samples and non-constant arrays to compute correlation; CI needs n > 3.
    corr_df = pd.DataFrame(index=good_metrics, columns=good_attn, dtype=float)
    p_df = pd.DataFrame(index=good_metrics, columns=good_attn, dtype=float)
    ci_lower_df = pd.DataFrame(index=good_metrics, columns=good_attn, dtype=float)
    ci_upper_df = pd.DataFrame(index=good_metrics, columns=good_attn, dtype=float)
    for mcol in good_metrics:
        for acol in good_attn:
            a = numeric[acol]
            b = numeric[mcol]
            mask = a.notna() & b.notna()
            n_paired = int(mask.sum())
            if n_paired < 2:
                r_val = np.nan
                p_val = np.nan
                lo = np.nan
                hi = np.nan
            else:
                aa = a[mask].astype(float).to_numpy()
                bb = b[mask].astype(float).to_numpy()
                if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
                    r_val = np.nan
                    p_val = np.nan
                    lo = np.nan
                    hi = np.nan
                else:
                    try:
                        r_val, p_val = pearsonr(aa, bb)
                    except Exception:
                        # fallback to numpy-based estimate if pearsonr fails
                        try:
                            r_val = float(np.corrcoef(aa, bb)[0, 1])
                            p_val = np.nan
                        except Exception:
                            r_val = np.nan
                            p_val = np.nan
                    # 95% CI using Fisher z-transform when enough samples
                    if n_paired > 3 and (not np.isnan(r_val)) and abs(r_val) < 0.9999999:
                        try:
                            fisher_z = np.arctanh(r_val)
                            se = 1.0 / math.sqrt(n_paired - 3)
                            z_crit = norm.ppf(0.975)
                            lo = np.tanh(fisher_z - z_crit * se)
                            hi = np.tanh(fisher_z + z_crit * se)
                        except Exception:
                            lo = np.nan
                            hi = np.nan
                    else:
                        lo = np.nan
                        hi = np.nan
            corr_df.at[mcol, acol] = float(r_val) if not (isinstance(r_val, float) and np.isnan(r_val)) else np.nan
            p_df.at[mcol, acol] = float(p_val) if not (isinstance(p_val, float) and np.isnan(p_val)) else np.nan
            ci_lower_df.at[mcol, acol] = float(lo) if not (isinstance(lo, float) and np.isnan(lo)) else np.nan
            ci_upper_df.at[mcol, acol] = float(hi) if not (isinstance(hi, float) and np.isnan(hi)) else np.nan

    # Prepare plotting: heatmap with rows=metrics, cols=attention columns
    metric_list = good_metrics
    n_metrics = len(metric_list)
    attn_list = good_attn
    n_attn = len(attn_list)

    # Also compute mean attention loss per attn column for diagnostics
    attn_means = {acol: float(numeric[acol].mean()) if int(numeric[acol].notna().sum()) > 0 else np.nan for acol in attn_list}

    # Ensure output directory exists under `validation_metrics_correlation_wandb_per_metric/<timestamp>`
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = 'validation_metrics_correlation_wandb_per_metric'
    os.makedirs(out_root, exist_ok=True)
    # include optional postfix in output directory name for easier identification
    postfix = '' if not output_postfix else f"_{output_postfix}"
    out_dir = os.path.join(out_root, f"{ts}{postfix}")
    os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(out_path)

    # Build metadata for attention columns (layer, head, step, and short name)
    attn_meta = []
    for acol in attn_list:
        layer, head, step = _parse_attn_name(acol)
        parts = acol.split('/')
        short = parts[-1]
        # if the last segment is a head (e.g., 'head9'), prefer the previous segment as the short/type
        if re.match(r'head[_/:-]?(\d+)$', short) and len(parts) >= 2:
            short = parts[-2]
        # derive a type string after the 'unet{N}_' prefix if present
        t = short
        if layer is not None and layer != -1:
            t = re.sub(rf'unet{layer}_?', '', short)
        attn_meta.append({'col': acol, 'short': short, 'layer': layer if layer is not None else -1, 'head': head if head is not None else -1, 'step': step if step is not None else -1, 'type': t})

    # Group attention columns by step -> by (layer, type) and aggregate correlations
    steps_present = sorted({m['step'] for m in attn_meta})
    if not steps_present:
        steps_present = [-1]

    # For each step produce plots per variation/type (e.g., 'pointmap', 'pointmap_Psoftargmax_l2', 'track_head', ...)
    metric_display = {m: m if not m.endswith('_inv') else m for m in metric_list}
    for step in steps_present:
        cols_in_step = [m['col'] for m in attn_meta if m['step'] == step]
        if not cols_in_step:
            continue
        sub = corr_df[cols_in_step]

        # map col -> meta for this step
        meta_map = {m['col']: m for m in attn_meta if m['col'] in cols_in_step}

        # collect unique types for this step; group by type and then aggregate per unet layer
        # types will be something like 'vggtpoint_map/cross_entropy' or similar
        types = sorted({m['type'] for m in meta_map.values()})
        if not types:
            continue

        cmap = 'RdBu_r'
        vmin, vmax = -1.0, 1.0

        for t_type in types:
            # For this type, group columns by unet layer and within each layer include all heads (head1..N) as separate rows
            layers = sorted({m['layer'] for m in meta_map.values() if m['type'] == t_type})
            if not layers:
                continue

            for layer in layers:
                # gather columns for this (type, layer), organized by head
                head_to_cols = {}
                for col, meta in meta_map.items():
                    if meta['type'] != t_type or meta['layer'] != layer:
                        continue
                    head = meta.get('head', -1)
                    head_to_cols.setdefault(head, []).append(col)

                if not head_to_cols:
                    continue

                # build rows: compute mean-of-explicit-heads (mean_heads) then per-head rows (keep headNone aggregated row if present)
                ordered_heads = sorted(head_to_cols.keys())
                # compute per-head series first
                head_series = {}
                for h in ordered_heads:
                    cols_for_head = head_to_cols[h]
                    head_series[h] = sub[cols_for_head].astype(float).mean(axis=1).to_numpy()

                rows = []
                row_labels = []
                explicit_heads = [h for h in ordered_heads if h != -1]
                # add computed mean across explicit heads (mean_heads) if available
                if explicit_heads:
                    head_rows = [head_series[h] for h in explicit_heads]
                    try:
                        mean_row = np.nanmean(np.vstack(head_rows), axis=0)
                    except Exception:
                        mean_row = np.nanmean(np.array(head_rows, dtype=float), axis=0)
                    rows.append(mean_row)
                    row_labels.append(f"Unet {layer} - mean_heads")

                # append all head rows (including headNone if present)
                for h in ordered_heads:
                    rows.append(head_series[h])
                    label = f"Unet {layer} - head{h}" if h != -1 else f"Unet {layer} - headNone"
                    row_labels.append(label)

                heat = np.vstack(rows) if rows else np.empty((0, len(metric_list)))
                df = pd.DataFrame(heat, index=row_labels, columns=metric_list)

                # plot heatmap for this layer (all heads)
                nrows = max(1, df.shape[0])
                fig, ax = plt.subplots(1, 1, figsize=(6, max(2.5, 0.35 * nrows)))
                if df.empty:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center')
                    ax.axis('off')
                else:
                    sns.heatmap(df, annot=True, fmt='.3f', cmap=cmap, vmin=vmin, vmax=vmax, cbar=True, ax=ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                    ax.set_title(f'{t_type}')

                step_label = f'step{step}' if step != -1 else 'global'
                # sanitize type string for filename and include layer
                t_fname = re.sub(r'[^0-9A-Za-z_\-]', '_', t_type)
                out_file = os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_{t_fname}_unet{layer}.png")
                fig.suptitle(f'Correlation per-layer ({step_label}) - {t_type} - Unet {layer}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(out_file, dpi=150)
                plt.close(fig)

                # save csv for this layer/type
                df.to_csv(os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_{t_fname}_unet{layer}.csv"))

                # Build a single detailed CSV per layer that aggregates r/p/ci/mean across heads
                try:
                    cols_in_layer = [c for cols in head_to_cols.values() for c in cols]
                    long_rows = []
                    for metric in metric_list:
                        for h, cols in head_to_cols.items():
                            # compute mean r/p/ci across the attn columns that belong to this head
                            try:
                                r_vals = [corr_df.at[metric, c] for c in cols if c in corr_df.columns]
                                p_vals = [p_df.at[metric, c] for c in cols if c in p_df.columns]
                                ci_lo_vals = [ci_lower_df.at[metric, c] for c in cols if c in ci_lower_df.columns]
                                ci_hi_vals = [ci_upper_df.at[metric, c] for c in cols if c in ci_upper_df.columns]
                            except Exception:
                                r_vals = []
                                p_vals = []
                                ci_lo_vals = []
                                ci_hi_vals = []

                            def _nanmean_or_nan(arr):
                                try:
                                    a = np.array([float(x) for x in arr if not (isinstance(x, float) and np.isnan(x))])
                                    if a.size == 0:
                                        return np.nan
                                    return float(np.nanmean(a))
                                except Exception:
                                    return np.nan

                            r_mean = _nanmean_or_nan(r_vals)
                            p_mean = _nanmean_or_nan(p_vals)
                            ci_lo_mean = _nanmean_or_nan(ci_lo_vals)
                            ci_hi_mean = _nanmean_or_nan(ci_hi_vals)
                            attn_mean_val = _nanmean_or_nan([attn_means.get(c, np.nan) for c in cols])
                            n_samples = int(sum(int(numeric[c].notna().sum()) for c in cols)) if cols else 0

                            long_rows.append({
                                'type': t_type,
                                'unet_layer': layer,
                                'head': h,
                                'metric': metric,
                                'r_mean': r_mean,
                                'p_mean': p_mean,
                                'ci_lower_mean': ci_lo_mean,
                                'ci_upper_mean': ci_hi_mean,
                                'attn_mean': attn_mean_val,
                                'n_samples': n_samples,
                                'attn_cols': '|'.join(cols)
                            })

                    long_df = pd.DataFrame(long_rows)
                    long_df.to_csv(os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_{t_fname}_unet{layer}_details.csv"), index=False)
                except Exception:
                    # skip if something fails building the detailed CSV
                    pass

            # Also produce an aggregated heatmap across all layers for this type (rows=layers, cols=metrics)
            try:
                layer_rows = []
                layer_labels = []
                for layer in layers:
                    cols_in_layer = [m['col'] for m in meta_map.values() if m['type'] == t_type and m['layer'] == layer]
                    if not cols_in_layer:
                        continue
                    vals = sub[cols_in_layer].astype(float).mean(axis=1)
                    layer_rows.append(vals.to_numpy())
                    layer_labels.append(f"Unet {layer}")

                if layer_rows:
                    heat_all = np.vstack(layer_rows)
                    df_layers = pd.DataFrame(heat_all, index=layer_labels, columns=metric_list)

                    # plot aggregated layers heatmap
                    nrows_all = max(1, df_layers.shape[0])
                    fig2, ax2 = plt.subplots(1, 1, figsize=(6, max(2.5, 0.35 * nrows_all)))
                    sns.heatmap(df_layers, annot=True, fmt='.3f', cmap=cmap, vmin=vmin, vmax=vmax, cbar=True, ax=ax2)
                    ax2.set_ylabel('')
                    ax2.set_xlabel('')
                    ax2.set_title(f'{t_type} - layer means')
                    t_fname_all = re.sub(r'[^0-9A-Za-z_\-]', '_', t_type)
                    out_file_all = os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_{t_fname_all}_layers.png")
                    fig2.suptitle(f'Correlation per-layer ({step_label}) - {t_type} (layers)')
                    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig2.savefig(out_file_all, dpi=150)
                    plt.close(fig2)

                    # save csv for aggregated layers
                    df_layers.to_csv(os.path.join(out_dir, f"{os.path.basename(base)}_{step_label}_{t_fname_all}_layers.csv"))
            except Exception:
                # non-fatal: skip aggregated layers plotting if anything fails
                pass

    # Save overall corr, p-values, CI and means as diagnostics
    corr_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_metric_corr.csv")
    corr_df.to_csv(corr_csv)
    p_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_metric_pvalues.csv")
    p_df.to_csv(p_csv)
    ci_lo_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_metric_ci_lower.csv")
    ci_lower_df.to_csv(ci_lo_csv)
    ci_hi_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_metric_ci_upper.csv")
    ci_upper_df.to_csv(ci_hi_csv)
    means_csv = os.path.join(out_dir, f"{os.path.basename(base)}_attn_means.csv")
    pd.DataFrame.from_dict(attn_means, orient='index', columns=['mean']).to_csv(means_csv)

    # Also save a combined long-form CSV with r, p, ci, and attn metadata per (metric, attn_col)
    rows = []
    for m in good_metrics:
        for a in good_attn:
            meta = next((it for it in attn_meta if it['col'] == a), None)
            layer = meta['layer'] if meta is not None else -1
            step = meta['step'] if meta is not None else -1
            short = meta['short'] if meta is not None else a.split('/')[-1]
            rows.append({
                'metric': m,
                'attn_col': a,
                'attn_short': short,
                'unet_layer': layer,
                'step': step,
                'r': corr_df.at[m, a],
                'p_value': p_df.at[m, a],
                'ci_lower': ci_lower_df.at[m, a],
                'ci_upper': ci_upper_df.at[m, a],
                'attn_mean': attn_means.get(a, np.nan),
                'n_samples': int(numeric[a].notna().sum()),
            })
    combined = pd.DataFrame(rows)
    combined.to_csv(os.path.join(out_dir, f"{os.path.basename(base)}_attn_metric_combined.csv"), index=False)

    # finished: return used columns and numeric dataframe
    good_cols = good_metrics + good_attn
    return good_cols, numeric

    # For downstream summary keep behavior compatible: return used numeric columns and numeric df
    good_cols = good_metrics + good_attn
    return good_cols, numeric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='validation_metrics_correlation_wandb.png')
    parser.add_argument('--run_id', type=str, default=None, help='WandB run path, e.g. user/project/runid')
    parser.add_argument('--output_postfix', type=str, default=None, help='Optional postfix to append to output folder name')
    args = parser.parse_args()

    os.environ['WANDB_API_KEY'] = "5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe"
    # require run_id via CLI or environment
    if args.run_id is None:
        raise RuntimeError('Please provide --run_id (wandb run path)')

    # If run_id doesn't contain '/', assume it's just the run_id and construct full path
    run_path = args.run_id
    if '/' not in run_path:
        # Default to current user and project
        run_path = f"jsh0423_/nvs-vggt-distill/{run_path}"
        print(f"Converting run_id '{args.run_id}' to full path: {run_path}")
    
    print(f"Fetching run metrics for: {run_path}")
    history = fetch_run_metrics(run_path)

    # Diagnostic: print available columns and non-null counts
    print('Available columns in run history:')
    for c in history.columns:
        non_null = int(history[c].notna().sum())
        print(f"  {c}: non-null={non_null}")

    # Compute and save correlation plot; receive selected good_cols and numeric df
    good_cols, numeric_df = compute_and_save_correlation(history, args.out, args.output_postfix)

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


