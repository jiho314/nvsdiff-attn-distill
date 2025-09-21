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


def collect_and_organize_runs_summary(project_name="jsh0423_/nvs-vggt-distill", run_name_prefixes=["EXP_val_run"]):
    """
    WandB에서 지정된 prefix(distill, naive)로 시작하는 모든 run을 가져와서 
    summary 데이터를 run별로 정리하여 반환하는 함수
    
    Args:
        project_name (str): WandB 프로젝트 경로 (user/project 형식)
        run_name_prefixes (list): 검색할 run 이름의 prefix 리스트
        
    Returns:
        pandas.DataFrame: run별로 정리된 summary 데이터 (정렬됨)
    """
    api = wandb.Api()
    
    # 프로젝트의 모든 run 가져오기
    runs = api.runs(project_name)
    
    # prefix에 해당하는 run들 필터링
    filtered_runs = []
    for run in runs:
        run_name = run.name
        if any(run_name.startswith(prefix) for prefix in run_name_prefixes):
            filtered_runs.append(run)
    
    print(f"총 {len(filtered_runs)}개의 run을 찾았습니다 (prefix: {run_name_prefixes})")
    
    # 각 run의 summary 데이터 수집
    summary_data = []
    for run in filtered_runs:
        run_info = {
            'run_name': run.name,
            'run_id': run.id,
            'run_path': f"{project_name}/{run.id}",
            'state': run.state,
            'created_at': run.created_at if hasattr(run, 'created_at') else None,
        }
        
        # updated_at이 있는 경우에만 추가 (일부 WandB API 버전에서는 없을 수 있음)
        if hasattr(run, 'updated_at'):
            run_info['updated_at'] = run.updated_at
        elif hasattr(run, 'heartbeat_at'):
            run_info['updated_at'] = run.heartbeat_at
        else:
            run_info['updated_at'] = None
        
        # summary 데이터 추가
        if hasattr(run, 'summary') and run.summary:
            for key, value in run.summary.items():
                # summary/ prefix가 있는 키들만 수집
                if key.startswith('summary/'):
                    run_info[key] = value
                # summary/ prefix가 없는 경우도 포함 (backward compatibility)
                elif not key.startswith('_'):  # wandb internal keys 제외
                    run_info[f"summary/{key}"] = value
        
        summary_data.append(run_info)
    
    # DataFrame으로 변환
    df = pd.DataFrame(summary_data)
    
    if df.empty:
        print("수집된 데이터가 없습니다.")
        return df
    
    # run_name으로 정렬
    df = df.sort_values('run_name').reset_index(drop=True)
    
    # 결과 출력
    print(f"\n수집된 run들:")
    for idx, row in df.iterrows():
        print(f"  {row['run_name']} (ID: {row['run_id']}, State: {row['state']})")
    
    # summary 컬럼들 출력
    summary_cols = [col for col in df.columns if col.startswith('summary/')]
    if summary_cols:
        print(f"\n수집된 summary 메트릭들:")
        for col in sorted(summary_cols):
            non_null_count = df[col].notna().sum()
            print(f"  {col}: {non_null_count}/{len(df)} runs에서 사용 가능")
    
    return df


def plot_summary_graphs(csv_file_path="runs_summary_organized.csv"):
    """
    CSV 파일에서 summary 데이터를 읽어와서 run 숫자 기준으로 정렬하고 
    주요 메트릭들에 대한 그래프를 생성하는 함수
    
    Args:
        csv_file_path (str): CSV 파일 경로
    """
    import re
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("CSV 파일이 비어있습니다.")
        return
    
    # run_name에서 숫자 추출하는 함수
    def extract_run_number(run_name):
        # distill_val_run_숫자 또는 naive_val_run_숫자 패턴에서 숫자 추출
        match = re.search(r'(distill|naive)_val_run_(\d+)', str(run_name))
        if match:
            return int(match.group(2))
        return 0
    
    # run 타입 추출하는 함수 (distill/naive)
    def extract_run_type(run_name):
        if str(run_name).startswith('distill'):
            return 'distill'
        elif str(run_name).startswith('naive'):
            return 'naive'
        return 'unknown'
    
    # 숫자와 타입 추출
    df['run_number'] = df['run_name'].apply(extract_run_number)
    df['run_type'] = df['run_name'].apply(extract_run_type)
    
    # run_number로 정렬
    df = df.sort_values('run_number').reset_index(drop=True)
    
    # 완료된 run들만 필터링 (finished 상태)
    finished_df = df[df['state'] == 'finished'].copy()
    
    if finished_df.empty:
        print("완료된 run이 없습니다.")
        return
    
    print(f"그래프 생성을 위한 완료된 run 개수: {len(finished_df)}")
    
    # 출력 디렉토리 생성
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"summary_plots_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 주요 메트릭 컬럼들 찾기
    metric_cols = {}
    
    # PSNR, SSIM, LPIPS 찾기
    for col in finished_df.columns:
        if 'psnr' in col.lower() and 'summary' in col:
            metric_cols['PSNR'] = col
        elif 'ssim' in col.lower() and 'summary' in col:
            metric_cols['SSIM'] = col  
        elif 'lpips' in col.lower() and 'summary' in col:
            metric_cols['LPIPS'] = col
    
    # attention loss 관련 메트릭들 찾기
    attn_cols = [col for col in finished_df.columns if 'attention_loss' in col and 'summary' in col and 'mean' in col]
    
    print(f"발견된 주요 메트릭: {list(metric_cols.keys())}")
    print(f"발견된 attention loss 메트릭 개수: {len(attn_cols)}")
    
    # 그래프 생성
    fig_width = 12
    fig_height = 8
    
    # 1. 주요 메트릭들 (PSNR, SSIM, LPIPS) 그래프
    if metric_cols:
        fig, axes = plt.subplots(len(metric_cols), 1, figsize=(fig_width, fig_height * len(metric_cols)))
        if len(metric_cols) == 1:
            axes = [axes]
        
        for idx, (metric_name, col_name) in enumerate(metric_cols.items()):
            ax = axes[idx]
            
            # distill과 naive 분리해서 플롯
            distill_data = finished_df[finished_df['run_type'] == 'distill']
            naive_data = finished_df[finished_df['run_type'] == 'naive']
            
            if not distill_data.empty:
                ax.plot(distill_data['run_number'], distill_data[col_name], 
                       'o-', label='Distill', linewidth=2, markersize=6)
            
            if not naive_data.empty:
                ax.plot(naive_data['run_number'], naive_data[col_name], 
                       's-', label='Naive', linewidth=2, markersize=6)
            
            ax.set_xlabel('Run Number')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Run Number')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # y축 범위 자동 조정
            if not finished_df[col_name].isna().all():
                y_min = finished_df[col_name].min()
                y_max = finished_df[col_name].max()
                y_range = y_max - y_min
                ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'main_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"주요 메트릭 그래프 저장: {os.path.join(output_dir, 'main_metrics.png')}")
    
    # 2. Attention Loss 메트릭들 그래프 (상위 몇 개만)
    if attn_cols:
        # 상위 6개 attention loss 메트릭만 선택
        selected_attn_cols = attn_cols[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(fig_width * 1.5, fig_height))
        axes = axes.flatten()
        
        for idx, col_name in enumerate(selected_attn_cols):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # distill과 naive 분리해서 플롯
            distill_data = finished_df[finished_df['run_type'] == 'distill']
            naive_data = finished_df[finished_df['run_type'] == 'naive']
            
            if not distill_data.empty and not distill_data[col_name].isna().all():
                ax.plot(distill_data['run_number'], distill_data[col_name], 
                       'o-', label='Distill', linewidth=2, markersize=6)
            
            if not naive_data.empty and not naive_data[col_name].isna().all():
                ax.plot(naive_data['run_number'], naive_data[col_name], 
                       's-', label='Naive', linewidth=2, markersize=6)
            
            # 컬럼명에서 간단한 제목 추출
            title = col_name.replace('summary/', '').replace('attention_loss/', '').replace('_mean', '')
            ax.set_xlabel('Run Number')
            ax.set_ylabel('Attention Loss')
            ax.set_title(title, fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # y축 범위 자동 조정
            if not finished_df[col_name].isna().all():
                y_min = finished_df[col_name].min()
                y_max = finished_df[col_name].max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
        
        # 사용하지 않는 subplot 숨기기
        for idx in range(len(selected_attn_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_loss_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention Loss 메트릭 그래프 저장: {os.path.join(output_dir, 'attention_loss_metrics.png')}")
    
    # 3. 요약 통계 저장
    summary_stats = []
    for run_type in ['distill', 'naive']:
        type_data = finished_df[finished_df['run_type'] == run_type]
        if not type_data.empty:
            for metric_name, col_name in metric_cols.items():
                if not type_data[col_name].isna().all():
                    stats = {
                        'run_type': run_type,
                        'metric': metric_name,
                        'count': len(type_data),
                        'mean': type_data[col_name].mean(),
                        'std': type_data[col_name].std(),
                        'min': type_data[col_name].min(),
                        'max': type_data[col_name].max()
                    }
                    summary_stats.append(stats)
    
    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        stats_file = os.path.join(output_dir, 'summary_statistics.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"요약 통계 저장: {stats_file}")
    
    print(f"\n모든 그래프와 통계가 '{output_dir}' 폴더에 저장되었습니다.")
    
    return output_dir


def analyze_timestep_layer_ce(csv_file_path="runs_summary_organized.csv"):
    """
    TIMESTEP별로 각 레이어의 Cross Entropy(CE) STD와 MEAN 변화를 시계열로 분석하는 함수
    
    Args:
        csv_file_path (str): CSV 파일 경로
        
    Returns:
        str: 출력 디렉토리 경로
    """
    import re
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)
    
    if df.empty:
        print("CSV 파일이 비어있습니다.")
        return
    
    # run_name에서 숫자와 타입 추출 (timestep으로 사용)
    def extract_run_info(run_name):
        match = re.search(r'(distill|naive)_val_run_(\d+)', str(run_name))
        if match:
            return match.group(1), int(match.group(2))
        return 'unknown', 0
    
    df['run_type'] = df['run_name'].apply(lambda x: extract_run_info(x)[0])
    df['timestep'] = df['run_name'].apply(lambda x: extract_run_info(x)[1])  # timestep으로 사용
    
    # 완료된 run들만 필터링
    finished_df = df[df['state'] == 'finished'].copy()
    
    if finished_df.empty:
        print("완료된 run이 없습니다.")
        return
    
    print(f"TIMESTEP별 레이어 CE 분석을 위한 완료된 run 개수: {len(finished_df)}")
    
    # 출력 디렉토리 생성
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"timestep_layer_ce_analysis_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cross Entropy 관련 컬럼들 찾기 (cross_entropy가 포함되고 head가 없는 컬럼만)
    ce_cols = [col for col in finished_df.columns 
               if 'cross_entropy' in col and 'summary' in col and 'head' not in col]
    
    print(f"발견된 Cross Entropy 컬럼 (head 제외): {len(ce_cols)}")
    
    # 레이어 정보 추출 함수
    def extract_layer_info(col_name):
        # unet 뒤의 숫자 추출
        layer_match = re.search(r'unet(\d+)', col_name)
        layer_num = int(layer_match.group(1)) if layer_match else None
        
        return layer_num
    
    # 시계열 데이터 정리
    timeseries_data = []
    
    for col in ce_cols:
        layer_num = extract_layer_info(col)
        if layer_num is None:
            continue
        
        # 각 run의 데이터 수집
        for _, row in finished_df.iterrows():
            if pd.notna(row[col]):  # 유효한 값이 있는 경우만
                timeseries_data.append({
                    'timestep': row['timestep'],
                    'run_type': row['run_type'],
                    'layer': layer_num,
                    'ce_value': row[col],
                    'run_name': row['run_name'],
                    'column_name': col
                })
    
    if not timeseries_data:
        print("Cross Entropy 데이터를 찾을 수 없습니다.")
        return output_dir
    
    # DataFrame으로 변환
    ts_df = pd.DataFrame(timeseries_data)
    
    # timestep으로 정렬
    ts_df = ts_df.sort_values(['timestep', 'layer']).reset_index(drop=True)
    
    # CSV로 저장
    ts_file = os.path.join(output_dir, 'timestep_layer_ce_data.csv')
    ts_df.to_csv(ts_file, index=False)
    print(f"TIMESTEP별 레이어 CE 데이터 저장: {ts_file}")
    
    # 레이어별로 시계열 그래프 생성
    unique_layers = sorted(ts_df['layer'].unique())
    unique_timesteps = sorted(ts_df['timestep'].unique())
    
    print(f"발견된 레이어: {unique_layers}")
    print(f"발견된 timesteps: {unique_timesteps}")
    
    # 각 레이어별로 그래프 생성 (head 제외)
    for layer in unique_layers:
        layer_data = ts_df[ts_df['layer'] == layer]
        
        if layer_data.empty:
            continue
        
        # 단순한 레이어별 시계열 분석 (head 없음)
        # 1) combined (distill vs naive) + difference (as before)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # combined plot (distill vs naive)
        ax = axes[0]
        for run_type in ['distill', 'naive']:
            type_data = layer_data[layer_data['run_type'] == run_type]
            if not type_data.empty:
                marker = 'o' if run_type == 'distill' else 's'
                ax.plot(type_data['timestep'], type_data['ce_value'], 
                       marker=marker, label=f'{run_type.capitalize()}', 
                       linewidth=2, markersize=6)

        ax.set_xlabel('Timestep')
        ax.set_ylabel('CE Value')
        ax.set_title(f'Layer {layer} - CE Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # difference plot (distill - naive)
        ax = axes[1]
        distill_data = layer_data[layer_data['run_type'] == 'distill'].set_index('timestep')['ce_value']
        naive_data = layer_data[layer_data['run_type'] == 'naive'].set_index('timestep')['ce_value']

        # 공통 timestep에서만 차이 계산
        common_timesteps = set(distill_data.index) & set(naive_data.index)
        if common_timesteps:
            common_timesteps = sorted(common_timesteps)
            diff_values = [distill_data[ts] - naive_data[ts] for ts in common_timesteps]

            ax.plot(common_timesteps, diff_values, 'ro-', linewidth=2, markersize=6)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('CE Difference (Distill - Naive)')
            ax.set_title(f'Layer {layer} - Distill vs Naive Difference')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'layer_{layer}_ce_timeseries.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer} CE 시계열 그래프 저장: {plot_file}")

        # 2) separate plots: distill-only and naive-only per layer (요청사항)
        # distill-only
        distill_df = layer_data[layer_data['run_type'] == 'distill']
        if not distill_df.empty:
            fig_d, ax_d = plt.subplots(1, 1, figsize=(8, 4))
            ax_d.plot(distill_df['timestep'], distill_df['ce_value'], 'o-', color='tab:blue', linewidth=2, markersize=6)
            ax_d.set_xlabel('Timestep')
            ax_d.set_ylabel('CE Value')
            ax_d.set_title(f'Layer {layer} - Distill CE Over Time')
            ax_d.grid(True, alpha=0.3)
            out_d = os.path.join(output_dir, f'layer_{layer}_ce_distill.png')
            fig_d.tight_layout()
            fig_d.savefig(out_d, dpi=300, bbox_inches='tight')
            plt.close(fig_d)
            print(f"Layer {layer} Distill CE 그래프 저장: {out_d}")

        # naive-only
        naive_df = layer_data[layer_data['run_type'] == 'naive']
        if not naive_df.empty:
            fig_n, ax_n = plt.subplots(1, 1, figsize=(8, 4))
            ax_n.plot(naive_df['timestep'], naive_df['ce_value'], 's-', color='tab:orange', linewidth=2, markersize=6)
            ax_n.set_xlabel('Timestep')
            ax_n.set_ylabel('CE Value')
            ax_n.set_title(f'Layer {layer} - Naive CE Over Time')
            ax_n.grid(True, alpha=0.3)
            out_n = os.path.join(output_dir, f'layer_{layer}_ce_naive.png')
            fig_n.tight_layout()
            fig_n.savefig(out_n, dpi=300, bbox_inches='tight')
            plt.close(fig_n)
            print(f"Layer {layer} Naive CE 그래프 저장: {out_n}")
    
    # 전체 요약 그래프 생성 (모든 레이어 한번에)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 모든 레이어의 평균 CE 시계열
    ax = axes[0, 0]
    for run_type in ['distill', 'naive']:
        type_data = ts_df[ts_df['run_type'] == run_type]
        if not type_data.empty:
            # timestep별 전체 평균
            overall_means = type_data.groupby('timestep')['ce_value'].mean()
            marker = 'o' if run_type == 'distill' else 's'
            ax.plot(overall_means.index, overall_means.values, 
                   marker=marker, label=f'{run_type.capitalize()}', 
                   linewidth=2, markersize=6)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Overall CE Mean')
    ax.set_title('Overall CE Mean Over Time (All Layers)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 레이어별 CE 평균 히트맵
    ax = axes[0, 1]
    pivot_data = ts_df.groupby(['timestep', 'layer'])['ce_value'].mean().unstack(fill_value=np.nan)
    im = ax.imshow(pivot_data.values, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Timestep')
    ax.set_title('CE Mean Heatmap (Timestep vs Layer)')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index)
    plt.colorbar(im, ax=ax)
    
    # 3. 레이어별 최종 성능 비교
    ax = axes[1, 0]
    final_timestep = max(unique_timesteps)
    final_data = ts_df[ts_df['timestep'] == final_timestep]
    
    for run_type in ['distill', 'naive']:
        type_final = final_data[final_data['run_type'] == run_type]
        if not type_final.empty:
            layer_means = type_final.groupby('layer')['ce_value'].mean()
            marker = 'o' if run_type == 'distill' else 's'
            ax.plot(layer_means.index, layer_means.values, 
                   marker=marker, label=f'{run_type.capitalize()}', 
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Final CE Value')
    ax.set_title(f'Final CE by Layer (Timestep {final_timestep})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 전체 차이 시계열
    ax = axes[1, 1]
    for layer in unique_layers:
        layer_data = ts_df[ts_df['layer'] == layer]
        distill_data = layer_data[layer_data['run_type'] == 'distill'].groupby('timestep')['ce_value'].mean()
        naive_data = layer_data[layer_data['run_type'] == 'naive'].groupby('timestep')['ce_value'].mean()
        
        common_timesteps = set(distill_data.index) & set(naive_data.index)
        if common_timesteps:
            common_timesteps = sorted(common_timesteps)
            diff_values = [distill_data[ts] - naive_data[ts] for ts in common_timesteps]
            ax.plot(common_timesteps, diff_values, 'o-', label=f'Layer {layer}', alpha=0.7)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('CE Difference (Distill - Naive)')
    ax.set_title('CE Difference by Layer Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot = os.path.join(output_dir, 'overall_ce_timeseries_summary.png')
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"전체 요약 그래프 저장: {summary_plot}")

    # --- Combined figure: two subplots (distill / naive), y-axis = timestep, lines per layer ---
    try:
        import colorsys
        layers = unique_layers
        if layers:
            # palette per layer (consistent across subplots)
            base_palette = sns.color_palette('hls', n_colors=len(layers))

            figC, axesC = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
            for idx, run_type in enumerate(['distill', 'naive']):
                ax = axesC[idx]
                for i, layer in enumerate(layers):
                    ld = ts_df[(ts_df['layer'] == layer) & (ts_df['run_type'] == run_type)].sort_values('timestep')
                    if ld.empty:
                        continue
                    base_rgb = base_palette[i % len(base_palette)]
                    # adjust saturation: distill more saturated, naive less
                    sat_factor = 1.0 if run_type == 'distill' else 0.6
                    r, g, b = base_rgb
                    h, l, s = colorsys.rgb_to_hls(r, g, b)
                    new_r, new_g, new_b = colorsys.hls_to_rgb(h, l, max(0.0, min(1.0, s * sat_factor)))
                    color = (new_r, new_g, new_b)
                    # plot timestep (x) vs CE (y)
                    ax.plot(ld['timestep'], ld['ce_value'], marker='o', label=f'Layer {layer}', color=color, linewidth=2, markersize=5)

                ax.set_xlabel('Timestep')
                ax.set_title(run_type.capitalize())
                ax.grid(True, alpha=0.3)

            axesC[0].set_ylabel('CE Value')
            # place legend outside the subplots to the right
            handles, labels = axesC[0].get_legend_handles_labels()
            if handles:
                figC.legend(handles, labels, loc='upper center', ncol=min(8, len(labels)), bbox_to_anchor=(0.5, 0.99))

            figC.suptitle('Per-layer CE over Timesteps (Distill vs Naive)')
            figC.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_combined = os.path.join(output_dir, 'combined_distill_naive_layers.png')
            figC.savefig(out_combined, dpi=300, bbox_inches='tight')
            plt.close(figC)
            print(f"Combined distill/naive layers plot 저장: {out_combined}")
    except Exception:
        # non-fatal: continue if combined plotting fails
        pass
    
    # 통계 요약 생성
    summary_stats = []
    for layer in unique_layers:
        layer_data = ts_df[ts_df['layer'] == layer]
        
        for run_type in ['distill', 'naive']:
            type_data = layer_data[layer_data['run_type'] == run_type]
            if not type_data.empty:
                stats = {
                    'layer': layer,
                    'run_type': run_type,
                    'timesteps_count': len(type_data['timestep'].unique()),
                    'ce_mean': type_data['ce_value'].mean(),
                    'ce_std': type_data['ce_value'].std(),
                    'ce_min': type_data['ce_value'].min(),
                    'ce_max': type_data['ce_value'].max(),
                    'final_timestep': type_data['timestep'].max(),
                    'final_ce': type_data[type_data['timestep'] == type_data['timestep'].max()]['ce_value'].mean()
                }
                summary_stats.append(stats)
    
    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        stats_file = os.path.join(output_dir, 'timestep_ce_summary_stats.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"통계 요약 저장: {stats_file}")
    
    print(f"\n모든 TIMESTEP별 레이어 CE 분석 결과가 '{output_dir}' 폴더에 저장되었습니다.")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='validation_metrics_correlation_wandb.png')
    parser.add_argument('--run_id', type=str, default=None, help='WandB run path, e.g. user/project/runid')
    parser.add_argument('--ckpts', type=str, default=None, help='Comma-separated checkpoint numbers to find corresponding runs')
    parser.add_argument('--output_postfix', type=str, default=None, help='Optional postfix to append to output folder name')
    parser.add_argument('--collect_summary', action='store_true', help='Collect and organize summary data from distill/naive runs')
    parser.add_argument('--plot_graphs', action='store_true', help='Generate plots from runs_summary_organized.csv')
    parser.add_argument('--analyze_layers', action='store_true', help='Analyze timestep-wise layer CE mean/std changes')
    args = parser.parse_args()

    os.environ['WANDB_API_KEY'] = "5e4d6a67a9287ff9ad9b05ccc97582fcb1d48dfe"
    
    # 새로운 summary 수집 기능 처리
    if args.collect_summary:
        print("Collecting and organizing summary data from distill/naive runs...")
        summary_df = collect_and_organize_runs_summary()
        
        # CSV로 저장
        output_file = "runs_summary_organized.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary 데이터가 {output_file}에 저장되었습니다.")
        
        # 간단한 통계 출력
        if not summary_df.empty:
            print(f"\n요약:")
            print(f"  총 run 개수: {len(summary_df)}")
            
            # run type별 개수
            distill_count = len([name for name in summary_df['run_name'] if name.startswith('distill')])
            naive_count = len([name for name in summary_df['run_name'] if name.startswith('naive')])
            print(f"  distill runs: {distill_count}")
            print(f"  naive runs: {naive_count}")
            
            # 상태별 개수
            state_counts = summary_df['state'].value_counts()
            print(f"  상태별 개수: {dict(state_counts)}")
        
        return
    
    # 새로운 그래프 생성 기능 처리
    if args.plot_graphs:
        print("Generating plots from summary data...")
        csv_path = "validation_metrics_correlation_wandb_per_metric/runs_summary_organized.csv"
        if not os.path.exists(csv_path):
            csv_path = "runs_summary_organized.csv"
        
        if os.path.exists(csv_path):
            output_dir = plot_summary_graphs(csv_path)
            print(f"그래프 생성 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        else:
            print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            print("먼저 --collect_summary를 실행해서 데이터를 수집하세요.")
        return
    
    # TIMESTEP별 레이어 CE 분석 기능 처리
    if args.analyze_layers:
        print("Analyzing timestep-wise layer CE changes...")
        csv_path = "validation_metrics_correlation_wandb_per_metric/runs_summary_organized.csv"
        if not os.path.exists(csv_path):
            csv_path = "runs_summary_organized.csv"
        
        if os.path.exists(csv_path):
            output_dir = analyze_timestep_layer_ce(csv_path)
            print(f"TIMESTEP별 레이어 CE 분석 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        else:
            print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            print("먼저 --collect_summary를 실행해서 데이터를 수집하세요.")
        return
    
    # Simple run selection logic inside this script:
    # - If --ckpts is provided, search WandB project runs for names containing each checkpoint string and process matches
    # - Else if --run_id provided, process that run
    # - Else auto-detect latest finished run
    api = wandb.Api()
    project = 'jsh0423_/nvs-vggt-distill'

    run_paths_to_process = []

    if args.ckpts:
        ckpts = [s.strip() for s in args.ckpts.split(',') if s.strip()]
        try:
            runs = api.runs(project, per_page=200)
            for ck in ckpts:
                candidates = [r for r in runs if (str(ck) in (getattr(r, 'name', '') or '') or str(ck) in (getattr(r, 'displayName', '') or ''))]
                if not candidates:
                    print(f"No wandb runs found containing checkpoint '{ck}' in name")
                    continue
                picked = sorted(candidates, key=lambda r: getattr(r, 'created_at', None) or 0, reverse=True)[0]
                # Handle case where path is a list
                path = picked.path
                if isinstance(path, list):
                    path = '/'.join(path)
                run_paths_to_process.append(path)
                print(f"Found run for ckpt {ck}: {path}")
        except Exception as e:
            print(f"[WARN] failed to search runs for ckpts: {e}")

    if args.run_id:
        run_paths_to_process.append(args.run_id if '/' in args.run_id else f"{project}/{args.run_id}")

    if not run_paths_to_process:
        try:
            runs = api.runs(project, per_page=200)
            finished = [r for r in runs if getattr(r, 'state', '') == 'finished']
            if finished:
                picked = sorted(finished, key=lambda r: getattr(r, 'created_at', None) or 0, reverse=True)[0]
                # Handle case where path is a list
                path = picked.path
                if isinstance(path, list):
                    path = '/'.join(path)
                run_paths_to_process.append(path)
                print(f"Auto-detected latest finished run: {path}")
        except Exception as e:
            print(f"[WARN] failed to auto-detect run: {e}")

    if not run_paths_to_process:
        raise RuntimeError('No runs found to process. Provide --run_id or --ckpts')

    for run_path in run_paths_to_process:
        print(f"Fetching run metrics for: {run_path}")
        try:
            history = fetch_run_metrics(run_path)
        except Exception as e:
            print(f"Failed to fetch run {run_path}: {e}")
            continue

        # Diagnostic: print available columns and non-null counts
        print('Available columns in run history:')
        for c in history.columns:
            non_null = int(history[c].notna().sum())
            print(f"  {c}: non-null={non_null}")

        try:
            postfix = args.output_postfix or run_path.split('/')[-1]
            compute_and_save_correlation(history, args.out, postfix)
        except Exception as e:
            print(f"Error processing run {run_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

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


