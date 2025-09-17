#!/usr/bin/env python3
"""
Post-process attention detail parquet files into a merged parquet and optionally upload a
wandb Artifact and preview table.

Usage:
  python scripts/attn_postprocess.py --attn_dir /path/to/attn_maps --upload-wandb
"""
import argparse
import os
import glob
import time
import pandas as pd


def find_parquets(attn_dir: str):
    patterns = [os.path.join(attn_dir, "*.parquet"), os.path.join(attn_dir, "*.csv")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # prefer parquet files
    parquets = [f for f in files if f.endswith('.parquet')]
    if parquets:
        return parquets
    return [f for f in files if f.endswith('.csv')]


def merge_files(files):
    dfs = []
    for f in files:
        try:
            if f.endswith('.parquet'):
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_dir', type=str, default='.', help='Directory containing attn detail parquet/csv files')
    # always upload to wandb (no flag)
    parser.add_argument('--top-k', type=int, default=10, help='Top-K rows by value for preview')
    parser.add_argument('--random-k', type=int, default=10, help='Random-K rows for preview')
    args = parser.parse_args()

    files = find_parquets(args.attn_dir)
    if not files:
        print(f"No parquet/csv files found in {args.attn_dir}")
        return

    print(f"Found {len(files)} files, merging...")
    df = merge_files(files)
    if df.empty:
        print("No data after merge.")
        return

    ts = int(time.time())
    out_fn = os.path.join(args.attn_dir, f"attn_merged_{ts}.parquet")
    try:
        df.to_parquet(out_fn, compression='snappy', index=False)
    except Exception:
        out_fn = out_fn.replace('.parquet', '.csv')
        df.to_csv(out_fn, index=False)

    print(f"Wrote merged file: {out_fn}")

    # build preview
    preview_df = pd.DataFrame()
    if 'value' in df.columns:
        try:
            topk = df.nlargest(args.top_k, 'value')
        except Exception:
            topk = df.head(args.top_k)
    else:
        topk = df.head(args.top_k)
    randk = df.sample(min(args.random_k, len(df)))
    preview_df = pd.concat([topk, randk]).drop_duplicates().reset_index(drop=True)

    # always upload to wandb
    try:
        import wandb
        run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'attn-postprocess'), reinit=True)
        art = wandb.Artifact(f"attn-merged-{ts}", type='attn-data')
        art.add_file(out_fn)
        run.log_artifact(art)
        try:
            table = wandb.Table(dataframe=preview_df)
            run.log({'attn/preview_table': table})
        except Exception:
            pass
        run.finish()
        print("Uploaded artifact and preview to wandb")
    except Exception as e:
        print(f"Warning: failed to upload to wandb: {e}")


if __name__ == '__main__':
    main()


