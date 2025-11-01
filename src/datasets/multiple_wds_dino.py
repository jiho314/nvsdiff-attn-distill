import os, tarfile
import webdataset as wds
from functools import partial
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import transforms

from .co3d_wds import postprocess_co3d_mvgen_dino
from .re10k_wds import postprocess_re10k_mvgen_dino

def postprocess_mvgen_multi(sample, re10k_process_kwargs, co3d_process_kwargs, **kwargs):
    if 'co3d' in sample['__url__'].lower(): # Caution: depends on data url(tar file name) 
        return postprocess_co3d_mvgen_dino(sample, **co3d_process_kwargs, **kwargs)
    else:
        return postprocess_re10k_mvgen_dino(sample, **re10k_process_kwargs, **kwargs)

# ---------- DINO feature resolver ----------
def _key_from_member_name(member_name: str):
    """
    Extract sample key from a tar member name.

    Supports:
      key/file pattern:   "<key>/dino_tokens.npy"
      flat-dot pattern:   "<key>.dino_tokens.npy"
    """
    base = os.path.basename(member_name)
    if base == "dino_tokens.npy":
        # assume "<key>/dino_tokens.npy"
        key = member_name.split("/", 1)[0] if "/" in member_name else None
        return key, key is not None

    if ".dino_tokens" in base and base.endswith(".npy"):
        # "<key>.dino_tokens.npy"
        key = base.split(".dino_tokens", 1)[0]
        return key, bool(key)

    return None, False

def build_dino_index(dino_paths):
    """
    Walk dino_paths, open each *.tar, and index members that look like DINO tokens.
    Returns: dict { key: (tar_path, member_name) }
    Earlier dino_paths take precedence (first seen wins).
    """
    index = {}
    for rootdir in dino_paths or []:
        if not os.path.isdir(rootdir):
            continue

        for fname in sorted(os.listdir(rootdir)):
            if not fname.endswith(".tar"):
                continue
            tar_path = os.path.join(rootdir, fname)
            # Scan the tar quickly
            try:
                with tarfile.open(tar_path, "r") as tf:
                    for m in tf:
                        if not m.isreg():
                            continue
                        base = os.path.basename(m.name)
                        # only index likely DINO token files
                        if base != "dino_tokens.npy" and ".dino_tokens" not in base:
                            continue
                        key, ok = _key_from_member_name(m.name)
                        if ok and key not in index:
                            index[key] = (tar_path, m.name)
            except Exception as e:
                print(f"[WARN] Failed to scan {tar_path}: {e}")

    return index

def attach_dino_feature(sample, dino_index, dtype=np.float32, key_field="__key__"):
    """
    WebDataset .map hook: attaches 'dino_tokens' (numpy array) if available.
    """
    try:
        skey = sample.get(key_field, None)
        if skey is None:
            return sample
        dpath = dino_index.get(skey)
        if dpath is None:
            return sample
        arr = np.load(dpath, allow_pickle=False)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        sample["dino_tokens"] = arr
        return sample
    except Exception as e:
        # Keep training resilient: skip errors but continue the pipeline
        sample["__dino_error__"] = str(e)
        return sample

# ---------- Your loader with DINO injection ----------
def build_multiple_wds_dino(
    url_paths = [],
    dataset_length=20970,
    resampled=True,
    shardshuffle=True,
    shuffle_buffer_size=None,
    num_viewpoints=4,
    batch_size=None,
    re10k_process_kwargs = {},
    co3d_process_kwargs = {},
    **kwargs,
    ):
    """
    return: wds.WebDataset
        First view is the target view
        -  images: torch.Size([B, Views, 3, 512, 512])
        -  points: torch.Size([B, Views, 3, 512, 512])
        -  intrinsic: torch.Size([B, Views, 3, 3])
        -  extrinsic: torch.Size([B, Views, 3, 4])
        -  dino_feat: (optional) numpy array attached here
    """
    # 1) gather shard URLs (fix: sort AFTER collecting)
    urls = []
    for path in url_paths:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".tar"):
                    urls.append(os.path.join(root, file))
    urls = sorted(urls)

    # 2) build DINO indices from both re10k & co3d paths
    #    (we just merge them; keys are globally unique across shards)
    re10k_dino_paths = (re10k_process_kwargs or {}).get("dino_paths", []) or []
    co3d_dino_paths  = (co3d_process_kwargs  or {}).get("dino_paths", []) or []
    dino_index = build_dino_index(re10k_dino_paths + co3d_dino_paths)
    import pdb; pdb.set_trace()
    # 3) your existing postprocess (unchanged)
    postprocess_fn = partial(
        postprocess_mvgen_multi,
        num_viewpoints=num_viewpoints,
        re10k_process_kwargs=re10k_process_kwargs,
        co3d_process_kwargs=co3d_process_kwargs,
        inference=False,
        inference_view_range=None,
        get_square_extrinsic=True,
        **kwargs
    )

    # 4) assemble dataset
    dataset = wds.WebDataset(
                    urls,
                    resampled=resampled,
                    shardshuffle=shardshuffle,
                    nodesplitter=wds.split_by_node,
                    workersplitter=wds.split_by_worker,
                    handler=wds.ignore_and_continue,
              )

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    # Inject DINO before your postprocess (so your fn can consume sample["dino_feat"])
    dataset = (
        dataset
        .decode("pil")
        .map(partial(attach_dino_feature, dino_index=dino_index))  # <-- add this
        .map(postprocess_fn)
        .with_length(dataset_length)
        .with_epoch(dataset_length)
    )

    if batch_size is not None:
        dataset = dataset.batched(batch_size, partial=False)

    return dataset
