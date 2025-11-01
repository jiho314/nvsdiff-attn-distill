import os 
import webdataset as wds
from functools import partial
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import transforms

from .co3d_wds import postprocess_co3d_mvgen
from .re10k_wds import postprocess_re10k_mvgen

def postprocess_mvgen_multi(sample, re10k_process_kwargs, co3d_process_kwargs, **kwargs):
    if 'co3d' in sample['__url__'].lower(): # Caution: depends on data url(tar file name) 
        return postprocess_co3d_mvgen(sample, **co3d_process_kwargs, **kwargs)
    else:
        return postprocess_re10k_mvgen(sample, **re10k_process_kwargs, **kwargs)

def build_multiple_wds(
    # epoch=10000,
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
    '''
        return: wds.WebDataset
            First view is the target view
            -  images: torch.Size([B, Views, 3, 512, 512])
            -  points: torch.Size([B, Views, 3, 512, 512])
            -  intrinsic: torch.Size([B, Views, 3, 3])
            -  extrinsic: torch.Size([B, Views, 3, 4])
    '''
    urls = []
    for path in url_paths:
        urls = sorted(urls)
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".tar"):
                    urls.append(os.path.join(root, file))


    postprocess_fn = partial(postprocess_mvgen_multi, num_viewpoints=num_viewpoints, re10k_process_kwargs = re10k_process_kwargs, co3d_process_kwargs = co3d_process_kwargs,
                             inference= False, inference_view_range = None, get_square_extrinsic = True, **kwargs )
    dataset = wds.WebDataset(urls, 
                        resampled=resampled,
                        shardshuffle=shardshuffle, 
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker,
                        handler=wds.ignore_and_continue)
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = ( dataset
                .decode("pil")
                .map(postprocess_fn)
                # .select(lambda x: x is not None)  # Filter out None values
                .with_length(dataset_length)
                .with_epoch(dataset_length)
        )
    if batch_size is not None:
        dataset = dataset.batched(batch_size, partial=False)
    
    return dataset
