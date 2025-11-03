import os 
import webdataset as wds
from functools import partial
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import transforms

# crop_transform = transforms.Compose([
#     transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize shortest side to 512
#     transforms.CenterCrop(512),  # Center crop to 512x512
#     transforms.ToTensor()    
# ])

# def c2w_to_ray_map(c2w1, c2w2, intrinsics, h, w):
#     c2w = np.linalg.inv(c2w1) @ c2w2
#     i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
#     grid = np.stack([i, j, np.ones_like(i)], axis=-1)
#     ro = c2w[:3, 3]
#     rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
#     rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
#     rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
#     ro = np.broadcast_to(ro, (h, w, 3))
#     ray_map = np.concatenate([ro, rd], axis=-1)
#     return ray_map

RE10K_MINSEOP_URL1 = "/mnt/data1/minseop/co3d_wds1" 
RE10K_MINSEOP_URL2 = "/mnt/data2/minseop/co3d_wds2"

def postprocess_co3d_mvgen(sample, num_viewpoints=3, min_view_range=5, max_view_range=15, 
                           inference=False, inference_view_range=6, inference_ref_idx=[],
                           get_square_extrinsic=False, **kwargs):
    '''
        First, sample (view_range) from [min_view_range, max_view_range]
        Then, sample (num_viewpoints) frames among (view_range) frames
    '''
    assert num_viewpoints > 2, "num_viewpoints should be greater than 2"
    try:
        image_list = [obj for obj in sample.keys() if "frame" in obj]
        image_list = sorted(image_list, key=lambda x : int(x.split("_")[-1][:-4]))
        num_frames = len(image_list)
        
        if not inference:
            min_view_range = max(min_view_range, num_viewpoints)
            max_view_range = min(max_view_range, num_frames - 1)
            if max_view_range < min_view_range:
                # print(f"wds: max_view_range < min_view_range, {max_view_range}, {min_view_range}, {num_frames}")
                return None
            view_range = random.randint(min_view_range, max_view_range)
            if num_frames <= view_range:
                # print(f"wds: num_frames <= view_range, {num_frames}, {view_range}")
                return None
            start = random.randint(0, num_frames - view_range - 1)
            end = start + view_range
            sampled_frames = random.sample(range(start + 1, end), num_viewpoints-2)
            idxs = [start, end] + sampled_frames
            idxs = sorted(idxs)
        else:
            # 11/02 jiho: update custom ref idx
            # 1) uniform sampling from view range
            max_view_point = min(num_frames, inference_view_range)
            data_idxs = np.linspace(0, max_view_point - 1, num_viewpoints, dtype=int).tolist()
            # 2) custom ref idx
            ids = range(num_viewpoints)
            tgt_ids = [i for i in ids if i not in inference_ref_idx]
            ids = inference_ref_idx + tgt_ids
            data_idxs = [ data_idxs[i] for i in ids ]
            idxs = data_idxs



            
        points = sample["pointmap.npy"]
        intrinsic = sample["intrinsic.npy"]
        extrinsic = sample["extrinsic.npy"]
        extrinsic = torch.tensor(extrinsic)[idxs]
        if get_square_extrinsic:
            if extrinsic.shape[-2] == 3:
                new_extrinsic = torch.zeros((num_viewpoints, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
                new_extrinsic[:, :3, :4] = extrinsic
                new_extrinsic[:, 3, 3] = 1.0
                extrinsic = new_extrinsic

        images = []
        for i in idxs:
            image_key = image_list[i]
            images.append(sample[image_key])
        
        intri = torch.tensor(intrinsic[idxs])
        
        images = []
        for i in idxs:
            image_key = image_list[i]
            img = sample[image_key]
            if not torch.is_tensor(img):
                img = transforms.ToTensor()(img)
            images.append(img)
        images = torch.stack(images)  # (V, 3, H, W), already normalized to [0, 1] by ToTensor()
        pts = torch.tensor(points[idxs]).permute(0,3,1,2)  # (V, 3, H, W)
        # tag = ['co3d'] * len(idxs)
        tag = 'co3d'
        # print("pts shape: ", pts.shape, " images shape: ", images.shape, " intri shape: ", intri.shape, " extrinsic shape: ", extrinsic.shape, " idxs: ", idxs)
        # print(f"wds: {sample['__key__']}, num_frames: {num_frames}, num_viewpoints: {num_viewpoints}, min_view_range: {min_view_range}, max_view_range: {max_view_range}")
        output = dict(image=images, point_map=pts, intrinsic=intri, extrinsic=extrinsic, idx=torch.tensor(idxs), tag= tag)
        #   intrinsic_original = intrinsic[
        # print(f"wds Inference: nothing occured , {sample['__key__']}, num_frames: {num_frames}, num_viewpoints: {num_viewpoints}, min_view_range: {min_view_range}, max_view_range: {max_view_range}")
        return output
    
    except Exception as e:
        print("Exception type:", type(e))
        print("Exception message:", e)
        import traceback
        traceback.print_exc()
        if inference:
            print(f"wds Inference: exception occured , {sample.get('__key__', 'unknown')}")
        else:
            print(f"wds Training: exception occured , {sample.get('__key__', 'unknown')}")
        return None
    
    
# def postprocess_co3d_mvgen_dino(sample, num_viewpoints=3, min_view_range=5, max_view_range=15, 
#                            inference=False, inference_view_range=6,
#                            get_square_extrinsic=False, **kwargs):
#     '''
#         First, sample (view_range) from [min_view_range, max_view_range]
#         Then, sample (num_viewpoints) frames among (view_range) frames
#     '''
#     assert num_viewpoints > 2, "num_viewpoints should be greater than 2"
#     try:
#         image_list = [obj for obj in sample.keys() if "frame" in obj]
#         image_list = sorted(image_list, key=lambda x : int(x.split("_")[-1][:-4]))
#         num_frames = len(image_list)
        
#         if not inference:
#             min_view_range = max(min_view_range, num_viewpoints)
#             max_view_range = min(max_view_range, num_frames - 1)
#             if max_view_range < min_view_range:
#                 # print(f"wds: max_view_range < min_view_range, {max_view_range}, {min_view_range}, {num_frames}")
#                 return None
#             view_range = random.randint(min_view_range, max_view_range)
#             if num_frames <= view_range:
#                 # print(f"wds: num_frames <= view_range, {num_frames}, {view_range}")
#                 return None
#             start = random.randint(0, num_frames - view_range - 1)
#             end = start + view_range
#             sampled_frames = random.sample(range(start + 1, end), num_viewpoints-2)
#             idxs = [start, end] + sampled_frames
#             idxs = sorted(idxs)
#         else:
#             max_view_point = min(num_frames, inference_view_range)
#             idxs = np.linspace(0, max_view_point - 1, num_viewpoints, dtype=int).tolist()

            
#         points = sample["pointmap.npy"]
#         intrinsic = sample["intrinsic.npy"]
#         extrinsic = sample["extrinsic.npy"]
#         extrinsic = torch.tensor(extrinsic)[idxs]
#         if get_square_extrinsic:
#             if extrinsic.shape[-2] == 3:
#                 new_extrinsic = torch.zeros((num_viewpoints, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
#                 new_extrinsic[:, :3, :4] = extrinsic
#                 new_extrinsic[:, 3, 3] = 1.0
#                 extrinsic = new_extrinsic

#         images = []
#         for i in idxs:
#             image_key = image_list[i]
#             images.append(sample[image_key])
        
#         intri = torch.tensor(intrinsic[idxs])
        
#         images = []
#         for i in idxs:
#             image_key = image_list[i]
#             img = sample[image_key]
#             if not torch.is_tensor(img):
#                 img = transforms.ToTensor()(img)
#             images.append(img)
#         images = torch.stack(images)  # (V, 3, H, W), already normalized to [0, 1] by ToTensor()
#         pts = torch.tensor(points[idxs]).permute(0,3,1,2)  # (V, 3, H, W)
#         # print("pts shape: ", pts.shape, " images shape: ", images.shape, " intri shape: ", intri.shape, " extrinsic shape: ", extrinsic.shape, " idxs: ", idxs)
#         # print(f"wds: {sample['__key__']}, num_frames: {num_frames}, num_viewpoints: {num_viewpoints}, min_view_range: {min_view_range}, max_view_range: {max_view_range}")
#         feat = sample.get("dino_tokens", None)
        
#         if feat is not None and not torch.is_tensor(feat):
#             feat = torch.from_numpy(feat) # (V, N, D)
#             V,N,D = feat.size
#             H = int(torch.sqrt(N))
#             W = H
#             feat = feat.reshape(V, H, W, D)
#             feat = feat[idxs]
            
#         output = dict(image=images, point_map=pts, intrinsic=intri, extrinsic=extrinsic, idx=torch.tensor(idxs), dino_feat = feat)
        
#         #   intrinsic_original = intrinsic[
#         # print(f"wds Inference: nothing occured , {sample['__key__']}, num_frames: {num_frames}, num_viewpoints: {num_viewpoints}, min_view_range: {min_view_range}, max_view_range: {max_view_range}")
#         return output
    
#     except Exception as e:
#         print("Exception type:", type(e))
#         print("Exception message:", e)
#         import traceback
#         traceback.print_exc()
#         if inference:
#             print(f"wds Inference: exception occured , {sample.get('__key__', 'unknown')}")
#         else:
#             print(f"wds Training: exception occured , {sample.get('__key__', 'unknown')}")
#         return None

def build_co3d_wds(
    # epoch=10000,
    url_paths = [],
    dataset_length=20970,
    resampled=True,
    shardshuffle=True,

    num_viewpoints=3,
    min_view_range=5,
    max_view_range=15,
    inference=False,
    inference_view_range= None,
    inference_ref_idx = [],
    process_kwargs = {},
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

    postprocess_fn = partial(postprocess_co3d_mvgen, num_viewpoints=num_viewpoints, min_view_range=min_view_range, max_view_range=max_view_range, 
                                                        inference= inference, inference_view_range = inference_view_range, inference_ref_idx=inference_ref_idx, 
                                                        **process_kwargs)
    
    dataset = (wds.WebDataset(urls, 
                        resampled=resampled,
                        shardshuffle=shardshuffle, 
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker,
                        handler=wds.ignore_and_continue)
                .decode("pil")
                .map(postprocess_fn)
                # .select(lambda x: x is not None)  # Filter out None values
                .with_length(dataset_length)
                .with_epoch(dataset_length)
        )
    
    return dataset

