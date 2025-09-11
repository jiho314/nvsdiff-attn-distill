import os 
import webdataset as wds
from functools import partial
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import transforms

crop_transform = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize shortest side to 512
    transforms.CenterCrop(512),  # Center crop to 512x512
    transforms.ToTensor()    
])

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

RE10K_MINSEOP_URL = "/mnt/data1/minseop/realestate_wds" 
def transform_numpy(array):
    """
    Transform a NumPy array image by:
      1. Resizing so the shortest side is 512 pixels (bicubic interpolation).
      2. Center cropping to 512x512.
      3. Normalizing pixel values to [0, 1].
    
    Args:
        image (np.ndarray): Input image in shape (H, W, C) or (H, W).
    
    Returns:
        np.ndarray: Transformed image.
    """
    # Convert numpy array to PIL Image
    # Determine new size keeping the aspect ratio,
    # so that the shortest side becomes 512.
    array = torch.tensor(array).permute(0,3,1,2)
    _, _, height, width = array.shape
    
    if width < height:
        new_width = 512
        new_height = int(512 * height / width)
    else:
        new_height = 512
        new_width = int(512 * width / height)

    # Resize image using bicubic interpolation
    array_resized = F.interpolate(array, size=(new_height, new_width), mode="bilinear")
    # im_resized = im.resize((new_width, new_height), resample=Image.BICUBIC)

    # Compute coordinates for center crop of 512x512
    left = (new_width - 512) // 2
    top = (new_height - 512) // 2
    right = left + 512
    bottom = top + 512

    # Center crop the image
    array = array_resized[...,top:bottom,left:right]

    return array

def postprocess_combined(sample, num_viewpoints = 2, interpolate_only=False, everything_out = False):
    raise NotImplementedError("This function is not implemented yet.")
    
    transform = transforms.ToTensor()

    try:
        image_list = [obj for obj in sample.keys() if "image" in obj]

        idxs = random.sample(range(len(image_list)), num_viewpoints+1)

        if interpolate_only:
            idxs = sorted(idxs)

        points = sample["points.npy"]
        focals = sample["focals.npy"]
        poses = sample["poses.npy"]

        images = []

        for i in idxs:
            image_key = image_list[i]
            images.append(transform(sample[image_key]))

        output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs])

        return output

    except:
        return None

def re10k_postprocess_vggt(
    sample,
    *,
    num_viewpoints: int = 2,
    interpolate_only: bool = False,
    view_range: int = 8,
    deterministic: bool = False,
    uniform_sampling: bool = False,
    sampling_views: int = 10,
    target_policy: str = "random",
):
    # 1) collect frame keys
    image_names = [k for k in sample.keys() if "frame" in k]
    if len(image_names) == 0:
        return None
    try:
        image_names = sorted(image_names, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except Exception:
        image_names = sorted(image_names)

    # 2) Determine available views across modalities, then select indices (target + refs)
    import random
    rng = random.Random(0)
    if deterministic:
        seed_src = "|".join(image_names)
        seed_int = abs(hash(seed_src)) % (2**32)
        rng = random.Random(seed_int)

    # Ensure we only sample from frames that have points/intrinsics/extrinsics
    points = sample.get("pointmap.npy", None)
    intrinsics = sample.get("intrinsic.npy", None)
    extrinsics = sample.get("extrinsic.npy", None)
    if points is None or intrinsics is None or extrinsics is None:
        return None
    try:
        n_img = len(image_names)
        import numpy as _np
        pts_arr = _np.asarray(points)
        K_arr = _np.asarray(intrinsics)
        E_arr = _np.asarray(extrinsics)
        max_v = max(0, min(n_img, int(pts_arr.shape[0]) if pts_arr.ndim >= 1 else 0, int(K_arr.shape[0]) if K_arr.ndim >= 1 else 0, int(E_arr.shape[0]) if E_arr.ndim >= 1 else 0))
    except Exception:
        max_v = len(image_names)

    desired_len = int(num_viewpoints) + 1
    if max_v < desired_len:
        return None  # skip sample lacking enough views

    if not uniform_sampling:
        try:
            start = rng.sample(range(max(1, max_v - view_range - 1)), 1)[0]
            idxs = rng.sample(range(start, min(max_v, start + view_range)), desired_len)
        except Exception:
            idxs = rng.sample(range(max_v), desired_len)
    else:
        if sampling_views <= 1:
            idxs = [0]
        else:
            import numpy as np
            cand = np.linspace(0, max_v - 1, sampling_views, dtype=int).tolist()
            cand = sorted(set(int(x) for x in cand if 0 <= int(x) < max_v))
            if len(cand) >= desired_len:
                idxs = cand[: desired_len]
            else:
                remaining = [i for i in range(max_v) if i not in cand]
                need = desired_len - len(cand)
                extra = rng.sample(remaining, need) if len(remaining) >= need else remaining[:need]
                idxs = cand + extra

    # target policy
    if interpolate_only:
        idxs_sorted = sorted(idxs)
        if target_policy == "center":
            mid = max(1, min(len(idxs_sorted) - 2, len(idxs_sorted) // 2))
            tgt_idx = idxs_sorted[mid]
        elif target_policy == "first":
            tgt_idx = idxs_sorted[1] if len(idxs_sorted) > 2 else idxs_sorted[0]
        else:
            inner = idxs_sorted[1:-1] if len(idxs_sorted) > 2 else idxs_sorted
            tgt_idx = rng.sample(inner, 1).pop() if len(inner) else idxs_sorted[0]
    else:
        if target_policy == "center":
            idxs_sorted = sorted(idxs)
            tgt_idx = idxs_sorted[len(idxs_sorted) // 2]
        elif target_policy == "first":
            tgt_idx = sorted(idxs)[0]
        else:
            tgt_idx = rng.sample(idxs, 1).pop()
    ref_idxs = [i for i in idxs if i != tgt_idx]
    # Unify ordering to [ref0, ref1, ..., tgt]
    idxs = [tgt_idx] + ref_idxs[: num_viewpoints]

    # 3) load tensors
    images = [crop_transform(sample[image_names[i]]) for i in idxs]
    images = torch.stack(images)  # [V,3,512,512]

    points = torch.as_tensor(points)[idxs]
    # points: [V,H,W,3] or [V,3,H,W]
    if points.dim() == 4 and points.shape[-1] == 3:
        points = points.permute(0, 3, 1, 2)
    points = points.to(torch.float32)
    points = torch.nn.functional.interpolate(points, size=(512, 512), mode="bilinear", align_corners=False)

    intrinsics = torch.as_tensor(intrinsics)[idxs]
    extrinsics = torch.as_tensor(extrinsics)[idxs]

    return {"image": images, "points": points, "intrinsic": intrinsics, "extrinsic": extrinsics}

# def re10k_postprocess_vggt(sample, num_viewpoints = 2, interpolate_only=False, view_range=8, inference=False, uniform_sampling=False, sampling_views=10):
      
#     # if "+" in sample['__key__']:
#     #     dataset = 0 # co3d
#     #     view_range = int(1.0 * view_range)
#     # else:
#     #     dataset = 1 # realestate
        
#     try:
#         image_names = [obj for obj in sample.keys() if "frame" in obj]
#         image_names = sorted(image_names, key=lambda x : int(x.split("_")[-1][:-4]))
#         # print("image names[0]: ", image_names[0])
#         # print("image length : ", len(image_names))
        
#         if not inference:
#             if not uniform_sampling:
#                 try:
#                     range_start_idx = random.sample(range(len(image_names) - view_range - 1),1)[0]
#                     idxs = random.sample(range(range_start_idx, range_start_idx + view_range), num_viewpoints+1)
#                 except:
#                     idxs = random.sample(range(len(image_names)), num_viewpoints+1)
#                     print(f"Image less than {view_range}.")
#             else:
#                 idxs = np.linspace(0, len(image_names)-1, sampling_views, dtype=int).tolist()
#         else:
#             idxs = [0, 3] # JIHO TODO: why [0, 3]?
#             raise NotImplementedError("Inference mode is not implemented yet. why [0, 3]?")
                    
#         # tgt idx select
#         if interpolate_only:
#             idxs_sorted = sorted(idxs)
#             tgt_idx = random.sample(idxs_sorted[1:-1], 1).pop()
#         else:
#             tgt_idx = random.sample(idxs, 1).pop()
#         ref_idxs = [i for i in idxs if i != tgt_idx]
#         idxs = [tgt_idx] + ref_idxs # first view == target view


#         points = sample["pointmap.npy"]
#         intrinsics = sample["intrinsic.npy"]
#         extrinsic = sample["extrinsic.npy"]
#         # conf = sample["conf.npy"]

#         points, intrinsic, extrinsic = points[idxs], intrinsic[idxs], extrinsic[idxs]
#         image_names = [image_names[i] for i in idxs] 

#         images = [ crop_transform(sample[n]) for n in image_names ]
#         images = torch.stack(images)

#         points = transform_numpy(points)

#         bottom_row = np.tile(np.array([0, 0, 0, 1]), (extrinsics.shape[0], 1, 1))  # shape: (N, 1, 4)
#         extrinsics = np.concatenate([extrinsics, bottom_row], axis=1)  # shape: (N, 4, 4)
#         output = dict(images= images, points = points, intrinsic = intrinsic, extrinsic = extrinsics)
        
#         # if get_ray_map:
#         #     # TODO: check if extrinsic is w2c or c2w here
#             # w2cs = extrinsics # N 3 4
#             # bottom_row = np.tile(np.array([0, 0, 0, 1]), (w2cs.shape[0], 1, 1))  # shape: (N, 1, 4)
#             # w2cs = np.concatenate([w2cs, bottom_row], axis=1)  # shape: (N, 4, 4)
#         #     c2ws = np.linalg.inv(w2cs)
#         #     c2w_0 = c2ws[0]
#         #     raymaps = [c2w_to_ray_map(c2w_0, c2w, intrinsics[i], 512, 512) for i, c2w in enumerate(c2ws)]
#         #     raymaps = np.stack(raymaps, axis=0)  # shape: (N, 512, 512, 6)
#         #     output['raymaps'] = raymaps
#         return output
#     except:
#         return None

def build_re10k_minseop(
        batch_size=1,
        epoch=10000,
        dataset_length=2000000,
        resampled=True,
        shardshuffle=True,

        num_ref_viewpoints=2,
        interpolate_only=False,
        view_range=8,
        deterministic=False,
        uniform_sampling=False,
        sampling_views=10,
        target_policy="random",
        process_vggt=True,
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
    
    realestate_directory = RE10K_MINSEOP_URL
    urls = []
    for root, _, files in os.walk(realestate_directory):
        for file in files:
            if file.endswith(".tar"):
                urls.append(os.path.join(root, file))

    if process_vggt:
        postprocess_fn = partial(re10k_postprocess_vggt, 
                                 num_viewpoints=num_ref_viewpoints,
                                 interpolate_only = interpolate_only,
                                 view_range=view_range, 
                                 deterministic = deterministic,
                                 uniform_sampling = uniform_sampling,
                                 sampling_views = sampling_views,
                                 target_policy = target_policy
                                 )
    else:
        postprocess_fn = partial(postprocess_combined, num_viewpoints=num_ref_viewpoints, interpolate_only = interpolate_only)
    


    dataset = (wds.WebDataset(urls, 
                        resampled=resampled,
                        shardshuffle=shardshuffle, 
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker,
                        handler=wds.ignore_and_continue)
                .decode("pil")
                .map(postprocess_fn)
                .with_length(dataset_length)
                .with_epoch(batch_size * epoch)
        )
    return dataset


    