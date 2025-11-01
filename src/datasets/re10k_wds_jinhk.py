# import os 
# import webdataset as wds
# from functools import partial
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms as T
# from torchvision import transforms

# crop_transform = transforms.Compose([
#     transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize shortest side to 512
#     transforms.CenterCrop(512),  # Center crop to 512x512
#     transforms.ToTensor()    
# ])

# # def c2w_to_ray_map(c2w1, c2w2, intrinsics, h, w):
# #     c2w = np.linalg.inv(c2w1) @ c2w2
# #     i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
# #     grid = np.stack([i, j, np.ones_like(i)], axis=-1)
# #     ro = c2w[:3, 3]
# #     rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
# #     rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
# #     rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
# #     ro = np.broadcast_to(ro, (h, w, 3))
# #     ray_map = np.concatenate([ro, rd], axis=-1)
# #     return ray_map

# RE10K_MINSEOP_URL1 = "/mnt/data1/minseop/realestate_wds" 
# RE10K_MINSEOP_URL2 = "/mnt/data2/minseop/realestate_train_wds"
# def transform_numpy(array):
#     """
#     Transform a NumPy array image by:
#       1. Resizing so the shortest side is 512 pixels (bicubic interpolation).
#       2. Center cropping to 512x512.
#       3. Normalizing pixel values to [0, 1].
    
#     Args:
#         image (np.ndarray): Input image in shape (H, W, C) or (H, W).
    
#     Returns:
#         np.ndarray: Transformed image.
#     """
#     # Convert numpy array to PIL Image
#     # Determine new size keeping the aspect ratio,
#     # so that the shortest side becomes 512.
#     array = torch.tensor(array).permute(0,3,1,2)
#     _, _, height, width = array.shape
    
#     if width < height:
#         new_width = 512
#         new_height = int(512 * height / width)
#     else:
#         new_height = 512
#         new_width = int(512 * width / height)

#     # Resize image using bicubic interpolation
#     array_resized = F.interpolate(array, size=(new_height, new_width), mode="bilinear")
#     # im_resized = im.resize((new_width, new_height), resample=Image.BICUBIC)

#     # Compute coordinates for center crop of 512x512
#     left = (new_width - 512) // 2
#     top = (new_height - 512) // 2
#     right = left + 512
#     bottom = top + 512

#     # Center crop the image
#     array = array_resized[...,top:bottom,left:right]

#     return array


# def postprocess_vggt_mvgen(sample, num_viewpoints=3, min_view_range=5, max_view_range=15, 
#                            inference=False, inference_view_range=6,
#                            get_square_extrinsic=False, far_view_qual=False, **kwargs):
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
#         elif far_view_qual:
#             idxs = list(range(0, num_viewpoints-1)) + [num_frames - 1]
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
#             images.append(crop_transform(sample[image_key]))
        
#         # from torchvision.utils import save_image
#         # import pdb; pdb.set_trace()
#         # images = torch.stack([crop_transform(sample[image_list[i]]) for i in idxs], dim=-2).reshape(3, 512, -1)
#         # save_image(images, 'a.png')
        

#         intri = torch.tensor(intrinsic[idxs])
#         # new intri for 512 512 (TODO could be more simplified)
#         intri_h, intri_w = 2*intri[..., 0, 2], 2*intri[..., 1, 2]
#         fov_h = 2 * torch.atan((intri_h / 2) / intri[..., 1, 1])
#         fov_w = 2 * torch.atan((intri_w / 2) / intri[..., 0, 0])
#         H, W = 512, 512
#         fy = (H / 2.0) / torch.tan(fov_h / 2.0)
#         fx = (W / 2.0) / torch.tan(fov_w / 2.0)
#         intri_new = torch.zeros(intri.shape[:-2] + (3, 3))
#         intri_new[..., 0, 0] = fx
#         intri_new[..., 1, 1] = fy
#         intri_new[..., 0, 2] = W / 2
#         intri_new[..., 1, 2] = H / 2
#         intri_new[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
#         intri = intri_new
        
#         images = torch.stack(images)
#         pts = transform_numpy(points[idxs])
#         output = dict(image=images, point_map=pts, intrinsic=intri, extrinsic=extrinsic, idx=torch.tensor(idxs))
#         #   intrinsic_original = intrinsic[
#         # print(f"wds Inference: nothing occured , {sample['__key__']}, num_frames: {num_frames}, num_viewpoints: {num_viewpoints}, min_view_range: {min_view_range}, max_view_range: {max_view_range}")
#         return output
    
#     except:
#         if inference:
#             print(f"wds Inference: exception occured , {sample['__key__']}, num_frames: {num_frames}, num_viewpoints: {num_viewpoints}")
#             # raise ValueError("Inference error, please check")
#         return None

# def build_re10k_wds(
#     # epoch=10000,
#     url_paths = [],
#     dataset_length=20970,
#     resampled=True,
#     shardshuffle=True,

#     num_viewpoints=3,
#     min_view_range=5,
#     max_view_range=15,
#     inference=False,
#     inference_view_range= None,
#     process_kwargs = {},
#     **kwargs,
#     ):
#     '''
#         return: wds.WebDataset
#             First view is the target view
#             -  images: torch.Size([B, Views, 3, 512, 512])
#             -  points: torch.Size([B, Views, 3, 512, 512])
#             -  intrinsic: torch.Size([B, Views, 3, 3])
#             -  extrinsic: torch.Size([B, Views, 3, 4])
#     '''
#     urls = []
#     for path in url_paths:
#         urls = sorted(urls)
#         for root, _, files in os.walk(path):
#             for file in files:
#                 if file.endswith(".tar"):
#                     urls.append(os.path.join(root, file))


#     postprocess_fn = partial(postprocess_vggt_mvgen, num_viewpoints=num_viewpoints, min_view_range=min_view_range, max_view_range=max_view_range, inference= inference, inference_view_range = inference_view_range, **process_kwargs)
    
#     dataset = (wds.WebDataset(urls, 
#                         resampled=resampled,
#                         shardshuffle=shardshuffle, 
#                         nodesplitter=wds.split_by_node,
#                         workersplitter=wds.split_by_worker,
#                         handler=wds.ignore_and_continue)
#                 .decode("pil")
#                 .map(postprocess_fn)
#                 # .select(lambda x: x is not None)  # Filter out None values
#                 .with_length(dataset_length)
#                 .with_epoch(dataset_length)
#         )
    
#     return dataset

