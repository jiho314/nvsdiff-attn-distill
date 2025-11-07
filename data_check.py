from src.datasets.re10k_wds import build_re10k_wds
from src.datasets import co3d_wds
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
import torch
from torchvision.utils import make_grid, save_image
import os

from accelerate.utils import ProjectConfiguration, set_seed

# Initialize accelerator
accelerator = Accelerator()
set_seed(0)
ho = build_re10k_wds(
    url_paths = ["/mnt/data2/minseop/realestate_val_wds",  ], # "/mnt/data2/minseop/realestate_train_wds"  "/mnt/data1/minseop/realestate_wds", # “/mnt/data2/minseop/realestate_val_wds” "/mnt/data2/minseop/realestate_val_wds"
    dataset_length = 10,
    # url_paths = ['/mnt/data1/minseop/realestate_wds'],
    num_viewpoints = 4,
    resampled=True,
    shardshuffle=True,
    min_view_range=6,
    max_view_range=6,
    # inference=True,
    # inference_view_range=10,
    process_kwargs={}
)
ho2 = co3d_wds(
    url_paths = ["/mnt/data1/minseop/co3d_wds1"],
    dataset_length = 40,
    num_viewpoints = 4,
    resampled=True,
    shardshuffle=True,
    min_view_range=6,
    max_view_range=6,
    
)
# HO = ho + ho2
import webdataset as wds
# ho = ho.shuffle(len(ho))
# ho2 = ho2.shuffle(len(ho2))
# HO = wds.RandomMix([ho, ho2], probs=[0.5,0.5])

# HO= ho.shuffle(len(ho))
# print("hotype : ", type(ho), " ho2 type: ", type(ho2), " HO type: ", type(HO))
# HO = (ho + ho2).shuffle(20)
# HO = HO.shuffle(len(HO))
dl = DataLoader(
    ho,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=True,
)
from itertools import cycle
import pdb ; pdb.set_trace()
# dl_iter = iter(dl)
# dl_cycle = cycle(dl)
from src.modules.position_encoding import depth_from_pointmap

for ep in range(3):
    print(f"Epoch {ep} -----------------")
    for i, batch in enumerate(dl):
        # batch_iter = next(dl_cycle)
        print(i)
        print("key : ", batch['__key__'])
        # print("pm size: ", batch['point_map'].shape)
import pdb ; pdb.set_trace()
        # print(i)
        # print(batch_iter['idx'])
        # print(batch['idx'])
        


# Create output directory
# os.makedirs('batch_viz', exist_ok=True)

scenes=[]
for batch_idx, batch in enumerate(dl):
    import pdb ; pdb.set_trace() 
    print(batch_idx)
    scenes += batch['__key__']
    # image = batch['image'].cuda()  # B V C H W
    # B, V, C, H, W = image.shape
    # # Reshape to (B*V, C, H, W) for make_grid
    # grid_images = image.view(B * V, C, H, W)
    
    # # Normalize to [0,1] if needed
    # if grid_images.max() > 1.0:
    #     grid_images = grid_images / 255.0
    
    # # Create grid with B rows, V columns
    # grid = make_grid(grid_images, nrow=V, padding=2, normalize=False, pad_value=1.0)
    
    # # Save the grid
    # save_image(grid.cpu(), f'batch_viz/batch_{batch_idx}_grid.png')
    
    # print(f"Saved batch {batch_idx} grid to batch_viz/batch_{batch_idx}_grid.png")
    # print(f"Batch shape: {image.shape}")
    
    # if batch_idx >= 34:  # Save first 3 batches
    #     break

print("Grid visualization saved!")
import pdb ; pdb.set_trace() 
# Prepare with accelerator
# # dl = accelerator.prepare(dl)

# print(f"Process {accelerator.process_index}/{accelerator.num_processes} starting...")
# rank = accelerator.device
# for i, batch in enumerate(dl):
#     # if accelerator.is_main_process:
#     print(f"Rank:{rank}, Batch {i}: keys = {list(batch.keys())}")
#     print("__key__:", batch['__key__'])

#     for key, value in batch.items():
#         if hasattr(value, 'shape'):
#             print(f"  {key}: shape = {value.shape}")
#         else:
#             print(f"  {key}: type = {type(value)}")

#     if i >= 5:  # Just check first few batches
#         break

# print(f"Process {accelerator.process_index} finished.")

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 data_check.py