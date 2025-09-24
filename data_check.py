from src.datasets.re10k_wds import build_re10k_wds
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
    url_paths = ["/mnt/data2/minseop/realestate_val_wds"], # "/mnt/data1/minseop/realestate_wds", # “/mnt/data2/minseop/realestate_val_wds”
    dataset_length = 280,
    # url_paths = ['/mnt/data1/minseop/realestate_wds'],
    num_viewpoints = 4,
    resampled=False,
    shardshuffle=False,
    # min_view_range=6,
    # max_view_range=6,
    inference=True,
    inference_view_range=10,
    process_kwargs={}
)

dl = DataLoader(
    ho,
    batch_size=4,
    num_workers=0,
)

# Create output directory
os.makedirs('batch_viz', exist_ok=True)

scenes=[]
for batch_idx, batch in enumerate(dl):
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