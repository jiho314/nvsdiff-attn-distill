# val_wds_dataset_config:
#   url_paths: [ "/scratch/kaist-cvlab/dataset/real10k_validation"  ] 
#   dataset_length: 100
#   resampled: False
#   shardshuffle: False
#   min_view_range: 6
#   max_view_range: 10
#   process_kwargs:
#     get_square_extrinsic: true

# Set seed for reproducibility
import torch
import numpy as np
import random
from PIL import Image
import matplotlib.cm as cm
import matplotlib as mpl
import os
import cv2
import torchvision
from typing import List

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_attn_map_whole(attn_layer, background):
    """
    Generates an attention map visualization by overlaying a heatmap on a
    background image and marking the single point of highest attention.

    Args:
        attn_layer (torch.Tensor): 2D attention map [H_small, W_small], values in [0,1].
        background (torch.Tensor): Background image [3, H_large, W_large], values in [0,1] or [0,255].

    Returns:
        np.ndarray: Final visualized image [H_large, W_large, 3] uint8 (ready to save/show).
    """

    H_small, W_small = attn_layer.shape      # e.g. 32, 64
    _, H_large, W_large = background.shape   # e.g. 3, 512, 1024

    # 1. Resize attention map to match background size
    resize_transform = torchvision.transforms.Resize(
        (H_large, W_large),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    attn_layer_resized = resize_transform(attn_layer.unsqueeze(0).unsqueeze(0)).squeeze(0)  # [1, H, W] -> [H, W]

    # 2. Convert to numpy for colormap
    attn_np_resized = attn_layer_resized.cpu().detach().numpy()
    normalizer = mpl.colors.Normalize(vmin=attn_np_resized.min(), vmax=attn_np_resized.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    heatmap = (mapper.to_rgba(attn_np_resized)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    # 3. Convert background to numpy (HWC uint8)
    bg = background.permute(1, 2, 0).cpu().detach().numpy()
    if bg.max() <= 1.0:
        bg = bg * 255.0
    background_uint8 = bg.astype(np.uint8)

    # 4. Blend heatmap with background
    blended_map = cv2.addWeighted(background_uint8, 0.3, heatmap, 0.7, 0)

    # 5. Find max in ORIGINAL small attention map
    attn_small_np = attn_layer.cpu().detach().numpy()
    max_idx_y, max_idx_x = np.unravel_index(np.argmax(attn_small_np), attn_small_np.shape)

    # 6. Scale coordinates to large image
    max_x = int((max_idx_x + 0.5) * (W_large / W_small))
    max_y = int((max_idx_y + 0.5) * (H_large / H_small))

    # 7. Draw marker
    final_map = cv2.circle(blended_map, (max_x, max_y), 10, (255, 255, 255), -1)
    final_map = cv2.circle(final_map, (max_x, max_y), 6, (255, 0, 0), -1)

    return final_map

def stitch_side_by_side_whole(images: List[Image.Image], padding: int = 0, bg_color: tuple = (0, 0, 0)):
    """
    Stitches a list of PIL Images with varying widths horizontally.

    Args:
        images (List[PIL.Image]): A list of images to stitch.
        padding (int): Pixels of space between images.
        bg_color (tuple): Background color (R, G, B) for the canvas.

    Returns:
        PIL.Image: The stitched panoramic image.
    """
    # Fix: Compute canvas size dynamically from the actual images
    # Sum the widths of all images and add the padding space
    total_width = sum(img.width for img in images) + padding * (len(images) - 1)
    # Find the maximum height to ensure all images fit
    max_height = max(img.height for img in images)

    # Create the empty canvas with the correct dimensions
    canvas = Image.new('RGB', (total_width, max_height), color=bg_color)

    # Paste each image onto the canvas
    x_offset = 0
    for img in images:
        # Vertically center the image on the canvas
        y_offset = (max_height - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))
        # Update the horizontal offset for the next image
        x_offset += img.width + padding

    return canvas

def mark_point_on_img(tgt_img, x, y, radius=6):
    """
    tgt_img: torch.Tensor [3, 512, 512], values in [0,1] or [0,255]
    x, y: coordinates to mark (0–511)
    radius: circle radius
    returns: np.ndarray [512, 512, 3], uint8
    """
    # Convert to numpy HWC uint8, ensure contiguous
    img = tgt_img.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:   # handle [0,1] case
        img = img * 255.0
    img = img.astype(np.uint8)
    img = np.ascontiguousarray(img)

    # Draw white outer circle + blue inner circle
    img = cv2.circle(img, (int(x), int(y)), radius+4, (255, 255, 255), -1)  # white
    img = cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)        # blue (BGR)

    return img

# 2) Slice attnmap: 
def slice_attnmap(attnmap, query_idx, key_idx):
    B, Head, Q, K = attnmap.shape
    HW = Q // F
    attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
    attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
    attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
    return attnmap

# 1) Resize token: gt -> pred
def resize_tok(tok, target_size):
    B, Head, FHW, C = tok.shape
    HW = FHW // F
    H = W = int(math.sqrt(HW))
    tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', B=B, Head=Head, F=F, H=H, W=W, C=C)
    tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
    tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C',  B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
    return tok

val_wds_dataset_config = {
    'url_paths': [ "/mnt/data2/minseop/realestate_train_wds", ],
    'dataset_length': 100,
    'resampled': False,
    'shardshuffle': False,
    'min_view_range': 6,
    'max_view_range': 10,
    'process_kwargs': {
        'get_square_extrinsic': True
    }
}
from src.datasets.re10k_wds import build_re10k_wds
val_dataset = build_re10k_wds(
    num_viewpoints=4,
    **val_wds_dataset_config
) 
from torch.utils.data import DataLoader
loader = DataLoader(
    val_dataset,
    shuffle=False, # jiho TODO
    batch_size=1,
)

vggt_distill_config = {
    "cache_attn_layer_ids": [],
    "cache_costmap_types": ["track_head"]
}
from vggt.models.vggt import VGGT
# with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_distill_config).eval()
for p in vggt_model.parameters():
    p.requires_grad = False

device = 'cuda:0'
vggt_model = vggt_model.to(device)

import math
import einops
for batch in  loader:
    images = batch['image'].to(device)  # B V C H W
    vggt_pred = vggt_model(images) # jiho TODO make image [0,1]
    if True: 
        gt_query, gt_key = vggt_pred['attn_cache']['track_head']['query'], vggt_pred['attn_cache']['track_head']['key']  # B V H W C
        B, F, _, H, W = images.shape

        pred_query_size, pred_key_size = 32, 32
        gt_query, gt_key = resize_tok(gt_query, target_size=pred_query_size), resize_tok(gt_key, target_size=pred_key_size) # torch.Size([16, 1, 4096, 128])
        gt_attn_logit = gt_query @ gt_key.transpose(-1, -2)
        
        gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=[0], key_idx=[1,2,3])
        
        
        attn_maps = gt_attn_logit.squeeze()
        x_coord, y_coord = 256, 256 # over 512
        y_feat_cost = int((y_coord / 512) * pred_query_size)
        x_feat_cost = int((x_coord / 512) * pred_query_size)
        query_token_idx_cost = y_feat_cost * pred_query_size + x_feat_cost
        all_scores = torch.softmax(attn_maps[query_token_idx_cost]/8, dim=-1)
        
        score1 = all_scores[:1024].reshape(32,32)
        score2 = all_scores[1024:2048].reshape(32,32)
        score3 = all_scores[2048:].reshape(32,32)
        
        score = torch.cat([score1, score2, score3], dim=-1)

    images = images.squeeze()
    vis_list = []
    tgt_image = images[0]
    tgt_image = mark_point_on_img(tgt_image, x_coord, y_coord)
    vis_list.append(tgt_image)
    
    ref_image = images[1:]
    background = []
    for i in range(3):
        background_elem = ref_image[i].squeeze().permute(1,2,0).cpu().detach()
        background.append(background_elem)
    background = torch.stack(background, dim=1).reshape(512, -1, 3).numpy()
    background = np.clip(background * 255.0, 0, 255).astype(np.uint8)
    attn_heatmap_img = get_attn_map_whole(score, background, )
    
    vis_list.append(Image.fromarray(attn_heatmap_img.astype(np.uint8)))
    combined_img = stitch_side_by_side_whole(vis_list)
    combined_img.save(f"outputs/stitched.png")
    print(f"Saved visualization")
    # Save final stitched image
    # Save final image
    # final_img = Image.fromarray(canvas)
    # os.makedirs("outputs", exist_ok=True)
    # final_img.save("outputs/final_concat.png")
    # import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
    
    
    
    
    
    
    
    
    
    

    
    

