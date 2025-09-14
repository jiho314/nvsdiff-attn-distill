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
import matplotlib.pyplot as plt
import os
import cv2
import torchvision
from typing import List
from vggt.models.vggt import VGGT
import math
import einops
from src.datasets.re10k_wds import build_re10k_wds
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from torchvision.utils import save_image
from datetime import datetime
import json

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



import torch
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import torchvision.transforms as T

def get_attn_map_whole(attn_layer: torch.Tensor, background: np.ndarray) -> np.ndarray:
    """
    Overlay an attention heatmap on a background image and mark the highest attention point.

    Args:
        attn_layer (torch.Tensor): 2D attention map, shape (H_small, W_small).
        background (np.ndarray): Background image, shape (H_large, W_large, 3).

    Returns:
        np.ndarray: Visualization with heatmap and marked point.
    """
    H_small, W_small = attn_layer.shape
    H_large, W_large = background.shape[:2]

    # 1. Resize attn_layer to background size
    resize_transform = T.Resize(
        (H_large, W_large),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias=True
    )
    attn_layer_resized = resize_transform(attn_layer.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
    attn_layer_resized = attn_layer_resized.squeeze().cpu().detach().numpy()     # [H, W]

    # 2. Create heatmap
    normalizer = mpl.colors.Normalize(vmin=attn_layer_resized.min(), vmax=attn_layer_resized.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap="viridis")
    heatmap = (mapper.to_rgba(attn_layer_resized)[:, :, :3] * 255).astype(np.uint8)

    # 3. Blend with background
    background_uint8 = background.astype(np.uint8).copy()
    blended_map = cv2.addWeighted(background_uint8, 0.3, heatmap, 0.7, 0)

    # 4. Find the max attention location in the small map
    attn_layer_small_np = attn_layer.cpu().detach().numpy()
    max_idx_y, max_idx_x = np.unravel_index(np.argmax(attn_layer_small_np), attn_layer_small_np.shape)

    # 5. Scale to large dimensions
    max_x = int((max_idx_x + 0.5) * (W_large / W_small))
    max_y = int((max_idx_y + 0.5) * (H_large / H_small))

    # 6. Draw a circle at the max point
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
    img = cv2.circle(img, (int(x), int(y)), radius, (0, 0, 255), -1)        # blue (BGR)

    return img

# 2) Slice attnmap: 
def slice_attnmap(attnmap, query_idx, key_idx):
    B, Head, Q, K = attnmap.shape
    F = 4 # warn: hardcoding
    HW = Q // F
    attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
    attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
    attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
    return attnmap

# 1) Resize token: gt -> pred
def resize_tok(tok, target_size):
    B, Head, FHW, C = tok.shape
    F = 4 # warn: hardcoding
    HW = FHW // F
    H = W = int(math.sqrt(HW))
    tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', B=B, Head=Head, F=F, H=H, W=W, C=C)
    tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
    tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C',  B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
    return tok

def overlay_grid_and_save(img_tensor: torch.Tensor, spacing=64, out_path="grid_overlay.png"):
    """
    Overlays a grid every `spacing` pixels, but only labels the
    x-axis at the top edge and the y-axis at the left edge,
    with labels in black. The image is shown without flipping.
    
    img_tensor: [H, W] or [C, H, W]
    """
    # 1. Convert to H×W or H×W×C numpy
    if img_tensor.ndim == 3:
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
    elif img_tensor.ndim == 2:
        img = img_tensor.cpu().numpy()
    else:
        raise ValueError(f"Unsupported shape {img_tensor.shape}")

    H, W = img.shape[:2]
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    # 2. Plot
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=None if img.ndim == 3 else "gray", origin="upper")
    # — no ax.invert_xaxis(), so image is not flipped

    # 3. Draw grid lines
    for y in ys:
        ax.axhline(y, color="white", linewidth=0.8)
    for x in xs:
        ax.axvline(x, color="white", linewidth=0.8)

    # 4. Set ticks at grid lines
    ax.set_xticks(xs)
    ax.set_yticks(ys)

    # 5. Label ticks in black
    ax.set_xticklabels([str(x) for x in xs], color="black", fontsize=8)
    ax.set_yticklabels([str(y) for y in ys], color="black", fontsize=8)

    # 6. Show x labels on top only, y labels on left only
    ax.tick_params(
        axis='x', which='both',
        labelbottom=False,
        labeltop=True,
        bottom=False, top=False
    )
    ax.tick_params(
        axis='y', which='both',
        labelleft=True,
        labelright=False,
        left=False, right=False
    )

    # 7. Remove frame lines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def get_costmap(gt_query, gt_key, x_coord, y_coord, mode):
    pred_query_size, pred_key_size = 32, 32
    # Convert pixel coords -> feature map index
    y_feat_cost = int((y_coord / 512) * pred_query_size)
    x_feat_cost = int((x_coord / 512) * pred_query_size)
    query_token_idx_cost = y_feat_cost * pred_query_size + x_feat_cost

    if mode == "tracking":
        query, key = resize_tok(gt_query, target_size=pred_query_size), resize_tok(gt_key, target_size=pred_key_size),
        gt_attn_logit = query @ key.transpose(-1, -2)
        gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=[0], key_idx=[1,2,3])
        attn_maps = gt_attn_logit.squeeze()
        all_scores = torch.softmax(attn_maps[query_token_idx_cost] / 8, dim=-1)
    elif mode == "attention":
        query, key = resize_tok(gt_query, target_size=pred_query_size), resize_tok(gt_key, target_size=pred_key_size),
        gt_attn_logit = query @ key.transpose(-1, -2)
        gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=[0], key_idx=[1,2,3])
        attn_maps = gt_attn_logit.mean(dim=1) # average over head
        attn_maps = attn_maps.squeeze()
        all_scores = torch.softmax(attn_maps[query_token_idx_cost] / 8, dim=-1)
    elif mode == "pointmap":
        def distance_softmax(query_points, ref_points, temperature=1.0, cross_only=True):
            """
            query_points: (B, 3, H, W)
            ref_points:   (B, V, 3, H, W)
            returns:      (B, HW, VHW) probability maps
            """
            # Compute pairwise differences: (B, N, N, 3)
            B, _, H, W = query_points.shape
            V = ref_points.shape[1]
            query_points = query_points.reshape(B, 3, -1).permute(0, 2, 1) # (B, HW, 3)
            ref_points = ref_points.reshape(B, V, 3, -1).permute(0, 1, 3, 2).reshape(B, -1, 3) # (B, VHW, 3)
            if not cross_only:
                ref_points = torch.cat((query_points, ref_points), dim=1) # (B, (V+1)HW, 3)
            diff = query_points.unsqueeze(2) - ref_points.unsqueeze(1)
            # Euclidean distance
            dist = torch.norm(diff, dim=-1) # (B, HW, VHW)
            # Convert to probabilities: smaller distance → higher prob
            logits = - dist / temperature
            probs = torch.softmax(logits, dim=-1)
            return probs.squeeze()
        gt_query = gt_query.reshape(-1, 3, 512, 512)
        gt_key = gt_key.reshape(-1, 3, 512, 512)
        gt_query = torch.nn.functional.interpolate(gt_query, size=(pred_query_size, pred_key_size), mode='bilinear')
        gt_key = torch.nn.functional.interpolate(gt_key, size=(pred_query_size, pred_key_size), mode='bilinear')
        gt_query = gt_query.reshape(1, 3, pred_query_size, pred_key_size)
        gt_key = gt_key.reshape(1, 3, 3, pred_query_size, pred_key_size)
        attn_maps = distance_softmax(gt_query, gt_key, temperature=0.1)
        all_scores = attn_maps[query_token_idx_cost]
    score1 = all_scores[:1024].reshape(pred_query_size,pred_key_size)
    score2 = all_scores[1024:2048].reshape(pred_query_size,pred_key_size)
    score3 = all_scores[2048:].reshape(pred_query_size,pred_key_size)
    score = torch.cat([score1, score2, score3], dim=-1)
    return score
    
def save_image_total(images, x_coord, y_coord, score):
    # Visualization
    images = images.squeeze()
    vis_list = []

    # Target image with marked point
    # warn
    # tgt_image = images[0]
    # tgt_image = mark_point_on_img(tgt_image, x_coord, y_coord)
    # vis_list.append(Image.fromarray(tgt_image))

    # Background from reference images
    ref_image = images[1:]
    background = []
    for i in range(3):
        background_elem = ref_image[i].squeeze().permute(1,2,0).cpu().detach()
        background.append(background_elem)
    background = torch.stack(background, dim=1).reshape(512, -1, 3).numpy()
    background = np.clip(background * 255.0, 0, 255).astype(np.uint8)

    attn_heatmap_img = get_attn_map_whole(score, background)
    vis_list.append(Image.fromarray(attn_heatmap_img.astype(np.uint8)))

    combined_img = stitch_side_by_side_whole(vis_list)
    return combined_img
        

def main(cfg):
    val_wds_dataset_config = {
        'url_paths': [ "/mnt/data2/minseop/realestate_train_wds", ],
        'dataset_length': 200,
        'resampled': False,
        'shardshuffle': False,
        'min_view_range': 6,
        'max_view_range': 10,
        'process_kwargs': {
            'get_square_extrinsic': True
        }
    }

    val_dataset = build_re10k_wds(
        num_viewpoints=4,
        **val_wds_dataset_config
    ) 

    loader = DataLoader(
        val_dataset,
        shuffle=False, # jiho TODO
        batch_size=1,
    )

    vggt_distill_config = {
        "cache_attn_layer_ids": cfg.attn_idx if "attention" in cfg.mode else [],
        "cache_costmap_types": cfg.mode
    }

    # with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_distill_config).eval()
    for p in vggt_model.parameters():
        p.requires_grad = False

    device = 'cuda:0'
    vggt_model = vggt_model.to(device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for i, batch in enumerate(loader):
        outdir = os.path.join("outputs", timestamp, f"sample{i}")
        os.makedirs(outdir, exist_ok=True)
        
        images = batch['image'].to(device)  # B V C H W
        vggt_pred = vggt_model(images)

        # --- get target image ---
        tgt_image = images.squeeze()[0]      # [C, H, W]

        # --- visualize with grid before input ---
        
        if cfg.view_select:
            overlay_grid_and_save(tgt_image, spacing=64, out_path="VIS_OVERLAY.png")
            save_image(images.squeeze()[1:], "VIS_REFERENCE.png")
            while True:
                answer = input("Do you want to visualize? (yes/no): ").strip().lower()
                if answer == "yes":
                    # run visualization code
                    counter_out = False
                    break
                elif answer == "no":
                    # skip this batch
                    counter_out = True
                    break  # goes back to your for-loop
                else:
                    print("Invalid input. Please type 'yes' or 'no'.")
                    
            if counter_out:
                continue
        
        if cfg.coord_select:
            if cfg.view_select is False:
                overlay_grid_and_save(tgt_image, spacing=64, out_path="VIS_OVERLAY.png")
                save_image(images.squeeze()[1:], "VIS_REFERENCE.png")
            # Ask user for coordinates
            while True: 
                x_coord = int(input("Enter x coordinate (0–512): "))
                y_coord = int(input("Enter y coordinate (0–512): "))
                if 0 < x_coord < 512 and 0 < y_coord < 512: 
                    break
                else: 
                    print("Invalid input. Please type valid values. ")
        else:
            # Random selection
            x_coord = random.randint(0, 511)
            y_coord = random.randint(0, 511)
        coords = {"x": x_coord, "y": y_coord}
        json_path = os.path.join(outdir, "coords.json")
        with open(json_path, "w") as f:
            json.dump(coords, f, indent=4)      
    
        if "track_head" in cfg.mode:
            gt_query, gt_key = vggt_pred['attn_cache']['track_head']['query'], vggt_pred['attn_cache']['track_head']['key'] # torch.Size([1, 1, 2668324, 128])
            score = get_costmap(gt_query, gt_key, x_coord, y_coord, "tracking") # 32, 3*32
            combined_img = save_image_total(images, x_coord, y_coord, score)
            combined_img.save(f"{outdir}/tracking.png")
        if "attention" in cfg.mode:
            for layer in cfg.attn_idx:
                gt_query, gt_key = vggt_pred['attn_cache'][f'{layer}']['query'], vggt_pred['attn_cache'][f'{layer}']['key'] # torch.Size([1, 16, 5476, 64])
                score = get_costmap(gt_query, gt_key, x_coord, y_coord, "attention") # 32, 3*32
                combined_img = save_image_total(images, x_coord, y_coord, score)
                combined_img.save(f"{outdir}/attn{layer}.png")
        if "pointmap" in cfg.mode:
            # pointmap = batch['point_map'].permute(0,1,3,4,2) # [1, 4, 512, 512, 3]
            # gt_query, gt_key = pointmap[:, :1, ...], pointmap[:, 1:, ...] 
            # gt_query = gt_query.reshape(1, 1, -1, 3) # torch.Size([1, 1, 512, 3])
            # gt_key = gt_key.reshape(1, 1, -1, 3) # torch.Size([1, 1, 786432, 3])
            pointmap = batch['point_map']
            gt_query = pointmap[:, :1, ...]
            gt_key = pointmap[:, 1:, ...]
            score = get_costmap(gt_query, gt_key, x_coord, y_coord, "pointmap")
            combined_img = save_image_total(images, x_coord, y_coord, score)
            combined_img.save(f"{outdir}/pointmap.png")
        
        save_image(tgt_image, f"{outdir}/TARGET.png")
        tgt_image_marked = mark_point_on_img(tgt_image, x_coord, y_coord)
        target_marked = torch.from_numpy(tgt_image_marked).permute(2, 0, 1).float() / 255.0
        save_image(target_marked, f"{outdir}/TARGET_MARKED.png")

        ref_imgs = images.squeeze()[1:]
        ref_concat = torch.cat([img for img in ref_imgs], dim=-1)  # shape [C, H, W*3] # save as one single image save_image(ref_concat, f"{outdir}/REFERENCE.png")
        save_image(ref_concat, f"{outdir}/REFERENCE.png")
        print(f"Saved visualization to outputs.png")
        
        if i == 200: break
        
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/data1/jiho/vggt-nvs/MVGenMaster/costmap_visualization.yaml")
    args = parser.parse_args()
    set_seed(42)
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    main(config)
    
    
    
    
    
    
    
    

    
    

