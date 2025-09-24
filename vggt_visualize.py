from train import *


from src.distill_utils.attn_processor_cache import set_attn_cache, unset_attn_cache, pop_cached_attn, clear_attn_cache
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
import torchvision.transforms as T
from src.distill_utils.attn_visualize import mark_point_on_img, save_image_total, overlay_grid_and_save, save_image_jinhk


# dataloader_workers: 16
# dataset:
#   dataset_path: /scratch2/ljeadec31/re10k_pixelsplat/re10k_lvsm/train/full_list.txt
#   use_view_idx_file: false
#   view_idx_file_path:  
#   image_size : 512
#   num_ref_views: 2
#   num_tgt_views: 1
#   min_frame_dist: 25 # 25
#   max_frame_dist: 100 # 100 
#   shuffle_prob: 0.5 # extrapolate
  
# eval_dataset:
#   dataset_path: /scratch2/ljeadec31/re10k_pixelsplat/re10k_lvsm/test/full_list.txt
#   use_view_idx_file: true
#   view_idx_file_path: dataset/evaluation_index_re10k_0.json 
#   image_size : 512
#   num_ref_views: 2
#   num_tgt_views: 1

def concat_vertical(pil_images):
    """Concatenate a list of PIL Images vertically (row direction)."""
    if len(pil_images) == 1:
        return pil_images[0]
    widths, heights = zip(*(im.size for im in pil_images))
    w_max = max(widths)
    h_sum = sum(heights)
    canvas = Image.new("RGB", (w_max, h_sum), (0, 0, 0))
    y = 0
    for im in pil_images:
        # center each image horizontally
        x = (w_max - im.size[0]) // 2
        canvas.paste(im, (x, y))
        y += im.size[1]
    return canvas

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def main(nframe, cond_num, inference_view_range, config, rank = 0):

    seed_everything(0)
    device= f"cuda"

    # 2) data
    from src.datasets.re10k_wds import build_re10k_wds
    val_dataset = build_re10k_wds(
        url_paths = [ "/mnt/data2/minseop/realestate_val_wds" ] ,
        dataset_length = 100000,
        resampled = False,
        shardshuffle=False,
        num_viewpoints= nframe,
        inference=True,
        inference_view_range = inference_view_range
        
    ) 
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=None,
        batch_size=1,
        num_workers=0,
        drop_last=True,
    )
    
    from src.distill_utils.attn_logit_head import cycle_consistency_checker
    def get_consistency_mask(logit):
        # assert config.distill_config.distill_query == "target", "consistency check only support distill_query to target"
        B, F1HW, F2HW = logit.shape  
        assert F1HW == F2HW, "first process full(square) consistency mask"
        # assert Head == 1, "costmap should have only one head, while consistency checking?"
        F = 1
        HW = F1HW // F
        # logit = einops.rearrange(logit, 'B Head (F1 HW1) (F2 HW2) -> (B Head F1 F2) HW1 HW2', B=B,Head=Head, F1=F, F2=F, HW1=HW, HW2=HW)
        mask_per_view = cycle_consistency_checker(logit, pixel_threshold=3) # (B Head F1 F2) HW 1
        # mask_per_view = einops.rearrange(mask_per_view, '(B Head F1 F2) HW 1 -> B Head (F1 HW) (F2 1)', B=B, Head=Head, F1=F, F2=F, HW=HW)
        # mask = mask.any(dim=-1) # B Head Q(F1HW) 
        return mask_per_view # B Head Q(F1HW) F2
    
    def resize_tok(tok, target_size):
        B, Head, FHW, C = tok.shape
        F = config.vggt_visualize.nframe
        HW = FHW // F
        H = W = int(math.sqrt(HW))
        tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', B=B, Head=Head, F=F, H=H, W=W, C=C)
        tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
        tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C',  B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
        return tok

    def get_costmap(sample_idx, query_idx, key_idx, gt_query, gt_key, x_coord, y_coord, mode, per_view: bool = False):
        # jinhyeok: hard-coded for analysis qual. 
        '''
        gt_query: torch.Size([B, 1, VHW, C])
        gt_key  : torch.Size([B, 1, VHW, C])
        x_coord, y_coord: pixel (0..511)
        mode: "tracking" | "attention" | "pointmap"
        per_view: if True, softmax is applied per key view (not across concatenated views)
        return: score in shape [HW, VHW]
        '''
        H, W = 32, 32
        HW = H * W

        # pixel → token index on 32x32 fmap
        y_feat_cost = int((y_coord / 512) * H)
        x_feat_cost = int((x_coord / 512) * H)
        query_token_idx_cost = y_feat_cost * H + x_feat_cost

        if mode == "pointmap":
            def distance_softmax(query_points, ref_points, temperature=1.0, cross_only=True, per_view=False):
                """
                query_points: (B, 3, H, W)
                ref_points  : (B, V, 3, H, W)
                returns     : (B, HW, VHW) probability maps
                            if per_view=True, softmax is per-view over the last HW of each view.
                """
                B, _, Hx, Wx = query_points.shape
                assert Hx == H and Wx == W
                V = ref_points.shape[1]

                # Flatten
                # query -> (B, HW, 3)
                query_flat = query_points.reshape(B, 3, -1).permute(0, 2, 1).contiguous()
                # refs  -> (B, V, HW, 3)
                refs_flat = ref_points.reshape(B, V, 3, -1).permute(0, 1, 3, 2).contiguous()

                if not cross_only:
                    # Optional: prepend query to keys along the "V" axis as its own 'view'
                    # (B, V+1, HW, 3), with first slot = query itself
                    query_as_view = query_flat.unsqueeze(1)  # (B,1,HW,3)
                    refs_flat = torch.cat([query_as_view, refs_flat], dim=1)
                    V = V + 1

                # Compute pairwise distances
                # We want (B, HW, V, HW): every query token to every token in each key view
                # Expand query to (B, 1, HW, 1, 3) then broadcast vs (B, V, 1, HW, 3)
                q_exp = query_flat.unsqueeze(1).unsqueeze(3)           # (B, 1,   HW, 1, 3)
                k_exp = refs_flat.unsqueeze(2)                          # (B, V,   1,  HW, 3)
                diff  = q_exp - k_exp                                   # (B, V, HW, HW, 3) but with dims permutable
                # put as (B, HW, V, HW, 3) for norm over last axis
                diff  = diff.permute(0, 2, 1, 3, 4).contiguous()        # (B, HW, V, HW, 3)
                dist  = torch.norm(diff, dim=-1)                        # (B, HW, V, HW)

                # Map distance to logits (smaller dist → higher logit)
                logits = -dist / max(temperature, 1e-8)                 # (B, HW, V, HW)

                if per_view:
                    # softmax within each view across that view's HW
                    # jinhyeok: for analysis qual. remove inconsistent keys
                    drop_key_idx = []
                    if sample_idx == 73 and query_idx == 0 and key_idx == [0,1,2]: # hardcoding for analysis qual
                        drop_key_idx = [1, 2]
                    elif sample_idx == 73 and query_idx == 2 and key_idx == [0,1,2]: # hardcoding for analysis qual
                        drop_key_idx = [0]
                    else:
                        for i in range(V):
                            logit = logits[:, :, i, :]                        # (B, HW, HW)
                            mask = get_consistency_mask(logit)  # (B, HW,)
                            if mask[:, query_token_idx_cost] == 0:
                                drop_key_idx.append(i)
                    if len(drop_key_idx) > 0:
                        print(f"drop inconsistent keys: {drop_key_idx} at query idx {query_idx}, key idx {key_idx}")
                        logits[:, :, drop_key_idx, :] = -100.0  # very low logit to ignore
                    probs = torch.softmax(logits, dim=-1)               # (B, HW, V, HW)
                    probs = probs.reshape(B, HW, V * HW)                # (B, HW, VHW)
                else:
                    # softmax across all concatenated keys (V*HW)
                    logits_cat = logits.reshape(B, HW, V * HW)          # (B, HW, VHW)
                    probs = torch.softmax(logits_cat, dim=-1)           # (B, HW, VHW)

                return probs  # (B, HW, VHW)

            # Resize tokens to 32x32 and put channel last for slicing
            query_tok = resize_tok(gt_query, target_size=H)  # B Head (FHW) C
            key_tok   = resize_tok(gt_key,   target_size=W)  # B Head (FHW) C
            query_tok = query_tok.permute(0, 1, 3, 2)        # B Head C (FHW)
            key_tok   = key_tok.permute(0, 1, 3, 2)          # B Head C (FHW)

            # select target view tokens: (B=1, 3, H, W)
            q_start = (query_idx) * HW
            q_end   = (query_idx + 1) * HW
            query = query_tok[:, 0, :, q_start:q_end].reshape(1, 3, H, W)

            # collect reference tokens for all key_idx → (1, V, 3, H, W)
            keys = []
            for idx in key_idx:
                k_start = idx * HW
                k_end   = (idx + 1) * HW
                ref = key_tok[:, 0, :, k_start:k_end].reshape(1, 3, H, W)
                keys.append(ref)
            keys = torch.stack(keys, dim=1)

            # compute attention probabilities
            attn_maps = distance_softmax(
                query, keys,
                temperature=config.vggt_visualize.temperature,
                cross_only=True,
                per_view=per_view
            )  # (B=1, HW, VHW)

            all_scores = attn_maps[0, query_token_idx_cost]  # (VHW,)

            # layout into a horizontal strip (H, V*W)
            tokens_per_img = H * W
            scores_split = all_scores.split(tokens_per_img)  # list of V tensors [HW]
            # score = torch.cat([s.reshape(H, W) for s in scores_split], dim=-1)
            score = torch.cat([F.interpolate(s.reshape(H, W).unsqueeze(0).unsqueeze(0), size=(16, 16), mode='nearest').squeeze(0).squeeze(0)
                                for s in scores_split], dim=-1)
            return score

    
    if config.vggt_visualize.mode != "pointmap":
        raise NotImplementedError("Only pointmap mode is implemented in this version.")
        vggt_distill_config = {
            "cache_attn_layer_ids": [],
            "cache_costmap_types": config.vggt_visualize.mode,
        }
        
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_distill_config).eval()
        for p in vggt_model.parameters():
            p.requires_grad = False

        vggt_model = vggt_model.to(device)
        vggt_model.eval()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    

    if config.vggt_visualize.idx_set:
            end_idx = config.vggt_visualize.idx_set[-1]

    for idx, batch in enumerate(val_dataloader):
        if config.vggt_visualize.idx_set and idx not in config.vggt_visualize.idx_set:
            continue
        elif config.vggt_visualize.idx_set is None and idx < config.vggt_visualize.start_data_idx:
            continue

        outdir_root = os.path.join("outputs_vggt_map", timestamp, f"sample{idx}")
        os.makedirs(outdir_root, exist_ok=True)

        images = batch['image'].to(device)  # B V C H W  (V == nframe)
        Bv, V, C, H, W = images.shape
        assert V == nframe, f"batch frames {V} != nframe {nframe}"

        # ---------- choose coordinates (one pair for all targets by default) ----------
        # If you want different coords per target, move this inside the per-target loop below.
        # --- visualize with grid before input ---
        if config.vggt_visualize.view_select:
            # For multi-target, overlay grid on ALL target frames for clarity
            for t in range(cond_num, nframe):  # <<< CHANGED >>>
                tgt_image_t = images.squeeze()[t]  # [C,H,W]
                overlay_grid_and_save(tgt_image_t, spacing=64,
                                      out_path=os.path.join(outdir_root, f"VIS_OVERLAY_t{t}.png"))
            save_image(images.squeeze()[:cond_num], os.path.join(outdir_root, "VIS_REFERENCE.png"))
            while True:
                answer = input("Do you want to visualize? (yes/no): ").strip().lower()
                if answer in ("yes", "no"):
                    counter_out = (answer == "no")
                    break
                else:
                    print("Invalid input. Please type 'yes' or 'no'.")
            if counter_out:
                continue
        
        if config.vggt_visualize.coord_select:
            if config.vggt_visualize.view_select is False:
                for t in range(cond_num, nframe):  # <<< CHANGED >>>
                    tgt_image_t = images.squeeze()[t]
                    overlay_grid_and_save(tgt_image_t, spacing=64,
                                          out_path=os.path.join(outdir_root, f"VIS_OVERLAY_t{t}.png"))
                save_image(images.squeeze()[:cond_num], os.path.join(outdir_root, "VIS_REFERENCE.png"))
            while True: 
                x_coord = int(input("Enter x coordinate (0–512): "))
                y_coord = int(input("Enter y coordinate (0–512): "))
                if 0 < x_coord < 512 and 0 < y_coord < 512: 
                    break
                else: 
                    print("Invalid input. Please type valid values. ")
        else:
            x_coord = random.randint(0, 511)
            y_coord = random.randint(0, 511)

        coords = {"x": x_coord, "y": y_coord}
        with open(os.path.join(outdir_root, "coords.json"), "w") as f:
            json.dump(coords, f, indent=4) 

        # pred = vggt_model(images[:, -1])

        stacked_images = []
        for t in range(0, nframe):  # <<< CHANGED >>>
            query_idx = [t]
            if config.vggt_visualize.cross_only:
                key_idx = [i for i in range(nframe) if i not in query_idx]             # <<< CHANGED >>> refs only
            else:
                key_idx = list(range(nframe))               # refs + all frames

            if "pointmap" in config.vggt_visualize.mode:
                pointmap = batch['point_map'] # [1, V, 3, 512, 512]
                gt_query = pointmap.permute(0,1,3,4,2).reshape(1, 1, -1, 3) # [1, 4, 512, 512, 3]
                gt_key = gt_query
                # hard-coding for analysis qual
                score = get_costmap(idx, t, key_idx, gt_query, gt_key, x_coord, y_coord,
                                    config.vggt_visualize.mode, per_view = config.vggt_visualize.per_view)
                
                combined_img = save_image_jinhk(images, t, x_coord, y_coord, score)
                stacked_images.append(combined_img)
        stacked_images = concat_vertical(stacked_images)
        stacked_images.save(os.path.join(outdir_root, f"pointmap.png"))

        if config.vggt_visualize.save_stack:
            # Save stacked images (ref + tgt)
            save_image(images.squeeze(), os.path.join(outdir_root, f"VIS_STACKED.png"))

        if config.vggt_visualize.idx_set:
            if idx >= end_idx:
                break
        if config.vggt_visualize.idx_set is None and idx >= config.vggt_visualize.end_data_idx:
            break
    
    print("Done!")
        

if __name__ == "__main__":

    config_file_path = 'configs/viz_vggt.yaml'
    config = EasyDict(OmegaConf.load(config_file_path))
    # data
    nframe = config.vggt_visualize.nframe # View length
    cond_num = config.vggt_visualize.cond_num # Condition among nframe
    inference_view_range = config.vggt_visualize.inference_view_range # change if you want 
    #caching_unet_attn_layers = range(15)

    main(nframe, cond_num, inference_view_range, config=config)