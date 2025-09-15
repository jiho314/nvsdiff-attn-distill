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
from src.distill_utils.attn_visualize import mark_point_on_img, save_image_total, overlay_grid_and_save

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

def main(cfg):
    # val_wds_dataset_config = {
    #     'url_paths': [ "/mnt/data2/minseop/realestate_train_wds", ],
    #     'dataset_length': 200,
    #     'resampled': False,
    #     'shardshuffle': False,
    #     'min_view_range': 10,
    #     'max_view_range': 20,
    #     'process_kwargs': {
    #         'get_square_extrinsic': True
    #     }
    # }

    # val_dataset = build_re10k_wds(
    #     num_viewpoints=cfg.nframes,
    #     **val_wds_dataset_config
    # ) 
    
    val_dataset = build_re10k_wds(
        url_paths = [ "/mnt/data2/minseop/realestate_val_wds" ] ,
        dataset_length = 100000,
        resampled = False,
        shardshuffle=False,
        num_viewpoints=cfg.nframes,
        inference=True,
        inference_view_range = cfg.inference_view_range
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
    
    # 2) Slice attnmap: 
    def slice_attnmap(attnmap, query_idx, key_idx):
        B, Head, Q, K = attnmap.shape
        F = cfg.nframes # warn: hardcoding
        HW = Q // F
        attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
        attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
        attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
        return attnmap

    # 1) Resize token: gt -> pred
    def resize_tok(tok, target_size):
        B, Head, FHW, C = tok.shape
        F = cfg.nframes # warn: hardcoding
        HW = FHW // F
        H = W = int(math.sqrt(HW))
        tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', B=B, Head=Head, F=F, H=H, W=W, C=C)
        tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
        tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C',  B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
        return tok


    def get_costmap(gt_query, gt_key, x_coord, y_coord, mode):
        '''
        gt_query: query in shape # torch.Size([B, 1, VHW, C])
        gt_key: key in shape # torch.Size([B, 1, VHW, C])
        x_coord: (0, 511)
        y_coord: (0, 511)
        mode: three options, "tracking", "attention", "point"
        return: score in shape [HW, VHW]
        '''
        H, W = 32, 32
        HW = H * W
        # Convert pixel coords -> feature map index
        y_feat_cost = int((y_coord / 512) * H)
        x_feat_cost = int((x_coord / 512) * H)
        query_token_idx_cost = y_feat_cost * H + x_feat_cost
        
        query_idx = [-1]
        if config.cross_only: 
            key_idx = list(range(cfg.nframes-1))
        else:
            key_idx = list(range(cfg.nframes))

        if mode == "tracking":
            query, key = resize_tok(gt_query, target_size=H), resize_tok(gt_key, target_size=W),
            gt_attn_logit = query @ key.transpose(-1, -2)
            gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=query_idx, key_idx=key_idx)
            attn_maps = gt_attn_logit.squeeze()
            all_scores = torch.softmax(attn_maps[query_token_idx_cost] / 8, dim=-1)
        elif mode == "attention":
            query, key = resize_tok(gt_query, target_size=H), resize_tok(gt_key, target_size=W),
            gt_attn_logit = query @ key.transpose(-1, -2)
            gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=query_idx, key_idx=key_idx)
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
                B, _, H, W = query_points.shape
                V = ref_points.shape[1]

                # Flatten query: (B, HW, 3)
                query_points = query_points.reshape(B, 3, -1).permute(0, 2, 1)

                # Flatten refs: (B, VHW, 3)
                ref_points = ref_points.reshape(B, V, 3, -1).permute(0, 1, 3, 2).reshape(B, -1, 3)

                if not cross_only:
                    # Concatenate query to refs → (B, (V+1)HW, 3)
                    ref_points = torch.cat((query_points, ref_points), dim=1)

                # Pairwise distances (B, HW, VHW)
                diff = query_points.unsqueeze(2) - ref_points.unsqueeze(1)
                dist = torch.norm(diff, dim=-1)

                # Convert to probabilities (smaller distance = higher prob)
                logits = -torch.log(dist + 1e-6) / temperature
                probs = torch.softmax(logits, dim=-1)

                return probs.squeeze()
            query, key = resize_tok(gt_query, target_size=H), resize_tok(gt_key, target_size=W)  # B Head (FHW) C
            query, key = query.permute(0, 1, 3, 2), key.permute(0, 1, 3, 2)  # B Head C (FHW)
            HW = H * W
            # take target tokens → shape (1, 3, H, W)
            query = query[:, 0, :, (cfg.nframes-1) * HW:].reshape(1, 3, H, W)
            # collect reference tokens for all key_idx
            ref_imgs = []
            for idx in key_idx:
                start = idx * HW
                end   = (idx + 1) * HW
                ref = key[:, 0, :, start:end].reshape(1, 3, H, W)
                ref_imgs.append(ref)

            # stack along V dimension → (1, V, 3, H, W)
            ref_imgs = torch.stack(ref_imgs, dim=1)
            attn_maps = distance_softmax(query, ref_imgs)
            all_scores = attn_maps[query_token_idx_cost]
        
        tokens_per_img = H * W
        scores_split = all_scores.split(tokens_per_img)
        score = torch.cat([s.reshape(H, W) for s in scores_split], dim=-1)
        return score
    
    # with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_distill_config).eval()
    for p in vggt_model.parameters():
        p.requires_grad = False

    device = 'cuda:0'
    vggt_model = vggt_model.to(device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for i, batch in enumerate(loader):
        
        images = batch['image'].to(device)  # B V C H W
        vggt_pred = vggt_model(images)

        # --- get target image ---
        tgt_image = images.squeeze()[-1]      # [C, H, W]

        # --- visualize with grid before input ---
        
        if cfg.view_select:
            overlay_grid_and_save(tgt_image, spacing=64, out_path="VIS_OVERLAY.png")
            save_image(images.squeeze()[:-1], "VIS_REFERENCE.png")
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
        outdir = os.path.join("outputs_vggt", timestamp, f"sample{i}")
        os.makedirs(outdir, exist_ok=True)
        coords = {"x": x_coord, "y": y_coord}
        json_path = os.path.join(outdir, "coords.json")
        with open(json_path, "w") as f:
            json.dump(coords, f, indent=4)      
    
        if "track_head" in cfg.mode:
            gt_query, gt_key = vggt_pred['attn_cache']['track_head']['query'], vggt_pred['attn_cache']['track_head']['key'] # torch.Size([1, 1, 2668324, 128])
            score = get_costmap(gt_query, gt_key, x_coord, y_coord, "tracking") # 32, 3*32
            combined_img = save_image_total(images, x_coord, y_coord, score,  mode="vertical")
            combined_img.save(f"{outdir}/tracking.png")
        if "attention" in cfg.mode:
            for layer in cfg.attn_idx:
                gt_query, gt_key = vggt_pred['attn_cache'][f'{layer}']['query'], vggt_pred['attn_cache'][f'{layer}']['key'] # torch.Size([1, 16, 5476, 64])
                score = get_costmap(gt_query, gt_key, x_coord, y_coord, "attention") # 32, 3*32
                combined_img = save_image_total(images, x_coord, y_coord, score, mode="vertical")
                combined_img.save(f"{outdir}/attn{layer}.png")
        if "pointmap" in cfg.mode:
            pointmap = batch['point_map'] # [1, V, 3, 512, 512]
            gt_query = pointmap.permute(0,1,3,4,2).reshape(1, 1, -1, 3) # [1, 4, 512, 512, 3]
            gt_key = gt_query
            score = get_costmap(gt_query, gt_key, x_coord, y_coord, "pointmap") 
            combined_img = save_image_total(images, x_coord, y_coord, score,  mode="vertical")
            combined_img.save(f"{outdir}/pointmap.png")
        
        if cfg.save_originals: 
            save_image(tgt_image, f"{outdir}/TARGET.png")
            tgt_image_marked = mark_point_on_img(tgt_image, x_coord, y_coord)
            target_marked = torch.from_numpy(tgt_image_marked).permute(2, 0, 1).float() / 255.0
            save_image(target_marked, f"{outdir}/TARGET_MARKED.png")

            ref_imgs = images.squeeze()[:-1]
            #ref_concat = torch.cat([img for img in ref_imgs], dim=-2)  # shape [C, H, W*3] # save as one single image save_image(ref_concat, f"{outdir}/REFERENCE.png")
            for i, img in enumerate(ref_imgs):
                save_image(img, f"{outdir}/REFERENCE{i}.png")
            print(f"Saved visualization to outputs.png")
        
        if i == 200: break
        
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/data1/jiho/vggt-nvs/nvsdiff-attn-distill/costmap_visualization.yaml")
    args = parser.parse_args()
    set_seed(0)
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    main(config)
    
    
    
    
    
    
    
    

    
    

