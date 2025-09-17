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
from src.distill_utils.attn_visualize import mark_point_on_img, save_image_total, overlay_grid_and_save


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



def main(nframe, cond_num, inference_view_range, 
         caching_unet_attn_layers, noise_timestep, 
         resume_checkpoint, config, rank = 0):
    
    vggt_distill_config = {
        "cache_attn_layer_ids": config.attn_idx if "attention" in config.mode else [],
        "cache_costmap_types": config.mode
    }
    def slice_softmax(attnmap):
        '''
        attnmap: [fH*fW, nfrmaes*fH*fW]
        nframes: # of frames
        '''
        feature_size, nfeature_size = attnmap.shape
        nframes = nfeature_size // feature_size
        total_attn_map = []
        for i in range(nframes):
            tmp = attnmap[:, i*feature_size:(i+1)*feature_size]
            tmp = torch.softmax(tmp, dim=-1)
            total_attn_map.append(tmp)
        
        return torch.cat(total_attn_map, dim=1)
    
    def slice_attnmap(attnmap, query_idx, key_idx):
        B, Head, Q, K = attnmap.shape
        F = nframe # warn: hardcoding
        HW = Q // F
        attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
        attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
        attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
        return attnmap

    # 1) Resize token: gt -> pred
    def resize_tok(tok, target_size):
        B, Head, FHW, C = tok.shape
        F = nframe # warn: hardcoding
        HW = FHW // F
        H = W = int(math.sqrt(HW))
        tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', B=B, Head=Head, F=F, H=H, W=W, C=C)
        tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
        tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C',  B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
        return tok
    
    def get_costmap(gt_query, gt_key, mode, H):
        '''
        gt_query: query in shape # torch.Size([B, 1, VHW, C])
        gt_key: key in shape # torch.Size([B, 1, VHW, C])
        x_coord: (0, 511)
        y_coord: (0, 511)
        mode: three options, "tracking", "attention", "point"
        return: score in shape [HW, VHW]
        '''
        H, W = int(math.sqrt(H)), int(math.sqrt(H))
        HW = H * W
        # Convert pixel coords -> feature map index
        # y_feat_cost = int((y_coord / 512) * H)
        # x_feat_cost = int((x_coord / 512) * H)
        # query_token_idx_cost = y_feat_cost * H + x_feat_cost
        
        query_idx = [-1]
        if config.cross_only: 
            key_idx = list(range(config.nframes-1))
        else:
            key_idx = list(range(config.nframes))

        if mode == "tracking":
            query, key = resize_tok(gt_query, target_size=H), resize_tok(gt_key, target_size=W),
            gt_attn_logit = query @ key.transpose(-1, -2)
            gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=query_idx, key_idx=key_idx)
            attn_maps = gt_attn_logit.squeeze()
            all_scores = slice_softmax(attn_maps / 8) if config.split else torch.softmax(attn_maps / 8, dim=-1)
            #all_scores = torch.softmax(attn_maps / 8, dim=-1)
        elif mode == "attention":
            query, key = resize_tok(gt_query, target_size=H), resize_tok(gt_key, target_size=W),
            gt_attn_logit = query @ key.transpose(-1, -2)
            gt_attn_logit = slice_attnmap(gt_attn_logit, query_idx=query_idx, key_idx=key_idx)
            attn_maps = gt_attn_logit.mean(dim=1) # average over head
            attn_maps = attn_maps.squeeze()
            all_scores = slice_softmax(attn_maps / 8) if config.split else torch.softmax(attn_maps / 8, dim=-1)
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
                #logits = -torch.log(dist + 1e-6) / temperature
                logits = -dist / temperature
                probs = slice_softmax(logits) if config.split else torch.softmax(logits, dim=-1)

                return probs.squeeze()
            query, key = resize_tok(gt_query, target_size=H), resize_tok(gt_key, target_size=W)  # B Head (FHW) C
            query, key = query.permute(0, 1, 3, 2), key.permute(0, 1, 3, 2)  # B Head C (FHW)
            HW = H * W
            # take target tokens → shape (1, 3, H, W)
            query = query[:, 0, :, (config.nframes-1) * HW:].reshape(1, 3, H, W)
            # collect reference tokens for all key_idx
            ref_imgs = []
            for idx in key_idx:
                start = idx * HW
                end   = (idx + 1) * HW
                ref = key[:, 0, :, start:end].reshape(1, 3, H, W)
                ref_imgs.append(ref)

            # stack along V dimension → (1, V, 3, H, W)
            ref_imgs = torch.stack(ref_imgs, dim=1)
            attn_maps = distance_softmax(query, ref_imgs, config.point_temperature)
            all_scores = attn_maps
        
        # tokens_per_img = H * W
        # scores_split = all_scores.split(tokens_per_img)
        # score = torch.cat([s.reshape(H, W) for s in scores_split], dim=-1)
        return all_scores
    # args, _, cfg = parse_args()
    
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_distill_config).eval()
    for p in vggt_model.parameters():
        p.requires_grad = False

    device = 'cuda'
    vggt_model = vggt_model.to(device)

    set_seed(0)
    device= f"cuda"
    # 1) model
    vae = AutoencoderKL.from_pretrained(f"{config.pretrained_model_name_or_path}", subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                    subfolder="unet",
                                                    rank=rank,
                                                    model_cfg=config.model_cfg,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True).to(device)
    vae.eval()
    unet.eval()

    # if args.use_ema:
    #     ema_unet = copy.deepcopy(unet)
    #     ema_unet = EMAModel(
    #         ema_unet.parameters(),
    #         model_cls=UNet2DConditionModel,
    #         model_config=ema_unet.config,
    #         foreach=True,
    #         decay=args.ema_decay,
    #         min_decay=args.min_decay,
    #         ema_decay_step=args.ema_decay_step
    #     )
    # else:
    #     ema_unet = None

    ema_unet = None



    # checkpoint load
    
    # reload weights for unet here
    weights = torch.load(f"{resume_checkpoint}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
    unet.load_state_dict(weights['module'], strict=False)  # we usually need to resume partial weights here
    
    # if args.use_ema:
    #     if os.path.exists(f"{resume_checkpoint}/ema_unet.pt"):
    #         print("Find ema weights, load it!")
    #         weights = torch.load(f"{resume_checkpoint}/ema_unet.pt", map_location="cpu")
    #         # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
    #         unet.load_state_dict(weights, strict=False)  # unet load first
    #     else:
    #         print("No ema weights, load original weights instead!")
    #         weights = torch.load(f"{resume_checkpoint}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
    #         # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
    #         unet.load_state_dict(weights['module'], strict=False)  # unet load first
    #     ema_unet.load_state_dict({"shadow_params": [p.clone().detach() for p in list(unet.parameters())]})
    #     if not args.reset_ema_step:
    #         ema_params = torch.load(f"{resume_checkpoint}/ema_unet_params.pt", map_location="cpu")
    #         ema_unet.optimization_step = ema_params['optimization_step']
    

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
    
    
    def slice_attnmap(attnmap, query_idx, key_idx):
        B, Head, Q, K = attnmap.shape
        F = nframe
        HW = Q // F
        attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
        attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
        attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
        return attnmap
    
    # 3) Model Inference
    def unet_inference(batch):
        # batch: always order by seq idx
        # uniform_push_batch: set condition (uniform sampling, no randomness) to front, tgt to last
        # batch = uniform_push_batch(batch, cond_num)
        image = batch["image"].to(device)  #  0 1 tensor [B,F,3,H,W]
        extri_, intri_ = batch["extrinsic"], batch["intrinsic"]

        b, f, _, h, w = image.shape
        if extri_.shape[-2] == 3:
            new_extri_ = torch.zeros((b, f, 4, 4), device=device, dtype=extri_.dtype)
            new_extri_[:, :, :3, :4] = extri_
            new_extri_[:, :, 3, 3] = 1.0
            extri_ = new_extri_
        extri_ = einops.rearrange(extri_, "b f c1 c2 -> (b f) c1 c2", f=f)
        intri_ = einops.rearrange(intri_, "b f c1 c2 -> (b f) c1 c2", f=f)
        camera_embedding = get_camera_embedding(intri_.to(device), extri_.to(device),
                                                    b, f, h, w, config=config).to(device=device)  # b,f,c,h,w
        image_normalized = image * 2.0 - 1.0
        latents = slice_vae_encode(vae, image_normalized, sub_size=16)
        latents = latents * vae.config.scaling_factor
        _, _, _, latent_h, latent_w = latents.shape

        # build masks (cond / gen), valid:0, mask:1
        masks = torch.ones((b, nframe, 1, h, w), device=device, dtype=latents.dtype)
        latent_masks = torch.ones((b, nframe, 1, latent_h, latent_w), device=device, dtype=latents.dtype)
        masks[:, :cond_num] = 0
        latent_masks[:, :cond_num] = 0

        train_noise_scheduler = get_diffusion_scheduler(config, name="DDPM")
        if config.get("adaptive_betas", False):
            noise_scheduler = train_noise_scheduler[nframe - cond_num]
        else:
            noise_scheduler = train_noise_scheduler

        timesteps = torch.tensor([noise_timestep] * b, device=latents.device).long()
        # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents.device).long()
        noise = torch.randn_like(latents)
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        inputs = noisy_latents  # [B,F,4,h,w]
        add_inputs = torch.cat([masks, camera_embedding], dim=2)  # [B,F,1+6,H,W]

        # get class label (domain switcher)
        domain_dict = config.model_cfg.get("domain_dict", None)
        if domain_dict is not None:
            tags = batch["tag"][::f]
            class_labels = [domain_dict.get(tag, domain_dict['others']) for tag in tags]
            class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
        else:
            class_labels = None

        model_pred = unet(inputs, timesteps, encoder_hidden_states=None, add_inputs=add_inputs,
                                class_labels=class_labels, coords=None, return_dict=False)[0]  # [BF,C,H,W]
        # diff_loss = torch.nn.functional.mse_loss(model_pred.float()[:, cond_num:], target.float()[:, cond_num:], reduction="mean")
        return model_pred

    set_attn_cache(unet, caching_unet_attn_layers)
    
    
    ## visualization 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    similarity_score = {}
    mode = config.mode[0]
    parent_name = os.path.basename(os.path.dirname(resume_checkpoint))   # lr1_cosine_noema
    ckpt_name = os.path.basename(resume_checkpoint)                     # checkpoint-30000
    outdir = os.path.join("outputs_comparison", timestamp, parent_name, ckpt_name, f"{noise_timestep}, {config.mode[0]}, {config.similarity_measure}")
    os.makedirs(outdir, exist_ok=True)
    
    # minkyung TODO: 여러 idx에 대해서 viz
    for idx, batch in enumerate(val_dataloader):
        if idx == config.nsample: 
            averaged_score = {}
            for k, vals in similarity_score.items():
                # convert list of tensors -> stacked tensor -> mean -> float
                vals_tensor = torch.tensor([v.item() if torch.is_tensor(v) else v for v in vals])
                averaged_score[k] = float(vals_tensor.mean().item())
            print(averaged_score)
            # Save to JSON
            with open(os.path.join(outdir,"similarity_score_avg.json"), "w") as f:
                json.dump(averaged_score, f, indent=4)
            print("successfully saved the .json")
            break
        print(f"processing sample {idx}")
        images = batch['image'].to(device)  # B V C H W
        #tgt_image = images.squeeze()[0]      # [C, H, W]

        # --- visualize with grid before input ---
        
        # if config.view_select:
        #     overlay_grid_and_save(tgt_image, spacing=64, out_path="VIS_OVERLAY.png")
        #     save_image(images.squeeze()[1:], "VIS_REFERENCE.png")
        #     while True:
        #         answer = input("Do you want to visualize? (yes/no): ").strip().lower()
        #         if answer == "yes":
        #             # run visualization code
        #             counter_out = False
        #             break
        #         elif answer == "no":
        #             # skip this batch
        #             counter_out = True
        #             break  # goes back to your for-loop
        #         else:
        #             print("Invalid input. Please type 'yes' or 'no'.")
                    
        #     if counter_out:
        #         continue
        
        # if config.coord_select:
        #     if config.view_select is False:
        #         overlay_grid_and_save(tgt_image, spacing=64, out_path="VIS_OVERLAY.png")
        #         save_image(images.squeeze()[1:], "VIS_REFERENCE.png")
        #     # Ask user for coordinates
        #     while True: 
        #         x_coord = int(input("Enter x coordinate (0–512): "))
        #         y_coord = int(input("Enter y coordinate (0–512): "))
        #         if 0 < x_coord < 512 and 0 < y_coord < 512: 
        #             break
        #         else: 
        #             print("Invalid input. Please type valid values. ")
        # else:
        #     # Random selection
        #     x_coord = random.randint(0, 511)
        #     y_coord = random.randint(0, 511)
        # coords = {"x": x_coord, "y": y_coord}
        # json_path = os.path.join(outdir, "coords.json")
        # with open(json_path, "w") as f:
        #     json.dump(coords, f, indent=4) 

        # batch: always order by seq idx
        # uniform_push_batch: set condition (uniform sampling, no randomness) to front, tgt to last
        _ = unet_inference(batch)
        unet_attn_cache = pop_cached_attn(unet)
        ''' unet_attn_cache(dict): {layer_id(str): attnmap tensor(B, head, Q(VHW), K(VHW)}
        '''
        for unet_layer in caching_unet_attn_layers:
            #print(f"layer: {unet_layer} shape : {unet_attn_cache[str(unet_layer)].shape}")
            unet_attn_logit = unet_attn_cache[str(unet_layer)] # [B, H, Q, K]
            B, H, Q, K = unet_attn_logit.shape
            # if B == 3 : # self attention
            #     query_size = int(math.sqrt(Q))
            #     # Convert pixel coords -> feature map index
            #     y_feat_cost = int((y_coord / 512) * query_size)
            #     x_feat_cost = int((x_coord / 512) * query_size)
            #     query_token_idx_cost = y_feat_cost * query_size + x_feat_cost
            #     attn_maps = unet_attn_logit.mean(dim=1) # [3, Q, K]
            #     all_scores = torch.softmax(attn_maps[:, query_token_idx_cost], dim=-1)
            #     score1 = all_scores[0].reshape(query_size, query_size)
            #     score2 = all_scores[1].reshape(query_size, query_size)
            #     score = torch.cat([score1, score2], dim=-1)
            # else: 
            query_idx = [nframe-1]
            if config.cross_only: 
                key_idx = list(range(nframe-1))
            else:
                key_idx = list(range(nframe))
            unet_attn_logit = slice_attnmap(unet_attn_logit, query_idx=query_idx, key_idx=key_idx) # [B, H, Q, K]
            # average over head
            unet_attn_logit = unet_attn_logit.mean(dim=1)   # [B, Q, K]
            unet_attn_logit = unet_attn_logit[0]                  # take batch 0 → [Q, K]
            # query_size = int(math.sqrt(Q))
            # Convert pixel coords -> feature map index
            # y_feat_cost = int((y_coord / 512) * query_size)
            # x_feat_cost = int((x_coord / 512) * query_size)
            # query_token_idx_cost = y_feat_cost * query_size + x_feat_cost
            
            # attention for chosen query token
            all_scores = slice_softmax(unet_attn_logit) if config.split else torch.softmax(unet_attn_logit, dim=-1)
            del unet_attn_logit
            H, W = all_scores.shape
            vggt_pred = vggt_model(images)
            if "track_head" in config.mode:
                gt_query, gt_key = vggt_pred['attn_cache']['track_head']['query'], vggt_pred['attn_cache']['track_head']['key'] # torch.Size([1, 1, 2668324, 128])
                score_vggt = get_costmap(gt_query, gt_key, "tracking", H) # 32, 3*32
                # combined_img = save_image_total(images, x_coord, y_coord, score)
                # combined_img.save(f"{outdir}/tracking.png")
                # save_individuals(combined_img, f"{outdir}/tracking")  
            if "attention" in config.mode:
                for layer in config.attn_idx:
                    gt_query, gt_key = vggt_pred['attn_cache'][f'{layer}']['query'], vggt_pred['attn_cache'][f'{layer}']['key'] # torch.Size([1, 16, 5476, 64])
                    score_vggt = get_costmap(gt_query, gt_key, "attention", H) # 32, 3*32
                    # combined_img = save_image_total(images, x_coord, y_coord, score)
                    # combined_img.save(f"{outdir}/attn{layer}.png")
                    # if cfg.save_individuals: 
                    #     save_individuals(combined_img, f"{outdir}/attn{layer}")  
            if "pointmap" in config.mode:
                pointmap = batch['point_map'] # [1, V, 3, 512, 512]
                gt_query = pointmap.permute(0,1,3,4,2).reshape(1, 1, -1, 3) # [1, 4, 512, 512, 3]
                gt_key = gt_query
                score_vggt = get_costmap(gt_query, gt_key, "pointmap", H) 
                # combined_img = save_image_total(images, x_coord, y_coord, score)
                # combined_img.save(f"{outdir}/pointmap.png")
                
                # if cfg.save_individuals: 
                #     save_individuals(combined_img, f"{outdir}/pointmap")
            
            # combined_img = save_image_total(images, x_coord, y_coord, score)
            # combined_img.save(f"{outdir}/attn{unet_layer}.png")

            score_vggt = score_vggt.to(device)
            if config.split: 
                pass
                if config.similarity_measure == "cross_entropy":
                    eps = 1e-12
                    cross_entropy = -(score_vggt * torch.log(all_scores + eps)).sum(axis=1)
                    cross_entropy_mean = cross_entropy.mean()
                    measure = cross_entropy_mean
                elif config.similarity_measure == "l1":
                    l1_loss = torch.abs(score_vggt - all_scores).sum(axis=1)  # sum over features
                    l1_loss_mean = l1_loss.mean()  # average over rows
                    measure = l1_loss_mean
            else:
                if config.similarity_measure == "cross_entropy":
                    eps = 1e-12
                    cross_entropy = -(score_vggt * torch.log(all_scores + eps)).sum(axis=1)
                    cross_entropy_mean = cross_entropy.mean()
                    measure = cross_entropy_mean
                elif config.similarity_measure == "l1":
                    l1_loss = torch.abs(score_vggt - all_scores).sum(axis=1)  # sum over features
                    l1_loss_mean = l1_loss.mean()  # average over rows
                    measure = l1_loss_mean
              
            if f'{unet_layer}' not in similarity_score.keys():
                similarity_score[f'{unet_layer}'] = [] 
            similarity_score[f'{unet_layer}'].append(measure.item())
        # if config.save_originals:
        #     save_image(tgt_image, f"{outdir}/TARGET.png")
        #     tgt_image_marked = mark_point_on_img(tgt_image, x_coord, y_coord)
        #     target_marked = torch.from_numpy(tgt_image_marked).permute(2, 0, 1).float() / 255.0
        #     save_image(target_marked, f"{outdir}/TARGET_MARKED.png")

        #     ref_imgs = images.squeeze()[:-1]
        #     ref_concat = torch.cat([img for img in ref_imgs], dim=-1)  # shape [C, H, W*3] # save as one single image save_image(ref_concat, f"{outdir}/REFERENCE.png")
        #     save_image(ref_concat, f"{outdir}/REFERENCE.png")
        #     print(f"Saved visualization to outputs.png")

        

if __name__ == "__main__":
    
    # minkyung TODO
    # unet: (down_blocks(6), mid_block(1), up_blocks(9))
    #  size: [64,64,32,32,16,16] [8] [16,16,16,32,32,32,64,64,64]
    # checkpoint
    config_file_path = 'attn_vggt_similarity.yaml'
    config = EasyDict(OmegaConf.load(config_file_path))
    # noise
    noise_timestep = config.unet_visualize.noise_timestep
    # data
    nframe = config.nframes # View length
    cond_num = config.unet_visualize.cond_num # Condition among nframe
    inference_view_range = config.inference_view_range # change if you want 
    caching_unet_attn_layers = config.unet_visualize.caching_unet_attn_layers
    #caching_unet_attn_layers = range(15)

    main(nframe, cond_num, inference_view_range, caching_unet_attn_layers, noise_timestep=noise_timestep, resume_checkpoint=config.unet_visualize.resume_checkpoint, config=config)