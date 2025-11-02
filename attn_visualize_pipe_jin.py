from train_old_jinhk import *

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
import yaml
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline

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

def main(nframe, cond_num, inference_view_range, 
         caching_unet_attn_layers, noise_timestep, 
         resume_checkpoint, config, config_file_path, rank = 0):
    # args, _, cfg = parse_args()

    seed_everything(0)
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

    for param in vae.parameters():
        param.requires_grad = False
    for param in unet.parameters():
        param.requires_grad = False

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
    if not config.unet_visualize.is_scratch:
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
    # from src.datasets.re10k_wds import build_re10k_wds
    # val_dataset = build_re10k_wds(
    #     url_paths = [ "/mnt/data2/minseop/realestate_val_wds" ] ,
    #     dataset_length = 100000,
    #     resampled = False,
    #     shardshuffle=False,
    #     num_viewpoints= nframe,
    #     inference=True,
    #     inference_view_range = inference_view_range,
    #     process_kwargs={'get_square_extrinsic': True}
    # ) 
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     shuffle=False,
    #     sampler=None,
    #     batch_size=1,
    #     num_workers=0,
    #     drop_last=True,
    # )

    from src.datasets.co3d_wds import build_co3d_wds
    val_dataset = build_co3d_wds(
        url_paths = [ "/mnt/data2/minseop/co3d_wds_val_subset/unseen_scene" ] ,
        dataset_length = 240,
        resampled = False,
        shardshuffle=False,
        num_viewpoints= nframe,
        inference=True,
        inference_view_range = 6,
        process_kwargs = {"get_square_extrinsic" : True}
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
    # scheduler
    noise_scheduler = get_diffusion_scheduler(config, name="DDIM")
    if config.get("adaptive_betas", False):
        noise_scheduler = noise_scheduler[nframe - cond_num]
    # for attr_name in (
    #         "alphas_cumprod",
    #         "alphas_cumprod_prev",
    #         "betas",
    #         "alphas",
    #         "sigmas",
    #         "sqrt_alphas_cumprod",
    #         "sqrt_one_minus_alphas_cumprod",
    #     ):
    #     attr_value = getattr(noise_scheduler, attr_name, None)
    #     if isinstance(attr_value, torch.Tensor) and attr_value.device != latents.device:
    #         setattr(noise_scheduler, attr_name, attr_value.to(latents.device))
    noise_scheduler.set_timesteps(50)

    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to(device)

    # bad code but quick fix for skipping some scenes
    def denoising_this_scene_or_not(scene_idx, idxs):
        if idxs is None:
            return True
        elif scene_idx in idxs:
            return True
        else:
            print(f"Skipping scene {scene_idx} ...")
            return False
    
    def set_perturb_attn(unet, perturb_layers = []):
        ''' 
        perturb_layers: list of int layer index to cache attention (0~15)
        '''
        attn_proc_names = sorted(list(unet.attn_processors.keys()))
        for i, name in enumerate(attn_proc_names):
            if i in perturb_layers:
                unet.attn_processors[name].perturb = True
            else:
                unet.attn_processors[name].perturb = False
    
    if config.unet_visualize.perturb_attn:
        set_perturb_attn(pipeline.unet, config.unet_visualize.perturb_layers)
 
    # must set cache and pop cache, just one timestep to prevent OOM
    def set_attn_cache_hook(
            caching_layers,
            caching_t,
            set_attn_cache=set_attn_cache,
        ):
        def hook(module, input):
            if not hasattr(module, 'cached'):
                setattr(module, 'cached', False)
            t = input[1].item()
            if not module.cached and t <= caching_t:
                set_attn_cache(module, caching_layers)
                print(f"Set attn cache at timestep {t} for layers {caching_layers}")

        return hook
    
    def clear_attn_cache_hook(
            caching_t,
            unet_attn_cache = {},
            unset_attn_cache=unset_attn_cache,
            clear_attn_cache=clear_attn_cache,
        ):
        def hook(module, input, output):
            t = input[1].item()
            if not module.cached and t <= caching_t:
                unet_attn_cache['attn'] = pop_cached_attn(module)
                print(f"Attn saved at timestep {t}")
                setattr(module, 'cached', True)
                unset_attn_cache(module)
                clear_attn_cache(module)
            return output
        return hook

    cached = {'attn': None}

    if config.unet_visualize.cache_attn:
        # set attn cache before forward call
        pipeline.unet.register_forward_pre_hook(
            set_attn_cache_hook(caching_unet_attn_layers, noise_timestep)
        )
        # clear attn cache and pop after forward call
        pipeline.unet.register_forward_hook(
            clear_attn_cache_hook(noise_timestep, cached, unset_attn_cache, clear_attn_cache)
        )

    
    def unet_inference(batch, go_this_scene = True):
        # batch: always order by seq idx
        # uniform_push_batch: set condition (uniform sampling, no randomness) to front, tgt to last
        # batch = uniform_push_batch(batch, cond_num)
        image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
        image = image[:nframe]
        intri_ = intri_[:nframe]
        extri_ = extri_[:nframe]

        image_normalized = image * 2.0 - 1.0
        extrinsic, intrinsic = extri_, intri_

        tag, sequence_name, depth = None, None , None
        preds = pipeline(images=image_normalized, nframe=nframe, cond_num=cond_num,
                            height=image_normalized.shape[2], width=image_normalized.shape[3],
                            intrinsics=intrinsic, extrinsics=extrinsic,
                            num_inference_steps=50, guidance_scale=config.unet_visualize.cfg_scale,
                            output_type="np", config=config, tag=tag,
                            sequence_name=sequence_name, depth=depth, vae=vae,
                            interrupt=(not go_this_scene),
                            ).images
        x_0 = preds[cond_num:]  # numpy array, [V-cond_num, H, W, C]
        setattr(pipeline.unet, 'cached', False)
        return torch.tensor(x_0).permute(0, 3, 1, 2)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # output folder
    exp_name = config.unet_visualize.exp_name
    view = "per_view" if config.unet_visualize.per_view else "full"
    out_root = config.unet_visualize.out_root
    outdir_root = os.path.join(out_root, timestamp, exp_name, view, f"{noise_timestep}")
    os.makedirs(outdir_root, exist_ok=True)

    with open(os.path.join(outdir_root, "config.yaml"), "w") as f:
        config_file = OmegaConf.load(config_file_path)
        yaml.dump(OmegaConf.to_container(config_file.unet_visualize, resolve=True), f)

    if config.unet_visualize.idx_set:
        end_idx = config.unet_visualize.idx_set[-1]

    print(f"Skipping some useless idxs ...")
    for idx, batch in enumerate(val_dataloader):
        # if config.unet_visualize.idx_set and idx not in config.unet_visualize.idx_set:
        #     continue
        # elif config.unet_visualize.idx_set is None and idx < config.unet_visualize.start_data_idx:
        #     continue
        go_this_scene = denoising_this_scene_or_not(idx, config.unet_visualize.idx_set)

        
        images = batch['image'].to(device)  # B V C H W  (V == nframe)
        B, V, C, H, W = images.shape
        assert V == nframe, f"batch frames {V} != nframe {nframe}"

        # ---------- choose coordinates (one pair for all targets by default) ----------
        # If you want different coords per target, move this inside the per-target loop below.
        # --- visualize with grid before input ---
        if config.unet_visualize.view_select:
            # For multi-target, overlay grid on ALL target frames for clarity
            for t in range(cond_num, nframe):  # <<< CHANGED >>>
                tgt_image_t = images.squeeze()[t]  # [C,H,W]
                overlay_grid_and_save(tgt_image_t, spacing=64,
                                      out_path=os.path.join(outdir_root, f"VIS_OVERLAY_{t}.png"))
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

        if config.unet_visualize.reproduce_coords_dir is not None:
            dirs = os.listdir(config.unet_visualize.reproduce_coords_dir)
            reproduce_idxs = [int(d.split("_")[-1]) for d in dirs if d.startswith("sample_")]
        else:
            reproduce_idxs = []
        
        x_coords, y_coords = {}, {}

        if idx in reproduce_idxs:
            with open(os.path.join(config.unet_visualize.reproduce_coords_dir, f"sample_{idx}", "coords.json"), "r") as f:
                coords = json.load(f)
            for t in range(0, nframe):
                x_coords[t] = coords["x"][str(t)]
                y_coords[t] = coords["y"][str(t)]
                
                print(f"Reproducing coordinates from sample_{idx}, frame {t}: (x, y) = ({x_coords[t]}, {y_coords[t]})")

        elif config.unet_visualize.coord_select:
            if config.unet_visualize.view_select is False:
                for t in range(0, nframe):
                    tgt_image_t = images.squeeze()[t]
                    overlay_grid_and_save(tgt_image_t, spacing=64,
                                          out_path=os.path.join(outdir_root, f"VIS_OVERLAY_{t}.png"))
                save_image(images.squeeze()[:cond_num], os.path.join(outdir_root, "VIS_REFERENCE.png"))
            if config.unet_visualize.coord_per_frame:
                for t in range(0, nframe):
                    while True: 
                        try:
                            x_coords[t] = int(input(f"Enter x coordinate for frame {t} (0–512): "))
                            y_coords[t] = int(input(f"Enter y coordinate for frame {t} (0–512): "))
                            if 0 < x_coords[t] < 512 and 0 < y_coords[t] < 512: 
                                break
                            else: 
                                print("Invalid input. Please type valid values. ")
                        except ValueError:
                            print("Invalid input. Please type integer values. ")
            else:
                while True: 
                    try:
                        x_coord = int(input("Enter x coordinate (0–512): "))
                        y_coord = int(input("Enter y coordinate (0–512): "))
                        if 0 < x_coord < 512 and 0 < y_coord < 512: 
                            break
                        else: 
                            print("Invalid input. Please type valid values. ")
                    except ValueError:
                        print("Invalid input. Please type integer values. ")
                # set same coord for all targets
                for t in range(0, nframe):
                    x_coords[t] = x_coord
                    y_coords[t] = y_coord
        else:
            x_coord = random.randint(0, 511)
            y_coord = random.randint(0, 511)
            for t in range(0, nframe):
                x_coords[t] = x_coord
                y_coords[t] = y_coord 

        x_0 = unet_inference(batch, go_this_scene)
        if not go_this_scene: continue

        sample_root = os.path.join(outdir_root, f"sample_{idx}")
        os.makedirs(sample_root, exist_ok=True)

        coords = {"x": x_coords, "y": y_coords}    
        with open(os.path.join(sample_root, "coords.json"), "w") as f:
            json.dump(coords, f, indent=4)

        unet_attn_cache = cached['attn']
        images_original = images.clone()
        x_0 = einops.rearrange(x_0, "(b f) c h w -> b f c h w", b=B, f=V-cond_num)
        images[:, cond_num:] = x_0  # (1, V, C, H, W), replace noisy targets with denoised ones for visualization
        # dict: {layer_id(str): attn[B, H, Q(VHW), K(VHW)]}
        # save generated images
        if config.unet_visualize.save_stack:
            stacked = torch.cat([images_original, images], dim=-2).squeeze(0) # [V, C, 2H, W]
            stacked = einops.rearrange(stacked, "V C H W -> C H (V W)")
        save_image(stacked, os.path.join(sample_root, f"VIS_STACKED.png"))
        
        # ---------- per-target visualization loop ----------
        # Targets are frames [cond_num, nframe)
        if unet_attn_cache is not None:
            for unet_layer in caching_unet_attn_layers:
                stacked_images = []
                for t in range(0, nframe):
                    x_coord, y_coord = x_coords[t], y_coords[t]
                    tgt_gt_image = images_original.squeeze()[t]  # [C, H, W]  (this target)
                    print(f"[frame={t}] layer: {unet_layer} shape : {unet_attn_cache[str(unet_layer)].shape}")
                    unet_attn_logit = unet_attn_cache[str(unet_layer)]  # [B, H, Q, K]
                    Bc, Hh, Q, K = unet_attn_logit.shape

                    # Self-attn vs cross-attn handling stays the same,
                    # but query index is now the CURRENT TARGET t (not nframe-1).
                    if Bc == 3:  # (kept from your original heuristic)
                        query_size = int(math.sqrt(Q))
                        y_feat_cost = int((y_coord / 512) * query_size)
                        x_feat_cost = int((x_coord / 512) * query_size)
                        query_token_idx_cost = y_feat_cost * query_size + x_feat_cost

                        attn_maps = unet_attn_logit.mean(dim=1)  # [3, Q, K]
                        # Use the head-averaged attention from stream index == t if applicable.
                        # Original code indexed [0], [1], [2] explicitly. Keep compatibility:
                        # we’ll clamp t to [0, 2] here to avoid OOB if Bc==3 encodes 3 streams.
                        sidx = max(0, min(2, t))                 # <<< CHANGED >>>
                        all_scores = torch.softmax(attn_maps[sidx, query_token_idx_cost], dim=-1)
                        score = all_scores.reshape(query_size, query_size)  # visualize the chosen stream only
                    else:
                        # Cross-attention (common case): slice Q/K by frame dimensions
                        def slice_attnmap(attnmap, query_idx, key_idx):
                            B_, Head, Q_, K_ = attnmap.shape
                            F = nframe
                            HW = Q_ // F
                            attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2',
                                                    B=B_, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
                            attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
                            attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)',
                                                    B=B_, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
                            return attnmap

                        query_idx = [t]
                        if config.unet_visualize.cross_only:
                            key_idx = [i for i in range(nframe) if i not in query_idx]   # refs only
                        else:
                            key_idx = list(range(nframe))                                # refs + all frames

                        sliced = slice_attnmap(unet_attn_logit, query_idx=query_idx, key_idx=key_idx)  # [B, H, Q, K]
                        # We'll handle head-mean either before or after softmax depending on per_view.
                        # Keep full heads for correct per-view softmax on logits:
                        attn_maps = sliced[0]  # [Head, Q, K]

                        query_size = int(math.sqrt(attn_maps.shape[1]))
                        y_feat_cost = int((y_coord / 512) * query_size)
                        x_feat_cost = int((x_coord / 512) * query_size)
                        query_token_idx_cost = y_feat_cost * query_size + x_feat_cost

                        # logits for the chosen query token across all keys: [Head, K] where K = V * HW
                        head_logits = attn_maps[:, query_token_idx_cost]  # [Head, K]
                        tokens_per_img = query_size * query_size
                        V_keys = head_logits.shape[-1] // tokens_per_img

                        if config.unet_visualize.per_view:
                            # Softmax PER VIEW on logits (not on concatenated keys)
                            # Split into V chunks (each length HW), softmax per chunk, then concat.
                            chunks = head_logits.split(tokens_per_img, dim=-1)           # list of V tensors, each [Head, HW]
                            probs_chunks = [torch.softmax(c / 1.5, dim=-1) for c in chunks]    # each [Head, HW], softmax per view
                            # Head-mean after per-view softmax, then concat back to [K]
                            all_scores = torch.cat([pc.mean(dim=0) for pc in probs_chunks], dim=-1)  # [K]
                        else:
                            # Original behavior: one softmax over all concatenated keys, then head-mean
                            probs = torch.softmax(head_logits, dim=-1)  # [Head, K]
                            all_scores = probs.mean(dim=0)              # [K]

                        scores_split = all_scores.split(tokens_per_img)  # V tensors, each [HW]
                        # Concatenate attention maps horizontally for each key frame
                        score = torch.cat([s.reshape(query_size, query_size) for s in scores_split], dim=-1)
                        # score = torch.cat([F.interpolate(s.reshape(query_size, query_size).unsqueeze(0).unsqueeze(0),
                        #              size=(16, 16), mode='bilinear').squeeze(0).squeeze(0)
                        #             for s in scores_split], dim=-1)


                    combined_img = save_image_jinhk(tgt_gt_image, images, t, x_coord, y_coord, score)
                    stacked_images.append(combined_img)
                stacked_images = concat_vertical(stacked_images)
                stacked_images.save(os.path.join(sample_root, f"attn{unet_layer}.png"))

        print(f"Saved sample {idx}")
        if config.unet_visualize.idx_set:
            if idx >= end_idx:
                break
        if config.unet_visualize.idx_set is None and idx >= config.unet_visualize.end_data_idx:
            break
        

if __name__ == "__main__":
    # unet: (down_blocks(6), mid_block(1), up_blocks(9))
    # size: [64,64,32,32,16,16] [8] [16,16,16,32,32,32,64,64,64]
    # checkpoint
    config_file_path = 'configs/viz_full_perview.yaml'
    config = EasyDict(OmegaConf.load(config_file_path))
    # noise
    noise_timestep = config.unet_visualize.noise_timestep
    # data
    nframe = config.unet_visualize.nframe # View length
    cond_num = config.unet_visualize.cond_num # Condition among nframe
    inference_view_range = config.unet_visualize.inference_view_range # change if you want 
    caching_unet_attn_layers = config.unet_visualize.caching_unet_attn_layers
    #caching_unet_attn_layers = range(15)

    main(nframe, cond_num, inference_view_range, caching_unet_attn_layers, noise_timestep=noise_timestep,
         resume_checkpoint=config.unet_visualize.resume_checkpoint,
         config=config,
         config_file_path=config_file_path
         )