import argparse
import collections
import copy
import gc
import logging
import math
import os
import random
import shutil
import wandb
import accelerate
import cv2
import diffusers
import einops
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from easydict import EasyDict
from omegaconf import OmegaConf, ListConfig, DictConfig
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
from my_diffusers.training_utils import EMAModel
# from src.datasets.global_datasets import load_global_dataset
# from src.datasets.global_sampler import GlobalConcatSampler
from src.modules.camera import get_camera_embedding
from src.modules.position_encoding import depth_freq_encoding, global_position_encoding_3d, get_3d_priors
from src.modules.schedulers import get_diffusion_scheduler
from utils import get_lpips_score, _seq_name_to_seed

from src.distill_utils.attn_logit_head import cycle_consistency_checker
from src.distill_utils.attn_processor_cache import set_attn_cache, unset_attn_cache, pop_cached_attn, clear_attn_cache, \
                                                    set_feat_cache, unset_feat_cache, pop_cached_feat, clear_feat_cache
from src.modules.timestep_sample import truncated_normal

from src.datasets import shuffle_batch, uniform_push_batch
logger = get_logger(__name__, log_level="INFO")


# def resort_batch(batch, nframe, bsz):
#     sequence_name = np.array(batch["sequence_name"]).reshape(bsz, nframe)
#     indices = np.argsort(sequence_name, axis=1)
#     batch["image"] = einops.rearrange(batch["image"], "(b f) c h w -> b f c h w", f=nframe)
#     batch["intrinsic"] = einops.rearrange(batch["intrinsic"], "(b f) c1 c2 -> b f c1 c2", f=nframe)
#     batch["extrinsic"] = einops.rearrange(batch["extrinsic"], "(b f) c1 c2 -> b f c1 c2", f=nframe)

#     if "depth" in batch and batch["depth"] is not None:
#         batch["depth"] = einops.rearrange(batch["depth"], "(b f) c h w -> b f c h w", f=nframe)

#     for i in range(bsz):  # we do not need to sort megascenes and front3d
#         if batch["tag"][i * nframe] not in ("megascenes", "front3d"):
#             batch["image"][i] = batch["image"][i, indices[i]]
#             batch["intrinsic"][i] = batch["intrinsic"][i, indices[i]]
#             batch["extrinsic"][i] = batch["extrinsic"][i, indices[i]]
#             sequence_name[i] = sequence_name[i, indices[i]]

#             if "depth" in batch and batch["depth"] is not None:
#                 batch["depth"][i] = batch["depth"][i, indices[i]]

#     batch["image"] = einops.rearrange(batch["image"], "b f c h w -> (b f) c h w", f=nframe)
#     batch["intrinsic"] = einops.rearrange(batch["intrinsic"], "b f c1 c2 -> (b f) c1 c2", f=nframe)
#     batch["extrinsic"] = einops.rearrange(batch["extrinsic"], "b f c1 c2 -> (b f) c1 c2", f=nframe)
#     batch["sequence_name"] = list(sequence_name.reshape(-1))

#     if "depth" in batch and batch["depth"] is not None:
#         batch["depth"] = einops.rearrange(batch["depth"], "b f c h w -> (b f) c h w", f=nframe)

#     return batch


def slice_vae_encode(vae, image, sub_size):  # vae fails to encode large tensor directly, we need to slice it
    with torch.no_grad(), torch.autocast("cuda", enabled=True):
        if len(image.shape) == 5:  # [B,F,C,H,W]
            b, f, _, h, w = image.shape
            image = einops.rearrange(image, "b f c h w -> (b f) c h w")
        else:
            b, f, h, w = None, None, None, None

        if (image.shape[-1] > 256 and image.shape[0] > sub_size) or (image.shape[0] > 192):
            slice_num = image.shape[0] // sub_size
            if image.shape[0] % sub_size != 0:
                slice_num += 1
            latents = []
            for i in range(slice_num):
                latents_ = vae.encode(image[i * sub_size:(i + 1) * sub_size]).latent_dist.sample()
                latents.append(latents_)
            latents = torch.cat(latents, dim=0)
        else:
            latents = vae.encode(image).latent_dist.sample()

        if f is not None:
            latents = einops.rearrange(latents, "(b f) c h w -> b f c h w", f=f)

        return latents


def get_pipeline(accelerator, config, vae, unet, weight_dtype):
    scheduler = get_diffusion_scheduler(config, name="DDIM")
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline


# def get_show_images(input_images, pred_images, cond_num, depth=None):
#     pred_images = [pred_images[i] for i in range(pred_images.shape[0])]
#     pred_images = np.clip(np.concatenate(pred_images, axis=1) * 255, 0, 255).astype(np.uint8)  # [H,W*F,c]
#     input_images = (input_images + 1) / 2
    # ground_truths = np.concatenate([np.array(ToPILImage()(input_images[i])) for i in range(input_images.shape[0])], axis=1)
#     input_images[cond_num:] = 0
#     inputs = np.concatenate([np.array(ToPILImage()(input_images[i])) for i in range(input_images.shape[0])], axis=1)
#     if depth is not None:
#         min_vals = depth.amin(dim=[2, 3], keepdim=True)
#         max_vals = depth.amax(dim=[2, 3], keepdim=True)
#         depth = (depth - min_vals) / (max_vals - min_vals)
#         depth[cond_num:] = 0
#         depth = np.concatenate([np.array(ToPILImage()(depth[i]).convert("RGB")) for i in range(depth.shape[0])], axis=1)
#         show_image = np.concatenate([inputs, depth, ground_truths, pred_images], axis=0)
#     else:
#         show_image = np.concatenate([inputs, ground_truths, pred_images], axis=0)

#     return show_image

def get_show_images(input_images, pred_images, cond_num, depth=None):
    assert depth == None
    F, _, H, W = input_images.shape
    ground_truths = input_images.permute(1,2,0,3).reshape(3, H, F*W).clone()
    input_images[cond_num:] = 0
    input_images = input_images.permute(1,2,0,3).reshape(3, H, F*W)
    pred_images = pred_images.permute(1,2,0,3).reshape(3, H, F*W)

    show_image = torch.cat([input_images, ground_truths, pred_images], dim=1)

    return show_image   



@torch.no_grad()        
def log_test(accelerator, config, args, pipeline, dataloader, step, device,
             cond_num, nframe, cfg, num_inference_steps,
             compute_fid = False,
             viz_len = 0,
             process_batch_fn = lambda x: x,
             eval_size = 512,
             eval_w_mask = False,
             save_csv_path = None,
             test_dataset_name = "",
             **kwargs):
    from src.metrics import compute_psnr, compute_ssim, compute_lpips
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    from filelock import FileLock
    import pandas  as pd
    if eval_w_mask:    
        from warp import render_points_pytorch3d


    ''' Caution, batch=1 !
    '''
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    if compute_fid:
        if accelerator.is_main_process:
            fid_calculator = FrechetInceptionDistance(normalize=True).to(device)
            fid_calculator.reset()

    resize_fn = transforms.Resize(eval_size, interpolation=transforms.InterpolationMode.BICUBIC)
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    psnr_masked_scores = []
    psnr_unmasked_scores = []

    show_images = []
    # show_save_dict = collections.defaultdict(int)
    val_iter = 0

    log_prefix = f"test/{test_dataset_name}"
    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            batch = process_batch_fn(batch)
            # unbatchify
            # image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
            image, intri_, extri_ = batch['image'].to(device), batch['intrinsic'].to(device), batch['extrinsic'].to(device)
            if eval_w_mask: # make mask if necessary
                assert batch['point_map'].shape[2] == 3, f"point_map should be (B F 3 H W), but got {batch['point_map'].shape}"
                point_map = batch['point_map'] # B F 3 H W
                B, _, _, H, W = image.shape
                ref_point_map, tgt_extri_, tgt_intri_ = point_map[:, :cond_num], extri_[:,cond_num:], intri_[:,cond_num:]
                ref_point_map, tgt_extri_, tgt_intri_ = ref_point_map.to(device), tgt_extri_.to(device), tgt_intri_.to(device)
                ref_pcd = ref_point_map.permute(0,1,3,4,2).reshape(B,-1,3)
                ones = torch.ones((B, ref_pcd.shape[1],1), device=device, dtype=ref_pcd.dtype)
                mask = [] # 1: visible region, 0: non-visible(occlusion) region
                for i in range(nframe - cond_num):
                    warped, _ = render_points_pytorch3d(
                                            ref_pcd.clone(), ones, 
                                            tgt_intri_[:,i],
                                            tgt_extri_[:,i],
                                            render_resolution=(H,W),
                                            ndc_radius = None, # automatic scale
                                            background=-1000.
                                        )
                    warped = warped.permute(0,2,3,1) # B C h w -> B h w C
                    mask.append(torch.all(warped != -1000., dim=-1)) # B h w 
                mask = torch.stack(mask, dim=1) # B f h w
            b, f, _, h, w = image.shape
            assert f == nframe, f"args.test_nframe({args.test_nframe}) doesn't match with data nframe({f}), image_shape({image.shape}) "
            image_normalized = image * 2.0 - 1.0

            extrinsic, intrinsic = extri_, intri_
            tag = batch.get("tag", ['others'] * b )
            sequence_name, depth = None, None
            preds = pipeline.batch_call(images=image_normalized, nframe=nframe, cond_num=cond_num,
                             height=image_normalized.shape[-2], width=image_normalized.shape[-1],
                             intrinsics=intrinsic, extrinsics=extrinsic,
                             num_inference_steps=num_inference_steps, guidance_scale=cfg,
                             output_type="pt", config=config, tag=tag,
                             sequence_name=sequence_name, depth=depth, vae=kwargs['vae']).images  # [bf,c,h,w]
            color_warps = None
            
            # if batch['tag'][0] not in show_save_dict or show_save_dict[batch['tag'][0]] < 10:  # 每个dataset显示10个
                # show_save_dict[batch['tag'][0]] += 1
            if val_iter < viz_len:
                image_viz = einops.rearrange(image, 'b f c h w -> (b f) c h w').clone()
                preds_viz = preds.clone()
                if depth is not None:
                    show_image = get_show_images(image_viz, preds_viz, cond_num, batch["depth"])
                else:
                    show_image = get_show_images(image_viz, preds_viz, cond_num)

                if color_warps is not None:
                    h, w = image.shape[-2], image.shape[-1]
                    show_image[h:h * 2, cond_num * w:] = color_warps[0][:, cond_num * w:]
                show_images.append(show_image)

            # Extract target views
            gt_images = einops.rearrange(image[:, cond_num:], 'b f c h w -> (b f) c h w')
            preds = einops.rearrange(preds, '(b f) c h w -> b f c h w', f= nframe)[:, cond_num:].flatten(0,1) # bf c h w
            if eval_w_mask: mask = einops.rearrange(mask, 'b f h w -> (b f) h w')

            if eval_size != h:
                gt_images = resize_fn(gt_images)
                preds = resize_fn(preds)
                if eval_w_mask: mask = resize_fn(mask.unsqueeze(1).float()).bool().squeeze(1)
            if compute_fid:
                # gt_imgs_fid, preds_fid = torch.tensor(gt_images).permute(0, 3, 1, 2).to(device),  torch.tensor(preds).permute(0, 3, 1, 2).to(device)
                gt_imgs_fid, preds_fid = accelerator.gather(gt_images), accelerator.gather(preds)
                if accelerator.is_main_process:
                    gt_imgs_fid = einops.rearrange(gt_imgs_fid,'... c h w -> (...) c h w').to(device)
                    preds_fid = einops.rearrange(preds_fid,'... c h w -> (...) c h w').to(device)
                    fid_calculator.update(gt_imgs_fid, real=True)
                    fid_calculator.update(preds_fid, real=False)

            psnr_scores += [compute_psnr(gt_images, preds)]
            ssim_scores += [compute_ssim(gt_images, preds)]
            lpips_scores += [compute_lpips(gt_images, preds)]

            if eval_w_mask:
                psnr_masked_ = []
                for i in range(preds.shape[0]):
                    p = compute_psnr(gt_images[i].permute(1,2,0)[mask[i]][None,...].float(), preds[i].permute(1,2,0)[mask[i]][None,...].float())
                    psnr_masked_.append(p)
                psnr_masked_scores.append(torch.cat(psnr_masked_))
                for i in range(preds.shape[0]):
                    p = compute_psnr(gt_images[i].permute(1,2,0)[~mask[i]][None,...].float(), preds[i].permute(1,2,0)[~mask[i]][None,...].float())
                    psnr_unmasked_scores.append(p)
            val_iter += 1
    
    # unify all results
    log_value_dict = {}
    # psnr_score = psnr_metric.compute()
    # ssim_score = ssim_metric.compute()
    # lpips_score = lpips_metric.compute()
    psnr_scores = torch.cat(psnr_scores)
    ssim_scores = torch.cat(ssim_scores)
    lpips_scores = torch.cat(lpips_scores)

    psnr_scores = accelerator.gather(psnr_scores).mean().item()
    ssim_scores = accelerator.gather(ssim_scores).mean().item()
    lpips_scores = accelerator.gather(lpips_scores).mean().item()
    log_value_dict.update({f"psnr": psnr_scores, f"ssim": ssim_scores, f"lpips": lpips_scores})
    accelerator.print("test/psnr: ", psnr_scores)
    accelerator.print("test/ssim: ", ssim_scores)
    accelerator.print("test/lpips: ", lpips_scores)
    if eval_w_mask:
        psnr_masked_scores = torch.cat(psnr_masked_scores)
        # psnr_masked_score = sum(psnr_masked_scores) / len(psnr_masked_scores)
        psnr_masked_scores = accelerator.gather(psnr_masked_scores)
        psnr_masked_scores = psnr_masked_scores[~torch.isnan(psnr_masked_scores)]
        psnr_masked_scores = psnr_masked_scores.mean().item() # avg except nan
        log_value_dict.update({f"psnr_masked": psnr_masked_scores})
        accelerator.print("test/psnr_masked: ", psnr_masked_scores)

        psnr_unmasked_scores = torch.cat(psnr_unmasked_scores)
        psnr_unmasked_scores = accelerator.gather(psnr_unmasked_scores)
        psnr_unmasked_scores = psnr_unmasked_scores[~torch.isnan(psnr_unmasked_scores)]
        psnr_unmasked_scores = psnr_unmasked_scores.mean().item() # avg except nan
        log_value_dict.update({f"psnr_unmasked": psnr_unmasked_scores})
        accelerator.print("test/psnr_unmasked: ", psnr_unmasked_scores)

    if compute_fid:
        if accelerator.is_main_process:
            print("fid compute start")
            import time
            st = time.time()
            fid_val = fid_calculator.compute()
            real_num, fake_num = fid_calculator.real_features_num_samples, fid_calculator.fake_features_num_samples
            fid_calculator.reset()
            del fid_calculator
            torch.cuda.empty_cache()
            print(f"fid compute end: {time.time() - st}" )
            accelerator.print("test/fid: ", fid_val)
            accelerator.print("test/fid_real_num: ", real_num)
            accelerator.print("test/fid_fake_num: ", fake_num)

            log_value_dict.update({f"fid": fid_val, f"fid_real_num": real_num, f"fid_fake_num": fake_num})
        
    
    wandb_log = {f"{log_prefix}/{k}": v for k,v in log_value_dict.items()}
    accelerator.log(wandb_log, step=step)

    show_images_full = torch.cat(show_images, dim=1).to(device)
    show_images_full = accelerator.gather(show_images_full).reshape(-1,*show_images_full.shape).permute(1,0,2,3).flatten(1,2)
    if accelerator.is_main_process:
        # for j in range(len(show_image)):
        #     if config.image_size > 256:
        #         show_images[j] = cv2.resize(show_images[j], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # accelerator.log({f"test/gt_masked_pred_images{j}": wandb.Image(show_images[j])}, step=step)
        accelerator.log({f"{log_prefix}/viz": wandb.Image(show_images_full)}, step=step)
        if save_csv_path is not None:
            data = {
                'run_name': args.run_name,
                'cond_num': cond_num,
                'dataset_full' : test_dataset_name,
                'dataset': test_dataset_name.split("_")[0],
                'view_range': test_dataset_name.split('_')[-1],
                "train_step" : step
            }
            data.update(log_value_dict)
            data = pd.DataFrame([data])
            lock = FileLock(save_csv_path +".lock")
            with lock:    
                if not os.path.exists(save_csv_path):
                    data.to_csv(save_csv_path, mode="w", header=True, index=False)
                else:
                    data.to_csv(save_csv_path, mode="a", header=False, index=False)     



    del loss_fn_alex
    torch.cuda.empty_cache()

    return lpips_scores

# @torch.no_grad()
# def log_validation(accelerator, config, args, pipeline, val_dataloader, step, device, log_prefix = 'val',**kwargs):
#     ''' Caution, batch=1 !
#     '''
#     if accelerator.is_main_process:
#         logger.info(f"Validation log in step {step}")

#     loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

#     manual_val_len = config.get("val_manual_len", None)
#     compute_fid = config.get("val_compute_fid", False) 
#     viz_len = config.get("val_viz_len", 30)
#     if compute_fid:
#         if accelerator.is_main_process:
#             from torchmetrics.image.fid import FrechetInceptionDistance
#             fid_calculator = FrechetInceptionDistance(normalize=True).to(device)
#             fid_calculator.reset()

#     if manual_val_len is not None:
#         assert manual_val_len % (accelerator.num_processes * 1) == 0
#         manual_val_step = manual_val_len // (accelerator.num_processes * 1)

#     psnr_scores = []
#     ssim_scores = []
#     lpips_scores = []
#     show_images = []
#     # show_save_dict = collections.defaultdict(int)
#     val_iter = 0

#     cond_num = config.val_cond_num
#     with torch.no_grad(), torch.autocast("cuda"):
#         for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):
#             batch = uniform_push_batch(batch, cond_num)
#             # unbatchify
#             image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
#             image_normalized = image * 2.0 - 1.0

#             extrinsic, intrinsic = extri_, intri_
#             tag, sequence_name, depth = None, None , None
#             preds = pipeline(images=image_normalized, nframe=config.val_nframe, cond_num=cond_num,
#                              height=image_normalized.shape[2], width=image_normalized.shape[3],
#                              intrinsics=intrinsic, extrinsics=extrinsic,
#                              num_inference_steps=50, guidance_scale=args.val_cfg,
#                              output_type="np", config=config, tag=tag,
#                              sequence_name=sequence_name, depth=depth, vae=kwargs['vae']).images  # [f,h,w,c]

#             if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
#                 color_warps = global_position_encoding_3d(config, depth, batch['intrinsic'],
#                                                           batch['extrinsic'], 1,
#                                                           nframe=config.val_nframe, device=accelerator.device,
#                                                           pe_scale=1 / 8,
#                                                           embed_dim=config.model_cfg.get("coord_dim", 192),
#                                                           colors=image)
#             else:
#                 color_warps = None

#             # if batch['tag'][0] not in show_save_dict or show_save_dict[batch['tag'][0]] < 10:  # 每个dataset显示10个
#                 # show_save_dict[batch['tag'][0]] += 1
#             if val_iter < viz_len:
#                 if depth is not None:
#                     show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num, batch["depth"])
#                 else:
#                     show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num)

#                 if color_warps is not None:
#                     h, w = image.shape[2], image.shape[3]
#                     show_image[h:h * 2, cond_num * w:] = color_warps[0][:, cond_num * w:]
#                 show_images.append(show_image)

#             gt_images = (image_normalized[cond_num:].permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 # -1 1 → 0 1
#             preds = preds[cond_num:]

#             if compute_fid:
#                 gt_imgs_fid, preds_fid = torch.tensor(gt_images).permute(0, 3, 1, 2).to(device),  torch.tensor(preds).permute(0, 3, 1, 2).to(device)
#                 gt_imgs_fid, preds_fid = accelerator.gather(gt_imgs_fid), accelerator.gather(preds_fid)
#                 if accelerator.is_main_process:
#                     fid_calculator.update(einops.rearrange(gt_imgs_fid,'... c h w -> (...) c h w'), real=True)
#                     fid_calculator.update(einops.rearrange(preds_fid,'... c h w -> (...) c h w'), real=False)

#             for i in range(preds.shape[0]):
#                 psnr_ = peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0)
#                 psnr_scores.append(psnr_)
#                 ssim_ = structural_similarity(cv2.cvtColor(gt_images[i], cv2.COLOR_RGB2GRAY),
#                                               cv2.cvtColor(preds[i], cv2.COLOR_RGB2GRAY), data_range=1.0)
#                 ssim_scores.append(ssim_)
#                 lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device)
#                 lpips_scores.append(lpips_)
            
#             val_iter += 1 
#             if manual_val_len is not None:
#                 if val_iter >= manual_val_step:
#                     break
    
#     # unify all results
#     psnr_score = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
#     ssim_score = torch.tensor(np.mean(ssim_scores), device=device, dtype=torch.float32)
#     lpips_score = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)

#     psnr_score = accelerator.gather(psnr_score).mean().item()
#     ssim_score = accelerator.gather(ssim_score).mean().item()
#     lpips_score = accelerator.gather(lpips_score).mean().item()
#     if compute_fid:

#         if accelerator.is_main_process:
#             print("fid compute start")
#             fid_val = fid_calculator.compute()
#             real_num, fake_num = fid_calculator.real_features_num_samples, fid_calculator.fake_features_num_samples
#             fid_calculator.reset()
#             del fid_calculator
#             torch.cuda.empty_cache()
#             print("fid compute end")
#             accelerator.log({f"{log_prefix}/fid": fid_val, f"{log_prefix}/fid_real_num": real_num, f"{log_prefix}/fid_fake_num": fake_num}, step=step)
#             # accelerator.log({"val/fid": fid_val, "val/fid_real_num": real_num, "val/fid_fake_num": fake_num}, step=step)

#     accelerator.log({f"{log_prefix}/psnr": psnr_score, f"{log_prefix}/ssim": ssim_score, f"{log_prefix}/lpips": lpips_score}, step=step)
#     # accelerator.log({"val/psnr": psnr_score, "val/ssim": ssim_score, "val/lpips": lpips_score}, step=step)

    
#     show_images_full = torch.cat(show_images, dim=1).to(device)
#     show_images_full = accelerator.gather(show_images_full).reshape(-1,*show_images_full.shape).permute(1,0,2,3).flatten(1,2)
#     if accelerator.is_main_process:
#         # for j in range(len(show_image)):
#         #     if config.image_size > 256:
#         #         show_images[j] = cv2.resize(show_images[j], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#             # accelerator.log({f"val/gt_masked_pred_images{j}": wandb.Image(show_images[j])}, step=step)
#         accelerator.log({f"{log_prefix}/gt_masked_pred_images": wandb.Image(show_images_full)}, step=step)
#         # accelerator.log({"val/show_images": wandb.Image(show_images_full)}, step=step)

#     del loss_fn_alex
#     torch.cuda.empty_cache()

#     return lpips_score


def log_train(accelerator, config, args, pipeline, weight_dtype, batch, step, **kwargs):
    logger.info(f"Train log in step {step}")
    nframe = kwargs['nframe']

    # image = batch["image"][:nframe]  # only show one group
    # intrinsic = batch["intrinsic"][:nframe]
    # extrinsic = batch["extrinsic"][:nframe]
    # tag = batch["tag"][:nframe]
    # sequence_name = batch["sequence_name"][:nframe]

    # batch = uniform_push_batch(batch, kwargs.get("random_cond_num", 1))
    image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
    image_normalized = image * 2.0 - 1.0
    if extri_.shape[-2] == 3:
        f = image.shape[0]
        new_extri_ = torch.zeros((f, 4, 4), device=extri_.device, dtype=extri_.dtype)
        new_extri_[:, :3, :4] = extri_
        new_extri_[:, 3, 3] = 1.0
        extri_ = new_extri_

    extrinsic, intrinsic = extri_, intri_
    tag = [ batch.get("tag", [None])[0] ]
    sequence_name = None

    if config.model_cfg.get("enable_depth", False):
        depth = batch["depth"][:nframe].to(torch.float32)
        depth_freq = config.model_cfg.get("depth_freq", None)
        if depth_freq is not None:
            depth = depth_freq_encoding(depth, device=kwargs['device'], embed_dim=depth_freq)
    else:
        depth = None

    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        preds = pipeline(images=image_normalized, nframe=nframe, cond_num=kwargs.get("random_cond_num", 1),
                         height=image_normalized.shape[2], width=image_normalized.shape[3],
                         intrinsics=intrinsic, extrinsics=extrinsic,
                         num_inference_steps=50, guidance_scale=args.val_cfg,
                         output_type="np", config=config, tag=tag, sequence_name=sequence_name,
                         depth=depth, vae=kwargs['vae']).images  # [f,h,w,c]

        if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
            color_warps = global_position_encoding_3d(config, depth, intrinsic,
                                                      extrinsic, 1,
                                                      nframe=nframe, device=accelerator.device,
                                                      pe_scale=1 / 8,
                                                      embed_dim=config.model_cfg.get("coord_dim", 192),
                                                      colors=image_normalized)
        else:
            color_warps = None

    if depth is not None:
        show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), kwargs.get("random_cond_num", 1), batch["depth"][:nframe])
    else:
        show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), kwargs.get("random_cond_num", 1))

    if color_warps is not None:
        h, w = image.shape[2], image.shape[3]
        show_image[h:h * 2, kwargs.get("random_cond_num", 1) * w:] = color_warps[0][:, kwargs.get("random_cond_num", 1) * w:]

    # tracker = accelerator.get_tracker("wandb", unwrap=True)
    # tracker.add_images("train/gt_masked_pred_images", show_image, step, dataformats="HWC")
    accelerator.log({"train/gt_masked_pred_images": wandb.Image(show_image), }, step=step)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="MV-Gen",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--train_log_interval", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--val_cfg", type=float, default=1.0)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--restart_global_step", default=0, type=int)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--min_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay_step", type=int, default=10)
    parser.add_argument("--reset_ema_step", action="store_true")
    parser.add_argument("--only_resume_weight", action="store_true")
    parser.add_argument("--max_nframe", type=int, default=None)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_at_first", action="store_true")
    parser.add_argument("--val_w_mask", action="store_true")

    parser.add_argument("--num_workers_per_gpu", type=int, default=1, help="dataloader num_workers per GPU")

    parser.add_argument("--checkpointing_last_steps", type=int, default=5000 )
    parser.add_argument("--resume_from_last", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if (args.resume_from_checkpoint is not None and args.resume_path is None) or args.resume_from_last:
        config = EasyDict(OmegaConf.load(os.path.join(args.output_dir, "config.yaml")))
        cfg = dict()
        # for data_name in config.dataset_names:
        #     data_name = data_name.replace('_easy', '')
        #     cfg[data_name] = EasyDict(OmegaConf.load(os.path.join(args.output_dir, f"{data_name}.yaml")))
    else:
        config = EasyDict(OmegaConf.load(args.config_file))
        cfg = None

    # Sanity checks
    # assert config.dataset_names is not None and len(config.dataset_names) > 0

    # dynamic nframe training
    if args.max_nframe is not None and args.max_nframe > config.nframe:
        config.dynamic_nframe = True
        config.origin_nframe = config.nframe
        config.min_nframe = config.nframe
        config.max_nframe = args.max_nframe
        config.nframe = args.max_nframe
    else:
        config.dynamic_nframe = False

    return args, config, cfg


def main():
    args, config, cfg = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        # save configs
    if accelerator.is_main_process:
        OmegaConf.save(dict(config), os.path.join(args.output_dir, 'config.yaml'))
        # for data_name in cfg:
        #     OmegaConf.save(dict(cfg[data_name]), os.path.join(args.output_dir, f'{data_name}.yaml'))

    # Load scheduler, tokenizer and models.
    train_noise_scheduler = get_diffusion_scheduler(config, name="DDPM")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    accelerator.print("Loading model weights...")
    # take text_encoder and vae away from parameter sharding across multi-gpu in ZeRO
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(f"{config.pretrained_model_name_or_path}", subfolder="vae")
        vae.requires_grad_(False)


    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                rank=accelerator.process_index,
                                                model_cfg=config.model_cfg,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)
    unet.train()

    do_attn_distill = len(config.distill_config.distill_pairs) > 0 if hasattr(config, "distill_config") else False
    if config.vggt_on_fly:
        vggt_distill_config = {
            "cache_attn_layer_ids": list(set([p[1] for p in config.distill_config.distill_pairs if isinstance(p[1], int)])),
            "cache_costmap_types": list(set([p[1] for p in config.distill_config.distill_pairs if isinstance(p[1], str)]))
        }
        from vggt.models.vggt import VGGT
        # with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_distill_config).eval()
        for p in vggt_model.parameters():
            p.requires_grad = False
    else:
        vggt_model = None
        assert not config.get("use_vggt_camera", False), "use_vggt_camera is only available when vggt_on_fly is set"
    
    if config.get("dino_on_fly", False):
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for p in dino_model.parameters():
            p.requires_grad = False
        dino_model.eval()
    else:
        dino_model = None
    
    if do_attn_distill:
        distill_config = config.distill_config
        distill_pairs = config.distill_config.distill_pairs

        # Cost Metric
        from src.distill_utils.query_key_cost_metric import COST_METRIC_FN
        distill_cost_fn_list = [COST_METRIC_FN[c.lower()] for c in distill_config.cost_metric]
        # distill_cost_fn = COST_METRIC_FN[distill_config.cost_metric.lower()]
        
        from src.distill_utils.attn_distill_loss import ATTN_LOSS_FN
        from src.distill_utils.attn_distill_loss import cosine_loss_weight_scheduler
        distill_loss_fn_list = [ATTN_LOSS_FN[l.lower()] for l in distill_config.distill_loss_fn]
        distill_loss_weight_list = distill_config.distill_loss_weight
        distill_loss_fn_config_list = distill_config.get('distill_loss_fn_config', [{} for _ in range(len(distill_loss_fn_list))])
    
        # distill_loss_fn = ATTN_LOSS_FN[distill_config.distill_loss_fn.lower()]
        # distill_loss_weight = config.distill_config.distill_loss_weight
        # distill_loss_fn_config = config.distill_config.get('distill_loss_fn_config', {})

        # caching target : attn map vs qk
        from src.distill_utils.feat_head import FEAT_HEAD_CLS
        from src.distill_utils.attn_processor_cache import SDXL_ATTN_HEAD_NUM, SDXL_ATTN_DIM
        unet_query_head, unet_key_head = nn.ModuleDict(), nn.ModuleDict()
        qk_caching_unet_layers, attn_caching_unet_layers = [], []
        for i, (unet_layer, _) in enumerate(distill_pairs):
            if distill_config.unet_query_head[i].lower() == 'identity' and distill_config.unet_key_head[i].lower() == 'identity':
                attn_caching_unet_layers += [str(unet_layer)] # use computed attnmap instead of qk
            else:
                qk_caching_unet_layers += [str(unet_layer)]
                unet_layer_config = {"in_head_num" : SDXL_ATTN_HEAD_NUM[int(unet_layer)], "in_dim" : SDXL_ATTN_DIM[int(unet_layer)] // SDXL_ATTN_HEAD_NUM[int(unet_layer)] }
                unet_query_head[str(unet_layer)] = FEAT_HEAD_CLS[distill_config.unet_query_head[i].lower()](**distill_config.unet_query_head_kwargs[i], **unet_layer_config) 
                unet_key_head[str(unet_layer)] = FEAT_HEAD_CLS[distill_config.unet_key_head[i].lower()](**distill_config.unet_key_head_kwargs[i], **unet_layer_config)
        
        # logit heads: input: [B, Head, Q, K]
        from src.distill_utils.attn_logit_head import LOGIT_HEAD_CLS 
        from src.distill_utils.attn_processor_cache import SDXL_ATTN_HEAD_NUM
        unet_logit_head = nn.ModuleDict()
        vggt_logit_head = nn.ModuleDict()
        for i, (unet_layer, vggt_layer) in enumerate(distill_pairs):
            unet_layer_config = {"in_head_num" : SDXL_ATTN_HEAD_NUM[int(unet_layer)]}
            unet_logit_head[str(unet_layer)] = LOGIT_HEAD_CLS[distill_config.unet_logit_head[i].lower()](**distill_config.unet_logit_head_kwargs[i], **unet_layer_config) 
            vggt_logit_head[str(vggt_layer)] = LOGIT_HEAD_CLS[distill_config.vggt_logit_head[i].lower()](**distill_config.vggt_logit_head_kwargs[i])
            
        # add to unet (only single module supported in deepspeed...)
        unet.unet_query_head = unet_query_head
        unet.unet_key_head = unet_key_head
        del unet_query_head, unet_key_head
        unet.unet_logit_head = unet_logit_head
        unet.vggt_logit_head = vggt_logit_head
        del vggt_logit_head, unet_logit_head
        

    else:
        distill_pairs = []
        distill_cost_fn = None
        distill_loss_fn = None
        distill_loss_weight = 0.0
        distill_loss_fn_config = {}
        
            
    # set feat cache for repa distill # JIHO TODO: repa     
    if config.get('repa_config', None) is not None:
        do_repa = len(config.repa_config.distill_layers) > 0
    else:
        do_repa = False

    if do_repa:
        assert config.vggt_on_fly == True or config.get("dino_on_fly", False), "use vggt dinov2 for now"
        from src.distill_utils.attn_logit_head import build_mlp
        from src.distill_utils.attn_processor_cache import SDXL_ATTN_DIM 
        repa_feat_head = nn.ModuleDict()
        for unet_layer in config.repa_config.distill_layers:
            unet_feat_dim = SDXL_ATTN_DIM[int(unet_layer)]
            repa_feat_head[str(unet_layer)] = build_mlp(unet_feat_dim, 768, mlp_ratio = None,mlp_depth=1,  mid_dim=2048) # following repa
        unet.repa_feat_head = repa_feat_head
        del repa_feat_head
        






    if args.use_ema:
        ema_unet = copy.deepcopy(unet)
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=True,
            decay=args.ema_decay,
            min_decay=args.min_decay,
            ema_decay_step=args.ema_decay_step
        )
    else:
        ema_unet = None

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.opt_cfg.scale_lr:
        config.opt_cfg.learning_rate = (
                config.opt_cfg.learning_rate * config.gradient_accumulation_steps * ( config.train_batch_size * config.nframe / config.opt_cfg.scale_lr_base_batch_size )* accelerator.num_processes
        )
    # set trainable parameters
    trainable_params = []
    sub_modules = ["unet_query_head", "unet_key_head", "unet_logit_head", "vggt_logit_head", "repa_feat_head"]
    params = [p for n, p in unet.named_parameters() if p.requires_grad and not any([sn in n for sn in sub_modules])]
    trainable_params.append({'params': params, 'lr': config.opt_cfg.learning_rate})
    print("Trainable unet parameters: ", sum([p.numel() for p in params]))
    if do_attn_distill:
        head_params = [p for n, p in unet.named_parameters() if any([sn in n for sn in sub_modules])]
        print("Trainable distill head parameters: ", sum([p.numel() for p in head_params]))
        trainable_params.append({'params': head_params, 'lr': config.distill_config.get("learning_rate", 1e-4)})

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.opt_cfg.learning_rate,
        betas=(config.opt_cfg.adam_beta1, config.opt_cfg.adam_beta2),
        weight_decay=config.opt_cfg.adam_weight_decay,
        eps=config.opt_cfg.adam_epsilon,
    )

    # Dataset
    from src import datasets
    cls_name, d_kwargs = config.train_dataset.cls_name, config.train_dataset.config
    d_kwargs.batch_size = config.train_batch_size # Caution! Apply batch size to dataset code instead of loader
    train_dataset = datasets.__dict__[cls_name](num_viewpoints=config.nframe, **d_kwargs)

    val_datasets_dict = {}
    for data_name, data_config in config.val_datasets.items():
        cls_name, d_kwargs = data_config.cls_name, data_config.config
        dataset = datasets.__dict__[cls_name]( num_viewpoints = config.val_nframe, **d_kwargs)
        val_datasets_dict[data_name] = dataset

    print("Creating data loaders...")
    import webdataset as wds
    train_dataloader = wds.WebLoader(
        train_dataset,
        shuffle=False, # jiho TODO
        sampler=None,
        # batch_size=config.train_batch_size, # batch_size directly applied to webdataset 
        batch_size=None,
        num_workers= args.num_workers_per_gpu,
        # num_workers=accelerator.num_processes * args.num_workers_per_gpu, # keep num_workers_per_gpu = 1 to avoid memory error
        # persistent_workers=True, # not activated to avoid memory error
        # pin_memory=True, # not activated to avoid memory error
        # prefetch_factor=2, # not activated to avoid memory error
    )
    val_dataloaders_dict = {}
    for data_name, val_dataset in val_datasets_dict.items():
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            sampler=None,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            # num_workers=accelerator.num_processes,
        )
        val_dataloaders_dict[data_name] = val_dataloader

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.opt_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.opt_cfg.lr_warmup_steps * accelerator.num_processes if args.resume_from_checkpoint is None else 0,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return [total_num, trainable_num]

    param_info_vae = get_parameter_number(vae)
    accelerator.print(f'########## VAE, Total:{param_info_vae[0] / 1e6}M, Trainable:{param_info_vae[1] / 1e6}M ##################')
    param_info_unet = get_parameter_number(unet)
    accelerator.print(f'########## Unet, Total:{(param_info_unet[0]) / 1e6}M, Trainable:{(param_info_unet[1]) / 1e6}M ##################')
    

    global_step = 0
    first_epoch = 0
    if args.only_resume_weight:
        if accelerator.is_main_process:
            print("Loading weight only...")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            resume_path = args.resume_path
        else:
            raise NotImplementedError
        # else:
        #     # Get the most recent checkpoint
        #     resume_path = args.resume_path if args.resume_path is not None else args.output_dir
        #     dirs = os.listdir(resume_path)
        #     dirs = [d for d in dirs if d.startswith("checkpoint")]
        #     dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        #     path = dirs[-1] if len(dirs) > 0 else None

        if args.use_ema:
            if os.path.exists(f"{resume_path}/{path}/ema_unet.pt"):
                print("Find ema weights, load it!")
                weights = torch.load(f"{resume_path}/{path}/ema_unet.pt", map_location="cpu")
                # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
                unet.load_state_dict(weights, strict=False)  # unet load first
            else:
                print("No ema weights, load original weights instead!")
                weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
                # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
                unet.load_state_dict(weights['module'], strict=False)  # unet load first
            ema_unet.load_state_dict({"shadow_params": [p.clone().detach() for p in list(unet.parameters())]})
            if not args.reset_ema_step:
                ema_params = torch.load(f"{resume_path}/{path}/ema_unet_params.pt", map_location="cpu")
                ema_unet.optimization_step = ema_params['optimization_step']

        # reload weights for unet here
        weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
        unet.load_state_dict(weights['module'], strict=False)  # we usually need to resume partial weights here
        
        first_epoch = global_step // num_update_steps_per_epoch
        global_step = int(path.split("-")[1])
    
    if args.resume_from_last:
        assert args.only_resume_weight == False, "resume weight is not compatible with resume from last"
        resume_path = args.output_dir
        path = "checkpoint-last"
        if args.use_ema:
            if os.path.exists(f"{resume_path}/{path}/ema_unet.pt"):
                print("Find ema weights, load it!")
                weights = torch.load(f"{resume_path}/{path}/ema_unet.pt", map_location="cpu")
                # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
                unet.load_state_dict(weights, strict=False)  # unet load first
            else:
                print("No ema weights, load original weights instead!")
                weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
                # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
                unet.load_state_dict(weights['module'], strict=False)  # unet load first
            ema_unet.load_state_dict({"shadow_params": [p.clone().detach() for p in list(unet.parameters())]})
            if not args.reset_ema_step:
                ema_params = torch.load(f"{resume_path}/{path}/ema_unet_params.pt", map_location="cpu")
                ema_unet.optimization_step = ema_params['optimization_step']
        # reload weights for unet here
        weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
        unet.load_state_dict(weights['module'], strict=False)  # we usually need to resume partial weights here

        first_epoch = global_step // num_update_steps_per_epoch
        step_file = os.path.join(resume_path, path, "global_step.txt")
        with open(step_file, "r", encoding="utf-8") as f:
            global_step = int(f.read().strip())

    # Prepare 
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.train_batch_size
    # unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)  # train_dataloader
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)  # train_dataloader
    # match lr
    for i in range(global_step):
        lr_scheduler.step()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        logger.info("MixedPrecision: FP16 training!")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        logger.info("MixedPrecision: BF16 training!")



    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if config.vggt_on_fly:
        vggt_model.to(accelerator.device, dtype=weight_dtype)
    if config.get("dino_on_fly", False):
        dino_model.to(accelerator.device, dtype=weight_dtype)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            init_kwargs={"wandb": {"name": args.run_name}} if args.run_name is not None else {},
        )

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    # Not used currently
    # # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #         resume_path = args.resume_path
    #     else:
    #         # Get the most recent checkpoint
    #         resume_path = args.resume_path if args.resume_path is not None else args.output_dir
    #         dirs = os.listdir(resume_path)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
    #         args.resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:

    #         if not args.only_resume_weight:
    #             accelerator.print(f"Resuming from checkpoint {path}")
    #             accelerator.load_state(os.path.join(resume_path, path))

    #         if args.restart_global_step != 0:
    #             global_step = args.restart_global_step
    #         else:
    #             global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch

    #         if args.use_ema and not args.only_resume_weight:
    #             # here we load state because we need to restore all ema states, while weights are not changed
    #             temp_unet = copy.deepcopy(accelerator.unwrap_model(unet)).cpu()
    #             if os.path.exists(f"{resume_path}/{path}/ema_unet.pt"):
    #                 print("Find ema weights, load it!")
    #                 weights = torch.load(f"{resume_path}/{path}/ema_unet.pt", map_location="cpu")
    #                 temp_unet.load_state_dict(weights)
    #                 if not args.reset_ema_step:
    #                     ema_params = torch.load(f"{resume_path}/{path}/ema_unet_params.pt", map_location="cpu")
    #                     ema_unet.optimization_step = ema_params['optimization_step']
    #             else:
    #                 print("No ema weights, load original weights instead!")
    #                 weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
    #                 temp_unet.load_state_dict(weights['module'])
    #             ema_unet.load_state_dict({"shadow_params": [p.clone().detach() for p in list(temp_unet.parameters())]})
    #             del temp_unet
    # else:
    #     initial_global_step = 0

    if args.use_ema:
        ema_unet.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    device = unet.device
    best_metric = 1000
    first_batch = True
    
    import time
    
    for epoch in range(first_epoch, config.num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        end_time = time.time()
        for step, batch in enumerate(train_dataloader):
            # # TODO: remove (for data debug)
            # batch_time = time.time() - end_time #TODO: log?
            # print(f'batch key{step}: {batch_time:.3f}', batch['__key__'])
            # accelerator.log({"train/batch_time": batch_time}, step=global_step)
            # global_step += 1
            # progress_bar.update(1)
            # end_time = time.time()
            # continue
            batch_time = time.time() - end_time #TODO: log?
            unet.train()
            with accelerator.accumulate(unet):
                # 1. Prepare Data
                
                #  --- View Order Selection ---
                # 1) Random nframe (full)
                if config.dynamic_nframe: # jiho TODO: other data too
                    random_nframe = random.Random(step).randint(config.min_nframe, config.max_nframe)
                    batch['image'] = batch["image"][:, :random_nframe]  #  0 1 tensor [B,F,3,H,W]
                    batch['intrinsic'] = batch['intrinsic'][:, :random_nframe]
                    batch['extrinsic'] = batch['extrinsic'][:, :random_nframe]
                    # new_tags = []
                    # for tag_ in batch['tag'][::config.nframe]: # TODO tag in batch? 
                    #     new_tags.extend([tag_] * random_nframe)
                    # batch['tag'] = new_tags
                    if "depth" in batch:
                        batch['depth'] = batch['depth'][:, :random_nframe]
                    if "point_map" in batch:
                        batch['point_map'] = batch['point_map'][:, :random_nframe]
                        
                # 2) Random cond_num among nframe
                if config.get("fix_cond_num", None) is not None:
                    random_cond_num = config.fix_cond_num
                else:
                    random_cond_num = random.randint(config.min_cond_num, config.max_cond_num)

                # 3) Shuffle or Push tgt Views to last 
                if config.get("shuffle_train_rate", 0) > 0 and random.random() < config.get("shuffle_train_rate", 0):
                    batch = shuffle_batch(batch)
                else:
                    batch = uniform_push_batch(batch, random_cond_num)

                
                # ---------------------------------------
                image = batch["image"].to(device)  #  0 1 tensor [B,F,3,H,W]
                b, f, _, h, w = image.shape

                image_normalized = image * 2.0 - 1.0
                latents = slice_vae_encode(vae, image_normalized.to(weight_dtype), sub_size=16).to(weight_dtype)
                latents = latents * vae.config.scaling_factor
                _, _, _, latent_h, latent_w = latents.shape

                # build masks (random visible frames), valid:0, mask:1
                random_masks = torch.ones((b, f, 1, h, w), device=device, dtype=latents.dtype)
                random_latent_masks = torch.ones((b, f, 1, latent_h, latent_w), device=device, dtype=latents.dtype)
                random_masks[:, :random_cond_num] = 0
                random_latent_masks[:, :random_cond_num] = 0

                if config.get("adaptive_betas", False):
                    noise_scheduler = train_noise_scheduler[config.nframe - random_cond_num]
                else:
                    noise_scheduler = train_noise_scheduler

                # build cameras
                # CFG (cond/uncond) train
                if random.random() < config.model_cfg.cfg_training_rate:
                    add_channel = np.sum(config.model_cfg.additional_in_channels) - 1  # 减去mask channel
                    # if config.model_cfg.get("enable_depth", False) and not config.model_cfg.get("priors3d", False):  # TODO: temp code
                    #     ex_depth_ch = config.model_cfg.get("depth_freq", None)
                    #     if ex_depth_ch is None:
                    #         ex_depth_ch = 1
                    #     add_channel -= ex_depth_ch
                    # if "pixel" in config.model_cfg.get("prior_type", "3dpe"):
                    #     add_channel -= 4
                    # if "h3dpe" in config.model_cfg.get("prior_type", "3dpe"):
                    #     add_channel -= 48
                    camera_embedding = torch.zeros((b, f, add_channel, h, w), device=device, dtype=latents.dtype)
                else:
                    extri_, intri_ = batch["extrinsic"], batch["intrinsic"]
                    if extri_.shape[-2] == 3:
                        new_extri_ = torch.zeros((b, f, 4, 4), device=device, dtype=extri_.dtype)
                        new_extri_[:, :, :3, :4] = extri_
                        new_extri_[:, :, 3, 3] = 1.0
                        extri_ = new_extri_
                    extri_ = einops.rearrange(extri_, "b f c1 c2 -> (b f) c1 c2", f=f)
                    intri_ = einops.rearrange(intri_, "b f c1 c2 -> (b f) c1 c2", f=f)
                    camera_embedding = get_camera_embedding(intri_.to(device), extri_.to(device), 
                                                                b, f, h, w, config=config).to(device=device, dtype=latents.dtype) # b,f,c,h,w

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                if config.input_perturbation:
                    new_noise = noise + config.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                if config.train_timestep_schedule == "uniform":
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                elif config.train_timestep_schedule == "gaussian":
                    timesteps = truncated_normal((bsz,), **config.train_timestep_schedule_config).to(device=latents.device).long()
                else:
                    raise NotImplementedError
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # ensure valid latents through masks
                noisy_latents = latents * (1 - random_latent_masks) + noisy_latents * random_latent_masks

                # Get the text embedding for conditioning; set prompt to "" in some probability.
                encoder_hidden_states = None

                # Get the target for loss depending on the prediction type
                if config.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # concat all inputs
                inputs = noisy_latents  # [B,F,4,h,w]
                add_inputs = torch.cat([random_masks, camera_embedding], dim=2) # [B,F,1+6,H,W]
                coords = None
                # cat3d: no 3d priors
                # if config.model_cfg.get("enable_depth", False):
                #     if config.model_cfg.get("priors3d", False):
                #         if random.random() < config.model_cfg.cfg_training_rate:  # cfg coords
                #             if config.model_cfg.get("prior_type", "3dpe") == "3dpe+pixel":
                #                 coords = [torch.zeros((b * f, 192 + 1, latent_h, latent_w),
                #                                       device=device, dtype=latents.dtype),
                #                           torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)]
                #             elif config.model_cfg.get("prior_type", "3dpe") == "h3dpe+pixel":
                #                 coords = [torch.zeros((b * f, 48 + 1, h, w), device=device, dtype=latents.dtype),
                #                           torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)]
                #             elif config.model_cfg.get("prior_type", "3dpe") in ("3dpe+latent", "3dpe+warp_latent"):
                #                 coords = [torch.zeros((b * f, 192 + 1, latent_h, latent_w),
                #                                       device=device, dtype=latents.dtype),
                #                           torch.zeros((b * f, 5, latent_h, latent_w), device=device, dtype=latents.dtype)]
                #             elif config.model_cfg.get("prior_type", "3dpe") == "pixel":
                #                 coords = torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)
                #             else:
                #                 coords = torch.zeros((b * f, config.model_cfg.get("coord_dim", 192) + 1, latent_h, latent_w),
                #                                      device=device, dtype=latents.dtype)
                #         else:
                #             coords = get_3d_priors(config, batch['depth'], batch['intrinsic'], batch['extrinsic'],
                #                                    random_cond_num, nframe=f, device=device, colors=batch['image'],
                #                                    latents=latents, vae=vae, prior_type=config.model_cfg.get("prior_type", "3dpe"))

                #             # mask掉特定view的coords为全0 [(b f) c h w]
                #             coords_cfg = config.model_cfg.get("coords_cfg", 0.0)
                #             if coords_cfg > 0:
                #                 cmask = np.ones((b * f))
                #                 cmask[:int(b * f * coords_cfg)] = 0
                #                 np.random.shuffle(cmask)
                #                 cmask = cmask.reshape(b, f, 1, 1, 1)
                #                 cmask[:, :random_cond_num] = 1  # 确保condition不为0
                #                 cmask = torch.tensor(cmask, device=device, dtype=torch.float32)

                #                 if type(coords) == list:
                #                     for j in range(len(coords)):
                #                         coords[j] = einops.rearrange(coords[j], '(b f) c h w -> b f c h w', f=f)
                #                         coords[j] = coords[j] * cmask
                #                         coords[j] = einops.rearrange(coords[j], 'b f c h w -> (b f) c h w')
                #                 else:
                #                     coords = einops.rearrange(coords, '(b f) c h w -> b f c h w', f=f)
                #                     coords = coords * cmask
                #                     coords = einops.rearrange(coords, 'b f c h w -> (b f) c h w')

                #     else:
                #         depth_freq = config.model_cfg.get("depth_freq", None)
                #         if depth_freq is not None:
                #             depth = depth_freq_encoding(batch["depth"], device=device, embed_dim=depth_freq)
                #         else:
                #             depth = batch["depth"].to(torch.float32)
                #         depth = einops.rearrange(depth, "(b f) c h w -> b f c h w", b=bsz, f=f).to(device=device)
                #         depth[:, random_cond_num:] = 0
                #         add_inputs = torch.cat([add_inputs, depth], dim=2)
                #         coords = None

                # get class label (domain switcher)
                domain_dict = config.model_cfg.get("domain_dict", None)
                if domain_dict is not None:
                    # tags = batch["tag"][::f]
                    tag = batch['tag']
                    class_labels = [domain_dict.get(t, domain_dict['others']) for t in tag]
                    class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
                else:
                    class_labels = None

                # Predict the noise residual and compute loss
                if do_attn_distill: # Cache mode selected
                    distill_pairs = list(config.distill_config.distill_pairs).copy()
                    distill_pairs_idx = range(len(distill_pairs))
                    # Random pair selection: VRAM Issue
                    if config.distill_config.get('distill_pair_num', None) is not None:
                        distill_pairs_idx = random.sample(distill_pairs_idx, config.distill_config.get('distill_pair_num'))
                        distill_pairs = [distill_pairs[i] for i in distill_pairs_idx]
                    # distill_student_unet_attn_layers = [p[0] for p in distill_pairs]
                    clear_attn_cache(unet)
                    set_attn_cache(unet, attn_caching_unet_layers, print_=False, cache_qk=False)
                    set_attn_cache(unet, qk_caching_unet_layers, print_=False, cache_qk=True)
                if do_repa:
                    repa_layers = list(config.repa_config.distill_layers)
                    clear_feat_cache(unet)
                    set_feat_cache(unet, repa_layers)
                # with torch.autocast(device_type="cuda", dtype = weight_dtype):
                with torch.cuda.amp.autocast(enabled=True, dtype = weight_dtype):
                    inputs, add_inputs = inputs.to(weight_dtype), add_inputs.to(weight_dtype)
                    model_pred = unet(inputs, timesteps, encoder_hidden_states, add_inputs=add_inputs,
                                        class_labels=class_labels, coords=coords, return_dict=False)[0]  # [BF,C,H,W]
                model_pred = einops.rearrange(model_pred, "(b f) c h w -> b f c h w", f=f)

                diff_loss = torch.nn.functional.mse_loss(model_pred.float()[:, random_cond_num:], target.float()[:, random_cond_num:], reduction="mean")

                loss = diff_loss

                # VGGT 
                if config.vggt_on_fly:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype = weight_dtype):
                        image = batch["image"].to(device)  #  0 1 tensor [B,F,3,H,W]
                        vggt_pred = vggt_model(image) # jiho TODO make image [0,1]
                        batch['vggt_attn_cache'] = vggt_pred['attn_cache']
                    assert config.get('use_vggt_camera', False) != True, "use_vggt_camera while training not supproted! should be False"
                    # if config.use_vggt_camera:
                    #     batch['extrinsic'], batch['intrinsic'] = vggt_pred['extrinsic'], vggt_pred['intrinsic']  # currently using pointcloud from camera & depth ('world_points' to use pointmap pred)
                    if config.get('use_vggt_point_map', False ):
                        batch['point_map'] = vggt_pred.get('world_points_from_depth', None)
                    if do_repa:
                        dino_feat = vggt_pred['dino_feat'] # B S H W C
                        dino_feat = einops.rearrange(dino_feat, "B S H W C -> B 1 (S H W) C")  # B Head(1) N(FHW) C
                        batch['dino_feat'] = dino_feat
                        
                if config.get("dino_on_fly", False):
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype = weight_dtype):
                        image = batch["image"].to(device)  #  0 1 tensor [B,F,3,H,W]
                        B, F, _, H, W = image.shape
                        dino_image = image.clone()
                        dino_image = torch.nn.functional.interpolate(dino_image.flatten(0,1), size=(518,518), mode='bilinear')
                        dino_feat = dino_model.forward_features(dino_image)['x_norm_patchtokens']
                        dino_feat = einops.rearrange(dino_feat, "(B F) (h w) C -> B 1 (F h w) C", B=B, F=F, h=37, w=37)
                        batch['dino_feat'] = dino_feat # B Head(1) N(FHW) C
                
                # Distill Loss
                if do_attn_distill:
                    with torch.autocast(device_type="cuda", dtype = weight_dtype):
                        distill_loss_name_dict = {}
                        distill_loss_dict = {}
                        distill_gt_dict = {}
                        if config.vggt_on_fly:
                            distill_gt_dict.update( batch['vggt_attn_cache'] ) 
                        if config.get("dino_on_fly", False):
                            dino_feat = batch['dino_feat'].to(device) # B Head(1) N(FHW) C
                            distill_gt_dict["dino_feat"] = dict(query=dino_feat, key=dino_feat)

                        distill_vggt_gt = [p[1] for p in distill_pairs]
                        if "point_map" in distill_vggt_gt: # process point_map -> query / key tokens
                            point_map = batch["point_map"].to(device)  # b,f,3,h,w
                            assert point_map.shape[2] == 3, f"pointmap coord im should be on 2 (b,f,3,h,w) but {point_map.shape}"
                            assert point_map.shape[-1] == h and point_map.shape[-2] == w, f"pointmap im should be {h} {w} but {point_map.shape}"
                            point_map = einops.rearrange(point_map, "b f c h w -> b 1 (f h w) c") # B,1,FHW,3
                            distill_gt_dict["point_map"] = dict(query=point_map, key=point_map)

                        
                        B, F, _, H, W = batch['image'].shape
                        unet_attn_cache = pop_cached_attn(unet)
                        for idx, (unet_layer, vggt_layer) in zip(distill_pairs_idx, distill_pairs):
                            '''
                                gt(VGGT): query and key (B Head N(FHW) C) 
                                    - if using cost map feat, Head=1
                                pred(UNet): attention map(logit, before softmax) ( B Head Q(FHW) K(FHW) )
                            '''
                            # 0. pred attn logit
                            if str(unet_layer) in attn_caching_unet_layers:
                                # pred_attn_logit = unet_attn_cache[str(unet_layer)]['attn']  # B Head Q(FHW) K(FHW)
                                pred_attn_logit = unet_attn_cache[str(unet_layer)]
                            elif str(unet_layer) in qk_caching_unet_layers:
                                # use cached qk from unet
                                pred_q, pred_k = unet_attn_cache[str(unet_layer)]['q'], unet_attn_cache[str(unet_layer)]['k']  # B Head Q(FHW) C, B Head K(FHW) C
                                # unet Query/Key -> logit map
                                qk_head_kwargs = {'num_view': F}
                                pred_q = unet.unet_query_head[str(unet_layer)](pred_q, **qk_head_kwargs)  # B Head Q(FHW) C
                                pred_k = unet.unet_key_head[str(unet_layer)](pred_k, **qk_head_kwargs) # B Head K(FHW) C
                                scale_factor = 1 / math.sqrt(pred_q.size(-1))
                                pred_attn_logit = pred_q @ pred_k.transpose(-1, -2) * scale_factor # B Head Q(FHW) K(FHW)
                            else:
                                raise ValueError(f"unet layer {unet_layer} not in caching layers!")
                            
                            B, Head, Q, K = pred_attn_logit.shape
                            assert Q==K, f"pred attn should have same Q,K dim, but {pred_attn_logit.shape}"
                            # 1. gt attn logit
                            gt_query, gt_key = distill_gt_dict[str(vggt_layer)]['query'].detach(), distill_gt_dict[str(vggt_layer)]['key'].detach() # B Head VHW C, B Head VHW C
                            with torch.no_grad():
                                # resize qk -> match to pred attn size
                                def resize_tok(tok, target_size):
                                    B, Head, FHW, C = tok.shape
                                    HW = FHW // F
                                    H = W = int(math.sqrt(HW))
                                    tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', B=B, Head=Head, F=F, H=H, W=W, C=C)
                                    tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
                                    tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C',  B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
                                    return tok
                                pred_query_size, pred_key_size = int(math.sqrt(Q // F)), int(math.sqrt(K // F))
                                gt_query, gt_key = resize_tok(gt_query, target_size=pred_query_size), resize_tok(gt_key, target_size=pred_key_size)
                                # qk -> logit map
                                gt_attn_logit = distill_cost_fn_list[idx](gt_query, gt_key)
                            
                            # 2. Mask
                            # mask_per_view = torch.ones((B,Head,Q,F)).bool().to(device)
                            # (Option 1) Consistency Checker
                            # assert config.distill_config.get("consistency_check", True) is True
                            if config.distill_config.consistency_check[idx]:
                                def get_consistency_mask(logit):
                                    # assert config.distill_config.distill_query == "target", "consistency check only support distill_query to target"
                                    B, Head, F1HW, F2HW = logit.shape  
                                    assert F1HW == F2HW, "first process full(square) consistency mask"
                                    # assert Head == 1, "costmap should have only one head, while consistency checking?"
                                    HW = F1HW // F
                                    logit = einops.rearrange(logit, 'B Head (F1 HW1) (F2 HW2) -> (B Head F1 F2) HW1 HW2', B=B,Head=Head, F1=F, F2=F, HW1=HW, HW2=HW)
                                    mask_per_view = cycle_consistency_checker(logit, **config.distill_config.consistency_check_cfg[idx] ) # (B Head F1 F2) HW 1
                                    mask_per_view = einops.rearrange(mask_per_view, '(B Head F1 F2) HW 1 -> B Head (F1 HW) (F2 1)', B=B, Head=Head, F1=F, F2=F, HW=HW)
                                    # mask = mask.any(dim=-1) # B Head Q(F1HW) 
                                    return mask_per_view # B Head Q(F1HW) F2
                                mask_per_view = get_consistency_mask(gt_attn_logit) # B Head Q(F1HW) F2
                            else:
                                mask_per_view = None


                            # 3. Loss Compute
                            ''' View idx order: [Reference, Target]'''
                            def slice_attnmap(attnmap, query_idx, key_idx, exclude_selfattn=False):
                                B, Head, Q, K = attnmap.shape
                                HW = Q // F
                                C = K // F
                                attnmap_ = einops.rearrange(attnmap, 'B Head (F1 HW) (F2 C) -> B Head F1 HW F2 C', B=B, Head=Head, F1=F, HW=HW, F2=F, C=C)
                                attnmap_sliced = []
                                for i in query_idx:
                                    a = attnmap_[:,:,[i]] # B Head 1 HW F HW
                                    k_id = key_idx.copy()
                                    if (i in k_id) and exclude_selfattn: # remove self if necessary
                                        k_id.remove(i)
                                    a = a[:,:,:,:,k_id] # B Head 1 HW f2 HW
                                    attnmap_sliced += [a]
                                attnmap_sliced = torch.cat(attnmap_sliced, dim=2) # B Head f1 HW f2 HW
                                f1, f2 = attnmap_sliced.shape[2], attnmap_sliced.shape[4] 
                                attnmap_sliced = einops.rearrange(attnmap_sliced, 'B Head f1 HW f2 C -> B Head (f1 HW) (f2 C)', B=B, Head=Head, f1=f1, f2=f2, HW=HW, C=C)
                                return attnmap_sliced, f1, f2  # B Head f1HW f2HW
                            

                            distill_key_for_reference_query = list(config.distill_config.distill_key_for_reference_query[idx]).copy()
                            distill_key_for_target_query = list(config.distill_config.distill_key_for_target_query[idx]).copy()
                            # if save vram, select query type
                            if len(distill_key_for_reference_query)>0 and len(distill_key_for_target_query) > 0 and  \
                                config.distill_config.get('save_vram_query_axis', False):
                                if random.random() > 0.5:
                                    distill_key_for_reference_query.clear()
                                else:
                                    distill_key_for_target_query.clear()

                                
                            # Reference Query Attnmap
                            ref_query_idx = list(range(random_cond_num))
                            key_list = distill_key_for_reference_query # among ["self", "reference","target"]
                            key_idx = []
                            if "reference" in key_list:
                                key_idx += list(range(random_cond_num))
                            if "target" in  key_list:
                                key_idx += list(range(random_cond_num, F))
                            '''Caution: "self" applied in slice_attnmap'''
                            key_idx =sorted(key_idx)
                            if len(key_idx) > 0 :
                                # 1) slice
                                ref_pred_attn, f1, f2 = slice_attnmap(pred_attn_logit, ref_query_idx, key_idx, exclude_selfattn="self" not in key_list)
                                ref_gt_attn, _, _ = slice_attnmap(gt_attn_logit, ref_query_idx, key_idx, exclude_selfattn="self" not in key_list)
                                if mask_per_view is not None:
                                    ref_mask_per_view, _, _ = slice_attnmap(mask_per_view, ref_query_idx, key_idx, exclude_selfattn="self" not in key_list)
                                # 2) logit head
                                logit_head_kwargs = {'num_view': f2, 'num_view_query': f1}
                                ref_pred, ref_gt = unet.unet_logit_head[str(unet_layer)](ref_pred_attn, **logit_head_kwargs), unet.vggt_logit_head[str(vggt_layer)](ref_gt_attn, **logit_head_kwargs) # [B head f1HW f2HW]
                                ''' [B Head Q K] 
                                    - if softargmax: (always per_view == True)
                                        -- B Head Q K(V*2)
                                '''
                                # 3) apply consistency mask
                                if config.distill_config.get("mask_per_view_consistency", [False])[idx]:
                                    assert config.distill_config.get("mask_full_view_consistency", [False])[idx] == False
                                    ref_pred = einops.rearrange(ref_pred, 'B Head f1HW (f2 C) -> B Head f1HW f2 C', f2 = f2)
                                    ref_gt =  einops.rearrange(ref_gt, 'B Head f1HW (f2 C) -> B Head f1HW f2 C', f2 = f2)
                                    ref_pred, ref_gt = ref_pred[ref_mask_per_view.bool()], ref_gt[ref_mask_per_view.bool()]
                                elif config.distill_config.get("mask_full_view_consistency", [False])[idx]:
                                    ref_mask = ref_mask_per_view.any(dim=-1) # B Head f1HW f2 -> B Head f1HW
                                    ref_pred, ref_gt = ref_pred[ref_mask.bool()], ref_gt[ref_mask.bool()]
                                # 4) loss
                                ref_query_distill_loss = distill_loss_fn_list[idx](ref_pred.float(), ref_gt.float(), **distill_loss_fn_config_list[idx])
                            else:
                                ref_query_distill_loss = 0

                            # Target Query Attnmap
                            tgt_query_idx = list(range(random_cond_num, F))
                            key_list = distill_key_for_target_query # among ["self", "reference","target"]
                            key_idx = []
                            if "reference" in key_list:
                                key_idx += list(range(random_cond_num))
                            if "target" in  key_list:
                                key_idx += list(range(random_cond_num, F))
                            key_idx =sorted(key_idx)
                            if len(key_idx) > 0 :
                                # 1) slice
                                tgt_pred_attn, f1, f2 = slice_attnmap(pred_attn_logit, tgt_query_idx, key_idx, exclude_selfattn="self" not in key_list)
                                tgt_gt_attn, _, _ = slice_attnmap(gt_attn_logit, tgt_query_idx, key_idx, exclude_selfattn="self" not in key_list)
                                if mask_per_view is not None:
                                    tgt_mask_per_view, _, _ = slice_attnmap(mask_per_view, tgt_query_idx, key_idx, exclude_selfattn="self" not in key_list)
                                # 2) logit head
                                logit_head_kwargs = {'num_view': f2, 'num_view_query': f1}
                                tgt_pred, tgt_gt = unet.unet_logit_head[str(unet_layer)](tgt_pred_attn, **logit_head_kwargs), unet.vggt_logit_head[str(vggt_layer)](tgt_gt_attn, **logit_head_kwargs) # [B head f1HW f2HW]
                                ''' [B Head Q K] 
                                    - if softargmax: (always per_view == True)
                                        -- B Head Q K(V*2)
                                '''
                                
                                # 3) apply consistency mask
                                if config.distill_config.get("mask_per_view_consistency", [False])[idx]:
                                    assert config.distill_config.get("mask_full_view_consistency", [False])[idx] == False
                                    tgt_pred = einops.rearrange(tgt_pred, 'B Head f1HW (f2 C) -> B Head f1HW f2 C', f2 = f2)
                                    tgt_gt =  einops.rearrange(tgt_gt, 'B Head f1HW (f2 C) -> B Head f1HW f2 C', f2 = f2)
                                    tgt_pred, tgt_gt = tgt_pred[tgt_mask_per_view.bool()], tgt_gt[tgt_mask_per_view.bool()]
                                elif config.distill_config.get("mask_full_view_consistency", [False])[idx]:
                                    tgt_mask = tgt_mask_per_view.any(dim=-1) # B Head f1HW f2 -> B Head f1HW
                                    tgt_pred, tgt_gt = tgt_pred[tgt_mask.bool()], tgt_gt[tgt_mask.bool()]

                                # 4) loss
                                tgt_query_distill_loss = distill_loss_fn_list[idx](tgt_pred.float(), tgt_gt.float(), **distill_loss_fn_config_list[idx])
                            else:
                                tgt_query_distill_loss = 0

                            distill_loss =  ref_query_distill_loss + tgt_query_distill_loss # TODO: weight?
                            if torch.isnan(distill_loss).any().item():
                                accelerator.print("NaN in distill_loss, skip this layer distill...")
                                continue
                            distill_loss_name_dict[idx] = f"train/distill_loss/unet{unet_layer}_vggt{vggt_layer}" # for log
                            distill_loss_dict[idx] = distill_loss

                        weighted_distill_loss = 0
                        for idx in distill_loss_dict.keys():
                            weighted_distill_loss = weighted_distill_loss + distill_loss_dict[idx] * distill_loss_weight_list[idx]
                        # weighted_distill_loss = sum(weighted_distill_loss_list) if len(weighted_distill_loss_list) > 0 else torch.tensor(0.0).to(device, dtype=weight_dtype)
                        
                        if config.distill_config.get("distill_loss_weight_scheduling", None) is not None:
                            raise NotImplementedError("distill loss scheduling not supported")
                            if config.distill_config.distill_loss_weight_scheduling == "cosine":
                                distill_loss_weight = cosine_loss_weight_scheduler(global_step, **config.distill_config.distill_loss_weight_scheduling_config )
                            else:
                                raise NotImplementedError
                        loss = loss + weighted_distill_loss
                    
                if do_repa:
                    with torch.autocast(device_type="cuda", dtype = weight_dtype):
                        B, F, _, H, W = image.shape
                        unet_feat_cache = pop_cached_feat(unet)
                        gt_feat = batch['dino_feat'].to(device)  # B Head(1) N(FHW) C
                        hw = gt_feat.shape[2] // F
                        h =w= int(hw**0.5)
                        gt_feat = einops.rearrange(gt_feat, 'B 1 (F h w) C -> B F h w C', B=B, F=F, h=h, w=w)
                        repa_view = []
                        if 'reference' in config.repa_config.distill_view:
                            repa_view += list(range(random_cond_num))
                        if 'target' in config.repa_config.distill_view:
                            repa_view += list(range(random_cond_num, F))
                        repa_view = sorted(repa_view)

                        repa_loss_dict = {}
                        for unet_layer in repa_layers:
                            '''
                                gt_feat (dino, from vggt): B F H W C
                                pred(UNet): feat B (f hw) C
                            '''
                            # slice view
                            gt_feat = gt_feat[:, repa_view]
                            pred_feat = unet_feat_cache[str(unet_layer)] # B (f hw) C
                            pred_feat = einops.rearrange(pred_feat, 'B (F HW) C -> B F HW C', F=F)[:, repa_view]
                            _,_,pred_hw, pred_c = pred_feat.shape
                            pred_h=pred_w = int(pred_hw**0.5)
                            pred_feat= pred_feat.flatten(1,2)

                            # resize gt -> unet
                            gt_feat = einops.rearrange(gt_feat, 'B f H W C -> (B f) C H W', f =len(repa_view))
                            gt_feat = torch.nn.functional.interpolate(gt_feat, size=(pred_h,pred_w), mode='bilinear')
                            gt_feat = einops.rearrange(gt_feat, '(B f) C h w -> B (f h w) C', f = len(repa_view))

                            # pred_feat
                            pred_feat = unet.repa_feat_head[str(unet_layer)](pred_feat)
                            pred_feat, gt_feat = torch.nn.functional.normalize(pred_feat, dim=-1), torch.nn.functional.normalize(gt_feat, dim= -1)
                            repa_loss_dict[f"train/repa/unet{unet_layer}"] = - (gt_feat * pred_feat).sum(dim=-1).mean()
                        repa_loss = sum(repa_loss_dict.values()) / len(repa_loss_dict.values()) if len(repa_loss_dict) >0 else torch.tensor(0.0).to(device, dtype=weight_dtype)
                        loss = loss + repa_loss * config.repa_config.distill_loss_weight



                # DEBUG for NaN loss
                if torch.isnan(loss).item():
                    accelerator.log({"train/nan/diff_loss": diff_loss}, step=global_step)
                    if do_attn_distill:
                        accelerator.log({"train/nan/distill_loss": distill_loss.item()}, step=global_step)
                        accelerator.log({k.replace("train", "train/nan"):v.item() for k,v in distill_loss_dict.items() })
                    import pickle
                    with open(f"{args.output_dir}/nan_batch_rank{accelerator.process_index}.pkl", "wb") as w:
                        save = {"batch": batch, "timesteps": timesteps, "model_pred": model_pred, "target": target}
                        if do_attn_distill:
                            save.update({'pred_attn_logit': pred_attn_logit.detach().cpu(), "gt_attn_logit": gt_attn_logit.detach().cpu()})
                        pickle.dump(save, w)
                    save_path = os.path.join(args.output_dir, "nan.ckpt")
                    torch.save(accelerator.unwrap_model(unet).state_dict(), save_path)
                    raise ValueError("NaN loss, stop training...")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.opt_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                    train_loss = avg_loss.item() / config.gradient_accumulation_steps
                    accelerator.log({"train/loss": train_loss}, step=global_step)
                    accelerator.log({"train/diff_loss": diff_loss}, step=global_step)
                    accelerator.log({"train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    accelerator.log({"train/batch_time": batch_time}, step=global_step)
                    if args.use_ema and ema_unet.cur_decay_value is not None:
                        accelerator.log({"train/ema_decay": ema_unet.cur_decay_value}, step=global_step)
                    if do_attn_distill:  
                        accelerator.log({"train/distill_loss": distill_loss.item()}, step=global_step)
                        # accelerator.log({"train/distill_loss_weighted": (distill_loss_weight * distill_loss).item()}, step=global_step)
                        # accelerator.log(distill_loss_dict, step=global_step)
                        accelerator.log({name: distill_loss_dict[idx].item() for idx, name in distill_loss_name_dict.items()}, step=global_step)
                        # softmax temperature info
                        for unet_layer, vggt_layer in distill_pairs:
                            unet_layer_logit_head = unet.unet_logit_head[str(unet_layer)]
                            if hasattr(unet_layer_logit_head, 'softmax_temp'):
                                t = unet_layer_logit_head.softmax_temp
                                t = t.detach().cpu().item() if torch.is_tensor(t) else t
                                accelerator.log({f"train/distill/unet{unet_layer}_temp": t}, step=global_step)
                            vggt_layer_logit_head = unet.vggt_logit_head[str(vggt_layer)]
                            if hasattr(vggt_layer_logit_head, 'softmax_temp'):
                                t = vggt_layer_logit_head.softmax_temp
                                t = t.detach().cpu().item() if torch.is_tensor(t) else t
                                accelerator.log({f"train/distill/vggt{vggt_layer}_temp": vggt_layer_logit_head.softmax_temp}, step=global_step)
                    if do_repa:
                        accelerator.log({"train/repa_loss": repa_loss.item()}, step = global_step)
                        accelerator.log(repa_loss_dict, step = global_step)
                        
                    # logger.info(f"Loss: {train_loss}")

                if (global_step % args.train_log_interval == 0 or first_batch) and accelerator.is_main_process:
                    
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    if do_attn_distill: unset_attn_cache(unet)
                    if do_repa: unset_feat_cache(unet)
                    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
                    log_train(accelerator=accelerator, config=config, args=args,
                              pipeline=pipeline, weight_dtype=weight_dtype, batch=batch,
                              step=global_step, random_cond_num=random_cond_num,
                              device=device, vae=vae, nframe=f)
                    # set_attn_cache(unet, distill_student_unet_attn_layers, print_=False)
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
                    first_batch = False

                if (global_step % config.checkpointing_steps == 0) or (global_step in config.checkpointing_steps_manual):
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.use_ema and accelerator.is_main_process:
                        ema_unet.save_pretrained(accelerator.unwrap_model(unet), f"{save_path}/ema_unet.pt")
                    logger.info(f"Saved state to {save_path}")
                    
                    # also save last
                    save_path = os.path.join(args.output_dir, f"checkpoint-last")
                    accelerator.save_state(save_path)
                    if args.use_ema and accelerator.is_main_process:
                        ema_unet.save_pretrained(accelerator.unwrap_model(unet), f"{save_path}/ema_unet.pt")
                    # save global step
                    with open(os.path.join(save_path, "global_step.txt"), "w") as f:
                        f.write(str(global_step))
                    logger.info(f"Saved state to {save_path}")
                    
                if (global_step % args.checkpointing_last_steps == 0):
                    save_path = os.path.join(args.output_dir, f"checkpoint-last")
                    accelerator.save_state(save_path)
                    if args.use_ema and accelerator.is_main_process:
                        ema_unet.save_pretrained(accelerator.unwrap_model(unet), f"{save_path}/ema_unet.pt")
                    # save global step
                    with open(os.path.join(save_path, "global_step.txt"), "w") as f:
                        f.write(str(global_step))
                    logger.info(f"Saved state to {save_path}")

                if ((global_step == 1 and args.val_at_first) or global_step % args.val_interval == 0) and not args.no_val:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    if do_attn_distill: unset_attn_cache(unet)
                    if do_repa: unset_feat_cache(unet)
                    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
                    for data_name, val_dataloader in val_dataloaders_dict.items():
                        res = log_test(accelerator=accelerator, config=config, args=args,
                                pipeline=pipeline, dataloader=val_dataloader, process_batch_fn=lambda x: x, # 11/02 uniform push batch disabled, (done in code for inference setting)
                                step=global_step, device=accelerator.device, vae=vae,
                                cond_num = config.val_cond_num, nframe = config.val_nframe, cfg = args.val_cfg,
                                compute_fid = False,
                                viz_len = config.val_viz_len,
                                num_inference_steps = 50,
                                eval_size = 512,
                                eval_w_mask = args.val_w_mask,
                                test_dataset_name = data_name
                                )
                        # res = log_validation(accelerator=accelerator, config=config, args=args,
                        #                      pipeline=pipeline, val_dataloader=val_dataloader,
                        #                      step=global_step, device=device, vae=vae, log_prefix='val_' + data_name)
                    # set_attn_cache(unet, distill_student_unet_attn_layers, print_=False)
                    
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
                    # if res <= best_metric:
                    #     best_metric = res
                    #     save_path = os.path.join(args.output_dir, "best")
                    #     accelerator.save_state(save_path)
                    #     if args.use_ema and accelerator.is_main_process:
                    #         ema_unet.save_pretrained(accelerator.unwrap_model(unet), f"{save_path}/ema_unet.pt")
                    #     logger.info(f"Saved best state to {save_path}")
                    

            logs = {"epoch": epoch + 1, "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break
            
            end_time = time.time()

    accelerator.end_training()


if __name__ == "__main__":
    main()

