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
from omegaconf import OmegaConf
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

# from src.distill_utils.attn_logit_head import cycle_consistency_checker
from src.distill_utils.attn_processor_cache import set_attn_cache, unset_attn_cache, pop_cached_attn, clear_attn_cache, set_feat_cache, unset_feat_cache, pop_cached_feat, clear_feat_cache
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
def log_validation(accelerator, config, args, pipeline, val_dataloader, step, device, **kwargs):
    ''' Caution, batch=1 !
    '''
    if accelerator.is_main_process:
        logger.info(f"Validation log in step {step}")

    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    manual_val_len = config.get("val_manual_len", None)
    compute_fid = config.get("val_compute_fid", False) 
    viz_len = config.get("val_viz_len", 30)
    if compute_fid:
        if accelerator.is_main_process:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid_calculator = FrechetInceptionDistance(normalize=True).to(device)
            fid_calculator.reset()

    if manual_val_len is not None:
        assert manual_val_len % (accelerator.num_processes * 1) == 0
        manual_val_step = manual_val_len // (accelerator.num_processes * 1)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    show_images = []
    # show_save_dict = collections.defaultdict(int)
    val_iter = 0

    cond_num = config.val_cond_num
    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            batch = uniform_push_batch(batch, cond_num)
            # unbatchify
            image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
            image_normalized = image * 2.0 - 1.0

            extrinsic, intrinsic = extri_, intri_
            tag, sequence_name, depth = None, None , None
            preds = pipeline(images=image_normalized, nframe=config.val_nframe, cond_num=cond_num,
                             height=image_normalized.shape[2], width=image_normalized.shape[3],
                             intrinsics=intrinsic, extrinsics=extrinsic,
                             num_inference_steps=50, guidance_scale=args.val_cfg,
                             output_type="np", config=config, tag=tag,
                             sequence_name=sequence_name, depth=depth, vae=kwargs['vae']).images  # [f,h,w,c]

            if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
                color_warps = global_position_encoding_3d(config, depth, batch['intrinsic'],
                                                          batch['extrinsic'], 1,
                                                          nframe=config.val_nframe, device=accelerator.device,
                                                          pe_scale=1 / 8,
                                                          embed_dim=config.model_cfg.get("coord_dim", 192),
                                                          colors=image)
            else:
                color_warps = None

            # if batch['tag'][0] not in show_save_dict or show_save_dict[batch['tag'][0]] < 10:  # 每个dataset显示10个
                # show_save_dict[batch['tag'][0]] += 1
            if val_iter < viz_len:
                if depth is not None:
                    show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num, batch["depth"])
                else:
                    show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num)

                if color_warps is not None:
                    h, w = image.shape[2], image.shape[3]
                    show_image[h:h * 2, cond_num * w:] = color_warps[0][:, cond_num * w:]
                show_images.append(show_image)

            gt_images = (image_normalized[cond_num:].permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 # -1 1 → 0 1
            preds = preds[cond_num:]

            if compute_fid:
                gt_imgs_fid, preds_fid = torch.tensor(gt_images).permute(0, 3, 1, 2).to(device),  torch.tensor(preds).permute(0, 3, 1, 2).to(device)
                gt_imgs_fid, preds_fid = accelerator.gather(gt_imgs_fid), accelerator.gather(preds_fid)
                if accelerator.is_main_process:
                    fid_calculator.update(einops.rearrange(gt_imgs_fid,'... c h w -> (...) c h w'), real=True)
                    fid_calculator.update(einops.rearrange(preds_fid,'... c h w -> (...) c h w'), real=False)

            for i in range(preds.shape[0]):
                psnr_ = peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0)
                psnr_scores.append(psnr_)
                ssim_ = structural_similarity(cv2.cvtColor(gt_images[i], cv2.COLOR_RGB2GRAY),
                                              cv2.cvtColor(preds[i], cv2.COLOR_RGB2GRAY), data_range=1.0)
                ssim_scores.append(ssim_)
                lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device)
                lpips_scores.append(lpips_)
            
            val_iter += 1 
            if manual_val_len is not None:
                if val_iter >= manual_val_step:
                    break
    
    # unify all results
    psnr_score = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
    ssim_score = torch.tensor(np.mean(ssim_scores), device=device, dtype=torch.float32)
    lpips_score = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)

    psnr_score = accelerator.gather(psnr_score).mean().item()
    ssim_score = accelerator.gather(ssim_score).mean().item()
    lpips_score = accelerator.gather(lpips_score).mean().item()
    if compute_fid:

        if accelerator.is_main_process:
            print("fid compute start")
            fid_val = fid_calculator.compute()
            real_num, fake_num = fid_calculator.real_features_num_samples, fid_calculator.fake_features_num_samples
            fid_calculator.reset()
            del fid_calculator
            torch.cuda.empty_cache()
            print("fid compute end")
            accelerator.log({"val/fid": fid_val, "val/fid_real_num": real_num, "val/fid_fake_num": fake_num}, step=step)


    accelerator.log({"val/psnr": psnr_score, "val/ssim": ssim_score, "val/lpips": lpips_score}, step=step)

    
    show_images_full = torch.cat(show_images, dim=1).to(device)
    show_images_full = accelerator.gather(show_images_full).reshape(-1,*show_images_full.shape).permute(1,0,2,3).flatten(1,2)
    if accelerator.is_main_process:
        # for j in range(len(show_image)):
        #     if config.image_size > 256:
        #         show_images[j] = cv2.resize(show_images[j], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # accelerator.log({f"val/gt_masked_pred_images{j}": wandb.Image(show_images[j])}, step=step)
        accelerator.log({"val/show_images": wandb.Image(show_images_full)}, step=step)

    del loss_fn_alex
    torch.cuda.empty_cache()

    return lpips_score


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
    tag, sequence_name = None, None

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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resume_from_checkpoint is not None and args.resume_path is None:
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
    return


if __name__ == "__main__":
    main()

