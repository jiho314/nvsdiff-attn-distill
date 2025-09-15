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
from src.datasets.global_datasets import load_global_dataset
from src.datasets.global_sampler import GlobalConcatSampler
from src.modules.camera import get_camera_embedding
from src.modules.position_encoding import depth_freq_encoding, global_position_encoding_3d, get_3d_priors
from src.modules.schedulers import get_diffusion_scheduler
from utils import get_lpips_score, _seq_name_to_seed

from src.distill_utils.attn_processor_cache import set_attn_cache, unset_attn_cache, pop_cached_attn, clear_attn_cache
from src.modules.attention_visualization_callback import AttentionVisualizationCallback
from src.modules.timestep_sample import truncated_normal

from src.distill_utils.attn_logit_head import LOGIT_HEAD_CLS # JIHO TODO: save/load params
from src.distill_utils.query_key_cost_metric import COST_METRIC_FN

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
logger = get_logger(__name__, log_level="INFO")




def cross_entropy(prob, prob_gt):
            """Cross entropy loss for attention probabilities."""
            eps = 1e-8
            return - (prob_gt * (prob + eps).log()).sum(dim=-1).mean()

def kl_divergence(prob, prob_gt):
    """Kullback-Leibler divergence loss for attention probabilities."""
    return (prob_gt * (prob_gt.log() - prob.log())).sum(dim=-1).mean()

ATTN_LOSS_FN = {
    "l1": torch.nn.functional.l1_loss,
    "cross_entropy": cross_entropy,
    "kl_divergence": kl_divergence,
}

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


def shuffle_batch(batch):
    ''' Caution data_name with "key" is not shuffled
    '''
    img = batch["image"]  # [B,F,3,H,W]
    B,F,_,H,W = img.shape
    perm = torch.randperm(F)

    # for key in data_keys:
    #     batch[key] = batch[key][:, perm]
    for k in batch.keys():
        if not "key" in k:
            batch[k] = batch[k][:, perm]
    # batch["image"] = img[:, perm]
    # batch["intrinsic"] = batch["intrinsic"][:, perm]
    # batch["extrinsic"] = batch["extrinsic"][:, perm]
    return batch

def uniform_push_batch(batch, random_cond_num=0):
    ''' Caution data_name with "key" is not applied
     1) uniformly sample target idx
     2) push target views to last
    '''
    img = batch["image"]  # [B,F,3,H,W]
    B,F,_,H,W = img.shape
    target_num = F - random_cond_num
    idx = torch.arange(F)
    tgt_idx = torch.linspace(1, F-2, target_num, dtype=torch.long)
    ref_idx = idx[~torch.isin(idx, tgt_idx)]
    new_idx = torch.cat([ref_idx, tgt_idx], dim=0)[:F]  # in case target_num +2 > F

    for k in batch.keys():
        if not "key" in k:
            batch[k] = batch[k][:, new_idx]
    return batch

def cycle_consistency_checker(costmap, threshold, width=32,):
    # Get dimensions from the input tensor
    B, HW, _ = costmap.shape
    device = costmap.device

    # Step 2: Find the best initial matches (already vectorized)
    max_idx = torch.argmax(costmap, dim=-1)  # (B, HW)
    transpose_costmap = costmap.transpose(1, 2)  # (B, 2HW, HW)
    b_idx = torch.arange(B, device=device)[:, None] # (B, 1)
    reverse_map = transpose_costmap[b_idx, max_idx] # (B, HW, HW)
    _, final_indices = torch.max(reverse_map, dim=-1) 
    # Create a tensor representing the original indices (0, 1, 2, ..., HW-1) for each batch item
    original_indices = torch.arange(HW, device=device).expand(B, -1) # (B, HW)

    # Convert 1D indices to 2D coordinates for both original and matched points
    x1 = original_indices // width
    y1 = original_indices % width
    x2 = final_indices // width
    y2 = final_indices % width

    # Calculate Euclidean distance and check if it's within the threshold
    distance = torch.sqrt(((x1 - x2)**2 + (y1 - y2)**2).float())
    is_close = (distance < threshold).float() # .float() converts boolean (True/False) to (1.0/0.0)

    # Step 6: Reshape final output tensor
    final_distance_tensor = is_close.view(B, HW, 1)

    return final_distance_tensor

def slice_vae_encode(vae, image, sub_size):  # vae fails to encode large tensor directly, we need to slice it
    # Use CUDA autocast for mixed precision where available
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
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
    
    # Handle VAE - it may not be wrapped by accelerator if not prepared
    try:
        unwrapped_vae = accelerator.unwrap_model(vae)
    except:
        unwrapped_vae = vae  # VAE is not prepared by accelerator
    
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=unwrapped_vae,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def get_show_images(input_images, pred_images, cond_num, depth=None):
    # pred_images는 target frames만 포함 (cond_num 이후)
    pred_images_list = [pred_images[i] for i in range(pred_images.shape[0])]
    pred_images_concat = np.clip(np.concatenate(pred_images_list, axis=1) * 255, 0, 255).astype(np.uint8)  # [H,W*target_frames,c]
    
    # 원본 이미지를 normalize
    input_images_orig = (input_images + 1) / 2
    
    # reference frames (condition frames)
    ref_frames = [np.array(ToPILImage()(input_images_orig[i])) for i in range(cond_num)]
    ref_concat = np.concatenate(ref_frames, axis=1)
    
    # target frames의 ground truth
    target_frames_count = input_images_orig.shape[0] - cond_num
    gt_target_frames = [np.array(ToPILImage()(input_images_orig[i])) for i in range(cond_num, input_images_orig.shape[0])]
    gt_target_concat = np.concatenate(gt_target_frames, axis=1)
    
    # pred_images와 gt_target의 크기를 맞춤
    pred_width = pred_images_concat.shape[1]
    gt_width = gt_target_concat.shape[1]
    
    if pred_width != gt_width:
        # 더 작은 크기로 맞춤
        min_width = min(pred_width, gt_width)
        pred_images_concat = pred_images_concat[:, :min_width]
        gt_target_concat = gt_target_concat[:, :min_width]
    
    # 첫 번째 행: 모델의 input 전체 + 생성 (ref1 ref2 pred)
    model_output_row = np.concatenate([ref_concat, pred_images_concat], axis=1)
    
    # 두 번째 행: GT (ref1 ref2 gt)
    ground_truth_row = np.concatenate([ref_concat, gt_target_concat], axis=1)
    
    if depth is not None:
        # depth 정보가 있으면 세 번째 행으로 추가
        min_vals = depth.amin(dim=[2, 3], keepdim=True)
        max_vals = depth.amax(dim=[2, 3], keepdim=True)
        depth = (depth - min_vals) / (max_vals - min_vals)
        depth_frames = [np.array(ToPILImage()(depth[i]).convert("RGB")) for i in range(depth.shape[0])]
        depth_row = np.concatenate(depth_frames, axis=1)
        show_image = np.concatenate([model_output_row, ground_truth_row, depth_row], axis=0)
    else:
        show_image = np.concatenate([model_output_row, ground_truth_row], axis=0)

    return show_image

@torch.no_grad()
def log_validation(accelerator, config, args, pipeline, val_dataloader, step, device, weight_dtype, cond_num = None, do_attn_visualize = False, vggt_model = None,
                    visualize_config = None,
                    **kwargs):
    ''' Caution, batch=1 !
    '''
    if accelerator.is_main_process:
        logger.info(f"Validation log in step {step}")

    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    compute_fid = config.get("val_compute_fid", True) 
    viz_len = config.get("val_viz_len", 52)
    
    if compute_fid:
        if accelerator.is_main_process:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid_calculator = FrechetInceptionDistance(normalize=True).to(device)
            fid_calculator.reset()

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    attention_loss_scores = {}  # 레이어별 attention loss 수집용 {layer_key: [scores]}
    show_images = []
    # show_save_dict = collections.defaultdict(int)
    val_iter = 0

    cond_num = config.fix_cond_num

    # visualize_loss_fn 초기화 (validation loop 전에 정의)
    # Note: loss_fn is selected per-pair in the callback; do not assume a global loss_fn here.

    # Match train.py: use torch.autocast with explicit dtype
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):

            # Defensive: if a sample is malformed, log error and raise so user can fix data.
            if batch is None:
                logger.error("Encountered None batch from dataloader during validation. Raising error to surface data issue.")
                raise RuntimeError("Validation dataloader yielded None batch. Check dataset postprocessing and WebDataset tars.")
            if not isinstance(batch, dict) and not hasattr(batch, 'keys'):
                logger.error(f"Encountered unexpected batch type {type(batch)} during validation. Raising error.")
                raise RuntimeError(f"Unexpected batch type from dataloader: {type(batch)}")

            # ensure required keys exist
            if 'image' not in batch or 'intrinsic' not in batch or 'extrinsic' not in batch:
                logger.error(f"Batch missing required keys. Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else None}")
                raise RuntimeError("Validation batch missing one of 'image','intrinsic','extrinsic' keys.")

            # proceed with normal processing
            batch = uniform_push_batch(batch, cond_num)
            # unbatchify
            image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
            image_normalized = image * 2.0 - 1.0

            extrinsic, intrinsic = extri_, intri_
            tag, sequence_name, depth = None, None , None
            # attention visualization callback 설정
            attention_callback = None
            if do_attn_visualize:
                # visualize_config 간소화
                visualize_config_with_fn = visualize_config.copy()
                visualize_config_with_fn['viz_log_wandb'] = True
                
                # 시퀀스 이름 설정
                seq_name = f"sample_{val_iter:03d}"
                if 'sequence_name' in batch:
                    seq_batch = batch['sequence_name']
                    if isinstance(seq_batch, (list, tuple)) and len(seq_batch) > 0:
                        seq_name = str(seq_batch[0])
                    elif seq_batch is not None:
                        seq_name = str(seq_batch)
                visualize_config_with_fn['viz_seq_name'] = seq_name
                
                # attention visualization도 샘플별 step 사용
                visualize_config_with_fn['viz_wandb_step_base'] = val_iter
                
                attention_callback = AttentionVisualizationCallback(
                    vggt_model=vggt_model,
                    visualize_config=visualize_config_with_fn,
                    batch=batch,
                    cond_num=cond_num,
                    device=accelerator.device,
                    do_attn_visualize=do_attn_visualize,
                    accelerator=accelerator
                )
            
            # 디버그: 첫 번째 샘플만 입력 차원 확인 (로그 정리)
            if val_iter == 0:
                print(f"DEBUG - Pipeline input shapes:")
                print(f"  image_normalized.shape: {image_normalized.shape}")
                print(f"  config.nframe: {config.nframe}")
                print(f"  cond_num: {cond_num}")
            
            preds = pipeline(images=image_normalized, nframe=config.nframe, cond_num=cond_num,
                             height=image_normalized.shape[2], width=image_normalized.shape[3],
                             intrinsics=intrinsic, extrinsics=extrinsic,
                             num_inference_steps=50, guidance_scale=args.val_cfg,
                             output_type="np", config=config, tag=tag,
                             sequence_name=sequence_name, depth=depth, vae=kwargs['vae'],
                             callback_on_step_end=attention_callback,
                             callback_on_step_end_tensor_inputs=["latents"]).images  # [f,h,w,c]
            
            # Pipeline 완료 후 VGGT cache 정리
            if do_attn_visualize and attention_callback is not None:
                attention_callback.clear_vggt_cache()

            # Strict validation: if pipeline returned invalid predictions or targets are empty,
            # raise immediately with detailed debug info so user can inspect dataset.
            try:
                preds_arr = np.array(preds)
            except Exception:
                logger.error(f"Pipeline returned non-convertible preds type: {type(preds)}; batch keys: {list(batch.keys())}")
                raise RuntimeError("Pipeline returned invalid preds; convert error. Check pipeline outputs and data.")

            img_f = image_normalized.shape[0] if hasattr(image_normalized, 'shape') else None
            pred_f = preds_arr.shape[0] if preds_arr.ndim >= 1 else 0

            seq_name = None
            if 'sequence_name' in batch:
                seq_name = batch['sequence_name']
                if isinstance(seq_name, (list, tuple)):
                    seq_name = seq_name[0]

            # If there are no predicted frames or no ground-truth target frames (after cond_num), fail fast
            if pred_f == 0 or img_f is None or img_f == 0:
                logger.error(f"Empty preds or images: pred_f={pred_f}, img_f={img_f}, cond_num={cond_num}, nframe={config.nframe}, batch_keys={list(batch.keys())}, sequence_name={seq_name}")
                raise RuntimeError("Validation pipeline produced empty predictions or images. Check dataset postprocessing and pipeline.")

            if cond_num >= img_f:
                logger.error(f"cond_num >= available frames: cond_num={cond_num}, img_f={img_f}, config.nframe={config.nframe}, sequence_name={seq_name}")
                raise RuntimeError("cond_num is greater than or equal to number of frames in sample — no target frames to evaluate.")

            if pred_f <= cond_num:
                logger.error(f"Predicted frames ({pred_f}) <= cond_num ({cond_num}); no preds for target frames. sequence_name={seq_name}")
                raise RuntimeError("Pipeline produced fewer predicted frames than expected (<= cond_num).")

            # attention visualization 데이터 가져오기
            attention_loss_data = {}
            attention_images_data = {}
            if do_attn_visualize and attention_callback is not None:
                structured_losses = attention_callback.get_structured_losses()
                attention_images = attention_callback.get_attention_images()
                print(f"[DEBUG] structured_losses keys: {list(structured_losses.keys()) if structured_losses else 'None'}")
                print(f"[DEBUG] attention_images keys: {list(attention_images.keys()) if attention_images else 'None'}")
                print(f"[DEBUG] step_losses length: {len(attention_callback.step_losses)}")
                print(f"[DEBUG] layer_losses keys: {list(attention_callback.layer_losses.keys())}")
                
                # attention loss 데이터 준비 (레이어별)
                layer_summary = structured_losses.get('layer_summary', {})
                overall_summary = structured_losses.get('overall_summary', {})
                
                attention_loss_data = {}
                
                # 전체 평균 추가
                if overall_summary:
                    attention_loss_data[f"val/attention_loss/overall_mean"] = overall_summary.get('mean', 0.0)
                
                # 레이어별 loss 추가
                for layer_key, layer_stats in layer_summary.items():
                    layer_mean = layer_stats.get('mean', 0.0)
                    attention_loss_data[f"val/attention_loss/{layer_key}"] = layer_mean
                    
                    # correlation 분석을 위해 레이어별로 수집
                    if layer_key not in attention_loss_scores:
                        attention_loss_scores[layer_key] = []
                    attention_loss_scores[layer_key].append(layer_mean)
                
                # 공통 메타데이터 추가
                if attention_loss_data:
                    attention_loss_data.update({
                        f"val/sample_name": seq_name if seq_name is not None else f"sample_{val_iter}",
                        "val_sample_idx": val_iter
                    })
                
                # attention 이미지 데이터 준비
                attention_images_data = attention_images
                    
            # if batch['tag'][0] not in show_save_dict or show_save_dict[batch['tag'][0]] < 10:  # 每个dataset显示10个
                # show_save_dict[batch['tag'][0]] += 1
            # if val_iter < viz_len:
            #     if depth is not None:
            #         show_image = get_show_images(image_normalized, preds, cond_num, batch["depth"])
            #     else:
            #         show_image = get_show_images(image_normalized, preds, cond_num)

            gt_images = (image_normalized[cond_num:].permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 # -1 1 → 0 1
            preds = preds[cond_num:]

            if compute_fid:
                gt_imgs_fid, preds_fid = torch.tensor(gt_images).permute(0, 3, 1, 2),  torch.tensor(preds).permute(0, 3, 1, 2)
                gt_imgs_fid, preds_fid = accelerator.gather(gt_imgs_fid), accelerator.gather(preds_fid)
                if accelerator.is_main_process:
                    fid_calculator.update(einops.rearrange(gt_imgs_fid,'... c h w -> (...) c h w'), real=True)
                    fid_calculator.update(einops.rearrange(preds_fid,'... c h w -> (...) c h w'), real=False)

            # compute per-frame metrics for this sample
            sample_frame_psnr = []
            sample_frame_ssim = []
            sample_frame_lpips = []
            
            for i in range(preds.shape[0]):  # 이 샘플의 target frames에 대해
                psnr_ = peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0)
                ssim_ = structural_similarity(cv2.cvtColor(gt_images[i], cv2.COLOR_RGB2GRAY),
                                              cv2.cvtColor(preds[i], cv2.COLOR_RGB2GRAY), data_range=1.0)
                lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device)
                
                # 전역 리스트에 추가 (전체 평균 계산용)
                psnr_scores.append(psnr_)
                ssim_scores.append(ssim_)
                lpips_scores.append(lpips_)
                
                # 샘플별 리스트에 추가 (이 샘플의 평균 계산용)
                sample_frame_psnr.append(psnr_)
                sample_frame_ssim.append(ssim_)
                sample_frame_lpips.append(lpips_)

            # 이 샘플의 평균 메트릭 계산 (여러 target frames의 평균)
            sample_psnr_mean = float(np.mean(sample_frame_psnr)) if len(sample_frame_psnr) > 0 else 0.0
            sample_ssim_mean = float(np.mean(sample_frame_ssim)) if len(sample_frame_ssim) > 0 else 0.0
            sample_lpips_mean = float(np.mean(sample_frame_lpips)) if len(sample_frame_lpips) > 0 else 0.0

            # wandb 로깅 (메인 프로세스에서만) - 각 샘플별로 개별 global step 사용
            if accelerator.is_main_process:
                import wandb
                
                # 시퀀스 이름 결정
                seq_name_log = f"sample_{val_iter:03d}"
                if 'sequence_name' in batch:
                    seq_batch = batch['sequence_name']
                    if isinstance(seq_batch, (list, tuple)) and len(seq_batch) > 0:
                        seq_name_log = str(seq_batch[0])
                    elif seq_batch is not None:
                        seq_name_log = str(seq_batch)

                # 이미지 생성
                if depth is not None:
                    show_image_local = get_show_images(image_normalized, preds, cond_num, batch["depth"])
                else:
                    show_image_local = get_show_images(image_normalized, preds, cond_num)

                # wandb 로깅 데이터 수집집
                log_data = {
                    f"val/generated": wandb.Image(show_image_local),
                    f"val/sample_name": seq_name_log,  # 샘플 이름 추가
                    f"val/psnr": sample_psnr_mean,
                    f"val/ssim": sample_ssim_mean,
                    f"val/lpips": sample_lpips_mean,
                }
                
                # attention loss 데이터가 있으면 추가
                if attention_loss_data:
                    log_data.update(attention_loss_data)
                
                # attention 이미지들이 있으면 추가 (callback에서 키에 step 포함됨)
                if attention_images_data:
                    for key, img_data in attention_images_data.items():
                        log_data[key] = wandb.Image(img_data['image'], caption=img_data['caption'])

                # callback의 step-level loss들도 wandb에 업로드
                if do_attn_visualize and attention_callback is not None:
                    loss_dict = attention_callback.get_loss_dict()
                    for loss_key, loss_val in loss_dict.items():
                        # loss_val may be a tensor; coerce to float for logging
                        try:
                            if hasattr(loss_val, 'item'):
                                log_data[loss_key] = float(loss_val.item())
                            else:
                                log_data[loss_key] = float(loss_val)
                        except Exception:
                            # fallback: store as-is if conversion fails
                            log_data[loss_key] = loss_val
                
                print(f"[DEBUG] log_data keys: {list(log_data.keys())}")
                # 모든 데이터를 한 번에 로깅
                accelerator.log(log_data, step=val_iter)
                
                # 로깅 완료 후 callback 상태 초기화 (다음 배치를 위해)
                if do_attn_visualize and attention_callback is not None:
                    attention_callback.reset()
            
            val_iter += 1 
    
    # unify all results
    # If no metrics were computed, raise an error so the root cause can be investigated
    if len(psnr_scores) == 0 or len(ssim_scores) == 0 or len(lpips_scores) == 0:
        logger.error("No valid metric samples collected during validation. This likely indicates every sample had empty targets or failed processing.")
        raise RuntimeError("Validation produced no metric samples (empty psnr/ssim/lpips lists). Check cond_num, data frames and pipeline outputs.")
    
    # Attention loss 전체 통계 로깅 (모든 샘플 종합)
    
    # 전체 validation에 대한 평균 메트릭 계산 (모든 frames의 평균)
    overall_psnr = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
    overall_ssim = torch.tensor(np.mean(ssim_scores), device=device, dtype=torch.float32)
    overall_lpips = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)

    # Multi-GPU 환경에서 평균 계산
    overall_psnr = accelerator.gather(overall_psnr).mean().item()
    overall_ssim = accelerator.gather(overall_ssim).mean().item()
    overall_lpips = accelerator.gather(overall_lpips).mean().item()
    if compute_fid:
        if accelerator.is_main_process:
            fid_val = fid_calculator.compute()
            real_num, fake_num = fid_calculator.real_features_num_samples, fid_calculator.fake_features_num_samples
            fid_calculator.reset()
            del fid_calculator
            torch.cuda.empty_cache()
            accelerator.log({"summary/fid": fid_val, "summary/fid_real_num": real_num, "summary/fid_fake_num": fake_num}, step=step)

    if accelerator.is_main_process:
        # 전체 validation 요약 메트릭 로깅 (별도의 높은 step 사용)
        summary_step = 99999  # 요약 데이터는 별도 step에 저장
        total_samples = val_iter  # 실제 처리된 샘플 수
        total_frames = len(psnr_scores)  # 실제 처리된 프레임 수
        
        summary_log_data = {
            "summary/psnr_avg": overall_psnr,  # 모든 프레임의 평균
            "summary/ssim_avg": overall_ssim, 
            "summary/lpips_avg": overall_lpips,
            "summary/total_samples": total_samples,  # 샘플 수
            "summary/total_frames": total_frames,    # 프레임 수
            "summary/frames_per_sample": total_frames / max(1, total_samples)  # 샘플당 평균 프레임 수
        }
        
        # 레이어별 attention loss 통계 추가
        for layer_key, layer_scores in attention_loss_scores.items():
            if layer_scores:
                # unet 숫자만 추출 (correlation과 동일한 로직)
                import re
                unet_match = re.search(r'unet(\d+)', layer_key)
                layer_match = re.search(r'_(\d+)$', layer_key)
                
                if unet_match and layer_match:
                    unet_num = unet_match.group(1)
                    layer_num = layer_match.group(1)
                    clean_layer_name = f'UNet{unet_num}_L{layer_num}'
                else:
                    clean_layer_name = layer_key.replace('_', '')[:10]  # fallback
                
                summary_log_data[f"summary/attention_loss/{clean_layer_name}_mean"] = np.mean(layer_scores)
                summary_log_data[f"summary/attention_loss/{clean_layer_name}_std"] = np.std(layer_scores)
                summary_log_data[f"summary/attention_loss/{clean_layer_name}_min"] = np.min(layer_scores)
                summary_log_data[f"summary/attention_loss/{clean_layer_name}_max"] = np.max(layer_scores)
        
        accelerator.log(summary_log_data, step=summary_step)

        # 샘플 이미지들 로깅 (요약용)
        for j in range(min(len(show_images), 5)):
            img = show_images[j]
            if config.image_size > 256:
                img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            accelerator.log({f"val/summary/sample_image_{j}": wandb.Image(img)}, step=summary_step)

    del loss_fn_alex
    
    # Validation 완료 후 모든 attention cache 정리
    if do_attn_visualize:
        from src.distill_utils.attn_processor_cache import unset_attn_cache
        # UNet attention cache 설정 해제
        unset_attn_cache(pipeline.unet)
        logger.info("Unset UNet attention cache after validation")
    
    torch.cuda.empty_cache()

    return overall_lpips



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument("--visualize_attention_maps", action="store_true", help="Whether to visualize attention maps.")
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
        "--validation_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether to validate a specific checkpoint. Use a path saved by"
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
    parser.add_argument("--val_cfg", type=float, default=1.0)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--config_file", type=str, default="configs/cat3d.yaml")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    try: 
        config = EasyDict(OmegaConf.load(os.path.join(args.val_path, "config.yaml")))
    except:
        if args.config_file is None:
            raise FileNotFoundError(f"No config file found in {args.val_path} and {args.config_file}")
        print(f"Warning: No config file found in {args.val_path}, using {args.config_file}")
        config = EasyDict(OmegaConf.load(args.config_file))

    # Sanity checks
    # assert config.dataset_names is not None and len(config.dataset_names) > 0



    return args, config


def main():
    args, config = parse_args()
    logging_dir = os.path.join(args.val_path, "val_logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.val_path, logging_dir=logging_dir)

    accelerator = Accelerator(
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
        if args.val_path is not None:
            assert os.path.exists(args.val_path), f"Validation path {args.val_path} does not exist"

    # Load scheduler, tokenizer and models.

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

    ## Seonghu TODO: 임시 visualize config
    visualize_config = {
        'timestep_interval': 1,
        
        # 'vggt_layer': "track_head",
        'vggt_layer': "point_map",
        # 'costmap_metric': 'neg_log_l2',

        'vggt_logit_head': "softmax_headmean",
        'vggt_logit_head_kwargs': {"softmax_temp": 0.01},
        'unet_logit_head': "softmax_headmean",
        'unet_logit_head_kwargs': {"softmax_temp": 1.0},
        # 'costmap_metric': 'dot_product',
        
        # Loss 계산용 설정 (train과 동일하게)
        'loss_query': "target",      # loss 계산 시 사용할 query
        'loss_key': "reference",     # loss 계산 시 사용할 key
        # Visualization용 설정 (시각화에서만 사용)
        'viz_query': "target",       # 시각화 시 사용할 query
        'viz_key': "all",            # 시각화 시 사용할 key (self attention 포함)
        'student_unet_attn_layers': list((2,4,6,8,10,12)),
        
        
        # 시각화 설정 추가
        # 빈 리스트 = 모든 스텝에서 시각화/로그, 아니면 명시된 스텝만 처리
        'viz_steps': [],  # visualization (images) 저장/생성 스텝
        'loss_steps': [0, 10, 20, 30, 40],                 # loss 계산/수집에 사용할 스텝 (빈 리스트 = 모든 스텝)
        # per_head_loss: if True, compute loss per attention head and log each head separately
        'per_head_loss': False,
        'viz_log_wandb': True,
        'viz_alpha': 0.6,
        'viz_query_xy': None,  # None이면 랜덤 선택
        'viz_query_index': 150,  # None이면 랜덤 선택
        # pairs: list of dicts defining per-pair settings
        # format: {'unet_layer': int, 'vggt_layer': str_or_int, 'costmap_metric': str, 'loss_fn': str}
        "softargmax_num_key_views": 2,
        'pairs': [
            {'unet_layer': l, 'vggt_layer': 'point_map', 'costmap_metric': 'inverse_l2', 'loss_fn': 'softargmax_l2'}
            for l in (2, 4, 6, 8, 10, 12)
        ] + [
            {'unet_layer': l, 'vggt_layer': 'point_map', 'costmap_metric': 'inverse_l2', 'loss_fn': 'argmax_l2'}
            for l in (2, 4, 6, 8, 10, 12)
        ],
    }
    # costmap metric 설정: 'neg_log_l2', 'neg_l2', 'inverse_l2', 'dot_product'
    vggt_layer = visualize_config.get('vggt_layer', None)
    do_attn_visualize = args.visualize_attention_maps
    if do_attn_visualize:
        # if hasattr(config, "distill_config"):
        #     vggt_visualize_config = {
        #         "cache_attn_layer_ids": list(set([p[1] for p in config.distill_config.distill_pairs if isinstance(p[1], int)])),
        #         "cache_costmap_types": list(set([p[1] for p in config.distill_config.distill_pairs if isinstance(p[1], str)]))
        #     }
        # else:
        #     ## seonghu TODO: default vggt layer; track head
        #     print("WARNING: No vggt layer provided for visualize_attention_maps. Defaulting to track head")
        #     vggt_visualize_config = {
        #         "cache_attn_layer_ids": [],
        #         "cache_costmap_types": ["track_head"]
        #     }
        from vggt.models.vggt import VGGT
        
        # derive cache_costmap_types from visualize_config['pairs'] (support dict entries)
        pairs_list = visualize_config.get('pairs', [])
        cache_costmap_types = []
        for p in pairs_list:
            if isinstance(p, dict):
                vt = p.get('vggt_layer', None)
            else:
                try:
                    _, vt = p
                except Exception:
                    vt = None
            if vt is not None:
                if vt not in cache_costmap_types:
                    cache_costmap_types.append(vt)
        if not cache_costmap_types:
            # fallback to single vggt_layer if provided, otherwise to track_head
            if vggt_layer is not None:
                cache_costmap_types = [vggt_layer]
            else:
                cache_costmap_types = ["track_head"]

        print(f"vggt caching costmap types: {cache_costmap_types}")
        vggt_visualize_config = {
            "cache_attn_layer_ids": [],
            "cache_costmap_types": cache_costmap_types
        }

        # with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Keep VGGT in full precision here; mixed dtypes caused xformers attention to fail
        # (query/key/value dtype mismatch). We rely on later `.to(device, dtype=weight_dtype)`
        # to cast weights appropriately.
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_visualize_config).eval()
        for p in vggt_model.parameters():
            p.requires_grad = False
        
    else:
        vggt_model = None
        assert not config.get("use_vggt_camera", False), "use_vggt_camera is only available when vggt_on_fly is set"

    
    # Save attention maps to a separate directory sibling to `args.val_path` (not inside `val/`)
    if args.val_path:
        attn_root = os.path.join(os.path.dirname(args.val_path), "attn_maps")
    else:
        attn_root = os.path.join(os.getcwd(), "attn_maps")
    visualize_config['viz_save_dir'] = attn_root
    # visualize_config['pairs'] is directly used (can be list of dicts)
    ## =====================================
    
    if do_attn_visualize:
        # logit heads: input: [B, Head, Q, K]
        unet_logit_head = LOGIT_HEAD_CLS[visualize_config['unet_logit_head'].lower()](**visualize_config['unet_logit_head_kwargs'])
        vggt_logit_head = LOGIT_HEAD_CLS[visualize_config['vggt_logit_head'].lower()](**visualize_config['vggt_logit_head_kwargs'])
        vggt_layer = visualize_config['vggt_layer']

        # Set Attention Cache for distillation
        ## TODO Seonghu : check max layer number
        visualize_student_unet_attn_layers = visualize_config['student_unet_attn_layers']
        print(f"Number of unet attention layers: {len(list(unet.attn_processors.keys()))}")
        set_attn_cache(unet, visualize_student_unet_attn_layers)
    
        # add to unet (only single module supported in deepspeed... fuk )
        unet.vggt_logit_head = vggt_logit_head
        unet.unet_logit_head = unet_logit_head
        del vggt_logit_head, unet_logit_head
    else:
        visualize_pairs = visualize_config['pairs']
        visualize_student_unet_attn_layers = []
        
    # set feat cache for repa distill # JIHO TODO: repa 
    # from src.modules.attn_processor_cache import set_feat_cache, unset_feat_cache, pop_cached_feat, clear_feat_cache
    
    # Assume no ema
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

    # if config.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # set trainable parameters
   
    # # Get the datasets
    # if cfg is None:
    #     cfg = dict()
    #     for data_name in config.dataset_names:
    #         if "_easy" in data_name:
    #             data_name = data_name.replace("_easy", "")
    #             cfg[data_name] = EasyDict(OmegaConf.load(f"configs/datasets/{data_name}_easy.yaml"))
    #         else:
    #             cfg[data_name] = EasyDict(OmegaConf.load(f"configs/datasets/{data_name}.yaml"))
    #         cfg[data_name]["image_height"] = config.image_size
    #         cfg[data_name]["image_width"] = config.image_size

    # train_dataset, val_dataset = load_global_dataset(config, cfg, rank=accelerator.process_index)
    # Dataset jiho TODO
    # from src.datasets.re10k_minseop import build_re10k_minseop    

    from src.datasets.re10k_wds import build_re10k_wds

    val_wds_dataset_config = {'url_paths': [ "/mnt/data2/minseop/realestate_val_wds", ],
        'dataset_length': 53,
        'resampled': False,
        'shardshuffle': False,
        'num_viewpoints': 3,
        # min_view_range=6,
        # max_view_range=6,
        'inference': True,
        'inference_view_range': 8,
        'process_kwargs': {
            'get_square_extrinsic': True
        }
    }
    
    ## TODO: for validation, set nframe to 3
    config.nframe = 3
    config.fix_cond_num = 2
    
    val_dataset = build_re10k_wds(
        **val_wds_dataset_config
    ) 
    
    
    val_num_workers = 0
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=data_config.get("eval_batch_size", 1),
    #     # persistent_workers=True,
    #     num_workers=0, # for iterableDataset(maybe), num_workers should be 0 (or same data iters)
    # )

    if accelerator.is_main_process:
        OmegaConf.save(dict(config), os.path.join(args.val_path, 'config.yaml'))
        # for data_name in cfg:
        #     OmegaConf.save(dict(cfg[data_name]), os.path.join(args.output_dir, f'{data_name}.yaml'))
    
    sampler = None
    # sampler = GlobalConcatSampler(train_dataset,
    #                               n_frames_per_sample=config.nframe,
    #                               shuffle=True,
    #                               dynamic_sampling=config.get("dynamic_sampling", False),
    #                               multi_scale=config.get("multi_scale", False),
    #                               batch_per_gpu=config.train_batch_size // config.nframe,
    #                               rank=accelerator.process_index,
    #                               num_replicas=accelerator.num_processes,
    #                               data_config=cfg)
    val_sampler = None
    # val_sampler = GlobalConcatSampler(val_dataset,
    #                                   n_frames_per_sample=config.nframe,
    #                                   shuffle=False,
    #                                   rank=accelerator.process_index,
    #                                   multi_scale=config.get("multi_scale", False),
    #                                   batch_per_gpu=1,
    #                                   num_replicas=accelerator.num_processes,
    #                                   data_config=cfg)

    # assert config.train_batch_size % config.nframe == 0

    # Validation DataLoaders creation:
    # Collate function: filter out keys that are non-tensor (e.g., strings) to avoid
    # accelerate concatenation errors when gathering batches across processes.
    def _validate_collate_fn(batch_list):
        # batch_list is a list of dataset samples (EasyDict or dict)
        # Build a dict containing only tensor-like keys; skip strings/None to avoid
        # accelerate concatenation errors. Always return a dict.
        if not batch_list:
            return {}

        merged = {}
        sample0 = batch_list[0]
        for k in sample0.keys():
            vals = [s.get(k, None) for s in batch_list]
            # skip keys with missing values
            if any(v is None for v in vals):
                continue

            # if all are tensors, stack
            if all(isinstance(v, torch.Tensor) for v in vals):
                try:
                    merged[k] = torch.stack(vals, dim=0)
                    continue
                except Exception:
                    pass

            # try converting numpy arrays/scalars to tensors
            converted = []
            can_convert = True
            for v in vals:
                if isinstance(v, torch.Tensor):
                    converted.append(v)
                else:
                    try:
                        converted.append(torch.as_tensor(v))
                    except Exception:
                        can_convert = False
                        break

            if can_convert:
                try:
                    merged[k] = torch.stack(converted, dim=0)
                except Exception:
                    # if stacking fails, keep converted list
                    merged[k] = converted
            else:
                # skip non-convertible keys (e.g., strings)
                continue

        return merged

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=1,
        num_workers=val_num_workers,
        collate_fn=_validate_collate_fn,
    )

    # Scheduler and math around the number of training steps.
    print("Loading trained weights for validation...")
    if args.validation_checkpoint != "latest":
        path = os.path.basename(args.validation_checkpoint)
        resume_path = args.val_path
    else:
        # Get the most recent checkpoint (keep original simple logic)
        resume_path = args.val_path if args.val_path is not None else args.val_path
        # Basic safety: ensure resume_path exists before listing
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Validation path does not exist: {resume_path}")
        dirs = os.listdir(resume_path)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        # keep original numeric sort but guard against malformed names
        def _idx(name):
            try:
                return int(name.split("-")[1])
            except Exception:
                return -1
        dirs = sorted(dirs, key=_idx)
        path = dirs[-1] if len(dirs) > 0 else None

    # Minimal existence checks while preserving original behavior
    if resume_path is None or path is None:
        raise FileNotFoundError(f"No checkpoint path determined. resume_path={resume_path}, path={path}")

    ckpt_file = os.path.join(resume_path, path, "pytorch_model", "mp_rank_00_model_states.pt")
    if not os.path.exists(ckpt_file):
        # helpful message: show what checkpoint dirs exist (possibly empty)
        available = []
        try:
            available = [d for d in os.listdir(resume_path) if d.startswith("checkpoint")]
        except Exception:
            available = []
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}. Available checkpoint dirs: {available}")

    # reload weights for unet here
    weights = torch.load(ckpt_file, map_location="cpu")
    unet.load_state_dict(weights['module'], strict=False)  # we usually need to resume partial weights here
    print(f"Loaded weights for unet from {ckpt_file}")

    # Prepare 
    # If accelerate/DeepSpeed is enabled in the environment, ensure validation does not use ZeRO stage>0
    # (validation has no optimizer and DeepSpeed stage 2 requires an optimizer). Force zero stage 0
    # and set micro batch size for validation so DS initialization won't require an optimizer.
    if accelerate.state.is_initialized() and AcceleratorState().deepspeed_plugin is not None:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        # set per-device batch size for validation
        ds_cfg["train_micro_batch_size_per_gpu"] = config.get("val_batch_size", 1)
        # force zero optimization to stage 0
        if "zero_optimization" in ds_cfg:
            try:
                ds_cfg["zero_optimization"]["stage"] = 0
            except Exception:
                ds_cfg["zero_optimization"] = {"stage": 0}
        else:
            ds_cfg["zero_optimization"] = {"stage": 0}

    # Set weight dtype according to accelerator BEFORE prepare and model operations
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    logger.info(f"Using weight_dtype: {weight_dtype} based on accelerator.mixed_precision: {accelerator.mixed_precision}")

    # Cast all models to correct dtype BEFORE prepare
    unet.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    if do_attn_visualize and vggt_model is not None:
        vggt_model.to(dtype=weight_dtype)
        logger.info(f"VGGT model cast to {weight_dtype}")

    # Pass validation dataloader into prepare so DeepSpeed/Accelerate can infer batch size when needed
    unet, val_dataloader = accelerator.prepare(unet, val_dataloader)

    # Move non-prepared models to device (they already have correct dtype from above)
    vae.to(accelerator.device)
    if do_attn_visualize and vggt_model is not None:
        vggt_model.to(accelerator.device)
        logger.info(f"VGGT model moved to device: {accelerator.device}")
        
        # Verify all VGGT model buffers are on correct device
        for name, buffer in vggt_model.named_buffers():
            if buffer.device != accelerator.device:
                logger.warning(f"VGGT buffer {name} is on {buffer.device}, moving to {accelerator.device}")
                buffer.data = buffer.data.to(accelerator.device)
        
        # Double-check the problematic buffers specifically
        if hasattr(vggt_model, 'aggregator') and hasattr(vggt_model.aggregator, '_resnet_mean'):
            logger.info(f"VGGT _resnet_mean device: {vggt_model.aggregator._resnet_mean.device}")
            logger.info(f"VGGT _resnet_std device: {vggt_model.aggregator._resnet_std.device}")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            init_kwargs={"wandb": {"name": args.run_name}} if args.run_name is not None else {},
        )
        
        # wandb 초기화 확인 및 메트릭 정의
        print("Wandb initialized for validation")
        
        # attention loss 메트릭을 validation step 기준으로 정의
        import wandb
        wandb.define_metric("val/attention_loss/*")
        wandb.define_metric("val/psnr") 
        wandb.define_metric("val/ssim")
        wandb.define_metric("val/lpips")

    # Validate!
    total_batch_size = 1 * accelerator.num_processes

    logger.info("***** Running Validation *****")
    logger.info(f"  Num examples = {len(val_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {1}")
    logger.info(f"  Total validation batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0

    device = unet.device
    
    ##

    # Build pipeline with the same weight dtype as training so autocast behaves consistently
    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
    
    _ = log_validation(accelerator=accelerator, config=config, args=args,
                            pipeline=pipeline, val_dataloader=val_dataloader,
                            step=global_step, device=device, weight_dtype=weight_dtype, vae=vae, 
                            do_attn_visualize=do_attn_visualize, vggt_model=vggt_model,
                            visualize_config=visualize_config)

    
    torch.cuda.empty_cache()
    gc.collect()
    
                    


if __name__ == "__main__":
    main()
