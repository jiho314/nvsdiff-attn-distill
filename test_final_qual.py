from train import *
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
from torchvision import transforms
# New imports for local saving
import os
from PIL import Image
import numpy as np
import cv2


logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_test(accelerator, config, args, pipeline, dataloader, step, device,
             process_batch_fn = lambda x: x,
             eval_size = 512,
             **kwargs):
    ''' Caution, batch=1 !
    '''
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    compute_fid = args.test_compute_fid
    viz_len = args.test_viz_num
    if compute_fid:
        if accelerator.is_main_process:
            fid_calculator = FrechetInceptionDistance(normalize=True).to(device)
            fid_calculator.reset()

    resize_fn = transforms.Resize(eval_size, interpolation=transforms.InterpolationMode.BICUBIC)
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    show_images = [] # Only used if args.use_wandb is True
    val_iter = 0

    cond_num = args.test_cond_num
    nframe = args.test_nframe
    cfg = args.test_cfg
    num_inference_steps = args.test_num_inference_steps
    if args.individual_test:
        target_ids = set(args.test_viz_id)
        found_ids_count = 0
        if not target_ids:
            logger.warning("Individual test mode is enabled, but no IDs were provided with --test_viz_id.")
            return # Exit early if there's nothing to do


    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            # --- 1. Data Loading and Preprocessing ---
            batch = process_batch_fn(batch)
            image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
            
            image = image[:nframe]
            intri_ = intri_[:nframe]
            extri_ = extri_[:nframe]
            
            f, _, h, w = image.shape
            assert f == nframe, f"args.test_nframe({args.test_nframe}) doesn't match with data nframe({f}), image_shape({image.shape}) "
            image_normalized = image * 2.0 - 1.0
            extrinsic, intrinsic = extri_, intri_

            # --- 2. Model Inference ---
            tag, sequence_name, depth = None, None , None
            preds = pipeline(images=image_normalized, nframe=nframe, cond_num=cond_num,
                             height=image_normalized.shape[2], width=image_normalized.shape[3],
                             intrinsics=intrinsic, extrinsics=extrinsic,
                             num_inference_steps=num_inference_steps, guidance_scale=cfg,
                             output_type="np", config=config, tag=tag,
                             sequence_name=sequence_name, depth=depth, vae=kwargs['vae']).images

            # --- 3. Visualization Logic ---
            process_this_batch = False
            if args.individual_test:
                if val_iter in target_ids:
                    process_this_batch = True
            else:
                if val_iter < viz_len:
                    process_this_batch = True

            if process_this_batch:
                if args.use_wandb:
                    if depth is not None:
                        show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num, batch["depth"])
                    else:
                        show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num)
                    show_images.append(show_image)
                else:
                    if accelerator.is_main_process:
                        sample_dir = os.path.join(args.output_dir, f"sample_{val_iter:04d}")
                        os.makedirs(sample_dir, exist_ok=True)

                        # --- MODIFIED HELPER: More robust for reuse ---
                        def to_numpy_uint8(img_data):
                            """Converts a PyTorch tensor or a NumPy array to a uint8 NumPy array (H, W, C)."""
                            if isinstance(img_data, torch.Tensor):
                                # From (C, H, W) to (H, W, C)
                                img_np = (img_data.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            else:
                                # Assumes (H, W, C) already
                                img_np = (img_data * 255).astype(np.uint8)
                            return img_np

                        # --- EXISTING LOGIC to save individual images ---
                        for i in range(cond_num):
                            path = os.path.join(sample_dir, f"input_view_{i+1}.jpg")
                            Image.fromarray(to_numpy_uint8(image[i])).save(path)

                        for i in range(cond_num, nframe):
                            path = os.path.join(sample_dir, f"target_view_{i - cond_num + 1}.jpg")
                            Image.fromarray(to_numpy_uint8(image[i])).save(path)

                        generated_preds = preds[cond_num:]
                        for i, pred_img in enumerate(generated_preds):
                            path = os.path.join(sample_dir, f"generated_view_{i+1}.jpg")
                            Image.fromarray(to_numpy_uint8(pred_img)).save(path)
                        
                        # --- NEW CODE START: Stitch and save the combined image ---
                        try:
                            # Note: This assumes one generated image (nframe=3, cond_num=2)
                            # Your request order: (generated | target | input1 | input2)
                            generated_img_np = to_numpy_uint8(preds[cond_num])
                            target_img_np    = to_numpy_uint8(image[cond_num])
                            input1_img_np    = to_numpy_uint8(image[0])
                            input2_img_np    = to_numpy_uint8(image[1])
                            
                            # Concatenate images horizontally
                            stitched_image = np.concatenate(
                                [generated_img_np, target_img_np, input1_img_np, input2_img_np], 
                                axis=1
                            )

                            # Save the final stitched image
                            stitched_path = os.path.join(sample_dir, "stitched_comparison.jpg")
                            Image.fromarray(stitched_image).save(stitched_path)

                        except IndexError as e:
                            logger.warning(f"Could not create stitched image for sample {val_iter}. "
                                           f"Check if nframe/cond_num settings are as expected. Error: {e}")
                        except Exception as e:
                            logger.warning(f"An unexpected error occurred while creating stitched image for sample {val_iter}: {e}")
                        # --- NEW CODE END ---
                
                if args.individual_test:
                    found_ids_count += 1
    # with torch.no_grad(), torch.autocast("cuda"):
    #     for batch in tqdm(dataloader, desc=f"Validation rank{accelerator.process_index}..."):
    #         batch = process_batch_fn(batch)
    #         image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
    #         # ✂️ Add these three lines to trim the data from 4 frames to 3
    #         image = image[:nframe]
    #         intri_ = intri_[:nframe]
    #         extri_ = extri_[:nframe]
    #         f, _, h, w = image.shape
    #         assert f == nframe, f"args.test_nframe({args.test_nframe}) doesn't match with data nframe({f}), image_shape({image.shape}) "
    #         image_normalized = image * 2.0 - 1.0

    #         extrinsic, intrinsic = extri_, intri_
    #         tag, sequence_name, depth = None, None , None
    #         preds = pipeline(images=image_normalized, nframe=nframe, cond_num=cond_num,
    #                          height=image_normalized.shape[2], width=image_normalized.shape[3],
    #                          intrinsics=intrinsic, extrinsics=extrinsic,
    #                          num_inference_steps=num_inference_steps, guidance_scale=cfg,
    #                          output_type="np", config=config, tag=tag,
    #                          sequence_name=sequence_name, depth=depth, vae=kwargs['vae']).images

    #         # --- VISUALIZATION LOGIC ---
    #         process_this_batch = False
    #         if args.individual_test:
    #             if val_iter == args.test_viz_id:
    #                 process_this_batch = True
    #         else:
    #             if val_iter < viz_len:
    #                 process_this_batch = True
            
    #         if process_this_batch:
    #             if args.use_wandb:
    #                 # Original logic: create grid for wandb
    #                 if depth is not None:
    #                     show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num, batch["depth"])
    #                 else:
    #                     show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num)
    #                 show_images.append(show_image)
    #             else:
    #                 # New logic: save individual images locally
    #                 if accelerator.is_main_process:
    #                     sample_dir = os.path.join(args.output_dir, f"sample_{val_iter:04d}")
    #                     os.makedirs(sample_dir, exist_ok=True)

    #                     def save_image_local(img_data, file_path):
    #                         """Helper to convert tensor/numpy array to savable image."""
    #                         if isinstance(img_data, torch.Tensor): # Ground Truth (C, H, W)
    #                             img_np = (img_data.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #                         else: # Prediction (H, W, C)
    #                             img_np = (img_data * 255).astype(np.uint8)
    #                         Image.fromarray(img_np).save(file_path)

    #                     # Save Input Views (from ground truth)
    #                     for i in range(cond_num):
    #                         path = os.path.join(sample_dir, f"input_view_{i+1}.jpg")
    #                         save_image_local(image[i], path)

    #                     # Save Target Views (from ground truth)
    #                     for i in range(cond_num, nframe):
    #                         path = os.path.join(sample_dir, f"target_view_{i - cond_num + 1}.jpg")
    #                         save_image_local(image[i], path)

    #                     # Save Generated Views (from predictions)
    #                     generated_preds = preds[cond_num:]
    #                     for i, pred_img in enumerate(generated_preds):
    #                         path = os.path.join(sample_dir, f"generated_view_{i+1}.jpg")
    #                         save_image_local(pred_img, path)    

    #         # If we are in individual mode and just processed the target scene, stop the loop.
    #         if args.individual_test and process_this_batch:
    #             logger.info(f"Successfully processed individual scene ID {args.test_viz_id}. Stopping iteration.")
    #             break                           

            # if val_iter < viz_len:
            #     if args.use_wandb:
            #         # Original logic: create grid for wandb
            #         if depth is not None:
            #             show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num, batch["depth"])
            #         else:
            #             show_image = get_show_images(image, torch.tensor(preds).permute(0,3,1,2), cond_num)
            #         show_images.append(show_image)
            #     else:
            #         # New logic: save individual images locally
            #         if accelerator.is_main_process:
            #             sample_dir = os.path.join(args.output_dir, f"sample_{val_iter:04d}")
            #             os.makedirs(sample_dir, exist_ok=True)

            #             def save_image_local(img_data, file_path):
            #                 """Helper to convert tensor/numpy array to savable image."""
            #                 if isinstance(img_data, torch.Tensor): # Ground Truth (C, H, W)
            #                     img_np = (img_data.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            #                 else: # Prediction (H, W, C)
            #                     img_np = (img_data * 255).astype(np.uint8)
            #                 Image.fromarray(img_np).save(file_path)

            #             # Save Input Views (from ground truth)
            #             for i in range(cond_num):
            #                 path = os.path.join(sample_dir, f"input_view_{i+1}.jpg")
            #                 save_image_local(image[i], path)

            #             # Save Target Views (from ground truth)
            #             for i in range(cond_num, nframe):
            #                 path = os.path.join(sample_dir, f"target_view_{i - cond_num + 1}.jpg")
            #                 save_image_local(image[i], path)

            #             # Save Generated Views (from predictions)
            #             generated_preds = preds[cond_num:]
            #             for i, pred_img in enumerate(generated_preds):
            #                 path = os.path.join(sample_dir, f"generated_view_{i+1}.jpg")
            #                 save_image_local(pred_img, path)



            # --- 4. Metric Calculation (Unchanged) ---
            gt_images = (image[cond_num:].permute(0, 2, 3, 1).cpu().numpy())
            preds_for_metric = preds[cond_num:]
            if eval_size != h:
                gt_images = resize_fn(gt_images)
                preds_for_metric = resize_fn(preds_for_metric)

            if compute_fid:
                gt_imgs_fid = torch.tensor(gt_images).permute(0, 3, 1, 2).to(device)
                preds_fid = torch.tensor(preds_for_metric).permute(0, 3, 1, 2).to(device)
                gt_imgs_fid, preds_fid = accelerator.gather(gt_imgs_fid), accelerator.gather(preds_fid)
                if accelerator.is_main_process:
                    gt_imgs_fid = einops.rearrange(gt_imgs_fid,'... c h w -> (...) c h w').to(device)
                    preds_fid = einops.rearrange(preds_fid,'... c h w -> (...) c h w').to(device)
                    fid_calculator.update(gt_imgs_fid, real=True)
                    fid_calculator.update(preds_fid, real=False)

            for i in range(preds_for_metric.shape[0]):
                psnr_ = peak_signal_noise_ratio(gt_images[i], preds_for_metric[i], data_range=1.0)
                psnr_scores.append(psnr_)
                ssim_ = structural_similarity(cv2.cvtColor((gt_images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                                              cv2.cvtColor((preds_for_metric[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), data_range=255.0)
                ssim_scores.append(ssim_)
                lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds_for_metric[i], device)
                lpips_scores.append(lpips_)

            # --- 5. Loop Control (Unchanged) ---
            if args.individual_test and len(target_ids) > 0 and found_ids_count == len(target_ids):
                logger.info(f"Successfully processed all {len(target_ids)} requested scenes: {sorted(list(target_ids))}. Stopping iteration.")
                break
                
            val_iter += 1

    # --- FINAL LOGGING (Unchanged) ---
    log_value_dict = {}
    psnr_score = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
    ssim_score = torch.tensor(np.mean(ssim_scores), device=device, dtype=torch.float32)
    lpips_score = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)

    psnr_score = accelerator.gather(psnr_score).mean().item()
    ssim_score = accelerator.gather(ssim_score).mean().item()
    lpips_score = accelerator.gather(lpips_score).mean().item()
    log_value_dict.update({"test/psnr": psnr_score, "test/ssim": ssim_score, "test/lpips": lpips_score})

    if args.use_wandb:
        accelerator.log(log_value_dict, step=step)
        if accelerator.is_main_process and show_images:
            show_images_full = torch.cat(show_images, dim=1).to(device)
            show_images_full = accelerator.gather(show_images_full).reshape(-1,*show_images_full.shape).permute(1,0,2,3).flatten(1,2)
            accelerator.log({"test/show_images": wandb.Image(show_images_full)}, step=step)
    else:
        if accelerator.is_main_process:
            logger.info(f"Saved {viz_len} visualization samples to {args.output_dir}")
            for key, value in log_value_dict.items():
                logger.info(f"{key}: {value}")


    accelerator.print("test/psnr: ", psnr_score)
    accelerator.print("test/ssim: ", ssim_score)
    accelerator.print("test/lpips: ", lpips_score)

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

            if args.use_wandb:
                log_value_dict.update({"test/fid": fid_val, "test/fid_real_num": real_num, "test/fid_fake_num": fake_num})
                accelerator.log({"test/fid": fid_val}, step=step)
            else:
                 logger.info(f"test/fid: {fid_val}")


    del loss_fn_alex
    torch.cuda.empty_cache()
    return lpips_score


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # --- Add new arguments for logging control ---
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to Weights & Biases.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs when not using wandb.")

    parser.add_argument("--config_file", type=str, default=None)
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
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--restart_global_step", default=0, type=int)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--min_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay_step", type=int, default=10)
    parser.add_argument("--reset_ema_step", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--test_dataset_config", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--test_nframe", type=int, required=True)
    parser.add_argument("--test_cond_num", type=int, required=True)
    parser.add_argument("--test_use_vggt_camera", action="store_true")
    parser.add_argument("--test_run_path", type=str, required=True)
    parser.add_argument("--test_train_step", type=int, required=True)
    parser.add_argument("--test_cfg", type=float, required=True)
    parser.add_argument("--test_num_inference_steps", type=int, default=50, required=True)
    parser.add_argument("--test_viz_num", type=int, default=15)

    # ✅ Add these two new arguments for individual testing
    parser.add_argument("--individual_test", action="store_true", help="Process a single specific scene by its ID.")
    parser.add_argument("--test_viz_id", type=int, nargs='+', default=[], help="A list of specific scene IDs to visualize when --individual_test is enabled.")


    parser.add_argument("--test_compute_fid", action="store_true")
    parser.add_argument("--test_eval_size", type=int, default = 512)
    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.config_file is not None:
        config = EasyDict(OmegaConf.load(args.config_file))
    else:
        args.config_file = os.path.join(args.test_run_path, "config.yaml")
        config = EasyDict(OmegaConf.load(args.config_file))

    cfg = None

    return args, config, cfg


def main():
    args, config, cfg = parse_args()
    
    log_with = "wandb" if args.use_wandb else None
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
    )

    if not args.use_wandb and accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

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

    set_seed(args.seed)


    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    accelerator.print("Loading model weights...")
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(f"{config.pretrained_model_name_or_path}", subfolder="vae")
        vae.requires_grad_(False)


    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                rank=accelerator.process_index,
                                                model_cfg=config.model_cfg,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)


    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    if accelerator.is_main_process:
        print("Loading weight only...")


    run_path = args.test_run_path
    ckpt_dir = f"checkpoint-{args.test_train_step}"
    ckpt_path = os.path.join(run_path, ckpt_dir)

    if args.use_ema:
        ema_path = os.path.join(ckpt_path, "ema_unet.pt")
        weights = torch.load(ema_path, map_location="cpu")
        unet.load_state_dict(weights, strict=False)
    else:
        model_path = os.path.join(ckpt_path,"pytorch_model", "mp_rank_00_model_states.pt" )
        weights = torch.load(model_path, map_location="cpu")
        unet.load_state_dict(weights['module'], strict=False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process and args.use_wandb:
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            init_kwargs={"wandb": {"name": args.run_name}} if args.run_name is not None else {},
        )

    if args.test_use_vggt_camera:
        from vggt.models.vggt import VGGT
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").eval()
        vggt_model.to(accelerator.device, dtype=weight_dtype)
    else:
        vggt_model = None
    
    from src import datasets
    test_dataset_config = EasyDict(OmegaConf.load(args.test_dataset_config))[args.test_dataset]
    test_dataset = datasets.__dict__[test_dataset_config.cls_name](**test_dataset_config.config)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        sampler=None,
        batch_size=1,
        num_workers=0,
        drop_last=True,
    )
    assert len(test_dataloader) % (accelerator.num_processes * 1) == 0, "Please make sure testidation dataset length is divisible by num_processes"

    logger.info("***** Running Test *****")
    logger.info(f"  Num Test Dataset = {len(test_dataset)}")
    
    @torch.no_grad()
    def process_batch_fn(batch, dataset_class, cond_num = None, 
                     use_vggt_camera=False, vggt_model = None, device ='cuda'):
        # ordering
        if dataset_class == "re10k_wds":
            '''re10k_wds: ordered frames'''
            batch = uniform_push_batch(batch, cond_num)
        elif dataset_class == "lvsm_dataset":
            '''lvsm_dataset: cond + tgt ordered (modified from original code)'''
            pass
        else:
            raise NotImplementedError
        
        # use vggt scaled camera
        if use_vggt_camera:
            image = batch["image"].to(accelerator.device)
            vggt_pred = vggt_model(image)
            batch['extrinsic'], batch['intrinsic'] = vggt_pred['extrinsic'], vggt_pred['intrinsic']
            del vggt_pred
        return batch

    process_batch_fn_1 = partial(process_batch_fn, dataset_class =test_dataset_config.cls_name, cond_num = args.test_cond_num, use_vggt_camera = args.test_use_vggt_camera, vggt_model=vggt_model,
                                                    device=accelerator.device)

    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
    res = log_test(accelerator=accelerator, config=config, args=args,
                            pipeline=pipeline, dataloader=test_dataloader, process_batch_fn=process_batch_fn_1,
                            step=args.test_train_step, device=accelerator.device, vae=vae,
                            eval_size = args.test_eval_size)
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()