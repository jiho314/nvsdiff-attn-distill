
from train import *
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
from torchvision import transforms
logger = get_logger(__name__, log_level="INFO")
from filelock import FileLock
import pandas  as pd

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
    show_images = []
    # show_save_dict = collections.defaultdict(int)
    val_iter = 0

    cond_num = args.test_cond_num
    nframe = args.test_nframe
    cfg = args.test_cfg
    num_inference_steps = args.test_num_inference_steps
    log_prefix = f"test/{args.test_dataset}"
    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            batch = process_batch_fn(batch)
            # unbatchify
            # image, intri_, extri_ = batch['image'][0], batch['intrinsic'][0], batch['extrinsic'][0]
            image, intri_, extri_ = batch['image'], batch['intrinsic'], batch['extrinsic']
            b, f, _, h, w = image.shape
            assert f == nframe, f"args.test_nframe({args.test_nframe}) doesn't match with data nframe({f}), image_shape({image.shape}) "
            image_normalized = image * 2.0 - 1.0

            extrinsic, intrinsic = extri_, intri_
            # import pdb ; pdb.set_trace()
            tag, sequence_name, depth = None, None , None
            preds = pipeline.batch_call(images=image_normalized, nframe=nframe, cond_num=cond_num,
                             height=image_normalized.shape[-2], width=image_normalized.shape[-1],
                             intrinsics=intrinsic, extrinsics=extrinsic,
                             num_inference_steps=num_inference_steps, guidance_scale=cfg,
                             output_type="np", config=config, tag=tag,
                             sequence_name=sequence_name, depth=depth, vae=kwargs['vae']).images  # [bf,h,w,c]
            color_warps = None

            # if batch['tag'][0] not in show_save_dict or show_save_dict[batch['tag'][0]] < 10:  # 每个dataset显示10个
                # show_save_dict[batch['tag'][0]] += 1
            if val_iter < viz_len:
                image_viz = einops.rearrange(image, 'b f c h w -> (b f) c h w')
                preds_viz = preds
                if depth is not None:
                    show_image = get_show_images(image_viz, torch.tensor(preds_viz).permute(0,3,1,2), cond_num, batch["depth"])
                else:
                    show_image = get_show_images(image_viz, torch.tensor(preds_viz).permute(0,3,1,2), cond_num)

                if color_warps is not None:
                    h, w = image.shape[-2], image.shape[-1]
                    show_image[h:h * 2, cond_num * w:] = color_warps[0][:, cond_num * w:]
                show_images.append(show_image)
            # slice gt/pred, channel to last, to numpy
            gt_images = einops.rearrange(image_normalized[:, cond_num:], 'b f c h w -> (b f) h w c')
            gt_images= (gt_images.cpu().numpy() + 1 ) / 2
            # gt_images = (image_normalized[:, cond_num:].permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 # -1 1 → 0 1
            preds = einops.rearrange(preds, '(b f) h w c -> b f h w c', f= nframe)[:, cond_num:]
            preds = einops.rearrange(preds, 'b f h w c -> (b f) h w c')
            if eval_size != h:
                gt_images = resize_fn(gt_images)
                preds = resize_fn(preds)
            if compute_fid:
                gt_imgs_fid, preds_fid = torch.tensor(gt_images).permute(0, 3, 1, 2).to(device),  torch.tensor(preds).permute(0, 3, 1, 2).to(device)
                gt_imgs_fid, preds_fid = accelerator.gather(gt_imgs_fid), accelerator.gather(preds_fid)
                if accelerator.is_main_process:
                    gt_imgs_fid = einops.rearrange(gt_imgs_fid,'... c h w -> (...) c h w').to(device)
                    preds_fid = einops.rearrange(preds_fid,'... c h w -> (...) c h w').to(device)
                    fid_calculator.update(gt_imgs_fid, real=True)
                    fid_calculator.update(preds_fid, real=False)

            for i in range(preds.shape[0]):
                psnr_ = peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0)
                psnr_scores.append(psnr_)
                ssim_ = structural_similarity(cv2.cvtColor(gt_images[i], cv2.COLOR_RGB2GRAY),
                                              cv2.cvtColor(preds[i], cv2.COLOR_RGB2GRAY), data_range=1.0)
                ssim_scores.append(ssim_)
                lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device)
                lpips_scores.append(lpips_)
            
            val_iter += 1
    
    # unify all results
    log_value_dict = {}
    psnr_score = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
    ssim_score = torch.tensor(np.mean(ssim_scores), device=device, dtype=torch.float32)
    lpips_score = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)

    psnr_score = accelerator.gather(psnr_score).mean().item()
    ssim_score = accelerator.gather(ssim_score).mean().item()
    lpips_score = accelerator.gather(lpips_score).mean().item()
    log_value_dict.update({f"psnr": psnr_score, f"ssim": ssim_score, f"lpips": lpips_score})
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
        accelerator.log({"test/show_images": wandb.Image(show_images_full)}, step=step)
        if args.test_save_csv_path is not None:
            data = {
                'run_name': args.run_name,
                'cond_num': args.test_cond_num,
                'dataset_full' : args.test_dataset,
                'dataset': args.test_dataset.split("_")[0],
                'view_range': args.test_dataset.split('_')[-1],
                "train_step" : args.test_train_step
            }
            data.update(log_value_dict)
            data = pd.DataFrame([data])
            save_csv_path = args.test_save_csv_path
            lock = FileLock(save_csv_path +".lock")
            with lock:    
                if not os.path.exists(save_csv_path):
                    data.to_csv(save_csv_path, mode="w", header=True, index=False)
                else:
                    data.to_csv(save_csv_path, mode="a", header=False, index=False)     





    del loss_fn_alex
    torch.cuda.empty_cache()

    return lpips_score



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
    
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--run_base_path", type=str, default="check_points") # 'check_points'
    # test data
    parser.add_argument("--test_dataset_config", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--test_nframe", type=int, required=True)
    parser.add_argument("--test_cond_num", type=int, required=True)
    parser.add_argument("--test_use_vggt_camera", action="store_true")
    # test model
    parser.add_argument("--test_train_step", type=int, required=True)
    # test inference setting
    parser.add_argument("--test_cfg", type=float, required=True)
    parser.add_argument("--test_num_inference_steps", type=int, default=50, required=True)

    # test logging/metric
    parser.add_argument("--test_viz_num", type=int, default=15)
    parser.add_argument("--test_compute_fid", action="store_true")
    parser.add_argument("--test_eval_size", type=int, default = 512)

    # test gist
    parser.add_argument("--test_remote", action="store_true")
    parser.add_argument("--test_remote_base_path", type=str , default=None)
    parser.add_argument("--test_remote_remove_ckpt", action='store_true')
    parser.add_argument("--test_save_csv_path",type=str, default= None)

    parser.add_argument("--test_batch_size", type=int, default=1)

    
    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # assert config.dataset_names is not None and len(config.dataset_names) > 0

    return args


def main():
    args = parse_args()

    if args.test_remote:
        if int(os.environ.get("RANK", "0")) == 0 :
            # down path (remote)
            remote_ckpt_path = os.path.join(args.test_remote_base_path, args.run_name, f"checkpoint-{args.test_train_step}", 'pytorch_model/mp_rank_00_model_states.pt')
            remote_config_path = os.path.join(args.test_remote_base_path, args.run_name, "config.yaml")
            # run path (local)
            os.makedirs(os.path.join(args.run_base_path, args.run_name, f"checkpoint-{args.test_train_step}", 'pytorch_model'), exist_ok=True)
            ckpt_path = os.path.join(args.run_base_path, args.run_name, f"checkpoint-{args.test_train_step}", 'pytorch_model/mp_rank_00_model_states.pt')
            config_path = os.path.join(args.run_base_path, args.run_name, "config.yaml")
            if os.path.isfile(ckpt_path) and  os.path.isfile(config_path):
                print('ckpt/config already exists skip downloading')
                ckpt_already_downloaded = True
            else:
                ckpt_already_downloaded = False
                print('downloading : ' , remote_ckpt_path)
                print('downloading : ' , remote_config_path)
                import paramiko
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect("210.125.69.5", username="kaist-cvlab", password="Cvlab@123!", port=51015)
                sftp = ssh.open_sftp()
                sftp.get(remote_config_path, config_path)
                sftp.get(remote_ckpt_path, ckpt_path)
                sftp.close()
                ssh.close()


    if args.config_file is not None:
        config = EasyDict(OmegaConf.load(args.config_file))
    else:
        args.config_file = os.path.join(args.run_base_path, args.run_name, "config.yaml")
        config = EasyDict(OmegaConf.load(args.config_file))

    

    # logging_dir = os.path.join(args.output_dir, "logs")
    # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        # project_config=accelerator_project_config,
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

    set_seed(args.seed)


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


    # if config.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    if accelerator.is_main_process:
        print("Loading weight only...")


    ckpt_dir = os.path.join(args.run_base_path, args.run_name, f"checkpoint-{args.test_train_step}")

    if args.use_ema:
        ema_path = os.path.join(ckpt_dir, "ema_unet.pt")
        weights = torch.load(ema_path, map_location="cpu")
        # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
        unet.load_state_dict(weights, strict=False)  # unet load first
    else:
        model_path = os.path.join(ckpt_dir,"pytorch_model", "mp_rank_00_model_states.pt" )
        weights = torch.load(model_path, map_location="cpu")
        unet.load_state_dict(weights['module'], strict=False)  # we usually need to resume partial weights here

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    config.run_name = args.run_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            init_kwargs={"wandb": {"name": args.run_name}} if args.run_name is not None else {},
            config = config
        )


    
    # device = accelerator.device
    if args.test_use_vggt_camera:
        from vggt.models.vggt import VGGT
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").eval()
        vggt_model.to(accelerator.device, dtype=weight_dtype)
    else:
        vggt_model = None
    
    # Dataset
    from src import datasets
    test_dataset_config = EasyDict(OmegaConf.load(args.test_dataset_config))[args.test_dataset]
    test_dataset = datasets.__dict__[test_dataset_config.cls_name](**test_dataset_config.config)
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        sampler=None,
        batch_size=args.test_batch_size,
        num_workers=0,
        drop_last=True,
        # num_workers=accelerator.num_processes,
    )
    
    assert len(test_dataloader) % (accelerator.num_processes * 1) == 0, "Please make sure testidation dataset length is divisible by num_processes"

    logger.info("***** Running Test *****")
    logger.info(f"  Num Test Dataset = {len(test_dataset)}")
    
    @torch.no_grad()
    def process_batch_fn(batch, dataset_class, cond_num = None, 
                     use_vggt_camera=False, vggt_model = None, device ='cuda'):
        # 1) ordering
        if dataset_class == "re10k_wds":
            '''re10k_wds: ordered frames'''
            batch = uniform_push_batch(batch, cond_num)
            # TODO: extrapolate idx

        elif dataset_class == "lvsm_dataset":
            '''lvsm_dataset: cond + tgt ordered (modified from original code)'''
            pass
        else:
            raise NotImplementedError
        
        # 2) use vggt scaled camera
        if use_vggt_camera:
            image = batch["image"].to(accelerator.device)  #  0 1 tensor [B,F,3,H,W]
            vggt_pred = vggt_model(image) # jiho TODO make image [0,1]
            batch['extrinsic'], batch['intrinsic'] = vggt_pred['extrinsic'], vggt_pred['intrinsic']
            del vggt_pred
        return batch

    process_batch_fn_1 = partial(process_batch_fn, dataset_class =test_dataset_config.cls_name, cond_num = args.test_cond_num, use_vggt_camera = args.test_use_vggt_camera, vggt_model=vggt_model,
                                                    device=accelerator.device)

    # unset_attn_cache(unet)
    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
    res = log_test(accelerator=accelerator, config=config, args=args,
                            pipeline=pipeline, dataloader=test_dataloader, process_batch_fn=process_batch_fn_1,
                            step=args.test_train_step, device=accelerator.device, vae=vae,
                            eval_size = args.test_eval_size)
    # set_attn_cache(unet, distill_student_unet_attn_layers, print_=False)
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    if args.test_remote_remove_ckpt and accelerator.is_main_process:
        os.remove(ckpt_path)
        print(f"{ckpt_path} deleted successfully.")


if __name__ == "__main__":
    main()

