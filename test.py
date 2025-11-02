
from train import *
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial
from torchvision import transforms
logger = get_logger(__name__, log_level="INFO")
from filelock import FileLock
import pandas  as pd
from warp import render_points_pytorch3d
from train import log_test
# from torchmetrics.image.psnr import PeakSignalNoiseRatio
# from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.metrics import compute_psnr, compute_ssim, compute_lpips



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
    # parser.add_argument("--test_nframe", type=int, required=True) # nframe from config
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
    parser.add_argument("--test_eval_w_mask", action="store_true")

    # test gist
    parser.add_argument("--test_remote", action="store_true")
    parser.add_argument("--test_remote_base_path", type=str , default=None)
    parser.add_argument("--test_remote_remove_ckpt", action='store_true')
    parser.add_argument("--test_save_csv_path",type=str, default= None)

    parser.add_argument("--test_batch_size", type=int, default=1)


    # test idx setting
    parser.add_argument("--test_custom_order_batch", type=bool, action='store_true')
    parser.add_argument("--test_custom_ref_idx", type=int, nargs='+', default=[])
    
    

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
                print(f'ckpt-{args.test_train_step}/config already exists skip downloading')
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
    # dataset_kwargs = OmegaConf.to_container(test_dataset_config.config, resolve=True)
    # dataset_kwargs["use_vggt_camera"] = args.test_use_vggt_camera
    # test_dataset_config.config = dataset_kwargs
    dataset_kwargs = test_dataset_config.config
    # test_dataset = datasets.__dict__[test_dataset_config.cls_name](num_viewpoints=args.test_nframe, **dataset_kwargs)
    test_dataset = datasets.__dict__[test_dataset_config.cls_name](**dataset_kwargs)
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
    
    from src.datasets import custom_order_batch
    def process_batch_fn(batch, dataset_class, cond_num = None, 
                     use_vggt_camera=False, vggt_model = None, device ='cuda', custom_order = False, custom_ref_idx = []):
        # 11/02 jiho: ordering done in data code (just for inference!)
        # # 1) ordering
        # if not custom_order:
        #     if dataset_class == "re10k_wds":
        #         '''re10k_wds: ordered frames'''
        #         batch = uniform_push_batch(batch, cond_num)
        #         # TODO: extrapolate idx
        #     elif dataset_class == "co3d_wds":
        #         '''co3d_wds: ordered frames'''
        #         batch = uniform_push_batch(batch, cond_num)
        #     elif dataset_class == "lvsm_dataset":
        #         '''lvsm_dataset: cond + tgt ordered (modified from original code)'''
        #         pass
        #     else:
        #         raise NotImplementedError
        # else:
        #     assert len(custom_ref_idx) == cond_num, f"custom_ref_idx length {len(custom_ref_idx)} must be equal to cond_num {cond_num}"
        #     batch = custom_order_batch(batch, ref_idx=list(custom_ref_idx))

        
        # 2) use vggt scaled camera
        if use_vggt_camera:
            image = batch["image"].to(accelerator.device)  #  0 1 tensor [B,F,3,H,W]
            vggt_pred = vggt_model(image) # jiho TODO make image [0,1]
            batch['extrinsic'], batch['intrinsic'] = vggt_pred['extrinsic'], vggt_pred['intrinsic']
            del vggt_pred

        return batch

    process_batch_fn_test = partial(process_batch_fn, dataset_class =test_dataset_config.cls_name, cond_num = args.test_cond_num, 
                                                    custom_order = args.test_custom_order_batch, custom_ref_idx = args.test_custom_ref_idx,
                                                    use_vggt_camera = args.test_use_vggt_camera, vggt_model=vggt_model,
                                                    device=accelerator.device)

    # unset_attn_cache(unet)
    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
    cond_num = args.test_cond_num
    # nframe = args.test_nframe
    nframe = dataset_kwargs.get('num_viewpoints')
    cfg = args.test_cfg
    num_inference_steps = args.test_num_inference_steps
    res = log_test(accelerator=accelerator, config=config, args=args,
                            pipeline=pipeline, dataloader=test_dataloader, process_batch_fn=process_batch_fn_test,
                            step=args.test_train_step, device=accelerator.device, vae=vae,
                            cond_num = cond_num, nframe = nframe, cfg = cfg, num_inference_steps = num_inference_steps,
                            compute_fid = args.test_compute_fid,
                            viz_len = args.test_viz_num,
                            eval_size = args.test_eval_size,
                            eval_w_mask = args.test_eval_w_mask,
                            save_csv_path= args.test_save_csv_path,
                            test_dataset_name = args.test_dataset,
                            )
    # set_attn_cache(unet, distill_student_unet_attn_layers, print_=False)
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    if args.test_remote_remove_ckpt and args.test_remote and accelerator.is_main_process:
        if not ckpt_already_downloaded:
            os.remove(ckpt_path)
            print(f"{ckpt_path} deleted successfully.")


if __name__ == "__main__":
    main()
