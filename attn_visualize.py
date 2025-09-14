from train import *


from src.distill_utils.attn_processor_cache import set_attn_cache, unset_attn_cache, pop_cached_attn, clear_attn_cache



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
    # args, _, cfg = parse_args()

    set_seed(0)
    device= f"cuda:{rank}"
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
    # 3) Model Inference
    def unet_inference(batch):
        # batch: always order by seq idx
        # uniform_push_batch: set condition (uniform sampling, no randomness) to front, tgt to last
        batch = uniform_push_batch(batch, cond_num)
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

    # minkyung TODO: 여러 idx에 대해서 viz
    for idx, batch in enumerate(val_dataloader):
        # batch: always order by seq idx
        # uniform_push_batch: set condition (uniform sampling, no randomness) to front, tgt to last
        _ = unet_inference(batch)
        unet_attn_cache = pop_cached_attn(unet)
        ''' unet_attn_cache(dict): {layer_id(str): attnmap tensor(B, head, Q(VHW), K(VHW)}
        '''
        for unet_layer in caching_unet_attn_layers:
            unet_attn_logit = unet_attn_cache[str(unet_layer)] # B Head VHW VHW
            # minkyung TODO: visualize attention maps (with image)
            import pdb ; pdb.set_trace()
        

if __name__ == "__main__":
    
    # minkyung TODO
    # unet: (down_blocks(6), mid_block(1), up_blocks(9))
    #  size: [64,64,32,32,16,16] [8] [16,16,16,32,32,32,64,64,64]
    caching_unet_attn_layers = [6, 12] # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # noise
    noise_timestep = 100
    # data
    nframe = 3 # View length
    cond_num = 2 # Condition among nframe
    inference_view_range = 6 # change if you want 
    # checkpoint
    resume_checkpoint = 'check_points/lr1_cosine_noema/checkpoint-30000'
    config_file_path = 'configs/viz.yaml'
    config = EasyDict(OmegaConf.load(config_file_path))

    rank = 0
    main(nframe, cond_num, inference_view_range, caching_unet_attn_layers, noise_timestep=noise_timestep, resume_checkpoint=resume_checkpoint, config=config, rank = 0)