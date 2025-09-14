from train import * 
from train import log_validation
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




dataset = Dataset(**config["eval_dataset"])
dataloader = DataLoader( # TODO check if this works
    dataset,
    batch_size=1,
    # persistent_workers=True,
    num_workers=config["dataloader_workers"],
    shuffle=False,
    drop_last=True,
)

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


pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
res = log_validation(accelerator=accelerator, config=config, args=args,
                        pipeline=pipeline, val_dataloader=val_dataloader,
                        step=global_step, device=device, vae=vae)