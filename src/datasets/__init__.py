from .dataset_lvsm import Dataset as lvsm_dataset
from .re10k_wds import build_re10k_wds as re10k_wds
from .co3d_wds import build_co3d_wds as co3d_wds
from .multiple_wds import build_multiple_wds as multi_wds
from .multiple_wds_dino import build_multiple_wds_dino as multi_wds_dino
# from .re10k_wds_jinhk import build_re10k_wds as re10k_wds_jinhk
import torch



def shuffle_batch(batch):
    ''' Caution data_name with "key" is not shuffled
    '''
    img = batch["image"]  # [B,F,3,H,W]
    B,F,_,H,W = img.shape
    perm = torch.randperm(F)
    # shuffle idx
    for k in batch.keys():
        data = batch[k]
        if torch.is_tensor(data):
            if data.ndim >= 2:
                batch[k] = batch[k][:, perm]
    return batch


def uniform_push_batch(batch, cond_num=0):
    ''' Caution data_name with "key" is not applied
     1) uniformly sample target idx
     2) push target views to last
    '''
    # return batch
    if cond_num == 1:
        '''select middle index'''
        img = batch["image"]  # [B,F,3,H,W]
        B,F,_,H,W = img.shape
        idx = torch.arange(F)
        ref_idx = torch.tensor([F // 2 - 1], dtype=torch.long)
        tgt_idx = idx[~torch.isin(idx, ref_idx)]
        new_idx = torch.cat([ref_idx, tgt_idx], dim=0)[:F] # in case target_num +2 > F
    else:
        img = batch["image"]  # [B,F,3,H,W]
        B,F,_,H,W = img.shape
        idx = torch.arange(F)
        ref_idx = torch.linspace(0, F-1, cond_num,dtype=torch.long)
        tgt_idx = idx[~torch.isin(idx, ref_idx)]
        new_idx = torch.cat([ref_idx, tgt_idx], dim=0)[:F]  # in case target_num +2 > F
    
    # shuffle idx
    for k in batch.keys():
        data = batch[k]
        if torch.is_tensor(data):
            if data.ndim >= 2:
                batch[k] = batch[k][:, new_idx]
    return batch

# # TODO: remove, not good 
# def uniform_push_batch_reverse(batch, cond_num):
#     if cond_num == 1:
#         '''select middle index'''
#         img = batch["image"]  # [B,F,3,H,W]
#         B,F,_,H,W = img.shape
#         idx = torch.arange(F)
#         ref_idx = torch.tensor([F // 2 - 1], dtype=torch.long)
#         tgt_idx = idx[~torch.isin(idx, ref_idx)]
#         new_idx = torch.cat([ref_idx, tgt_idx], dim=0)[:F] # in case target_num +2 > F
#     else:
#         img = batch["image"]  # [B,F,3,H,W]
#         B,F,_,H,W = img.shape
#         assert F ==4, "only F==4 supported yet"
#         idx = torch.arange(F)
#         tgt_idx = torch.linspace(0, F-1, cond_num,dtype=torch.long)
#         ref_idx = idx[~torch.isin(idx, tgt_idx)]
#         new_idx = torch.cat([ref_idx, tgt_idx], dim=0)[:F]  # in case target_num +2 > F
        
#         # shuffle idx
#         for k in batch.keys():
#             data = batch[k]
#             if torch.is_tensor(data):
#                 if data.ndim >= 2:
#                     batch[k] = batch[k][:, new_idx]
#         return batch

