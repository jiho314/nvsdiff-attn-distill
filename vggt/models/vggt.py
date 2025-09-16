import torch
import torch.nn as nn
import torch.nn.functional as F


from vggt.layers.attention import Attention
from vggt.layers.block import Block

from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

from typing import Literal, Union

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class AttentionCacheable(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_attn_logit = False
        # self.save_attn_prob = False
        self.attn_logit = []
        self.attn_prob = []

        self.save_query = False
        self.save_key = False
        self.query = []
        self.key = []
    
    def forward(self, x, pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        
        if self.save_query:
            self.query.append(q) # B Head N C
        if self.save_key:
            self.key.append(k)

        if self.save_attn_logit \
            or not self.fused_attn:
            q = q * self.scale
            attn_logit = q @ k.transpose(-2, -1)
            attn_prob = attn_logit.softmax(dim=-1)
            attn = self.attn_drop(attn_prob)
            x = attn @ v

        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            raise NotImplementedError("Unsupported attention type")
        
        if self.save_attn_logit:
            self.attn_logit.append(attn_logit)
        # if self.save_attn_prob:
        #     self.attn_prob.append(attn_prob)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockCacheable(Block):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(
            attn_class=AttentionCacheable,
            *args, **kwargs,
        )

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 # -- caching attnmap
                 cache_attn_layer_ids: list[int]=[],
                #  cache_attn_resolution = (32, 32),
                 cache_attn_block_type = "global", # "global", "frame"
                #  cache_attn_interpolate_mode = "bilinear",
                 # -- caching costmap
                 cache_costmap_types: list[str] = [""], # "track_head", "point_head"
                #  cache_costmap_resolution = (32, 32),
                #  cache_costmap_interpolate_mode = "bilinear",
                #  # -- returning pointmap/depthmap/warped pixel...
                #  return_targets: Union[Literal["pointmap", "depthmap", "warped_rgb", "track_costmap"]] = [],
                 # --
                 img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()
        
        self.aggregator = Aggregator(block_fn= BlockCacheable,
                                     img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None


        assert cache_attn_block_type == "global", "Only global block attention caching is supported."

        self.cache_attn_layer_ids = cache_attn_layer_ids
        # self.cache_attn_resolution = cache_attn_resolution
        # self.cache_attn_interpolate_mode = cache_attn_interpolate_mode
        self.cache_attn_block_type = cache_attn_block_type

        self.cache_costmap_types = cache_costmap_types
        # self.cache_costmap_resolution = cache_costmap_resolution
        # self.cache_costmap_interpolate_mode = cache_costmap_interpolate_mode
        self.set_attn_cache(
            attn_layer_ids=cache_attn_layer_ids,
            block_type=cache_attn_block_type
        )

    def set_attn_cache(vggt, attn_layer_ids=[], block_type="global"):
        if block_type == "global":
            tgt_blocks = vggt.aggregator.global_blocks
        elif block_type == "frame":
            tgt_blocks = vggt.aggregator.frame_blocks
            raise NotImplementedError("Frame block attention caching is not used.")
        else:
            raise ValueError(f"Unsupported block type: {block_type}")        

        for i in attn_layer_ids:
            tgt_attn = tgt_blocks[i].attn
            assert isinstance(tgt_attn, AttentionCacheable), f"Expected AttentionCacheable, but got {type(tgt_attn)} at block {i}"
            tgt_attn.save_query, tgt_attn.save_key = True, True

        print(f"Set attention saving for layer : {block_type}_{attn_layer_ids}, costmap {vggt.cache_costmap_types}")
    
    @torch.no_grad()
    def forward(
        self, 
        images: torch.Tensor,
        query_points: torch.Tensor = None
    ):
        '''
         images: [B, S, 3, H, W], range [0,1]
         return: costmap/attnmap, 
            - First image as Query
            - Shape: [B, Head, q, K(=S*k)]  
        '''
        # -- Forward --
        for param in self.parameters():
            dtype = param.dtype
            break
        B, S, _, H, W = images.shape
        assert H == W, "input img should be square"
        if (H,W) != (518, 518):
            images_vggt = F.interpolate(images.reshape(B*S, 3, H,W), size=(518, 518), mode="bilinear").reshape(B, S, 3, 518, 518)  
        images_vggt = images_vggt.to(dtype=dtype)
        aggregated_tokens_list, patch_start_idx, dino_feat = self.aggregator(images_vggt)

        predictions = {}
        # save dino feat
        assert dino_feat.ndim == 3, "dino_feat should be patchified"
        dino_H = dino_W = int(dino_feat.shape[1] ** 0.5)
        dino_feat = dino_feat.reshape(B, S, dino_H, dino_W, dino_feat.shape[-1]) # B S H W C
        predictions['dino_feat'] = dino_feat
        with torch.cuda.amp.autocast(dtype = dtype):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            # if self.depth_head is not None:
            #     depth, depth_conf = self.depth_head(
            #         aggregated_tokens_list, images=images_vggt, patch_start_idx=patch_start_idx
            #     )
            #     depth = F.interpolate(depth.reshape(B*S, *depth.shape[2:] ).permute(0, 3, 1, 2 ), size=(H, W), mode='bilinear')
            #     depth_conf = F.interpolate(depth_conf.reshape(B*S, *depth_conf.shape[2:], 1).permute(0, 3, 1, 2 ), size=(H, W), mode='bilinear')
            #     predictions["depth"] = depth.permute(0, 2, 3, 1).reshape(B, S, H, W, 1)
            #     predictions["depth_conf"] = depth_conf.permute(0, 2, 3, 1).reshape(B, S, H, W)

            # if self.point_head is not None:
            #     pts3d, pts3d_conf = self.point_head(
            #         aggregated_tokens_list, images=images_vggt, patch_start_idx=patch_start_idx
            #     )
            #     # predictions['world_points_518'] = pts3d
            #     # predictions['images_518'] = images_vggt
            #     pts3d = F.interpolate(pts3d.reshape(B*S, *pts3d.shape[2:]).permute(0,3,1,2), size=(H, W), mode='bilinear')
            #     pts3d_conf = F.interpolate(pts3d_conf.reshape(B*S, *pts3d_conf.shape[2:], 1).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
            #     predictions["world_points"] = pts3d.permute(0, 2, 3, 1).reshape(B, S, H, W, 3)
            #     predictions["world_points_conf"] = pts3d_conf.permute(0, 2, 3, 1).reshape(B, S, H, W)

        # if self.track_head is not None and query_points is not None:
        #     track_list, vis, conf = self.track_head(
        #         aggregated_tokens_list, images=images_vggt, patch_start_idx=patch_start_idx, query_points=query_points
        #     )
        #     predictions["track"] = track_list[-1]  # track of the last iteration
        #     predictions["vis"] = vis
        #     predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        # -- Process --
        # 1) camera params
        predictions['fovy'], predictions['fovx'] = predictions['pose_enc'][..., 7] , predictions['pose_enc'][..., 8]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:]) # original img resolution
        predictions['extrinsic'], predictions['intrinsic'] = extrinsics, intrinsics
        
        # # 2) pointmap w/ depth head & Camera
        # depth head가 비활성화되어 있으므로 depth 처리도 주석 처리
        # depth = predictions["depth"]
        # depth, extrinsics, intrinsics = depth.reshape(B*S, *depth.shape[2:]).to(torch.float32), extrinsics.reshape(B*S, *extrinsics.shape[2:]).to(torch.float32), intrinsics.reshape(B*S, *intrinsics.shape[2:]).to(torch.float32)
        # world_points_from_depth = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics) # S H W 3
        # world_points_from_depth = world_points_from_depth.reshape(B, S, *world_points_from_depth.shape[1:])  # B S H W 3
        # predictions['world_points_from_depth'] = torch.tensor(world_points_from_depth).to(dtype=dtype, device=images.device)
        

        # -- Distill --
        # attnmap
        attn_cache = {}
        if self.cache_attn_block_type == "global":
            blocks = self.aggregator.global_blocks 
            for l_id in self.cache_attn_layer_ids:
                query_cache, key_cache = blocks[l_id].attn.query, blocks[l_id].attn.key
                assert len(query_cache) == 1 and len(key_cache) == 1, "Only single query/key should be cached per layer"
                query, key = query_cache.pop(), key_cache.pop()
                query_cache.clear()
                key_cache.clear()
                # Exclude extra tokens
                _, head, tok_N, tok_C = query.shape
                query, key = query.reshape(B, head, S, tok_N//S, tok_C), key.reshape(B, head, S, tok_N//S, tok_C)
                query, key = query[:,:,:, patch_start_idx:, :], key[:,:,:, patch_start_idx:,:] # B Head SHW C
                query, key = query.reshape(B, head, -1 ,tok_C), key.reshape(B, head, -1, tok_C) # B Head SHW C
                attn_cache[str(l_id)] = {
                    "query": query, # B Head N(SHW) C
                    "key": key # B Head N(SHW) C
                }

        elif self.cache_attn_block_type == "frame":
            raise NotImplementedError("Frame block attention caching is not used.") 

        # costmap
        if "track_head" in self.cache_costmap_types:
            track_feat = self.track_head.feature_extractor(aggregated_tokens_list, images_vggt, patch_start_idx) # B S C H W
            _, _, track_C, track_H, track_W = track_feat.shape
            track_feat = track_feat.permute(0,1,3,4,2).reshape(B, 1, S*track_H*track_W, track_C)  # B 1(attn head dim) S*H*W C
            attn_cache["track_head"] = {
                'query': track_feat, # B Head(1) N(SHW) C
                "key" : track_feat # B Head(1) N(SHW) C
            }
            # fmaps = fmaps.reshape(B, S, C_feat, *self.cache_costmap_resolution)
            # fmaps = fmaps.permute(0, 2, 1, 3, 4) # B C_feat S H W
            # # costmap
            # fmap_tgt = fmaps[:, :, 0, ...] 
            # fmap_tgt, fmaps = fmap_tgt.reshape(B, C_feat, -1), fmaps.reshape(B, C_feat, -1) # B C_feat S*H*W
            # costmap = torch.matmul(fmap_tgt.transpose(1, 2), fmaps) # B H*W S*H*W 
            # costmap =costmap.unsqueeze(1) # B 1 H*W S*H*W
            # costmap_dict["track_head"] = costmap
        elif "point_head" in self.cache_costmap_types: 
            # TODO:
            raise NotImplementedError("Point head costmap caching is not implemented yet.")
        elif "point_map" in self.cache_costmap_types:
            pass # already saved in batch

        # for jinhyeok
        elif "refine_track_head" in self.cache_costmap_types:
            def make_query_points(batch_size, device="cpu"):
                coords = torch.arange(0, 518, 16.5, device=device)  # (32,)
                yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # each (32,32)

                grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (1024, 2)

                grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 1024, 2)

                return grid
            query_points = make_query_points(B, device=images.device).to(dtype=dtype) # B HW 2
            with torch.cuda.amp.autocast(dtype = torch.float32):
                track_feat = self.track_head.feature_extractor(aggregated_tokens_list, images_vggt, patch_start_idx) # B S C H W
                coord_preds, vis_e, track_feats, query_track_feat, conf_e = self.track_head.tracker(query_points=query_points, fmaps=track_feat, iters=4, return_feat=True)
            
            attn_cache["refine_track_head"] = {
                'query': track_feats.reshape(B, -1, 128).unsqueeze(1), # B 1 SHW 128
                "key" : track_feats.reshape(B, -1, 128).unsqueeze(1) # B S HW 128
            }

        predictions['attn_cache'] = attn_cache

        return predictions

