import torch
import math
import einops
from typing import Dict, List, Tuple, Optional, Any
from my_diffusers.callbacks import PipelineCallback
from src.distill_utils.attn_processor_cache import pop_cached_attn, clear_attn_cache
from src.distill_utils.query_key_cost_metric import COST_METRIC_FN
from torch import nn


class Softmax(nn.Module):
    def __init__(self, 
                 softmax_temp=1.0, num_view_for_per_view=None, **kwargs):
        super(Softmax, self).__init__()
        self.num_view = num_view_for_per_view
        if num_view_for_per_view is not None:
            self.per_view = True
        else:
            self.per_view = False
        self.softmax_temp = softmax_temp
    
    def forward(self, x, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // self.num_view
            x = x.reshape(*x.shape[:-1], self.num_view, HW)
        x = x / self.softmax_temp
        return x.softmax(dim=-1)
    
class Softmax_HeadMean(Softmax):
    def forward(self, x, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // self.num_view
            x = x.reshape(*x.shape[:-1], self.num_view, HW)
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        return x.mean(dim=1, keepdim=True)


import os
import math
import numpy as np
import torch
import torch.nn.functional as FNN
import time
import uuid
from PIL import Image as _PILImage, ImageDraw as _PILDraw



class AttentionVisualizationCallback(PipelineCallback):
    """
    VGGT와 UNet attention을 비교하여 시각화 로스를 계산하는 callback 클래스
    매 denoising step마다 호출되어 attention logit을 수집하고 distillation loss를 계산합니다.
    """
    
    tensor_inputs = ["latents"]
    
    def __init__(
        self,
        vggt_model: torch.nn.Module,
        visualize_config: Dict[str, Any],
        batch: Dict[str, Any],
        cond_num: int,
        device: torch.device,
        do_attn_visualize: bool = True,
        accelerator=None
    ):
        """
        Args:
            vggt_model: VGGT 모델
            visualize_config: 시각화 설정 (pairs, query, key, loss_fn 등)
            batch: 배치 데이터 (image, vggt_attn_cache 등)
            cond_num: condition frame 수
            device: 디바이스
            do_attn_visualize: attention 시각화 활성화 여부
        """
        super().__init__()
        self.vggt_model = vggt_model
        self.visualize_config = visualize_config
        
        # Loss function 설정
        self._setup_loss_function()
        self.batch = batch
        self.cond_num = cond_num
        self.device = device
        self.do_attn_visualize = do_attn_visualize
        self.accelerator = accelerator  # accelerator 객체 저장
        self.visualize_loss_dict = {}
        self.step_losses = []
        self.step_layer_losses = {}  # step별 layer별 loss 저장
        self.layer_losses = {}  # layer별 누적 loss 저장
        
        # VGGT attention cache 미리 계산
        if self.do_attn_visualize and self.vggt_model is not None:
            self._prepare_vggt_cache()
    
    def _setup_loss_function(self):
        """Loss function을 문자열에서 실제 함수로 변환"""
        # define canonical loss functions and store mapping for per-pair overrides
        # Helper: normalize inputs and perform head/view collapsing
        def _prepare_for_loss(pred: torch.Tensor, gt: torch.Tensor):
            """
            Accepts pred/gt in either per-view form (B,Head,Q,V,HW) or global form (B,Head,Q,K)
            and returns a tuple (mode, P, G, meta) where
              - mode is 'per_view' or 'global'
              - P, G are tensors already converted to float and on same device
              - meta contains dict with B, Head, Q, V (if per_view) and K
            """
            if pred is None or gt is None:
                raise ValueError("pred and gt must be tensors")
            # ensure same dtype/device
            gt = gt.to(dtype=pred.dtype, device=pred.device)

            if pred.dim() == 5:
                # (B,Head,Q,V,HW)
                B, Head, Q, V, HW = pred.shape
                mode = 'per_view'
                meta = dict(B=B, Head=Head, Q=Q, V=V, HW=HW)
                return mode, pred, gt, meta
            elif pred.dim() == 4:
                # (B,Head,Q,K)
                B, Head, Q, K = pred.shape
                mode = 'global'
                meta = dict(B=B, Head=Head, Q=Q, K=K)
                return mode, pred, gt, meta
            else:
                raise ValueError(f"Unsupported pred shape for loss: {pred.shape}")

        def _collapse_heads_as_batch(x: torch.Tensor):
            # (B,Head,...) -> (B*Head,...)
            B, H = x.shape[0], x.shape[1]
            return x.view(B * H, *x.shape[2:])

        # Core loss implementations operating on collapsed shapes
        def _cross_entropy(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            # pred_coll, gt_coll: (N, Q, K) probabilities over last dim
            eps = 1e-8
            qloss = - (gt_coll * (pred_coll + eps).log()).sum(dim=-1)  # (N, Q)
            per_sample = qloss.mean(dim=1)  # (N,)
            return per_sample.mean()

        def _kl_divergence(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            qloss = (gt_coll * (gt_coll.log() - pred_coll.log())).sum(dim=-1)
            per_sample = qloss.mean(dim=1)
            return per_sample.mean()

        def _l1(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            # mean abs diff over Q and K
            l1 = torch.abs(pred_coll - gt_coll)
            per_sample = l1.mean(dim=(1, 2))
            return per_sample.mean()

        # Using decorator approach to avoid duplicated wrapper logic. Decorator converts a
        # per-sample (N,Q,K)->(N,) function into a full loss callable accepting pred/gt
        # in either per-view (B,Head,Q,V,HW) or global (B,Head,Q,K) formats and applying
        # optional pre/post head-mean semantics.
        def loss_variant(variant: Optional[str] = None):
            def decorator(per_sample_fn):
                def loss_callable(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
                    mode, P, G, meta = _prepare_for_loss(pred, gt)

                    # pre-headmean
                    if variant == 'pre':
                        P = P.mean(dim=1, keepdim=True)
                        G = G.mean(dim=1, keepdim=True)

                    if mode == 'per_view':
                        B, H, Q, V, HW = P.shape
                        P_coll = P.permute(0, 1, 3, 2, 4).contiguous().view(B * H * V, Q, HW)
                        G_coll = G.permute(0, 1, 3, 2, 4).contiguous().view(B * H * V, Q, HW)
                        per_sample = per_sample_fn(P_coll, G_coll)  # (N,) where N == B*H*V

                        # Return per-head values (1D tensor) when no head-mean requested
                        if variant is None:
                            per_sample = per_sample.view(B, H, V)
                            per_head = per_sample.mean(dim=2)  # (B, H)
                            return per_head.view(-1)

                        # For pre/post head-mean variants return a single-value tensor (length 1)
                        scalar = float(per_sample.mean().item())
                        return torch.tensor([scalar], device=P.device, dtype=P.dtype)

                    else:  # global
                        B, H, Q, K = P.shape
                        P_coll = _collapse_heads_as_batch(P)
                        G_coll = _collapse_heads_as_batch(G)
                        per_sample = per_sample_fn(P_coll, G_coll)  # (N,) where N == B*H

                        if variant is None:
                            per_sample = per_sample.view(B, H)
                            return per_sample.view(-1)

                        scalar = float(per_sample.mean().item())
                        return torch.tensor([scalar], device=P.device, dtype=P.dtype)

                return loss_callable

            return decorator

        # Per-sample loss implementations (return per-sample vector of shape (N,))
        def _cross_entropy_per_sample(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            eps = 1e-8
            # pred_coll, gt_coll: (N, Q, K)
            qloss = - (gt_coll * (pred_coll + eps).log()).sum(dim=-1)  # (N, Q)
            per_sample = qloss.mean(dim=1)  # (N,)
            return per_sample

        def _kl_per_sample(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            qloss = (gt_coll * (gt_coll.log() - pred_coll.log())).sum(dim=-1)
            per_sample = qloss.mean(dim=1)
            return per_sample

        def _l1_per_sample(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            l1 = torch.abs(pred_coll - gt_coll)
            per_sample = l1.mean(dim=(1, 2))
            return per_sample

        # Base per-sample loss functions mapping.
        #
        # Helper for adding custom loss functions:
        # - Implement a per-sample function with signature:
        #       def fn(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor
        #   where the inputs are "collapsed" tensors as described below and the
        #   output is a 1D tensor of per-sample losses (length N):
        #     * Global mode: pred_coll / gt_coll shape = (N, Q, K)
        #         - N == B * Head (batch times head)
        #     * Per-view mode: the callback will collapse to
        #         pred_coll / gt_coll shape = (N, Q, K') where
        #         - N == B * Head * V (batch * head * num_views)
        #         - K' == HW (flattened per-view spatial keys)
        #   The per-sample fn must return a 1D tensor of length N (per-sample losses).
        #
        # To register an external per-sample fn, pass a dict into
        # `visualize_config['loss_fn_map']` mapping base name -> callable.
        base_per_sample = {
            'cross_entropy': _cross_entropy_per_sample,
            'kl_divergence': _kl_per_sample,
            'l1': _l1_per_sample,
        }

        # Allow validate.py to pass an external mapping via visualize_config['loss_fn_map']
        external_map = self.visualize_config.get('loss_fn_map', None)

        # decorator factory
        def loss_variant(variant: Optional[str] = None):
            def decorator(per_sample_fn):
                def loss_callable(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
                    mode, P, G, meta = _prepare_for_loss(pred, gt)

                    # pre-headmean
                    if variant == 'pre':
                        P = P.mean(dim=1, keepdim=True)
                        G = G.mean(dim=1, keepdim=True)

                    if mode == 'per_view':
                        B, H, Q, V, HW = P.shape
                        P_coll = P.permute(0, 1, 3, 2, 4).contiguous().view(B * H * V, Q, HW)
                        G_coll = G.permute(0, 1, 3, 2, 4).contiguous().view(B * H * V, Q, HW)
                        per_sample = per_sample_fn(P_coll, G_coll)

                        if variant == 'post':
                            per_sample = per_sample.view(B, H, V).mean(dim=1).mean()
                            return per_sample
                        return per_sample.mean()

                    else:  # global
                        B, H, Q, K = P.shape
                        P_coll = _collapse_heads_as_batch(P)
                        G_coll = _collapse_heads_as_batch(G)
                        per_sample = per_sample_fn(P_coll, G_coll)

                        if variant == 'post':
                            per_sample = per_sample.view(B, H).mean(dim=1).mean()
                            return per_sample
                        return per_sample.mean()

                return loss_callable

            return decorator

        # Store base per-sample functions (may be overridden by external_map).
        # expose base per-sample map (possibly overridden by external_map)
        base_map = {}
        for base_name, per_sample_fn in base_per_sample.items():
            if external_map is not None and base_name in external_map:
                per_sample_fn = external_map[base_name]
            base_map[base_name] = per_sample_fn

        # also expose ready-to-call wrapped variants for convenience (so callers can request 'cross_entropy_pre' etc.)
        wrapped_map = {}
        for base_name, per_sample_fn in base_map.items():
            wrapped_map[base_name] = loss_variant(None)(per_sample_fn)
            wrapped_map[f"{base_name}_pre"] = loss_variant('pre')(per_sample_fn)
            wrapped_map[f"{base_name}_post"] = loss_variant('post')(per_sample_fn)

        # store both
        self._ATTN_LOSS_BASE = base_map
        self.ATTN_LOSS_FN = wrapped_map

        # # set default loss_fn
        # if callable(loss_fn):
        #     self.loss_fn = loss_fn
        # elif isinstance(loss_fn, str):
        #     key = loss_fn.lower()
        #     if key not in ATTN_LOSS_FN:
        #         raise ValueError(f"Unsupported loss_fn string: {loss_fn}")
        #     self.loss_fn = ATTN_LOSS_FN[key]
        #     print(f"Using loss_fn: {key}")
        # else:
        #     raise ValueError(f"Unsupported loss_fn type: {type(loss_fn)}")
    
    def _prepare_vggt_cache(self):
        """VGGT attention cache를 미리 계산"""
        with torch.no_grad():
            image = self.batch['image'].to(self.device)  # [B,F,3,H,W]
            vggt_pred = self.vggt_model(image)
            self.batch['vggt_attn_cache'] = vggt_pred['attn_cache']

    def _logits_to_probs(self, logits: torch.Tensor, num_key_views: int, mode: Optional[str]) -> torch.Tensor:
        """Convert logits to probabilities according to `mode`.

        This is a class-level helper (accessible via self) — pair-level callers must
        provide an explicit `mode` (no global fallbacks).
        """
        if logits is None:
            return logits
        if mode is None:
            raise ValueError("_logits_to_probs requires explicit 'mode' (pair-level configuration only)")

        # preserve dtype/device
        device = logits.device
        dtype = logits.dtype

        # quick check: if values along K already sum to ~1, assume probs and return
        try:
            sums = logits.sum(dim=-1)
            if torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
                return logits
        except Exception:
            pass

        if mode == 'global':
            return logits.softmax(dim=-1)

        # expect K divisible by num_key_views
        K = int(logits.shape[-1])
        num_k = int(num_key_views)
        if num_k <= 0 or K % num_k != 0:
            raise ValueError(f"softmax per-view requires num_key_views that divides K: num_key_views={num_k}, K={K}")
        per = K // num_k

        if mode == 'per_view':
            # reshape last dim to (..., num_k, per) and softmax over per
            K = int(logits.shape[-1])
            if num_k <= 0 or K % num_k != 0:
                # Provide a clearer error message to help debugging malformed shapes
                raise ValueError(f"per_view softmax requires num_key_views that divides K: num_key_views={num_k}, K={K}")
            per = K // num_k
            leading = logits.shape[:-1]
            # defensive reshape: ensure the computed shape matches element count
            try:
                view_shaped = logits.view(*leading, num_k, per)
            except Exception as e:
                raise RuntimeError(f"Failed to reshape logits for per_view softmax: attempted shape {*leading, num_k, per} from logits.shape={logits.shape}: {e}")
            probs = view_shaped.softmax(dim=-1)
            return probs.view(*leading, K)

        raise ValueError(f"Unsupported softmax mode: {mode}")
    


    def _resolve_heads(self, pair: Dict[str, Any]):
        """
        Return a (unet_head, vggt_head) tuple by instantiating from per-pair
        overrides or falling back to the global `visualize_config` settings.
        """
        # Always use the unified head class `softmax` for both UNet and VGGT.
        # Allow per-pair or global kwargs to configure temperatures and per_view.
        logit_head_cls = Softmax
        viz_head_cls = Softmax_HeadMean
        
        if logit_head_cls is None:
            raise RuntimeError("Unified logit head class 'softmax' not found")
        if viz_head_cls is None:
            raise RuntimeError("Unified viz head class 'softmax_headmean' not found")

        # kwargs resolution: prefer per-pair kwargs, then global visualize_config
        unet_kwargs = pair.get('unet_logit_head_kwargs', None) if isinstance(pair, dict) else None
        vggt_kwargs = pair.get('vggt_logit_head_kwargs', None) if isinstance(pair, dict) else None

        mode_loss = str(pair.get('loss_softmax_mode', 'global')).lower()
        mode_viz = str(pair.get('viz_softmax_mode', 'global')).lower()
        # dict + dict 연산 오류 수정: dict.update() 또는 {**dict1, **dict2} 사용
        unet_loss_kwargs = {**(unet_kwargs or {}), 'per_view': mode_loss == 'per_view'}
        vggt_loss_kwargs = {**(vggt_kwargs or {}), 'per_view': mode_loss == 'per_view'}
        unet_viz_kwargs = {**(unet_kwargs or {}), 'per_view': mode_viz == 'per_view'}
        vggt_viz_kwargs = {**(vggt_kwargs or {}), 'per_view': mode_viz == 'per_view'}
        
        unet_loss_head = logit_head_cls(**unet_loss_kwargs)
        vggt_loss_head = logit_head_cls(**vggt_loss_kwargs)
        unet_viz_head = viz_head_cls(**unet_viz_kwargs)
        vggt_viz_head = viz_head_cls(**vggt_viz_kwargs)
        return unet_loss_head, vggt_loss_head, unet_viz_head, vggt_viz_head


    def _resize_token(self, tok: torch.Tensor, target_size: int, F: int) -> torch.Tensor:
        """토큰을 target_size로 리사이즈"""
        # Support inputs with and without explicit Head dimension.
        # Acceptable input shapes:
        # - (B, Head, FHW, C)
        # - (B, FHW, C)  -> treated as Head=1
        if tok.dim() == 4:
            B, Head, FHW, C = tok.shape
            has_head = True
        elif tok.dim() == 3:
            B, FHW, C = tok.shape
            Head = 1
            has_head = False
            # add head dim at dim=1 for consistent processing
            tok = tok.unsqueeze(1)  # (B,1,FHW,C)
        else:
            raise ValueError(f"Unexpected token tensor shape for _resize_token: {tok.shape}")

        HW = FHW // F
        H = W = int(math.sqrt(HW))
        tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', 
                              B=B, Head=Head, F=F, H=H, W=W, C=C)
        tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
        tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C', 
                              B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)

        # If original input had no head dim, keep Head dim=1 (caller should handle)
        return tok
    
    def _slice_attention_map(self, attnmap: torch.Tensor, query_idx: List[int], 
                           key_idx: List[int], F: int) -> torch.Tensor:
        """attention map을 query/key index에 따라 슬라이싱"""
        B, Head, Q, K = attnmap.shape
        HW = Q // F
        attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', 
                                  B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
        attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
        attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', 
                                  B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
        return attnmap
    
    def _extract_view_indices(self, F: int, mode: str) -> Tuple[List[int], List[int]]:
        """Loss 계산용 query와 key의 view index 추출"""
        # Loss Query index 추출
        if self.visualize_config[f'{mode}_query'] == "target":
            query_idx = list(range(self.cond_num, F))
        elif self.visualize_config[f'{mode}_query'] == "all":
            query_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['loss_query'] {self.visualize_config['loss_query']} not implemented")
        
        # Loss Key index 추출
        if self.visualize_config[f'{mode}_key'] == "reference":
            key_idx = list(range(0, self.cond_num))
        elif self.visualize_config[f'{mode}_key'] == "all":
            key_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['loss_key'] {self.visualize_config['loss_key']} not implemented")
        
        return query_idx, key_idx

    
    ##### VISUALIZATION UTILS #####
    def _attn_gray_to_rgb(self, gray: np.ndarray) -> np.ndarray:
        """Map normalized [0,1] gray heatmap to RGB uint8 (simple HSV-like colormap)."""
        h = (1.0 - gray) * (2.0 / 3.0)
        s = np.ones_like(gray)
        v = np.ones_like(gray)
        hp = (h * 6.0)
        i = np.floor(hp).astype(np.int32) % 6
        f = hp - np.floor(hp)
        p = np.zeros_like(gray)
        q = 1.0 - f
        t = f
        r = np.empty_like(gray)
        g = np.empty_like(gray)
        b = np.empty_like(gray)
        mask = (i == 0)
        r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
        mask = (i == 1)
        r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
        mask = (i == 2)
        r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
        mask = (i == 3)
        r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
        mask = (i == 4)
        r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
        mask = (i == 5)
        r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb

    def _draw_query_point(self, img: _PILImage.Image, qx: int, qy: int, q_side: int) -> _PILImage.Image:
        d = _PILDraw.Draw(img)
        w, h = img.size
        sx = w / float(q_side)
        sy = h / float(q_side)
        cx = int((qx + 0.5) * sx)
        cy = int((qy + 0.5) * sy)
        r_in = max(2, min(w, h) // 120)
        # outline + fill
        d.ellipse((cx - r_in*2, cy - r_in*2, cx + r_in*2, cy + r_in*2), fill=(255, 255, 255))
        d.ellipse((cx - r_in, cy - r_in, cx + r_in, cy + r_in), fill=(0, 128, 255))
        return img

    def _draw_max_attention_point(self, img: _PILImage.Image, attention_tile: np.ndarray, tile_side: int) -> _PILImage.Image:
        """각 view별로 가장 확률이 높은 부분에 파란 점을 그림"""
        d = _PILDraw.Draw(img)
        w, h = img.size
        # compute argmax index
        flat = attention_tile.flatten()
        max_idx = int(np.argmax(flat))
        max_y, max_x = np.unravel_index(max_idx, attention_tile.shape)

        # compute softargmax (expectation) on the tile for consistency with debug
        xs = np.arange(tile_side)
        # note: attention_tile shape is (tile_side, tile_side)
        total = attention_tile.sum()
        if total > 0:
            pred_sx = (attention_tile * xs.reshape(1, tile_side)).sum(axis=(0, 1)) / (total + 1e-12)
            pred_sy = (attention_tile * xs.reshape(tile_side, 1)).sum(axis=(0, 1)) / (total + 1e-12)
        else:
            pred_sx = float(max_x)
            pred_sy = float(max_y)

        # tile 좌표를 이미지 좌표로 변환
        sx = w / float(tile_side)
        sy = h / float(tile_side)
        cx_hard = int((max_x + 0.5) * sx)
        cy_hard = int((max_y + 0.5) * sy)
        cx_soft = int((pred_sx + 0.5) * sx)
        cy_soft = int((pred_sy + 0.5) * sy)

        # dot sizes
        r_in = max(3, min(w, h) // 80)

        # draw argmax (red) with white border
        r_small = max(2, r_in // 2)
        d.ellipse((cx_hard - (r_small+2), cy_hard - (r_small+2), cx_hard + (r_small+2), cy_hard + (r_small+2)), fill=(255,255,255))
        d.ellipse((cx_hard - r_small, cy_hard - r_small, cx_hard + r_small, cy_hard + r_small), fill=(255,0,0))

        # draw softargmax (yellow) with black border for contrast
        d.ellipse((cx_soft - (r_in+2), cy_soft - (r_in+2), cx_soft + (r_in+2), cy_soft + (r_in+2)), fill=(0,0,0))
        d.ellipse((cx_soft - r_in, cy_soft - r_in, cx_soft + r_in, cy_soft + r_in), fill=(255,255,0))

        return img

    def _coords_argmax(self, tile: np.ndarray) -> Tuple[int, int]:
        """Return (x,y) integer coords of argmax within tile."""
        idx = int(np.argmax(tile.flatten()))
        y, x = np.unravel_index(idx, tile.shape)
        return int(x), int(y)

    def _coords_softargmax(self, tile: np.ndarray) -> Tuple[float, float]:
        """Return (x,y) soft-argmax (expectation) within tile as floats."""
        side = tile.shape[0]
        xs = np.arange(side)
        total = float(tile.sum())
        if total <= 0:
            # fallback to argmax if tile is all zeros
            ax, ay = self._coords_argmax(tile)
            return float(ax), float(ay)
        # pred_sx sums over columns for x coordinate, pred_sy sums over rows for y
        pred_sx = (tile * xs.reshape(1, side)).sum(axis=(0, 1)) / (total + 1e-12)
        pred_sy = (tile * xs.reshape(side, 1)).sum(axis=(0, 1)) / (total + 1e-12)
        return float(pred_sx), float(pred_sy)

    def _maybe_save_attn_overlay(
            self,
            *,
            step_index: int,
            layer_key: str,
            pred_logits: torch.Tensor,  # [B, Head, Q, K]
            gt_logits: torch.Tensor,    # [B, Head, Q, K]
            F: int,
            query_idx_list: List[int],
            key_idx_list: List[int],

    ) -> None:
        """Save attention overlay image similar to example, if configured.

        Rules:
        - Target (query) view is last frame (index F-1).
        - Query point can be specified via visualize_config['viz_query_xy'] (x,y) or
          visualize_config['viz_query_index'] (int). Otherwise a random point inside the
          target view is sampled each time this function is called.
        """
        save_dir = self.visualize_config.get('viz_save_dir', None)
       

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Move to CPU float32 for visualization
        pred = pred_logits.detach().to(dtype=torch.float32, device='cpu')  # [B,H,Q,K]
        gt = gt_logits.detach().to(dtype=torch.float32, device='cpu')

        B, Hh, Q, K = pred.shape
        
        # token geometry
        num_q_views = max(1, len(query_idx_list))
        
        q_hw = Q // num_q_views
        q_side = int(math.sqrt(q_hw))
        assert q_side * q_side == q_hw, f"Q per-view must be square, got {q_hw}"
        
        num_k_views = max(1, len(key_idx_list))
        k_hw = K // num_k_views
        k_side = int(math.sqrt(k_hw))
        assert k_side * k_side == k_hw, f"K per-view must be square, got {k_hw}"

        # choose query token inside target view (F-1)
        assert (F - 1) in query_idx_list, "Target view (last) must be included in query indices."
        tgt_q_view_pos = query_idx_list.index(F - 1)
        
        # query point 선택
        qxy = self.visualize_config.get('viz_query_xy', None)
        qindex = self.visualize_config.get('viz_query_index', None)
        if qxy is not None:
            qx, qy = int(qxy[0]), int(qxy[1])
            q_in_view = int(qy) * q_side + int(qx)
        elif qindex is not None:
            q_in_view = int(qindex) % (q_side * q_side)
            qy, qx = divmod(q_in_view, q_side)
        else:
            q_in_view = int(torch.randint(0, q_side * q_side, (1,), device=torch.device('cpu')).item())
            qy, qx = divmod(q_in_view, q_side)
        q_idx = tgt_q_view_pos * q_hw + q_in_view

        # `pred` and `gt` are expected to already be probabilities (softmax applied
        # by the logit head used upstream). Use them directly and average over heads
        # for aggregated visualization.
        pred_prob = pred.mean(dim=1)[0]  # [Q,K]
        gt_prob = gt.mean(dim=1)[0]      # [Q,K]

        # select query row
        pred_vec = pred_prob[q_idx]  # [K]
        gt_vec = gt_prob[q_idx]      # [K]

        # per key-view heatmap tiles
        def tiles_from_vec(vec: torch.Tensor, num_views: int, side: int) -> List[np.ndarray]:
            tiles: List[np.ndarray] = []
            vec_np = vec.detach().cpu().numpy()
            for v in range(num_views):
                seg = vec_np[v * side * side:(v + 1) * side * side]
                m = seg.min(); M = seg.max()
                if M > m:
                    seg = (seg - m) / (M - m)
                else:
                    seg = np.zeros_like(seg)
                tiles.append(seg.reshape(side, side))
            return tiles

        pred_tiles = tiles_from_vec(pred_vec, num_k_views, k_side)
        gt_tiles = tiles_from_vec(gt_vec, num_k_views, k_side)

        # backgrounds in the same order as key_idx_list
        bg_images: List[np.ndarray] = []
        with torch.no_grad():
            imgs = self.batch['image'][0]  # [F,3,H,W]
            for vidx in key_idx_list:
                t = imgs[vidx].to(dtype=torch.float32, device='cpu')
                if t.min() < 0:  # [-1,1] → [0,1]
                    t = (t + 1.0) / 2.0
                t = torch.clamp(t, 0, 1)
                arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                bg_images.append(arr)

        # target (query) background is last frame F-1
        q_bg = self.batch['image'][0, F - 1].detach().to(dtype=torch.float32, device='cpu')
        if q_bg.min() < 0:
            q_bg = (q_bg + 1.0) / 2.0
        q_bg = torch.clamp(q_bg, 0, 1)
        q_bg_np = (q_bg.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # build canvases (Q image + per-view overlays for pred and gt)
        def build_row(tiles: List[np.ndarray], backgrounds: List[np.ndarray]) -> _PILImage.Image:
            assert len(tiles) == len(backgrounds)
            outs: List[np.ndarray] = []
            for view_idx, (tile, bg) in enumerate(zip(tiles, backgrounds)):
                Hc, Wc = bg.shape[0], bg.shape[1]
                # colorize heatmap and resize to bg size
                color = self._attn_gray_to_rgb(tile)
                color_img = _PILImage.fromarray(color).resize((Wc, Hc), _PILImage.NEAREST)
                color_np = np.array(color_img)
                alpha = float(self.visualize_config.get('viz_alpha', 0.6))
                blended = (bg.astype(np.float32) * (1.0 - alpha) + color_np.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
                
                # 각 view별로 가장 확률이 높은 부분에 파란 점 추가
                blended_pil = _PILImage.fromarray(blended)
                blended_pil = self._draw_max_attention_point(blended_pil, tile, k_side)
                
                # self attention 영역 (target view)에 특별한 경계선 추가
                # current_key_view_idx = key_idx_list[view_idx]
                # if current_key_view_idx == F - 1:  # target view (self attention)
                #     # 노란색 경계선으로 self attention 영역 표시
                #     draw = _PILDraw.Draw(blended_pil)
                #     border_width = 3
                #     w, h = blended_pil.size
                #     # 노란색 경계선 그리기
                #     for i in range(border_width):
                #         draw.rectangle([i, i, w-1-i, h-1-i], outline=(255, 255, 0), width=1)
                
                blended = np.array(blended_pil)
                outs.append(blended)
            row = np.concatenate(outs, axis=1)
            return _PILImage.fromarray(row)

        row_pred = build_row(pred_tiles, bg_images)
        row_gt = build_row(gt_tiles, bg_images)

        # build Q panel (with query dot)
        q_side_px = min(q_bg_np.shape[0], q_bg_np.shape[1])
        q_panel = _PILImage.fromarray(q_bg_np).resize((q_side_px, q_side_px), _PILImage.NEAREST)
        q_panel = self._draw_query_point(q_panel, qx, qy, q_side)

        # compose final image: two rows (Q+Pred) and (Q+GT)
        header_h = max(18, int(q_side_px * 0.20))
        gap = 8
        row_w = q_panel.width + row_pred.width
        row_h1 = max(q_panel.height, row_pred.height)
        row_h2 = max(q_panel.height, row_gt.height)
        canvas_w = row_w
        canvas_h = header_h + row_h1 + gap + header_h + row_h2
        canvas = _PILImage.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))

        # headers text (simple)
        def put_text(img: _PILImage.Image, text: str, xy: Tuple[int, int]):
            d = _PILDraw.Draw(img)
            d.text(xy, text, fill=(255, 255, 255))

        # first row
        x, y = 0, header_h
        canvas.paste(q_panel, (x, y)); x += q_panel.width
        canvas.paste(row_pred, (x, y))
        put_text(canvas, "Q", (4, 2))
        put_text(canvas, f"Pred {layer_key}", (q_panel.width + 4, 2))

        # second row
        x, y = 0, header_h + row_h1 + gap + header_h
        canvas.paste(q_panel, (x, y)); x += q_panel.width
        canvas.paste(row_gt, (x, y))
        put_text(canvas, "Q", (4, header_h + row_h1 + gap + 2))
        put_text(canvas, f"GT {layer_key}", (q_panel.width + 4, header_h + row_h1 + gap + 2))

        # Optional: save to disk if a directory is provided
        out_path = None
        if save_dir is not None:
            out_path = os.path.join(save_dir, f"attn_step{int(step_index)}_{layer_key}_q{int(q_idx)}.png")
            canvas.save(out_path)
            if step_index % 10 == 0:
                print(f"Saved attention overlay: {out_path}")

        # attention 이미지를 저장만 하고 wandb 로깅은 하지 않음 (메인에서 일괄 처리)
        if self.visualize_config.get('viz_log_wandb', True):
            # sequence name: prefer explicit config name then batch
            seq = self.visualize_config.get('viz_seq_name', None)
            if seq is None and isinstance(self.batch, dict) and ('sequence_name' in self.batch):
                seq = self.batch['sequence_name']
                if isinstance(seq, (list, tuple)):
                    seq = seq[0]
            if seq is None:
                seq = 'sample'

            # attention 이미지를 callback 내부에 저장 (wandb 로깅은 메인에서)
            # include step in the key so multiple viz steps don't overwrite each other
            key = f"attn/{seq}/step{int(step_index)}/{layer_key}"
            if not hasattr(self, 'attention_images'):
                self.attention_images = {}

            self.attention_images[key] = {
                'image': canvas,
                'caption': f"{seq} | {layer_key} | step {int(step_index)}"
            }

    def _roll_gt_map(self, gt_tensor: torch.Tensor) -> torch.Tensor:
        """Circularly roll the GT map along the last dimension if configured.

        Minimal, dtype/device-preserving implementation. If `visualize_config['roll_gt_map']`
        is missing or zero, returns the input unchanged.
        """
        shift = self.visualize_config.get('roll_gt_map', 0)
        if not shift:
            return gt_tensor

        try:
            shift_int = int(shift)
        except Exception:
            print(f"Warning: invalid roll_gt_map value: {shift}; skipping roll")
            return gt_tensor

        try:
            print(f"Rolling GT map by {shift_int} positions")
            return torch.roll(gt_tensor, shifts=shift_int, dims=-1)
        except Exception as e:
            print(f"Warning: roll_gt_map failed: {e}")
            return gt_tensor

    def _is_step_enabled(self, step_index):
        # Support separate viz_steps and loss_steps.
        # Semantics: `None` (not provided) => all steps; empty list [] => no steps;
        # otherwise only steps in the list are enabled.
        viz_enabled = False
        loss_enabled = False
        viz_steps = self.visualize_config.get('viz_steps', None)
        loss_steps = self.visualize_config.get('loss_steps', None)
        if viz_steps is None:
            viz_enabled = True
        elif isinstance(viz_steps, (list, tuple)) and len(viz_steps) == 0:
            viz_enabled = False
        else:
            viz_enabled = step_index in viz_steps
        if loss_steps is None:
            loss_enabled = True
        elif isinstance(loss_steps, (list, tuple)) and len(loss_steps) == 0:
            loss_enabled = False
        else:
            loss_enabled = step_index in loss_steps
        return viz_enabled, loss_enabled
    
    def _process_layer_pair(self):
        # 각 layer pair에 대해 loss 계산
        visualize_pairs = list(self.visualize_config['pairs'])
        
        # sort pairs so same unet_layer are consecutive -> avoids popping UNet cache early
        def _get_unet_layer(p):
            if isinstance(p, dict):
                return p.get('unet_layer')
            try:
                return p[0]
            except Exception:
                return None
            
        visualize_pairs = [p for p in visualize_pairs if _get_unet_layer(p) is not None]
        visualize_pairs.sort(key=_get_unet_layer)
        print(f"[DEBUG] Processing {len(visualize_pairs)} layer pairs (sorted): {visualize_pairs}")
        return visualize_pairs
    
    def _get_pred_attn_logit(self, unet_layer: int, current_unet: int, unet_attn_cache: dict):
        if current_unet != unet_layer:
                    # Free previous intermediates for the previous UNet layer.
                    # Use guarded deletes instead of swallowing exceptions.
                    if current_unet is not None:
                        if 'pred_attn_logit' in locals():
                            del pred_attn_logit
                        # safe-pop from cache if present
                        if str(current_unet) in unet_attn_cache:
                            unet_attn_cache.pop(str(current_unet), None)
                        # best-effort GPU memory cleanup
                        torch.cuda.empty_cache()

                    current_unet = unet_layer
                    # ensure UNet cache exists for this unet_layer
                    if str(current_unet) not in unet_attn_cache:
                        print(f"UNet attention cache missing for layer {current_unet}; skipping group")
                        pred_attn_logit = None
                        return None
                    pred_attn_logit = unet_attn_cache[str(current_unet)]
                    
                    ## CFG - second batch is conditional branch
                    if pred_attn_logit.dim() == 4 and pred_attn_logit.shape[1] != 1:
                        pred_attn_logit = pred_attn_logit[-1].unsqueeze(0)
                    return pred_attn_logit
                    
    def _get_gt_tokens(self, vggt_layer: str):
         # Obtain GT tokens for the chosen vggt_layer.
                # - For 'point_map' the batch must include a point_map tensor shaped
                #   (B, V, 3, H, W) which we convert to (B, 1, VHW, C).
                # - For other layers we require `vggt_attn_cache` to contain the
                #   requested layer key and both 'query' and 'key' tensors.
        if vggt_layer == "point_map":
            if 'point_map' not in self.batch:
                raise RuntimeError("[DEBUG] point_map requested for visualization but missing from batch")
            pointmap = self.batch['point_map']
            Bp, Vp, Cp, Hp, Wp = pointmap.shape
            return pointmap.shape, pointmap.permute(0, 1, 3, 4, 2).reshape(Bp, 1, -1, Cp), pointmap.permute(0, 1, 3, 4, 2).reshape(Bp, 1, -1, Cp)
        else:
            if 'vggt_attn_cache' not in self.batch or str(vggt_layer) not in self.batch['vggt_attn_cache']:
                raise RuntimeError(f"[DEBUG] vggt_attn_cache missing layer {vggt_layer} in batch")
            layer_cache = self.batch['vggt_attn_cache'][str(vggt_layer)]
            if 'query' not in layer_cache or 'key' not in layer_cache:
                raise RuntimeError(f"[DEBUG] vggt_attn_cache[{vggt_layer}] missing 'query'/'key')")
            gt_query = layer_cache['query'].detach()
            gt_key = layer_cache['key'].detach()
            return gt_query.shape, gt_query, gt_key
    
    def _get_gt_costmap(self, gt_query_resized, gt_key_resized, pair_metric: str, num_head: int):
        metric_fn = COST_METRIC_FN.get(pair_metric, None)
        print(f"[DEBUG] Using costmap metric: {pair_metric}")
        if metric_fn is None:
            raise ValueError(f"Unknown costmap metric {pair_metric}, falling back to neg_log_l2")
        
        # gt_attn_logit: (B, Head, pred_query_size, pred_key_size)
        gt_attn_logit = metric_fn(gt_query_resized, gt_key_resized)
        
        # Head expansion for GT
        # gt_query_resized : (B, 1, pred_query_size, pred_query_size, C) -> (B, Head, pred_query_size, pred_query_size, C)
        # gt_key_resized : (B, 1, pred_key_size, pred_key_size, C) -> (B, Head, pred_key_size, pred_key_size, C)
        pred_head = num_head
        head_gt = gt_query_resized.shape[1] if gt_query_resized.dim() == 4 else 1
        if head_gt == 1 and pred_head > 1:
            gt_query_resized = gt_query_resized.expand(-1, pred_head, -1, -1).contiguous()
            gt_key_resized = gt_key_resized.expand(-1, pred_head, -1, -1).contiguous()
        return gt_attn_logit
    
    
    def _get_loss_fn(self, pair_loss_fn: str):
        # determine chosen loss name (pair override or global)
        if pair_loss_fn is None:
            raise ValueError(f"Pair must provide 'loss_fn' entry for loss calculation: pair={pair}")
        # support passing dict: {'fn': 'cross_entropy', 'head_mean': 'pre'}
        if isinstance(pair_loss_fn, dict):
            fn_name = pair_loss_fn.get('fn') or pair_loss_fn.get('name')
            head_mean = pair_loss_fn.get('head_mean', None)
            if fn_name is None:
                raise ValueError(f"loss_fn dict must contain 'fn'/'name' key: {pair_loss_fn}")
            key = str(fn_name).lower()
            if head_mean is not None:
                hm = str(head_mean).lower()
                if hm in ('none', 'null', 'no'):
                    # treat as no head-mean
                    key = f"{key}_pre"
                    is_head_mean = False
                elif hm in ('pre', 'post'):
                    key = f"{key}_{hm}"
                    is_head_mean = True
                else:
                    raise ValueError(f"Unsupported head_mean value: {head_mean}")
        else:
            # default; pre-headmean
            is_head_mean = True
            key = f"{str(pair_loss_fn).lower()}_pre"
        loss_fn = self.ATTN_LOSS_FN.get(key, None)
        
        if loss_fn is None:
            raise ValueError(f"Unknown/unsupported loss key requested: {key}. Available: {list(self.ATTN_LOSS_FN.keys())}")
        print(f"[DEBUG] Using loss_fn: {key}")    
        return loss_fn, key, is_head_mean
    
    def __call__(
        self,
        pipeline,
        step_index: int,
        timestep: int,
        callback_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        매 denoising step마다 호출되는 callback 함수
        UNet attention을 캐시에서 가져와 VGGT attention과 비교하여 loss 계산
        """
        viz_enabled, loss_enabled = self._is_step_enabled(step_index) 
          
        if not (viz_enabled or loss_enabled):
            print(f"[DEBUG] Skipping attention visualization callback for step {step_index}")
            clear_attn_cache(pipeline.unet)
            return callback_kwargs
        
        print(f"[DEBUG] AttentionVisualizationCallback.__call__ invoked: step_index={step_index}, timestep={timestep}")
        print(f"[DEBUG] do_attn_visualize: {self.do_attn_visualize}")
        
        ### 주 Logic ###
        try:
            # UNet attention cache 가져오기
            unet_attn_cache = pop_cached_attn(pipeline.unet)
            print(f"[DEBUG] UNet attention cache keys: {list(unet_attn_cache.keys()) if unet_attn_cache else 'None'}")
            
            if not unet_attn_cache:
                # attention cache가 비어있으면 스킵
                print("[DEBUG] UNet attention cache is empty, skipping")
                return callback_kwargs
            
            # 이미지 shape 정보
            image = self.batch['image']
            B, F, _, H, W = image.shape
            
            step_loss_dict = {}
            current_unet = None
            pred_attn_logit = None
            
            visualize_pairs = self._process_layer_pair()

            for pair in visualize_pairs:
                
                if isinstance(pair, dict):
                    unet_layer = pair.get('unet_layer')
                    vggt_layer = pair.get('vggt_layer')
                    pair_metric = pair.get('costmap_metric', None)
                    pair_loss_fn = pair.get('loss_fn', None)
                else:
                    raise ValueError(f"invalid pair entry: {pair}")

                pred_attn_logit = self._get_pred_attn_logit(unet_layer, current_unet, unet_attn_cache)

                print(f"[DEBUG] Processing layer pair: {unet_layer}, {vggt_layer}")
                print(f"[DEBUG] UNet attention cache keys: {list(unet_attn_cache.keys()) if unet_attn_cache else 'None'}")
                print(f"[DEBUG] VGGT attention cache keys: {list(self.batch.get('vggt_attn_cache', {}).keys()) if self.batch.get('vggt_attn_cache') else 'None'}")

                gt_query_shape, gt_query, gt_key = self._get_gt_tokens(vggt_layer)

                # Ensure pred exists 
                if pred_attn_logit is None:
                    raise RuntimeError(f"[DEBUG] missing UNet attention logits for unet layer {unet_layer}")
                Q, K = pred_attn_logit.shape[-2], pred_attn_logit.shape[-1]
                if Q != K:
                    raise ValueError(f"pred attn must have equal Q and K dims, got {pred_attn_logit.shape}")
                if Q % F != 0 or K % F != 0:
                    raise ValueError(f"pred attnmap must be divisible by F; got {pred_attn_logit.shape} and F={F}")

                #  move GT to the same dtype/device as pred
                target_dtype = pred_attn_logit.dtype
                target_device = pred_attn_logit.device
                gt_query = gt_query.to(dtype=target_dtype, device=target_device)
                gt_key = gt_key.to(dtype=target_dtype, device=target_device)
            
                pred_query_size = int(math.sqrt(Q // F))
                pred_key_size = int(math.sqrt(K // F))
                gt_query_resized = self._resize_token(gt_query, pred_query_size, F)
                gt_key_resized = self._resize_token(gt_key, pred_key_size, F)
                
                ### 차원 정리
                # gt_query_resized: (B, 1, pred_query_size, pred_query_size, C)
                # gt_key_resized: (B, 1, pred_key_size, pred_key_size, C)
                # pred_attn_logit: (B, Head, pred_query_size, pred_key_size)
                
                unet_softmax_head, vggt_softmax_head, unet_viz_head, vggt_viz_head = self._resolve_heads(pair)
                if unet_softmax_head is None or vggt_softmax_head is None or unet_viz_head is None or vggt_viz_head is None:
                    raise RuntimeError(f"Missing unet/vggt logit head for pair: {pair}")

                if loss_enabled:
                    layer_key = f"unet{unet_layer}_vggt{vggt_layer}"
                    print(f"Calculating loss for step {step_index}, layer {layer_key}")
                    loss_query_idx, loss_key_idx = self._extract_view_indices(F, mode="loss")
                    
                    # pred_attn_logit_sliced: (B, Head, pred_query_size, pred_key_size) / (2,10,1024,2048)
                    pred_attn_logit_sliced = self._slice_attention_map(pred_attn_logit, loss_query_idx, loss_key_idx, F)
                    
                    # gt_attn_logit_sliced: (B, Head, pred_query_size, pred_key_size)
                    gt_attn_logit = self._get_gt_costmap(gt_query_resized, gt_key_resized, pair_metric, num_head=pred_attn_logit.shape[1])
                    gt_attn_logit_sliced = self._slice_attention_map(gt_attn_logit, loss_query_idx, loss_key_idx, F)

                    # Instantiate logit-heads for UNet and VGGT outputs.
                    # Preference order: 1) per-pair explicit head class, 2) global visualize_config
                    # if per-view => (View, Head, HW, View, HW)
                    # else global => (view, Head, HW, View*HW)
                    pred_processed_viz = unet_softmax_head(pred_attn_logit_sliced)
                    gt_processed_viz = vggt_softmax_head(gt_attn_logit_sliced)
                    
                    # aggregated loss 계산
                    loss_fn, chosen_fn_str, is_head_mean = self._get_loss_fn(pair_loss_fn)
                    loss_value = loss_fn(pred_processed_viz, gt_processed_viz)

                    # include chosen loss function name in the step-level key
                    chosen_fn_str = str(chosen_fn_str).replace('/', '_')
                    
                    # store average for backwards compatibility
                    per_head_list = [float(x) for x in loss_value.detach().cpu().view(-1).tolist()]
                    loss_scalar = float(sum(per_head_list) / len(per_head_list))
                    step_loss_dict[f"val/step{step_index}/{layer_key}/{chosen_fn_str}"] = loss_scalar

                    # store per-head entries under head indices so logging can read head-wise metrics
                    for hid, hv in enumerate(per_head_list):
                        step_loss_dict[f"val/step{step_index}/{layer_key}/{chosen_fn_str}/head{hid}"] = hv

                    print(f"[DEBUG] Calculated loss for {layer_key} (fn={chosen_fn_str}): avg={loss_scalar}, per_head={per_head_list}")

                    if layer_key not in self.layer_losses:
                        self.layer_losses[layer_key] = []
                    # store the per-head list (not scalar) for later analysis
                    self.layer_losses[layer_key].append(per_head_list)
                    
                if viz_enabled:
                    viz_query_idx, viz_key_idx = self._extract_view_indices(F, mode="viz")
                    
                    pred_attn_logit_viz = self._slice_attention_map(pred_attn_logit, viz_query_idx, viz_key_idx, F)
                    gt_attn_logit_viz = self._slice_attention_map(gt_attn_logit, viz_query_idx, viz_key_idx, F)

                    # For visualization, always use Softmax_HeadMean and only consider
                    viz_mode = pair.get('viz_softmax_mode', None)
                    # determine number of key-views for viz (prefer pair, then global config)
                    viz_num_k = pair.get('viz_num_key_views', None)
                    if viz_num_k is None:
                        viz_num_k = int(self.visualize_config.get('viz_num_key_views', self.visualize_config.get('loss_num_key_views', 1)))
                    per_view_flag = str(viz_mode).lower() == 'per_view'
                    num_view_for_per_view = int(viz_num_k) if per_view_flag else None

                    unet_head = Softmax_HeadMean(softmax_temp=1.0, num_view_for_per_view=num_view_for_per_view)
                    vggt_head = Softmax_HeadMean(softmax_temp=1.0, num_view_for_per_view=num_view_for_per_view)

                    pred_processed_viz = unet_head(pred_attn_logit_viz)
                    gt_processed_viz = vggt_head(gt_attn_logit_viz)

                    self._maybe_save_attn_overlay(
                        step_index=step_index,
                        layer_key=layer_key,
                        pred_logits=pred_processed_viz,
                        gt_logits=gt_processed_viz,
                        F=F,
                        query_idx_list=viz_query_idx,
                        key_idx_list=viz_key_idx,
                    )

                # cleanup per-pair intermediates
                try:
                    del gt_query, gt_key, gt_query_resized, gt_key_resized, gt_attn_logit, gt_attn_logit_sliced
                except Exception:
                    pass
                try:
                    del pred_attn_logit_viz, gt_attn_logit_viz, pred_attn_logit_sliced
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # after finishing all pairs for all unet layers, free any remaining pred cache entries
            # Now aggregate step-level losses collected in step_loss_dict
            if step_loss_dict:
                try:
                    step_avg_loss = sum(step_loss_dict.values()) / len(step_loss_dict)
                except Exception:
                    # convert tensors to floats if necessary
                    vals = [v.item() if hasattr(v, 'item') else float(v) for v in step_loss_dict.values()]
                    step_avg_loss = sum(vals) / len(vals)
                step_loss_dict[f"val/step{step_index}/avg_loss"] = step_avg_loss

                # step별 layer별 loss 저장
                self.step_layer_losses[step_index] = {}
                for key, loss_value in step_loss_dict.items():
                    if key.startswith(f"val/step{step_index}/") and key != f"val/step{step_index}/avg_loss":
                        # Preserve the rest of the key after val/step{step_index}/ as-is
                        layer_key = key.replace(f"val/step{step_index}/", "")
                        # store numeric value for metric
                        self.step_layer_losses[step_index][layer_key] = loss_value.item() if hasattr(loss_value, 'item') else float(loss_value)

                self.step_losses.append(step_avg_loss.item() if hasattr(step_avg_loss, 'item') else float(step_avg_loss))
                print(f"[DEBUG] Step {step_index} avg loss: {float(step_avg_loss):.6f}")

                # 전체 loss dict에 추가
                self.visualize_loss_dict.update(step_loss_dict)
                
                
        except Exception as e:
            print(f"Error in attention visualization callback at step {step_index}: {e}")
            import traceback
            traceback.print_exc()
            
            
        finally:
            # 각 step 이후 UNet attention cache 정리
            if hasattr(pipeline, 'unet'):
                clear_attn_cache(pipeline.unet)
                if step_index % 10 == 0:  # 10 step마다만 로그 출력
                    print(f"Cleared UNet attention cache after step {step_index}")
        
        return callback_kwargs
    
    def get_final_loss(self) -> torch.Tensor:
        """모든 step의 평균 loss 반환"""
        if not self.step_losses:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        return torch.tensor(sum(self.step_losses) / len(self.step_losses), 
                          device=self.device, dtype=torch.float32)
    
    def get_loss_dict(self) -> Dict[str, torch.Tensor]:
        """전체 loss dictionary 반환"""
        # 최종 평균 loss도 포함
        final_dict = self.visualize_loss_dict.copy()
        final_dict["val/visualize/final_avg_loss"] = self.get_final_loss()
        return final_dict
    
    def get_structured_losses(self) -> Dict[str, Any]:
        """구조화된 loss 정보 반환 (wandb 로깅용)"""
        structured_losses = {
            'step_losses': self.step_losses,
            'step_layer_losses': self.step_layer_losses,
            'layer_summary': {},
            'overall_summary': {}
        }
        
        # Layer별 통계 계산
        for layer_key, losses in self.layer_losses.items():
            if losses:
                # losses may contain per-step per-head lists; normalize to per-step scalars
                scalar_vals = []
                for item in losses:
                    if isinstance(item, (list, tuple)):
                        # average per-head values to a scalar for summary
                        try:
                            scalar_vals.append(float(sum(item) / len(item)))
                        except Exception:
                            # fallback: coerce first element
                            scalar_vals.append(float(item[0]))
                    else:
                        scalar_vals.append(float(item))

                mean_val = sum(scalar_vals) / len(scalar_vals)
                min_val = min(scalar_vals)
                max_val = max(scalar_vals)
                std_val = (sum((x - mean_val) ** 2 for x in scalar_vals) / len(scalar_vals)) ** 0.5 if len(scalar_vals) > 1 else 0.0

                structured_losses['layer_summary'][layer_key] = {
                    'mean': mean_val,
                    'min': min_val,
                    'max': max_val,
                    'std': std_val,
                    'count': len(losses)
                }
        
        # 전체 통계
        if self.step_losses:
            structured_losses['overall_summary'] = {
                'mean': sum(self.step_losses) / len(self.step_losses),
                'min': min(self.step_losses),
                'max': max(self.step_losses),
                'std': (sum((x - sum(self.step_losses)/len(self.step_losses))**2 for x in self.step_losses) / len(self.step_losses))**0.5 if len(self.step_losses) > 1 else 0.0,
                'total_steps': len(self.step_losses)
            }
        
        return structured_losses
    
    def get_attention_images(self) -> Dict[str, Dict[str, Any]]:
        """저장된 attention 이미지들 반환 (wandb 로깅용)"""
        if hasattr(self, 'attention_images'):
            return self.attention_images.copy()
        return {}
    
    def clear_vggt_cache(self):
        """VGGT attention cache 정리 (pipeline 완료 후 호출)"""
        if 'vggt_attn_cache' in self.batch:
            # VGGT cache 메모리 정리
            for layer_key in list(self.batch['vggt_attn_cache'].keys()):
                if 'query' in self.batch['vggt_attn_cache'][layer_key]:
                    del self.batch['vggt_attn_cache'][layer_key]['query']
                if 'key' in self.batch['vggt_attn_cache'][layer_key]:
                    del self.batch['vggt_attn_cache'][layer_key]['key']
                del self.batch['vggt_attn_cache'][layer_key]
            del self.batch['vggt_attn_cache']
            print("Cleared VGGT attention cache")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
    
    def reset(self):
        """callback 상태 초기화"""
        self.visualize_loss_dict.clear()
        self.step_losses.clear()
        self.step_layer_losses.clear()
        self.layer_losses.clear()
        if hasattr(self, 'attention_images'):
            self.attention_images.clear()
