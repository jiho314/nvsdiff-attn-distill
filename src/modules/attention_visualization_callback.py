import torch
import math
import einops
from typing import Dict, List, Tuple, Optional, Any
from my_diffusers.callbacks import PipelineCallback
from src.distill_utils.attn_processor_cache import pop_cached_attn, clear_attn_cache
from src.distill_utils.query_key_cost_metric import COST_METRIC_FN
from src.distill_utils.attn_logit_head import LOGIT_HEAD_CLS


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
        loss_fn = self.visualize_config.get('loss_fn', 'cross_entropy')

        # define canonical loss functions and store mapping for per-pair overrides
        def cross_entropy(prob, prob_gt):
            """Cross entropy loss for attention probabilities.

            Expects inputs to already be probabilities over last dim.
            """
            eps = 1e-8
            return - (prob_gt * (prob + eps).log()).sum(dim=-1).mean()

        def kl_divergence(prob, prob_gt):
            """Kullback-Leibler divergence loss for attention probabilities."""
            return (prob_gt * (prob_gt.log() - prob.log())).sum(dim=-1).mean()

        def softargmax_l2_from_prob(prob_pred, prob_gt, num_key_views: int):
            """Compute L2 between 2D soft-argmax coordinates from probability inputs.

            Simplified strict version: requires `num_key_views` (int). Assumes inputs are
            probabilities shaped [..., Q, K] and sum to 1 along the last dim. K must equal
            num_key_views * (kp**2) where kp is integer spatial side.

            This function returns L2 computed on *pixel/index coordinates* (0..kp-1), not
            normalized [0,1] coordinates.

            Returns scalar mean L2 averaged over batch, heads, queries and key-views.
            """
            # last-dim K
            K = int(prob_pred.shape[-1])
            if K <= 0:
                raise ValueError(f"Invalid K for softargmax_l2: K={K}")

            num_k = int(num_key_views)
            if num_k <= 0 or K % num_k != 0:
                raise ValueError(f"num_key_views={num_k} is not a valid divisor of K={K}")

            other = K // num_k
            kp = int(round(math.sqrt(other)))
            if kp * kp != other:
                raise ValueError(f"Given num_key_views={num_k} does not yield a square kp for K={K}")

            # reshape to [..., Q, num_k, kp, kp]
            p_pred = prob_pred.view(*prob_pred.shape[:-1], num_k, kp, kp)
            p_gt = prob_gt.view(*prob_gt.shape[:-1], num_k, kp, kp)

            # coordinate grids in pixel/index space: 0..kp-1
            xs = torch.arange(0, kp, device=p_pred.device, dtype=p_pred.dtype)
            ys = torch.arange(0, kp, device=p_pred.device, dtype=p_pred.dtype)
            # build broadcastable grids matching p_pred shape [..., num_k, kp, kp]
            nd = p_pred.dim()
            leading = nd - 3  # number of leading dims before (num_k, kp, kp)

            # grid_x should have shape [..., num_k, 1, kp]
            grid_x = xs
            for _ in range(leading + 2):
                grid_x = grid_x.unsqueeze(0)
            grid_x = grid_x.to(device=p_pred.device, dtype=p_pred.dtype)
            grid_x = grid_x.expand(*p_pred.shape[:-2], 1, kp)

            # grid_y should have shape [..., num_k, kp, 1]
            grid_y = ys
            for _ in range(leading + 1):
                grid_y = grid_y.unsqueeze(0)
            grid_y = grid_y.unsqueeze(-1)
            grid_y = grid_y.to(device=p_pred.device, dtype=p_pred.dtype)
            grid_y = grid_y.expand(*p_pred.shape[:-2], kp, 1)

            # expected coordinates per-key-view: [..., Q, num_k]
            pred_x = (p_pred * grid_x).sum(dim=(-2, -1))
            pred_y = (p_pred * grid_y).sum(dim=(-2, -1))
            gt_x = (p_gt * grid_x).sum(dim=(-2, -1))
            gt_y = (p_gt * grid_y).sum(dim=(-2, -1))

            # stack to coords: [..., Q, num_k, 2]
            pred_coords = torch.stack([pred_x, pred_y], dim=-1)
            gt_coords = torch.stack([gt_x, gt_y], dim=-1)

            # Euclidean distance (L2) per key-view: sqrt(sum((x-y)^2))
            diff = pred_coords - gt_coords
            per_view_sq = diff.pow(2).sum(dim=-1)  # squared L2 [..., Q, num_k]
            per_view_l2 = per_view_sq.sqrt()  # Euclidean L2 [..., Q, num_k]

            # average L2 over batch, heads, queries and key-views
            loss = per_view_l2.mean()

            # # Detailed runtime logging to help debug small loss values
            # print(f"[softargmax_l2] prob_pred.shape={prob_pred.shape}, prob_gt.shape={prob_gt.shape}, K={K}, num_k={num_k}, kp={kp}")
            # print(f"[softargmax_l2] pred_prob mean/min/max = {float(prob_pred.mean()):.6e}/{float(prob_pred.min()):.6e}/{float(prob_pred.max()):.6e}")
            # print(f"[softargmax_l2] gt_prob   mean/min/max = {float(prob_gt.mean()):.6e}/{float(prob_gt.min()):.6e}/{float(prob_gt.max()):.6e}")
            # # sample a few coordinates (first batch/head/query/keyview)
            # try:
            #     sample_pred = pred_coords.detach().cpu().numpy()
            #     sample_gt = gt_coords.detach().cpu().numpy()
            #     # print first few coords for first query & first key-view
            #     print(f"[softargmax_l2] sample pred_coords[0,0,0,:5] = {sample_pred.reshape(-1, sample_pred.shape[-1])[0:5,:].tolist()}")
            #     print(f"[softargmax_l2] sample gt_coords[0,0,0,:5]   = {sample_gt.reshape(-1, sample_gt.shape[-1])[0:5,:].tolist()}")
            # except Exception:
            #     pass
            # try:
            #     print(f"[softargmax_l2] per_view_l2 stats mean/min/max = {float(per_view_l2.mean()):.6e}/{float(per_view_l2.min()):.6e}/{float(per_view_l2.max()):.6e}")
            # except Exception:
            #     pass
            # print(f"[softargmax_l2] loss = {float(loss):.6e}")
            # # Save small visualization: overlay pred/gt heatmaps for first sample
            
            # save_dir = "batch_viz/"
            # if save_dir is not None:
            #     os.makedirs(save_dir, exist_ok=True)
            #     # take first batch, first head, first query
            #     p_sample = prob_pred[0, 0, :].view(num_k, kp, kp).detach().cpu().numpy()
            #     g_sample = prob_gt[0, 0, :].view(num_k, kp, kp).detach().cpu().numpy()
            #     # build side-by-side image per key-view
            #     rows = []
            #     for v in range(num_k):
            #         p_img = (p_sample[v] - p_sample[v].min()) / (np.ptp(p_sample[v]) + 1e-8)
            #         g_img = (g_sample[v] - g_sample[v].min()) / (np.ptp(g_sample[v]) + 1e-8)
            #         p_rgb = (self._attn_gray_to_rgb(p_img) )
            #         g_rgb = (self._attn_gray_to_rgb(g_img) )
            #         row = np.concatenate([p_rgb, g_rgb], axis=1)
            #         rows.append(row)
            #     comp = np.concatenate(rows, axis=0)
            #     fname = os.path.join(save_dir, f"softargmax_viz_{int(time.time())}_{uuid.uuid4().hex[:6]}.png")
            #     _PILImage.fromarray(comp).save(fname)
            #     print(f"[softargmax_l2] saved viz: {fname}")

            return loss

        def _logits_to_probs(logits: torch.Tensor, num_key_views: int, mode: Optional[str] = None) -> torch.Tensor:
            """Convert logits to probabilities.

            NOTE: per-view and per_view_batch modes removed. Always apply a global
            softmax over the last dimension. If inputs already look like
            probabilities (sum ~1) they are returned unchanged.
            """
            if logits is None:
                return logits

            # quick heuristic: if values along K already sum to ~1, assume probs and return
            try:
                sums = logits.sum(dim=-1)
                if torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
                    return logits
            except Exception:
                pass

            # always use global softmax across last dim
            return logits.softmax(dim=-1)

        def cross_entropy_per_view(prob_pred, prob_gt, num_key_views: int):
            """Per-view cross entropy. Expects inputs shaped [..., K] or [B,Head,Q,K].

            This computes cross-entropy per key-view tile and averages.
            """
            # unify shapes to [..., K]
            if prob_pred.dim() == 4:
                # [B,Head,Q,K] -> [..., K]
                flat_pred = prob_pred
                flat_gt = prob_gt
            else:
                flat_pred = prob_pred
                flat_gt = prob_gt

            K = int(flat_pred.shape[-1])
            num_k = int(num_key_views)
            if num_k <= 0 or K % num_k != 0:
                raise ValueError(f"num_key_views must divide K: num_key_views={num_k}, K={K}")
            per = K // num_k

            # reshape to [..., num_k, per]
            p_pred = flat_pred.view(*flat_pred.shape[:-1], num_k, per)
            p_gt = flat_gt.view(*flat_gt.shape[:-1], num_k, per)

            # ensure each tile is probability distribution
            # apply softmax per tile if sums not ~1
            pred_flat = p_pred.view(-1, per)
            gt_flat = p_gt.view(-1, per)
            if not torch.allclose(pred_flat.sum(dim=-1), torch.ones_like(pred_flat.sum(dim=-1)), atol=1e-3):
                pred_flat = pred_flat.softmax(dim=-1)
            if not torch.allclose(gt_flat.sum(dim=-1), torch.ones_like(gt_flat.sum(dim=-1)), atol=1e-3):
                gt_flat = gt_flat.softmax(dim=-1)

            # compute cross entropy per tile
            eps = 1e-8
            ce = - (gt_flat * (pred_flat + eps).log()).sum(dim=-1)
            # average across tiles and remaining dims
            return ce.view(*p_pred.shape[:-2], num_k).mean()

        def argmax_l2_from_prob(prob_pred, prob_gt, num_key_views: int):
            """Compute L2 between argmax coordinates from probability inputs.

            This is a discrete (hard) argmax variant: for each key-view we take the
            index of the maximum probability and compute pixel/index coordinates
            then L2 to the GT argmax. Inputs have same shape/assumptions as
            `softargmax_l2_from_prob`.
            """
            # last-dim K
            K = int(prob_pred.shape[-1])
            if K <= 0:
                raise ValueError(f"Invalid K for argmax_l2: K={K}")

            num_k = int(num_key_views)
            if num_k <= 0 or K % num_k != 0:
                raise ValueError(f"num_key_views={num_k} is not a valid divisor of K={K}")

            other = K // num_k
            kp = int(round(math.sqrt(other)))
            if kp * kp != other:
                raise ValueError(f"Given num_key_views={num_k} does not yield a square kp for K={K}")

            # reshape to [..., Q, num_k, kp, kp]
            p_pred = prob_pred.view(*prob_pred.shape[:-1], num_k, kp, kp)
            p_gt = prob_gt.view(*prob_gt.shape[:-1], num_k, kp, kp)

            # compute argmax indices per tile -> (..., Q, num_k)
            # flatten last two dims and argmax
            p_pred_flat = p_pred.view(*p_pred.shape[:-2], -1)
            p_gt_flat = p_gt.view(*p_gt.shape[:-2], -1)
            pred_idx = p_pred_flat.argmax(dim=-1)
            gt_idx = p_gt_flat.argmax(dim=-1)

            # convert 1D index to 2D coords (y,x): y = idx // kp, x = idx % kp
            pred_y = (pred_idx // kp).to(dtype=prob_pred.dtype, device=prob_pred.device)
            pred_x = (pred_idx % kp).to(dtype=prob_pred.dtype, device=prob_pred.device)
            gt_y = (gt_idx // kp).to(dtype=prob_gt.dtype, device=prob_gt.device)
            gt_x = (gt_idx % kp).to(dtype=prob_gt.dtype, device=prob_gt.device)

            pred_coords = torch.stack([pred_x, pred_y], dim=-1)
            gt_coords = torch.stack([gt_x, gt_y], dim=-1)

            diff = pred_coords - gt_coords
            per_view_sq = diff.pow(2).sum(dim=-1)
            per_view_l2 = per_view_sq.sqrt()

            loss = per_view_l2.mean()
            return loss

        def gaussian_a2b_nll_from_prob(prob_pred, prob_gt, num_key_views: int, sigma: float):
            """Gaussian NLL interpreted as A->B: -sum_x A(x) log K_sigma(x,y*)

            Here prob_gt is expected to be a one-hot (delta) per tile; we infer y* from argmax of prob_gt.
            The NLL reduces (up to const) to (1/(2 sigma^2)) * E_x[||x - y*||^2].
            We compute E[x^2+y^2] - 2 E[x]·y* + ||y*||^2 per tile and average.
            """
            # unify shapes [..., K]
            flat_pred = prob_pred
            flat_gt = prob_gt
            K = int(flat_pred.shape[-1])
            num_k = int(num_key_views)
            if num_k <= 0 or K % num_k != 0:
                raise ValueError(f"num_key_views must divide K: num_key_views={num_k}, K={K}")
            per = K // num_k

            # reshape to [..., num_k, per]
            p_pred = flat_pred.view(*flat_pred.shape[:-1], num_k, per)
            p_gt = flat_gt.view(*flat_gt.shape[:-1], num_k, per)

            # coordinate grids
            device = p_pred.device
            dtype = p_pred.dtype
            xs = torch.arange(0, per, device=device, dtype=dtype)
            ys = torch.arange(0, per, device=device, dtype=dtype)
            # build grid coords (x,y) for index idx -> (x=idx%kp, y=idx//kp) where kp = sqrt(per)
            kp = int(round(math.sqrt(per)))
            gx = (xs % kp).to(dtype=dtype)
            gy = (xs // kp).to(dtype=dtype)

            # compute expectations per tile: E[x], E[y], E[x^2+y^2]
            # p_pred shape [..., num_k, per]
            ex = (p_pred * gx.view(*([1] * (p_pred.dim() - 2)), kp).reshape([1]*(p_pred.dim()-2) + [kp]).view(-1)).sum(dim=-1)
            # above broadcasting is tricky; instead compute with explicit tensors
            # compute gx,gy as tensors of shape (per,)
            gx = gx.view(1, 1, 1, -1) if p_pred.dim() == 4 else gx.view(1, -1)
            gy = gy.view(1, 1, 1, -1) if p_pred.dim() == 4 else gy.view(1, -1)
            # fallback generic computation using broadcasting with torch
            # reshape p_pred to (-1, per) to compute easily
            pred_flat = p_pred.view(-1, per)
            ex_flat = (pred_flat * (torch.arange(0, per, device=device, dtype=dtype) % kp)).sum(dim=-1)
            ey_flat = (pred_flat * (torch.arange(0, per, device=device, dtype=dtype) // kp)).sum(dim=-1)
            ex2_flat = (pred_flat * ((torch.arange(0, per, device=device, dtype=dtype) % kp)**2 + (torch.arange(0, per, device=device, dtype=dtype) // kp)**2)).sum(dim=-1)

            # get gt hard coords per tile from p_gt argmax
            gt_flat = p_gt.view(-1, per)
            gt_idx = gt_flat.argmax(dim=-1)
            gt_x = (gt_idx % kp).to(dtype=dtype)
            gt_y = (gt_idx // kp).to(dtype=dtype)

            # Align counts between pred tiles and gt tiles if broadcasting occurred
            pred_flat = p_pred.view(-1, per)
            Np = pred_flat.shape[0]
            Ng = gt_flat.shape[0]
            if Np != Ng:
                if Np % Ng == 0:
                    factor = Np // Ng
                    gt_x = gt_x.repeat_interleave(factor)
                    gt_y = gt_y.repeat_interleave(factor)
                elif Ng % Np == 0:
                    factor = Ng // Np
                    # replicate pred expectations to match gt (rare); do by repeating rows
                    ex_flat = ex_flat.repeat_interleave(factor)
                    ey_flat = ey_flat.repeat_interleave(factor)
                    ex2_flat = ex2_flat.repeat_interleave(factor)
                else:
                    raise ValueError(f"Incompatible pred/gt tile counts for gaussian_a2b: pred_tiles={Np}, gt_tiles={Ng}")

            # compute expected squared dist per tile
            # E||x - y*||^2 = E[x^2+y^2] - 2(E[x]*y_x + E[y]*y_y) + (y_x^2 + y_y^2)
            term = ex2_flat - 2.0 * (ex_flat * gt_x + ey_flat * gt_y) + (gt_x * gt_x + gt_y * gt_y)

            # reshape back to [..., num_k]
            out = term.view(*p_pred.shape[:-2], num_k)
            # mean over tiles and remaining dims, scale by 1/(2 sigma^2)
            loss = out.mean() / (2.0 * (sigma**2))
            return loss

        def gaussian_b2a_nll_from_prob(prob_pred, prob_gt, num_key_views: int, sigma: float, eps: float = 1e-8):
            """Compute -log( (K_sigma * A)(y*) ) where y* is GT argmax per tile.

            This computes for each tile: -log( sum_x A(x) * exp(-||x-y*||^2/(2 sigma^2)) ).
            """
            flat_pred = prob_pred
            flat_gt = prob_gt
            K = int(flat_pred.shape[-1])
            num_k = int(num_key_views)
            if num_k <= 0 or K % num_k != 0:
                raise ValueError(f"num_key_views must divide K: num_key_views={num_k}, K={K}")
            per = K // num_k

            p_pred = flat_pred.view(-1, num_k, per)
            p_gt = flat_gt.view(-1, num_k, per)

            kp = int(round(math.sqrt(per)))
            # create coordinate grid indices 0..per-1
            idx = torch.arange(0, per, device=flat_pred.device, dtype=flat_pred.dtype)
            gx = (idx % kp).to(dtype=flat_pred.dtype)
            gy = (idx // kp).to(dtype=flat_pred.dtype)

            # gt indices
            gt_flat = p_gt.view(-1, per)
            gt_idx = gt_flat.argmax(dim=-1)
            gt_x = (gt_idx % kp).to(dtype=flat_pred.dtype).unsqueeze(-1)
            gt_y = (gt_idx // kp).to(dtype=flat_pred.dtype).unsqueeze(-1)

            # compute squared distances (per gt tile vs all x)
            dx2 = (gx.unsqueeze(0) - gt_x)**2
            dy2 = (gy.unsqueeze(0) - gt_y)**2
            dist2 = dx2 + dy2  # shape (Ntiles, per)

            # gaussian kernel weights (unnormalized)
            weights = torch.exp(-dist2 / (2.0 * sigma**2))

            pred_flat = p_pred.view(-1, per)

            # Align counts: if pred_flat rows != weights rows, try broadcasting by repeating gt rows
            Np = pred_flat.shape[0]
            Nw = weights.shape[0]
            if Np != Nw:
                if Np % Nw == 0:
                    factor = Np // Nw
                    weights = weights.repeat_interleave(factor, dim=0)
                elif Nw % Np == 0:
                    factor = Nw // Np
                    pred_flat = pred_flat.repeat_interleave(factor, dim=0)
                else:
                    raise ValueError(f"Incompatible pred/gt tile counts for gaussian_b2a: pred_tiles={Np}, weight_tiles={Nw}")

            numer = (pred_flat * weights).sum(dim=-1)
            vals = -torch.log(numer + eps)
            loss = vals.mean()
            return loss

        ATTN_LOSS_FN = {
            "l1": torch.nn.functional.l1_loss,
            "cross_entropy": cross_entropy,
            "softargmax_l2": softargmax_l2_from_prob,
            "argmax_l2": argmax_l2_from_prob,
            "kl_divergence": kl_divergence,
            "gauss_a2b": lambda pred, gt, num_k: gaussian_a2b_nll_from_prob(pred, gt, num_k, sigma=2.0),
            "gauss_b2a": lambda pred, gt, num_k: gaussian_b2a_nll_from_prob(pred, gt, num_k, sigma=2.0),
            "gauss_both": lambda pred, gt, num_k: 0.5 * (gaussian_a2b_nll_from_prob(pred, gt, num_k, sigma=2.0) + gaussian_b2a_nll_from_prob(pred, gt, num_k, sigma=2.0))
        }
        # always expose mapping attribute
        self.ATTN_LOSS_FN = ATTN_LOSS_FN

        # set default loss_fn
        if callable(loss_fn):
            self.loss_fn = loss_fn
        elif isinstance(loss_fn, str):
            key = loss_fn.lower()
            if key not in ATTN_LOSS_FN:
                raise ValueError(f"Unsupported loss_fn string: {loss_fn}")
            self.loss_fn = ATTN_LOSS_FN[key]
            print(f"Using loss_fn: {key}")
        else:
            raise ValueError(f"Unsupported loss_fn type: {type(loss_fn)}")
    
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
    
    def _extract_loss_view_indices(self, F: int) -> Tuple[List[int], List[int]]:
        """Loss 계산용 query와 key의 view index 추출"""
        # Loss Query index 추출
        if self.visualize_config['loss_query'] == "target":
            query_idx = list(range(self.cond_num, F))
        elif self.visualize_config['loss_query'] == "all":
            query_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['loss_query'] {self.visualize_config['loss_query']} not implemented")
        
        # Loss Key index 추출
        if self.visualize_config['loss_key'] == "reference":
            key_idx = list(range(0, self.cond_num))
        elif self.visualize_config['loss_key'] == "all":
            key_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['loss_key'] {self.visualize_config['loss_key']} not implemented")
        
        return query_idx, key_idx

    def _extract_viz_view_indices(self, F: int) -> Tuple[List[int], List[int]]:
        """Visualization용 query와 key의 view index 추출"""
        # Viz Query index 추출
        if self.visualize_config['viz_query'] == "target":
            query_idx = list(range(self.cond_num, F))
        elif self.visualize_config['viz_query'] == "all":
            query_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['viz_query'] {self.visualize_config['viz_query']} not implemented")
        
        # Viz Key index 추출
        if self.visualize_config['viz_key'] == "reference":
            key_idx = list(range(0, self.cond_num))
        elif self.visualize_config['viz_key'] == "all":
            key_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['viz_key'] {self.visualize_config['viz_key']} not implemented")
        
        return query_idx, key_idx
    
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
            viz_mode: str,
            viz_num_k: int,
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

        # convert logits to probabilities using pair-provided viz_mode and viz_num_k
        pred_p = self._logits_to_probs(pred, int(viz_num_k), mode=viz_mode)
        gt_p = self._logits_to_probs(gt, int(viz_num_k), mode=viz_mode)
        pred_prob = pred_p.mean(dim=1)[0]  # [Q,K]
        gt_prob = gt_p.mean(dim=1)[0]      # [Q,K]

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
                current_key_view_idx = key_idx_list[view_idx]
                if current_key_view_idx == F - 1:  # target view (self attention)
                    # 노란색 경계선으로 self attention 영역 표시
                    draw = _PILDraw.Draw(blended_pil)
                    border_width = 3
                    w, h = blended_pil.size
                    # 노란색 경계선 그리기
                    for i in range(border_width):
                        draw.rectangle([i, i, w-1-i, h-1-i], outline=(255, 255, 0), width=1)
                
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
    
    def _get_gt_costmap(self, vggt_layer: str, pair_metric: str, num_head: int):
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
                # support dict entries
                if isinstance(pair, dict):
                    unet_layer = pair.get('unet_layer')
                    vggt_layer = pair.get('vggt_layer')
                    pair_metric = pair.get('costmap_metric', None)
                    pair_loss_fn_name = pair.get('loss_fn', None)
                else:
                    raise ValueError(f"Invalid pair entry: {pair}")

                # when encountering a new unet layer, free previous pred and load new one
                pred_attn_logit = self._get_pred_attn_logit(unet_layer, current_unet, unet_attn_cache)

                print(f"[DEBUG] Processing layer pair: {unet_layer}, {vggt_layer}")
                print(f"[DEBUG] UNet attention cache keys: {list(unet_attn_cache.keys()) if unet_attn_cache else 'None'}")
                print(f"[DEBUG] VGGT attention cache keys: {list(self.batch.get('vggt_attn_cache', {}).keys()) if self.batch.get('vggt_attn_cache') else 'None'}")

                gt_query_shape, gt_query, gt_key = self._get_gt_tokens(vggt_layer)
                Bp, Vp, Cp, Hp, Wp = gt_query_shape

                # Ensure pred exists 
                if pred_attn_logit is None:
                    raise RuntimeError(f"[DEBUG] Missing UNet attention logits for unet layer {unet_layer}")
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
                
                ### 정리
                # gt_query_resized: (B, 1, pred_query_size, pred_query_size, C)
                # gt_key_resized: (B, 1, pred_key_size, pred_key_size, C)
                # pred_attn_logit: (B, Head, pred_query_size, pred_key_size)

                if loss_enabled:
                    layer_key = f"unet{unet_layer}_vggt{vggt_layer}"
                    print(f"Calculating loss for step {step_index}, layer {layer_key}")
                    loss_query_idx, loss_key_idx = self._extract_loss_view_indices(F)
                    
                    # pred_attn_logit_sliced: (B, Head, pred_query_size, pred_key_size) / (2,10,1024,2048)
                    pred_attn_logit_sliced = self._slice_attention_map(pred_attn_logit, loss_query_idx, loss_key_idx, F)
                    
                    # gt_attn_logit_sliced: (B, Head, pred_query_size, pred_key_size)
                    gt_attn_logit = self._get_gt_costmap(vggt_layer, pair_metric, num_head=pred_attn_logit.shape[1])
                    gt_attn_logit_sliced = self._slice_attention_map(gt_attn_logit, loss_query_idx, loss_key_idx, F)

                    # Instantiate logit-heads for UNet and VGGT outputs.
                    # Preference order:
                    # 1) per-pair explicit head class in pair dict
                    # 2) global visualize_config head names
                    def _resolve_head(name_key, kwargs_key, layer_key):
                        """
                        Instantiate a logit head for the given keys.

                        NOTE: Do NOT prefer or attempt to use any module already attached
                        to `pipeline.unet`. Always instantiate from the per-pair override
                        or fall back to the global `visualize_config` settings.
                        """
                        # 1) If pair explicitly provides a head name, instantiate that class
                        if isinstance(pair, dict) and pair.get(name_key, None) is not None:
                            name = pair[name_key].lower()
                            kw = pair.get(kwargs_key, {})
                            return LOGIT_HEAD_CLS[name](**kw)

                        # 2) Fall back to global visualize_config names (instantiate)
                        gname = self.visualize_config.get(name_key, None)
                        gkw = self.visualize_config.get(kwargs_key, {})
                        if gname is not None:
                            return LOGIT_HEAD_CLS[gname.lower()](**gkw)

                        return None

                    unet_head = _resolve_head('unet_logit_head', 'unet_logit_head_kwargs', unet_layer)
                    vggt_head = _resolve_head('vggt_logit_head', 'vggt_logit_head_kwargs', vggt_layer)
                    if unet_head is None or vggt_head is None:
                        raise RuntimeError(f"Missing unet/vggt logit head for pair: {pair}")

                    # For loss calculations we keep the raw logits and apply
                    # deterministic conversion to probabilities below according
                    # to the global softmax mode. For visualization we process
                    # through the configured logit heads so viz respects
                    # head-specific temperature/processing.
                    pred_processed_loss = pred_attn_logit_sliced
                    gt_processed_loss = gt_attn_logit_sliced
                    pred_processed_viz = unet_head(pred_attn_logit_sliced)
                    gt_processed_viz = vggt_head(gt_attn_logit_sliced)
                    
                    if self.visualize_config.get('per_head_loss', False):
                        # per-head loss 계산
                        raise NotImplementedError(f"Not implemented per-head loss")
                    else:
                        # aggregated loss 계산
                        # determine chosen loss name (pair override or global)
                        if pair_loss_fn_name is not None:
                            # pair may provide a string name
                            chosen_fn = pair_loss_fn_name.lower() if isinstance(pair_loss_fn_name, str) else None
                        else:
                            # pair MUST provide loss function name
                            raise ValueError(f"Pair must provide 'loss_fn' entry for loss calculation: pair={pair}")

                        # log the chosen function (or note fallback)
                        print(f"Using loss_fn: {chosen_fn if chosen_fn is not None else 'default_callable'}")

                        local_loss_fn = self.ATTN_LOSS_FN.get(chosen_fn, None) if chosen_fn is not None else None
                        # For losses that expect probability inputs (cross_entropy, kl_divergence)
                        # convert logits -> probs using global softmax settings. Otherwise keep raw.
                        prob_required = chosen_fn in ('cross_entropy', 'kl_divergence')
                        if prob_required:
                            # Use logits -> log_softmax(path) with per-head temperature when available.
                            # Determine temperatures: prefer head attributes, then per-pair kwargs, then global config.
                            def _resolve_temp(head_obj, pair_kwargs_key, global_cfg_key):
                                t = None
                                try:
                                    if hasattr(head_obj, 'softmax_temp'):
                                        t = head_obj.softmax_temp
                                except Exception:
                                    t = None
                                if t is None:
                                    # pair-level override
                                    if isinstance(pair, dict) and pair.get(pair_kwargs_key, None) is not None:
                                        t = pair.get(pair_kwargs_key, {}).get('softmax_temp', None)
                                if t is None:
                                    t = self.visualize_config.get(global_cfg_key, {}).get('softmax_temp', 1.0)
                                # if tensor, convert to float
                                try:
                                    if torch.is_tensor(t):
                                        t = t.detach().cpu().item()
                                except Exception:
                                    pass
                                return float(t)

                            pred_temp = _resolve_temp(unet_head, 'unet_logit_head_kwargs', 'unet_logit_head_kwargs')
                            gt_temp = _resolve_temp(vggt_head, 'vggt_logit_head_kwargs', 'vggt_logit_head_kwargs')

                            # compute log-prob for pred and prob for gt in a numerically stable way
                            # Support per-view (split) softmax: reshape last dim to (..., num_k, per)
                            use_per_view = bool(self.visualize_config.get('split', False)) or \
                                (self.visualize_config.get('softmax_mode', 'global') == 'per_view') or \
                                (isinstance(pair, dict) and pair.get('loss_softmax_mode', None) == 'per_view')
                            loss_k = int(self.visualize_config.get('loss_num_key_views', 1))

                            if use_per_view and loss_k > 1:
                                # Debugging: print shapes to trace reshape failures
                                try:
                                    print(f"[DEBUG] pred_processed_loss.shape={getattr(pred_processed_loss, 'shape', None)}, gt_processed_loss.shape={getattr(gt_processed_loss, 'shape', None)}, loss_k={loss_k}, pair={pair}")
                                    print(f"[DEBUG] pred_processed_loss.numel={pred_processed_loss.numel() if hasattr(pred_processed_loss, 'numel') else 'NA'}, gt_processed_loss.numel={gt_processed_loss.numel() if hasattr(gt_processed_loss, 'numel') else 'NA'}")
                                except Exception:
                                    pass
                                
                                import pdb; pdb.set_trace()

                                # pred: logits -> log_softmax per tile
                                K = int(pred_processed_loss.shape[-1])
                                if K % loss_k != 0:
                                    raise ValueError(f"loss_num_key_views={loss_k} does not divide K={K}")
                                per = K // loss_k

                                # use each tensor's leading shape for safe reshape
                                leading_pred = pred_processed_loss.shape[:-1]
                                leading_gt = gt_processed_loss.shape[:-1]

                                # reshape pred using its own leading dims
                                try:
                                    p_logits = pred_processed_loss.view(*leading_pred, loss_k, per)
                                    logp = FNN.log_softmax(p_logits / pred_temp, dim=-1).view(*leading_pred, K)
                                except Exception as e:
                                    raise RuntimeError(f"Failed to reshape pred_processed_loss for per-view softmax: shape={pred_processed_loss.shape}, attempted (*{leading_pred}, {loss_k}, {per}): {e}")

                                # reshape gt using its own leading dims
                                try:
                                    g_logits = gt_processed_loss.view(*leading_gt, loss_k, per)
                                    prob_gt = FNN.softmax(g_logits / gt_temp, dim=-1).view(*leading_gt, K)
                                except Exception as e:
                                    raise RuntimeError(f"Failed to reshape gt_processed_loss for per-view softmax: shape={gt_processed_loss.shape}, attempted (*{leading_gt}, {loss_k}, {per}): {e}")

                                # If leading shapes differ, do NOT implicitly expand GT to match pred.
                                # Require caller/upstream to produce matching batch sizes to avoid silent incorrect broadcasts.
                                if leading_gt != leading_pred:
                                    raise RuntimeError(f"Mismatched leading shapes for pred vs gt in per-view loss: pred_leading={leading_pred}, gt_leading={leading_gt}. Upstream must provide matching batch sizes.")
                            else:
                                logp = FNN.log_softmax(pred_processed_loss / pred_temp, dim=-1)
                                prob_gt = FNN.softmax(gt_processed_loss / gt_temp, dim=-1)

                            # optional GT roll
                            prob_gt = self._roll_gt_map(prob_gt)

                            # final CE: -sum(q * logp)
                            loss_value = - (prob_gt * logp).sum(dim=-1).mean()
                        else:
                            # non-prob losses use raw logits (or previously-resolved callable)
                            pred_proc = pred_processed_loss
                            gt_proc = self._roll_gt_map(gt_processed_loss)
                            if local_loss_fn is None:
                                loss_value = self.loss_fn(pred_proc, gt_proc)
                            else:
                                loss_value = local_loss_fn(pred_proc, gt_proc)

                        # include chosen loss function name in the step-level key
                        chosen_fn_str = chosen_fn if chosen_fn is not None else 'default_callable'
                        # sanitize chosen_fn_str for use in metric key
                        try:
                            chosen_fn_str = str(chosen_fn_str).replace('/', '_')
                        except Exception:
                            chosen_fn_str = 'default_callable'
                        step_loss_dict[f"val/step{step_index}/{layer_key}/P{chosen_fn_str}"] = loss_value
                        print(f"Calculated loss for {layer_key} (fn={chosen_fn_str}): {loss_value.item()}")
                        if layer_key not in self.layer_losses:
                            self.layer_losses[layer_key] = []
                        self.layer_losses[layer_key].append(loss_value.item())
                        
                if viz_enabled:
                    viz_query_idx, viz_key_idx = self._extract_viz_view_indices(F)
                    pred_attn_logit_viz = self._slice_attention_map(pred_attn_logit, viz_query_idx, viz_key_idx, F)
                    gt_attn_logit_viz = self._slice_attention_map(gt_attn_logit, viz_query_idx, viz_key_idx, F)

                    # Process through configured logit heads so visualization respects softmax_temp / head processing
                    # per-pair viz heads
                    if isinstance(pair, dict) and pair.get('unet_logit_head', None) is not None:
                        unet_head_name = pair['unet_logit_head'].lower()
                        unet_head_kwargs = pair.get('unet_logit_head_kwargs', {})
                        unet_head = LOGIT_HEAD_CLS[unet_head_name](**unet_head_kwargs)
                    else:
                        raise ValueError(f"Pair must provide 'unet_logit_head' for visualization: {pair}")
                    if isinstance(pair, dict) and pair.get('vggt_logit_head', None) is not None:
                        vggt_head_name = pair['vggt_logit_head'].lower()
                        vggt_head_kwargs = pair.get('vggt_logit_head_kwargs', {})
                        vggt_head = LOGIT_HEAD_CLS[vggt_head_name](**vggt_head_kwargs)
                    else:
                        raise ValueError(f"Pair must provide 'vggt_logit_head' for visualization: {pair}")

                    pred_processed_viz = unet_head(pred_attn_logit_viz)
                    gt_processed_viz = vggt_head(gt_attn_logit_viz)

                    print(f"Calling _maybe_save_attn_overlay for step {step_index}, layer {layer_key} (using processed logits)")
                    # derive viz softmax settings from pair (fallback to loss settings)
                    # use global softmax mode/num_k from visualize_config only
                    viz_mode = self.visualize_config.get('softmax_mode', 'global')
                    viz_num_k = int(self.visualize_config.get('loss_num_key_views', 1))
                    self._maybe_save_attn_overlay(
                        step_index=step_index,
                        layer_key=layer_key,
                        pred_logits=pred_processed_viz,
                        gt_logits=gt_processed_viz,
                        F=F,
                        query_idx_list=viz_query_idx,
                        key_idx_list=viz_key_idx,
                        viz_mode=viz_mode,
                        viz_num_k=viz_num_k,
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
                print(f"Step {step_index} avg loss: {float(step_avg_loss):.6f}")

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
                structured_losses['layer_summary'][layer_key] = {
                    'mean': sum(losses) / len(losses),
                    'min': min(losses),
                    'max': max(losses),
                    'std': (sum((x - sum(losses)/len(losses))**2 for x in losses) / len(losses))**0.5 if len(losses) > 1 else 0.0,
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
