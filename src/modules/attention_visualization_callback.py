import torch
import math
import einops
from typing import Dict, List, Tuple, Optional, Any
from my_diffusers.callbacks import PipelineCallback
from src.distill_utils.attn_processor_cache import pop_cached_attn, clear_attn_cache


import os
import math
import numpy as np
import torch
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
        
        if callable(loss_fn):
            # 이미 함수인 경우 (validate.py에서 전달된 경우)
            self.loss_fn = loss_fn
        elif isinstance(loss_fn, str):
            # 문자열인 경우 함수로 변환
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
            self.loss_fn = ATTN_LOSS_FN[loss_fn.lower()]
        else:
            raise ValueError(f"Unsupported loss_fn type: {type(loss_fn)}")
    
    def _prepare_vggt_cache(self):
        """VGGT attention cache를 미리 계산"""
        with torch.no_grad():
            image = self.batch['image'].to(self.device)  # [B,F,3,H,W]
            vggt_pred = self.vggt_model(image)
            self.batch['vggt_attn_cache'] = vggt_pred['attn_cache']
    
    def _resize_token(self, tok: torch.Tensor, target_size: int, F: int) -> torch.Tensor:
        """토큰을 target_size로 리사이즈"""
        B, Head, FHW, C = tok.shape
        HW = FHW // F
        H = W = int(math.sqrt(HW))
        tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', 
                              B=B, Head=Head, F=F, H=H, W=W, C=C)
        tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
        tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C', 
                              B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)
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
    
    def _extract_view_indices(self, F: int) -> Tuple[List[int], List[int]]:
        """query와 key의 view index 추출"""
        # Query index 추출
        if self.visualize_config['query'] == "target":
            query_idx = list(range(self.cond_num, F))
        elif self.visualize_config['query'] == "all":
            query_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['query'] {self.visualize_config['query']} not implemented")
        
        # Key index 추출
        if self.visualize_config['key'] == "reference":
            key_idx = list(range(0, self.cond_num))
        elif self.visualize_config['key'] == "all":
            key_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['key'] {self.visualize_config['key']} not implemented")
        
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
        
        # attention_tile에서 최대값 위치 찾기
        max_idx = np.argmax(attention_tile.flatten())
        max_y, max_x = np.unravel_index(max_idx, attention_tile.shape)
        
        # tile 좌표를 이미지 좌표로 변환
        sx = w / float(tile_side)
        sy = h / float(tile_side)
        cx = int((max_x + 0.5) * sx)
        cy = int((max_y + 0.5) * sy)
        
        # 점 크기 계산
        r_in = max(3, min(w, h) // 80)
        
        # 빨간 점 그리기 (흰색 테두리 + 빨간색 내부, 크기 줄임)
        r_small = max(2, r_in // 2)  # 크기를 절반으로 줄임
        d.ellipse((cx - r_small*2, cy - r_small*2, cx + r_small*2, cy + r_small*2), fill=(255, 255, 255))
        d.ellipse((cx - r_small, cy - r_small, cx + r_small, cy + r_small), fill=(255, 0, 0))
        
        return img

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

        # softmax over K and mean over heads → [B,Q,K]
        pred_prob = pred.softmax(dim=-1).mean(dim=1)[0]  # [Q,K]
        gt_prob = gt.softmax(dim=-1).mean(dim=1)[0]      # [Q,K]

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
                color_img = _PILImage.fromarray(color).resize((Wc, Hc), _PILImage.BILINEAR)
                color_np = np.array(color_img)
                alpha = float(self.visualize_config.get('viz_alpha', 0.6))
                blended = (bg.astype(np.float32) * (1.0 - alpha) + color_np.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
                
                # 각 view별로 가장 확률이 높은 부분에 파란 점 추가
                blended_pil = _PILImage.fromarray(blended)
                blended_pil = self._draw_max_attention_point(blended_pil, tile, k_side)
                blended = np.array(blended_pil)
                
                outs.append(blended)
            row = np.concatenate(outs, axis=1)
            return _PILImage.fromarray(row)

        row_pred = build_row(pred_tiles, bg_images)
        row_gt = build_row(gt_tiles, bg_images)

        # build Q panel (with query dot)
        q_side_px = min(q_bg_np.shape[0], q_bg_np.shape[1])
        q_panel = _PILImage.fromarray(q_bg_np).resize((q_side_px, q_side_px), _PILImage.BILINEAR)
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
            key = f"attn/{seq}/{layer_key}"
            if not hasattr(self, 'attention_images'):
                self.attention_images = {}
            
            self.attention_images[key] = {
                'image': canvas,
                'caption': f"{seq} | {layer_key} | step {int(step_index)}"
            }

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
        
        viz_steps = self.visualize_config.get('viz_steps', [])
        
        if step_index not in viz_steps:
            print(f"Skipping attention visualization callback for step {step_index}")
            clear_attn_cache(pipeline.unet)
            return callback_kwargs
        
        print(f"AttentionVisualizationCallback.__call__ invoked: step_index={step_index}, timestep={timestep}")
        print(f"do_attn_visualize: {self.do_attn_visualize}")
        
        
        if not self.do_attn_visualize:
            print("do_attn_visualize is False, returning early")
            return callback_kwargs
        
        try:
            # UNet attention cache 가져오기
            unet_attn_cache = pop_cached_attn(pipeline.unet)
            print(f"UNet attention cache keys: {list(unet_attn_cache.keys()) if unet_attn_cache else 'None'}")
            
            if not unet_attn_cache:
                # attention cache가 비어있으면 스킵
                print("UNet attention cache is empty, skipping")
                return callback_kwargs
            
            # 이미지 shape 정보
            image = self.batch['image']
            B, F, _, H, W = image.shape
            
            step_loss_dict = {}
            
            # 각 layer pair에 대해 loss 계산
            visualize_pairs = self.visualize_config['pairs']
            print(f"Processing {len(visualize_pairs)} layer pairs: {visualize_pairs}")
            
            for unet_layer, vggt_layer in visualize_pairs:
                if str(unet_layer) not in unet_attn_cache:
                    continue
                if str(vggt_layer) not in self.batch['vggt_attn_cache']:
                    continue
                
                # GT (VGGT) query, key 가져오기
                gt_query = self.batch['vggt_attn_cache'][str(vggt_layer)]['query'].detach()  # B Head VHW C
                gt_key = self.batch['vggt_attn_cache'][str(vggt_layer)]['key'].detach()      # B Head VHW C
                
                # Pred (UNet) attention logit 가져오기
                pred_attn_logit = unet_attn_cache[str(unet_layer)]  # B Head Q(FHW) K(FHW)
                
                # dtype과 device 일치 확인
                target_dtype = pred_attn_logit.dtype
                target_device = pred_attn_logit.device
                gt_query = gt_query.to(dtype=target_dtype, device=target_device)
                gt_key = gt_key.to(dtype=target_dtype, device=target_device)
                
                Q, K = pred_attn_logit.shape[-2], pred_attn_logit.shape[-1]
                if Q != K:
                    print(f"Warning: pred attn should have same Q,K dim, but got {pred_attn_logit.shape}")
                    continue
                
                # F truncate 로직: Q가 F로 나누어떨어지지 않으면 자르기; Up block concat issue 
                if Q % F != 0 or K % F != 0:
                    Q_truncated = (Q // (F+1)) * F
                    K_truncated = (K // (F+1)) * F
                    pred_attn_logit = pred_attn_logit[:, :, :Q_truncated, :K_truncated]
                    Q, K = Q_truncated, K_truncated
                    print(f"Truncated attention logit: Q {Q_truncated}, K {K_truncated} (F={F})")
                
                # 1) 토큰 크기 맞추기: gt -> pred
                pred_query_size = int(math.sqrt(Q // F))
                pred_key_size = int(math.sqrt(K // F))
                gt_query_resized = self._resize_token(gt_query, pred_query_size, F)
                gt_key_resized = self._resize_token(gt_key, pred_key_size, F)
                
                # 2) View index 추출
                query_idx, key_idx = self._extract_view_indices(F)
                
                # 3) Attention map 슬라이싱
                pred_attn_logit_sliced = self._slice_attention_map(pred_attn_logit, query_idx, key_idx, F)
                
                # GT attention logit 계산
                gt_attn_logit = gt_query_resized @ gt_key_resized.transpose(-1, -2)
                gt_attn_logit_sliced = self._slice_attention_map(gt_attn_logit, query_idx, key_idx, F)
                
                # 4) Logit head 통과시켜 최종 loss 계산
                pred_processed = pipeline.unet.unet_logit_head(pred_attn_logit_sliced)
                gt_processed = pipeline.unet.vggt_logit_head(gt_attn_logit_sliced)
                
                loss_value = self.loss_fn(pred_processed, gt_processed)

                layer_key = f"unet{unet_layer}_vggt{vggt_layer}"
                step_loss_dict[f"step{step_index}/{layer_key}"] = loss_value
                
                print(f"Calculated loss for {layer_key}: {loss_value.item()}")
                
                # layer별 누적 loss 저장
                if layer_key not in self.layer_losses:
                    self.layer_losses[layer_key] = []
                self.layer_losses[layer_key].append(loss_value.item())

                # Optional visualization save
                print(f"Calling _maybe_save_attn_overlay for step {step_index}, layer {layer_key}")
                self._maybe_save_attn_overlay(
                    step_index=step_index,
                    layer_key=layer_key,
                    pred_logits=pred_attn_logit_sliced,
                    gt_logits=gt_attn_logit_sliced,
                    F=F,
                    query_idx_list=query_idx,
                    key_idx_list=key_idx,
                )
            
            # Step별 데이터 저장
            if step_loss_dict:
                step_avg_loss = sum(step_loss_dict.values()) / len(step_loss_dict)
                step_loss_dict[f"step{step_index}/avg_loss"] = step_avg_loss
                
                # step별 layer별 loss 저장
                self.step_layer_losses[step_index] = {}
                for key, loss_value in step_loss_dict.items():
                    if key.startswith(f"step{step_index}/") and key != f"step{step_index}/avg_loss":
                        # key에서 step 부분을 제거하여 layer_key만 추출
                        layer_key = key.replace(f"step{step_index}/", "")
                        self.step_layer_losses[step_index][layer_key] = loss_value.item()
                
                self.step_losses.append(step_avg_loss.item())
                print(f"Step {step_index} avg loss: {step_avg_loss.item():.6f}")
            
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
        final_dict["train/visualize/final_avg_loss"] = self.get_final_loss()
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
