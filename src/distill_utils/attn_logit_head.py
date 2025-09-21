import math
import torch
import torch.nn as nn
import numpy as np
'''
x = [B, Head, Q, K]
'''
def build_mlp(in_dim, out_dim, mlp_ratio=4.0, mlp_depth=1, mid_dim = None):
    
    mlp = []
    if mlp_depth <= 0:
        mlp += [nn.Linear(in_dim, out_dim)]
    else:
        if mid_dim is not None:
            assert mlp_ratio is None
        else:
            mid_dim = int(in_dim * mlp_ratio)
        mlp += [nn.Linear(in_dim, mid_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            mlp += [nn.Linear(mid_dim, mid_dim), nn.GELU()]
        mlp += [nn.Linear(mid_dim, out_dim)]
    return nn.Sequential(*mlp)

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x, **kwargs):
        return x
    
class Softmax(nn.Module):
    def __init__(self, 
                 per_view = False,
                 softmax_temp=1.0, learnable_temp = False, **kwargs):
        super(Softmax, self).__init__()
        self.per_view = per_view
        self.softmax_temp = softmax_temp
        if learnable_temp:
            self.softmax_temp = nn.Parameter(torch.tensor(softmax_temp))

    def forward(self, x, num_view=None, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // num_view
            x = x.reshape(*x.shape[:-1], num_view, HW)
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        if self.per_view:
            x = x.reshape(*x.shape[:-2], num_view*HW)
        return x
 
class Softmax_HeadMean(Softmax):
    def forward(self, x, num_view=None, **kwargs):
        ''' x : B Head Q K(num_view HW)
        '''
        x = super().forward(x, num_view, **kwargs)
        return x.mean(dim=1, keepdim=True)

class Softmax_HeadMlp(Softmax):
    def __init__(self, 
            in_head_num = 24, out_head_num = 1,
            mlp_ratio = 4.0, mlp_depth = 1,
            final_activation = None,
            **kwargs
        ):
        super(Softmax_HeadMlp, self).__init__(**kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x, num_view= None, **kwargs):
        x = super().forward(x, num_view, **kwargs)
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        x = self.final_activation(x)
        return x

class HeadMlp_Softmax(Softmax):
    def __init__(self, 
            in_head_num = 24, out_head_num = 1,
            mlp_ratio = 4.0, mlp_depth = 1,
            **kwargs
        ):
        super(HeadMlp_Softmax, self).__init__(**kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)

    def forward(self, x, num_view=None,**kwargs):
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        x = super().forward(x, num_view, **kwargs)
        return x

# class HeadMlp(nn.Module):
#     def __init__(self, 
#                  in_head_num = 24, out_head_num = 1,
#                  mlp_ratio = 4.0, mlp_depth = 1, final_activation = None,
#                  **kwargs):
#         super(HeadMlp, self).__init__(**kwargs)
#         self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)
#         if final_activation == "sigmoid":
#             self.final_activation = nn.Sigmoid()
#         else:
#             self.final_activation = nn.Identity()

#     def forward(self, x):
#         x = x.permute(0,2,3,1) # B Q K Head
#         x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
#         x = self.final_activation(x)
#         return x

# def softargmax2d(logit, beta=100, num_view=1):
#     ''' https://github.com/david-wb/softargmax
#         - logit: ... num_view*hw
#     '''
#     h = int((logit.shape[-1] / num_view) ** 0.5)
#     w = num_view * h # width 로 쌓으면(h V*w) 기존 VHW와 dim 순서 다름 issue? 
#     assert h * w == logit.shape[-1], f"input size does not match, {h} * {w} != {logit.shape[-1]}"

#     prob = nn.functional.softmax(beta * logit, dim=-1)

#     indices_c, indices_r = np.meshgrid(
#         np.linspace(0, 1, w),
#         np.linspace(0, 1, h),
#         indexing='xy'
#     )

#     indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
#     indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

#     result_r = torch.sum((h - 1) * prob * indices_r, dim=-1) # (h-1) * n/(h-1) = n
#     result_c = torch.sum((w - 1) * prob * indices_c, dim=-1)

#     result = torch.stack([result_r, result_c], dim=-1)

#     return result

def per_view_softargmax2d(prob, num_view=1):
    ''' leffa: https://arxiv.org/pdf/2412.08486v2
        args
            - prob: [..., vhw]
        return
            - indices: [..., v, 2]
    '''
    K = prob.shape[-1]
    hw = K // num_view
    prob = prob.reshape(*prob.shape[:-1], num_view, hw)
    h = w = int(hw**0.5)

    indices_c, indices_r = np.meshgrid(
        np.linspace(-1, 1, w),
        np.linspace(-1, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(prob.device, dtype=prob.dtype)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(prob.device, dtype=prob.dtype)

    result_r = torch.sum(prob * indices_r, dim=-1)
    result_c = torch.sum(prob * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result

# class Argmax(nn.Module):
#     def __init__(self,
#         per_view = True,
#         compute_entropy = True,
#         entropy_temp = 1.0,
#     ):
#         self.per_view=per_view
#         assert per_view == True
#         self.compute_entropy = compute_entropy
#         self.entropy_temp = entropy_temp
#         self.entropy_val =  None

#     def forward(self, x, num_view, **kwargs):
#         ''' x : ... K(VHW)
#         '''
#         prob = x
#         if self.per_view:
#             K = prob.shape[-1]
#             hw = K // num_view
#             prob = prob.reshape(prob.shape[:-1], num_view, hw)
#             argmax_indices = per_view_softargmax2d(prob) # [... V 2]
#             if self.compute_entropy:
#                 eps = 1e-8
#                 self.entropy_val = - (prob * (prob + eps).log()).sum(dim=-1) # [... V]
#             return argmax_indices
#         else:
#             raise NotImplementedError

class HeadMean_SoftArgmax(Softmax_HeadMean):
    ''' Softmax -> HeadMean -> Prob * idx
    '''
    def __init__(self, 
        compute_entropy= False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.per_view == True, "SoftArgmax requires per_view computation"
        self.compute_entropy = compute_entropy
        self.entropy_val =  None
    def forward(self, x, num_view=None, **kwargs):
        assert num_view is not None, "softargmax requires num_view"
        x = super().forward(x, num_view) # per_view prob: B 1 Q K(VHW)
        if self.compute_entropy:
            K = x.shape[-1]
            HW = K // num_view
            per_view_prob = x.reshape(*x.shape[-1], num_view, HW)
            eps = 1e-8
            self.entropy_val = - (per_view_prob * (per_view_prob + eps).log()).sum(dim=-1) # B 1 Q V
        x = per_view_softargmax2d(x, num_view) # B 1 Q V 2
        return x.flatten(-2,-1) # B 1 Q V*2


class HeadMlp_SoftArgmax(HeadMlp_Softmax):
    def __init__(self,
        compute_entropy= False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.per_view == True, "SoftArgmax requires per_view computation"
        self.compute_entropy = compute_entropy
        self.entropy_val =  None

    def forward(self, x, num_view=None, **kwargs):
        ''' x: B Head Q K
            return: B Head Q 1 2 or  B Head Q key_num_view 2 (per view)
        '''
        assert num_view is not None, "softargmax must provide num_view"
        x = super().forward(x, num_view, **kwargs) # per_view prob: B 1 Q K(VHW)
        if self.compute_entropy:
            K = x.shape[-1]
            HW = K // num_view
            per_view_prob = x.reshape(*x.shape[-1], num_view, HW)
            eps = 1e-8
            self.entropy_val = - (per_view_prob * (per_view_prob + eps).log()).sum(dim=-1) # B 1 Q V
        x = per_view_softargmax2d(x, num_view) # B 1 Q V 2
        return x.flatten(-2,-1) # B 1 Q V*2



def cycle_consistency_checker(costmap, pixel_threshold=None):
    ''' costmap : [B, HW, (V-1)*HW]
        cross cost/attn only
    '''
    # Get dimensions from the input tensor
    B, HW, _ = costmap.shape
    device = costmap.device

    H = W = int(math.sqrt(HW))

    # Step 2: Find the best initial matches (already vectorized)
    max_idx = torch.argmax(costmap, dim=-1)  # (B, HW)
    transpose_costmap = costmap.transpose(1, 2)  # (B, 2HW, HW)
    b_idx = torch.arange(B, device=device)[:, None] # (B, 1)
    reverse_map = transpose_costmap[b_idx, max_idx] # (B, HW, HW)
    _, final_indices = torch.max(reverse_map, dim=-1) 
    # Create a tensor representing the original indices (0, 1, 2, ..., HW-1) for each batch item
    original_indices = torch.arange(HW, device=device).expand(B, -1) # (B, HW)

    # Convert 1D indices to 2D coordinates for both original and matched points
    x1 = original_indices // W
    y1 = original_indices % H
    x2 = final_indices // W
    y2 = final_indices % H

    if pixel_threshold is None:
        pixel_threshold = round(H / 10.0)

    # Calculate Euclidean distance and check if it's within the threshold
    distance = torch.sqrt(((x1 - x2)**2 + (y1 - y2)**2).float())
    is_close = (distance < pixel_threshold).float() # .float() converts boolean (True/False) to (1.0/0.0)

    # Step 6: Reshape final output tensor
    final_distance_tensor = is_close.view(B, HW, 1)

    return final_distance_tensor


from einops import rearrange
class OneHot(nn.Module):
    ''' non-differentiable, just for GT'''
    def forward(self, x, num_view=None, consistency_pixel_threshold=1.5 ,**kwargs):
        ''' x: B Head Q(f1HW) K(f2HW)
        '''
        f2 = num_view
        B, Head, Q, K = x.shape
        HW = K // f2
        f1 = Q // HW
        x = rearrange(x, "B Head (f1 HW1) (f2 HW2) -> (B Head f1 f2) HW1 HW2", B=B, Head=Head, f1=f1,f2=f2,HW1=HW, HW2=HW)

        max_idx = torch.argmax(x, dim=-1)  # (..., q(HW))
        mask_per_view = cycle_consistency_checker(x, consistency_pixel_threshold ) # (B Head F1 F2) HW1 1


        

        # mask_per_view = cycle_consistency_checker()

        # def get_consistency_mask(logit):
        #     # assert config.distill_config.distill_query == "target", "consistency check only support distill_query to target"
        #     B, Head, F1HW, F2HW = logit.shape  
        #     assert F1HW == F2HW, "first process full(square) consistency mask"
        #     # assert Head == 1, "costmap should have only one head, while consistency checking?"
        #     HW = F1HW // F
        #     logit = einops.rearrange(logit, 'B Head (F1 HW1) (F2 HW2) -> (B Head F1 F2) HW1 HW2', B=B,Head=Head, F1=F, F2=F, HW1=HW, HW2=HW)
        #     mask_per_view = cycle_consistency_checker(logit, **config.distill_config.get("consistency_check_cfg", {}) ) # (B Head F1 F2) HW
        #     mask_per_view = einops.rearrange(mask_per_view, '(B Head F1 F2) HW 1 -> B Head (F1 HW) (F2 1)', B=B, Head=Head, F1=F, F2=F, HW=HW)
        #     # mask = mask.any(dim=-1) # B Head Q(F1HW) 
        #     return mask_per_view # B Head Q(F1HW) F2


        pass


LOGIT_HEAD_CLS = {
    "identity": Identity,
    "softmax": Softmax,
    "softmax_headmean": Softmax_HeadMean,
    "softmax_headmlp": Softmax_HeadMlp,
    "headmlp_softmax": HeadMlp_Softmax,
    # "headmlp": HeadMlp,
    "headmlp_softargmax": HeadMlp_SoftArgmax,
    "headmean_softargmax": HeadMean_SoftArgmax,
}




# keep for debugging
# class HeadMlp_Soft_argmax(nn.Module):
#     def __init__(self, 
#                  in_head_num = 24, out_head_num = 1,
#                  mlp_ratio = 4.0, mlp_depth = 1,
#                  softmax_temp=1.0, learnable_temp = False, beta = 100, per_view = True, **kwargs):
#         super(HeadMlp_Soft_argmax, self).__init__(**kwargs)
#         self.softmax_temp = softmax_temp
#         self.beta = beta
#         self.per_view = per_view
#         self.entropy = None
#         self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)
#         if learnable_temp:
#             self.beta = nn.Parameter(torch.tensor(self.beta))

    
#     def forward(self, x):
#         x = x.permute(0,2,3,1) # B Q K Head
#         logit = self.mlp(x).permute(0,3,1,2).squeeze(1) # B Q K
#         num_k = int(x.shape[-1] // x.shape[-2])

#         if self.per_view:
#             K = logit.shape[-1]
#             if K % num_k != 0:
#                 raise ValueError(f"num_key_views={num_k} is not a valid divisor of K={K}")
#             other = K // num_k
#             kp = int(round((other) ** 0.5))
#             if kp * kp != other:
#                 raise ValueError(f"Given num_key_views={num_k} does not yield a square kp for K={K}")

#             argmax_idx_list = []
#             logit = logit.view(*logit.shape[:-1], num_k, kp, kp)

#             for i in range(num_k):
#                 l = logit[..., i, :, :]
#                 argmax_idx = softargmax2d(l.reshape(*l.shape[:-2], kp ** 2), beta=self.beta, num_view=1) # [batch, hw, 2]
#                 argmax_idx_list.append(argmax_idx)

#             argmax_indices = torch.stack(argmax_idx_list, dim=2) # [batch, hw, view, 2]
#             argmax_indices = argmax_indices.unsqueeze(1) # [batch, 1, hw, view, 2]
#             return argmax_indices
#         else:
#             argmax_indices = softargmax2d(logit, beta=self.betap, num_view=num_k) # [batch, hw, 2]
#             argmax_indices = argmax_indices.unsqueeze(1).unsqueeze(3) # [batch, 1, hw, 1, 2]
#             eps = 1e-8
#             prob = torch.nn.functional.softmax(logit / self.softmax_temp, dim=-1)
#             self.entropy = - (prob * (prob + eps).log()).sum(dim=-1).mean()
#             return argmax_indices

# class HeadMean_Soft_argmax(nn.Module):
#     def __init__(self, 
#                  in_head_num = 24, out_head_num = 1,
#                  mlp_ratio = 4.0, mlp_depth = 1,
#                  softmax_temp=1.0, learnable_temp = False, beta = 100, per_view = True, **kwargs):
#         super(HeadMean_Soft_argmax, self).__init__(**kwargs)
#         self.softmax_temp = softmax_temp
#         self.beta = beta
#         self.per_view = per_view
#         self.entropy_val = None
#         if learnable_temp:
#             self.beta = nn.Parameter(torch.tensor(self.beta))
    
#     def forward(self, x):
#         # x: B Head Q K
#         logit = x.mean(dim=1, keepdim=False) # B Q K
#         num_key_views = int(x.shape[-1] // x.shape[-2])

#         if self.per_view:
#             K = logit.shape[-1]
#             num_k = int(num_key_views)
#             if K % num_k != 0:
#                 raise ValueError(f"num_key_views={num_k} is not a valid divisor of K={K}")
#             other = K // num_k
#             kp = int(round((other) ** 0.5))
#             if kp * kp != other:
#                 raise ValueError(f"Given num_key_views={num_k} does not yield a square kp for K={K}")

#             argmax_idx_list = []
#             logit = logit.view(*logit.shape[:-1], num_k, kp, kp)

#             for i in range(num_k):
#                 l = logit[..., i, :, :]
#                 argmax_idx = softargmax2d(l.reshape(*l.shape[:-2], kp ** 2), beta=self.beta, num_view=1) # [batch, hw, 2]
#                 argmax_idx_list.append(argmax_idx)

#             argmax_indices = torch.stack(argmax_idx_list, dim=2) # [batch, hw, view, 2]
#             argmax_indices = argmax_indices.unsqueeze(1) # [batch, 1, hw, view, 2]
#             return argmax_indices
#         else:
#             argmax_indices = softargmax2d(logit, beta=self.beta, num_view=num_k) # [batch, hw, 2]
#             argmax_indices = argmax_indices.unsqueeze(1).unsqueeze(3) # [batch, 1, hw, 1, 2]
#             eps = 1e-8
#             prob = torch.nn.functional.softmax(logit / self.softmax_temp, dim=-1)
#             self.entropy_val = - (prob * (prob + eps).log()).sum(dim=-1).mean()
#             return argmax_indices # [batch, 1, hw, 1, 2]

