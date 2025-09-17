import math
import torch
import torch.nn as nn
import numpy as np
'''
x = [B, Head, Q, K]
'''
def build_mlp(in_dim, out_dim, mlp_ratio=4.0, mlp_depth=1):
    mlp = []
    if mlp_depth <= 0:
        mlp += [nn.Linear(in_dim, out_dim)]
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
            x = x.reshape(*x.shape[:-1], K, HW)
        x = x / self.softmax_temp
        return x.softmax(dim=-1)

class Softmax_HeadMean(Softmax):
    def forward(self, x, **kwargs):
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        return x.mean(dim=1, keepdim=True)


class Softmax_HeadMlp(Softmax):
    def __init__(self, 
                 in_head_num = 24, out_head_num = 1,
                 mlp_ratio = 4.0, mlp_depth = 1,
                 softmax_temp=1.0, learnable_temp = False, 
                 final_activation = None,
        **kwargs):
        super(Softmax_HeadMlp, self).__init__(softmax_temp, learnable_temp, **kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x, num_view= None, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // num_view
            x = x.reshape(*x.shape[:-1], K, HW)
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        x = self.final_activation(x)
        return x

class HeadMlp_Softmax(Softmax):
    def __init__(self, 
                 in_head_num = 24, out_head_num = 1,
                 mlp_ratio = 4.0, mlp_depth = 1,
                 softmax_temp=1.0, learnable_temp = False, **kwargs):
        super(HeadMlp_Softmax, self).__init__(softmax_temp, learnable_temp, **kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)

    def forward(self, x, num_view=None,**kwargs):
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        if self.per_view:
            K = x.shape[-1]
            HW = K // num_view
            x = x.reshape(*x.shape[:-1], K, HW)
        x = x / self.softmax_temp
        return x.softmax(dim=-1)

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

def softargmax2d(logit, beta=100, num_view=1):
    ''' logit: ... num_view*hw
    '''
    h = int((logit.shape[-1] / num_view) ** 0.5)
    w = num_view * h
    assert h * w == logit.shape[-1], f"input size does not match, {h} * {w} != {logit.shape[-1]}"

    prob = nn.functional.softmax(beta * logit, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

    result_r = torch.sum((h - 1) * prob * indices_r, dim=-1) # (h-1) * n/(h-1) = n
    result_c = torch.sum((w - 1) * prob * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result

class SoftArgmax(nn.Module):
    def __init__(self,
        beta,
        learnable_beta = False,
        per_view = False,
        entropy_temp = 1.0,
        compute_entropy = True,
    ):
        self.beta = nn.Parameter(torch.tensor(beta)) if learnable_beta else beta
        self.per_view=per_view
        self.entropy_temp = entropy_temp
        self.compute_entropy = compute_entropy
        self.entropy_val =  None

    def forward(self, x, num_view, **kwargs):
        ''' x : ... K(VHW)
        '''
        logit = x
        num_k = num_view

        if self.per_view:
            K = logit.shape[-1]
            if K % num_k != 0:
                raise ValueError(f"num_key_views={num_k} is not a valid divisor of K={K}")
            other = K // num_k
            kp = int(round((other) ** 0.5))
            if kp * kp != other:
                raise ValueError(f"Given num_key_views={num_k} does not yield a square kp for K={K}")

            argmax_idx_list = []
            logit = logit.view(*logit.shape[:-1], num_k, kp**2)

            for i in range(num_k):
                l = logit[..., i, :]
                argmax_idx = softargmax2d(l, beta=self.beta, num_view=1) # [..., hw, 2]
                argmax_idx_list.append(argmax_idx)
            argmax_indices = torch.stack(argmax_idx_list, dim=-2) # [..., hw, view, 2]
            
            if self.compute_entropy:
                ent = []
                for i in range(num_k):
                    l = logit[..., i, :]
                    prob = torch.nn.functional.softmax(l / self.entropy_temp, dim=-1)
                    ent += [- (prob * (prob + eps).log()).sum(dim=-1).mean()]
                self.entropy_val = sum(ent) / len(ent)
            return argmax_indices
        else:
            argmax_indices = softargmax2d(logit, beta=self.beta, num_view=num_k) # [..., hw, 2]
            argmax_indices = argmax_indices.unsqueeze(1).unsqueeze(3) # [..., 1, hw, 1, 2]
            eps = 1e-8
            if self.compute_entropy:
                prob = torch.nn.functional.softmax(logit / self.entropy_temp, dim=-1)
                self.entropy_val = - (prob * (prob + eps).log()).sum(dim=-1).mean()
            return argmax_indices


class HeadMlp_SoftArgmax(SoftArgmax):
    def __init__(self,
        # softargmax
        beta,
        learnable_beta = False,
        per_view = False,
        entropy_temp = 1.0,
        compute_entropy = True,
        # mlp
        in_head_num = 24, out_head_num = 1,
        mlp_ratio = 4.0, mlp_depth = 1, final_activation = None,
        **kwargs
    ):
        super().__init__(beta, learnable_beta, per_view, entropy_temp, compute_entropy)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)

    def forward(self, x, num_view=None, **kwargs):
        ''' x: B Head Q K
            return: B Head Q 1 2 or  B Head Q key_num_view 2 (per view)
        '''
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        return super().forward(x, num_view, **kwargs)

class HeadMean_SoftArgmax(SoftArgmax):
    def __init__(self,
        # softargmax
        beta,
        learnable_beta = False,
        per_view = False,
        entropy_temp = 1.0,
        compute_entropy = True,
        **kwargs
    ):
      super().__init__(beta, learnable_beta, per_view, entropy_temp, compute_entropy)  
    
    def forward(self, x, num_view=None, **kwargs):
        ''' x : B Head Q K(num_view*hw) 
            return: B Head(1) Q 1 2 or  B Head(1) Q key_num_view 2 (per view)
        '''
        x = x.mean(dim=1, keepdim=True)
        return super().forward(x, num_view, **kwargs)


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

