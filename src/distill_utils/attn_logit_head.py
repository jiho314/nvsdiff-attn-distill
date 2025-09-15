import math
import torch
import torch.nn as nn
'''
x = [B, Head, Q, K]
'''

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
class Softmax(nn.Module):
    def __init__(self, softmax_temp=1.0, **kwargs):
        super(Softmax, self).__init__()
        self.softmax_temp = softmax_temp

    def forward(self, x):
        x = x / self.softmax_temp
        return x.softmax(dim=-1)

class Softmax_HeadMean(nn.Module):
    def __init__(self, softmax_temp=1.0, **kwargs):
        super(Softmax_HeadMean, self).__init__()
        self.softmax_temp = softmax_temp

    def forward(self, x):
        x = x / self.softmax_temp
        return x.softmax(dim=-1).mean(dim=1)


class Softmax_HeadMlp(nn.Module):
    def __init__(self, in_head_num = 24, 
                 out_head_num = 16, depth = 1,
                 softmax_temp=1.0, **kwargs):
        super(Softmax_HeadMlp, self).__init__()
        self.softmax_temp = softmax_temp
        mlp = []
        for _ in range(depth - 1):
            mlp.append(nn.Linear(in_head_num, in_head_num))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(in_head_num, out_head_num))
        self.mlp = nn.Sequential(*mlp)

    # TODO: Pass query and key (or q @ k) to MLP, then compute attention probs
    def forward(self, x):
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        x = self.mlp(x)
        return x

# class Ref_Softmax_HeadMean(nn.Module):
#     def __init__(self, 
#         softmax_temp=1.0,
#         # ref_ids = [0,1] 
#         **kwargs
#     ):
#         super(Ref_Softmax_HeadMean, self).__init__()
#         self.softmax_temp = softmax_temp
#         # self.ref_ids = ref_ids

#     def forward(self, x):
#         B, head, HW, VHW = x.shape
#         # V = VHW // HW
#         x = x[:,:,:, HW:] # exclude self-attn logit        
#         x = x / self.softmax_temp
#         return x.softmax(dim=-1).mean(dim=1)

LOGIT_HEAD_CLS = {
    "identity": Identity,
    "softmax": Softmax,
    "softmax_headmean": Softmax_HeadMean,
    "softmax_headmlp": Softmax_HeadMlp,
    # "ref_softmax_headmean": Ref_Softmax_HeadMean,
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