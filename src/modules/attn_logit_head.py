
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