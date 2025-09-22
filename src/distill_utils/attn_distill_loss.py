import torch

# loss function
def cross_entropy(prob, prob_gt):
    """Cross entropy loss for attention probabilities."""
    eps = 1e-8
    return - (prob_gt * (prob + eps).log()).sum(dim=-1).mean()

def kl_divergence(prob, prob_gt):
    """Kullback-Leibler divergence loss for attention probabilities."""
    return (prob_gt * (prob_gt.log() - prob.log())).sum(dim=-1).mean()

# def soft_argmax(prob, prob_gt, alpha=0.5):
#     """Soft argmax loss for attention probabilities.
#     Additional regularization to match entropy between two distributions"""
#     idxs = torch.arange(prob.shape[-1], device=prob.device).float()
#     argmax = (prob * idxs).sum(dim=-1)
#     argmax_gt = (prob_gt * idxs).sum(dim=-1)
#     argmax_loss = torch.nn.functional.l1_loss(argmax, argmax_gt)
#     eps = 1e-8
#     entropy_loss = ((prob_gt * (prob_gt + eps).log()).sum(dim=-1) - (prob * (prob + eps).log()).sum(dim=-1)).mean()
#     return argmax_loss + alpha * entropy_loss

ATTN_LOSS_FN = {
    "l1": torch.nn.functional.l1_loss,
    "mse": torch.nn.functional.mse_loss,
    "cross_entropy": cross_entropy,
    "kl_divergence": kl_divergence,
    # "soft_argmax": soft_argmax,
}

import math
def cosine_loss_weight_scheduler(step: int, start_w: float, end_w: float, end_step: int) -> float:
    """
    Cosine interpolation from start_w -> end_w over [0, end_step].
    For step >= end_step returns end_w.
    """
    if end_step is None or end_step <= 0:
        return end_w
    if step >= end_step:
        return end_w
    t = step / float(end_step)  # in [0,1)
    # classic cosine: at t=0 -> start_w, at t=1 -> end_w
    return end_w + 0.5 * (start_w - end_w) * (1.0 + math.cos(math.pi * t))