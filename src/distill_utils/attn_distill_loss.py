import torch

# loss function
def cross_entropy(prob, prob_gt):
    """Cross entropy loss for attention probabilities."""
    eps = 1e-8
    return - (prob_gt * (prob + eps).log()).sum(dim=-1).mean()

def kl_divergence(prob, prob_gt):
    """Kullback-Leibler divergence loss for attention probabilities."""
    return (prob_gt * (prob_gt.log() - prob.log())).sum(dim=-1).mean()

def soft_argmax(prob, prob_gt, alpha=0.5, num_key_views: int = None):
    """Soft argmax loss for attention probabilities.
    Additional regularization to match entropy between two distributions.

    If `num_key_views` is provided, enforce per-key-view normalization (per-view softmax)
    by normalizing `prob` and `prob_gt` within each key-view tile before computing
    expectations. This preserves previous behavior when inputs are already per-view
    probabilities but fixes cases where a global softmax was incorrectly assumed.
    """
    # Optionally reshape and normalize per key-view tile
    if num_key_views is not None:
        K = prob.shape[-1]
        num_k = int(num_key_views)
        if K % num_k != 0:
            raise ValueError(f"num_key_views={num_k} is not a valid divisor of K={K}")
        other = K // num_k
        kp = int(round((other) ** 0.5))
        if kp * kp != other:
            raise ValueError(f"Given num_key_views={num_k} does not yield a square kp for K={K}")

        # reshape to (..., Q, num_k, kp, kp) and normalize within tiles
        p = prob.view(*prob.shape[:-1], num_k, kp, kp)
        pg = prob_gt.view(*prob_gt.shape[:-1], num_k, kp, kp)
        eps = 1e-12
        p = p / (p.sum(dim=(-2, -1), keepdim=True) + eps)
        pg = pg / (pg.sum(dim=(-2, -1), keepdim=True) + eps)
        # flatten back to (..., Q, K)
        prob = p.view(*prob.shape[:-1], K)
        prob_gt = pg.view(*prob_gt.shape[:-1], K)

    idxs = torch.arange(prob.shape[-1], device=prob.device).float()
    argmax = (prob * idxs).sum(dim=-1)
    argmax_gt = (prob_gt * idxs).sum(dim=-1)
    argmax_loss = torch.nn.functional.l1_loss(argmax, argmax_gt)
    eps = 1e-8
    entropy_loss = torch.nn.functional.mse((prob * (prob + eps).log()).sum(dim=-1), (prob_gt * (prob_gt + eps).log()).sum(dim=-1))
    return argmax_loss + alpha * entropy_loss

ATTN_LOSS_FN = {
    "l1": torch.nn.functional.l1_loss,
    "cross_entropy": cross_entropy,
    "kl_divergence": kl_divergence,
    "soft_argmax": soft_argmax,
}