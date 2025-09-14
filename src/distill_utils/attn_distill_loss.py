import torch

# loss function
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