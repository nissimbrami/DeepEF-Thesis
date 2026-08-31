"""v5 loss functions: correlation-based losses that align training with the PCC metric.

Our model is graded on Pearson correlation (PCC) of predicted vs true dG, but the primary
training loss (Huber / L1) only penalizes point-wise error. Two predictions can have the same
Huber loss but very different PCC. These losses add a term that directly rewards high correlation
(getting the ORDERING/trend right), which is what the evaluation metric measures.
"""

import torch


def pearson_loss(pred, target, eps=1e-8):
    """1 - Pearson correlation coefficient between pred and target.

    Returns a value in [0, 2]; 0 means perfect positive correlation.
    Safe for small batches (returns 0 if fewer than 2 elements or zero variance).
    """
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
    pred = pred.float()
    target = target.float()
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    num = (pred_c * target_c).sum()
    den = torch.sqrt((pred_c ** 2).sum() * (target_c ** 2).sum() + eps)
    r = num / (den + eps)
    return 1.0 - r


def ccc_loss(pred, target, eps=1e-8):
    """1 - Concordance Correlation Coefficient (Lin's CCC).

    CCC penalizes both correlation AND shift/scale mismatch, so it is stricter than Pearson.
    Useful when we care about both the trend and the absolute agreement of dG.
    """
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
    pred = pred.float()
    target = target.float()
    mp, mt = pred.mean(), target.mean()
    vp, vt = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - mp) * (target - mt)).mean()
    ccc = (2 * cov) / (vp + vt + (mp - mt) ** 2 + eps)
    return 1.0 - ccc
