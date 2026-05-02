import torch.nn.functional as F


def causal_lm_cross_entropy(logits, targets, ignore_index=-100):
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
