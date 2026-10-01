"""Class-balanced focal loss for region-grounded closed questions."""
import torch
import torch.nn.functional as F


def class_balanced_focal(logits, labels, counts, gamma=2, beta=.999):
    if (counts <= 0).any():
        raise ValueError('Each class needs a positive training count.')
    weights = (1 - beta) / (1 - beta ** counts.float())
    weights = weights / weights.mean()
    logp = F.log_softmax(logits, -1).gather(1, labels[:, None]).squeeze(1)
    return (-weights[labels] * (1 - logp.exp()).pow(gamma) * logp).mean()
