"""Low-rank update over a frozen linear projection."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, in_features))
        self.b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.a), self.b) * self.scale
