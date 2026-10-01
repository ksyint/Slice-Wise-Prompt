"""Multimodal slice attention with relative slice-position biases."""
import torch
from torch import nn
from .lora import LoRALinear


class MultiHeadSliceAttention(nn.Module):
    """MSSA over [slice; question] tokens; R acts on slice/slice entries only."""
    def __init__(self, dim, heads=4, max_slices=64, rank=16):
        super().__init__()
        if dim % heads:
            raise ValueError('Dimension must be divisible by heads.')
        self.heads, self.max_slices = heads, max_slices
        self.qkv = LoRALinear(dim, dim * 3, rank)
        self.projection = LoRALinear(dim, dim, rank)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_slices - 1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, slices, text, text_mask):
        count = slices.shape[1]
        if count > self.max_slices:
            raise ValueError('Too many slices; resample or increase max_slices.')
        tokens = torch.cat((slices, text), 1)
        batch, length, dim = tokens.shape
        qkv = self.qkv(self.norm(tokens)).view(batch, length, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = query @ key.transpose(-1, -2) / (dim // self.heads) ** .5
        positions = torch.arange(count, device=tokens.device)
        relative = positions[:, None] - positions[None, :] + self.max_slices - 1
        bias = tokens.new_zeros(self.heads, length, length)
        bias[:, :count, :count] = self.relative_bias[:, relative]
        key_mask = torch.cat((torch.ones(batch, count, dtype=torch.bool, device=tokens.device), text_mask), 1)
        attention = torch.softmax((logits + bias).masked_fill(~key_mask[:, None, None], -torch.inf), -1)
        output = (attention @ value).transpose(1, 2).reshape(batch, length, dim)
        return tokens + self.projection(output), attention
