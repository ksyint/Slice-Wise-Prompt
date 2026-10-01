"""Question-conditioned SSA and causal early-block MSSA for Qwen2.5."""
import torch
from torch import nn
import torch.nn.functional as F


class SliceTextFusion(nn.Module):
    def __init__(self, dim, heads, max_slices):
        super().__init__()
        if dim % heads:
            raise ValueError('Qwen width must divide into MSSA heads.')
        self.heads, self.max_slices = heads, max_slices
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_slices - 1))
        self.scale = nn.Parameter(torch.tensor(.01))

    def forward(self, hidden, prefix, valid):
        batch, length, dim = hidden.shape
        qkv = self.qkv(self.norm(hidden)).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        bias = hidden.new_zeros(batch, self.heads, length, length)
        positions = torch.arange(prefix, device=hidden.device)
        relative = positions[:, None] - positions[None, :] + self.max_slices - 1
        bias[:, :, :prefix, :prefix] = self.relative_bias[:, relative]
        bias.masked_fill_(~(causal[None, None] & valid[:, None, None, :]), -torch.inf)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        output = attention.transpose(1, 2).reshape(batch, length, dim)
        return hidden + self.scale * self.out(output)


class SliceQuestionAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.visual_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, visual, question, question_mask):
        text = self.text_norm(question)
        fused, _ = self.attention(self.visual_norm(visual), text, text,
                                  key_padding_mask=~question_mask, need_weights=False)
        return visual + fused
