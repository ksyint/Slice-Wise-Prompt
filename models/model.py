"""Relative slice attention, LoRA projections, and a compact VQA baseline."""
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


def class_balanced_focal(logits, labels, counts, gamma=2, beta=.999):
    if (counts <= 0).any():
        raise ValueError('Each class needs a positive training count.')
    weights = (1 - beta) / (1 - beta ** counts.float())
    weights = weights / weights.mean()
    logp = F.log_softmax(logits, -1).gather(1, labels[:, None]).squeeze(1)
    return (-weights[labels] * (1 - logp.exp()).pow(gamma) * logp).mean()


class SlicePromptVLM(nn.Module):
    def __init__(self, vocab_size, classes, hidden_dim=64, heads=4, max_slices=64, rank=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, hidden_dim, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.encoder.requires_grad_(False)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.mssa = MultiHeadSliceAttention(hidden_dim, heads, max_slices, rank)
        self.decoder = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.language_head = nn.Linear(hidden_dim, vocab_size)
        self.class_head = nn.Linear(hidden_dim, classes)

    def context(self, volumes, questions):
        batch, slices, channels, height, width = volumes.shape
        with torch.no_grad():
            visual = self.encoder(volumes.reshape(-1, channels, height, width)).reshape(batch, slices, -1)
        text = self.embedding(questions)
        fused, _ = self.mssa(visual, text, questions != 0)
        return fused[:, :slices].mean(1)

    def forward(self, volumes, questions, inputs):
        context = self.context(volumes, questions)
        tokens = self.embedding(inputs)
        output, _ = self.decoder(torch.cat((tokens, context[:, None].expand(-1, inputs.shape[1], -1)), -1), context[None])
        return self.class_head(context), self.language_head(output)

    def lora_penalty(self):
        return sum(parameter.square().sum() for name, parameter in self.named_parameters() if name.endswith(('.a', '.b')))

    @torch.no_grad()
    def generate(self, volumes, questions, max_tokens=40):
        context = self.context(volumes, questions)
        hidden, token = context[None], torch.full((len(volumes), 1), 2, device=volumes.device, dtype=torch.long)
        results, done = [], torch.zeros(len(volumes), device=volumes.device, dtype=torch.bool)
        for _ in range(max_tokens):
            output, hidden = self.decoder(torch.cat((self.embedding(token), context[:, None]), -1), hidden)
            token = self.language_head(output).argmax(-1)
            token[done] = 0
            results.append(token)
            done |= token[:, 0] == 3
            if done.all():
                break
        return self.class_head(context), torch.cat(results, 1)
