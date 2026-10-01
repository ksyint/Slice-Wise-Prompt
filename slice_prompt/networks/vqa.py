"""Frozen slice encoder with trainable attention and language decoding."""
import torch
from torch import nn
from .attention import MultiHeadSliceAttention


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
