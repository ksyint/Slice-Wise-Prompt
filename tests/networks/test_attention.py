"""Regression tests for visual prompting."""
import numpy as np
import torch
from slice_prompt.networks.lora import LoRALinear
from slice_prompt.networks.attention import MultiHeadSliceAttention
from slice_prompt.losses import class_balanced_focal
from slice_prompt.preprocess.splits import patient_split
from slice_prompt.preprocess.contours import render_prompt, slice_contours


def test_attention_padding_and_lora_gradients():
    module = MultiHeadSliceAttention(16, heads=4, max_slices=4, rank=2)
    output, attention = module(torch.rand(2, 3, 16), torch.rand(2, 2, 16), torch.tensor([[True, False], [True, True]]))
    assert output.shape == (2, 5, 16)
    assert attention[0, :, :, -1].abs().max() == 0
    output.square().mean().backward()
    assert module.qkv.base.weight.grad is None
    assert module.qkv.b.grad.abs().sum() > 0
    assert torch.isfinite(module.relative_bias.grad).all()
