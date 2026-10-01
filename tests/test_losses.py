"""Regression tests for visual prompting."""
import numpy as np
import torch
from slice_prompt.losses import class_balanced_focal
from slice_prompt.preprocess.splits import patient_split
from slice_prompt.preprocess.contours import render_prompt, slice_contours


def test_focal_prefers_correct_predictions():
    counts = torch.tensor([10, 10])
    label = torch.tensor([0])
    assert class_balanced_focal(torch.tensor([[4., -4.]]), label, counts) < class_balanced_focal(torch.tensor([[-4., 4.]]), label, counts)
