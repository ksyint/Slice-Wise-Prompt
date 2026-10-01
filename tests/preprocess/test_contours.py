"""Regression tests for visual prompting."""
import numpy as np
import torch
from slice_prompt.losses import class_balanced_focal
from slice_prompt.preprocess.splits import patient_split
from slice_prompt.preprocess.contours import render_prompt, slice_contours


def test_contour_interior_and_empty_slices():
    mask = np.zeros((3, 12, 12), bool)
    mask[1, 2:10, 2:10] = True
    mask[1, 0, 0] = True
    contour = slice_contours(mask)
    assert not contour[0].any() and not contour[2].any()
    assert not contour[1, 0, 0] and not contour[1, 5, 5]
    assert contour[1].sum() == 28
    rgb, border = render_prompt(np.zeros_like(mask, dtype=float), mask)
    np.testing.assert_array_equal(rgb[border], np.tile([0, 0, 1], (28, 1)))
