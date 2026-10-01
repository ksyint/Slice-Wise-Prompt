"""Contour, patient partition, and focal objective invariants."""

import numpy as np
import torch
from slice_prompt.models.qwen.adaptation import class_balanced_focal
from slice_prompt.data.volumes.processing import patient_split, render_prompt, slice_contours


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


def test_patient_split_no_overlap():
    rows = [{'patient_id': str(index // 2)} for index in range(40)]
    split = patient_split(rows)
    groups = [{row['patient_id'] for row in split[name]} for name in ['train', 'val', 'test']]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sum(map(len, split.values())) == len(rows)
    assert [len(group) for group in groups] == [14, 2, 4]


def test_focal_prefers_correct_predictions():
    counts = torch.tensor([10, 10])
    label = torch.tensor([0])
    assert class_balanced_focal(torch.tensor([[4.0, -4.0]]), label, counts) < class_balanced_focal(
        torch.tensor([[-4.0, 4.0]]), label, counts
    )
