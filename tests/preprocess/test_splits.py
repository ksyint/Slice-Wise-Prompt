"""Regression tests for visual prompting."""
import numpy as np
import torch
from slice_prompt.losses import class_balanced_focal
from slice_prompt.preprocess.splits import patient_split
from slice_prompt.preprocess.contours import render_prompt, slice_contours


def test_patient_split_no_overlap():
    rows = [{'patient_id': str(index // 2)} for index in range(40)]
    split = patient_split(rows)
    groups = [{row['patient_id'] for row in split[name]} for name in ['train', 'val', 'test']]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sum(map(len, split.values())) == len(rows)
    assert [len(group) for group in groups] == [14, 2, 4]
