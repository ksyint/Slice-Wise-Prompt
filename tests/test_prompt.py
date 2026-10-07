"""Contour, patient partition, and focal objective invariants."""

import numpy as np
import pytest
import torch
from slice_prompt.cache import PromptCache
from slice_prompt.models.adaptation import class_balanced_focal
from slice_prompt.processing import patient_split, render_prompt, rewrite_question, slice_contours, system_prompt
from slice_prompt.records.rewriting import question_prefix, region_instruction, rewrite_record


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


@pytest.mark.parametrize('color', ['blue', 'red'])
@pytest.mark.parametrize('thickness', [1, 3])
def test_preparation_instructions_match_paper_fallbacks(color, thickness):
    instruction = system_prompt(color, thickness)
    assert instruction == region_instruction(color, thickness)
    assert f'{color} outline' in instruction
    assert 'ignore other regions' in instruction
    assert 'closed question with no relevant finding, answer No finding.' in instruction
    assert 'open question with no relevant abnormality, answer No, the requested abnormality is absent.' in instruction


@pytest.mark.parametrize('question, prefix', [
    ('What is enlarged?', 'Within the blue-outlined area of the CT volume,'),
    ('Is the spleen enlarged?', 'Inside the blue boundary,'),
])
def test_question_rewriting_is_consistent_and_idempotent(question, prefix):
    expected = f'{prefix} {question}'
    assert rewrite_question(question, 'blue') == expected
    assert rewrite_question(expected, 'blue') == expected
    assert question_prefix(question_prefix(question, 'red'), 'blue') == expected


@pytest.mark.parametrize('closed', [False, True])
def test_cached_and_detailed_rewriting_preserve_answer_contract(tmp_path, closed):
    record = dict(volume='ct.nii.gz', question='Is the spleen enlarged?', answer='No', closed=closed)
    artifact = dict(prompt='prompt.npy', pngs=[], contour_voxels=8, intersecting_slices=1)
    cached = PromptCache(tmp_path).conversation(record, artifact)
    detailed = rewrite_record(record, color='blue')
    assert cached['question'] == detailed['question']
    assert cached['conversations'][0] == detailed['conversations'][0]
    assert cached['answer'] == detailed['answer'] == ('No' if closed else 'Within the blue boundary, No')
