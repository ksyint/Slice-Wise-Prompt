"""Volume alignment, contour prompts, and question-answer batches."""

import numpy as np
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import ndimage
from torch.utils.data import Dataset


def load_volume(path):
    path = Path(path)
    if path.suffix == '.npy':
        volume = np.load(path, allow_pickle=False)
    elif path.name.endswith(('.nii', '.nii.gz')):
        import nibabel as nib

        # NIfTI voxel array (X,Y,Z) -> axial stack (Z,Y,X).
        volume = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj).transpose(2, 1, 0)
    else:
        raise ValueError('Use .npy (Z,Y,X) or .nii[.gz] volume files.')
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Expected finite scalar volume of shape (Z,Y,X).')
    return volume


def load_pair(volume_path, mask_path):
    """Align a NIfTI mask to the canonical image grid using nearest-neighbor resampling."""
    volume_path, mask_path = Path(volume_path), Path(mask_path)
    if volume_path.name.endswith(('.nii', '.nii.gz')) and mask_path.name.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        from nibabel.processing import resample_from_to

        image = nib.as_closest_canonical(nib.load(volume_path))
        mask = nib.as_closest_canonical(nib.load(mask_path))
        if image.shape != mask.shape or not np.allclose(image.affine, mask.affine):
            mask = resample_from_to(mask, image, order=0)
        values = np.asarray(image.dataobj).transpose(2, 1, 0)
        labels = np.asarray(mask.dataobj).transpose(2, 1, 0)
        if values.ndim != 3 or not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError('Image and mask must be finite scalar 3D volumes.')
        return values, labels
    return load_volume(volume_path), load_volume(mask_path)


COLORS = {"blue": (0, 0, 1), "red": (1, 0, 0)}


def largest_component(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def slice_contours(mask, thickness=1, largest=True):
    if thickness < 1 or thickness % 2 == 0:
        raise ValueError('Contour thickness must be an odd positive integer.')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError('Expected mask (Z,Y,X).')
    contours = np.zeros_like(mask)
    for index, region in enumerate(mask):
        region = largest_component(region) if largest else region
        boundary = region & ~ndimage.binary_erosion(region, structure=np.ones((3, 3)), border_value=0)
        if thickness > 1:
            boundary = ndimage.binary_dilation(boundary, iterations=thickness // 2, structure=np.ones((3, 3)))
        contours[index] = boundary
    return contours


def render_prompt(volume, mask, color='blue', thickness=1, center=-600, width=1500):
    if volume.shape != mask.shape or width <= 0:
        raise ValueError('Volume/mask shapes must match and window width must be positive.')
    if color not in COLORS:
        raise ValueError(f'Unsupported color: {color}')
    gray = np.clip((volume.astype('float32') - center + width / 2) / width, 0, 1)
    rgb = np.repeat(gray[:, :, :, None], 3, axis=-1)
    contours = slice_contours(mask, thickness)
    rgb[contours] = COLORS[color]
    return rgb, contours


def rewrite_question(question, color='blue'):
    return f'Within the {color}-outlined area of the CT volume, {question.strip()}'


def system_prompt(color='blue', thickness=1):
    return (
        f'The CT slices contain a {color} boundary of thickness {thickness} pixel(s). '
        'Answer the question using the marked region. Describe only the requested structure. '
        'If the requested abnormality is absent, answer No finding. Keep the answer concise.'
    )


def patient_split(records, seed=42, fractions=(0.7, 0.1, 0.2)):
    if not np.isclose(sum(fractions), 1) or any(value < 0 for value in fractions):
        raise ValueError('Split fractions must be nonnegative and sum to one.')
    patients = sorted({str(record['patient_id']) for record in records})
    np.random.default_rng(seed).shuffle(patients)
    cut1 = int(round(len(patients) * fractions[0], 10))
    cut2 = int(round(len(patients) * sum(fractions[:2]), 10))
    allocation = {
        patient: ('train' if index < cut1 else 'val' if index < cut2 else 'test')
        for index, patient in enumerate(patients)
    }
    result = {key: [] for key in ('train', 'val', 'test')}
    for record in records:
        result[allocation[str(record['patient_id'])]].append(record)
    return result


class PromptDataset(Dataset):
    def __init__(self, manifest, slices=16, image_size=64):
        self.root = Path(manifest).resolve().parent
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError('Empty manifest.')
        self.slices, self.image_size = slices, image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        volume = np.load(self.root / row['prompt'], allow_pickle=False)
        if volume.ndim != 4 or volume.shape[1] != 3 or not np.isfinite(volume).all():
            raise ValueError('Prompt array must be finite (Z,3,H,W).')
        from slice_prompt.volumes.selection import uniform_indices, validate_indices

        indices = (
            validate_indices(row['slice_indices'], len(volume), self.slices)
            if row.get('slice_indices') is not None
            else uniform_indices(len(volume), self.slices)
        )
        tensor = F.interpolate(
            torch.tensor(volume[indices], dtype=torch.float32),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        return dict(row, tensor=tensor, prompt=str((self.root / row['prompt']).resolve()))


def pack_rows(rows, tokenizer, classes, max_length=1024, training=True):
    sequences, labels, prompts, prompt_lengths = [], [], [], []
    for row in rows:
        system = next(
            (turn['content'] for turn in row.get('conversations', []) if turn['role'] == 'system'),
            'Answer the medical question using the marked region of the CT volume.',
        )
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': row['question']}],
            tokenize=False,
            add_generation_prompt=True,
        )
        before = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target = tokenizer(row.get('answer', ''), add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        if len(before) + (len(target) if training else 0) > max_length:
            raise ValueError('Question and answer exceed max_length; increase the configured context budget.')
        ids = before + target if training else before
        supervised = [-100] * len(before) + target if training and not row.get('closed', False) else [-100] * len(ids)
        sequences.append(ids)
        labels.append(supervised)
        prompts.append(prompt)
        prompt_lengths.append(len(before))
    length = max(map(len, sequences))
    tokens = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    targets = torch.full_like(tokens, -100)
    valid = torch.zeros_like(tokens)
    question = torch.zeros_like(tokens, dtype=torch.bool)
    for i, ids in enumerate(sequences):
        tokens[i, : len(ids)] = torch.tensor(ids)
        targets[i, : len(ids)] = torch.tensor(labels[i])
        valid[i, : len(ids)] = 1
        question[i, : prompt_lengths[i]] = True
    return dict(
        records=[{key: value for key, value in row.items() if key != 'tensor'} for row in rows],
        volumes=torch.stack([row['tensor'] for row in rows]),
        input_ids=tokens,
        labels=targets,
        attention_mask=valid,
        question_mask=question,
        prompts=prompts,
        class_labels=torch.tensor(
            [classes.get(row.get('answer'), -1) if row.get('closed', False) else -1 for row in rows]
        ),
    )
