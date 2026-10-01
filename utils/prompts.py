"""CT windowing, per-slice connected components and consistent contour prompts."""
from pathlib import Path

import numpy as np
from scipy import ndimage


COLORS = {'blue': (0, 0, 1), 'red': (1, 0, 0)}


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
    return (f'The CT slices contain a {color} boundary of thickness {thickness} pixel(s). '
            'Answer the question using the marked region. Describe only the requested structure. '
            'If the requested abnormality is absent, answer No finding. Keep the answer concise.')


def patient_split(records, seed=42, fractions=(.7, .1, .2)):
    if not np.isclose(sum(fractions), 1) or any(value < 0 for value in fractions):
        raise ValueError('Split fractions must be nonnegative and sum to one.')
    patients = sorted({str(record['patient_id']) for record in records})
    np.random.default_rng(seed).shuffle(patients)
    cut1 = int(round(len(patients) * fractions[0], 10))
    cut2 = int(round(len(patients) * sum(fractions[:2]), 10))
    allocation = {patient: ('train' if index < cut1 else 'val' if index < cut2 else 'test') for index, patient in enumerate(patients)}
    result = {key: [] for key in ('train', 'val', 'test')}
    for record in records:
        result[allocation[str(record['patient_id'])]].append(record)
    return result
