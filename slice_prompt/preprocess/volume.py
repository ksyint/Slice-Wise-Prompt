"""Scalar volume loading and CT windowing."""
from pathlib import Path
import numpy as np


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
