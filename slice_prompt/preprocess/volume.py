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
