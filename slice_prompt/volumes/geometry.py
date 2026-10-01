"""Canonical axial volume grids and physical-space region measurements."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VolumeGrid:
    shape: tuple
    spacing: tuple
    affine: np.ndarray
    orientation: tuple = ("R", "A", "S")

    def __post_init__(self):
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Volume shape must contain three positive dimensions")
        if len(self.spacing) != 3 or not np.isfinite(self.spacing).all() or min(self.spacing) <= 0:
            raise ValueError("Voxel spacing must contain three positive values")
        affine = np.asarray(self.affine, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("Voxel-to-world affine must be a finite 4 by 4 matrix")
        if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("Voxel grid affine is singular")

    @property
    def voxel_volume(self):
        return float(abs(np.linalg.det(self.affine[:3, :3])))

    def world_coordinates(self, zyx):
        points = np.asarray(zyx, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Voxel coordinates must have shape N by 3 in Z,Y,X order")
        homogeneous = np.concatenate((points[:, ::-1], np.ones((len(points), 1))), axis=1)
        return (homogeneous @ np.asarray(self.affine).T)[:, :3]

    def describe(self):
        return {
            "shape_zyx": list(self.shape),
            "spacing_zyx_mm": list(self.spacing),
            "affine_xyz_to_ras": np.asarray(self.affine).tolist(),
            "orientation": list(self.orientation),
            "voxel_volume_mm3": self.voxel_volume,
        }


def read_grid(path, npy_spacing=None):
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        spacing = tuple(npy_spacing or (1.0, 1.0, 1.0))
        affine = np.diag([*spacing[::-1], 1.0])
        return VolumeGrid(tuple(array.shape), spacing, affine)
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        image = nib.as_closest_canonical(nib.load(path))
        return VolumeGrid(
            tuple(image.shape[::-1]),
            tuple(float(value) for value in image.header.get_zooms()[:3][::-1]),
            np.asarray(image.affine),
            tuple(nib.aff2axcodes(image.affine)),
        )
    raise ValueError("Volume grids support NIfTI and NPY inputs")


def region_bounds(mask):
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("Region masks must have shape Z,Y,X")
    coordinates = np.argwhere(values)
    if not len(coordinates):
        return None
    return coordinates.min(0), coordinates.max(0) + 1


def bounding_box(mask, grid=None):
    bounds = region_bounds(mask)
    if bounds is None:
        return {"empty": True, "voxels": 0}
    low, high = bounds
    result = {
        "empty": False,
        "voxels": int(np.count_nonzero(mask)),
        "lower_zyx": low.tolist(),
        "upper_zyx_exclusive": high.tolist(),
        "extent_zyx": (high - low).tolist(),
        "centroid_zyx": np.argwhere(mask).mean(0).tolist(),
    }
    if grid is not None:
        corners = np.asarray(
            [
                [z, y, x]
                for z in (low[0], high[0] - 1)
                for y in (low[1], high[1] - 1)
                for x in (low[2], high[2] - 1)
            ]
        )
        world = grid.world_coordinates(corners)
        result.update(
            volume_mm3=result["voxels"] * grid.voxel_volume,
            world_minimum_ras_mm=world.min(0).tolist(),
            world_maximum_ras_mm=world.max(0).tolist(),
            centroid_ras_mm=grid.world_coordinates(np.asarray([result["centroid_zyx"]]))[
                0
            ].tolist(),
        )
    return result


def physical_crop(mask, grid, margin_mm=0.0):
    if margin_mm < 0 or not np.isfinite(margin_mm):
        raise ValueError("Physical crop margin must be finite and nonnegative")
    bounds = region_bounds(mask)
    if bounds is None:
        return tuple(slice(0, dimension) for dimension in grid.shape)
    low, high = bounds
    margin = np.ceil(margin_mm / np.asarray(grid.spacing)).astype(int)
    low = np.maximum(low - margin, 0)
    high = np.minimum(high + margin, grid.shape)
    return tuple(slice(int(start), int(end)) for start, end in zip(low, high))


def compare_grids(left, right, tolerance=1e-4):
    return {
        "shape_equal": tuple(left.shape) == tuple(right.shape),
        "spacing_equal": bool(np.allclose(left.spacing, right.spacing, atol=tolerance)),
        "affine_equal": bool(np.allclose(left.affine, right.affine, atol=tolerance)),
        "orientation_equal": left.orientation == right.orientation,
    }
