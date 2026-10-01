"""Prompt and mask quality measurements before model execution."""

import numpy as np

from slice_prompt.volumes.geometry import bounding_box
from slice_prompt.volumes.regions import axial_region_report, components_3d, contiguous_roi_runs
from slice_prompt.volumes.regions import component_shape_report
from slice_prompt.volumes.intensity import CTWindow, hu_histogram


def prompt_array_report(array):
    values = np.asarray(array)
    if values.ndim != 4 or values.shape[1] != 3 or min(values.shape) < 1:
        raise ValueError("Prompt arrays must have shape Z,3,H,W")
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Prompt array contains nonfinite values")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Prompt intensities must lie in [0,1]")
    spread = values.max(1) - values.min(1)
    colored = spread > 1e-5
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "colored_pixels": int(colored.sum()),
        "colored_slices": int(colored.any((1, 2)).sum()),
        "colored_fraction": float(colored.mean()),
        "channel_means": values.mean((0, 2, 3)).tolist(),
    }


def contour_agreement(prompt, contours, color="blue"):
    values = np.asarray(prompt)
    boundary = np.asarray(contours, dtype=bool)
    if boundary.ndim != 3 or values.shape != (len(boundary), 3, *boundary.shape[1:]):
        raise ValueError("Contour and prompt geometry differ")
    channel = {"red": 0, "blue": 2}.get(color)
    if channel is None:
        raise ValueError("Prompt color must be red or blue")
    rgb = np.moveaxis(values, 1, -1)
    expected = np.zeros(3, dtype=float)
    expected[channel] = 1
    detected = np.all(np.isclose(rgb, expected, atol=1e-6), axis=-1)
    overlap = int((detected & boundary).sum())
    return {
        "expected_pixels": int(boundary.sum()),
        "detected_pixels": int(detected.sum()),
        "matched_pixels": overlap,
        "missing_pixels": int((boundary & ~detected).sum()),
        "extra_pixels": int((detected & ~boundary).sum()),
    }


def volume_quality(volume, mask, grid=None, window=None):
    volume, mask = np.asarray(volume), np.asarray(mask, dtype=bool)
    if volume.shape != mask.shape or volume.ndim != 3:
        raise ValueError("Volume and mask must have identical scalar 3D grids")
    if not np.isfinite(volume).all():
        raise ValueError("Source volume contains nonfinite values")
    window = window or CTWindow()
    _, components = components_3d(mask, grid.spacing if grid else (1, 1, 1))
    slices = axial_region_report(mask)
    return {
        "grid": grid.describe() if grid else {"shape_zyx": list(volume.shape)},
        "region": bounding_box(mask, grid),
        "components": components,
        "component_shapes": component_shape_report(mask, grid.spacing if grid else (1, 1, 1)),
        "roi_runs": contiguous_roi_runs(mask),
        "slices": slices,
        "intensity": window.report(volume),
        "region_intensity": window.report(volume, mask),
        "histogram": hu_histogram(volume, mask),
        "empty_region": not bool(mask.any()),
        "constant_volume": bool(volume.min() == volume.max()),
        "border_slices": sum(row["touches_border"] for row in slices),
    }


def aggregate_quality(reports):
    reports = list(reports)
    if not reports:
        raise ValueError("Quality aggregation requires prepared records")
    return {
        "records": len(reports),
        "empty_regions": sum(row["empty_region"] for row in reports),
        "constant_volumes": sum(row["constant_volume"] for row in reports),
        "depth_range": [
            min(row["grid"]["shape_zyx"][0] for row in reports),
            max(row["grid"]["shape_zyx"][0] for row in reports),
        ],
        "mean_roi_voxels": float(np.mean([row["region"]["voxels"] for row in reports])),
        "maximum_components": max(len(row["components"]) for row in reports),
        "records_touching_border": sum(row["border_slices"] > 0 for row in reports),
    }
