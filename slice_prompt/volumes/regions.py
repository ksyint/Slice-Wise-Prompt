"""Anatomical label selection, connected components, and contour measurements."""

import numpy as np
from scipy import ndimage


def select_region(mask, labels=None):
    values = np.asarray(mask)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Anatomical masks must be finite arrays of shape Z,Y,X")
    if not np.equal(values, np.round(values)).all() or np.any(values < 0):
        raise ValueError("Anatomical mask labels must be nonnegative integers")
    available, counts = np.unique(values.astype(np.int64), return_counts=True)
    if labels is None:
        selected = values != 0
        labels = available[available != 0].tolist()
    else:
        labels = list(labels)
        if not labels or any(type(label) is not int or label <= 0 for label in labels):
            raise ValueError("Region selection requires positive integer label IDs")
        if len(set(labels)) != len(labels):
            raise ValueError("Region label selection contains duplicates")
        selected = np.isin(values, labels)
    report = {
        "selected_labels": labels,
        "available_labels": {str(label): int(count) for label, count in zip(available, counts)},
        "missing_labels": sorted(set(labels) - set(available.tolist())),
        "region_voxels": int(selected.sum()),
    }
    return selected, report


def components_3d(mask, spacing=(1, 1, 1), connectivity=1):
    if connectivity not in (1, 2, 3):
        raise ValueError("3D connectivity must be one, two, or three")
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("Component masks must have shape Z,Y,X")
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, count = ndimage.label(values, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels)
    volume = float(np.prod(spacing))
    reports = []
    for index, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        reports.append(
            {
                "component": index,
                "voxels": int(sizes[index]),
                "volume_mm3": float(sizes[index] * volume),
                "lower_zyx": [axis.start for axis in bounds],
                "upper_zyx_exclusive": [axis.stop for axis in bounds],
                "axial_slices": bounds[0].stop - bounds[0].start,
            }
        )
    return labels, sorted(reports, key=lambda row: (-row["voxels"], row["component"]))


def retain_components(mask, minimum_voxels=0, largest=None, connectivity=1):
    if minimum_voxels < 0 or (largest is not None and largest < 1):
        raise ValueError("Component retention bounds must be nonnegative")
    labels, reports = components_3d(mask, connectivity=connectivity)
    retained = [row for row in reports if row["voxels"] >= minimum_voxels]
    if largest is not None:
        retained = retained[:largest]
    selected = [row["component"] for row in retained]
    return np.isin(labels, selected), {
        "original_components": len(reports),
        "retained_components": selected,
        "original_voxels": int(np.count_nonzero(mask)),
        "retained_voxels": sum(row["voxels"] for row in retained),
    }


def axial_region_report(mask):
    values = np.asarray(mask, dtype=bool)
    rows = []
    for index, region in enumerate(values):
        points = np.argwhere(region)
        labels, count = ndimage.label(region, np.ones((3, 3)))
        boundary = region & ~ndimage.binary_erosion(region, np.ones((3, 3)), border_value=0)
        sizes = np.bincount(labels.ravel())
        rows.append(
            {
                "slice": index,
                "area_pixels": int(region.sum()),
                "boundary_pixels": int(boundary.sum()),
                "components": int(count),
                "largest_component_pixels": int(sizes[1:].max()) if count else 0,
                "centroid_yx": points.mean(0).tolist() if len(points) else None,
                "touches_border": bool(
                    region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
                ),
            }
        )
    return rows


def contiguous_roi_runs(mask):
    selected = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if not len(selected):
        return []
    runs = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
    return [
        {"start_slice": int(run[0]), "end_slice_exclusive": int(run[-1] + 1), "slices": len(run)}
        for run in runs
    ]


def component_shape_report(mask, spacing=(1.0, 1.0, 1.0)):
    values = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Shape measurements require finite positive voxel spacing")
    labels, components = components_3d(values, spacing)
    reports = []
    for component in components:
        identifier = component["component"]
        low = np.asarray(component["lower_zyx"])
        high = np.asarray(component["upper_zyx_exclusive"])
        crop = tuple(slice(int(start), int(end)) for start, end in zip(low, high))
        selected = labels[crop] == identifier
        points = np.argwhere(selected).astype(float) + low
        coordinates = points * spacing
        centered = coordinates - coordinates.mean(0)
        covariance = centered.T @ centered / max(1, len(points))
        eigenvalues = np.maximum(0, np.linalg.eigvalsh(covariance))[::-1]
        extent = (high - low) * spacing
        physical_boundary = boundary_surface_area(selected, spacing)
        volume = component["volume_mm3"]
        sphericity = (
            np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / physical_boundary
            if physical_boundary
            else None
        )
        reports.append(
            {
                **component,
                "centroid_zyx_mm": coordinates.mean(0).tolist(),
                "extent_zyx_mm": extent.tolist(),
                "covariance_eigenvalues_mm2": eigenvalues.tolist(),
                "elongation": float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
                if eigenvalues[0]
                else None,
                "flatness": float(np.sqrt(eigenvalues[2] / eigenvalues[0]))
                if eigenvalues[0]
                else None,
                "bounding_box_occupancy": len(points) / float(np.prod(high - low)),
                "surface_area_mm2": physical_boundary,
                "voxel_surface_sphericity": float(sphericity) if sphericity is not None else None,
            }
        )
    return reports


def boundary_surface_area(mask, spacing=(1.0, 1.0, 1.0)):
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("Surface measurements require a scalar 3D mask")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("Voxel surface spacing must be positive")
    padded = np.pad(values.astype(np.int8), 1)
    area = 0.0
    for axis in range(3):
        faces = np.abs(np.diff(padded, axis=axis)).sum()
        face_area = np.prod(np.delete(spacing, axis))
        area += float(faces * face_area)
    return area


def contour_topology(mask, contours):
    values = np.asarray(mask, dtype=bool)
    boundaries = np.asarray(contours, dtype=bool)
    if values.shape != boundaries.shape or values.ndim != 3:
        raise ValueError("Mask and contour topology require identical axial grids")
    rows = []
    for index, (region, boundary) in enumerate(zip(values, boundaries)):
        _, region_count = ndimage.label(region, np.ones((3, 3)))
        _, boundary_count = ndimage.label(boundary, np.ones((3, 3)))
        filled = ndimage.binary_fill_holes(region)
        holes = filled & ~region
        _, hole_count = ndimage.label(holes, np.ones((3, 3)))
        lost = region.any() and not boundary.any()
        rows.append(
            {
                "slice": index,
                "region_components": int(region_count),
                "boundary_components": int(boundary_count),
                "holes": int(hole_count),
                "hole_pixels": int(holes.sum()),
                "region_without_boundary": bool(lost),
                "boundary_inside_region": int((boundary & region).sum()),
                "boundary_outside_region": int((boundary & ~region).sum()),
            }
        )
    return rows


def mask_agreement(reference, candidate, spacing=(1.0, 1.0, 1.0)):
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("Mask comparison requires identical scalar 3D grids")
    intersection = int((reference & candidate).sum())
    union = int((reference | candidate).sum())
    total = int(reference.sum() + candidate.sum())
    result = {
        "reference_voxels": int(reference.sum()),
        "candidate_voxels": int(candidate.sum()),
        "intersection_voxels": intersection,
        "dice": 2 * intersection / total if total else 1.0,
        "iou": intersection / union if union else 1.0,
        "added_voxels": int((candidate & ~reference).sum()),
        "removed_voxels": int((reference & ~candidate).sum()),
    }
    if not reference.any() or not candidate.any():
        result.update(average_surface_distance_mm=None, hausdorff95_mm=None)
        return result
    structure = ndimage.generate_binary_structure(3, 1)
    first = reference & ~ndimage.binary_erosion(reference, structure)
    second = candidate & ~ndimage.binary_erosion(candidate, structure)
    distance_first = ndimage.distance_transform_edt(~first, sampling=spacing)
    distance_second = ndimage.distance_transform_edt(~second, sampling=spacing)
    distances = np.concatenate((distance_first[second], distance_second[first]))
    result.update(
        average_surface_distance_mm=float(distances.mean()),
        hausdorff95_mm=float(np.quantile(distances, 0.95)),
        maximum_surface_distance_mm=float(distances.max()),
    )
    return result
