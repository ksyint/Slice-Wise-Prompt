"""Axial sampling plans with explicit region coverage measurements."""

import numpy as np


def uniform_indices(depth, count):
    if depth < 1 or count < 1:
        raise ValueError("Volume depth and requested slice count must be positive")
    return np.linspace(0, depth - 1, count).round().astype(np.int64)


def region_indices(mask, count, context=0):
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or count < 1 or context < 0:
        raise ValueError("Invalid region sampling inputs")
    active = np.flatnonzero(np.any(values, axis=(1, 2)))
    if not len(active):
        return uniform_indices(len(values), count)
    start = max(0, int(active[0]) - context)
    end = min(len(values) - 1, int(active[-1]) + context)
    return np.linspace(start, end, count).round().astype(np.int64)


def coverage_indices(mask, count, seed=42):
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or count < 1:
        raise ValueError("Invalid coverage sampling inputs")
    depth = len(values)
    if count >= depth:
        return uniform_indices(depth, count)
    areas = values.sum(axis=(1, 2)).astype(float)
    if areas.sum() == 0:
        return uniform_indices(depth, count)
    distribution = areas / areas.sum()
    cdf = distribution.cumsum()
    quantiles = (np.arange(count) + 0.5) / count
    chosen = np.unique(np.searchsorted(cdf, quantiles).clip(max=depth - 1))
    remaining = count - len(chosen)
    if remaining:
        available = np.setdiff1d(np.arange(depth), chosen)
        priorities = areas[available]
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(len(available)), -priorities))
        chosen = np.concatenate((chosen, available[order[:remaining]]))
    return np.sort(chosen).astype(np.int64)


def sampling_plan(mask, count, strategy="uniform", context=0, seed=42):
    if strategy == "uniform":
        indices = uniform_indices(len(mask), count)
    elif strategy == "region":
        indices = region_indices(mask, count, context)
    elif strategy == "coverage":
        indices = coverage_indices(mask, count, seed)
    else:
        raise ValueError(f"Unknown axial sampling strategy: {strategy}")
    active = np.any(mask, axis=(1, 2))
    unique = np.unique(indices)
    return {
        "strategy": strategy,
        "indices": indices.tolist(),
        "original_depth": len(mask),
        "requested_slices": count,
        "unique_slices": len(unique),
        "repeated_slices": count - len(unique),
        "roi_slices": int(active.sum()),
        "selected_roi_slices": int(active[unique].sum()),
        "roi_slice_recall": float(active[unique].sum() / active.sum()) if active.any() else None,
        "selected_roi_voxels": int(np.asarray(mask)[unique].sum()),
        "original_roi_voxels": int(np.asarray(mask).sum()),
    }


def validate_indices(indices, depth, expected_count=None):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("Slice indices must form a nonempty vector")
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Slice positions must be integers")
    if np.any((indices < 0) | (indices >= depth)):
        raise ValueError("Slice positions fall outside the volume")
    if np.any(np.diff(indices) < 0):
        raise ValueError("Slice positions must preserve axial order")
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError("Slice count differs from the model configuration")
    return indices.astype(np.int64)


def relative_positions(indices, depth):
    values = validate_indices(indices, depth)
    return (values.astype(float) / max(1, depth - 1)).tolist()


def selection_ablation(mask, counts=(16, 32), strategies=("uniform", "region", "coverage")):
    return [sampling_plan(mask, count, strategy) for count in counts for strategy in strategies]
