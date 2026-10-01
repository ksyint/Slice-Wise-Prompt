"""CT windows and intensity distributions recorded with prepared prompt stacks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CTWindow:
    center: float = -600.0
    width: float = 1500.0

    def __post_init__(self):
        if not np.isfinite(self.center) or not np.isfinite(self.width) or self.width <= 0:
            raise ValueError("CT center and positive window width must be finite")

    def apply(self, values):
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("CT volume contains nonfinite intensities")
        return np.clip((values - self.center + self.width / 2) / self.width, 0, 1)

    def inverse(self, values):
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Windowed values must lie in [0,1]")
        return values * self.width + self.center - self.width / 2

    def report(self, values, mask=None):
        values = np.asarray(values)
        if mask is not None:
            if values.shape != np.shape(mask):
                raise ValueError("Intensity region and volume shapes differ")
            values = values[np.asarray(mask, dtype=bool)]
        values = values.reshape(-1)
        if not len(values):
            return {"voxels": 0, "center": self.center, "width": self.width}
        low, high = self.center - self.width / 2, self.center + self.width / 2
        return {
            "voxels": len(values),
            "center": self.center,
            "width": self.width,
            "lower_hu": low,
            "upper_hu": high,
            "clipped_low_fraction": float((values <= low).mean()),
            "clipped_high_fraction": float((values >= high).mean()),
            "hu_minimum": float(values.min()),
            "hu_maximum": float(values.max()),
            "hu_mean": float(values.mean()),
            "hu_std": float(values.std()),
            "hu_percentiles": np.percentile(values, [1, 10, 50, 90, 99]).tolist(),
        }


WINDOWS = {
    "lung": CTWindow(-600, 1500),
    "mediastinum": CTWindow(40, 400),
    "bone": CTWindow(400, 1800),
    "brain": CTWindow(40, 80),
}


def resolve_window(name=None, center=None, width=None):
    if name is not None:
        if center is not None or width is not None:
            raise ValueError("Select a named window or explicit center and width")
        if name not in WINDOWS:
            raise ValueError(f"Unknown CT window: {name}")
        return WINDOWS[name]
    if (center is None) != (width is None):
        raise ValueError("Explicit intensity windows need both center and width")
    return CTWindow() if center is None else CTWindow(center, width)


def slice_intensity_report(volume, mask, window):
    if np.shape(volume) != np.shape(mask) or np.ndim(volume) != 3:
        raise ValueError("Expected aligned scalar volume and mask stacks")
    rows = []
    for index, (image, region) in enumerate(zip(volume, mask)):
        report = window.report(image, region)
        rows.append(dict(slice=index, **report))
    return rows


def compare_windows(volume, mask, names=("lung", "mediastinum")):
    reports = {}
    for name in names:
        window = resolve_window(name)
        reports[name] = {
            "whole_volume": window.report(volume),
            "region": window.report(volume, mask),
        }
    return reports


def hu_histogram(volume, mask=None, edges=None):
    values = np.asarray(volume)
    if mask is not None:
        if values.shape != np.shape(mask):
            raise ValueError("Histogram mask shape differs from the volume")
        values = values[np.asarray(mask, dtype=bool)]
    edges = np.asarray(edges if edges is not None else np.linspace(-1200, 2000, 65))
    if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("Histogram edges must increase strictly")
    counts, _ = np.histogram(values, edges)
    return {
        "edges_hu": edges.tolist(),
        "counts": counts.tolist(),
        "below_range": int(np.sum(values < edges[0])),
        "above_range": int(np.sum(values > edges[-1])),
    }
