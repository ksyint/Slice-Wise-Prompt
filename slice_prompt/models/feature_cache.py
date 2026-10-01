"""Frozen Med3D visual features keyed by preprocessing and source arrays."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch as th


def array_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


class VisualFeatureCache:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings = {
            key: config.get(key)
            for key in ("vision_source", "vision_revision", "slices", "image_size", "precision")
        }
        self.memo = {}

    def identity(self, row):
        source = str(Path(row["prompt"]).resolve())
        stat = Path(source).stat()
        key = source, stat.st_mtime_ns, stat.st_size
        if key not in self.memo:
            self.memo[key] = array_identity(source)
        identity = {
            "format": "slice-visual-features-v1",
            "source": self.memo[key],
            "settings": self.settings,
            "slice_indices": row.get("slice_indices"),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return digest, identity

    def paths(self, row):
        key, identity = self.identity(row)
        directory = self.directory / key[:2]
        return directory / f"{key}.safetensors", directory / f"{key}.json", identity

    def read(self, row, device="cuda"):
        from safetensors.torch import load_file

        if th.device(device).type != "cuda":
            raise ValueError("Visual features are loaded onto CUDA")
        tensors_path, metadata_path, identity = self.paths(row)
        if not tensors_path.exists() or not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise ValueError("Visual feature identity differs from its preprocessing settings")
        values = load_file(str(tensors_path), device=str(device))
        if set(values) != {"native", "axial"}:
            raise ValueError("Visual cache must contain native and axial features")
        for name, value in values.items():
            if list(value.shape) != metadata["shapes"][name] or not th.isfinite(value).all():
                raise ValueError(f"Invalid cached {name} features")
        if values["native"].ndim != 2 or values["axial"].ndim != 2:
            raise ValueError("Per-volume cached features must have token and channel axes")
        if len(values["axial"]) != self.settings["slices"]:
            raise ValueError("Cached axial features differ from the configured slice count")
        return values

    def write(self, row, native, axial):
        from safetensors.torch import save_file

        tensors_path, metadata_path, identity = self.paths(row)
        tensors_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            "native": native.detach().contiguous().cpu(),
            "axial": axial.detach().contiguous().cpu(),
        }
        if any(value.ndim != 2 or not th.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Cannot cache invalid visual feature tensors")
        with tempfile.NamedTemporaryFile(dir=tensors_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            save_file(tensors, str(temporary))
            temporary.replace(tensors_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        metadata = {
            "identity": identity,
            "shapes": {name: list(value.shape) for name, value in tensors.items()},
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        with temporary_metadata.open("w") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_metadata.replace(metadata_path)

    @th.no_grad()
    def encode(self, visual, rows, volumes):
        if volumes.device.type != "cuda":
            raise ValueError("Visual encoding requires CUDA input stacks")
        if len(rows) != len(volumes):
            raise ValueError("Visual rows and tensor batches must be aligned")
        values = [self.read(row, volumes.device) for row in rows]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            native, axial = visual(volumes[missing])
            for offset, index in enumerate(missing):
                self.write(rows[index], native[offset], axial[offset])
                values[index] = {"native": native[offset], "axial": axial[offset]}
        return th.stack([value["native"] for value in values]), th.stack(
            [value["axial"] for value in values]
        )

    def audit(self):
        entries = []
        for tensor in sorted(self.directory.rglob("*.safetensors")):
            path = tensor.with_suffix(".json")
            if not path.is_file():
                entries.append(
                    {
                        "metadata": str(path),
                        "present": False,
                        "bytes": tensor.stat().st_size,
                        "shapes": {},
                    }
                )
                continue
            metadata = json.loads(path.read_text())
            entries.append(
                {
                    "metadata": str(path),
                    "present": tensor.is_file(),
                    "bytes": tensor.stat().st_size if tensor.exists() else 0,
                    "shapes": metadata["shapes"],
                }
            )
        return {
            "entries": entries,
            "count": len(entries),
            "total_bytes": sum(row["bytes"] for row in entries),
        }


@th.no_grad()
def feature_statistics(values):
    if values.ndim != 2 or values.device.type != "cuda":
        raise ValueError("Feature statistics require a CUDA token-by-channel matrix")
    values = values.float()
    if not th.isfinite(values).all():
        raise ValueError("Feature statistics cannot summarize nonfinite tensors")
    norms = values.norm(dim=1)
    normalized = th.nn.functional.normalize(values, dim=-1)
    adjacent = (normalized[:-1] * normalized[1:]).sum(-1)
    mean = values.mean(0)
    centered = values - mean
    variance = centered.square().mean(0)
    total = variance.sum()
    probabilities = variance / total.clamp_min(th.finfo(values.dtype).tiny)
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return {
        "tokens": len(values),
        "channels": values.shape[1],
        "mean_norm": float(norms.mean()),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "zero_tokens": int((norms == 0).sum()),
        "mean_adjacent_cosine": float(adjacent.mean()) if len(adjacent) else None,
        "minimum_adjacent_cosine": float(adjacent.min()) if len(adjacent) else None,
        "mean_channel_variance": float(variance.mean()),
        "maximum_channel_variance": float(variance.max()),
        "effective_varying_channels": float(entropy.exp()) if total > 0 else 0.0,
        "mean_vector_norm": float(mean.norm()),
    }


@th.no_grad()
def compare_visual_features(left, right):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Paired visual feature matrices must have identical shapes")
    if left.device.type != "cuda" or right.device.type != "cuda":
        raise ValueError("Visual feature comparisons require CUDA")
    first, second = left.float(), right.float()
    difference = second - first
    cosine = th.nn.functional.cosine_similarity(first, second, dim=-1)
    norm = first.norm()
    return {
        "tokens": len(first),
        "mean_token_cosine": float(cosine.mean()),
        "minimum_token_cosine": float(cosine.min()),
        "maximum_absolute_change": float(difference.abs().max()),
        "root_mean_square_change": float(difference.square().mean().sqrt()),
        "relative_norm_change": float(difference.norm() / norm) if norm > 0 else None,
        "per_token_cosine": cosine.cpu().tolist(),
    }


def cache_feature_report(cache, records, device="cuda", baseline=None):
    rows = []
    missing = []
    for record in records:
        features = cache.read(record, device)
        if features is None:
            missing.append(record["id"])
            continue
        row = {
            "id": record["id"],
            "patient_id": record["patient_id"],
            "native": feature_statistics(features["native"]),
            "axial": feature_statistics(features["axial"]),
        }
        if baseline is not None:
            previous = baseline.read(record, device)
            if previous is None:
                row["baseline_present"] = False
            else:
                row["baseline_present"] = True
                row["native_change"] = compare_visual_features(
                    previous["native"], features["native"]
                )
                row["axial_change"] = compare_visual_features(previous["axial"], features["axial"])
        rows.append(row)
    return {
        "records": rows,
        "missing_ids": missing,
        "requested": len(records),
        "present": len(rows),
        "settings": cache.settings,
    }
