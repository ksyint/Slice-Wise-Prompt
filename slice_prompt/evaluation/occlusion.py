"""CUDA slice occlusion scores for the closed-answer decision."""

import torch as th


def closed_probability(model, batch, target=None):
    output = model(batch)["classes"].float()
    probability = output.softmax(-1)
    if target is None:
        target = probability.argmax(-1)
    if target.shape != (len(output),):
        raise ValueError("Occlusion target labels must align with the batch")
    selected = probability.gather(1, target[:, None]).squeeze(1)
    return selected, target, probability


def neutral_slice(volumes, replacement="mean"):
    if replacement == "mean":
        return volumes.mean(dim=1, keepdim=True)
    if replacement == "zero":
        return th.zeros_like(volumes[:, :1])
    if replacement == "gray":
        gray = volumes.mean(dim=2, keepdim=True)
        return gray.mean(dim=1, keepdim=True).expand(-1, 1, 3, -1, -1)
    raise ValueError("Slice replacement must be mean, zero, or gray")


@th.no_grad()
def slice_occlusion(model, batch, target=None, replacement="mean", window=1, stride=1):
    volumes = batch["volumes"]
    if volumes.device.type != "cuda" or volumes.ndim != 5:
        raise ValueError("Occlusion expects CUDA volumes with shape B,Z,3,H,W")
    if window < 1 or stride < 1 or window > volumes.shape[1]:
        raise ValueError("Occlusion window must fit within the sampled axial stack")
    previous_mode = model.training
    model.eval()
    original_cache = getattr(model, "visual_cache", None)
    model.visual_cache = None
    try:
        with th.autocast("cuda", dtype=model.dtype):
            baseline, targets, probabilities = closed_probability(model, batch, target)
            reference = neutral_slice(volumes, replacement)
            positions = list(range(0, volumes.shape[1] - window + 1, stride))
            if positions[-1] != volumes.shape[1] - window:
                positions.append(volumes.shape[1] - window)
            contributions = th.zeros(volumes.shape[:2], device=volumes.device)
            counts = th.zeros(volumes.shape[1], device=volumes.device)
            windows = []
            for start in positions:
                modified = volumes.clone()
                modified[:, start : start + window] = reference
                changed = dict(batch, volumes=modified)
                prediction, _, _ = closed_probability(model, changed, targets)
                difference = baseline - prediction
                contributions[:, start : start + window] += difference[:, None]
                counts[start : start + window] += 1
                windows.append(
                    {
                        "start_slice": start,
                        "end_slice_exclusive": start + window,
                        "probability": prediction.float().cpu().tolist(),
                        "drop": difference.float().cpu().tolist(),
                    }
                )
            scores = contributions / counts.clamp_min(1)[None]
        return {
            "targets": targets.cpu().tolist(),
            "baseline_probability": baseline.cpu().tolist(),
            "baseline_distribution": probabilities.cpu().tolist(),
            "slice_scores": scores.cpu().tolist(),
            "evaluated_slices": (counts > 0).cpu().tolist(),
            "replacement": replacement,
            "window": window,
            "stride": stride,
            "windows": windows,
        }
    finally:
        model.visual_cache = original_cache
        model.train(previous_mode)


def roi_concentration(slice_scores, roi_areas):
    scores = th.as_tensor(slice_scores, dtype=th.float64)
    areas = th.as_tensor(roi_areas, dtype=th.float64)
    if scores.ndim != 1 or scores.shape != areas.shape:
        raise ValueError("Occlusion scores and ROI slice areas must align")
    if not th.isfinite(scores).all() or not th.isfinite(areas).all() or (areas < 0).any():
        raise ValueError("Slice measurements must be finite with nonnegative ROI areas")
    positive = scores.clamp_min(0)
    inside = areas > 0
    total = positive.sum()
    return {
        "roi_slices": int(inside.sum()),
        "total_slices": len(scores),
        "positive_mass_in_roi": float(positive[inside].sum() / total) if total > 0 else None,
        "mean_drop_inside": float(scores[inside].mean()) if inside.any() else None,
        "mean_drop_outside": float(scores[~inside].mean()) if (~inside).any() else None,
        "largest_drop_slice": int(scores.argmax()) if len(scores) else None,
    }
