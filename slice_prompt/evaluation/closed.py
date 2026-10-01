"""Closed-answer accuracy, confusion, and confidence measurements."""

from collections import Counter

import numpy as np


def normalized_answer(value):
    import re

    return " ".join(re.findall(r"\w+", str(value).casefold()))


def closed_records(records):
    rows = [row for row in records if row["closed"] and row.get("answer") is not None]
    if not rows:
        return []
    for row in rows:
        if not isinstance(row["prediction"], str) or not isinstance(row["answer"], str):
            raise ValueError("Closed answers and predictions must be strings")
    return rows


def closed_report(records):
    rows = closed_records(records)
    if not rows:
        return {"questions": 0, "accuracy": None}
    targets = [normalized_answer(row["answer"]) for row in rows]
    predictions = [normalized_answer(row["prediction"]) for row in rows]
    classes = sorted(set(targets) | set(predictions))
    lookup = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[lookup[target], lookup[prediction]] += 1
    per_class = []
    for index, label in enumerate(classes):
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        correct = int(matrix[index, index])
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        per_class.append(
            {
                "answer": label,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    supported = [row for row in per_class if row["support"]]
    return {
        "questions": len(rows),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_recall": float(np.mean([row["recall"] for row in supported])),
        "macro_f1": float(np.mean([row["f1"] for row in supported])),
        "classes": classes,
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "unknown_predictions": dict(
            Counter(prediction for prediction in predictions if prediction not in set(targets))
        ),
    }


def confidence_report(records, bins=10):
    rows = [row for row in closed_records(records) if row.get("confidence") is not None]
    if bins < 1:
        raise ValueError("Confidence bins must be positive")
    if not rows:
        return {"questions": 0, "ece": None, "bins": []}
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Closed-answer confidence must lie in [0,1]")
    correct = np.asarray(
        [normalized_answer(row["answer"]) == normalized_answer(row["prediction"]) for row in rows]
    )
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    report = []
    weighted_gap = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        average = float(confidence[selected].mean()) if count else None
        if count:
            weighted_gap += count * abs(accuracy - average)
        report.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "accuracy": accuracy,
                "confidence": average,
            }
        )
    return {"questions": len(rows), "ece": weighted_gap / len(rows), "bins": report}


def abstention_curve(records, thresholds=(0.0, 0.5, 0.7, 0.9)):
    rows = [row for row in closed_records(records) if row.get("confidence") is not None]
    result = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Abstention thresholds must lie in [0,1]")
        selected = [row for row in rows if row["confidence"] >= threshold]
        result.append(
            {
                "threshold": threshold,
                "retained": len(selected),
                "coverage": len(selected) / len(rows) if rows else None,
                "accuracy": sum(
                    normalized_answer(row["answer"]) == normalized_answer(row["prediction"])
                    for row in selected
                )
                / len(selected)
                if selected
                else None,
            }
        )
    return result


def answer_errors(records):
    return [
        {
            key: row.get(key)
            for key in (
                "id",
                "patient_id",
                "question",
                "answer",
                "prediction",
                "confidence",
                "anatomy",
                "dataset",
            )
        }
        for row in closed_records(records)
        if normalized_answer(row["answer"]) != normalized_answer(row["prediction"])
    ]
