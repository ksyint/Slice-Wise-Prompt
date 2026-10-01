"""Patient-weighted answer scores and paired resampling intervals."""

from collections import defaultdict

import numpy as np

from slice_prompt.evaluation.closed import normalized_answer
from slice_prompt.evaluation.opened import overlap_metrics


def question_scores(records):
    result = []
    for row in records:
        if row.get("answer") is None:
            continue
        score = (
            float(normalized_answer(row["prediction"]) == normalized_answer(row["answer"]))
            if row["closed"]
            else overlap_metrics(row["prediction"], row["answer"], True)["token_f1"]
        )
        result.append(
            {
                "id": str(row["id"]),
                "patient_id": str(row["patient_id"]),
                "closed": bool(row["closed"]),
                "answer": row["answer"],
                "score": score,
            }
        )
    identities = [row["id"] for row in result]
    if len(set(identities)) != len(identities):
        raise ValueError("Question IDs must be unique before patient aggregation")
    return result


def patient_scores(records):
    groups = defaultdict(list)
    for row in question_scores(records):
        groups[row["patient_id"]].append(row)
    result = []
    for patient, rows in sorted(groups.items()):
        closed = [row["score"] for row in rows if row["closed"]]
        opened = [row["score"] for row in rows if not row["closed"]]
        result.append(
            {
                "patient_id": patient,
                "questions": len(rows),
                "closed_questions": len(closed),
                "open_questions": len(opened),
                "closed_accuracy": float(np.mean(closed)) if closed else None,
                "open_token_f1": float(np.mean(opened)) if opened else None,
            }
        )
    return result


def bootstrap_mean(values, repeats=2000, seed=42, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or repeats < 1 or not 0 < confidence < 1:
        raise ValueError(
            "Patient bootstrap needs finite values, positive repeats, and confidence in (0,1)"
        )
    generator = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for repeat in range(repeats):
        samples[repeat] = values[generator.integers(0, len(values), len(values))].mean()
    tail = (1 - confidence) / 2
    return {
        "patients": len(values),
        "mean": float(values.mean()),
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1 - tail)),
        "confidence": confidence,
        "repeats": repeats,
    }


def patient_report(records, repeats=2000, seed=42):
    rows = patient_scores(records)
    report = {"patients": rows}
    for name in ("closed_accuracy", "open_token_f1"):
        values = [row[name] for row in rows if row[name] is not None]
        report[name] = bootstrap_mean(values, repeats, seed) if values else None
    return report


def paired_patients(baseline, candidate, repeats=2000, seed=42):
    left = {row["id"]: row for row in question_scores(baseline)}
    right = {row["id"]: row for row in question_scores(candidate)}
    if left.keys() != right.keys():
        raise ValueError("Paired evaluation requires identical question IDs")
    groups = defaultdict(list)
    for identity in left:
        a, b = left[identity], right[identity]
        for field in ("patient_id", "closed", "answer"):
            if a[field] != b[field]:
                raise ValueError(f"Paired evaluation changed {field} for {identity}")
        groups[(a["patient_id"], a["closed"])].append(b["score"] - a["score"])
    report = {}
    for closed, name in ((True, "closed_accuracy"), (False, "open_token_f1")):
        differences = [
            float(np.mean(values)) for (_, kind), values in groups.items() if kind == closed
        ]
        report[name] = bootstrap_mean(differences, repeats, seed) if differences else None
    report["questions"] = len(left)
    return report


def grouped_patient_reports(
    records, fields=("dataset", "anatomy", "question_type"), repeats=2000, seed=42
):
    report = {}
    for field in fields:
        groups = defaultdict(list)
        for row in records:
            if row.get(field) is not None:
                groups[str(row[field])].append(row)
        report[field] = {
            key: patient_report(rows, repeats, seed) for key, rows in sorted(groups.items())
        }
    return report
