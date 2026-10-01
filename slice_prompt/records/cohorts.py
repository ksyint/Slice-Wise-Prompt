"""Patient-disjoint cohort allocation and cross-validation manifests."""

from collections import Counter, defaultdict

import numpy as np


def patient_groups(records):
    groups = defaultdict(list)
    for row in records:
        patient = str(row["patient_id"]).strip()
        if not patient:
            raise ValueError("Patient identifiers must be nonempty")
        groups[patient].append(row)
    if not groups:
        raise ValueError("Patient allocation requires records")
    return dict(groups)


def cohort_summary(records):
    groups = patient_groups(records)
    return {
        "patients": len(groups),
        "questions": len(records),
        "closed_questions": sum(row["closed"] for row in records),
        "open_questions": sum(not row["closed"] for row in records),
        "questions_per_patient": {
            "minimum": min(map(len, groups.values())),
            "maximum": max(map(len, groups.values())),
            "mean": len(records) / len(groups),
        },
        "datasets": dict(Counter(str(row.get("dataset", "unspecified")) for row in records)),
        "anatomy": dict(Counter(str(row.get("anatomy", "unspecified")) for row in records)),
        "question_types": dict(
            Counter(str(row.get("question_type", "unspecified")) for row in records)
        ),
        "closed_answers": dict(Counter(row["answer"] for row in records if row["closed"])),
    }


def balanced_patient_folds(records, folds=5, seed=42):
    groups = patient_groups(records)
    if not 2 <= folds <= len(groups):
        raise ValueError("Fold count must lie between two and the patient count")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(groups)).tolist()
    order = sorted(shuffled, key=lambda patient: -len(groups[patient]))
    assignments = [[] for _ in range(folds)]
    questions = np.zeros(folds, dtype=int)
    closed = np.zeros(folds, dtype=int)
    target_closed = sum(row["closed"] for row in records) / len(records)
    for patient in order:
        values = groups[patient]
        count = len(values)
        closed_count = sum(row["closed"] for row in values)
        empty = [index for index in range(folds) if not assignments[index]]
        candidates = empty or list(range(folds))
        selected = min(
            candidates,
            key=lambda index: (
                questions[index],
                abs((closed[index] + closed_count) / (questions[index] + count) - target_closed),
                index,
            ),
        )
        assignments[selected].append(patient)
        questions[selected] += count
        closed[selected] += closed_count
    return assignments


def cross_validation(records, folds=5, seed=42):
    groups = patient_groups(records)
    assignments = balanced_patient_folds(records, folds, seed)
    result = []
    for index, validation_ids in enumerate(assignments):
        selected = set(validation_ids)
        train = [row for patient, rows in groups.items() if patient not in selected for row in rows]
        validation = [
            row for patient, rows in groups.items() if patient in selected for row in rows
        ]
        result.append(
            {
                "fold": index,
                "train": train,
                "validation": validation,
                "training_patients": sorted(set(groups) - selected),
                "validation_patients": sorted(selected),
            }
        )
    return result


def verify_cohorts(partitions):
    ownership = {}
    questions = {}
    reports = {}
    for split, rows in partitions.items():
        reports[split] = cohort_summary(rows)
        for row in rows:
            patient = str(row["patient_id"])
            if patient in ownership and ownership[patient] != split:
                raise ValueError(f"Patient occurs in more than one cohort: {patient}")
            ownership[patient] = split
            identifier = str(row["id"])
            if identifier in questions:
                raise ValueError(f"Question identity occurs more than once: {identifier}")
            questions[identifier] = split
    return reports


def vocabulary_coverage(training, evaluation):
    labels = {row["answer"] for row in training if row["closed"]}
    unknown = Counter(
        row["answer"] for row in evaluation if row["closed"] and row["answer"] not in labels
    )
    return {
        "training_classes": len(labels),
        "evaluation_closed": sum(row["closed"] for row in evaluation),
        "unknown_answers": dict(unknown),
        "unknown_questions": sum(unknown.values()),
    }
