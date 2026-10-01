"""Repeated source volumes and contradictory question-answer records."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize_question(question):
    text = question.casefold().strip()
    text = re.sub(
        r"^(within the (red|blue)-outlined area of the ct volume,|inside the (red|blue) boundary,)\s*",
        "",
        text,
    )
    return " ".join(re.findall(r"\w+", text))


def source_fingerprint(path, content=False):
    path = Path(path).resolve()
    if not content:
        return str(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record_key(row):
    values = {
        "volume": str(Path(row["volume"]).resolve()),
        "mask": str(Path(row["mask"]).resolve()),
        "labels": sorted(row.get("mask_labels", [])),
        "question": normalize_question(row["question"]),
        "closed": bool(row["closed"]),
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def question_duplicates(records):
    groups = defaultdict(list)
    for row in records:
        groups[record_key(row)].append(row)
    return [
        {
            "sha256": key,
            "ids": [row["id"] for row in rows],
            "patients": sorted({row["patient_id"] for row in rows}),
            "answers": sorted({row["answer"] for row in rows}),
            "conflicting_answers": len({row["answer"] for row in rows}) > 1,
        }
        for key, rows in sorted(groups.items())
        if len(rows) > 1
    ]


def shared_volumes(partitions, content=False):
    cache = {}
    locations = defaultdict(list)
    for split, rows in partitions.items():
        for row in rows:
            path = str(Path(row["volume"]).resolve())
            if path not in cache:
                cache[path] = source_fingerprint(path, content)
            locations[cache[path]].append(
                {
                    "split": split,
                    "patient_id": row["patient_id"],
                    "id": row["id"],
                    "volume": path,
                }
            )
    result = []
    for identity, rows in locations.items():
        splits = sorted({row["split"] for row in rows})
        patients = sorted({row["patient_id"] for row in rows})
        if len(splits) > 1 or len(patients) > 1:
            result.append(
                {
                    "identity": identity,
                    "splits": splits,
                    "patient_ids": patients,
                    "records": rows,
                }
            )
    return result


def deduplicate_questions(records):
    retained = {}
    removed = []
    for row in records:
        key = record_key(row)
        if key in retained:
            previous = retained[key]
            if previous["answer"] != row["answer"] or previous["patient_id"] != row["patient_id"]:
                raise ValueError(
                    "Duplicate questions have incompatible labels or patient identities"
                )
            removed.append({"id": row["id"], "retained": previous["id"]})
        else:
            retained[key] = row
    return list(retained.values()), removed


def duplicate_audit(partitions, content=False):
    return {
        "volume_identity": "sha256" if content else "resolved-path",
        "shared_volumes": shared_volumes(partitions, content),
        "questions": {split: question_duplicates(rows) for split, rows in partitions.items()},
    }
