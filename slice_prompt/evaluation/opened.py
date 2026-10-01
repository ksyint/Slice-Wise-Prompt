"""Open-answer token overlap and length diagnostics beside corpus VQA scores."""

import re
from collections import Counter

import numpy as np


def remove_prompt_prefix(text):
    return re.sub(r"^Within the (?:red|blue) boundary,\s*", "", text.strip(), flags=re.I)


def tokens(text, strip_prefix=False):
    if strip_prefix:
        text = remove_prompt_prefix(text)
    return re.findall(r"\w+", text.casefold())


def overlap_metrics(prediction, reference, strip_prefix=False):
    predicted = tokens(prediction, strip_prefix)
    expected = tokens(reference, strip_prefix)
    intersection = sum((Counter(predicted) & Counter(expected)).values())
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(expected) if expected else 0.0
    if not predicted and not expected:
        precision = recall = 1.0
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_match": predicted == expected,
        "prediction_tokens": len(predicted),
        "reference_tokens": len(expected),
        "overlap_tokens": intersection,
    }


def longest_common_subsequence(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1 if token == other else max(current[-1], previous[index])
            )
        previous = current
    return previous[-1]


def sequence_overlap(prediction, reference):
    predicted, expected = tokens(prediction), tokens(reference)
    shared = longest_common_subsequence(predicted, expected)
    precision = shared / len(predicted) if predicted else 0.0
    recall = shared / len(expected) if expected else 0.0
    return {
        "lcs_tokens": shared,
        "lcs_precision": precision,
        "lcs_recall": recall,
        "lcs_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def repeated_ngrams(text, size=3):
    sequence = tokens(text)
    if size < 1:
        raise ValueError("N-gram size must be positive")
    grams = Counter(
        tuple(sequence[index : index + size]) for index in range(max(0, len(sequence) - size + 1))
    )
    total = sum(grams.values())
    repeated = sum(count - 1 for count in grams.values())
    return {
        "ngrams": total,
        "repeated": repeated,
        "repetition_fraction": repeated / total if total else 0.0,
    }


def open_report(records, strip_prefix=False):
    rows = [row for row in records if not row["closed"] and row.get("answer") is not None]
    if not rows:
        return {"questions": 0}
    details = []
    for row in rows:
        values = overlap_metrics(row["prediction"], row["answer"], strip_prefix)
        values.update(sequence_overlap(row["prediction"], row["answer"]))
        values["repetition"] = repeated_ngrams(row["prediction"])
        details.append(dict(id=row["id"], patient_id=row["patient_id"], **values))
    metrics = (
        "token_precision",
        "token_recall",
        "token_f1",
        "exact_match",
        "prediction_tokens",
        "reference_tokens",
        "lcs_f1",
    )
    return {
        "questions": len(rows),
        "strip_region_prefix": strip_prefix,
        "averages": {name: float(np.mean([row[name] for row in details])) for name in metrics},
        "empty_predictions": sum(not row["prediction"].strip() for row in rows),
        "length_ratio": sum(row["prediction_tokens"] for row in details)
        / max(1, sum(row["reference_tokens"] for row in details)),
        "details": details,
    }


def vocabulary_statistics(records):
    predictions = Counter()
    references = Counter()
    for row in records:
        if row["closed"] or row.get("answer") is None:
            continue
        predictions.update(tokens(row["prediction"], True))
        references.update(tokens(row["answer"], True))
    return {
        "prediction_types": len(predictions),
        "reference_types": len(references),
        "prediction_tokens": sum(predictions.values()),
        "reference_tokens": sum(references.values()),
        "prediction_only_terms": [
            {"term": term, "count": count}
            for term, count in predictions.most_common()
            if term not in references
        ],
        "frequent_prediction_terms": predictions.most_common(50),
    }
