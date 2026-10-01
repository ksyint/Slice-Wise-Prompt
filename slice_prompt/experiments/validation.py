"""Validation-driven training and detailed patient-level answer reports."""

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch as th
from torch.utils.data import DataLoader

from slice_prompt.commands import read_config
from slice_prompt.processing import PromptDataset, pack_rows
from slice_prompt.experiments.training import (
    model_defaults,
    data_defaults,
    training_defaults,
    create_model_and_data,
    TrainLoop,
    require_cuda,
)
from slice_prompt.records.schema import read_manifest, manifest_digest
from slice_prompt.records.cohorts import verify_cohorts, vocabulary_coverage
from slice_prompt.models.checkpoints import check_resume
from slice_prompt.models.monitor import TrainingLog
from slice_prompt.models.optimization import trainable_summary
from slice_prompt.evaluation.closed import (
    closed_report,
    confidence_report,
    abstention_curve,
    answer_errors,
)
from slice_prompt.evaluation.opened import open_report, vocabulary_statistics
from slice_prompt.evaluation.patients import (
    patient_report,
    paired_patients,
    grouped_patient_reports,
)
from slice_prompt.volumes.export import write_json


@th.no_grad()
def validate_model(model, manifest, classes, config, device, max_new_tokens=128):
    dataset = PromptDataset(manifest, config["slices"], config["image_size"])
    dataset.rows = read_manifest(manifest, prepared=True)
    names = sorted(classes, key=classes.get)
    previous_mode = model.training
    model.eval()
    predictions = []
    try:
        with th.autocast("cuda", dtype=model.dtype):
            for row in dataset:
                batch = pack_rows(
                    [row], model.tokenizer, classes, config["max_length"], training=False
                )
                batch = {
                    name: value.to(device) if isinstance(value, th.Tensor) else value
                    for name, value in batch.items()
                }
                if row["closed"]:
                    if not names:
                        raise ValueError("Closed validation records require an answer vocabulary")
                    probabilities = model(batch)["classes"].float().softmax(-1)[0]
                    selected = int(probabilities.argmax())
                    prediction = names[selected]
                    confidence = float(probabilities[selected])
                else:
                    prediction = model.generate(batch, max_new_tokens)[0]
                    confidence = None
                result = {
                    key: row[key] for key in ("id", "patient_id", "answer", "question", "closed")
                }
                result.update(prediction=prediction, confidence=confidence)
                for field in ("dataset", "anatomy", "question_type"):
                    if field in row:
                        result[field] = row[field]
                predictions.append(result)
    finally:
        model.train(previous_mode)
    return predictions


def validation_score(records, metric):
    if metric == "closed_accuracy":
        report = closed_report(records)
        score = report["accuracy"]
    elif metric == "open_token_f1":
        report = open_report(records, strip_prefix=True)
        score = report.get("averages", {}).get("token_f1")
    else:
        raise ValueError("Validation metric must be closed_accuracy or open_token_f1")
    if score is None:
        raise ValueError("Validation split has no examples for the selected checkpoint metric")
    return score


def fit_validated_cli():
    parser = argparse.ArgumentParser(
        description="Train with a patient-disjoint validation manifest."
    )
    parser.add_argument("--config", default="recipes/prompt_vqa.yaml")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric", choices=("closed_accuracy", "open_token_f1"), default="closed_accuracy"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--visual-cache")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.checkpoint:
        checkpoint_parent = Path(args.checkpoint).resolve().parent
        if checkpoint_parent not in (output, output / "best"):
            raise ValueError("Resume in the original run directory with its best checkpoint")
        if not (output / "best" / "last.pt").is_file():
            raise FileNotFoundError("Resume requires the selected best/last.pt checkpoint")
    device = require_cuda(args.device)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = dict(**model_defaults(), **data_defaults(), **training_defaults())
    config.update(read_config(args.config))
    config["named_optimizer_groups"] = True
    state = (
        th.load(args.checkpoint, map_location=device, weights_only=False)
        if args.checkpoint
        else None
    )
    if state:
        config = dict(state["config"])
        if config.get("validation_metric") != args.metric:
            raise ValueError("Validation checkpoint metric changed during resume")
    if args.visual_cache:
        config["visual_cache"] = args.visual_cache
    partitions = {
        "train": read_manifest(args.train_manifest, prepared=True),
        "validation": read_manifest(args.validation_manifest, prepared=True),
    }
    cohort_report = verify_cohorts(partitions)
    coverage = vocabulary_coverage(partitions["train"], partitions["validation"])
    if coverage["unknown_questions"]:
        raise ValueError("Validation closed answers are absent from the training vocabulary")
    validation_identity = manifest_digest(partitions["validation"])
    config["validation_manifest_digest"] = validation_identity
    config["validation_metric"] = args.metric
    model, dataset, tokenizer, classes, counts = create_model_and_data(
        config, args.train_manifest, device
    )
    data_identity = manifest_digest(dataset.rows)
    if state:
        check_resume(state, config, classes, data_identity)
        if state["config"].get("validation_manifest_digest") != validation_identity:
            raise ValueError("Validation cohort changed during resume")
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        generator=th.Generator().manual_seed(args.seed),
        collate_fn=partial(
            pack_rows, tokenizer=tokenizer, classes=classes, max_length=config["max_length"]
        ),
    )
    loop = TrainLoop(model, loader, counts, config, device)
    loop.manifest_digest = data_identity
    if state:
        loop.restore(state)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cohorts.json", cohort_report)
    write_json(output / "trainable.json", trainable_summary(model))
    history = list(state.get("history", [])) if state else []
    best = max(
        (row.get("validation_score", float("-inf")) for row in history), default=float("-inf")
    )
    log = TrainingLog(output / "epochs.jsonl", loop.start_epoch, history)
    for row in loop.run_loop():
        answers = validate_model(
            model, args.validation_manifest, classes, config, device, args.max_new_tokens
        )
        score = validation_score(answers, args.metric)
        row.update(validation_metric=args.metric, validation_score=score)
        history.append(row)
        if score > best:
            best = score
            loop.save(output / "best", tokenizer, classes, history)
            write_json(output / "best" / "validation.json", answers)
        loop.save(output, tokenizer, classes, history)
        log.append(row["monitor"] | {"epoch": row["epoch"], "validation_score": score})
        print(json.dumps(row, allow_nan=False), flush=True)
    write_json(output / "training-summary.json", log.summary())


def read_predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Prediction file is empty")
    identities = [row["id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Prediction IDs must be unique")
    return rows


def detailed_report_cli():
    parser = argparse.ArgumentParser(
        description="Report closed and open answers with patient-level uncertainty."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-metrics", action="store_true")
    args = parser.parse_args()
    rows = read_predictions(args.predictions)
    report = {
        "closed": closed_report(rows),
        "confidence": confidence_report(rows),
        "abstention": abstention_curve(rows),
        "closed_errors": answer_errors(rows),
        "open": open_report(rows),
        "open_without_region_prefix": open_report(rows, strip_prefix=True),
        "vocabulary": vocabulary_statistics(rows),
        "patients": patient_report(rows, args.bootstrap, args.seed),
        "groups": grouped_patient_reports(rows, repeats=args.bootstrap, seed=args.seed),
    }
    if args.baseline:
        report["paired"] = paired_patients(
            read_predictions(args.baseline), rows, args.bootstrap, args.seed
        )
    if args.corpus_metrics:
        from slice_prompt.experiments.training import evaluate_records

        report["corpus"] = evaluate_records(rows)
    write_json(args.output, report)
    print(json.dumps({"questions": len(rows), "output": args.output}, indent=2))


def token_budget_report(records, tokenizer, max_length, slices=32):
    if max_length < 1 or slices < 1:
        raise ValueError("Token and visual prefix budgets must be positive")
    rows = []
    for row in records:
        system = next(
            (turn["content"] for turn in row.get("conversations", []) if turn["role"] == "system"),
            "Answer the medical question using the marked region of the CT volume.",
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        question = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
        answer_tokens = len(answer) + int(tokenizer.eos_token_id is not None)
        total = len(question) + answer_tokens
        rows.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "closed": row["closed"],
                "prompt_tokens": len(question),
                "answer_tokens": answer_tokens,
                "training_tokens": total,
                "visual_prefix_tokens": slices,
                "decoder_tokens": total + slices,
                "over_budget": total > max_length,
                "remaining_generation_tokens": max_length - len(question),
            }
        )
    if not rows:
        raise ValueError("Token budgeting requires question-answer records")
    lengths = np.asarray([row["training_tokens"] for row in rows])
    return {
        "questions": len(rows),
        "maximum_length": max_length,
        "over_budget": sum(row["over_budget"] for row in rows),
        "maximum_training_tokens": int(lengths.max()),
        "minimum_training_tokens": int(lengths.min()),
        "mean_training_tokens": float(lengths.mean()),
        "percentiles": {
            str(level): float(np.percentile(lengths, level)) for level in (50, 90, 95, 99)
        },
        "records": rows,
    }


def token_audit_cli():
    parser = argparse.ArgumentParser(
        description="Measure Qwen prompt and answer token budgets before training."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="recipes/prompt_vqa.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-fit", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    config = dict(**model_defaults(), **data_defaults(), **training_defaults())
    config.update(read_config(args.config))
    tokenizer = AutoTokenizer.from_pretrained(
        config["language_model"],
        revision=config["revision"],
        cache_dir=config["cache_dir"],
        local_files_only=config["offline"],
    )
    records = read_manifest(args.manifest, prepared=True)
    report = token_budget_report(records, tokenizer, config["max_length"], config["slices"])
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    if args.require_fit and report["over_budget"]:
        raise ValueError("Some supervised examples exceed the configured Qwen context budget")


def compare_prepared_cohorts(left, right):
    first = {row["id"]: row for row in left}
    second = {row["id"]: row for row in right}
    if len(first) != len(left) or len(second) != len(right):
        raise ValueError("Prepared cohort comparisons require unique question IDs")
    shared = sorted(first.keys() & second.keys())
    changes = []
    for identity in shared:
        before, after = first[identity], second[identity]
        for field in ("patient_id", "closed"):
            if before[field] != after[field]:
                raise ValueError(f"Prepared variants changed {field} for {identity}")
        if before["closed"] and before["answer"] != after["answer"]:
            raise ValueError("Closed-answer labels differ across prepared variants")
        changes.append(
            {
                "id": identity,
                "question_changed": before["question"] != after["question"],
                "answer_changed": before["answer"] != after["answer"],
                "prompt_changed": before["prompt"] != after["prompt"],
                "slice_indices_changed": before.get("slice_indices") != after.get("slice_indices"),
                "contour_voxel_change": after.get("contour_voxels", 0)
                - before.get("contour_voxels", 0),
                "intersecting_slice_change": after.get("intersecting_slices", 0)
                - before.get("intersecting_slices", 0),
            }
        )
    return {
        "left_questions": len(left),
        "right_questions": len(right),
        "shared_questions": len(shared),
        "left_only": sorted(first.keys() - second.keys()),
        "right_only": sorted(second.keys() - first.keys()),
        "changes": changes,
    }


def compare_preparation_cli():
    parser = argparse.ArgumentParser(
        description="Compare paired prompt preparations without changing source answers."
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_prepared_cohorts(
        read_manifest(args.left, prepared=True),
        read_manifest(args.right, prepared=True),
    )
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, indent=2))
