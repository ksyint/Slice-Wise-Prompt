"""Contour ablations, frozen-feature preparation, and axial occlusion runs."""

import argparse
import json
from pathlib import Path

import torch as th

from slice_prompt.commands import read_config
from slice_prompt.processing import PromptDataset, pack_rows
from slice_prompt.models.adaptation import FrozenMed3D, QwenSliceVLM
from slice_prompt.models.feature_cache import VisualFeatureCache
from slice_prompt.experiments.training import (
    model_defaults,
    data_defaults,
    training_defaults,
    require_cuda,
)
from slice_prompt.experiments.prepare_volumes import prepare_record
from slice_prompt.records.rewriting import AnatomyLexicon
from slice_prompt.records.schema import read_manifest, write_manifest
from slice_prompt.evaluation.occlusion import slice_occlusion, roi_concentration
from slice_prompt.volumes.export import write_json


def feature_cache_cli():
    parser = argparse.ArgumentParser(
        description="Precompute frozen DCFormer and Mixer features on CUDA."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="recipes/prompt_vqa.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Visual encoding batch size must be positive")
    config = dict(**model_defaults(), **data_defaults(), **training_defaults())
    config.update(read_config(args.config))
    device = require_cuda(args.device)
    dtype = th.bfloat16 if config["precision"] == "bf16" else th.float16
    visual = FrozenMed3D(config).to(device=device, dtype=dtype).eval()
    data = PromptDataset(args.manifest, config["slices"], config["image_size"])
    data.rows = read_manifest(args.manifest, prepared=True)
    cache = VisualFeatureCache(args.cache, config)
    with th.inference_mode(), th.autocast("cuda", dtype=dtype):
        for start in range(0, len(data), args.batch_size):
            rows = [data[index] for index in range(start, min(start + args.batch_size, len(data)))]
            volumes = th.stack([row["tensor"] for row in rows]).to(device)
            cache.encode(visual, rows, volumes)
    report = cache.audit()
    write_json(
        Path(args.cache) / "summary.json",
        {"records": len(data), "entries": report["count"], "total_bytes": report["total_bytes"]},
    )
    print(json.dumps({"records": len(data), "entries": report["count"]}, indent=2))


def prompt_ablation_cli():
    parser = argparse.ArgumentParser(
        description="Prepare paired color and contour-width variants with unchanged answer labels."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--colors", nargs="+", choices=("red", "blue"), default=("red", "blue"))
    parser.add_argument("--thicknesses", nargs="+", type=int, default=(1, 3))
    parser.add_argument("--window", choices=("lung", "mediastinum"), default="lung")
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    variants = []
    for color in dict.fromkeys(args.colors):
        for thickness in dict.fromkeys(args.thicknesses):
            if thickness < 1 or thickness % 2 == 0:
                raise ValueError("Ablation thickness must be an odd positive integer")
            settings = dict(
                color=color,
                thickness=thickness,
                window=args.window,
                center=None,
                width=None,
                slices=args.slices,
                sampling="uniform",
                context=0,
                seed=args.seed,
                minimum_component=0,
                largest_components=None,
                png=False,
                preview=False,
            )
            destination = output / f"{color}-{thickness}"
            destination.mkdir(exist_ok=True)
            prepared = [
                prepare_record(row, destination, settings, AnatomyLexicon())[0] for row in rows
            ]
            manifest = destination / "prepared.jsonl"
            write_manifest(manifest, prepared)
            variants.append(
                {
                    "color": color,
                    "thickness": thickness,
                    "manifest": str(manifest.resolve()),
                    "questions": len(prepared),
                }
            )
    write_json(
        output / "variants.json",
        {"source": str(Path(args.manifest).resolve()), "variants": variants},
    )
    print(json.dumps({"variants": len(variants), "questions_per_variant": len(rows)}, indent=2))


def occlusion_cli():
    parser = argparse.ArgumentParser(
        description="Measure closed-answer sensitivity to axial slices on CUDA."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replacement", choices=("mean", "zero", "gray"), default="mean")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    device = require_cuda(args.device)
    state = th.load(args.checkpoint, map_location=device, weights_only=False)
    config, classes = state["config"], state["classes"]
    model = QwenSliceVLM(config, len(classes)).to(device)
    model.restore_trainable(state["trainable"])
    model.eval()
    dataset = PromptDataset(args.manifest, config["slices"], config["image_size"])
    dataset.rows = read_manifest(args.manifest, prepared=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index in range(len(dataset)):
        row = dataset[index]
        if not row["closed"]:
            continue
        batch = pack_rows([row], model.tokenizer, classes, config["max_length"], training=False)
        batch = {
            key: value.to(device) if isinstance(value, th.Tensor) else value
            for key, value in batch.items()
        }
        result = slice_occlusion(
            model, batch, replacement=args.replacement, window=args.window, stride=args.stride
        )
        result.update(id=row["id"], patient_id=row["patient_id"])
        if row.get("roi_areas") is not None:
            from slice_prompt.volumes.selection import uniform_indices

            indices = (
                row.get("slice_indices")
                or uniform_indices(len(row["roi_areas"]), config["slices"]).tolist()
            )
            result["roi_concentration"] = roi_concentration(
                result["slice_scores"][0], [row["roi_areas"][position] for position in indices]
            )
        reports.append(result)
        if args.limit is not None and len(reports) >= args.limit:
            break
    write_json(
        output / "occlusion.json",
        {"checkpoint": str(Path(args.checkpoint).resolve()), "records": reports},
    )
    print(json.dumps({"questions": len(reports), "output": str(output)}, indent=2))


def feature_audit_cli():
    parser = argparse.ArgumentParser(
        description="Inspect cached DCFormer and Mixer token variation on CUDA."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="recipes/prompt_vqa.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from slice_prompt.models.feature_cache import cache_feature_report

    config = dict(**model_defaults(), **data_defaults(), **training_defaults())
    config.update(read_config(args.config))
    records = read_manifest(args.manifest, prepared=True)
    cache = VisualFeatureCache(args.cache, config)
    report = cache_feature_report(cache, records, require_cuda(args.device))
    write_json(args.output, report)
    print(json.dumps({"requested": report["requested"], "present": report["present"]}, indent=2))
