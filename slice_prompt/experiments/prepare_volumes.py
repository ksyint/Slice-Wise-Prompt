"""Prepare region prompts with geometry, intensity, and axial sampling records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from slice_prompt.processing import load_pair, render_prompt, patient_split
from slice_prompt.records.schema import read_manifest, write_manifest
from slice_prompt.records.rewriting import AnatomyLexicon, rewrite_record, rewriting_report
from slice_prompt.records.cohorts import cohort_summary, cross_validation, verify_cohorts
from slice_prompt.records.duplicates import duplicate_audit, deduplicate_questions
from slice_prompt.volumes.geometry import read_grid
from slice_prompt.volumes.intensity import resolve_window
from slice_prompt.volumes.regions import select_region, retain_components
from slice_prompt.volumes.regions import contour_topology, mask_agreement
from slice_prompt.volumes.selection import sampling_plan, selection_ablation
from slice_prompt.volumes.quality import (
    volume_quality,
    aggregate_quality,
    prompt_array_report,
    contour_agreement,
)
from slice_prompt.volumes.export import export_prompt, write_json


def prepare_record(row, output, settings, lexicon):
    volume, labels = load_pair(row["volume"], row["mask"])
    region, selected = select_region(labels, row.get("mask_labels"))
    original_region = region.copy()
    cleaned = None
    if settings["minimum_component"] or settings["largest_components"]:
        region, cleaned = retain_components(
            region,
            settings["minimum_component"],
            settings["largest_components"],
        )
    grid = read_grid(row["volume"], settings.get("npy_spacing"))
    window = resolve_window(settings["window"], settings["center"], settings["width"])
    quality = volume_quality(volume, region, grid, window)
    rgb, contours = render_prompt(
        volume,
        region,
        settings["color"],
        settings["thickness"],
        window.center,
        window.width,
    )
    prompt = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2), dtype=np.float32)
    plan = sampling_plan(
        region, settings["slices"], settings["sampling"], settings["context"], settings["seed"]
    )
    quality["prompt"] = prompt_array_report(prompt)
    quality["contour"] = contour_agreement(prompt, contours, settings["color"])
    quality["selection"] = selected
    quality["component_filter"] = cleaned
    quality["contour_topology"] = contour_topology(region, contours)
    if cleaned is not None:
        quality["component_filter_agreement"] = mask_agreement(
            original_region, region, grid.spacing
        )
    quality["sampling"] = plan
    digest = hashlib.sha256(str(row["id"]).encode()).hexdigest()[:20]
    artifact = export_prompt(
        Path(output) / "volumes" / digest,
        prompt,
        contours,
        quality,
        png=settings["png"],
        preview=settings["preview"],
    )
    prepared = dict(row, prompt=artifact["prompt"], contour=artifact["contour"])
    prepared["slice_indices"] = plan["indices"]
    prepared["roi_areas"] = region.sum((1, 2)).tolist()
    prepared["contour_voxels"] = int(contours.sum())
    prepared["intersecting_slices"] = int(np.any(contours, axis=(1, 2)).sum())
    prepared["preparation_metadata"] = str(
        (Path(output) / "volumes" / digest / "prepared.json").resolve()
    )
    prepared = rewrite_record(prepared, settings["color"], settings["thickness"], lexicon)
    return prepared, quality


def prepare_volumes_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--color", choices=("red", "blue"), default="red")
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--window", choices=("lung", "mediastinum", "bone", "brain"))
    parser.add_argument("--center", type=float)
    parser.add_argument("--width", type=float)
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--sampling", choices=("uniform", "region", "coverage"), default="uniform")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--npy-spacing", type=float, nargs=3)
    parser.add_argument("--minimum-component", type=int, default=0)
    parser.add_argument("--largest-components", type=int)
    parser.add_argument("--lexicon")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()
    settings = vars(args).copy()
    settings["preview"] = not args.no_preview
    resolve_window(args.window, args.center, args.width)
    rows = read_manifest(args.manifest)
    removed = []
    if args.deduplicate:
        rows, removed = deduplicate_questions(rows)
    lexicon = AnatomyLexicon.read(args.lexicon) if args.lexicon else AnatomyLexicon()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared, reports = [], []
    for row in rows:
        record, report = prepare_record(row, output, settings, lexicon)
        prepared.append(record)
        reports.append(report)
    partitions = (
        patient_split(prepared, args.seed, args.fractions) if args.split else {"prepared": prepared}
    )
    if any(not records for records in partitions.values()):
        raise ValueError("Patient splitting produced an empty partition")
    cohorts = verify_cohorts(partitions)
    for split, records in partitions.items():
        write_manifest(output / f"{split}.jsonl", records)
    result = {
        "format": "slice-detailed-preparation-v1",
        "settings": settings,
        "quality": aggregate_quality(reports),
        "cohorts": cohorts,
        "duplicates": duplicate_audit(partitions),
        "rewriting": rewriting_report(rows, prepared),
        "removed_questions": removed,
    }
    write_json(output / "preparation.json", result)
    print(json.dumps(result["quality"], indent=2))


def cohort_cli():
    parser = argparse.ArgumentParser(
        description="Create patient-disjoint cross-validation folds from prepared questions."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--content-hashes", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(args.manifest, prepared=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for fold in cross_validation(rows, args.folds, args.seed):
        directory = output / f"fold-{fold['fold']}"
        directory.mkdir(exist_ok=True)
        partitions = {key: fold[key] for key in ("train", "validation")}
        verify_cohorts(partitions)
        for split, records in partitions.items():
            write_manifest(directory / f"{split}.jsonl", records)
        audit = duplicate_audit(partitions, args.content_hashes)
        if audit["shared_volumes"]:
            raise ValueError("Different patient identifiers refer to shared source volumes")
        result.append(
            {
                "fold": fold["fold"],
                "train": cohort_summary(fold["train"]),
                "validation": cohort_summary(fold["validation"]),
            }
        )
    write_json(output / "folds.json", {"seed": args.seed, "folds": result})
    print(
        json.dumps({"folds": len(result), "patients": cohort_summary(rows)["patients"]}, indent=2)
    )


def sampling_audit_cli():
    parser = argparse.ArgumentParser(
        description="Compare axial sampling coverage without running a model."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = []
    for row in read_manifest(args.manifest):
        _, mask = load_pair(row["volume"], row["mask"])
        region, _ = select_region(mask, row.get("mask_labels"))
        reports.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "settings": selection_ablation(region, args.counts),
            }
        )
    write_json(args.output, {"records": reports})
    print(json.dumps({"records": len(reports), "output": args.output}, indent=2))
