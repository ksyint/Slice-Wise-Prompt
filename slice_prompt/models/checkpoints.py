"""Atomic trainable checkpoints, reproducible epoch resumes, and run metadata."""

import hashlib
import json
import math
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from slice_prompt.records.vocabulary import AnswerVocabulary


def validate_training_config(config):
    positive = (
        "slices",
        "image_size",
        "max_length",
        "rank",
        "heads",
        "fusion_layers",
        "epochs",
        "batch_size",
        "accumulation_steps",
    )
    for name in positive:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("workers", "warmup_steps", "distillation_epochs"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if config["precision"] not in ("fp16", "bf16"):
        raise ValueError("precision must be fp16 or bf16")
    for name in ("lr", "weight_decay", "lora_l2"):
        value = config[name]
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"Invalid optimization value for {name}")
    if config["distillation_epochs"] > config["epochs"]:
        raise ValueError("Distillation epochs cannot exceed the training schedule")
    if config["fusion_layers"] > 2:
        raise ValueError("Slice fusion is attached to the first two decoder blocks")
    if not isinstance(config["offline"], bool):
        raise ValueError("offline must be a Boolean value")
    return dict(config)


def capture_random_state(loader=None):
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "gaussian": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }
    if loader is not None and loader.generator is not None:
        result["loader"] = loader.generator.get_state()
    return result


def restore_random_state(state, loader=None):
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["gaussian"],
            numpy_state["cached_gaussian"],
        )
    )
    if "loader" in state:
        if loader is None or loader.generator is None:
            raise ValueError("Checkpoint includes a shuffle generator but the DataLoader has none")
        loader.generator.set_state(state["loader"].cpu())


def checkpoint_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def merge_history(previous, current):
    by_epoch = {}
    for row in [*previous, *current]:
        epoch = row["epoch"]
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Training history epochs must be positive integers")
        if epoch in by_epoch and by_epoch[epoch] != row:
            raise ValueError(f"Two different histories exist for epoch {epoch}")
        by_epoch[epoch] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def save_training_checkpoint(loop, output, tokenizer, classes, history):
    if not history:
        raise ValueError("Cannot save a checkpoint without completed epoch statistics")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.json"
    complete = merge_history([], history)
    if [row["epoch"] for row in complete] != list(range(1, complete[-1]["epoch"] + 1)):
        raise ValueError("Checkpoint history must cover every completed epoch")
    counts = [int(value) for value in loop.counts.detach().cpu().tolist()]
    vocabulary = AnswerVocabulary(classes, counts)
    state = dict(
        trainable=loop.model.trainable_state(),
        classes=classes,
        config=loop.config,
        optimizer=loop.optimizer.state_dict(),
        scheduler=loop.scheduler.state_dict(),
        scaler=loop.scaler.state_dict(),
        epoch=history[-1]["epoch"],
        step=loop.step,
        format_version=2,
        vocabulary=vocabulary.as_dict(),
        random_state=capture_random_state(loop.loader),
        history=complete,
    )
    if hasattr(loop, "manifest_digest"):
        state["manifest_digest"] = loop.manifest_digest
    atomic_torch_save(state, output / "last.pt")
    tokenizer.save_pretrained(output / "tokenizer")
    vocabulary.save(output / "vocabulary.json")
    atomic_json(history_path, complete)
    counts = {name: value.numel() for name, value in state["trainable"].items()}
    atomic_json(
        output / "checkpoint.json",
        {
            "format": "slice-checkpoint-v2",
            "epoch": state["epoch"],
            "step": state["step"],
            "classes": len(classes),
            "trainable_tensors": len(counts),
            "trainable_parameters": sum(counts.values()),
            "config": loop.config,
            "vocabulary_digest": vocabulary.digest(),
        },
    )


def check_resume(state, config, classes, manifest_digest=None):
    protected = (
        "language_model",
        "vision_source",
        "vision_revision",
        "revision",
        "rank",
        "heads",
        "fusion_layers",
        "slices",
        "image_size",
        "max_length",
        "precision",
    )
    for field in protected:
        if state["config"][field] != config[field]:
            raise ValueError(f"Resume setting differs from the checkpoint: {field}")
    if state["classes"] != classes:
        raise ValueError("Resume requires the saved closed-answer class mapping")
    if manifest_digest and state.get("manifest_digest") not in (None, manifest_digest):
        raise ValueError("Resume manifest records differ from the saved training data")
    if config["epochs"] <= state["epoch"]:
        raise ValueError("Total epochs must exceed the completed checkpoint epoch")
    return state


def checkpoint_cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--verify-file", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory)
    report = json.loads((directory / "checkpoint.json").read_text())
    vocabulary = AnswerVocabulary.load(directory / "vocabulary.json")
    if vocabulary.digest() != report["vocabulary_digest"]:
        raise ValueError("Vocabulary metadata does not match the checkpoint manifest")
    if args.verify_file:
        report["weights"] = checkpoint_identity(directory / "last.pt")
    print(json.dumps(report, indent=2))


def tensor_file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def export_trainable_bundle(checkpoint, destination, maximum_shard_bytes=256 * 1024 * 1024):
    from safetensors.torch import save_file

    if maximum_shard_bytes < 1:
        raise ValueError("Adapter shard budget must be positive")
    source = Path(checkpoint).resolve()
    output = Path(destination).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Portable adapter destination must be empty")
    state = torch.load(source, map_location="cpu", weights_only=False)
    config = validate_training_config(state["config"])
    vocabulary = AnswerVocabulary.from_checkpoint(state)
    tensors = state["trainable"]
    if not tensors or any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError("Training checkpoint does not contain named trainable tensors")
    output.mkdir(parents=True, exist_ok=True)
    groups, pending, size = [], {}, 0
    for name, value in sorted(tensors.items()):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid trainable tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        required = tensor.numel() * tensor.element_size()
        if pending and size + required > maximum_shard_bytes:
            groups.append(pending)
            pending, size = {}, 0
        pending[name] = tensor
        size += required
    if pending:
        groups.append(pending)
    shards = []
    weight_map = {}
    for index, group in enumerate(groups, 1):
        filename = f"adapter-{index:05d}-of-{len(groups):05d}.safetensors"
        path = output / filename
        save_file(group, str(path))
        shards.append(
            {
                "file": filename,
                "sha256": tensor_file_digest(path),
                "bytes": path.stat().st_size,
                "tensors": len(group),
            }
        )
        for name, value in group.items():
            weight_map[name] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "elements": value.numel(),
            }
    tokenizer_source = source.parent / "tokenizer"
    tokenizer_files = []
    if tokenizer_source.is_dir():
        import shutil

        for path in sorted(tokenizer_source.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError("Portable tokenizer files cannot be symbolic links")
            relative = path.relative_to(tokenizer_source)
            target = output / "tokenizer" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            tokenizer_files.append(
                {
                    "file": target.relative_to(output).as_posix(),
                    "sha256": tensor_file_digest(target),
                }
            )
    vocabulary.save(output / "vocabulary.json")
    metadata = {
        "format": "slice-trainable-bundle-v1",
        "config": config,
        "epoch": state["epoch"],
        "step": state["step"],
        "classes": state["classes"],
        "vocabulary": vocabulary.as_dict(),
        "manifest_digest": state.get("manifest_digest"),
        "shards": shards,
        "weight_map": weight_map,
        "tokenizer_files": tokenizer_files,
        "source_sha256": tensor_file_digest(source),
        "trainable_parameters": sum(value.numel() for value in tensors.values()),
    }
    atomic_json(output / "adapter.json", metadata)
    return metadata


def safe_bundle_member(root, filename):
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Adapter bundle members must use paths inside the bundle")
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Adapter bundle member escapes its directory")
    return path


def inspect_trainable_bundle(directory, verify=True):
    root = Path(directory)
    metadata = json.loads((root / "adapter.json").read_text())
    if metadata.get("format") != "slice-trainable-bundle-v1":
        raise ValueError("Unsupported trainable adapter bundle")
    validate_training_config(metadata["config"])
    vocabulary = AnswerVocabulary.from_checkpoint(metadata)
    saved = AnswerVocabulary.load(root / "vocabulary.json")
    if vocabulary.digest() != saved.digest():
        raise ValueError("Portable answer vocabulary differs from adapter metadata")
    filenames = {entry["file"] for entry in metadata["shards"]}
    if len(filenames) != len(metadata["shards"]):
        raise ValueError("Adapter bundle lists a shard more than once")
    for entry in metadata["shards"]:
        path = safe_bundle_member(root, entry["file"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("Adapter tensor shard size differs from metadata")
        if verify and tensor_file_digest(path) != entry["sha256"]:
            raise ValueError("Adapter tensor shard checksum differs from metadata")
    for entry in metadata["tokenizer_files"]:
        path = safe_bundle_member(root, entry["file"])
        if verify and tensor_file_digest(path) != entry["sha256"]:
            raise ValueError("Portable tokenizer checksum differs from metadata")
    if any(value["file"] not in filenames for value in metadata["weight_map"].values()):
        raise ValueError("Adapter weight map refers to a missing shard")
    return metadata


def read_inference_checkpoint(path, device="cuda"):
    path = Path(path)
    if not path.is_dir():
        return torch.load(path, map_location=device, weights_only=False)
    from safetensors.torch import load_file

    metadata = inspect_trainable_bundle(path)
    tensors = {}
    for shard in metadata["shards"]:
        loaded = load_file(str(safe_bundle_member(path, shard["file"])), device=str(device))
        if len(loaded) != shard["tensors"]:
            raise ValueError("Adapter shard tensor count differs from metadata")
        for name, value in loaded.items():
            if name in tensors or name not in metadata["weight_map"]:
                raise ValueError("Adapter shards contain duplicate or unknown tensor names")
            expected = metadata["weight_map"][name]
            if (
                expected["file"] != shard["file"]
                or list(value.shape) != expected["shape"]
                or str(value.dtype) != expected["dtype"]
            ):
                raise ValueError(f"Adapter tensor differs from its saved specification: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite adapter tensor: {name}")
            tensors[name] = value
    if set(tensors) != set(metadata["weight_map"]):
        raise ValueError("Portable adapter is missing trainable tensors")
    return {
        "trainable": tensors,
        "config": metadata["config"],
        "classes": metadata["classes"],
        "vocabulary": metadata["vocabulary"],
        "epoch": metadata["epoch"],
        "step": metadata["step"],
    }


def bundle_checkpoint_identity(path):
    path = Path(path)
    if not path.is_dir():
        return checkpoint_identity(path)
    metadata = inspect_trainable_bundle(path)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return {
        "path": str(path.resolve()),
        "format": metadata["format"],
        "sha256": digest,
        "trainable_parameters": metadata["trainable_parameters"],
        "epoch": metadata["epoch"],
    }


def export_adapter_cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export trainable slice adapters into checked safetensors shards."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-mb", type=int, default=256)
    args = parser.parse_args()
    result = export_trainable_bundle(args.checkpoint, args.output, args.shard_mb * 1024 * 1024)
    print(
        json.dumps(
            {
                "parameters": result["trainable_parameters"],
                "shards": len(result["shards"]),
                "output": args.output,
            },
            indent=2,
        )
    )
