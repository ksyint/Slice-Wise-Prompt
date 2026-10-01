"""Per-epoch loss, gradient, and CUDA memory records."""

import json
import time
from pathlib import Path

import torch as th

from slice_prompt.models.objectives import LossAccumulator


class EpochMonitor:
    def __init__(self, device, epoch):
        self.device = th.device(device)
        if self.device.type != "cuda":
            raise ValueError("Training monitoring requires CUDA")
        self.epoch = int(epoch)
        self.losses = LossAccumulator()
        self.updates = 0
        self.skipped = 0
        self.gradients = []
        self.started = None

    def start(self):
        th.cuda.synchronize(self.device)
        th.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    def record_loss(self, values):
        self.losses.update(values)

    def record_update(self, update):
        if update["updated"]:
            self.updates += 1
            self.gradients.append(update["gradient_norm"])
        else:
            self.skipped += 1

    def finish(self):
        if self.started is None:
            raise RuntimeError("Epoch monitoring has not started")
        th.cuda.synchronize(self.device)
        duration = time.perf_counter() - self.started
        return {
            "epoch": self.epoch,
            **self.losses.result(),
            "optimizer_updates": self.updates,
            "skipped_updates": self.skipped,
            "mean_gradient_norm": sum(self.gradients) / len(self.gradients)
            if self.gradients
            else None,
            "maximum_gradient_norm": max(self.gradients) if self.gradients else None,
            "seconds": duration,
            "questions_per_second": self.losses.questions / max(duration, 1e-9),
            "peak_allocated_bytes": th.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": th.cuda.max_memory_reserved(self.device),
        }


def module_gradient_report(model):
    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module = name.split(".")[0]
        record = rows.setdefault(
            module, {"parameters": 0, "with_gradient": 0, "squared_norm": 0.0, "nonfinite": 0}
        )
        record["parameters"] += parameter.numel()
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        record["with_gradient"] += parameter.numel()
        record["squared_norm"] += float(gradient.square().sum())
        record["nonfinite"] += int((~th.isfinite(gradient)).sum())
    return [
        dict(module=module, norm=record.pop("squared_norm") ** 0.5, **record)
        for module, record in sorted(rows.items())
    ]


class TrainingLog:
    def __init__(self, path, resume_epoch=0, history=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if resume_epoch and history is not None:
            entries = list(history or [])
            if len(entries) != resume_epoch or any("monitor" not in row for row in entries):
                raise ValueError(
                    "Resume needs the original training log or checkpoint epoch monitoring"
                )
            self.rows = [
                row["monitor"]
                | {"epoch": row["epoch"], "validation_score": row["validation_score"]}
                for row in entries
            ]
            if [row["epoch"] for row in self.rows] != list(range(1, resume_epoch + 1)):
                raise ValueError("Checkpoint history does not cover the resumed epochs")
            self.path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows)
            )
            return
        if self.path.exists():
            if resume_epoch == 0:
                raise FileExistsError("Training log already exists, use checkpoint resume")
            self.rows = [
                json.loads(line) for line in self.path.read_text().splitlines() if line.strip()
            ]
            self.rows = [row for row in self.rows if row["epoch"] <= resume_epoch]
            if [row["epoch"] for row in self.rows] != list(range(1, resume_epoch + 1)):
                raise ValueError("Training log does not cover the resumed epochs")
            self.path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows)
            )

    def append(self, values):
        import os

        expected = self.rows[-1]["epoch"] + 1 if self.rows else 1
        if values["epoch"] != expected:
            raise ValueError("Training log epochs must increase consecutively")
        with self.path.open("a") as stream:
            stream.write(json.dumps(values, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(values)

    def summary(self):
        return {
            "epochs": len(self.rows),
            "seconds": sum(row["seconds"] for row in self.rows),
            "questions": sum(row["questions"] for row in self.rows),
            "optimizer_updates": sum(row["optimizer_updates"] for row in self.rows),
            "skipped_updates": sum(row["skipped_updates"] for row in self.rows),
            "peak_allocated_bytes": max(
                (row["peak_allocated_bytes"] for row in self.rows), default=0
            ),
        }
