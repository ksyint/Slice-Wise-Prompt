"""Supervised answer objectives with separately recorded loss components."""

import math

import torch as th
import torch.nn.functional as F

from slice_prompt.models.adaptation import class_balanced_focal, sigmoid_distillation


def answer_cross_entropy(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Answer logits and labels must align by batch and token")
    shifted = labels[:, 1:]
    tokens = int((shifted != -100).sum())
    if tokens == 0:
        return logits.sum() * 0, 0
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
    )
    return loss, tokens


def closed_answer_loss(logits, labels, counts, gamma=2.0, beta=0.999):
    if logits.ndim != 2 or labels.shape != (len(logits),):
        raise ValueError("Closed-answer predictions require aligned label IDs")
    selected = labels >= 0
    count = int(selected.sum())
    if count == 0:
        return logits.sum() * 0, 0
    if counts.shape != (logits.shape[1],):
        raise ValueError("Class-frequency vector differs from the answer vocabulary")
    if not 0 <= beta < 1 or gamma < 0:
        raise ValueError("Focal loss requires beta in [0,1) and gamma >= 0")
    loss = class_balanced_focal(logits[selected].float(), labels[selected], counts, gamma, beta)
    return loss, count


class AnswerObjective:
    def __init__(self, counts, config):
        self.counts = counts
        self.weights = dict(closed=1.0, open=1.0, distillation=1.0, lora=config["lora_l2"])
        self.weights.update(config.get("objective_weights", {}))
        if set(self.weights) != {"closed", "open", "distillation", "lora"}:
            raise ValueError("Unknown answer objective weight")
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Objective weights must be finite and nonnegative")
        self.gamma = float(config.get("focal_gamma", 2.0))
        self.beta = float(config.get("focal_beta", 0.999))

    def __call__(self, model, output, batch, teacher=None):
        closed, closed_count = closed_answer_loss(
            output["classes"], batch["class_labels"], self.counts, self.gamma, self.beta
        )
        opened, token_count = answer_cross_entropy(output["logits"], batch["labels"])
        lora = model.lora_penalty()
        distillation = output["representation"].sum() * 0
        if teacher is not None and self.weights["distillation"]:
            targets = teacher(output["native"], batch["prompts"])
            projected = model.distill_projection(output["representation"])
            distillation = sigmoid_distillation(
                projected.float(), targets, model.log_scale, model.log_bias
            )
        parts = {"closed": closed, "open": opened, "lora": lora, "distillation": distillation}
        loss = sum(self.weights[name] * value for name, value in parts.items())
        if not th.isfinite(loss):
            raise RuntimeError("Nonfinite slice prompting objective")
        measurements = {name: float(value.detach()) for name, value in parts.items()}
        measurements.update(
            total=float(loss.detach()),
            closed_questions=closed_count,
            answer_tokens=token_count,
            questions=len(batch["input_ids"]),
        )
        return loss, measurements


class LossAccumulator:
    def __init__(self):
        self.totals = {}
        self.weights = {}
        self.batches = 0
        self.questions = 0
        self.closed = 0
        self.tokens = 0

    def update(self, values):
        self.batches += 1
        self.questions += values["questions"]
        self.closed += values["closed_questions"]
        self.tokens += values["answer_tokens"]
        for key in ("total", "closed", "open", "distillation", "lora"):
            weight = (
                values["closed_questions"]
                if key == "closed"
                else values["answer_tokens"]
                if key == "open"
                else values["questions"]
            )
            self.totals[key] = self.totals.get(key, 0.0) + values[key] * weight
            self.weights[key] = self.weights.get(key, 0) + weight

    def result(self):
        return {
            "batches": self.batches,
            "questions": self.questions,
            "closed_questions": self.closed,
            "answer_tokens": self.tokens,
            "losses": {
                key: value / self.weights[key] if self.weights[key] else None
                for key, value in self.totals.items()
            },
        }
