"""Trainable module groups and mixed-precision optimization state."""

import math

import torch as th


def trainable_groups(model, config):
    groups = {}
    identities = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in identities:
            continue
        identities.add(id(parameter))
        if "lora_" in name:
            role = "lora"
        elif name.startswith(("ssa.", "mssa.")):
            role = "slice_attention"
        elif name.startswith(("bridge.", "global_context.", "distill_projection.")):
            role = "projection"
        else:
            role = "answer"
        group = groups.setdefault(role, {"params": [], "names": []})
        group["params"].append(parameter)
        group["names"].append(name)
    if not groups:
        raise ValueError("No trainable slice-prompt parameters were found")
    result, report = [], []
    multipliers = config.get("lr_multipliers", {})
    unknown = set(multipliers) - set(groups)
    if unknown:
        raise ValueError(f"Learning-rate multipliers name unknown groups: {sorted(unknown)}")
    for role, values in groups.items():
        learning_rate = config["lr"] * multipliers.get(role, 1.0)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Every learning-rate group must be finite and positive")
        decay = config["weight_decay"] if role == "lora" else 0.0
        result.append(
            {"params": values["params"], "lr": learning_rate, "weight_decay": decay, "name": role}
        )
        report.append(
            {
                "name": role,
                "parameters": sum(parameter.numel() for parameter in values["params"]),
                "learning_rate": learning_rate,
                "weight_decay": decay,
                "tensors": values["names"],
            }
        )
    return result, report


def build_optimizer(model, config):
    groups, report = trainable_groups(model, config)
    optimizer = th.optim.AdamW(groups, betas=(0.9, 0.98))
    return optimizer, report


def make_schedule(optimizer, config, loader_length):
    total = math.ceil(loader_length / config["accumulation_steps"]) * config["epochs"]
    if total < 1:
        raise ValueError("Training schedule requires at least one optimizer update")
    warmup = config["warmup_steps"]
    if warmup < 0:
        raise ValueError("Warmup updates must be nonnegative")

    def scale(update):
        if update < warmup:
            return (update + 1) / max(1, warmup)
        progress = min(1.0, (update - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return th.optim.lr_scheduler.LambdaLR(optimizer, scale)


def optimizer_update(model, optimizer, scaler, scheduler, clip=1.0):
    if clip <= 0:
        raise ValueError("Gradient clipping must be positive")
    scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = th.nn.utils.clip_grad_norm_(parameters, clip)
    if not scaler.is_enabled() and not th.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Slice prompt gradients contain nonfinite values")
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= previous
    if success:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"updated": success, "gradient_norm": float(norm), "loss_scale": scaler.get_scale()}


def trainable_summary(model):
    rows = []
    frozen = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rows.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "parameters": parameter.numel(),
                    "dtype": str(parameter.dtype),
                }
            )
        else:
            frozen += parameter.numel()
    trainable = sum(row["parameters"] for row in rows)
    return {
        "trainable": trainable,
        "frozen": frozen,
        "trainable_fraction": trainable / (trainable + frozen),
        "tensors": rows,
    }
