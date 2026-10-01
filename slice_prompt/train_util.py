"""Mixed-precision LoRA optimization followed by compact trainable checkpoints."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .script_util import require_cuda
from .losses import class_balanced_focal
from .networks.pretrained.teacher import Med3DTeacher, sigmoid_distillation


class TrainLoop:
    def __init__(self, model, loader, class_counts, config, device):
        self.device = require_cuda(device)
        self.model, self.loader, self.config = model, loader, config
        self.counts = torch.tensor(class_counts, device=device)
        lora, other = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (lora if 'lora_' in name else other).append(parameter)
        self.optimizer = torch.optim.AdamW([{'params': lora, 'weight_decay': config['weight_decay']},
                     {'params': other, 'weight_decay': 0.0}], lr=config['lr'], betas=(.9, .98))
        total = math.ceil(len(loader) / config['accumulation_steps']) * config['epochs']
        def factor(step):
            warmup = config['warmup_steps']
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16')
        self.step, self.start_epoch = 0, 0
        self.teacher = None

    def restore(self, state):
        self.model.restore_trainable(state['trainable'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.step, self.start_epoch = state['step'], state['epoch']

    def forward_loss(self, batch):
        batch = {name: value.to(self.device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
        with torch.autocast('cuda', dtype=self.model.dtype):
            output = self.model(batch)
            labels = batch['class_labels']
            closed = labels >= 0
            loss = output['classes'].sum() * 0
            if closed.any():
                loss = loss + class_balanced_focal(output['classes'][closed].float(), labels[closed], self.counts)
            shifted = batch['labels'][:, 1:]
            if (shifted != -100).any():
                loss = loss + F.cross_entropy(output['logits'][:, :-1].float().reshape(-1, output['logits'].shape[-1]),
                                              shifted.reshape(-1), ignore_index=-100)
            loss = loss + self.config['lora_l2'] * self.model.lora_penalty()
            if self.teacher is not None:
                targets = self.teacher(output['native'], batch['prompts'])
                projected = self.model.distill_projection(output['representation'])
                loss = loss + sigmoid_distillation(projected.float(), targets, self.model.log_scale, self.model.log_bias)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite VQA objective.')
        return loss

    def run_loop(self):
        for epoch in range(self.start_epoch, self.config['epochs']):
            if epoch < self.config['distillation_epochs'] and self.teacher is None:
                self.teacher = Med3DTeacher(self.config, self.device)
            if epoch >= self.config['distillation_epochs'] and self.teacher is not None:
                self.teacher = None
                torch.cuda.empty_cache()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            accumulation = self.config['accumulation_steps']
            for index, batch in enumerate(self.loader):
                loss = self.forward_loss(batch)
                block_size = min(accumulation, len(self.loader) - (index // accumulation) * accumulation)
                self.scaler.scale(loss / block_size).backward()
                losses.append(float(loss.detach()))
                if (index + 1) % accumulation == 0 or index + 1 == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
            yield dict(epoch=epoch + 1, step=self.step, loss=sum(losses) / len(losses), lr=self.scheduler.get_last_lr()[0])

    def save(self, output, tokenizer, classes, history):
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(dict(trainable=self.model.trainable_state(), classes=classes, config=self.config,
                        optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                        scaler=self.scaler.state_dict(), epoch=history[-1]['epoch'], step=self.step), destination / 'last.pt')
        tokenizer.save_pretrained(destination / 'tokenizer')
        (destination / 'history.json').write_text(json.dumps(history, indent=2))
