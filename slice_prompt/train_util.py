"""Batch update loop for closed classification and open answer generation."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .script_util import require_cuda
from .losses import class_balanced_focal


class TrainLoop:
    def __init__(self, model, loader, class_counts, config, device):
        device = require_cuda(device)
        self.model = model.to(device)
        self.loader = loader
        self.config = config
        self.device = device
        self.counts = torch.tensor(class_counts, device=device)
        self.optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                           lr=config['lr'], weight_decay=.02)
        self.step = 0

    def forward_backward(self, batch):
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        logits, language = self.model(batch['volumes'], batch['questions'], batch['inputs'])
        closed = batch['labels'] >= 0
        loss = language.sum() * 0
        if closed.any():
            loss = loss + class_balanced_focal(logits[closed], batch['labels'][closed], self.counts)
        if (~closed).any():
            loss = loss + F.cross_entropy(language[~closed].transpose(1, 2), batch['targets'][~closed], ignore_index=0)
        loss = loss + self.config['lora_l2'] * self.model.lora_penalty()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite VQA loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.step += 1
        return float(loss.detach())

    def run_loop(self):
        for epoch in range(self.config['epochs']):
            self.model.train()
            losses = [self.forward_backward(batch) for batch in self.loader]
            yield dict(epoch=epoch + 1, step=self.step, loss=sum(losses) / len(losses))

    def save(self, output, vocabulary, classes, history):
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=self.model.state_dict(), vocab=vocabulary.words, classes=classes,
                        config=self.config, step=self.step), destination / 'last.pt')
        (destination / 'history.json').write_text(json.dumps(history, indent=2))
