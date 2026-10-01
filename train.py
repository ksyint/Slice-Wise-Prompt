import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from models.model import SlicePromptVLM, class_balanced_focal
from utils.data import PromptDataset, Vocabulary, collate, make_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--manifest')
    parser.add_argument('--output', default='outputs/default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_demo(output / 'synthetic', args.seed) if args.smoke else args.manifest
    if not manifest:
        raise ValueError('Use --manifest or --smoke.')
    data = PromptDataset(manifest, **config['data'])
    vocab = Vocabulary().fit(data.rows)
    counts = Counter(row['answer'] for row in data.rows if row.get('closed', False))
    classes = {name: index for index, name in enumerate(sorted(counts))}
    class_counts = torch.tensor([counts[name] for name in classes], device=args.device)
    model = SlicePromptVLM(len(vocab.words), max(1, len(classes)), **config['model']).to(args.device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config['train']['lr'], weight_decay=.02)
    loader = DataLoader(data, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=partial(collate, vocab=vocab, classes=classes))
    history = []
    for epoch in range(2 if args.smoke else config['train']['epochs']):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            logits, language = model(batch['volumes'], batch['questions'], batch['inputs'])
            closed = batch['labels'] >= 0
            loss = language.sum() * 0
            if closed.any():
                loss = loss + class_balanced_focal(logits[closed], batch['labels'][closed], class_counts)
            if (~closed).any():
                loss = loss + F.cross_entropy(language[~closed].transpose(1, 2), batch['targets'][~closed], ignore_index=0)
            loss = loss + config['loss']['lora_l2'] * model.lora_penalty()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses)}
        history.append(record)
        print(json.dumps(record), flush=True)
    torch.save({'model': model.state_dict(), 'vocab': vocab.words, 'classes': classes, 'config': config}, output / 'last.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
