"""Restore pretrained backbones and learned slice adapters for free-text/closed VQA."""
import argparse
import json
from pathlib import Path

import torch

from slice_prompt.script_util import require_cuda
from slice_prompt.networks.pretrained.qwen import QwenSliceVLM
from slice_prompt.data.volumes import PromptDataset
from slice_prompt.data.text.packing import pack_rows
from slice_prompt.evaluation.metrics import evaluate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='predictions.jsonl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--language-model', help='Local Qwen directory override.')
    parser.add_argument('--vision-source', help='Local Med3DVLM directory override.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    device = require_cuda(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state['config'].copy()
    for key in ('cache_dir', 'language_model', 'vision_source'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.offline:
        config['offline'] = True
    classes = state['classes']
    model = QwenSliceVLM(config, len(classes)).to(device)
    model.restore_trainable(state['trainable'])
    model.eval()
    data = PromptDataset(args.manifest, slices=config['slices'], image_size=config['image_size'])
    names = sorted(classes, key=classes.get)
    records = []
    with torch.no_grad(), torch.autocast('cuda', dtype=model.dtype):
        for row in data:
            batch = pack_rows([row], model.tokenizer, classes, config['max_length'], training=False)
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            if row.get('closed', False):
                if not names:
                    raise ValueError('The checkpoint needs closed-answer classes for this record.')
                prediction = names[int(model(batch)['classes'].argmax(-1).item())]
            else:
                prediction = model.generate(batch, args.max_new_tokens)[0]
            records.append(dict(patient_id=row.get('patient_id'), prediction=prediction, answer=row.get('answer'), closed=row.get('closed', False)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    print(json.dumps(evaluate_records(records)))


if __name__ == '__main__':
    main()
