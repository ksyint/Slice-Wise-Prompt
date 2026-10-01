"""Train pretrained DCFormer/Qwen2.5 from prepared slice-prompt volumes."""
import argparse
from functools import partial
import json
from pathlib import Path

import torch as th
from torch.utils.data import DataLoader
import yaml

from slice_prompt.data.text.packing import pack_rows
from slice_prompt.script_util import (add_dict_to_argparser, create_model_and_data, data_defaults,
                                      model_defaults, training_defaults, require_cuda)
from slice_prompt.train_util import TrainLoop


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/prompt_vqa.yaml')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', help='Resume the trainable weights and optimizer from last.pt.')
    parser.add_argument('--output', default='outputs/default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    add_dict_to_argparser(parser, dict(**model_defaults(), **data_defaults(), **training_defaults()))
    return parser


def main():
    args = create_argparser().parse_args()
    device = require_cuda(args.device)
    th.manual_seed(args.seed)
    config = dict(**model_defaults(), **data_defaults(), **training_defaults())
    config.update(yaml.safe_load(Path(args.config).read_text()))
    state = th.load(args.checkpoint, map_location=device, weights_only=False) if args.checkpoint else None
    if state:
        config = state['config'].copy()
    config.update({key: getattr(args, key) for key in config if getattr(args, key, None) is not None})
    model, data, tokenizer, classes, counts = create_model_and_data(config, args.manifest, device)
    if state and classes != state['classes']:
        raise ValueError('Resuming requires the saved closed-answer class mapping.')
    loader = DataLoader(data, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'],
                        collate_fn=partial(pack_rows, tokenizer=tokenizer, classes=classes, max_length=config['max_length']))
    loop = TrainLoop(model, loader, counts, config, device)
    if state:
        loop.restore(state)
    history = []
    for statistics in loop.run_loop():
        history.append(statistics)
        print(json.dumps(statistics), flush=True)
        loop.save(args.output, tokenizer, classes, history)


if __name__ == '__main__':
    main()
