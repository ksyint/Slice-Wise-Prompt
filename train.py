"""Train from prepared prompt stacks; command-line values override YAML defaults."""
import argparse
from functools import partial
import json
from pathlib import Path

import torch as th
from torch.utils.data import DataLoader
import yaml

from slice_prompt.data.batching import collate
from slice_prompt.data.synthetic import make_demo
from slice_prompt.script_util import (
    add_dict_to_argparser, create_model_and_data, data_defaults,
    model_defaults, training_defaults,
)
from slice_prompt.train_util import TrainLoop


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/prompt_vqa.yaml')
    parser.add_argument('--manifest')
    parser.add_argument('--output', default='outputs/default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    add_dict_to_argparser(parser, dict(**model_defaults(), **data_defaults(), **training_defaults()))
    return parser


def main():
    args = create_argparser().parse_args()
    th.set_num_threads(2)
    th.manual_seed(args.seed)
    defaults = dict(**model_defaults(), **data_defaults(), **training_defaults())
    defaults.update(yaml.safe_load(Path(args.config).read_text()))
    defaults.update({key: getattr(args, key) for key in defaults if getattr(args, key, None) is not None})
    output = Path(args.output)
    manifest = make_demo(output / 'synthetic', args.seed) if args.smoke else args.manifest
    if not manifest:
        raise ValueError('Use --manifest or --smoke.')
    if args.smoke:
        defaults['epochs'] = 2
    model, data, vocabulary, classes, counts = create_model_and_data(defaults, manifest)
    loader = DataLoader(data, batch_size=defaults['batch_size'], shuffle=True,
                        collate_fn=partial(collate, vocab=vocabulary, classes=classes))
    loop = TrainLoop(model, loader, counts, defaults, args.device)
    history = []
    for statistics in loop.run_loop():
        history.append(statistics)
        print(json.dumps(statistics), flush=True)
    loop.save(output, vocabulary, classes, history)


if __name__ == '__main__':
    main()
