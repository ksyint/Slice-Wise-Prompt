"""Joint contour-preparation and VQA training recipes."""
import argparse
from itertools import product
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = {'lung': dict(center=-600, width=1500), 'mediastinum': dict(center=40, width=400)}
INDEX = ROOT / 'recipes/index.jsonl'


def compile_recipes():
    baseline = yaml.safe_load((ROOT / 'configs/prompt_vqa.yaml').read_text())
    recipes = []
    for color, thickness, window, slices, size, rank, fusion, rate in product(
            ('blue', 'red'), (1, 3), WINDOWS, (16, 32), ('0.5B', '1.5B', '3B'), (8, 16), (1, 2), (1e-5, 2e-5)):
        preparation = dict(color=color, thickness=thickness, **WINDOWS[window])
        training = dict(baseline, slices=slices, rank=rank, lr=rate, fusion_layers=fusion,
                        language_model=f'Qwen/Qwen2.5-{size}-Instruct')
        folder = Path('recipes/volumetric') / color / f'border-{thickness}' / window / f'slices-{slices}' / f'qwen-{size}'
        filename = f'rank-{rank}_fusion-{fusion}_lr-{rate:g}.yaml'
        path = ROOT / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dict(preparation=preparation, training=training), sort_keys=False))
        recipes.append(dict(name=f'{color}-{thickness}-{window}-{slices}-{size}-{rank}-{fusion}-{rate:g}',
                            recipe=str(path.relative_to(ROOT))))
    INDEX.write_text(''.join(json.dumps(row) + '\n' for row in recipes))
    return recipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    entries = compile_recipes() if args.compile else [json.loads(line) for line in INDEX.read_text().splitlines()]
    if args.list:
        for entry in entries:
            print(entry['name'] + '\t' + entry['recipe'])
    else:
        print(json.dumps(dict(recipes=len(entries), index=str(INDEX))))


if __name__ == '__main__':
    main()
