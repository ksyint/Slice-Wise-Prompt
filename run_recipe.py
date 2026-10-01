"""Apply a contour recipe to source volumes, then train its paired VQA setup."""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent


def prepare_command(recipe, manifest, output, seed, split):
    command = [sys.executable, str(ROOT / 'prepare.py'), '--manifest', str(manifest), '--output', str(output), '--seed', str(seed)]
    for key in ('color', 'thickness', 'center', 'width'):
        command.extend(['--' + key, str(recipe['preparation'][key])])
    if split:
        command.append('--split')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--manifest', required=True, help='Source volume/mask/question JSONL.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', action='store_true', help='Create patient-disjoint 70/10/20 partitions.')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / 'prepared'
    subprocess.run(prepare_command(recipe, Path(args.manifest).resolve(), prepared, args.seed, args.split), cwd=ROOT, check=True)
    training_config = output / 'training.yaml'
    training_config.write_text(yaml.safe_dump(recipe['training'], sort_keys=False))
    if not args.prepare_only:
        manifest = prepared / ('train.jsonl' if args.split else 'prepared.jsonl')
        subprocess.run([sys.executable, str(ROOT / 'train.py'), '--config', str(training_config),
                        '--manifest', str(manifest), '--output', str(output / 'model'), '--device', args.device,
                        '--seed', str(args.seed)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
