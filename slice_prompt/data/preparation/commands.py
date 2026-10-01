"""Dataset exports, anatomical masks, and paired preparation recipes."""

import argparse
import csv
import json
import numpy as np
import subprocess
import sys
import yaml
from pathlib import Path
from PIL import Image
from slice_prompt.data.volumes.processing import (
    load_pair,
    patient_split,
    render_prompt,
    rewrite_question,
    system_prompt,
)
from slice_prompt.engine.training.loop import require_cuda
from itertools import product


def export_manifest_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True)
    parser.add_argument('--root', required=True, help='Root for source paths in the CSV.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--columns', default='{}', help='JSON mapping canonical names to CSV column names.')
    parser.add_argument(
        '--single-slice-images',
        action='store_true',
        help='Convert already-windowed image and mask files to Z=1 arrays.',
    )
    parser.add_argument('--center', type=float, default=-600)
    parser.add_argument('--width', type=float, default=1500)
    args = parser.parse_args()
    mapping = json.loads(args.columns)
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(args.table, newline='', encoding='utf-8-sig') as stream:
        for index, source in enumerate(csv.DictReader(stream)):
            row = {
                key: source[mapping.get(key, key)]
                for key in ('volume', 'mask', 'question', 'answer', 'patient_id', 'closed')
            }
            row['closed'] = row['closed'].strip().lower() in ('true', '1', 'yes', 'closed')
            if not row['patient_id'].strip():
                raise ValueError('Every row needs a patient identifier, or an article identifier for PMC-VQA.')
            volume, mask = root / row['volume'], root / row['mask']
            if not volume.is_file() or not mask.is_file():
                raise FileNotFoundError(f'Missing volume or mask: {volume}, {mask}')
            label_column = mapping.get('mask_labels', 'mask_labels')
            if source.get(label_column):
                row['mask_labels'] = json.loads(source[label_column])
            if args.single_slice_images:
                folder = output.parent / 'single_slices' / f'{index:06d}'
                folder.mkdir(parents=True, exist_ok=True)
                gray = np.asarray(Image.open(volume).convert('L'), dtype=np.float32) / 255
                labels = np.asarray(Image.open(mask))
                if labels.ndim != 2 or labels.shape != gray.shape:
                    raise ValueError('The image and mask must have identical 2D pixel grids.')
                # Store the existing intensity window, preserving it through the configured intensity window.
                np.save(folder / 'volume.npy', (gray * args.width + args.center - args.width / 2)[None])
                np.save(folder / 'mask.npy', labels[None])
                volume, mask = folder / 'volume.npy', folder / 'mask.npy'
            row.update(volume=str(volume), mask=str(mask))
            records.append(row)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    print(json.dumps({'exported': len(records)}))


def export_manifest_cli():
    export_manifest_main()


def prepare_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--color', choices=['blue', 'red'], default='blue')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--center', type=float, default=-600)
    parser.add_argument('--width', type=float, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--split-fractions', type=float, nargs=3, default=(0.7, 0.1, 0.2))
    args = parser.parse_args()
    root = Path(args.manifest).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    prepared = []
    for index, record in enumerate(records):
        volume, mask = load_pair(root / record['volume'], root / record['mask'])
        if record.get('mask_labels'):
            mask = np.isin(mask, record['mask_labels'])
        rgb, contours = render_prompt(volume, mask, args.color, args.thickness, args.center, args.width)
        sample = output / f'sample_{index:06d}'
        sample.mkdir(exist_ok=True)
        np.save(sample / 'prompt.npy', rgb.transpose(0, 3, 1, 2))
        pngs = []
        for slice_id, image in enumerate(rgb):
            filename = sample / f'{slice_id:04d}.png'
            Image.fromarray((image * 255).astype('uint8')).save(filename)
            pngs.append(str(filename))
        question = rewrite_question(record['question'], args.color)
        answer = (
            record['answer'] if record.get('closed', False) else f"Within the {args.color} boundary, {record['answer']}"
        )
        prepared.append(
            dict(
                record,
                volume=str((root / record['volume']).resolve()),
                mask=str((root / record['mask']).resolve()),
                prompt=str(sample / 'prompt.npy'),
                question=question,
                answer=answer,
                contour_voxels=int(contours.sum()),
                conversations=[
                    {'role': 'system', 'content': system_prompt(args.color, args.thickness)},
                    {'role': 'user', 'content': question},
                    {
                        'role': 'vision',
                        'content': {'volume': str((root / record['volume']).resolve()), 'prompt_png': pngs},
                    },
                    {'role': 'assistant', 'content': answer},
                ],
            )
        )
    splits = patient_split(prepared, args.seed, args.split_fractions) if args.split else {'prepared': prepared}
    for key, rows in splits.items():
        (output / f'{key}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({key: len(value) for key, value in splits.items()}))


def prepare_cli():
    prepare_main()


def segment_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='NIfTI volume, question, answer, and vista_labels per row.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--bundle-dir', default='weights/bundles')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    device = require_cuda(args.device)
    bundle = Path(args.bundle_dir).resolve() / 'vista3d'
    if not (bundle / 'models/model.pt').exists():
        if args.offline:
            raise FileNotFoundError(f'Place the official vista3d bundle, including models/model.pt, at {bundle}')
        subprocess.run(
            [sys.executable, '-m', 'monai.bundle', 'download', 'vista3d', '--bundle_dir', str(bundle.parent)],
            check=True,
        )
    root = Path(args.manifest).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, line in enumerate(Path(args.manifest).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        volume = (root / row['volume']).resolve()
        if not volume.name.endswith(('.nii', '.nii.gz')):
            raise ValueError('VISTA3D segmentation input must be NIfTI.')
        labels = row['vista_labels']
        if not labels or any(not isinstance(value, int) or value <= 0 for value in labels):
            raise ValueError('vista_labels must contain positive official VISTA class IDs.')
        destination = output / f'mask_{index:06d}'
        subprocess.run(
            [
                sys.executable,
                '-m',
                'monai.bundle',
                'run',
                '--config_file',
                'configs/inference.json',
                '--input_dict',
                repr({'image': str(volume), 'label_prompt': labels}),
                '--output_dir',
                str(destination),
                '--output_postfix',
                'mask',
                '--device',
                f"$torch.device('{device}')",
            ],
            cwd=bundle,
            check=True,
        )
        masks = list(destination.rglob('*_mask.nii.gz'))
        if len(masks) != 1:
            raise RuntimeError(f'Expected one VISTA3D mask under {destination}, found {len(masks)}.')
        rows.append(dict(row, volume=str(volume), mask=str(masks[0]), mask_labels=labels))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def segment_cli():
    segment_main()


catalog_ROOT = Path(__file__).resolve().parents[3]
WINDOWS = {'lung': dict(center=-600, width=1500), 'mediastinum': dict(center=40, width=400)}
INDEX = catalog_ROOT / 'recipes/index.jsonl'


def compile_recipes():
    baseline = yaml.safe_load((catalog_ROOT / 'configs/prompt_vqa.yaml').read_text())
    recipes = []
    for color, thickness, window, slices, size, rank, fusion, rate in product(
        ('blue', 'red'), (1, 3), WINDOWS, (16, 32), ('0.5B', '1.5B', '3B'), (8, 16), (1, 2), (1e-5, 2e-5)
    ):
        preparation = dict(color=color, thickness=thickness, **WINDOWS[window])
        training = dict(
            baseline,
            slices=slices,
            rank=rank,
            lr=rate,
            fusion_layers=fusion,
            language_model=f'Qwen/Qwen2.5-{size}-Instruct',
        )
        folder = (
            Path('recipes/volumetric') / color / f'border-{thickness}' / window / f'slices-{slices}' / f'qwen-{size}'
        )
        filename = f'rank-{rank}_fusion-{fusion}_lr-{rate:g}.yaml'
        path = catalog_ROOT / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dict(preparation=preparation, training=training), sort_keys=False))
        recipes.append(
            dict(
                name=f'{color}-{thickness}-{window}-{slices}-{size}-{rank}-{fusion}-{rate:g}',
                recipe=str(path.relative_to(catalog_ROOT)),
            )
        )
    INDEX.write_text(''.join(json.dumps(row) + '\n' for row in recipes))
    return recipes


def catalog_main():
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


def catalog_cli():
    catalog_main()


run_recipe_ROOT = Path(__file__).resolve().parents[3]


def prepare_command(recipe, manifest, output, seed, split):
    command = [
        sys.executable,
        str(run_recipe_ROOT / 'prompt.py'),
        'prepare',
        '--manifest',
        str(manifest),
        '--output',
        str(output),
        '--seed',
        str(seed),
    ]
    for key in ('color', 'thickness', 'center', 'width'):
        command.extend(['--' + key, str(recipe['preparation'][key])])
    if split:
        command.append('--split')
    return command


def run_recipe_main():
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
    subprocess.run(
        prepare_command(recipe, Path(args.manifest).resolve(), prepared, args.seed, args.split),
        cwd=run_recipe_ROOT,
        check=True,
    )
    training_config = output / 'training.yaml'
    training_config.write_text(yaml.safe_dump(recipe['training'], sort_keys=False))
    if not args.prepare_only:
        manifest = prepared / ('train.jsonl' if args.split else 'prepared.jsonl')
        subprocess.run(
            [
                sys.executable,
                str(run_recipe_ROOT / 'prompt.py'),
                'train',
                '--config',
                str(training_config),
                '--manifest',
                str(manifest),
                '--output',
                str(output / 'model'),
                '--device',
                args.device,
                '--seed',
                str(args.seed),
            ],
            cwd=run_recipe_ROOT,
            check=True,
        )


def run_recipe_cli():
    run_recipe_main()
