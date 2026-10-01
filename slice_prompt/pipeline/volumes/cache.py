"""Reusable contour arrays keyed by volume, mask, and rendering settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

from slice_prompt.pipeline.volumes.records.schema import read_manifest, write_manifest
from slice_prompt.pipeline.volumes.processing import (
    load_pair,
    patient_split,
    render_prompt,
    rewrite_question,
    system_prompt,
)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def cache_key(record, settings):
    identity = {
        'format': 'slice-contour-cache-v1',
        'volume': file_identity(record['volume']),
        'mask': file_identity(record['mask']),
        'mask_labels': record.get('mask_labels'),
        'settings': settings,
    }
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest(), identity


class PromptCache:
    def __init__(self, root, color='blue', thickness=1, center=-600, width=1500, png=False):
        if color not in ('blue', 'red') or thickness < 1 or thickness % 2 != 1:
            raise ValueError('Use a blue/red contour and an odd positive thickness')
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            raise ValueError('CT center and positive window width must be finite')
        self.root = Path(root).resolve()
        self.settings = dict(color=color, thickness=thickness, center=float(center), width=float(width))
        self.png = bool(png)
        self.hits = self.misses = 0

    def locate(self, record):
        key, identity = cache_key(record, self.settings)
        return self.root / key[:2] / key, identity

    def read(self, record):
        directory, identity = self.locate(record)
        metadata = directory / 'metadata.json'
        array = directory / 'prompt.npy'
        if not metadata.is_file() or not array.is_file():
            return None
        saved = json.loads(metadata.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Prompt cache metadata differs at {directory}')
        data = np.load(array, mmap_mode='r', allow_pickle=False)
        if list(data.shape) != saved['shape'] or str(data.dtype) != 'float32':
            raise ValueError(f'Prompt cache array does not match its metadata: {array}')
        if self.png and not all((directory / value).is_file() for value in saved.get('pngs', [])):
            return None
        if self.png and not saved.get('pngs'):
            return None
        return dict(saved, prompt=str(array), directory=str(directory))

    def prepare(self, record):
        cached = self.read(record)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        directory, identity = self.locate(record)
        volume, mask = load_pair(record['volume'], record['mask'])
        if record.get('mask_labels'):
            mask = np.isin(mask, record['mask_labels'])
        rgb, contours = render_prompt(volume, mask, **self.settings)
        data = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2), dtype=np.float32)
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=directory, suffix='.npy', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.save(stream, data, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(directory / 'prompt.npy')
        pngs = []
        if self.png:
            for index, image in enumerate(rgb):
                filename = f'{index:04d}.png'
                Image.fromarray((image * 255).round().astype('uint8')).save(directory / filename)
                pngs.append(filename)
        metadata = {
            'identity': identity,
            'shape': list(data.shape),
            'dtype': 'float32',
            'contour_voxels': int(contours.sum()),
            'intersecting_slices': int(np.any(contours, axis=(1, 2)).sum()),
            'pngs': pngs,
        }
        temporary = directory / 'metadata.json.tmp'
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(directory / 'metadata.json')
        return dict(metadata, prompt=str(directory / 'prompt.npy'), directory=str(directory))

    def conversation(self, record, artifact):
        color, thickness = self.settings['color'], self.settings['thickness']
        question = rewrite_question(record['question'], color)
        answer = record['answer'] if record['closed'] else f'Within the {color} boundary, {record["answer"]}'
        vision = {'volume': record['volume'], 'prompt': artifact['prompt']}
        if artifact['pngs']:
            vision['prompt_png'] = [str(Path(artifact['directory']) / value) for value in artifact['pngs']]
        result = dict(
            record,
            prompt=artifact['prompt'],
            question=question,
            answer=answer,
            contour_voxels=artifact['contour_voxels'],
            intersecting_slices=artifact['intersecting_slices'],
        )
        result['conversations'] = [
            {'role': 'system', 'content': system_prompt(color, thickness)},
            {'role': 'user', 'content': question},
            {'role': 'vision', 'content': vision},
            {'role': 'assistant', 'content': answer},
        ]
        return result

    def build(self, records, output, split=False, seed=42, fractions=(0.7, 0.1, 0.2)):
        prepared = [self.conversation(row, self.prepare(row)) for row in records]
        partitions = patient_split(prepared, seed, fractions) if split else {'prepared': prepared}
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in partitions.items():
            write_manifest(output / f'{name}.jsonl', rows)
        report = {
            'format': 'slice-preparation-v1',
            'settings': self.settings,
            'cache': str(self.root),
            'hits': self.hits,
            'misses': self.misses,
            'partitions': {name: len(rows) for name, rows in partitions.items()},
            'seed': seed,
            'fractions': list(fractions) if split else None,
        }
        (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def cache_prepare_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--color', choices=('blue', 'red'), default='blue')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--center', type=float, default=-600)
    parser.add_argument('--width', type=float, default=1500)
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fractions', type=float, nargs=3, default=(0.7, 0.1, 0.2))
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    cache = PromptCache(args.cache, args.color, args.thickness, args.center, args.width, args.png)
    print(json.dumps(cache.build(records, args.output, args.split, args.seed, args.fractions), indent=2))
