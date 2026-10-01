"""Export a joined region-QA CSV to the volume/mask JSONL consumed by prepare.py."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True)
    parser.add_argument('--root', required=True, help='Root for source paths in the CSV.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--columns', default='{}', help='JSON mapping canonical names to CSV column names.')
    parser.add_argument('--single-slice-images', action='store_true', help='Convert already-windowed image and mask files to Z=1 arrays.')
    parser.add_argument('--center', type=float, default=-600)
    parser.add_argument('--width', type=float, default=1500)
    args = parser.parse_args()
    mapping = json.loads(args.columns)
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(args.table, newline='', encoding='utf-8-sig') as stream:
        for index, source in enumerate(csv.DictReader(stream)):
            row = {key: source[mapping.get(key, key)] for key in ('volume', 'mask', 'question', 'answer', 'patient_id', 'closed')}
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
                # Store the existing intensity window, preserving it through prepare.py's window transform.
                np.save(folder / 'volume.npy', (gray * args.width + args.center - args.width / 2)[None])
                np.save(folder / 'mask.npy', labels[None])
                volume, mask = folder / 'volume.npy', folder / 'mask.npy'
            row.update(volume=str(volume), mask=str(mask))
            records.append(row)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    print(json.dumps({'exported': len(records)}))


if __name__ == '__main__':
    main()
