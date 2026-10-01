import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.prompts import load_volume, patient_split, render_prompt, rewrite_question, system_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--color', choices=['blue', 'red'], default='blue')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--center', type=float, default=-600)
    parser.add_argument('--width', type=float, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()
    root = Path(args.manifest).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    prepared = []
    for index, record in enumerate(records):
        volume, mask = load_volume(root / record['volume']), load_volume(root / record['mask'])
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
        answer = record['answer'] if record.get('closed', False) else f"Within the {args.color} boundary, {record['answer']}"
        prepared.append(dict(record, volume=str((root / record['volume']).resolve()), mask=str((root / record['mask']).resolve()),
                             prompt=str(sample / 'prompt.npy'), question=question, answer=answer,
                             contour_voxels=int(contours.sum()), conversations=[
                                 {'role': 'system', 'content': system_prompt(args.color, args.thickness)},
                                 {'role': 'user', 'content': question},
                                 {'role': 'vision', 'content': {'volume': str((root / record['volume']).resolve()), 'prompt_png': pngs}},
                                 {'role': 'assistant', 'content': answer}]))
    splits = patient_split(prepared, args.seed) if args.split else {'prepared': prepared}
    for key, rows in splits.items():
        (output / f'{key}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps({key: len(value) for key, value in splits.items()}))


if __name__ == '__main__':
    main()
