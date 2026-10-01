"""Synthetic region-grounded volume fixture."""
import json
from pathlib import Path
import numpy as np
from slice_prompt.preprocess.contours import render_prompt
from slice_prompt.preprocess.conversation import rewrite_question


def make_demo(folder, seed=42):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    for index in range(16):
        volume = rng.normal(-700, 40, (4, 24, 24))
        mask = np.zeros_like(volume, dtype=bool)
        left = 3 if index % 2 == 0 else 13
        mask[:, 8:17, left:left + 7] = True
        volume[mask] += 500
        prompted, _ = render_prompt(volume, mask)
        filename = f'volume_{index:03d}.npy'
        np.save(folder / filename, prompted.transpose(0, 3, 1, 2))
        rows.append({'prompt': filename, 'patient_id': f'synthetic-{index}', 'question': rewrite_question('is the shape on the left?'),
                     'answer': 'yes' if index % 2 == 0 else 'no', 'closed': index % 3 != 0})
    path = folder / 'train.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path
