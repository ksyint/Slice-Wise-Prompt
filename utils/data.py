import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.prompts import render_prompt, rewrite_question


class Vocabulary:
    def __init__(self, words=None):
        self.words = words or ['<pad>', '<unk>', '<bos>', '<eos>']
        self.ids = {word: index for index, word in enumerate(self.words)}

    @staticmethod
    def split(text):
        return re.findall(r'\w+|[^\w\s]', str(text).lower())

    def fit(self, rows):
        words = {word for row in rows for key in ('question', 'answer') for word in self.split(row[key])}
        self.__init__(self.words + sorted(words - set(self.words)))
        return self

    def encode(self, text):
        return [self.ids.get(word, 1) for word in self.split(text)] or [1]

    def decode(self, tokens):
        result = []
        for token in tokens:
            if token == 3:
                break
            if token > 3:
                result.append(self.words[token])
        return ' '.join(result)


class PromptDataset(Dataset):
    def __init__(self, manifest, slices=16, image_size=64):
        self.root = Path(manifest).resolve().parent
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError('Empty manifest.')
        self.slices, self.image_size = slices, image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        volume = np.load(self.root / row['prompt'], allow_pickle=False)
        if volume.ndim != 4 or volume.shape[1] != 3 or not np.isfinite(volume).all():
            raise ValueError('Prompt array must be finite (Z,3,H,W).')
        indices = np.linspace(0, len(volume) - 1, self.slices).round().astype(int)
        tensor = F.interpolate(torch.tensor(volume[indices], dtype=torch.float32), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return dict(row, tensor=tensor)


def collate(rows, vocab, classes):
    def pad(sequences):
        output = torch.zeros(len(rows), max(map(len, sequences)), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            output[index, :len(sequence)] = torch.tensor(sequence)
        return output
    answers = pad([[2] + vocab.encode(row['answer']) + [3] for row in rows])
    labels = [classes.get(row['answer'], -1) if row.get('closed', False) else -1 for row in rows]
    return {'volumes': torch.stack([row['tensor'] for row in rows]), 'questions': pad([vocab.encode(row['question']) for row in rows]),
            'inputs': answers[:, :-1], 'targets': answers[:, 1:], 'labels': torch.tensor(labels)}


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
