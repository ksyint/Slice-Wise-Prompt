"""Resampled RGB prompt stacks and manifest records."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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
