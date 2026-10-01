"""Durable array and contact-sheet artifacts for region-prompt preparation."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_array(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def png_stack(array, directory, indices=None):
    values = np.asarray(array)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("PNG export expects Z,3,H,W prompt arrays")
    indices = np.arange(len(values)) if indices is None else np.asarray(indices, dtype=int)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for position, index in enumerate(indices):
        if not 0 <= index < len(values):
            raise ValueError("PNG slice index exceeds prompt depth")
        rgb = np.moveaxis(values[index], 0, -1)
        image = Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))
        path = directory / f"{position:04d}-z{index:04d}.png"
        image.save(path)
        files.append(str(path.resolve()))
    return files


def contact_sheet(array, path, indices=None, columns=4, tile_size=192):
    values = np.asarray(array)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("Contact sheets expect Z,3,H,W prompt arrays")
    if columns < 1 or tile_size < 16:
        raise ValueError("Contact sheet layout requires positive columns and tile size")
    if indices is None:
        indices = np.unique(
            np.linspace(0, len(values) - 1, min(16, len(values))).round().astype(int)
        )
    indices = list(indices)
    if not indices:
        raise ValueError("Contact sheets require at least one selected slice")
    rows = (len(indices) + columns - 1) // columns
    label_height = 24
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "#101010")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        if not 0 <= index < len(values):
            raise ValueError("Contact sheet slice index exceeds volume depth")
        rgb = np.moveaxis(values[index], 0, -1)
        image = Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))
        image.thumbnail((tile_size, tile_size), Image.Resampling.BILINEAR)
        left = position % columns * tile_size
        top = position // columns * (tile_size + label_height)
        sheet.paste(
            image, (left + (tile_size - image.width) // 2, top + (tile_size - image.height) // 2)
        )
        draw.text((left + 5, top + tile_size + 4), f"z = {index}", fill="white")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {"path": str(path.resolve()), "slices": indices, "size": list(sheet.size)}


def export_prompt(directory, prompt, contour, metadata, png=False, preview=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_array(directory / "prompt.npy", np.asarray(prompt, dtype=np.float32))
    save_array(directory / "contour.npy", np.asarray(contour, dtype=bool))
    result = {
        "prompt": str((directory / "prompt.npy").resolve()),
        "contour": str((directory / "contour.npy").resolve()),
        "shape": list(prompt.shape),
        "metadata": metadata,
    }
    if png:
        result["pngs"] = png_stack(prompt, directory / "png")
    if preview:
        result["preview"] = contact_sheet(prompt, directory / "preview.png")
    write_json(directory / "prepared.json", result)
    return result
