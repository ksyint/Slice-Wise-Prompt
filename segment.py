"""Run the official VISTA3D MONAI bundle and write a mask-ready source manifest."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from slice_prompt.script_util import require_cuda


def main():
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
        subprocess.run([sys.executable, '-m', 'monai.bundle', 'download', 'vista3d',
                        '--bundle_dir', str(bundle.parent)], check=True)
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
        subprocess.run([sys.executable, '-m', 'monai.bundle', 'run', '--config_file', 'configs/inference.json',
                        '--input_dict', repr({'image': str(volume), 'label_prompt': labels}),
                        '--output_dir', str(destination), '--output_postfix', 'mask',
                        '--device', f"$torch.device('{device}')"], cwd=bundle, check=True)
        masks = list(destination.rglob('*_mask.nii.gz'))
        if len(masks) != 1:
            raise RuntimeError(f'Expected one VISTA3D mask under {destination}, found {len(masks)}.')
        rows.append(dict(row, volume=str(volume), mask=str(masks[0]), mask_labels=labels))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


if __name__ == '__main__':
    main()
