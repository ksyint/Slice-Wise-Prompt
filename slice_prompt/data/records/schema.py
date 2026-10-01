"""Canonical volume-question records and patient-level manifest checks."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def record_identity(row):
    if str(row.get('id', '')).strip():
        return str(row['id'])
    fields = {key: row.get(key) for key in ('patient_id', 'volume', 'mask', 'question', 'answer')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def read_manifest(path, prepared=False, check_files=True):
    path = Path(path).resolve()
    records = []
    seen = set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected an object')
            row = dict(row)
            for field in ('patient_id', 'question', 'answer'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'{path}:{line_number}: {field} must be nonempty text')
            if not isinstance(row.get('closed'), bool):
                raise ValueError(f'{path}:{line_number}: closed must be true or false')
            fields = ('prompt',) if prepared else ('volume', 'mask')
            for field in fields:
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f'{path}:{line_number}: missing {field} path')
                resolved = (path.parent / value).resolve()
                if check_files and not resolved.is_file():
                    raise FileNotFoundError(resolved)
                row[field] = str(resolved)
            for field in ('volume', 'mask'):
                if field in row:
                    row[field] = str((path.parent / row[field]).resolve())
            labels = row.get('mask_labels')
            if labels is not None:
                if not isinstance(labels, list) or any(type(value) is not int or value < 1 for value in labels):
                    raise ValueError('mask_labels contains positive anatomical class IDs')
            row['id'] = record_identity(row)
            if row['id'] in seen:
                raise ValueError(f'Duplicate volume-question record: {row["id"]}')
            seen.add(row['id'])
            records.append(row)
    if not records:
        raise ValueError('Manifest contains no records')
    return records


def manifest_digest(records):
    value = json.dumps(records, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def check_prompt_arrays(records, slices=None):
    shapes = Counter()
    sources = {}
    for row in records:
        path = Path(row['prompt'])
        if str(path) in sources:
            continue
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.ndim != 4 or array.shape[1] != 3 or min(array.shape) < 1:
            raise ValueError(f'Prompt stack must have shape [Z,3,H,W]: {path}')
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f'Prompt stack must contain floating intensities: {path}')
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError(f'Prompt RGB values must be finite and within [0,1]: {path}')
        key = 'x'.join(map(str, array.shape))
        shapes[key] += 1
        sources[str(path)] = {
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'sampled_slices': int(slices or array.shape[0]),
        }
    return {'unique_stacks': len(sources), 'shapes': dict(shapes), 'stacks': sources}


def summarize_records(records):
    closed = [row for row in records if row['closed']]
    groups = Counter(row['patient_id'] for row in records)
    classes = Counter(row['answer'] for row in closed)
    return {
        'records': len(records),
        'patients': len(groups),
        'closed': len(closed),
        'open': len(records) - len(closed),
        'closed_answers': dict(sorted(classes.items())),
        'questions_per_patient': dict(sorted(groups.items())),
        'fingerprint': manifest_digest(records),
    }


def validate_partitions(partitions):
    owner = {}
    identity_owner = {}
    for name, records in partitions.items():
        for row in records:
            patient = row['patient_id']
            if owner.setdefault(patient, name) != name:
                raise ValueError(f'Patient occurs in multiple partitions: {patient}')
            identity = record_identity(row)
            if identity_owner.setdefault(identity, name) != name:
                raise ValueError(f'Record occurs in multiple partitions: {identity}')
    return {name: summarize_records(rows) for name, rows in partitions.items()}


def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    temporary.replace(path)
    return path


def grouped_questions(records, field='anatomy'):
    result = defaultdict(list)
    for row in records:
        value = row.get(field)
        if value is not None:
            result[str(value)].append(row['id'])
    return dict(sorted(result.items()))


def manifest_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--inspect-arrays', action='store_true')
    parser.add_argument('--compare', nargs='*', default=[])
    parser.add_argument('--output')
    args = parser.parse_args()
    rows = read_manifest(args.manifest, args.prepared)
    report = summarize_records(rows)
    if args.inspect_arrays:
        if not args.prepared:
            parser.error('--inspect-arrays requires --prepared')
        report['arrays'] = check_prompt_arrays(rows)
    if args.compare:
        partitions = {'input': rows}
        for index, path in enumerate(args.compare):
            partitions[f'comparison_{index}'] = read_manifest(path, args.prepared)
        report['partitions'] = validate_partitions(partitions)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
