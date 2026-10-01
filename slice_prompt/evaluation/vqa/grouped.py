"""Patient-aware VQA reports, answer confusion, and paired model comparisons."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from slice_prompt.engine.training.loop import evaluate_records, normalize


def read_predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Prediction file is empty')
    seen = set()
    for index, row in enumerate(rows):
        for field in ('prediction', 'answer'):
            if not isinstance(row.get(field), str):
                raise ValueError(f'Record {index} needs a text {field}')
        if not isinstance(row.get('closed'), bool):
            raise ValueError('Every answer must state whether its question is closed')
        identifier = str(row.get('id', index))
        if identifier in seen:
            raise ValueError(f'Duplicate prediction ID: {identifier}')
        seen.add(identifier)
        row['id'] = identifier
    return rows


def exact_match(row):
    return normalize(row['prediction']) == normalize(row['answer'])


def answer_confusion(rows):
    pairs = Counter()
    supports = Counter()
    correct = Counter()
    for row in rows:
        if not row['closed']:
            continue
        target = ' '.join(normalize(row['answer']))
        prediction = ' '.join(normalize(row['prediction']))
        pairs[target, prediction] += 1
        supports[target] += 1
        if target == prediction:
            correct[target] += 1
    table = [
        {'target': target, 'prediction': prediction, 'count': count}
        for (target, prediction), count in sorted(pairs.items())
    ]
    classes = [
        {'answer': label, 'support': count, 'correct': correct[label], 'recall': correct[label] / count}
        for label, count in sorted(supports.items())
    ]
    balanced = sum(row['recall'] for row in classes) / len(classes) if classes else None
    return {'pairs': table, 'classes': classes, 'balanced_closed_accuracy': balanced}


def group_answers(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if field in row and row[field] is not None:
            groups[str(row[field])].append(row)
    result = []
    for name, records in sorted(groups.items()):
        closed = [row for row in records if row['closed']]
        result.append(
            {
                'group': name,
                'records': len(records),
                'closed': len(closed),
                'open': len(records) - len(closed),
                **evaluate_records(records),
            }
        )
    return result


def patient_accuracy(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['closed']:
            if not row.get('patient_id'):
                raise ValueError('Patient accuracy requires patient_id on closed questions')
            groups[str(row['patient_id'])].append(exact_match(row))
    values = {key: sum(group) / len(group) for key, group in groups.items()}
    return {
        'patients': len(values),
        'macro_accuracy': float(np.mean(list(values.values()))) if values else None,
        'per_patient': values,
    }


def bootstrap_closed(rows, repeats=1000, seed=42, confidence=0.95):
    if repeats < 1 or not 0 < confidence < 1:
        raise ValueError('Bootstrap requires positive repeats and confidence in (0,1)')
    groups = defaultdict(list)
    for row in rows:
        if row['closed']:
            if not row.get('patient_id'):
                raise ValueError('Patient bootstrap requires patient_id')
            groups[str(row['patient_id'])].append(int(exact_match(row)))
    if not groups:
        return None
    names = sorted(groups)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        choices = generator.integers(0, len(names), size=len(names))
        sampled = [value for index in choices for value in groups[names[int(index)]]]
        values.append(sum(sampled) / len(sampled))
    tail = (1 - confidence) / 2
    return {
        'lower': float(np.quantile(values, tail)),
        'upper': float(np.quantile(values, 1 - tail)),
        'confidence': confidence,
        'repeats': repeats,
        'patients': len(names),
        'seed': seed,
    }


def paired_answers(before, after):
    left = {row['id']: row for row in before}
    right = {row['id']: row for row in after}
    if left.keys() != right.keys():
        raise ValueError('Paired predictions must contain identical question IDs')
    result = []
    for identity in sorted(left):
        a, b = left[identity], right[identity]
        if a['answer'] != b['answer'] or a['closed'] != b['closed']:
            raise ValueError(f'Target answers differ for question {identity}')
        result.append(
            {
                'id': identity,
                'closed': a['closed'],
                'answer': a['answer'],
                'before': a['prediction'],
                'after': b['prediction'],
                'before_exact': exact_match(a),
                'after_exact': exact_match(b),
            }
        )
    return result


def write_table(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def export_report(rows, output, fields=('anatomy', 'dataset', 'question_type'), bootstrap=0, seed=42, paired=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    confusion = answer_confusion(rows)
    report = {
        'format': 'slice-vqa-report-v1',
        'records': len(rows),
        'overall': evaluate_records(rows),
        'closed_answers': confusion,
        'groups': {field: group_answers(rows, field) for field in fields},
    }
    closed = [row for row in rows if row['closed']]
    if closed and all(row.get('patient_id') for row in closed):
        report['patients'] = patient_accuracy(rows)
    if bootstrap:
        report['closed_accuracy_interval'] = bootstrap_closed(rows, bootstrap, seed)
    write_table(output / 'answer_confusion.csv', confusion['pairs'])
    write_table(output / 'answer_classes.csv', confusion['classes'])
    for field, groups in report['groups'].items():
        write_table(output / f'group_{field}.csv', groups)
    if paired is not None:
        comparisons = paired_answers(paired, rows)
        write_table(output / 'paired.csv', comparisons)
        report['paired_closed_improvements'] = sum(
            row['closed'] and row['after_exact'] and not row['before_exact'] for row in comparisons
        )
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def report_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups', nargs='+', default=('anatomy', 'dataset', 'question_type'))
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare')
    args = parser.parse_args()
    if args.bootstrap < 0 or any(not value.replace('_', '').isalnum() for value in args.groups):
        parser.error('Use nonnegative bootstrap repeats and simple group field names')
    rows = read_predictions(args.predictions)
    paired = read_predictions(args.compare) if args.compare else None
    print(json.dumps(export_report(rows, args.output, args.groups, args.bootstrap, args.seed, paired), indent=2))
