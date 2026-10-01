"""Persisted closed-answer IDs, counts, and training-manifest identity."""

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from slice_prompt.pipeline.volumes.records.schema import manifest_digest, read_manifest


def canonical_answer(value, lowercase=False):
    if not isinstance(value, str):
        raise TypeError('Answer labels must be strings')
    answer = unicodedata.normalize('NFC', value).strip()
    if lowercase:
        answer = answer.casefold()
    if not answer:
        raise ValueError('Closed answers cannot be empty')
    return answer


class AnswerVocabulary:
    def __init__(self, classes, counts=None, normalization='exact', source_digest=None):
        if normalization not in ('exact', 'nfc', 'casefold'):
            raise ValueError('Unknown answer normalization')
        self.classes = dict(classes)
        self.normalization = normalization
        self.source_digest = source_digest
        values = list(self.classes.values())
        if any(type(value) is not int for value in values) or sorted(values) != list(range(len(values))):
            raise ValueError('Answer IDs must be contiguous and begin at zero')
        if any(not isinstance(key, str) or not key for key in self.classes):
            raise ValueError('Answer class names must be nonempty strings')
        self.names = sorted(self.classes, key=self.classes.get)
        self.counts = list(counts or [0] * len(values))
        if len(self.counts) != len(values) or any(type(value) is not int or value < 0 for value in self.counts):
            raise ValueError('Each class requires one nonnegative integer count')

    def normalize(self, answer):
        if self.normalization == 'exact':
            if not isinstance(answer, str) or not answer:
                raise ValueError('Answer labels must be nonempty strings')
            return answer
        return canonical_answer(answer, self.normalization == 'casefold')

    @classmethod
    def from_records(cls, records, normalization='exact'):
        prototype = cls({}, normalization=normalization)
        counts = Counter(prototype.normalize(row['answer']) for row in records if row['closed'])
        classes = {name: index for index, name in enumerate(sorted(counts))}
        return cls(classes, [counts[name] for name in classes], normalization, manifest_digest(records))

    @classmethod
    def from_checkpoint(cls, checkpoint):
        classes = checkpoint['classes']
        metadata = checkpoint.get('vocabulary', {})
        return cls(
            classes, metadata.get('counts'), metadata.get('normalization', 'exact'), metadata.get('source_digest')
        )

    def encode(self, answer, allow_unknown=False):
        value = self.normalize(answer)
        if value not in self.classes:
            if allow_unknown:
                return -1
            raise KeyError(f'Unknown closed answer: {answer}')
        return self.classes[value]

    def decode(self, index):
        if type(index) is not int or not 0 <= index < len(self.names):
            raise IndexError('Closed-answer index is outside the vocabulary')
        return self.names[index]

    def validate_records(self, records, require_coverage=False):
        counts = Counter()
        unknown = Counter()
        for row in records:
            if not row['closed']:
                continue
            answer = self.normalize(row['answer'])
            (counts if answer in self.classes else unknown)[answer] += 1
        if unknown:
            raise ValueError('Unknown closed answers: ' + json.dumps(dict(unknown), ensure_ascii=False))
        missing = [name for name in self.names if not counts[name]]
        if require_coverage and missing:
            raise ValueError(f'Training data omits saved answer classes: {missing}')
        return {
            'closed_records': sum(counts.values()),
            'observed_classes': len(counts),
            'absent_classes': missing,
            'counts': [counts[name] for name in self.names],
        }

    def reindex_rows(self, records):
        result = []
        for row in records:
            value = dict(row)
            if value['closed']:
                value['answer'] = self.normalize(value['answer'])
                value['class_id'] = self.encode(value['answer'])
            result.append(value)
        return result

    def as_dict(self):
        return {
            'format': 'slice-answer-vocabulary-v1',
            'classes': self.classes,
            'counts': self.counts,
            'normalization': self.normalization,
            'source_digest': self.source_digest,
        }

    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        temporary.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + '\n')
        temporary.replace(destination)

    @classmethod
    def load(cls, source):
        value = json.loads(Path(source).read_text())
        if value.get('format') != 'slice-answer-vocabulary-v1':
            raise ValueError('Unsupported answer vocabulary format')
        return cls(value['classes'], value['counts'], value['normalization'], value['source_digest'])

    def compare(self, other):
        shared = set(self.classes) & set(other.classes)
        changes = {
            name: [self.classes[name], other.classes[name]]
            for name in sorted(shared)
            if self.classes[name] != other.classes[name]
        }
        return {
            'added': sorted(set(other.classes) - set(self.classes)),
            'removed': sorted(set(self.classes) - set(other.classes)),
            'changed_ids': changes,
            'same_mapping': self.classes == other.classes,
        }


def vocabulary_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--existing')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--normalization', choices=('exact', 'nfc', 'casefold'), default='exact')
    args = parser.parse_args()
    rows = read_manifest(args.manifest, args.prepared)
    vocabulary = AnswerVocabulary.from_records(rows, args.normalization)
    report = {
        'classes': len(vocabulary.classes),
        'closed_records': sum(vocabulary.counts),
        'fingerprint': vocabulary.digest(),
    }
    if args.existing:
        report['comparison'] = AnswerVocabulary.load(args.existing).compare(vocabulary)
    vocabulary.save(args.output)
    print(json.dumps(report, indent=2))
