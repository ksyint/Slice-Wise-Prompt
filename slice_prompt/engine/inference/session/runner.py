"""Resumable CUDA answer generation with one durable result per question."""

import argparse
import json
import time
from pathlib import Path

import torch

from slice_prompt.data.records.schema import manifest_digest, read_manifest, write_manifest
from slice_prompt.data.records.vocabulary import AnswerVocabulary
from slice_prompt.data.volumes.processing import PromptDataset, pack_rows
from slice_prompt.engine.checkpoints.runtime import atomic_json, checkpoint_identity
from slice_prompt.engine.training.loop import require_cuda
from slice_prompt.models.qwen.adaptation import QwenSliceVLM


class PredictionJournal:
    def __init__(self, directory, identity, records, resume=False):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = list(records)
        self.identifiers = {row['id'] for row in self.records}
        manifest = self.directory / 'session.json'
        if manifest.exists():
            if not resume:
                raise FileExistsError('Prediction directory already has a session. Add --resume to continue')
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('Prediction checkpoint, inputs, or generation settings changed')
        else:
            atomic_json(manifest, identity)
        self.path = self.directory / 'answers.jsonl'
        self.completed = self._read()

    def _read(self):
        rows = {}
        if not self.path.is_file():
            return rows
        with self.path.open(encoding='utf-8') as stream:
            for index, line in enumerate(stream, 1):
                if not line.endswith('\n'):
                    raise ValueError(
                        f'Incomplete prediction record at line {index}. Remove that final line before resume'
                    )
                row = json.loads(line)
                identifier = row['id']
                if identifier not in self.identifiers or identifier in rows:
                    raise ValueError(f'Duplicate or unknown journal ID: {identifier}')
                rows[identifier] = row
        return rows

    def pending(self):
        return [index for index, row in enumerate(self.records) if row['id'] not in self.completed]

    def append(self, row):
        import os

        identifier = row['id']
        if identifier not in self.identifiers or identifier in self.completed:
            raise ValueError(f'Cannot append duplicate or unknown question ID: {identifier}')
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.completed[identifier] = row

    def export(self):
        ordered = [self.completed[row['id']] for row in self.records if row['id'] in self.completed]
        write_manifest(self.directory / 'predictions.jsonl', ordered)
        summary = {
            'questions': len(self.records),
            'completed': len(ordered),
            'remaining': len(self.records) - len(ordered),
            'closed': sum(row['closed'] for row in ordered),
            'elapsed_seconds': sum(row.get('elapsed_seconds', 0) for row in ordered),
        }
        atomic_json(self.directory / 'summary.json', summary)
        return summary


class InferenceSession:
    def __init__(
        self,
        checkpoint,
        manifest,
        output,
        device='cuda',
        resume=False,
        max_new_tokens=128,
        cache_dir=None,
        offline=False,
        language_model=None,
        vision_source=None,
    ):
        if max_new_tokens < 1:
            raise ValueError('Generation length must be positive')
        self.device = require_cuda(device)
        self.checkpoint = Path(checkpoint).resolve()
        self.manifest = Path(manifest).resolve()
        self.rows = read_manifest(self.manifest, prepared=True)
        self.overrides = {
            'cache_dir': cache_dir,
            'offline': bool(offline),
            'language_model': language_model,
            'vision_source': vision_source,
        }
        self.max_new_tokens = int(max_new_tokens)
        identity = {
            'format': 'slice-inference-v1',
            'checkpoint': checkpoint_identity(checkpoint),
            'manifest_digest': manifest_digest(self.rows),
            'overrides': self.overrides,
            'max_new_tokens': self.max_new_tokens,
        }
        self.journal = PredictionJournal(output, identity, self.rows, resume)
        self.model = None
        self.vocabulary = None
        self.data = None
        self.config = None

    def configure(self):
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.config = dict(state['config'])
        for name in ('cache_dir', 'language_model', 'vision_source'):
            if self.overrides[name] is not None:
                self.config[name] = self.overrides[name]
        if self.overrides['offline']:
            self.config['offline'] = True
        self.vocabulary = AnswerVocabulary.from_checkpoint(state)
        self.vocabulary.validate_records(self.rows)
        self.model = QwenSliceVLM(self.config, len(self.vocabulary.classes)).to(self.device)
        self.model.restore_trainable(state['trainable'])
        self.model.eval()
        self.data = PromptDataset(self.manifest, slices=self.config['slices'], image_size=self.config['image_size'])
        self.data.rows = self.rows
        del state

    def predict_one(self, index):
        row = self.data[index]
        before = time.perf_counter()
        batch = pack_rows(
            [row], self.model.tokenizer, self.vocabulary.classes, self.config['max_length'], training=False
        )
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
        }
        with torch.inference_mode(), torch.autocast('cuda', dtype=self.model.dtype):
            if row['closed']:
                if not self.vocabulary.classes:
                    raise ValueError('Checkpoint has no closed-answer classifier vocabulary')
                output = self.model(batch)['classes'].float()
                probabilities = output.softmax(-1)[0]
                selected = int(probabilities.argmax().item())
                prediction = self.vocabulary.decode(selected)
                confidence = float(probabilities[selected].item())
            else:
                prediction = self.model.generate(batch, self.max_new_tokens)[0]
                confidence = None
        result = {key: row[key] for key in ('id', 'patient_id', 'closed', 'answer', 'question')}
        result.update(prediction=prediction, confidence=confidence, elapsed_seconds=time.perf_counter() - before)
        for field in ('anatomy', 'dataset', 'question_type', 'volume', 'mask'):
            if field in row:
                result[field] = row[field]
        return result

    def run(self):
        pending = self.journal.pending()
        if pending:
            self.configure()
            for index in pending:
                self.journal.append(self.predict_one(index))
        return self.journal.export()


def session_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--language-model')
    parser.add_argument('--vision-source')
    args = parser.parse_args()
    session = InferenceSession(
        args.checkpoint,
        args.manifest,
        args.output,
        args.device,
        args.resume,
        args.max_new_tokens,
        args.cache_dir,
        args.offline,
        args.language_model,
        args.vision_source,
    )
    print(json.dumps(session.run(), indent=2))
