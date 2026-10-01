"""Atomic trainable checkpoints, reproducible epoch resumes, and run metadata."""

import hashlib
import json
import math
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from slice_prompt.data.records.vocabulary import AnswerVocabulary


def validate_training_config(config):
    positive = (
        'slices',
        'image_size',
        'max_length',
        'rank',
        'heads',
        'fusion_layers',
        'epochs',
        'batch_size',
        'accumulation_steps',
    )
    for name in positive:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('workers', 'warmup_steps', 'distillation_epochs'):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if config['precision'] not in ('fp16', 'bf16'):
        raise ValueError('precision must be fp16 or bf16')
    for name in ('lr', 'weight_decay', 'lora_l2'):
        value = config[name]
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        if value < 0 or (name == 'lr' and value == 0):
            raise ValueError(f'Invalid optimization value for {name}')
    if config['distillation_epochs'] > config['epochs']:
        raise ValueError('Distillation epochs cannot exceed the training schedule')
    if config['fusion_layers'] > 2:
        raise ValueError('Slice fusion is attached to the first two decoder blocks')
    if not isinstance(config['offline'], bool):
        raise ValueError('offline must be a Boolean value')
    return dict(config)


def capture_random_state(loader=None):
    numpy_state = np.random.get_state()
    result = {
        'python': random.getstate(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
        'numpy': {
            'algorithm': numpy_state[0],
            'keys': numpy_state[1].tolist(),
            'position': numpy_state[2],
            'gaussian': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
    }
    if loader is not None and loader.generator is not None:
        result['loader'] = loader.generator.get_state()
    return result


def restore_random_state(state, loader=None):
    if not state:
        return
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    numpy_state = state['numpy']
    np.random.set_state(
        (
            numpy_state['algorithm'],
            np.asarray(numpy_state['keys'], dtype=np.uint32),
            numpy_state['position'],
            numpy_state['gaussian'],
            numpy_state['cached_gaussian'],
        )
    )
    if 'loader' in state:
        if loader is None or loader.generator is None:
            raise ValueError('Checkpoint includes a shuffle generator but the DataLoader has none')
        loader.generator.set_state(state['loader'].cpu())


def checkpoint_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def merge_history(previous, current):
    by_epoch = {}
    for row in [*previous, *current]:
        epoch = row['epoch']
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Training history epochs must be positive integers')
        if epoch in by_epoch and by_epoch[epoch] != row:
            raise ValueError(f'Two different histories exist for epoch {epoch}')
        by_epoch[epoch] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def save_training_checkpoint(loop, output, tokenizer, classes, history):
    if not history:
        raise ValueError('Cannot save a checkpoint without completed epoch statistics')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / 'history.json'
    previous = json.loads(history_path.read_text()) if history_path.is_file() else []
    complete = merge_history(previous, history)
    counts = [int(value) for value in loop.counts.detach().cpu().tolist()]
    vocabulary = AnswerVocabulary(classes, counts)
    state = dict(
        trainable=loop.model.trainable_state(),
        classes=classes,
        config=loop.config,
        optimizer=loop.optimizer.state_dict(),
        scheduler=loop.scheduler.state_dict(),
        scaler=loop.scaler.state_dict(),
        epoch=history[-1]['epoch'],
        step=loop.step,
        format_version=2,
        vocabulary=vocabulary.as_dict(),
        random_state=capture_random_state(loop.loader),
        history=complete,
    )
    if hasattr(loop, 'manifest_digest'):
        state['manifest_digest'] = loop.manifest_digest
    atomic_torch_save(state, output / 'last.pt')
    tokenizer.save_pretrained(output / 'tokenizer')
    vocabulary.save(output / 'vocabulary.json')
    atomic_json(history_path, complete)
    counts = {name: value.numel() for name, value in state['trainable'].items()}
    atomic_json(
        output / 'checkpoint.json',
        {
            'format': 'slice-checkpoint-v2',
            'epoch': state['epoch'],
            'step': state['step'],
            'classes': len(classes),
            'trainable_tensors': len(counts),
            'trainable_parameters': sum(counts.values()),
            'config': loop.config,
            'vocabulary_digest': vocabulary.digest(),
        },
    )


def check_resume(state, config, classes, manifest_digest=None):
    protected = (
        'language_model',
        'vision_source',
        'vision_revision',
        'revision',
        'rank',
        'heads',
        'fusion_layers',
        'slices',
        'image_size',
        'max_length',
        'precision',
    )
    for field in protected:
        if state['config'][field] != config[field]:
            raise ValueError(f'Resume setting differs from the checkpoint: {field}')
    if state['classes'] != classes:
        raise ValueError('Resume requires the saved closed-answer class mapping')
    if manifest_digest and state.get('manifest_digest') not in (None, manifest_digest):
        raise ValueError('Resume manifest records differ from the saved training data')
    if config['epochs'] <= state['epoch']:
        raise ValueError('Total epochs must exceed the completed checkpoint epoch')
    return state


def checkpoint_cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--verify-file', action='store_true')
    args = parser.parse_args()
    directory = Path(args.directory)
    report = json.loads((directory / 'checkpoint.json').read_text())
    vocabulary = AnswerVocabulary.load(directory / 'vocabulary.json')
    if vocabulary.digest() != report['vocabulary_digest']:
        raise ValueError('Vocabulary metadata does not match the checkpoint manifest')
    if args.verify_file:
        report['weights'] = checkpoint_identity(directory / 'last.pt')
    print(json.dumps(report, indent=2))
