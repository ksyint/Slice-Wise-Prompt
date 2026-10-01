"""Dictionary defaults and factory functions used by the runnable scripts."""
from collections import Counter

from .data.volumes import PromptDataset
from .data.vocabulary import Vocabulary
from .networks.vqa import SlicePromptVLM


def model_defaults():
    return dict(hidden_dim=64, heads=4, max_slices=64, rank=16)


def data_defaults():
    return dict(slices=8, image_size=32)


def training_defaults():
    return dict(epochs=25, batch_size=4, lr=.001, lora_l2=.0001)


def args_to_dict(values, keys):
    return {key: values[key] for key in keys}


def add_dict_to_argparser(parser, defaults):
    for key, value in defaults.items():
        parser.add_argument('--' + key.replace('_', '-'), default=None, type=type(value))


def create_model_and_data(config, manifest):
    data = PromptDataset(manifest, **args_to_dict(config, data_defaults()))
    vocabulary = Vocabulary().fit(data.rows)
    counts = Counter(row['answer'] for row in data.rows if row.get('closed', False))
    classes = {name: index for index, name in enumerate(sorted(counts))}
    model = SlicePromptVLM(len(vocabulary.words), max(1, len(classes)), **args_to_dict(config, model_defaults()))
    return model, data, vocabulary, classes, [counts[name] for name in classes]


def require_cuda(value='cuda'):
    import torch as th
    device = th.device(value)
    if device.type != 'cuda':
        raise ValueError('VQA model execution requires cuda or cuda:N.')
    if not th.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and an accessible GPU are required.')
    return device
