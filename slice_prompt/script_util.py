"""Dictionary defaults and factories for pretrained Qwen slice-prompt experiments."""
from collections import Counter

from .data.volumes import PromptDataset
from .networks.pretrained.qwen import QwenSliceVLM
from .networks.pretrained.weights import MED3D_ID, MED3D_REVISION


def model_defaults():
    return dict(language_model='Qwen/Qwen2.5-1.5B-Instruct', vision_source=MED3D_ID,
                vision_revision=MED3D_REVISION, revision='main', heads=4, rank=16, fusion_layers=2,
                precision='bf16', cache_dir=None, offline=False)


def data_defaults():
    return dict(slices=32, image_size=256, max_length=1024)


def training_defaults():
    return dict(epochs=25, batch_size=4, lr=2e-5, lora_l2=1e-4, warmup_steps=500,
                weight_decay=.02, distillation_epochs=1, accumulation_steps=1, workers=0)


def args_to_dict(values, keys):
    return {key: values[key] for key in keys}


def add_dict_to_argparser(parser, defaults):
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument('--' + key.replace('_', '-'), action='store_true', default=None)
        else:
            parser.add_argument('--' + key.replace('_', '-'), default=None, type=type(value) if value is not None else str)


def create_model_and_data(config, manifest, device):
    require_cuda(device)
    data = PromptDataset(manifest, slices=config['slices'], image_size=config['image_size'])
    counts = Counter(row['answer'] for row in data.rows if row.get('closed', False))
    classes = {name: index for index, name in enumerate(sorted(counts))}
    model = QwenSliceVLM(config, len(classes)).to(device)
    return model, data, model.tokenizer, classes, [counts[name] for name in classes]


def require_cuda(value='cuda'):
    import torch as th
    device = th.device(value)
    if device.type != 'cuda':
        raise ValueError('VQA model execution requires cuda or cuda:N.')
    if not th.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and an accessible GPU are required.')
    return device
