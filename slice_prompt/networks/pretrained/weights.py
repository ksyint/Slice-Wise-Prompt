"""Load the released DCFormer and dual-stream Mixer without allocating its 7B decoder."""
import importlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

MED3D_ID = 'MagicXin/Med3DVLM-Qwen-2.5-7B'
MED3D_REVISION = '7d192aef4b06501a6ba191a474e6d684ca492824'


def hub_options(config, visual=False):
    return dict(cache_dir=config.get('cache_dir'), local_files_only=config.get('offline', False),
                revision=config.get('vision_revision', MED3D_REVISION) if visual else config.get('revision', 'main'))


def checkpoint_file(source, filename, **options):
    if Path(source).is_dir():
        path = Path(source) / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(source, filename=filename, **options))


class FrozenMed3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from safetensors import safe_open
        source = config.get('vision_source', MED3D_ID)
        options = hub_options(config, visual=True)
        native = AutoConfig.from_pretrained(source, trust_remote_code=True, **options)
        model_class = get_class_from_dynamic_module('modeling.VLMQwenForCausalLM', source, **options)
        module = importlib.import_module(model_class.__module__)
        self.vision_tower = module.build_vision_tower(native)
        self.mm_projector = module.build_mm_projector(native)
        self.output_dim = native.hidden_size
        self.native_shape = (native.input_size[2], native.input_size[0], native.input_size[1])
        index = json.loads(checkpoint_file(source, 'model.safetensors.index.json', **options).read_text())['weight_map']
        prefixes = ('model.vision_tower.', 'model.mm_projector.')
        selected = {key: shard for key, shard in index.items() if key.startswith(prefixes)}
        if not selected:
            raise ValueError('The checkpoint must contain pretrained DCFormer and Mixer tensors.')
        state = {}
        for shard in sorted(set(selected.values())):
            with safe_open(str(checkpoint_file(source, shard, **options)), framework='pt', device='cpu') as handle:
                for key in selected:
                    if selected[key] == shard:
                        state[key.removeprefix('model.')] = handle.get_tensor(key)
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Batch-normalization statistics are part of the frozen pretrained encoder.
        return super().train(False)

    @torch.no_grad()
    def forward(self, stacks):
        # Prepared stacks: B,Z,RGB,H,W. Preserve the checkpoint's scalar CT input convention.
        rgb = stacks.permute(0, 2, 1, 3, 4)
        coefficients = rgb.new_tensor([.299, .587, .114]).view(1, 3, 1, 1, 1)
        volume = (rgb * coefficients).sum(1, keepdim=True)
        volume = F.interpolate(volume, size=self.native_shape, mode='trilinear', align_corners=False)
        features = self.vision_tower(volume.to(next(self.parameters()).dtype))
        native = self.mm_projector(features)
        axial = []
        for feature, stride in zip(features, (32, 64)):
            depth, height, width = [value // stride for value in self.native_shape]
            spatial = feature.reshape(len(feature), depth, height, width, feature.shape[-1])
            slices = spatial.mean((2, 3)).transpose(1, 2)
            axial.append(F.interpolate(slices, size=stacks.shape[1], mode='linear', align_corners=False).transpose(1, 2))
        return native, torch.cat(axial, -1)
