"""Pretrained DCFormer and Qwen with slice-conditioned attention."""

import importlib
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from contextlib import contextmanager

MED3D_ID = 'MagicXin/Med3DVLM-Qwen-2.5-7B'
MED3D_REVISION = '7d192aef4b06501a6ba191a474e6d684ca492824'


def hub_options(config, visual=False):
    return dict(
        cache_dir=config.get('cache_dir'),
        local_files_only=config.get('offline', False),
        revision=config.get('vision_revision', MED3D_REVISION) if visual else config.get('revision', 'main'),
    )


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
        coefficients = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1, 1)
        volume = (rgb * coefficients).sum(1, keepdim=True)
        volume = F.interpolate(volume, size=self.native_shape, mode='trilinear', align_corners=False)
        features = self.vision_tower(volume.to(next(self.parameters()).dtype))
        native = self.mm_projector(features)
        axial = []
        for feature, stride in zip(features, (32, 64)):
            depth, height, width = [value // stride for value in self.native_shape]
            spatial = feature.reshape(len(feature), depth, height, width, feature.shape[-1])
            slices = spatial.mean((2, 3)).transpose(1, 2)
            axial.append(
                F.interpolate(slices, size=stacks.shape[1], mode='linear', align_corners=False).transpose(1, 2)
            )
        return native, torch.cat(axial, -1)


class SliceTextFusion(nn.Module):
    def __init__(self, dim, heads, max_slices):
        super().__init__()
        if dim % heads:
            raise ValueError('Qwen width must divide into MSSA heads.')
        self.heads, self.max_slices = heads, max_slices
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_slices - 1))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, hidden, prefix, valid):
        batch, length, dim = hidden.shape
        qkv = self.qkv(self.norm(hidden)).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        bias = hidden.new_zeros(batch, self.heads, length, length)
        positions = torch.arange(prefix, device=hidden.device)
        relative = positions[:, None] - positions[None, :] + self.max_slices - 1
        bias[:, :, :prefix, :prefix] = self.relative_bias[:, relative]
        bias.masked_fill_(~(causal[None, None] & valid[:, None, None, :]), -torch.inf)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        output = attention.transpose(1, 2).reshape(batch, length, dim)
        return hidden + self.scale * self.out(output)


class SliceQuestionAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.visual_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, visual, question, question_mask):
        text = self.text_norm(question)
        fused, _ = self.attention(
            self.visual_norm(visual), text, text, key_padding_mask=~question_mask, need_weights=False
        )
        return visual + fused


class QwenSliceVLM(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
        from peft import LoraConfig
        from peft import get_peft_model

        self.config = config
        self.dtype = torch.bfloat16 if config['precision'] == 'bf16' else torch.float16
        options = hub_options(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config['language_model'], **options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        decoder = AutoModelForCausalLM.from_pretrained(
            config['language_model'], torch_dtype=self.dtype, attn_implementation='sdpa', **options
        )
        if decoder.config.model_type != 'qwen2':
            raise ValueError('Select a Qwen2.5 decoder checkpoint.')
        lora = LoraConfig(
            r=config['rank'],
            lora_alpha=config['rank'] * 2,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type='CAUSAL_LM',
        )
        self.decoder = get_peft_model(decoder, lora)
        self.visual = FrozenMed3D(config).to(dtype=self.dtype)
        dim = decoder.config.hidden_size
        self.bridge = nn.Linear(384 + 768, dim)
        self.global_context = nn.Linear(self.visual.output_dim, dim)
        self.ssa = SliceQuestionAttention(dim, config['heads'])
        self.mssa = nn.ModuleList(
            SliceTextFusion(dim, config['heads'], config['slices']) for _ in range(config['fusion_layers'])
        )
        if not 0 < len(self.mssa) <= decoder.config.num_hidden_layers:
            raise ValueError('fusion_layers must fit inside Qwen decoder depth.')
        self.classifier = nn.Linear(dim, max(1, classes))
        self.distill_projection = nn.Linear(dim, self.visual.output_dim)
        self.log_scale = nn.Parameter(torch.tensor(2.6593))
        self.log_bias = nn.Parameter(torch.tensor(-10.0))
        self._fusion_context = None
        self._hooks = [
            layer.register_forward_pre_hook(self._hook(index), with_kwargs=True)
            for index, layer in enumerate(decoder.model.layers[: len(self.mssa)])
        ]

    def _hook(self, index):
        def apply(module, args, kwargs):
            if self._fusion_context is None:
                return args, kwargs
            hidden = args[0] if args else kwargs['hidden_states']
            prefix, valid = self._fusion_context
            if hidden.shape[1] != valid.shape[1]:
                raise ValueError('MSSA requires the full token sequence and its current attention mask.')
            updated = self.mssa[index](hidden, prefix, valid)
            if args:
                return (updated,) + args[1:], kwargs
            return args, dict(kwargs, hidden_states=updated)

        return apply

    @contextmanager
    def fusion_context(self, prefix, valid):
        self._fusion_context = (prefix, valid.bool())
        try:
            yield
        finally:
            self._fusion_context = None

    def prepare_inputs(self, batch):
        native, axial = self.visual(batch['volumes'])
        # DCFormer feature grids retain axial order. The frozen Mixer supplies global context.
        visual = self.bridge(axial.float()) + self.global_context(native.float().mean(1))[:, None]
        text = self.decoder.get_input_embeddings()(batch['input_ids'])
        visual = self.ssa(visual, text.float(), batch['question_mask'])
        prefix = visual.shape[1]
        valid = torch.cat(
            (torch.ones(len(text), prefix, device=text.device, dtype=torch.long), batch['attention_mask']), 1
        )
        return torch.cat((visual.to(text.dtype), text), 1), valid, native

    def forward(self, batch):
        inputs, valid, native = self.prepare_inputs(batch)
        prefix = self.config['slices']
        with self.fusion_context(prefix, valid):
            hidden = (
                self.decoder.get_base_model()
                .model(inputs_embeds=inputs, attention_mask=valid, use_cache=False, return_dict=True)
                .last_hidden_state
            )
        # Last question position is independent of the teacher-forced answer.
        end = batch['question_mask'].sum(1) - 1 + prefix
        representation = hidden[torch.arange(len(hidden), device=hidden.device), end]
        return dict(
            logits=self.decoder.get_output_embeddings()(hidden[:, prefix:]),
            classes=self.classifier(representation.float()),
            representation=representation.float(),
            native=native,
        )

    @torch.no_grad()
    def generate(self, batch, max_new_tokens=128):
        """Greedily decode full prefixes so every new token passes through MSSA."""
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive.')
        inputs, valid, _ = self.prepare_inputs(batch)
        if inputs.device.type != 'cuda':
            raise ValueError('Slice-prompt generation requires CUDA.')
        eos = self.decoder.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        eos_ids = torch.tensor(eos_ids, device=inputs.device, dtype=torch.long)
        generated = []
        # Decode each unpadded prompt at its actual length, preserving axial positions.
        for row in range(inputs.shape[0]):
            sequence = inputs[row : row + 1, valid[row].bool()]
            tokens = []
            for _ in range(max_new_tokens):
                mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device)
                with self.fusion_context(self.config['slices'], mask):
                    hidden = (
                        self.decoder.get_base_model()
                        .model(inputs_embeds=sequence, attention_mask=mask, use_cache=False, return_dict=True)
                        .last_hidden_state
                    )
                logits = self.decoder.get_output_embeddings()(hidden[:, -1:])
                token = logits[:, -1].float().argmax(-1)
                tokens.append(token.item())
                if torch.isin(token, eos_ids).any().item():
                    break
                embedding = self.decoder.get_input_embeddings()(token[:, None])
                sequence = torch.cat((sequence, embedding.to(sequence.dtype)), dim=1)
            generated.append(tokens)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def lora_penalty(self):
        return sum(p.float().square().sum() for name, p in self.decoder.named_parameters() if 'lora_' in name)

    def trainable_state(self):
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}

    def restore_trainable(self, state):
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError('Checkpoint trainable tensors do not match the selected model configuration.')
        self.load_state_dict(state, strict=False)


class Med3DTeacher(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        source = config.get('vision_source', MED3D_ID)
        options = hub_options(config, visual=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                source,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if config['precision'] == 'bf16' else torch.float16,
                **options
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(source, **options)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def forward(self, native_visual, prompts):
        text = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
            native_visual.device
        )
        embeddings = self.model.get_input_embeddings()(text['input_ids'])
        inputs = torch.cat((embeddings[:, :1], native_visual.to(embeddings.dtype), embeddings[:, 1:]), 1)
        valid = torch.cat(
            (
                text['attention_mask'][:, :1],
                torch.ones(native_visual.shape[:2], device=inputs.device, dtype=torch.long),
                text['attention_mask'][:, 1:],
            ),
            1,
        )
        hidden = self.model.get_model()(
            inputs_embeds=inputs, attention_mask=valid, use_cache=False, return_dict=True
        ).last_hidden_state
        end = valid.sum(1) - 1
        return hidden[torch.arange(len(hidden), device=hidden.device), end].float()


def sigmoid_distillation(student, teacher, scale, bias):
    scores = F.normalize(student, dim=-1) @ F.normalize(teacher.detach(), dim=-1).T
    signs = torch.eye(len(scores), device=scores.device).mul(2).sub(1)
    return -F.logsigmoid(signs * (scale.exp().clamp(max=100) * scores + bias)).sum() / len(scores)


def class_balanced_focal(logits, labels, counts, gamma=2, beta=0.999):
    if (counts <= 0).any():
        raise ValueError('Each class needs a positive training count.')
    weights = (1 - beta) / (1 - beta ** counts.float())
    weights = weights / weights.mean()
    logp = F.log_softmax(logits, -1).gather(1, labels[:, None]).squeeze(1)
    return (-weights[labels] * (1 - logp.exp()).pow(gamma) * logp).mean()
