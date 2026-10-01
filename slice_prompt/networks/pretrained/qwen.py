"""Frozen Med3DVLM vision, pretrained Qwen2.5 LoRA, and slice-aware fusion."""
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from .weights import FrozenMed3D, hub_options
from .fusion import SliceTextFusion, SliceQuestionAttention


class QwenSliceVLM(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.config = config
        self.dtype = torch.bfloat16 if config['precision'] == 'bf16' else torch.float16
        options = hub_options(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config['language_model'], **options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        decoder = AutoModelForCausalLM.from_pretrained(config['language_model'], torch_dtype=self.dtype,
                    attn_implementation='sdpa', **options)
        if decoder.config.model_type != 'qwen2':
            raise ValueError('Select a Qwen2.5 decoder checkpoint.')
        lora = LoraConfig(r=config['rank'], lora_alpha=config['rank'] * 2, lora_dropout=.05,
                          target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                          task_type='CAUSAL_LM')
        self.decoder = get_peft_model(decoder, lora)
        self.visual = FrozenMed3D(config).to(dtype=self.dtype)
        dim = decoder.config.hidden_size
        self.bridge = nn.Linear(384 + 768, dim)
        self.global_context = nn.Linear(self.visual.output_dim, dim)
        self.ssa = SliceQuestionAttention(dim, config['heads'])
        self.mssa = nn.ModuleList(SliceTextFusion(dim, config['heads'], config['slices'])
                                  for _ in range(config['fusion_layers']))
        if not 0 < len(self.mssa) <= decoder.config.num_hidden_layers:
            raise ValueError('fusion_layers must fit inside Qwen decoder depth.')
        self.classifier = nn.Linear(dim, max(1, classes))
        self.distill_projection = nn.Linear(dim, self.visual.output_dim)
        self.log_scale = nn.Parameter(torch.tensor(2.6593))
        self.log_bias = nn.Parameter(torch.tensor(-10.0))
        self._fusion_context = None
        self._hooks = [layer.register_forward_pre_hook(self._hook(index), with_kwargs=True)
                       for index, layer in enumerate(decoder.model.layers[:len(self.mssa)])]

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
        valid = torch.cat((torch.ones(len(text), prefix, device=text.device, dtype=torch.long), batch['attention_mask']), 1)
        return torch.cat((visual.to(text.dtype), text), 1), valid, native

    def forward(self, batch):
        inputs, valid, native = self.prepare_inputs(batch)
        prefix = self.config['slices']
        with self.fusion_context(prefix, valid):
            hidden = self.decoder.get_base_model().model(inputs_embeds=inputs, attention_mask=valid,
                                                         use_cache=False, return_dict=True).last_hidden_state
        # Last question position is independent of the teacher-forced answer.
        end = batch['question_mask'].sum(1) - 1 + prefix
        representation = hidden[torch.arange(len(hidden), device=hidden.device), end]
        return dict(logits=self.decoder.get_output_embeddings()(hidden[:, prefix:]),
                    classes=self.classifier(representation.float()), representation=representation.float(), native=native)

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
            sequence = inputs[row:row + 1, valid[row].bool()]
            tokens = []
            for _ in range(max_new_tokens):
                mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device)
                with self.fusion_context(self.config['slices'], mask):
                    hidden = self.decoder.get_base_model().model(
                        inputs_embeds=sequence, attention_mask=mask,
                        use_cache=False, return_dict=True).last_hidden_state
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
