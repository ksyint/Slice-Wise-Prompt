"""One-epoch SigLIP-style representation distillation from the frozen 7B baseline."""
import torch
from torch import nn
import torch.nn.functional as F

from .weights import MED3D_ID, hub_options


class Med3DTeacher(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        source = config.get('vision_source', MED3D_ID)
        options = hub_options(config, visual=True)
        self.model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True,
                  torch_dtype=torch.bfloat16 if config['precision'] == 'bf16' else torch.float16,
                  **options).to(device).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **options)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def forward(self, native_visual, prompts):
        text = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(native_visual.device)
        embeddings = self.model.get_input_embeddings()(text['input_ids'])
        inputs = torch.cat((embeddings[:, :1], native_visual.to(embeddings.dtype), embeddings[:, 1:]), 1)
        valid = torch.cat((text['attention_mask'][:, :1], torch.ones(native_visual.shape[:2], device=inputs.device, dtype=torch.long), text['attention_mask'][:, 1:]), 1)
        hidden = self.model.get_model()(inputs_embeds=inputs, attention_mask=valid, use_cache=False, return_dict=True).last_hidden_state
        end = valid.sum(1) - 1
        return hidden[torch.arange(len(hidden), device=hidden.device), end].float()


def sigmoid_distillation(student, teacher, scale, bias):
    scores = F.normalize(student, dim=-1) @ F.normalize(teacher.detach(), dim=-1).T
    signs = torch.eye(len(scores), device=scores.device).mul(2).sub(1)
    return -F.logsigmoid(signs * (scale.exp().clamp(max=100) * scores + bias)).sum() / len(scores)
