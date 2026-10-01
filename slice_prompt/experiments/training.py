"""Mixed-precision slice prompting and answer metrics."""

import math
import torch
import torch.nn.functional as F
import re
from slice_prompt.pipeline.volumes.processing import PromptDataset
from slice_prompt.models.qwen.adaptation import (
    QwenSliceVLM,
    MED3D_ID,
    MED3D_REVISION,
    class_balanced_focal,
    Med3DTeacher,
    sigmoid_distillation,
)


def model_defaults():
    return dict(
        language_model='Qwen/Qwen2.5-1.5B-Instruct',
        vision_source=MED3D_ID,
        vision_revision=MED3D_REVISION,
        revision='main',
        heads=4,
        rank=16,
        fusion_layers=2,
        precision='bf16',
        cache_dir=None,
        offline=False,
    )


def data_defaults():
    return dict(slices=32, image_size=256, max_length=1024)


def training_defaults():
    return dict(
        epochs=25,
        batch_size=4,
        lr=2e-5,
        lora_l2=1e-4,
        warmup_steps=500,
        weight_decay=0.02,
        distillation_epochs=1,
        accumulation_steps=1,
        workers=0,
    )


def args_to_dict(values, keys):
    return {key: values[key] for key in keys}


def add_dict_to_argparser(parser, defaults):
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument('--' + key.replace('_', '-'), action='store_true', default=None)
        else:
            parser.add_argument(
                '--' + key.replace('_', '-'), default=None, type=type(value) if value is not None else str
            )


def create_model_and_data(config, manifest, device):
    from slice_prompt.pipeline.volumes.records.schema import read_manifest
    from slice_prompt.pipeline.volumes.records.vocabulary import AnswerVocabulary
    from slice_prompt.models.qwen.checkpoints import validate_training_config

    validate_training_config(config)
    require_cuda(device)
    data = PromptDataset(manifest, slices=config['slices'], image_size=config['image_size'])
    data.rows = read_manifest(manifest, prepared=True)
    vocabulary = AnswerVocabulary.from_records(data.rows)
    classes = vocabulary.classes
    model = QwenSliceVLM(config, len(classes)).to(device)
    return model, data, model.tokenizer, classes, vocabulary.counts


def require_cuda(value='cuda'):
    import torch as th

    device = th.device(value)
    if device.type != 'cuda':
        raise ValueError('VQA model execution requires cuda or cuda:N.')
    if not th.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch and an accessible GPU are required.')
    return device


class TrainLoop:
    def __init__(self, model, loader, class_counts, config, device):
        self.device = require_cuda(device)
        self.model, self.loader, self.config = model, loader, config
        self.counts = torch.tensor(class_counts, device=device)
        lora, other = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (lora if 'lora_' in name else other).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [{'params': lora, 'weight_decay': config['weight_decay']}, {'params': other, 'weight_decay': 0.0}],
            lr=config['lr'],
            betas=(0.9, 0.98),
        )
        total = math.ceil(len(loader) / config['accumulation_steps']) * config['epochs']

        def factor(step):
            warmup = config['warmup_steps']
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16')
        self.step, self.start_epoch = 0, 0
        self.teacher = None

    def restore(self, state):
        self.model.restore_trainable(state['trainable'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.step, self.start_epoch = state['step'], state['epoch']
        from slice_prompt.models.qwen.checkpoints import restore_random_state

        restore_random_state(state.get('random_state'), self.loader)

    def forward_loss(self, batch):
        batch = {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()
        }
        with torch.autocast('cuda', dtype=self.model.dtype):
            output = self.model(batch)
            labels = batch['class_labels']
            closed = labels >= 0
            loss = output['classes'].sum() * 0
            if closed.any():
                loss = loss + class_balanced_focal(output['classes'][closed].float(), labels[closed], self.counts)
            shifted = batch['labels'][:, 1:]
            if (shifted != -100).any():
                loss = loss + F.cross_entropy(
                    output['logits'][:, :-1].float().reshape(-1, output['logits'].shape[-1]),
                    shifted.reshape(-1),
                    ignore_index=-100,
                )
            loss = loss + self.config['lora_l2'] * self.model.lora_penalty()
            if self.teacher is not None:
                targets = self.teacher(output['native'], batch['prompts'])
                projected = self.model.distill_projection(output['representation'])
                loss = loss + sigmoid_distillation(
                    projected.float(), targets, self.model.log_scale, self.model.log_bias
                )
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite VQA objective.')
        return loss

    def run_loop(self):
        for epoch in range(self.start_epoch, self.config['epochs']):
            if epoch < self.config['distillation_epochs'] and self.teacher is None:
                self.teacher = Med3DTeacher(self.config, self.device)
            if epoch >= self.config['distillation_epochs'] and self.teacher is not None:
                self.teacher = None
                torch.cuda.empty_cache()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            accumulation = self.config['accumulation_steps']
            for index, batch in enumerate(self.loader):
                loss = self.forward_loss(batch)
                block_size = min(accumulation, len(self.loader) - (index // accumulation) * accumulation)
                self.scaler.scale(loss / block_size).backward()
                losses.append(float(loss.detach()))
                if (index + 1) % accumulation == 0 or index + 1 == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    previous_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scaler.get_scale() >= previous_scale:
                        self.scheduler.step()
                        self.step += 1
                    self.optimizer.zero_grad(set_to_none=True)
            yield dict(
                epoch=epoch + 1, step=self.step, loss=sum(losses) / len(losses), lr=self.scheduler.get_last_lr()[0]
            )

    def save(self, output, tokenizer, classes, history):
        from slice_prompt.models.qwen.checkpoints import save_training_checkpoint

        save_training_checkpoint(self, output, tokenizer, classes, history)


def normalize(text):
    return re.findall(r'\w+', text.lower())


def evaluate_records(records):
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer

    labeled = [row for row in records if row.get('answer') is not None]
    closed = [row for row in labeled if row['closed']]
    opened = [row for row in labeled if not row['closed']]
    result = dict(
        closed_accuracy=(
            sum(normalize(row['prediction']) == normalize(row['answer']) for row in closed) / len(closed)
            if closed
            else None
        )
    )
    if opened:
        references = [[normalize(row['answer'])] for row in opened]
        predictions = [normalize(row['prediction']) for row in opened]
        scorer = RougeScorer(['rougeL'], use_stemmer=True)
        result['bleu4'] = corpus_bleu(references, predictions, smoothing_function=SmoothingFunction().method1)
        result['rougeL'] = sum(
            scorer.score(row['answer'], row['prediction'])['rougeL'].fmeasure for row in opened
        ) / len(opened)
        try:
            result['meteor'] = sum(
                meteor_score(reference, prediction) for reference, prediction in zip(references, predictions)
            ) / len(opened)
        except LookupError as error:
            raise RuntimeError(
                'Install the METEOR lexical resources with: python -m nltk.downloader wordnet omw-1.4'
            ) from error
    return result
