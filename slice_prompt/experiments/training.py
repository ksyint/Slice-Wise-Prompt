"""Mixed-precision slice prompting and answer metrics."""

import torch
import re
from slice_prompt.processing import PromptDataset
from slice_prompt.models.adaptation import (
    QwenSliceVLM,
    MED3D_ID,
    MED3D_REVISION,
    Med3DTeacher,
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
    from slice_prompt.records.schema import read_manifest
    from slice_prompt.records.vocabulary import AnswerVocabulary
    from slice_prompt.models.checkpoints import validate_training_config

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
        from slice_prompt.models.optimization import build_optimizer, make_schedule
        from slice_prompt.models.objectives import AnswerObjective

        if config.get('named_optimizer_groups', False):
            self.optimizer, self.optimizer_groups = build_optimizer(model, config)
        self.scheduler = make_schedule(self.optimizer, config, len(loader))
        self.objective = AnswerObjective(self.counts, config)
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['precision'] == 'fp16')
        self.step, self.start_epoch = 0, 0
        self.teacher = None

    def restore(self, state):
        self.model.restore_trainable(state['trainable'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.step, self.start_epoch = state['step'], state['epoch']
        from slice_prompt.models.checkpoints import restore_random_state

        restore_random_state(state.get('random_state'), self.loader)

    def forward_loss(self, batch):
        batch = {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()
        }
        with torch.autocast('cuda', dtype=self.model.dtype):
            output = self.model(batch)
            loss, self.last_losses = self.objective(self.model, output, batch, self.teacher)
        return loss

    def run_loop(self):
        from slice_prompt.models.monitor import EpochMonitor
        from slice_prompt.models.optimization import optimizer_update

        for epoch in range(self.start_epoch, self.config['epochs']):
            if epoch < self.config['distillation_epochs'] and self.teacher is None:
                self.teacher = Med3DTeacher(self.config, self.device)
            if epoch >= self.config['distillation_epochs'] and self.teacher is not None:
                self.teacher = None
                torch.cuda.empty_cache()
            monitor = EpochMonitor(self.device, epoch + 1)
            monitor.start()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            accumulation = self.config['accumulation_steps']
            for index, batch in enumerate(self.loader):
                loss = self.forward_loss(batch)
                monitor.record_loss(self.last_losses)
                block_size = min(accumulation, len(self.loader) - (index // accumulation) * accumulation)
                self.scaler.scale(loss / block_size).backward()
                losses.append(float(loss.detach()))
                if (index + 1) % accumulation == 0 or index + 1 == len(self.loader):
                    update = optimizer_update(
                        self.model, self.optimizer, self.scaler, self.scheduler,
                        self.config.get('max_grad_norm', 1.0),
                    )
                    monitor.record_update(update)
                    self.step += int(update['updated'])
            yield dict(
                epoch=epoch + 1, step=self.step, loss=sum(losses) / len(losses), lr=self.scheduler.get_last_lr()[0], monitor=monitor.finish()
            )

    def save(self, output, tokenizer, classes, history):
        from slice_prompt.models.checkpoints import save_training_checkpoint

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
