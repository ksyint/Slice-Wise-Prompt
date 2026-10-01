"""CUDA training and inference for slice-wise visual prompting."""

import argparse
import json
import torch as th
import torch
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from slice_prompt.processing import pack_rows, PromptDataset
from slice_prompt.experiments.training import (
    add_dict_to_argparser,
    create_model_and_data,
    data_defaults,
    model_defaults,
    training_defaults,
    require_cuda,
    TrainLoop,
    evaluate_records,
)
from slice_prompt.models.adaptation import QwenSliceVLM
from slice_prompt.commands import (
    export_manifest_cli,
    prepare_cli,
    segment_cli,
    run_recipe_cli,
    catalog_cli,
    read_config,
)


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='recipes/prompt_vqa.yaml')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', help='Resume the trainable weights and optimizer from last.pt.')
    parser.add_argument('--output', default='outputs/default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    add_dict_to_argparser(parser, dict(**model_defaults(), **data_defaults(), **training_defaults()))
    return parser


def train_main():
    args = create_argparser().parse_args()
    device = require_cuda(args.device)
    th.manual_seed(args.seed)
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = dict(**model_defaults(), **data_defaults(), **training_defaults())
    config.update(read_config(args.config))
    state = th.load(args.checkpoint, map_location=device, weights_only=False) if args.checkpoint else None
    if state:
        config = state['config'].copy()
    config.update({key: getattr(args, key) for key in config if getattr(args, key, None) is not None})
    model, data, tokenizer, classes, counts = create_model_and_data(config, args.manifest, device)
    from slice_prompt.records.schema import manifest_digest
    from slice_prompt.models.checkpoints import check_resume

    data_identity = manifest_digest(data.rows)
    if state:
        check_resume(state, config, classes, data_identity)
    loader = DataLoader(
        data,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=th.Generator().manual_seed(args.seed),
        num_workers=config['workers'],
        collate_fn=partial(pack_rows, tokenizer=tokenizer, classes=classes, max_length=config['max_length']),
    )
    loop = TrainLoop(model, loader, counts, config, device)
    loop.manifest_digest = data_identity
    if state:
        loop.restore(state)
    history = list(state.get('history', [])) if state else []
    for statistics in loop.run_loop():
        history.append(statistics)
        print(json.dumps(statistics), flush=True)
        loop.save(args.output, tokenizer, classes, history)


def train_cli():
    train_main()


def inference_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='predictions.jsonl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--language-model', help='Local Qwen directory override.')
    parser.add_argument('--vision-source', help='Local Med3DVLM directory override.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    device = require_cuda(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state['config'].copy()
    for key in ('cache_dir', 'language_model', 'vision_source'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.offline:
        config['offline'] = True
    classes = state['classes']
    model = QwenSliceVLM(config, len(classes)).to(device)
    model.restore_trainable(state['trainable'])
    model.eval()
    data = PromptDataset(args.manifest, slices=config['slices'], image_size=config['image_size'])
    names = sorted(classes, key=classes.get)
    records = []
    with torch.no_grad(), torch.autocast('cuda', dtype=model.dtype):
        for row in data:
            batch = pack_rows([row], model.tokenizer, classes, config['max_length'], training=False)
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
            }
            if row.get('closed', False):
                if not names:
                    raise ValueError('The checkpoint needs closed-answer classes for this record.')
                prediction = names[int(model(batch)['classes'].argmax(-1).item())]
            else:
                prediction = model.generate(batch, args.max_new_tokens)[0]
            records.append(
                dict(
                    patient_id=row.get('patient_id'),
                    prediction=prediction,
                    answer=row.get('answer'),
                    closed=row.get('closed', False),
                )
            )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    print(json.dumps(evaluate_records(records)))


def inference_cli():
    inference_main()


def eval_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate_records(records), indent=2))


def eval_cli():
    eval_main()


def download_weights_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='weights/cache')
    parser.add_argument('--language-model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--with-teacher', action='store_true', help='Download all 7B baseline shards for distillation.')
    parser.add_argument('--local-dir', help='Write portable med3d/ and qwen/ subdirectories.')
    args = parser.parse_args()
    from slice_prompt.models.adaptation import prepare_visual_artifacts, prepare_language_artifacts

    visual = str(Path(args.local_dir) / 'med3d') if args.local_dir else None
    language = str(Path(args.local_dir) / 'qwen') if args.local_dir else None
    prepared = {
        'visual': prepare_visual_artifacts(
            cache_dir=args.cache_dir,
            destination=visual,
            with_teacher=args.with_teacher,
        ),
        'language': prepare_language_artifacts(
            args.language_model,
            cache_dir=args.cache_dir,
            destination=language,
        ),
    }
    print(json.dumps(prepared, indent=2))


def download_weights_cli():
    download_weights_main()


def main():
    import sys

    from slice_prompt.records.schema import manifest_cli
    from slice_prompt.records.vocabulary import vocabulary_cli
    from slice_prompt.cache import cache_prepare_cli
    from slice_prompt.models.checkpoints import checkpoint_cli, export_adapter_cli
    from slice_prompt.experiments.inference import session_cli
    from slice_prompt.experiments.grouped import report_cli
    from slice_prompt.experiments.prepare_volumes import prepare_volumes_cli, cohort_cli, sampling_audit_cli
    from slice_prompt.experiments.validation import fit_validated_cli, detailed_report_cli
    from slice_prompt.experiments.validation import token_audit_cli, compare_preparation_cli
    from slice_prompt.experiments.ablation import feature_cache_cli, prompt_ablation_cli, occlusion_cli
    from slice_prompt.experiments.ablation import feature_audit_cli

    commands = {
        'export-adapter': export_adapter_cli,
        'feature-audit': feature_audit_cli,
        'token-audit': token_audit_cli,
        'compare-preparation': compare_preparation_cli,
        'prepare-detailed': prepare_volumes_cli,
        'cohorts': cohort_cli,
        'sampling-audit': sampling_audit_cli,
        'fit': fit_validated_cli,
        'report-detailed': detailed_report_cli,
        'feature-cache': feature_cache_cli,
        'prompt-ablation': prompt_ablation_cli,
        'occlusion': occlusion_cli,
        'manifest': manifest_cli,
        'vocabulary': vocabulary_cli,
        'prepare-cache': cache_prepare_cli,
        'checkpoint': checkpoint_cli,
        'infer-session': session_cli,
        'report': report_cli,
        'train': train_cli,
        'infer': inference_cli,
        'evaluate': eval_cli,
        'download': download_weights_cli,
        'export-manifest': export_manifest_cli,
        'prepare': prepare_cli,
        'segment': segment_cli,
        'recipe': run_recipe_cli,
        'catalog': catalog_cli,
    }
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print("Commands: " + ", ".join(commands))
        print("Use python prompt.py <command> --help for stage options.")
        return
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"Usage: python prompt.py <command> [options]\nCommands: {choices}")
    command = sys.argv.pop(1)
    commands[command]()


if __name__ == "__main__":
    main()
