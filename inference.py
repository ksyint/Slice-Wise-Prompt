import argparse
import json
from pathlib import Path

import torch

from slice_prompt.script_util import require_cuda
from slice_prompt.networks.vqa import SlicePromptVLM
from slice_prompt.data.volumes import PromptDataset
from slice_prompt.data.vocabulary import Vocabulary
from slice_prompt.data.batching import collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='predictions.jsonl')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.device = require_cuda(args.device)
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    vocab, classes = Vocabulary(state['vocab']), state['classes']
    model = SlicePromptVLM(len(vocab.words), max(1, len(classes)), **{key: state['config'][key] for key in ('hidden_dim', 'heads', 'max_slices', 'rank')}).to(args.device)
    model.load_state_dict(state['model'])
    model.eval()
    data, records = PromptDataset(args.manifest, **{key: state['config'][key] for key in ('slices', 'image_size')}), []
    class_names = list(classes)
    for row in data:
        batch = collate([row], vocab, classes)
        logits, tokens = model.generate(batch['volumes'].to(args.device), batch['questions'].to(args.device))
        if row.get('closed', False) and not class_names:
            raise ValueError('Checkpoint has no closed-ended classes.')
        prediction = class_names[int(logits.argmax(-1))] if row.get('closed', False) else vocab.decode(tokens[0].tolist())
        records.append({'patient_id': row.get('patient_id'), 'prediction': prediction, 'answer': row['answer'],
                        'closed': row.get('closed', False), 'correct': Vocabulary.split(prediction) == Vocabulary.split(row['answer'])})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(''.join(json.dumps(row) + '\n' for row in records))
    metrics = {}
    for closed, name in [(True, 'closed_accuracy'), (False, 'open_exact_match')]:
        subset = [row for row in records if row['closed'] == closed]
        metrics[name] = sum(row['correct'] for row in subset) / len(subset) if subset else None
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
