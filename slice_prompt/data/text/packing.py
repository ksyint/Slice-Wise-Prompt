"""Qwen chat packing with prompt-only fusion masks and answer-only labels."""
import torch


def pack_rows(rows, tokenizer, classes, max_length=1024, training=True):
    sequences, labels, prompts, prompt_lengths = [], [], [], []
    for row in rows:
        system = next((turn['content'] for turn in row.get('conversations', []) if turn['role'] == 'system'),
                      'Answer the medical question using the marked region of the CT volume.')
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system},
                   {'role': 'user', 'content': row['question']}], tokenize=False, add_generation_prompt=True)
        before = tokenizer(prompt, add_special_tokens=False)['input_ids']
        target = tokenizer(row.get('answer', ''), add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        if len(before) + (len(target) if training else 0) > max_length:
            raise ValueError('Question and answer exceed max_length; increase the configured context budget.')
        ids = before + target if training else before
        supervised = [-100] * len(before) + target if training and not row.get('closed', False) else [-100] * len(ids)
        sequences.append(ids); labels.append(supervised); prompts.append(prompt); prompt_lengths.append(len(before))
    length = max(map(len, sequences))
    tokens = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    targets = torch.full_like(tokens, -100)
    valid = torch.zeros_like(tokens)
    question = torch.zeros_like(tokens, dtype=torch.bool)
    for i, ids in enumerate(sequences):
        tokens[i, :len(ids)] = torch.tensor(ids)
        targets[i, :len(ids)] = torch.tensor(labels[i])
        valid[i, :len(ids)] = 1
        question[i, :prompt_lengths[i]] = True
    return dict(volumes=torch.stack([row['tensor'] for row in rows]), input_ids=tokens, labels=targets,
                attention_mask=valid, question_mask=question, prompts=prompts,
                class_labels=torch.tensor([classes.get(row.get('answer'), -1) if row.get('closed', False) else -1 for row in rows]))
