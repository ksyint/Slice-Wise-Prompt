"""Autoregressive answer packing and closed-answer class indices."""
import torch


def collate(rows, vocab, classes):
    def pad(sequences):
        output = torch.zeros(len(rows), max(map(len, sequences)), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            output[index, :len(sequence)] = torch.tensor(sequence)
        return output
    answers = pad([[2] + vocab.encode(row['answer']) + [3] for row in rows])
    labels = [classes.get(row['answer'], -1) if row.get('closed', False) else -1 for row in rows]
    return {'volumes': torch.stack([row['tensor'] for row in rows]), 'questions': pad([vocab.encode(row['question']) for row in rows]),
            'inputs': answers[:, :-1], 'targets': answers[:, 1:], 'labels': torch.tensor(labels)}
