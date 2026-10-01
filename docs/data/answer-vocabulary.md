# Closed-answer vocabulary

Closed-answer classifier IDs are deterministic: exact training answers are sorted and numbered from zero. `vocabulary.json` records this mapping and each class count alongside the trained checkpoint.

```bash
python prompt.py vocabulary --manifest outputs/prepared/train.jsonl --prepared --output outputs/vocabulary.json
python prompt.py vocabulary --manifest outputs/prepared/val.jsonl --prepared --existing outputs/vocabulary.json --output outputs/validation-vocabulary.json
```

The comparison reports added and removed labels and changed IDs. Training and checkpoint resume use exact answers so existing class-head state remains compatible. The standalone vocabulary tool also offers NFC and casefold normalization for preparing a consistently normalized dataset before training.
