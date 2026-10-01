# Patient-level answer statistics

Closed-answer macro accuracy first averages questions within each patient, then averages those patient values. The optional bootstrap resamples complete patients and keeps all their closed questions together.

```bash
python prompt.py report --predictions outputs/qwen15b/session/predictions.jsonl --output reports/qwen15b --bootstrap 1000 --seed 42
```

Every closed record needs `patient_id` for patient bootstrap. The report records the patient count, seed, repetition count, and percentile interval. Keep patient assignments identical when comparing model sizes or visual-prompt settings. Open-answer BLEU, ROUGE-L, and METEOR remain available in the overall and grouped score tables.
