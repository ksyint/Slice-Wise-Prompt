# Resumable answer generation

Use a prediction session for a manifest that may take several runs to complete. Each generated answer is appended and flushed before moving to the next question.

```bash
python prompt.py infer-session --checkpoint outputs/qwen15b/last.pt --manifest outputs/prepared/test.jsonl --output outputs/qwen15b/session --device cuda
python prompt.py infer-session --checkpoint outputs/qwen15b/last.pt --manifest outputs/prepared/test.jsonl --output outputs/qwen15b/session --device cuda --resume
```

`session.json` records the checkpoint checksum, normalized manifest digest, model overrides, and generation length. `answers.jsonl` is the durable journal. `predictions.jsonl` restores manifest order for reporting. Closed predictions include class confidence, and records retain anatomy, dataset, and question-type fields when provided.
