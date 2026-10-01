# Epoch checkpoint continuation

Training writes `last.pt` atomically after each complete epoch, together with tokenizer files, `vocabulary.json`, `history.json`, and a readable `checkpoint.json` manifest. The checkpoint stores trainable parameters, optimizer and scheduler state, scaler state, and random states for Python, NumPy, Torch, CUDA, and the DataLoader generator.

```bash
python prompt.py train --manifest outputs/prepared/train.jsonl --output outputs/qwen15b --device cuda
python prompt.py train --checkpoint outputs/qwen15b/last.pt --manifest outputs/prepared/train.jsonl --output outputs/qwen15b --epochs 30 --device cuda
python prompt.py checkpoint --directory outputs/qwen15b --verify-file
```

Resume checks model shape settings, class IDs, and the saved training-manifest digest. Existing epoch history remains available. FP16 schedules advance only after successful scaled optimizer updates.
