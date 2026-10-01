# Prepared contour stacks

Preparation produces float32 NumPy arrays in `[Z,3,H,W]` order with RGB values in [0,1]. Each prepared row points to its `prompt` array and retains the question, answer, patient, and source-volume identity.

```bash
python prompt.py manifest --manifest outputs/prepared/train.jsonl --prepared --inspect-arrays --output reports/prepared.json
```

The array inspection checks shape, dtype, finite values, intensity range, and reusable stack identities. Training samples the configured number of axial slices and resizes their RGB planes before the frozen visual tower's native volume conversion. Prepared paths can be absolute to share a cache across manifests.
