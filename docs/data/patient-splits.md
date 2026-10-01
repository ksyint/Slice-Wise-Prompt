# Patient-disjoint partitions

Split by patient before constructing training and held-out manifests. All questions from one volume and all studies from one patient belong to the same partition. Existing official partitions can be validated without reallocating their records.

```bash
python prompt.py manifest --manifest outputs/prepared/train.jsonl --prepared --compare outputs/prepared/val.jsonl outputs/prepared/test.jsonl
python prompt.py prepare-cache --manifest data/source.jsonl --cache data/prompt-cache --output outputs/prepared --split --fractions 0.7 0.1 0.2 --seed 42
```

The partition validator rejects repeated patients or question identities across splits. `--fractions 0.8 0.1 0.1` selects an article-level 80/10/10 partition when exported PMC records use article IDs as their grouping field.
