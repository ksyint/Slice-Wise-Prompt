# Source volume and question records

Each source JSONL row contains `patient_id`, `volume`, `mask`, `question`, `answer`, and Boolean `closed`. Paths resolve from the manifest. Keep patient IDs stable across all questions and studies belonging to one person. For article-based PMC records, use the article ID as the grouping identity.

```bash
python prompt.py manifest --manifest data/source.jsonl --output reports/source.json
```

The reader validates nonempty text, Boolean question type, existing paths, unique question IDs, and optional positive `mask_labels`. Explicit `id` fields remain unchanged. Otherwise an ID is derived from the patient, volume, mask, question, and answer. See [source-record.json](../../schemas/source-record.json) for the portable shape.
