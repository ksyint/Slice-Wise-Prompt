# Grouped question-answer reports

Prediction records contain `prediction`, `answer`, Boolean `closed`, and stable question IDs. Optional `anatomy`, `dataset`, and `question_type` fields produce separate report tables.

```bash
python prompt.py report --predictions outputs/qwen15b/session/predictions.jsonl --output reports/qwen15b --groups anatomy dataset question_type
```

The report includes ordinary closed/open metrics, class-wise closed recall, an answer confusion table, and grouped scores. `answer_confusion.csv` records normalized target/prediction pairs. Supply `--compare outputs/other/session/predictions.jsonl` to export paired answers after checking identical question IDs and target labels.
