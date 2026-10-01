# Workflow guides

## Data

- [Closed-answer vocabulary](data/answer-vocabulary.md)
- [Reusable visual prompt cache](data/contour-cache.md)
- [Patient-disjoint partitions](data/patient-splits.md)
- [Prepared contour stacks](data/prepared-records.md)
- [Source volume and question records](data/source-schema.md)

## Evaluation

- [Grouped question-answer reports](experiments/grouped-vqa.md)
- [Patient-level answer statistics](experiments/patient-bootstrap.md)

## Runtime

- [Resumable answer generation](experiments/inference-session.md)
- [Prepared pretrained directories](experiments/offline-models.md)
- [Epoch checkpoint continuation](experiments/training-resume.md)

## Detailed preparation and evaluation

`prepare-detailed` records canonical voxel geometry, selected mask labels, connected components, intensity clipping, and axial region coverage beside every prompt array. It preserves closed-answer labels and uses an optional JSON anatomy vocabulary for explicit text substitutions. The default axial plan is uniform. Region and coverage sampling are available as named preparation variants.

```bash
python prompt.py prepare-detailed --manifest data/source.jsonl --output data/prepared --window lung --color red --slices 32 --split
python prompt.py sampling-audit --manifest data/source.jsonl --counts 16 32 --output outputs/sampling.json
python prompt.py cohorts --manifest data/prepared/train.jsonl --folds 5 --output data/folds
```

NPY inputs use Z,Y,X order. Supply `--npy-spacing z y x` when physical voxel spacing is available. NIfTI spacing and world coordinates are read from the canonical image grid. Optional component filtering is recorded in each preparation artifact. The original region is retained unless filtering is selected.

```bash
python prompt.py feature-cache --manifest data/prepared/train.jsonl --config recipes/prompt_vqa.yaml --cache outputs/visual-cache
python prompt.py fit --train-manifest data/prepared/train.jsonl --validation-manifest data/prepared/val.jsonl --output outputs/validated --visual-cache outputs/visual-cache
python prompt.py infer-session --checkpoint outputs/validated/best/last.pt --manifest data/prepared/test.jsonl --output outputs/test
python prompt.py report-detailed --predictions outputs/test/predictions.jsonl --output outputs/test/report.json
```

Frozen DCFormer and Mixer features are cached by source-array hash, model revision, precision, image size, slice count, and axial indices. Cache misses run the visual encoder on CUDA. Training records closed-answer, open-answer, distillation, and LoRA losses separately. Validation chooses `best/last.pt` using `--metric closed_accuracy` or `--metric open_token_f1`. `last.pt` in the run directory retains the most recent epoch for resume.

The detailed report includes per-answer confusion, confidence calibration, answer length, token overlap with and without the region prefix, and patient bootstrap intervals. `--corpus-metrics` adds BLEU, ROUGE-L, and METEOR using the installed lexical resources. `--baseline` compares the same question IDs with paired patient resampling.

```bash
python prompt.py prompt-ablation --manifest data/source.jsonl --output data/contour-variants --colors red blue --thicknesses 1 3
python prompt.py occlusion --checkpoint outputs/validated/best/last.pt --manifest data/prepared/test.jsonl --output outputs/occlusion --window 2
```

Occlusion measures changes in a fixed closed-answer probability as consecutive axial slices are replaced. Each record includes the original answer distribution, per-window probability change, and region concentration when ROI slice areas are present. Visual feature caching is disabled during these altered-volume evaluations.

The geometry report includes component volume, physical surface area, principal-axis spread, contour topology, and overlap before and after optional component filtering. These measurements use the supplied voxel spacing.

```bash
python prompt.py token-audit --manifest data/prepared/train.jsonl --output outputs/token-budget.json --require-fit
python prompt.py feature-audit --manifest data/prepared/train.jsonl --cache outputs/visual-cache --output outputs/features.json
python prompt.py compare-preparation --left data/contour-variants/red-1/train.jsonl --right data/contour-variants/blue-1/train.jsonl --output outputs/preparation-comparison.json
python prompt.py export-adapter --checkpoint outputs/validated/best/last.pt --output outputs/adapter
python prompt.py infer-session --checkpoint outputs/adapter --manifest data/prepared/test.jsonl --output outputs/portable-test
```

`export-adapter` writes the trained parameters, answer vocabulary, tokenizer files when present, and tensor checksums. Inference accepts either this directory or the training checkpoint. Qwen and the visual backbone are prepared from their configured model locations on first use. Use `download --local-dir weights --with-teacher` to prepare portable `weights/qwen` and `weights/med3d` directories. Visual shard selection comes from the checkpoint weight index. The teacher option includes the language weights used during distillation. Pass the local directories with `--language-model`, `--vision-source`, and `--offline` to `infer-session` after transferring those files.
