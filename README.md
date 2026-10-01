# Enhancing 3D Medical Vision-Language Models with Slice-wise Visual Prompt

[Paper](https://link.springer.com/chapter/10.1007/978-3-032-06004-4_20)

Slice-wise visual prompting with **pretrained DCFormer, the Med3DVLM dual-stream MLP–Mixer, and Qwen2.5 0.5B / 1.5B / 3B**. The default is Qwen2.5-1.5B-Instruct with rank-16 LoRA, four-head MSSA, and 32 sampled axial slices.

## Model and training

The frozen visual tower and Mixer are loaded strictly from the released [Med3DVLM checkpoint](https://huggingface.co/MagicXin/Med3DVLM-Qwen-2.5-7B), linked by its [official implementation](https://github.com/mirthAI/Med3DVLM). Loading extracts only visual tensors from the published shard index. The original 3584-dimensional Mixer remains frozen. Axially ordered low/high DCFormer feature grids supply slice embeddings, and a trainable bridge combines them with global Mixer context at the selected Qwen width.

Prepared RGB contour stacks are converted to luminance and resampled to the checkpoint's scalar `128×256×256` CT input. Question-conditioned SSA aligns slice tokens with the complete question. Four-head MSSA adds relative slice bias before the first two Qwen decoder blocks. Causal masks exclude future answer tokens. Qwen base weights remain frozen while its attention/MLP LoRA adapters, fusion layers, bridges and closed-answer head train.

Training combines class-balanced focal loss (`gamma=2`) for closed questions, answer-token NLL for open questions, and a LoRA-only L2 penalty (`1e-4`). AdamW uses betas `(0.9,0.98)`, LoRA-only weight decay `0.02`, 500 warmup steps and cosine decay over 25 epochs. The default first epoch also applies pairwise SigLIP-style representation distillation from the frozen Med3DVLM 7B decoder. Set `distillation_epochs: 0` for the supervised ablation. Peak learning rate is `2e-5` for 0.5B/1.5B and `1e-5` for the 3B setting.

## Install and obtain pretrained weights

Use Python 3.10+ and a CUDA-enabled PyTorch build. Model commands require `cuda` or `cuda:N`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

Model loading automatically downloads the pinned Med3DVLM visual checkpoint and the selected official [Qwen2.5 model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) into the Hugging Face cache. The 7B baseline downloads on entering the distillation stage. Med3DVLM's published custom model definitions are loaded through Transformers with the fixed revision in the YAML configuration.

To prepare a portable local copy ahead of training:

```bash
python prompt.py download --language-model Qwen/Qwen2.5-1.5B-Instruct \
  --with-teacher --local-dir weights/pretrained --cache-dir weights/cache
```

This writes `weights/pretrained/med3d/` and `weights/pretrained/qwen/`. Pass them with `--vision-source`, `--language-model`, and `--offline`. `--cache-dir` selects an alternative cache. The same overrides are available during inference. Selecting `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct` loads that actual pretrained decoder and constructs the matching-width fusion layers.

## Segment and prepare volumes

Obtain the grounded CT volumes, region masks and QA tables from the official [RadGenome-ChestCT release](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT). Keep the downloaded directory layout and join each QA row to its referenced volume and anatomy mask. For the CT subset of [PMC-VQA](https://huggingface.co/datasets/RadGenome/PMC-VQA), retain the image/article identifiers and provide an aligned ROI mask for each selected question. Use patient identity for volumetric grouping and article identity for PMC grouping.

`prompt.py export-manifest` converts a joined CSV with columns `volume,mask,question,answer,patient_id,closed` into source JSONL. `--columns` maps those names to the corresponding columns in a dataset export. Paths resolve from `--root`. The output stores absolute paths. An optional `mask_labels` column contains a JSON list of anatomical label IDs.

```bash
python prompt.py export-manifest --table data/region_qa.csv --root data/RadGenome-ChestCT \
  --output data/source.jsonl
python prompt.py export-manifest --table data/pmc_ct_roi.csv --root data/PMC-VQA \
  --columns '{"volume":"image","patient_id":"article_id"}' \
  --single-slice-images --output data/pmc_source.jsonl
```

The single-slice option preserves already-windowed image intensities in a `(1,Y,X)` NumPy array and stores the aligned mask on the same grid. Use the same `--center` and `--width` for export and preparation. This path represents a 2D CT figure as one axial observation. The volume loader samples it to the configured input length.

Source JSONL records contain `volume`, `mask`, `patient_id`, `question`, `answer`, and `closed`. Volumes/masks are NIfTI or NumPy scalar arrays. `.npy` uses `(Z,Y,X)`. Relative paths resolve from the manifest directory. Binary masks are accepted directly. `mask_labels` selects desired IDs from a multiclass mask.

```json
{"volume":"ct/case001.nii.gz","mask":"masks/case001.nii.gz","mask_labels":[3],"patient_id":"case001","question":"Is the spleen enlarged?","answer":"No","closed":true}
```

For automatic masks, supply `vista_labels` instead of `mask` using the official [VISTA3D label definitions](https://github.com/Project-MONAI/model-zoo/blob/dev/models/vista3d/configs/metadata.json):

```bash
pip install -r requirements-segmentation.txt
python prompt.py segment --manifest data/volumes.jsonl --output outputs/segmented \
  --bundle-dir weights/bundles --device cuda
python prompt.py prepare --manifest outputs/segmented/manifest.jsonl --output outputs/prepared \
  --color blue --thickness 1 --center -600 --width 1500 --split
```

`prompt.py segment` downloads the official MONAI `vista3d` bundle on first use, selects the requested anatomical labels and writes a mask-ready manifest. For offline use, place the complete [official bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/vista3d) in `weights/bundles/vista3d/`, including `models/model.pt`, `configs/` and `scripts/`, then add `--offline`.

NIfTI preparation first canonicalizes image orientation and resamples a differently aligned mask onto the image grid using nearest-neighbor interpolation. Arrays then use `(Z,Y,X)` order. Hounsfield intensities are clipped with the selected center/width before RGB contour rendering. Preparation renders boundaries on each intersecting slice, keeps the largest component, rewrites region instructions, and preserves closed-answer labels. It writes RGB NumPy stacks, PNG slices and conversation JSONL. `--split` creates patient-disjoint 70/10/20 train/validation/test partitions. Omit it to use existing dataset partitions. Already prepared masks can be passed directly to `prompt.py prepare`. Keep official partitions when supplied by the dataset. Keep them separate before splitting. For a fresh PMC article-level partition, the exported `patient_id` contains the article ID and the paper's 80/10/10 ratio can be selected explicitly:

```bash
python prompt.py prepare --manifest data/pmc_source.jsonl --output outputs/pmc_prepared \
  --color blue --thickness 1 --split --split-fractions 0.8 0.1 0.1
```

## Train, resume and evaluate

```bash
python prompt.py train --config configs/prompt_vqa.yaml --manifest outputs/prepared/train.jsonl \
  --cache-dir weights/cache --device cuda --output outputs/qwen15b
python prompt.py train --checkpoint outputs/qwen15b/last.pt --manifest outputs/prepared/train.jsonl \
  --vision-source weights/pretrained/med3d --language-model weights/pretrained/qwen \
  --offline --device cuda --output outputs/qwen15b
python prompt.py infer --checkpoint outputs/qwen15b/last.pt --manifest outputs/prepared/test.jsonl \
  --device cuda --output outputs/qwen15b/predictions.jsonl
python prompt.py evaluate --predictions outputs/qwen15b/predictions.jsonl
```

Checkpoints store trainable fusion/LoRA/head tensors, optimizer/scaler/scheduler state, epoch, model identifiers and closed-answer vocabulary. Inference reconstructs pretrained backbones, restores those tensors, then generates open answers autoregressively or predicts a closed class. Evaluation reports closed accuracy, BLEU-4, ROUGE-L and METEOR on supplied answers. Use `--batch-size` and `--accumulation-steps` to select the memory and effective-batch settings for the GPU.

## Preparation and decoder recipes

The **384 executable recipes** pair contour preparation with real pretrained training. Axes are blue/red contours, one/three-pixel borders, lung/mediastinum windows, 16/32 slices, all three Qwen2.5 sizes, rank 8/16, one/two fusion blocks and learning rates `1e-5`/`2e-5`. Every field is consumed by preparation, the pretrained model factory or optimizer.

```bash
python prompt.py catalog --list
python prompt.py recipe \
  --recipe recipes/volumetric/blue/border-1/lung/slices-32/qwen-1.5B/rank-16_fusion-2_lr-2e-05.yaml \
  --manifest data/source.jsonl --output outputs/recipe15b --split --device cuda
python prompt.py catalog --compile
```

`prompt.py recipe` renders the selected contours, writes the resolved training configuration, and launches the CUDA Qwen training entrypoint. `--prepare-only` writes the paired dataset/configuration for a later training run.
