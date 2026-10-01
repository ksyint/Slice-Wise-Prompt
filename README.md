# Enhancing 3D Medical Vision-Language Models with Slice-wise Visual Prompt

Slice-wise anatomical contours connect region-grounded CT questions to visible spatial evidence. The pipeline renders prompt stacks, rewrites matching instructions, trains a frozen-CNN/MSSA/GRU VQA model and evaluates closed and open answers.

`slice_prompt/preprocess/` separates volume loading, contours, conversations and patient splits. `networks/` contains low-rank projections, relative slice attention and the VQA network. Dictionary factories connect preparation, training and inference.

## Install

Install CUDA-enabled PyTorch for your GPU and the project requirements.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available()"
```

Model commands accept `cuda` or `cuda:N`.

## Source volumes

Provide a CT volume, an aligned mask for the queried structure, and the question/answer pair:

```json
{"volume":"ct/case.npy","mask":"masks/case.npy","patient_id":"case-001","question":"Is a finding present?","answer":"yes","closed":true}
```

NumPy scalar volumes use `(Z,Y,X)` with Hounsfield-unit intensities. NIfTI volumes are canonicalized and transposed from `(X,Y,Z)` to axial `(Z,Y,X)` stacks. Volume and mask inputs must share the same physical grid.

```bash
python prepare.py --manifest data/source.jsonl --output data/prepared --color blue --thickness 1 --split
python train.py --manifest data/prepared/train.jsonl --config configs/prompt_vqa.yaml --output outputs/run --device cuda
python inference.py --checkpoint outputs/run/last.pt --manifest data/prepared/test.jsonl --output outputs/run/predictions.jsonl --device cuda
```

Preparation writes RGB arrays `(Z,3,H,W)`, PNG slices and conversation records. `--split` partitions patients 70/10/20; preparing existing official splits separately preserves their assignment. Training uniformly samples the configured number of slices. Closed answers retain their labels; open answers receive the selected color prefix.

## Paired preparation/training recipes

The catalog contains **256 executable recipes** connecting eight preparation and model settings:

| Setting | Values |
|---|---|
| Contour color | blue, red |
| Boundary thickness | 1, 3 pixels |
| CT window | lung: -600/1500; mediastinum: 40/400 |
| Sampled slices | 8, 16 |
| LoRA rank | 8, 16 |
| Learning rate | 0.0003, 0.001 |
| Training resolution | 32, 64 |
| Attention heads | 4, 8 |

Window values are center/width. Recipes are grouped by color, thickness, window and slice count under `recipes/volumetric/`.

```bash
python -m slice_prompt.recipes.catalog --list
python run_recipe.py --recipe RECIPE_PATH --manifest data/source.jsonl --output outputs/recipe_run --split --device cuda
python -m slice_prompt.recipes.catalog --compile
```

Use a path printed by `--list` as `RECIPE_PATH`. `run_recipe.py` applies all contour/window settings, writes the resulting training configuration, then trains on the prepared manifest. `--prepare-only` produces stacks and the resolved configuration for a separate training job.

## Attention and supervision

Contours use the largest 8-connected component on each slice and an inner morphological boundary. The selected color is shared by rasterization and text instructions. A frozen CNN produces slice embeddings; MSSA attends over concatenated visual/question tokens using scaled softmax attention and learned slice-relative bias. Padding keys are masked.

LoRA updates the attention projections. The compact decoder, embeddings, prediction heads, layer normalization and relative bias train alongside those updates. Closed questions use effective-number class-balanced focal loss; open questions use masked token likelihood. An L2 term regularizes LoRA matrices. Inference reports closed accuracy and open exact match.
