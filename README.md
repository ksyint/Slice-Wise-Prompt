# Enhancing 3D Medical Vision-Language Models with Slice-wise Visual Prompt

Independent implementation from the supplied paper by Soo Yong Kim and Seunghyeok Hong.

## Summary

Slice-wise anatomical contours connect a region-grounded CT question to explicit visual evidence. This repository implements offline prompt rendering, consistent visual/text color cues, patient-level splitting, relative slice attention, LoRA projections, class-balanced focal loss and autoregressive answer training. A small frozen CNN plus GRU makes the complete data/training/evaluation path executable without downloading foundation models.

The frozen CNN is randomly initialized and is an integration baseline; it is not DCFormer. Original DCFormer/MLP-Mixer, Qwen, VISTA3D, teacher checkpoints, SigLIP distillation and medical benchmark datasets are not included. No paper scores or claimed memory improvements are reproduced.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python train.py --smoke --output outputs/smoke
python eval.py --checkpoint outputs/smoke/last.pt --manifest outputs/smoke/synthetic/train.jsonl --output outputs/smoke/predictions.jsonl
```

The smoke run optimizes parameters on synthetic volumes. Reported scores on these same fixtures verify execution, not generalization.

## Prepare region-grounded data

Provide an existing segmentation mask. A source JSONL row is:

```json
{"volume":"ct/case.npy","mask":"masks/case.npy","patient_id":"case-001","question":"Is a finding present?","answer":"yes","closed":true}
```

NumPy input arrays use `(Z,Y,X)` in Hounsfield units with masks in exactly the same voxel grid. NIfTI inputs are canonicalized and transposed from `(X,Y,Z)` to `(Z,Y,X)`; volume and mask must have matching physical geometry before preparation. Shape equality alone does not establish registration. The tool preserves full volume depth; training uniformly samples the configured number of slices.

```bash
python prepare.py --manifest data/source.jsonl --output data/prepared --color blue --thickness 1 --split
python train.py --manifest data/prepared/train.jsonl --config configs/default.yaml --output outputs/run --device cuda
python eval.py --checkpoint outputs/run/last.pt --manifest data/prepared/test.jsonl
```

Preparation writes RGB `.npy` stacks `(Z,3,H,W)`, PNG slices and conversation JSONL. Use `--center` and `--width` for CT windowing. The default split is 70/10/20 by patient. For existing official splits, prepare each manifest separately without `--split`. Very small patient sets can produce empty partitions. Closed answers are preserved; open answers receive the chosen color prefix. External RadLex normalization is not fabricated.

## Method choices

Blue with a one-pixel inner boundary is the default; `--color red --thickness 3` exposes alternatives while keeping text and image consistent. Components are selected separately on each axial slice. An 8-connected erosion defines the boundary; optional dilation thickens it without filling the ROI interior. All disconnected components except the largest per slice are dropped.

MSSA uses four heads over concatenated slice/question tokens, scaled dot-product softmax, and a learned bias for relative slice positions. Padded text keys are masked. Attention weights are normalized with softmax. LoRA trains the QKV/output projections while their base weights remain frozen. In the compact baseline, the GRU, text embedding, heads, layer norm and relative bias also train; this does not claim the manuscript's 2.4% trainable-parameter figure. The standalone preliminary SSA block and placement inside Qwen's first k blocks are not implemented.

Closed rows optimize effective-number class-balanced focal loss (gamma 2); open rows optimize masked token NLL; a separate L2 term applies only to LoRA matrices. Evaluation reports closed accuracy and open exact match, not BLEU/METEOR. A larger pretrained visual/language stack is needed for meaningful medical evaluation.
