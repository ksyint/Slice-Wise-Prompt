# Prepared pretrained directories

The normal training and inference commands automatically download the fixed Med3DVLM visual source and the selected Qwen2.5 decoder. Distillation loads the 7B baseline when its stage begins.

```bash
python prompt.py download --language-model Qwen/Qwen2.5-1.5B-Instruct --with-teacher --local-dir weights/pretrained --cache-dir weights/cache
python prompt.py train --manifest outputs/prepared/train.jsonl --vision-source weights/pretrained/med3d --language-model weights/pretrained/qwen --offline --output outputs/qwen15b --device cuda
```

Retain the Med3DVLM model-definition files and shard index as well as downloaded tensors. The root README links the model release and gives the VISTA3D bundle layout for manual segmentation-model preparation. Inference sessions accept the same local-path and offline overrides.
