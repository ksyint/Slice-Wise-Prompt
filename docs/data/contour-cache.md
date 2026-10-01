# Reusable visual prompt cache

The contour cache identifies each volume, aligned mask, anatomical label selection, and rendering setting. Questions using the same visual prompt reuse one prepared array.

```bash
python prompt.py prepare-cache --manifest data/source.jsonl --cache data/prompt-cache --output outputs/prepared --color blue --thickness 1 --center -600 --width 1500 --png
```

Cache directories contain `prompt.npy`, `metadata.json`, and optional PNG slices. Metadata records source-file identities, array shape, contour voxel count, and intersecting slices. A second preparation run reuses entries whose files and rendering settings match. Conversation records are created for each question independently of the shared visual array.
