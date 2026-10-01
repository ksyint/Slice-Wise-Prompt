"""Download the official visual checkpoint and selected Qwen model into a reusable cache."""
import argparse
from huggingface_hub import snapshot_download
from slice_prompt.networks.pretrained.weights import MED3D_ID, MED3D_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='weights/cache')
    parser.add_argument('--language-model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--with-teacher', action='store_true', help='Download all 7B baseline shards for distillation.')
    parser.add_argument('--local-dir', help='Write portable med3d/ and qwen/ subdirectories.')
    args = parser.parse_args()
    patterns = None if args.with_teacher else ['*.json', '*.py', '*.txt', '*.model', 'merges.txt', 'model-00006-of-00007.safetensors']
    from pathlib import Path
    visual = str(Path(args.local_dir) / 'med3d') if args.local_dir else None
    language = str(Path(args.local_dir) / 'qwen') if args.local_dir else None
    print(snapshot_download(MED3D_ID, revision=MED3D_REVISION, allow_patterns=patterns, cache_dir=args.cache_dir, local_dir=visual))
    print(snapshot_download(args.language_model, cache_dir=args.cache_dir, local_dir=language))


if __name__ == '__main__':
    main()
