"""Evaluate generated JSONL answers without loading a model."""
import argparse
import json
from pathlib import Path
from slice_prompt.evaluation.metrics import evaluate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate_records(records), indent=2))


if __name__ == '__main__':
    main()
