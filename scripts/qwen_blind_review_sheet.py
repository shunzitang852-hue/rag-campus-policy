"""Offline Qwen review sheet and score import; never read a blind identity key."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_qwen_cross_family import OUTPUT_ROOT, output_directory
from src.experiments.blind_review_sheet import build_sheet, import_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "import"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    output_directory(args.experiment_id)
    kwargs = {"root": OUTPUT_ROOT, "csv_name": "qwen_deepseek_manual_review.csv",
              "review_id": "review_" + hashlib.sha256(args.experiment_id.encode()).hexdigest()[:16],
              "allow_empty_answers": True}
    if args.command == "build":
        result = build_sheet(args.experiment_id, **kwargs)
    else:
        if args.scores is None:
            parser.error("--scores is required for import")
        result = import_scores(args.experiment_id, args.scores, **kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
