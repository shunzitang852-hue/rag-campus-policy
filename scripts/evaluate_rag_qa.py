"""Run batch evaluation for the extractive RAG QA demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.evaluation import DEFAULT_CONFIG_PATH, run_batch_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--include-optional", action="store_true", help="Evaluate optional method/strategy combinations too.")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_batch_evaluation(
        config_path=args.config,
        include_optional=args.include_optional,
        output_root=args.output_root,
    )
    default_report = report["default_report"]
    print(f"question_count: {default_report.get('question_count')}")
    print(f"answer_non_empty_rate: {default_report.get('answer_non_empty_rate')}")
    print(f"citation_coverage_rate: {default_report.get('citation_coverage_rate')}")
    print(f"related_chunk_hit_rate: {default_report.get('related_chunk_hit_rate')}")
    print(f"output_root: {report['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
