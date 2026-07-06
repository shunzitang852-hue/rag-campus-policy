"""Evaluate generated RAG answer quality with rule-based review signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer_quality import DEFAULT_CONFIG_PATH, run_answer_quality_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_answer_quality_evaluation(config_path=args.config, output_root=args.output_root)
    summary = report["summary"]
    print(f"question_count: {summary['question_count']}")
    print(f"average_keyword_overlap_ratio: {summary['average_keyword_overlap_ratio']}")
    print(f"average_evidence_overlap_ratio: {summary['average_evidence_overlap_ratio']}")
    print(f"source_policy_hit_rate: {summary['source_policy_hit_rate']}")
    print(f"related_chunk_hit_rate: {summary['related_chunk_hit_rate']}")
    print(f"auto_quality_distribution: {summary['auto_quality_distribution']}")
    print(f"output_root: {report['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
