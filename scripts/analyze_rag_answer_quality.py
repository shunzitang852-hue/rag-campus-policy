"""Analyze v1 RAG answers that need manual review."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer_improvement import NEEDS_REVIEW_ANALYSIS_CSV, run_needs_review_analysis


def main() -> None:
    rows = run_needs_review_analysis()
    print(f"needs_review_count={len(rows)}")
    print(f"output={NEEDS_REVIEW_ANALYSIS_CSV}")


if __name__ == "__main__":
    main()
