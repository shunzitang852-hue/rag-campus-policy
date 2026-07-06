"""Build manual-review tables and a v2 RAG answer error analysis report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer_error_analysis import run_error_analysis


def main() -> int:
    report = run_error_analysis()
    summary = {
        "total_questions": report["total_questions"],
        "needs_review_count": report["needs_review_count"],
        "worsened_count": report["worsened_count"],
        "condition_case_count": report["condition_case_count"],
        "medium_case_count": report["medium_case_count"],
        "output_files": report["output_files"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
