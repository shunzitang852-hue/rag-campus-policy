"""Evaluate v3 RAG answer quality with existing rule-based metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer_v3_optimization import evaluate_v3_quality


def main() -> int:
    report = evaluate_v3_quality()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
