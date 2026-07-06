"""Generate and evaluate the first v2 rule-based RAG answer improvement run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer_improvement import run_improvement_evaluation


def main() -> None:
    report = run_improvement_evaluation()
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(f"report={report['output_files']['report_json']}")


if __name__ == "__main__":
    main()
