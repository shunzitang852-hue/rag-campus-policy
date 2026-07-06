"""Generate v3 targeted rule-based RAG answers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer_v3_optimization import V3_ANSWERS_JSONL, generate_v3_answers


def main() -> int:
    records = generate_v3_answers()
    print(json.dumps({"question_count": len(records), "output": str(V3_ANSWERS_JSONL)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
