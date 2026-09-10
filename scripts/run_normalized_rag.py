"""Run normalized extractive RAG experiments or validate the LLM batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.normalized_rag import GENERATION_VERSIONS, dry_run_llm, run_extractive, run_llm_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-version", required=True, choices=GENERATION_VERSIONS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-llm", action="store_true", help="Explicit opt-in for real external requests; never implied.")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry previously failed question IDs; default is disabled.")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    if args.generation_version == "llm_grounded_v1":
        if args.execute_llm:
            if not args.experiment_id:
                parser.error("--experiment-id is required with --execute-llm")
            result = run_llm_batch(args.experiment_id, execute_external_api=True, retry_failed=args.retry_failed)
        else:
            result = dry_run_llm()
    else:
        if args.execute_llm or args.dry_run:
            parser.error("extractive experiments do not accept --execute-llm or --dry-run")
        recorder = run_extractive(args.generation_version, experiment_id=args.experiment_id)
        result = {"experiment_id": recorder.experiment_id, "run_dir": str(recorder.run_dir), "status": "completed"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
