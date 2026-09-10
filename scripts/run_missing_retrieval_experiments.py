"""Run only missing experiments from the frozen retrieval protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_normalized_bm25_smoke import run_normalized_bm25_smoke
from scripts.run_normalized_real_dense_smoke import run_normalized_real_dense_smoke
from scripts.run_normalized_real_hybrid import run_normalized_real_hybrid
from src.experiments.final_matrix import assess_matrix
from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT
from src.experiments.registry import rebuild_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_clean_worktree() -> None:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain=v1"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip():
        raise RuntimeError("formal experiments require a clean Git worktree")


def _signature(row: dict) -> tuple:
    return (
        row["chunk_strategy"], row["retrieval_method"], row["top_k"],
        row.get("dense_weight"), row.get("bm25_weight"), row.get("rrf_k"),
    )


def run_missing(*, include_weight_sweep: bool = True, dry_run: bool = False) -> dict:
    rows = rebuild_registry(DEFAULT_EXPERIMENT_ROOT)
    assessment = assess_matrix(rows, include_weight_sweep=include_weight_sweep)
    missing = [row for row in assessment if row["status"] == "missing"]
    unique_missing: list[dict] = []
    seen: set[tuple] = set()
    for row in missing:
        signature = _signature(row)
        if signature not in seen:
            seen.add(signature)
            unique_missing.append(row)
    if dry_run:
        return {"missing": unique_missing, "executed": []}
    _require_clean_worktree()
    executed: list[dict] = []
    for spec in unique_missing:
        method = spec["retrieval_method"]
        common = {
            "chunk_strategy": spec["chunk_strategy"],
            "top_k": int(spec["top_k"]),
            "random_seed": 42,
        }
        if method == "bm25":
            result = run_normalized_bm25_smoke(**common)
        elif method == "real_dense":
            result = run_normalized_real_dense_smoke(
                **common, candidate_pool_k=50, offline=True
            )
        elif method == "real_hybrid_weighted":
            result = run_normalized_real_hybrid(
                **common,
                fusion_method="weighted",
                candidate_pool_k=50,
                dense_weight=float(spec["dense_weight"]),
                bm25_weight=float(spec["bm25_weight"]),
                offline=True,
            )
        elif method == "real_hybrid_rrf":
            result = run_normalized_real_hybrid(
                **common,
                fusion_method="rrf",
                candidate_pool_k=50,
                rrf_k=int(spec["rrf_k"]),
                offline=True,
            )
        else:  # pragma: no cover - frozen matrix guards this
            raise ValueError(f"unsupported retrieval method: {method}")
        executed.append(
            {
                "spec_key": spec["spec_key"],
                "experiment_id": result["experiment_id"],
                "Recall@5": result["metrics"].get("Recall@5"),
                "MRR@5": result["metrics"].get("MRR@5"),
            }
        )
    final_rows = rebuild_registry(DEFAULT_EXPERIMENT_ROOT)
    final_assessment = assess_matrix(final_rows, include_weight_sweep=include_weight_sweep)
    remaining = [row for row in final_assessment if row["status"] == "missing"]
    return {"missing": unique_missing, "executed": executed, "remaining": remaining}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_missing(
        include_weight_sweep=not args.main_only,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.dry_run or not result.get("remaining") else 1


if __name__ == "__main__":
    raise SystemExit(main())

