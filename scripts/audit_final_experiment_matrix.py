"""Audit the frozen final retrieval matrix against the experiment registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.final_matrix import assess_matrix
from src.experiments.manifests import atomic_write_csv, atomic_write_json, atomic_write_text
from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT
from src.experiments.registry import rebuild_registry


def _markdown(rows: list[dict]) -> str:
    fields = [
        "phase", "chunk_strategy", "retrieval_method", "top_k",
        "dense_weight", "bm25_weight", "rrf_k", "status",
        "matching_experiment_id",
    ]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(field) is None else str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def audit(output_dir: Path, *, include_weight_sweep: bool = True) -> dict:
    rows = rebuild_registry(DEFAULT_EXPERIMENT_ROOT)
    assessments = assess_matrix(rows, include_weight_sweep=include_weight_sweep)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_csv(output_dir / "final_experiment_matrix_status.csv", assessments)
    atomic_write_json(output_dir / "final_experiment_matrix_status.json", assessments)
    missing = [row for row in assessments if row["status"] == "missing"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_row_count": len(rows),
        "expected_count": len(assessments),
        "completed_count": len(assessments) - len(missing),
        "missing_count": len(missing),
        "missing_spec_keys": [row["spec_key"] for row in missing],
    }
    atomic_write_json(output_dir / "audit_summary.json", summary)
    atomic_write_text(
        output_dir / "final_experiment_matrix_status.md",
        "# 最终实验矩阵状态\n\n"
        f"- 已完成：{summary['completed_count']}\n"
        f"- 缺失：{summary['missing_count']}\n"
        "- 判定口径：仅接受 completed、legacy=false、clean worktree、full reproducibility 的规范运行。\n\n"
        + _markdown(assessments),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or DEFAULT_EXPERIMENT_ROOT / "audit_exports" / stamp
    summary = audit(output_dir, include_weight_sweep=not args.main_only)
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

