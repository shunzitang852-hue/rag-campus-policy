"""List and filter normalized and legacy experiments."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT
from src.experiments.registry import (
    filter_registry,
    load_registry,
    rebuild_registry,
)


DEFAULT_REGISTRY = DEFAULT_EXPERIMENT_ROOT / "experiment_registry.csv"
DISPLAY_FIELDS = [
    "experiment_id",
    "status",
    "task_type",
    "retrieval_method",
    "chunk_strategy",
    "top_k",
    "legacy",
    "completed_at",
]


def format_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    if not rows:
        return "No experiments found."
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in rows))
        for field in fields
    }
    header = "  ".join(field.ljust(widths[field]) for field in fields)
    divider = "  ".join("-" * widths[field] for field in fields)
    body = [
        "  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_rows(
    rows: list[dict[str, str]],
    output_format: str,
    *,
    detail: bool = False,
) -> str:
    if output_format == "json":
        return json.dumps(rows, ensure_ascii=False, indent=2)
    fields = list(rows[0]) if detail and rows else DISPLAY_FIELDS
    if output_format == "csv":
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )
        return output.getvalue().rstrip()
    return format_table(rows, fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--task-type")
    parser.add_argument("--retrieval-method")
    parser.add_argument("--chunk-strategy")
    parser.add_argument("--experiment-id")
    parser.add_argument("--format", choices=("table", "csv", "json"), default="table")
    args = parser.parse_args(argv)

    if args.rebuild or not args.registry.exists():
        rebuild_registry(args.experiment_root, args.registry)
    rows = filter_registry(
        load_registry(args.registry),
        task_type=args.task_type,
        retrieval_method=args.retrieval_method,
        chunk_strategy=args.chunk_strategy,
        experiment_id=args.experiment_id,
    )
    print(render_rows(rows, args.format, detail=bool(args.experiment_id)))
    return 0 if rows or not args.experiment_id else 1


if __name__ == "__main__":
    raise SystemExit(main())
