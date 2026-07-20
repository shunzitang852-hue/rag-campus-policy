"""Export CSV and Markdown thesis tables from the experiment registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT
from src.experiments.registry import rebuild_registry
from src.experiments.thesis_export import default_export_dir, export_thesis_tables


DEFAULT_REGISTRY = DEFAULT_EXPERIMENT_ROOT / "experiment_registry.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Use the existing registry without scanning runs and legacy outputs first.",
    )
    args = parser.parse_args(argv)
    if not args.no_rebuild:
        rows = rebuild_registry(args.experiment_root, args.registry)
        print(f"registry_rows: {len(rows)}")
    elif not args.registry.exists():
        parser.error(f"registry does not exist: {args.registry}")
    output_dir = args.output_dir or default_export_dir(args.experiment_root)
    manifest = export_thesis_tables(args.registry, output_dir)
    print(f"output_dir: {output_dir}")
    print(f"registry_row_count: {manifest['registry_row_count']}")
    print(f"missing_cell_count: {manifest['missing_cell_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
