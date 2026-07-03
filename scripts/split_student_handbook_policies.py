"""Split the HDU IES student handbook into policy-level text files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INTERIM_ROOT,
    DEFAULT_RAW_PATH,
    EXPECTED_RAW_SHA256,
    run_split,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--expected-sha256", default=EXPECTED_RAW_SHA256)
    args = parser.parse_args()

    policies, anomalies = run_split(
        raw_path=args.raw_path,
        catalog_path=args.catalog_path,
        output_root=args.output_root,
        expected_sha256=args.expected_sha256,
    )
    print(f"split_policies={len(policies)}")
    print(f"split_anomalies={len(anomalies)}")
    print(f"output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
