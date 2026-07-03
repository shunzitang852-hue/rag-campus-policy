"""Parse policy-level files into line-traceable structure records and clean text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import DEFAULT_INTERIM_ROOT
from src.preprocessing.structure_parser import run_parse
from src.preprocessing.text_cleaner import DEFAULT_PROCESSED_ROOT, run_clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies-path",
        type=Path,
        default=DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_policies.jsonl",
    )
    parser.add_argument(
        "--anomalies-path",
        type=Path,
        default=DEFAULT_INTERIM_ROOT / "parse_anomalies.jsonl",
    )
    parser.add_argument(
        "--articles-path",
        type=Path,
        default=DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_articles.jsonl",
    )
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    args = parser.parse_args()

    records, anomalies = run_parse(
        policies_path=args.policies_path,
        anomalies_path=args.anomalies_path,
        articles_path=args.articles_path,
    )
    cleaning_summary = run_clean(
        articles_path=args.articles_path,
        policies_path=args.policies_path,
        output_root=args.processed_root,
    )
    print(f"article_records={len(records)}")
    print(f"anomalies={len(anomalies)}")
    print(f"cleaning_character_delta={cleaning_summary['cleaning_character_delta']}")
    print(f"processed_root={args.processed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
