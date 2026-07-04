"""Generate chunk datasets for the HDU IES student handbook."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import generate_chunks, load_jsonl, write_jsonl
from src.chunking.config import DEFAULT_CONFIG_PATH, get_strategy_config
from src.chunking.validation import write_statistics_outputs
from src.preprocessing.policy_splitter import DATASETS_ROOT, DEFAULT_RAW_PATH, EXPECTED_RAW_SHA256, sha256_file


DEFAULT_ARTICLES = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "hdu_ies_student_handbook_articles_clean.jsonl"
DEFAULT_OUTPUT_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
DEFAULT_CLEANED_TEXT = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook" / "cleaned" / "hdu_ies_student_handbook_cleaned.txt"
KNOWN_CLEANED_TITLE_MISMATCHES = {
    "杭州电子科技大学信息工程学院学生解除处分管理办法": "正文标题已记录为受控差异，不改写 cleaned 文件。",
}


def write_manifest(path: Path, strategy: str, config_path: Path, chunks_path: Path, stats: dict) -> None:
    payload = {
        "stage": "chunk_generation",
        "strategy": strategy,
        "config_path": str(config_path),
        "chunks_path": str(chunks_path),
        "chunk_total_count": stats["chunk_total_count"],
        "body_coverage_ratio": stats["body_coverage_ratio"],
        "uncovered_source_record_count": stats["uncovered_source_record_count"],
        "notes": "No embedding, vector database, BM25, Hybrid Search, RAG, or reranker artifacts were generated.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_sample(path: Path, chunks: list[dict], limit: int = 30) -> None:
    fields = ["chunk_id", "policy_id", "policy_title", "article_number", "chunk_char_count", "chunk_text"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for chunk in chunks[:limit]:
            writer.writerow({field: chunk.get(field, "") for field in fields})


def run_strategy(strategy: str, records: list[dict], config_path: Path, output_root: Path) -> dict:
    config = get_strategy_config(strategy, config_path)
    chunks = generate_chunks(records, config)
    rows = [chunk.to_dict() for chunk in chunks]
    output_dir = output_root / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
    root_chunks_path = output_root / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
    write_jsonl(chunks_path, rows)
    write_jsonl(root_chunks_path, rows)
    shutil.copyfile(config_path, output_dir / "chunk_config_snapshot.yaml")
    stats = write_statistics_outputs(output_dir, rows, records, config.__dict__)
    coverage = {
        "strategy": strategy,
        "coverage_basis": "unique parsed structure record IDs",
        "source_record_total_count": stats["source_record_total_count"],
        "covered_source_record_count": stats["covered_source_record_count"],
        "uncovered_source_record_count": stats["uncovered_source_record_count"],
        "duplicate_covered_source_record_count": stats["duplicate_covered_source_record_count"],
        "body_coverage_ratio": stats["body_coverage_ratio"],
        "duplicate_coverage_ratio": stats["duplicate_coverage_ratio"],
        "note": stats["coverage_note"],
    }
    (output_dir / "chunk_coverage_report.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(output_dir / "chunk_run_manifest.json", strategy, config_path, chunks_path, stats)
    write_sample(output_dir / "chunk_manual_sample.csv", rows)
    stats["chunks_path"] = str(root_chunks_path)
    return stats


def write_root_report(output_root: Path, summary: dict[str, dict], records: list[dict], cleaned_path: Path) -> None:
    report = {
        "stage": "chunking",
        "input_file": str(cleaned_path),
        "output_dir": str(output_root),
        "strategy_order": list(summary),
        "policy_count": len({record["policy_id"] for record in records}),
        "title_check_note": "31项目录规范标题在cleaned正文中逐字匹配；学生解除处分管理办法为已知正文标题差异，按受控异常处理。",
        "strategies": summary,
        "acceptance": {
            "empty_chunk_count_all_zero": all(item["empty_chunk_count"] == 0 for item in summary.values()),
            "covered_policy_count_is_32": all(item["covered_policy_count"] == 32 for item in summary.values()),
            "missing_policy_count_all_zero": all(item["missing_policy_count"] == 0 for item in summary.values()),
            "entered_embedding_stage": False,
            "created_vector_database": False,
            "ran_bm25": False,
            "ran_dense_search": False,
            "ran_hybrid_search": False,
            "ran_reranker": False,
            "ran_rag": False,
        },
    }
    (output_root / "hdu_ies_student_handbook_chunking_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_root_anomalies(output_root: Path, summary: dict[str, dict]) -> None:
    path = output_root / "hdu_ies_student_handbook_chunking_anomalies.csv"
    fields = ["strategy", "anomaly_type", "count", "detail"]
    rows: list[dict[str, object]] = []
    for strategy, stats in summary.items():
        checks = [
            ("empty_chunk", stats["empty_chunk_count"], "Chunk text must not be empty."),
            ("overlong_chunk", stats["overlong_chunk_count"], "Chunk length exceeded configured max_chars."),
            ("missing_policy", stats["missing_policy_count"], "; ".join(stats["missing_policy_titles"])),
            ("cross_policy_chunk", stats["cross_policy_chunk_count"], "Chunk contains records from multiple policies."),
        ]
        for anomaly_type, count, detail in checks:
            if count:
                rows.append(
                    {
                        "strategy": strategy,
                        "anomaly_type": anomaly_type,
                        "count": count,
                        "detail": detail,
                    }
                )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text_for_title_check(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles-path", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--cleaned-path", type=Path, default=DEFAULT_CLEANED_TEXT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--strategy", choices=["fixed_length", "article_based", "structure_aware", "all"], default="all")
    args = parser.parse_args()

    if sha256_file(DEFAULT_RAW_PATH) != EXPECTED_RAW_SHA256:
        raise RuntimeError("raw source SHA-256 changed; abort chunk generation")
    if not args.cleaned_path.exists() or args.cleaned_path.stat().st_size == 0:
        raise RuntimeError(f"cleaned input is missing or empty: {args.cleaned_path}")
    cleaned_text = args.cleaned_path.read_text(encoding="utf-8-sig")
    records = load_jsonl(args.articles_path)
    normalized_cleaned_text = normalize_text_for_title_check(cleaned_text)
    missing_titles = sorted(
        {
            record["policy_title_canonical"]
            for record in records
            if record.get("policy_title_canonical")
            and normalize_text_for_title_check(record["policy_title_canonical"]) not in normalized_cleaned_text
            and record["policy_title_canonical"] not in KNOWN_CLEANED_TITLE_MISMATCHES
        }
    )
    if missing_titles:
        raise RuntimeError(f"cleaned input does not contain policy titles: {missing_titles[:5]}")
    strategies = ["fixed_length", "article_based", "structure_aware"] if args.strategy == "all" else [args.strategy]
    summary = {}
    for strategy in strategies:
        stats = run_strategy(strategy, records, args.config_path, args.output_root)
        summary[strategy] = stats
        print(f"{strategy}: chunks={stats['chunk_total_count']} coverage={stats['body_coverage_ratio']}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "chunk_generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_root_report(args.output_root, summary, records, args.cleaned_path)
    write_root_anomalies(args.output_root, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
