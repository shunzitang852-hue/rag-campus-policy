"""Validate generated student handbook parsing artifacts and write statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INTERIM_ROOT,
    DEFAULT_RAW_PATH,
    EXPECTED_RAW_SHA256,
    read_catalog,
    read_raw_lines,
    sha256_file,
)
from src.preprocessing.structure_parser import load_jsonl
from src.preprocessing.text_cleaner import DEFAULT_PROCESSED_ROOT


OFFICIAL_TITLE = "基于RAG技术的高校政策知识问答系统构建与检索策略优化研究"
WRONG_TITLE_FRAGMENT = (
    "基于RAG技术的高校政策知识问答系统构建与检索策略优化研究"
    + "——以杭州电子科技大学信息工程学院为例"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _ensure(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sum_policy_chars(policies: list[dict[str, Any]]) -> int:
    return sum(len(row.get("raw_text", "")) for row in policies)


def _source_span_chars(lines: list[str], policies: list[dict[str, Any]]) -> int:
    if not policies:
        return 0
    start = min(int(row["source_start_line"]) for row in policies)
    end = max(int(row["source_end_line"]) for row in policies)
    return len("\n".join(lines[start - 1 : end]))


def _load_clean_articles(processed_root: Path) -> list[dict[str, Any]]:
    path = processed_root / "hdu_ies_student_handbook_articles_clean.jsonl"
    return _read_jsonl(path)


def build_statistics(
    raw_path: Path,
    catalog_path: Path,
    interim_root: Path,
    processed_root: Path,
) -> dict[str, Any]:
    raw_lines = read_raw_lines(raw_path)
    raw_text = raw_path.read_text(encoding="utf-8")
    catalog = read_catalog(catalog_path)
    policies = _read_jsonl(interim_root / "hdu_ies_student_handbook_policies.jsonl")
    articles = _read_jsonl(interim_root / "hdu_ies_student_handbook_articles.jsonl")
    anomalies = _read_jsonl(interim_root / "parse_anomalies.jsonl")
    clean_articles = _load_clean_articles(processed_root)

    match_counts = Counter(row.get("title_match_status", "") for row in policies)
    parse_counts = Counter(row.get("parse_status", "") for row in articles)
    anomaly_counts = Counter(row.get("anomaly_type", "") for row in anomalies)
    cleaning_actions = Counter()
    for row in clean_articles:
        cleaning_actions.update(row.get("cleaning_actions", []))

    covered_chars = _sum_policy_chars(policies)
    span_chars = _source_span_chars(raw_lines, policies)
    clean_chars = sum(len(row.get("content_clean", "")) for row in clean_articles)
    raw_article_chars = sum(len(row.get("content_raw", "")) for row in articles)

    return {
        "raw_sha256": sha256_file(raw_path),
        "raw_total_lines": len(raw_lines),
        "raw_total_characters": len(raw_text),
        "catalog_policy_count": len(catalog),
        "split_policy_count": len(policies),
        "exact_title_match_count": match_counts.get("exact", 0),
        "fuzzy_title_match_count": match_counts.get("fuzzy", 0),
        "not_found_title_match_count": match_counts.get("not_found", 0),
        "empty_policy_file_count": sum(1 for row in policies if not row.get("raw_text", "").strip()),
        "chapter_count": parse_counts.get("chapter_heading", 0),
        "section_count": parse_counts.get("section_heading", 0),
        "article_count": parse_counts.get("article", 0),
        "clause_count": parse_counts.get("clause", 0),
        "item_count": parse_counts.get("item", 0) + parse_counts.get("cn_item", 0),
        "unnumbered_paragraph_count": parse_counts.get("paragraph", 0),
        "unclassified_paragraph_count": parse_counts.get("unclassified", 0),
        "anomaly_total_count": len(anomalies),
        "anomaly_type_counts": dict(sorted(anomaly_counts.items())),
        "raw_character_coverage": covered_chars / span_chars if span_chars else 0,
        "coverage_formula": "sum(len(policy.raw_text)) / len(raw text from first policy start line through last policy end line, joined with LF)",
        "raw_policy_characters": covered_chars,
        "raw_article_characters": raw_article_chars,
        "clean_article_characters": clean_chars,
        "cleaning_character_delta": clean_chars - raw_article_chars,
        "cleaning_action_counts": dict(sorted(cleaning_actions.items())),
        "has_body_text_outside_any_policy": covered_chars < span_chars,
        "generated_embedding_or_vector_files": False,
    }


def validate(
    raw_path: Path,
    catalog_path: Path,
    interim_root: Path,
    processed_root: Path,
    repo_root: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    actual_hash = sha256_file(raw_path)
    _ensure(actual_hash == expected_sha256, "raw_hash_mismatch", errors)

    policies = _read_jsonl(interim_root / "hdu_ies_student_handbook_policies.jsonl")
    articles = _read_jsonl(interim_root / "hdu_ies_student_handbook_articles.jsonl")
    clean_articles = _load_clean_articles(processed_root)

    _ensure(len(policies) == 32, "policy_count_not_32", errors)
    _ensure(len({row["policy_id"] for row in policies}) == len(policies), "policy_id_not_unique", errors)
    _ensure([row["sequence"] for row in policies] == list(range(1, len(policies) + 1)), "sequence_not_continuous", errors)
    _ensure(all(row.get("raw_text", "").strip() for row in policies), "empty_policy_text", errors)
    _ensure(all(int(row["source_start_line"]) <= int(row["source_end_line"]) for row in policies), "invalid_policy_line_span", errors)
    _ensure(
        all(int(policies[index]["source_end_line"]) < int(policies[index + 1]["source_start_line"]) for index in range(len(policies) - 1)),
        "policy_boundaries_overlap_or_unsorted",
        errors,
    )
    _ensure(len({row["record_id"] for row in articles}) == len(articles), "record_id_not_unique", errors)
    _ensure(all(row.get("policy_id") for row in articles), "article_missing_policy_id", errors)
    _ensure(all(row.get("content_raw", "") for row in articles), "article_missing_content_raw", errors)
    _ensure(all(int(row["source_start_line"]) <= int(row["source_end_line"]) for row in articles), "invalid_article_line_span", errors)
    _ensure(all(int(row["source_start_line"]) > 0 for row in articles), "non_positive_article_line", errors)
    _ensure(any(row["policy_id"] == "HDU_IES_SH_014" and row["title_match_status"] == "fuzzy" for row in policies), "fuzzy_title_not_recorded", errors)
    _ensure(len(clean_articles) == len(articles), "clean_article_count_mismatch", errors)

    combined_clean = "\n".join(row.get("content_clean", "") for row in clean_articles)
    for token in ["2023", "9号", "%", "〔"]:
        if token in "\n".join(row.get("content_raw", "") for row in articles):
            _ensure(token in combined_clean, f"cleaning_removed_token:{token}", errors)

    repo_text = "\n".join(
        path.read_text(encoding="utf-8-sig", errors="ignore")
        for path in repo_root.rglob("*")
        if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts and path.suffix.lower() in {".md", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".py", ".txt"}
    )
    _ensure(OFFICIAL_TITLE in repo_text, "official_title_not_found", errors)
    _ensure(WRONG_TITLE_FRAGMENT not in repo_text, "wrong_title_fragment_found", errors)

    vector_markers = ["*.sqlite3", "*.parquet", "*.faiss", "*.index"]
    generated_vector_files = []
    for pattern in vector_markers:
        generated_vector_files.extend((repo_root.parent / "datasets").rglob(pattern))
    _ensure(not generated_vector_files, "embedding_or_vector_file_found", errors)

    stats = build_statistics(raw_path, catalog_path, interim_root, processed_root)
    _ensure(stats["raw_character_coverage"] >= 0.99, "coverage_below_threshold", errors)
    return stats, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--expected-sha256", default=EXPECTED_RAW_SHA256)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    stats, errors = validate(
        raw_path=args.raw_path,
        catalog_path=args.catalog_path,
        interim_root=args.interim_root,
        processed_root=args.processed_root,
        repo_root=args.repo_root,
        expected_sha256=args.expected_sha256,
    )
    stats["validation_errors"] = errors
    stats_path = args.interim_root / "parsing_statistics.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"statistics_path={stats_path}")
    print(f"validation_error_count={len(errors)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
