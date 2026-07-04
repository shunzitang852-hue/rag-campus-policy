"""Chunk statistics and coverage validation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def chunk_stats(chunks: list[dict[str, Any]], records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    lengths = [int(chunk["chunk_char_count"]) for chunk in chunks]
    record_ids = {record["record_id"] for record in records if record.get("content_clean") or record.get("content_raw")}
    coverage_counter: Counter[str] = Counter()
    for chunk in chunks:
        coverage_counter.update(chunk.get("metadata", {}).get("source_record_ids", []))
    covered = set(coverage_counter)
    duplicate_records = {record_id for record_id, count in coverage_counter.items() if count > 1}
    policy_ids = {record["policy_id"] for record in records}
    policy_titles_by_id = {
        record["policy_id"]: record.get("policy_title_canonical", record["policy_id"])
        for record in records
    }
    chunk_policy_counts = Counter(chunk["policy_id"] for chunk in chunks)
    missing_policy_ids = sorted(policy_ids - set(chunk_policy_counts))

    stats = {
        "strategy": config["strategy"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "input_policy_count": len(policy_ids),
        "successful_policy_count": len(chunk_policy_counts),
        "chunk_total_count": len(chunks),
        "chunk_count": len(chunks),
        "chunk_count_by_policy": dict(sorted(chunk_policy_counts.items())),
        "average_char_count": statistics.mean(lengths) if lengths else 0,
        "avg_chars": statistics.mean(lengths) if lengths else 0,
        "min_char_count": min(lengths) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_char_count": max(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "median_char_count": statistics.median(lengths) if lengths else 0,
        "p25_char_count": percentile(lengths, 0.25),
        "p75_char_count": percentile(lengths, 0.75),
        "p90_char_count": percentile(lengths, 0.90),
        "p95_char_count": percentile(lengths, 0.95),
        "too_short_chunk_count": sum(1 for value in lengths if value < config["min_chunk_size"]),
        "too_long_chunk_count": sum(1 for value in lengths if value > config["max_chunk_size"]),
        "overlong_chunk_count": sum(1 for value in lengths if value > config["max_chunk_size"]),
        "empty_chunk_count": sum(1 for chunk in chunks if not chunk["chunk_text"].strip()),
        "duplicate_chunk_count": len(chunks) - len({chunk["chunk_text"] for chunk in chunks}),
        "missing_policy_title_count": sum(1 for chunk in chunks if not chunk.get("policy_title")),
        "missing_article_number_count": sum(1 for chunk in chunks if not chunk.get("article_number")),
        "missing_source_file_count": sum(1 for chunk in chunks if not chunk.get("source_file")),
        "missing_source_page_count": sum(1 for chunk in chunks if chunk.get("source_page_start") is None or chunk.get("source_page_end") is None),
        "cross_policy_chunk_count": sum(1 for chunk in chunks if len({rid.split("_R")[0] for rid in chunk.get("metadata", {}).get("source_record_ids", [])}) > 1),
        "cross_article_chunk_count": sum(1 for chunk in chunks if len(set(chunk.get("metadata", {}).get("article_numbers", []))) > 1),
        "source_record_total_count": len(record_ids),
        "covered_source_record_count": len(covered & record_ids),
        "uncovered_source_record_count": len(record_ids - covered),
        "duplicate_covered_source_record_count": len(duplicate_records & record_ids),
        "body_coverage_ratio": (len(covered & record_ids) / len(record_ids)) if record_ids else 0,
        "covered_char_ratio": (len(covered & record_ids) / len(record_ids)) if record_ids else 0,
        "duplicate_coverage_ratio": (len(duplicate_records & record_ids) / len(record_ids)) if record_ids else 0,
        "covered_policy_count": len(set(chunk_policy_counts)),
        "missing_policy_count": len(missing_policy_ids),
        "missing_policy_titles": [policy_titles_by_id[item] for item in missing_policy_ids],
        "anomaly_count": 0,
        "coverage_note": "Coverage is based on unique parsed structure record IDs, not summed chunk characters; overlap is tracked separately.",
    }
    stats["anomaly_count"] = (
        stats["empty_chunk_count"]
        + stats["too_long_chunk_count"]
        + stats["missing_policy_count"]
        + stats["cross_policy_chunk_count"]
    )
    return stats


def write_statistics_outputs(output_dir: Path, chunks: list[dict[str, Any]], records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    stats = chunk_stats(chunks, records, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunk_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [{"policy_id": key, "chunk_count": value} for key, value in stats["chunk_count_by_policy"].items()]
    write_csv(output_dir / "chunk_policy_statistics.csv", rows, ["policy_id", "chunk_count"])
    warnings = []
    if stats["uncovered_source_record_count"]:
        warnings.append({"warning_type": "uncovered_records", "count": stats["uncovered_source_record_count"]})
    if stats["cross_policy_chunk_count"]:
        warnings.append({"warning_type": "cross_policy_chunk", "count": stats["cross_policy_chunk_count"]})
    if stats["empty_chunk_count"]:
        warnings.append({"warning_type": "empty_chunk", "count": stats["empty_chunk_count"]})
    with (output_dir / "chunk_warnings.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for warning in warnings:
            handle.write(json.dumps(warning, ensure_ascii=False) + "\n")
    return stats
