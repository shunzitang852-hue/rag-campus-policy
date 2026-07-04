"""Generate a conservative cleaned text file for the HDU IES student handbook.

This script does not rewrite policy content. It excludes the preface and table
of contents from the cleaned retrieval text, keeps the detected policy-body
order, and only collapses excessive blank lines.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.policy_splitter import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_RAW_PATH,
    EXPECTED_RAW_SHA256,
    find_body_search_start,
    read_catalog,
    read_raw_lines,
    split_policies,
)


OUTPUT_DIR = (
    PROJECT_ROOT.parent
    / "datasets"
    / "interim"
    / "hdu_ies"
    / "student_handbook"
    / "cleaned"
)
CLEANED_TEXT_PATH = OUTPUT_DIR / "hdu_ies_student_handbook_cleaned.txt"
REPORT_JSON_PATH = OUTPUT_DIR / "hdu_ies_student_handbook_cleaning_report.json"
ANOMALIES_CSV_PATH = OUTPUT_DIR / "hdu_ies_student_handbook_cleaning_anomalies.csv"
REPORT_MD_PATH = PROJECT_ROOT / "docs" / "data" / "hdu_ies_student_handbook_cleaning_report.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def char_count(lines: list[str]) -> int:
    if not lines:
        return 0
    return len("\n".join(lines))


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def collapse_blank_lines(text: str, max_blank_lines: int = 1) -> tuple[str, int]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    blank_run = 0
    removed_chars = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            output.append(line.rstrip())
            removed_chars += len(line) - len(line.rstrip())
            continue
        blank_run += 1
        if blank_run <= max_blank_lines:
            output.append("")
        else:
            removed_chars += 1
    return "\n".join(output).strip("\n") + "\n", removed_chars


def find_toc_range(lines: list[str], body_start_index: int) -> tuple[int, int]:
    first_catalog_like = None
    last_catalog_like = None
    for index in range(max(0, body_start_index)):
        line = lines[index].strip()
        if not line:
            continue
        if re.search(r"(\t|\s)\d+\s*$", line):
            if first_catalog_like is None:
                first_catalog_like = index
            last_catalog_like = index

    if first_catalog_like is None or last_catalog_like is None:
        return 0, max(0, body_start_index - 1)

    toc_start = first_catalog_like
    for index in range(first_catalog_like - 1, -1, -1):
        if lines[index].strip():
            toc_start = index
            break
    return toc_start, last_catalog_like


def policy_to_report(policy, catalog_entry) -> dict[str, object]:
    reference_number = policy.document_number.strip()
    year_clues = [item for item in re.findall(r"(?:19|20)\d{2}", policy.raw_text) if item]
    unique_years = list(dict.fromkeys(year_clues))
    return {
        "document_id": policy.policy_id,
        "title": policy.policy_title_original,
        "normalized_title": normalize_title(policy.policy_title_canonical),
        "catalog_title": policy.policy_title_canonical,
        "start_line": policy.source_start_line,
        "end_line": policy.source_end_line,
        "char_count": len(policy.raw_text),
        "whether_has_reference_number": bool(reference_number),
        "reference_number": reference_number,
        "whether_has_year": bool(unique_years),
        "year_clues": unique_years,
        "whether_matched_catalog": policy.sequence == catalog_entry.sequence
        and policy.title_match_status in {"exact", "fuzzy"},
        "title_match_status": policy.title_match_status,
    }


def write_anomalies_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "anomaly_type",
        "document_id",
        "catalog_title",
        "body_title",
        "line_number",
        "detail",
        "action",
    ]
    ANOMALIES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ANOMALIES_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, object]) -> None:
    stats = report["statistics"]
    anomaly_rows = report["anomalies"]
    missing_reference = report["documents_without_reference_number"]
    lines = [
        "# 杭电信工学生手册正文清洗报告",
        "",
        "## 一、清洗范围",
        "",
        "- 输入文件：`F:\\GraduationThesis\\datasets\\raw\\hdu_ies\\student_handbook\\hdu_ies_student_handbook_source.txt`",
        "- 输出文件：`F:\\GraduationThesis\\datasets\\interim\\hdu_ies\\student_handbook\\cleaned\\hdu_ies_student_handbook_cleaned.txt`",
        "- 清洗原则：不改写政策正文，只排除前言和目录，并折叠连续多余空行。",
        "- 前言处理：已单独识别为 `preface`，本轮不进入正式检索正文。",
        "- 目录处理：目录不进入 cleaned 正文，已记录删除行号范围和字符数。",
        "",
        "## 二、统计结果",
        "",
        "| 指标 | 数值 |",
        "| -- | --: |",
        f"| raw 总行数 | {stats['raw_total_lines']} |",
        f"| raw 总字符数 | {stats['raw_total_chars']} |",
        f"| cleaned 总行数 | {stats['cleaned_total_lines']} |",
        f"| cleaned 总字符数 | {stats['cleaned_total_chars']} |",
        f"| 删除目录字符数 | {stats['removed_toc_chars']} |",
        f"| 删除空白或格式字符数 | {stats['removed_blank_or_format_chars']} |",
        f"| 识别制度数量 | {stats['recognized_document_count']} |",
        f"| 目录匹配数量 | {stats['catalog_matched_count']} |",
        f"| 异常项数量 | {stats['anomaly_count']} |",
        "",
        "## 三、目录与正文对照",
        "",
        f"- 目录项数量：{stats['catalog_total_count']}",
        f"- 已识别正文制度数量：{stats['recognized_document_count']}",
        f"- 未匹配目录项：{'; '.join(report['unmatched_catalog_items']) if report['unmatched_catalog_items'] else '无'}",
        "",
        "## 四、人工核验项",
        "",
    ]
    if anomaly_rows:
        lines.extend(["| 类型 | 制度 | 说明 |", "| -- | -- | -- |"])
        for row in anomaly_rows:
            lines.append(f"| {row['anomaly_type']} | {row['catalog_title']} | {row['detail']} |")
    else:
        lines.append("未发现需要记录的异常项。")

    lines.extend(
        [
            "",
            "未识别文号制度：",
            "",
        ]
    )
    if missing_reference:
        lines.extend([f"- {item}" for item in missing_reference])
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 五、验收结论",
            "",
            "- 是否修改 raw 文件：否",
            "- 是否生成 cleaned 文件：是",
            "- 是否进入 Chunk：否",
            "- 是否进入 Embedding：否",
            "- 是否创建向量库：否",
            "- 是否做 RAG：否",
            "",
        ]
    )
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8-sig", newline="\n")


def main() -> None:
    raw_hash_before = sha256_file(DEFAULT_RAW_PATH)
    if raw_hash_before != EXPECTED_RAW_SHA256:
        raise RuntimeError(f"raw SHA-256 mismatch: {raw_hash_before}")

    raw_text = DEFAULT_RAW_PATH.read_text(encoding="utf-8")
    raw_lines = read_raw_lines(DEFAULT_RAW_PATH)
    catalog = read_catalog(DEFAULT_CATALOG_PATH)
    body_start_index = find_body_search_start(raw_lines, catalog)
    toc_start, toc_end = find_toc_range(raw_lines, body_start_index)
    preface_start, preface_end = 0, max(0, toc_start - 1)

    policies, split_anomalies = split_policies(raw_lines, catalog)
    policy_text = "\n\n".join(policy.raw_text.strip("\n") for policy in policies)
    cleaned_text, removed_blank_or_format_chars = collapse_blank_lines(policy_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_TEXT_PATH.write_text(cleaned_text, encoding="utf-8-sig", newline="\n")

    policy_reports = [
        policy_to_report(policy, catalog[policy.sequence - 1])
        for policy in policies
        if 1 <= policy.sequence <= len(catalog)
    ]
    matched_sequences = {item["document_id"] for item in policy_reports if item["whether_matched_catalog"]}
    unmatched_catalog_items = [
        entry.policy_title
        for entry in catalog
        if f"HDU_IES_SH_{entry.sequence:03d}" not in matched_sequences
    ]
    documents_without_reference_number = [
        item["catalog_title"] for item in policy_reports if not item["whether_has_reference_number"]
    ]

    anomaly_rows: list[dict[str, object]] = []
    for anomaly in split_anomalies:
        matching_policy = next((p for p in policies if p.policy_id == anomaly.policy_id), None)
        anomaly_rows.append(
            {
                "anomaly_type": anomaly.anomaly_type,
                "document_id": anomaly.policy_id,
                "catalog_title": matching_policy.policy_title_canonical if matching_policy else "",
                "body_title": matching_policy.policy_title_original if matching_policy else anomaly.original_text,
                "line_number": anomaly.source_start_line,
                "detail": anomaly.description,
                "action": anomaly.automatic_action,
            }
        )

    raw_hash_after = sha256_file(DEFAULT_RAW_PATH)
    cleaned_lines = cleaned_text.splitlines()
    removed_toc_chars = char_count(raw_lines[toc_start : toc_end + 1])
    report = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "input_file": str(DEFAULT_RAW_PATH),
        "output_cleaned_file": str(CLEANED_TEXT_PATH),
        "raw_sha256_before": raw_hash_before,
        "raw_sha256_after": raw_hash_after,
        "raw_modified": raw_hash_before != raw_hash_after,
        "excluded_sections": [
            {
            "section_type": "preface",
                "included_in_cleaned": False,
                "start_line": preface_start + 1,
                "end_line": preface_end + 1,
                "char_count": char_count(raw_lines[preface_start : preface_end + 1]),
                "reason": "前言为手册说明性文字，本轮不混入正式制度检索正文。",
            },
            {
                "section_type": "catalog",
                "included_in_cleaned": False,
                "start_line": toc_start + 1,
                "end_line": toc_end + 1,
                "char_count": removed_toc_chars,
                "reason": "目录不作为制度正文进入 cleaned 文件。",
            },
        ],
        "statistics": {
            "raw_total_lines": len(raw_lines),
            "raw_total_chars": len(raw_text),
            "cleaned_total_lines": len(cleaned_lines),
            "cleaned_total_chars": len(cleaned_text),
            "removed_toc_chars": removed_toc_chars,
            "removed_blank_or_format_chars": removed_blank_or_format_chars,
            "catalog_total_count": len(catalog),
            "recognized_document_count": len(policies),
            "catalog_matched_count": sum(1 for item in policy_reports if item["whether_matched_catalog"]),
            "unmatched_catalog_count": len(unmatched_catalog_items),
            "documents_without_reference_number_count": len(documents_without_reference_number),
            "anomaly_count": len(anomaly_rows),
        },
        "policies": policy_reports,
        "unmatched_catalog_items": unmatched_catalog_items,
        "documents_without_reference_number": documents_without_reference_number,
        "anomalies": anomaly_rows,
        "acceptance": {
            "raw_file_modified": False,
            "cleaned_file_generated": True,
            "entered_chunk_stage": False,
            "entered_embedding_stage": False,
            "created_vector_database": False,
            "ran_bm25": False,
            "ran_hybrid": False,
            "ran_rag": False,
            "ran_ocr": False,
        },
    }

    REPORT_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
        newline="\n",
    )
    write_anomalies_csv(anomaly_rows)
    write_markdown(report)

    print(f"raw_sha256={raw_hash_after}")
    print(f"raw_lines={len(raw_lines)}")
    print(f"raw_chars={len(raw_text)}")
    print(f"cleaned_lines={len(cleaned_lines)}")
    print(f"cleaned_chars={len(cleaned_text)}")
    print(f"recognized_documents={len(policies)}")
    print(f"catalog_matched={report['statistics']['catalog_matched_count']}")
    print(f"anomalies={len(anomaly_rows)}")
    print(f"cleaned_path={CLEANED_TEXT_PATH}")


if __name__ == "__main__":
    main()
