"""Build the first QA test set for the HDU IES student handbook.

The script only creates evaluation data. It does not run embedding, vector
database, BM25, dense search, hybrid search, reranking, or RAG.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH, EXPECTED_RAW_SHA256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
CLEANED_PATH = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook" / "cleaned" / "hdu_ies_student_handbook_cleaned.txt"
ARTICLES_PATH = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "hdu_ies_student_handbook_articles_clean.jsonl"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
OUTPUT_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset"
CSV_PATH = OUTPUT_ROOT / "hdu_ies_student_handbook_qa_testset_v1.csv"
JSONL_PATH = OUTPUT_ROOT / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
REPORT_PATH = OUTPUT_ROOT / "hdu_ies_student_handbook_qa_testset_report.json"

QUESTION_TYPES = [
    "fact_lookup",
    "condition",
    "procedure",
    "responsible_body",
    "time_limit",
    "qualification",
    "comparison",
]
DIFFICULTIES = ["easy", "medium", "hard"]
ANSWER_GRANULARITIES = ["sentence", "paragraph", "multi_clause", "multi_policy"]
STRATEGIES = ["fixed_length", "article_based", "structure_aware"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_chunks_by_record() -> dict[str, dict[str, list[str]]]:
    mapping: dict[str, dict[str, list[str]]] = defaultdict(lambda: {strategy: [] for strategy in STRATEGIES})
    for strategy in STRATEGIES:
        path = CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
        chunks = load_jsonl(path)
        for chunk in chunks:
            for record_id in chunk.get("metadata", {}).get("source_record_ids", []):
                mapping[record_id][strategy].append(chunk["chunk_id"])
    return mapping


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def evidence_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        text = normalize_space(record.get("content_clean") or record.get("content_raw") or "")
        if record.get("parse_status") not in {"article", "clause", "item", "cn_item", "paragraph"}:
            continue
        if len(text) < 35 or len(text) > 420:
            continue
        if text == record.get("policy_title_canonical"):
            continue
        candidates.append(record)
    return candidates


def select_records(records: list[dict[str, Any]], count: int = 40) -> list[dict[str, Any]]:
    candidates = evidence_candidates(records)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_policy[record["policy_id"]].append(record)

    selected: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        if by_policy[policy_id]:
            selected.append(by_policy[policy_id][0])
        if len(selected) == count:
            return selected

    offset = 1
    while len(selected) < count:
        added = False
        for policy_id in sorted(by_policy):
            rows = by_policy[policy_id]
            if len(rows) > offset:
                selected.append(rows[offset])
                added = True
            if len(selected) == count:
                return selected
        if not added:
            break
        offset += 1
    return selected


def locate_evidence(cleaned_text: str, evidence: str, search_start: int = 0) -> tuple[int, int]:
    candidates = [evidence, evidence.replace(" ", "\u3000"), evidence.replace("\u3000", " ")]
    for candidate in candidates:
        index = cleaned_text.find(candidate, search_start)
        if index >= 0:
            return index, index + len(candidate)
    index = cleaned_text.find(evidence)
    if index >= 0:
        return index, index + len(evidence)
    return -1, -1


def question_for(record: dict[str, Any], question_type: str) -> str:
    title = record["policy_title_canonical"]
    article = record.get("article_number") or "相关条款"
    templates = {
        "fact_lookup": f"根据《{title}》，{article}规定的主要内容是什么？",
        "condition": f"根据《{title}》，{article}涉及哪些适用条件或情形？",
        "procedure": f"根据《{title}》，{article}体现了怎样的办理或处理要求？",
        "responsible_body": f"根据《{title}》，{article}中相关主体需要承担什么责任或义务？",
        "time_limit": f"根据《{title}》，{article}是否规定了时间、期限或阶段要求？具体是什么？",
        "qualification": f"根据《{title}》，{article}涉及哪些资格、条件或限制要求？",
        "comparison": f"根据《{title}》，{article}中不同情形或要求之间有什么区别？",
    }
    return templates[question_type]


def answer_granularity(evidence: str, question_type: str) -> str:
    if question_type == "comparison":
        return "multi_clause"
    if len(evidence) > 180:
        return "paragraph"
    if "；" in evidence or "，" in evidence:
        return "multi_clause"
    return "sentence"


def difficulty(evidence: str, question_type: str) -> str:
    if question_type in {"comparison", "procedure"} or len(evidence) > 220:
        return "hard"
    if question_type in {"condition", "qualification", "time_limit"} or len(evidence) > 120:
        return "medium"
    return "easy"


def evidence_keywords(record: dict[str, Any], evidence: str) -> list[str]:
    keywords = [record["policy_title_canonical"]]
    if record.get("article_number"):
        keywords.append(record["article_number"])
    for token in ["申请", "评定", "处分", "考试", "学籍", "奖学金", "学分", "期限", "管理", "学生"]:
        if token in evidence and token not in keywords:
            keywords.append(token)
    return keywords[:6]


def build_rows() -> list[dict[str, Any]]:
    raw_hash = sha256_file(DEFAULT_RAW_PATH)
    if raw_hash != EXPECTED_RAW_SHA256:
        raise RuntimeError(f"raw SHA-256 changed: {raw_hash}")
    if not CLEANED_PATH.exists() or CLEANED_PATH.stat().st_size == 0:
        raise RuntimeError(f"cleaned file is missing or empty: {CLEANED_PATH}")

    cleaned_text = CLEANED_PATH.read_text(encoding="utf-8-sig")
    records = load_jsonl(ARTICLES_PATH)
    chunk_map = load_chunks_by_record()
    selected = select_records(records, 40)
    if len(selected) != 40:
        raise RuntimeError(f"not enough candidate records: {len(selected)}")

    rows: list[dict[str, Any]] = []
    search_start = 0
    for index, record in enumerate(selected, start=1):
        evidence = normalize_space(record.get("content_clean") or record.get("content_raw") or "")
        start, end = locate_evidence(cleaned_text, evidence, search_start)
        if start < 0:
            start, end = locate_evidence(cleaned_text, evidence, 0)
        search_start = max(end, search_start)
        question_type = QUESTION_TYPES[(index - 1) % len(QUESTION_TYPES)]
        related = chunk_map.get(record["record_id"], {strategy: [] for strategy in STRATEGIES})
        row = {
            "question_id": f"HDU_IES_QA_{index:03d}",
            "question": question_for(record, question_type),
            "question_type": question_type,
            "expected_answer": evidence,
            "source_policy_title": record["policy_title_canonical"],
            "evidence_text": evidence,
            "evidence_start_char": start,
            "evidence_end_char": end,
            "evidence_keywords": evidence_keywords(record, evidence),
            "related_chunk_ids_fixed_length": related["fixed_length"],
            "related_chunk_ids_article_based": related["article_based"],
            "related_chunk_ids_structure_aware": related["structure_aware"],
            "difficulty": difficulty(evidence, question_type),
            "answer_granularity": answer_granularity(evidence, question_type),
            "notes": "基于学生手册cleaned正文和Chunk元数据生成；需人工复核问题表述和标准答案。",
        }
        rows.append(row)
    return rows


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key, value in list(result.items()):
        if isinstance(value, list):
            result[key] = ";".join(str(item) for item in value)
    return result


def write_outputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "question",
        "question_type",
        "expected_answer",
        "source_policy_title",
        "evidence_text",
        "evidence_start_char",
        "evidence_end_char",
        "evidence_keywords",
        "related_chunk_ids_fixed_length",
        "related_chunk_ids_article_based",
        "related_chunk_ids_structure_aware",
        "difficulty",
        "answer_granularity",
        "notes",
    ]
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_row(row) for row in rows)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    question_counter = Counter(row["question"] for row in rows)
    type_distribution = Counter(row["question_type"] for row in rows)
    difficulty_distribution = Counter(row["difficulty"] for row in rows)
    questions_per_policy = Counter(row["source_policy_title"] for row in rows)
    missing_evidence_count = sum(1 for row in rows if not row["evidence_text"].strip())
    missing_related_chunk_count = sum(
        1
        for row in rows
        if not (
            row["related_chunk_ids_fixed_length"]
            or row["related_chunk_ids_article_based"]
            or row["related_chunk_ids_structure_aware"]
        )
    )
    missing_all_strategy_chunk_count = sum(
        1
        for row in rows
        if not (
            row["related_chunk_ids_fixed_length"]
            and row["related_chunk_ids_article_based"]
            and row["related_chunk_ids_structure_aware"]
        )
    )
    report = {
        "stage": "qa_testset_v1",
        "csv_path": str(CSV_PATH),
        "jsonl_path": str(JSONL_PATH),
        "question_count": len(rows),
        "question_type_distribution": dict(type_distribution),
        "difficulty_distribution": dict(difficulty_distribution),
        "covered_policy_count": len(questions_per_policy),
        "covered_policy_titles": sorted(questions_per_policy),
        "questions_per_policy": dict(sorted(questions_per_policy.items())),
        "average_evidence_chars": sum(len(row["evidence_text"]) for row in rows) / len(rows),
        "missing_evidence_count": missing_evidence_count,
        "missing_related_chunk_count": missing_related_chunk_count,
        "missing_all_strategy_chunk_count": missing_all_strategy_chunk_count,
        "duplicate_question_count": sum(count - 1 for count in question_counter.values() if count > 1),
        "anomaly_count": missing_evidence_count + missing_related_chunk_count + sum(count - 1 for count in question_counter.values() if count > 1),
        "raw_sha256": sha256_file(DEFAULT_RAW_PATH),
        "cleaned_sha256": hashlib.sha256(CLEANED_PATH.read_bytes()).hexdigest().upper(),
        "notes": "本报告仅记录问答测试集构建结果；未运行检索、向量化或RAG流程。",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    rows = build_rows()
    report = write_outputs(rows)
    print(f"question_count={report['question_count']}")
    print(f"covered_policy_count={report['covered_policy_count']}")
    print(f"missing_evidence_count={report['missing_evidence_count']}")
    print(f"missing_related_chunk_count={report['missing_related_chunk_count']}")
    print(f"duplicate_question_count={report['duplicate_question_count']}")
    print(f"csv_path={CSV_PATH}")
    print(f"jsonl_path={JSONL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
