"""Rule-based error analysis for v2 RAG answer quality review outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.rag.answer_improvement import DEFAULT_IMPROVEMENT_ROOT, DEFAULT_V1_REVIEW_JSONL, V2_REVIEW_JSONL
from src.rag.answer_quality import DEFAULT_QA_JSONL, load_qa_questions
from src.retrieval.dense_index import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
DEFAULT_ERROR_ANALYSIS_ROOT = EXPERIMENT_ROOT / "rag_answer_quality_error_analysis"

NEEDS_REVIEW_MANUAL_CHECK_CSV = DEFAULT_ERROR_ANALYSIS_ROOT / "rag_v2_needs_review_manual_check.csv"
WORSENED_CASES_MANUAL_CHECK_CSV = DEFAULT_ERROR_ANALYSIS_ROOT / "rag_v2_worsened_cases_manual_check.csv"
CONDITION_MEDIUM_FOCUS_CSV = DEFAULT_ERROR_ANALYSIS_ROOT / "rag_v2_condition_medium_focus_cases.csv"
ERROR_ANALYSIS_REPORT_JSON = DEFAULT_ERROR_ANALYSIS_ROOT / "rag_answer_error_analysis_v2_report.json"

MANUAL_CHECK_FIELDS = [
    "question_id",
    "question",
    "question_type",
    "difficulty",
    "expected_answer",
    "reference_answer",
    "retrieved_evidence",
    "evidence_text",
    "generated_answer",
    "auto_quality",
    "keyword_overlap_ratio",
    "evidence_overlap_ratio",
    "v1_quality",
    "v2_quality",
    "change_type",
    "possible_error_type",
    "is_condition_focus",
    "is_medium_focus",
    "manual_review_result",
    "manual_error_type",
    "manual_notes",
]

ERROR_TYPES = {
    "evidence_missing",
    "evidence_partially_used",
    "condition_not_fully_covered",
    "answer_too_general",
    "answer_too_short",
    "answer_contains_extra_info",
    "evaluation_metric_limitation",
    "unknown",
}

LABEL_SCORE = {"insufficient": 0, "needs_review": 1, "good": 2}


def _quality_label(record: dict[str, Any]) -> str:
    return str(record.get("auto_quality_label", record.get("auto_quality", "")) or "")


def classify_change(v1_record: dict[str, Any] | None, v2_record: dict[str, Any]) -> str:
    if not v1_record:
        return "unknown"
    v1_score = LABEL_SCORE.get(_quality_label(v1_record), 0)
    v2_score = LABEL_SCORE.get(_quality_label(v2_record), 0)
    if v2_score > v1_score:
        return "improved"
    if v2_score < v1_score:
        return "worsened"
    delta = float(v2_record.get("keyword_overlap_ratio", 0.0) or 0.0) - float(v1_record.get("keyword_overlap_ratio", 0.0) or 0.0)
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "worsened"
    return "unchanged"


def classify_possible_error(record: dict[str, Any], *, change_type: str = "unknown") -> str:
    question_type = str(record.get("question_type", "") or "")
    difficulty = str(record.get("difficulty", "") or "")
    keyword_overlap = float(record.get("keyword_overlap_ratio", 0.0) or 0.0)
    evidence_overlap = float(record.get("evidence_overlap_ratio", 0.0) or 0.0)
    answer_len = int(record.get("answer_char_count", len(str(record.get("generated_answer", "") or ""))) or 0)
    missing_keywords = record.get("possible_missing_keywords", []) or []
    unsupported_terms = record.get("possible_unsupported_terms", []) or []

    if question_type == "condition" and (keyword_overlap < 0.55 or len(missing_keywords) >= 3):
        return "condition_not_fully_covered"
    if answer_len < 180 and keyword_overlap < 0.5:
        return "answer_too_short"
    if evidence_overlap < 0.45:
        return "evidence_missing"
    if evidence_overlap < 0.75:
        return "evidence_partially_used"
    if keyword_overlap < 0.5 and evidence_overlap >= 0.75:
        return "answer_too_general"
    if len(unsupported_terms) >= 8 and keyword_overlap < 0.65:
        return "answer_contains_extra_info"
    if _quality_label(record) == "needs_review" and evidence_overlap >= 0.85 and keyword_overlap >= 0.45:
        return "evaluation_metric_limitation"
    if difficulty == "medium" and change_type == "worsened":
        return "answer_too_general"
    return "unknown"


def _citation_evidence(record: dict[str, Any]) -> str:
    citations = record.get("citations", []) or []
    excerpts = [str(item.get("excerpt", "") or "") for item in citations if isinstance(item, dict)]
    return "\n".join(item for item in excerpts if item)


def build_manual_check_row(
    v2_record: dict[str, Any],
    *,
    v1_record: dict[str, Any] | None = None,
    qa_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    change_type = classify_change(v1_record, v2_record)
    possible_error_type = classify_possible_error(v2_record, change_type=change_type)
    expected_answer = str(v2_record.get("expected_answer", "") or qa_record.get("expected_answer", "") if qa_record else v2_record.get("expected_answer", "") or "")
    evidence_text = str(qa_record.get("evidence_text", "") if qa_record else "")
    return {
        "question_id": v2_record.get("question_id", ""),
        "question": v2_record.get("question", ""),
        "question_type": v2_record.get("question_type", ""),
        "difficulty": v2_record.get("difficulty", ""),
        "expected_answer": expected_answer,
        "reference_answer": expected_answer,
        "retrieved_evidence": _citation_evidence(v2_record),
        "evidence_text": evidence_text,
        "generated_answer": v2_record.get("generated_answer", ""),
        "auto_quality": _quality_label(v2_record),
        "keyword_overlap_ratio": v2_record.get("keyword_overlap_ratio", 0.0),
        "evidence_overlap_ratio": v2_record.get("evidence_overlap_ratio", 0.0),
        "v1_quality": _quality_label(v1_record or {}),
        "v2_quality": _quality_label(v2_record),
        "change_type": change_type,
        "possible_error_type": possible_error_type,
        "is_condition_focus": str(v2_record.get("question_type", "") or "") == "condition",
        "is_medium_focus": str(v2_record.get("difficulty", "") or "") == "medium",
        "manual_review_result": "",
        "manual_error_type": "",
        "manual_notes": "",
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_manual_check_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_CHECK_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in MANUAL_CHECK_FIELDS})


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "") or "unknown") for row in rows).items()))


def _nested_distribution(rows: list[dict[str, Any]], group_field: str, value_field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row.get(group_field, "") or "unknown")
        value = str(row.get(value_field, "") or "unknown")
        grouped[group][value] += 1
    return {group: dict(sorted(counter.items())) for group, counter in sorted(grouped.items())}


def _summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "error_type_distribution": _distribution(rows, "possible_error_type"),
        "quality_distribution": _distribution(rows, "auto_quality"),
        "change_type_distribution": _distribution(rows, "change_type"),
        "manual_review_status": "待人工复核",
    }


def build_error_analysis(
    *,
    v1_review_jsonl: Path = DEFAULT_V1_REVIEW_JSONL,
    v2_review_jsonl: Path = V2_REVIEW_JSONL,
    qa_jsonl: Path = DEFAULT_QA_JSONL,
) -> dict[str, Any]:
    v1_records = {row["question_id"]: row for row in load_jsonl(v1_review_jsonl)}
    v2_records = load_jsonl(v2_review_jsonl)
    qa_records = {row["question_id"]: row for row in load_qa_questions(qa_jsonl)}
    all_rows = [
        build_manual_check_row(
            row,
            v1_record=v1_records.get(row.get("question_id")),
            qa_record=qa_records.get(row.get("question_id")),
        )
        for row in v2_records
    ]
    needs_review_rows = [row for row in all_rows if row["auto_quality"] == "needs_review"]
    worsened_rows = [row for row in all_rows if row["change_type"] == "worsened"]
    condition_medium_rows = [row for row in all_rows if row["is_condition_focus"] or row["is_medium_focus"]]
    condition_rows = [row for row in all_rows if row["is_condition_focus"]]
    medium_rows = [row for row in all_rows if row["is_medium_focus"]]
    report = {
        "stage": "rag_answer_error_analysis_v2",
        "total_questions": len(all_rows),
        "needs_review_count": len(needs_review_rows),
        "worsened_count": len(worsened_rows),
        "condition_case_count": len(condition_rows),
        "medium_case_count": len(medium_rows),
        "error_type_distribution": _distribution(all_rows, "possible_error_type"),
        "question_type_error_distribution": _nested_distribution(all_rows, "question_type", "possible_error_type"),
        "difficulty_error_distribution": _nested_distribution(all_rows, "difficulty", "possible_error_type"),
        "condition_error_summary": _summary_for(condition_rows),
        "medium_error_summary": _summary_for(medium_rows),
        "recommendations_for_v3": [
            "优先处理 condition 类问题，补充多条件合并与条件完整性检查，避免只抽取单个条件句。",
            "单独复核 medium 难度问题，区分答案遗漏、表达过泛和自动指标偏严格三类情况。",
            "对 v2 中 worsened 的问题建立回归样例，v3 规则调整后必须单独比较这些问题。",
            "重点检查证据覆盖率高但仍为 needs_review 的问题，判断是否属于关键词指标局限或答案组织不充分。",
            "针对答案过短或条件表达不完整的问题，增加候选句补全规则，但继续限制为检索片段内抽取。",
        ],
        "output_root": str(DEFAULT_ERROR_ANALYSIS_ROOT),
        "source_files": {
            "v1_review_jsonl": str(v1_review_jsonl),
            "v2_review_jsonl": str(v2_review_jsonl),
            "qa_jsonl": str(qa_jsonl),
            "v2_improvement_root": str(DEFAULT_IMPROVEMENT_ROOT),
        },
        "output_files": {
            "needs_review_manual_check_csv": str(NEEDS_REVIEW_MANUAL_CHECK_CSV),
            "worsened_cases_manual_check_csv": str(WORSENED_CASES_MANUAL_CHECK_CSV),
            "condition_medium_focus_csv": str(CONDITION_MEDIUM_FOCUS_CSV),
            "report_json": str(ERROR_ANALYSIS_REPORT_JSON),
        },
        "modified_inputs": {"v1_outputs": False, "v2_outputs": False, "raw": False, "cleaned": False, "chunks": False, "qa": False},
        "used_llm_api": False,
        "used_reranker": False,
        "used_web_frontend": False,
        "manual_review_status": "待人工复核",
    }
    return {
        "all_rows": all_rows,
        "needs_review_rows": needs_review_rows,
        "worsened_rows": worsened_rows,
        "condition_medium_rows": condition_medium_rows,
        "report": report,
    }


def run_error_analysis(output_root: Path = DEFAULT_ERROR_ANALYSIS_ROOT) -> dict[str, Any]:
    analysis = build_error_analysis()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "needs_review": output_root / NEEDS_REVIEW_MANUAL_CHECK_CSV.name,
        "worsened": output_root / WORSENED_CASES_MANUAL_CHECK_CSV.name,
        "focus": output_root / CONDITION_MEDIUM_FOCUS_CSV.name,
        "report": output_root / ERROR_ANALYSIS_REPORT_JSON.name,
    }
    write_manual_check_csv(paths["needs_review"], analysis["needs_review_rows"])
    write_manual_check_csv(paths["worsened"], analysis["worsened_rows"])
    write_manual_check_csv(paths["focus"], analysis["condition_medium_rows"])
    report = dict(analysis["report"])
    report["output_root"] = str(output_root)
    report["output_files"] = {
        "needs_review_manual_check_csv": str(paths["needs_review"]),
        "worsened_cases_manual_check_csv": str(paths["worsened"]),
        "condition_medium_focus_csv": str(paths["focus"]),
        "report_json": str(paths["report"]),
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
