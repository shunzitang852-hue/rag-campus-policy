from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.answer_error_analysis import classify_change, classify_possible_error

EXPERIMENT_ROOT = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook")
OUTPUT_ROOT = EXPERIMENT_ROOT / "manual_review_package"

V1_REVIEW_JSONL = EXPERIMENT_ROOT / "rag_answer_quality" / "rag_answer_quality_review_v1.jsonl"
V2_ANSWERS_JSONL = (
    EXPERIMENT_ROOT
    / "rag_answer_quality_improvement"
    / "rag_answers_real_hybrid_weighted_fixed_length_v2.jsonl"
)
V2_REVIEW_JSONL = EXPERIMENT_ROOT / "rag_answer_quality_improvement" / "rag_answer_quality_review_v2.jsonl"
V2_NEEDS_REVIEW_CSV = (
    EXPERIMENT_ROOT
    / "rag_answer_quality_error_analysis"
    / "rag_v2_needs_review_manual_check.csv"
)
V2_WORSENED_CSV = (
    EXPERIMENT_ROOT
    / "rag_answer_quality_error_analysis"
    / "rag_v2_worsened_cases_manual_check.csv"
)
V2_CONDITION_MEDIUM_CSV = (
    EXPERIMENT_ROOT
    / "rag_answer_quality_error_analysis"
    / "rag_v2_condition_medium_focus_cases.csv"
)
V2_ERROR_REPORT_JSON = (
    EXPERIMENT_ROOT / "rag_answer_quality_error_analysis" / "rag_answer_error_analysis_v2_report.json"
)
V3_ANSWERS_JSONL = (
    EXPERIMENT_ROOT
    / "rag_answer_quality_v3_optimization"
    / "rag_answers_real_hybrid_weighted_fixed_length_v3.jsonl"
)
V3_REVIEW_JSONL = (
    EXPERIMENT_ROOT / "rag_answer_quality_v3_optimization" / "rag_answer_quality_review_v3.jsonl"
)
V2_V3_COMPARISON_JSONL = (
    EXPERIMENT_ROOT / "rag_answer_quality_v3_optimization" / "rag_answer_quality_comparison_v2_v3.jsonl"
)
V2_V3_COMPARISON_REPORT_JSON = (
    EXPERIMENT_ROOT
    / "rag_answer_quality_v3_optimization"
    / "rag_answer_quality_comparison_v2_v3_report.json"
)

NEEDS_REVIEW_OUTPUT = OUTPUT_ROOT / "manual_review_needs_review_cases.csv"
WORSENED_OUTPUT = OUTPUT_ROOT / "manual_review_worsened_cases.csv"
CONDITION_OUTPUT = OUTPUT_ROOT / "manual_review_condition_cases.csv"
MEDIUM_OUTPUT = OUTPUT_ROOT / "manual_review_medium_cases.csv"
UNKNOWN_OUTPUT = OUTPUT_ROOT / "manual_review_unknown_error_cases.csv"
ALL_CASES_OUTPUT = OUTPUT_ROOT / "manual_review_sample_all_cases.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "manual_review_package_summary.json"


NEEDS_REVIEW_FIELDS = [
    "review_id",
    "question_id",
    "question",
    "question_type",
    "difficulty",
    "v2_generated_answer",
    "v2_auto_quality",
    "v2_keyword_overlap_ratio",
    "v2_evidence_overlap_ratio",
    "v2_possible_error_type",
    "retrieved_evidence",
    "expected_answer",
    "reference_answer",
    "manual_answer_correctness",
    "manual_evidence_sufficiency",
    "manual_error_type",
    "manual_should_fix",
    "manual_notes",
]

WORSENED_FIELDS = [
    "review_id",
    "change_source",
    "question_id",
    "question",
    "question_type",
    "difficulty",
    "before_answer",
    "after_answer",
    "before_quality",
    "after_quality",
    "before_keyword_overlap_ratio",
    "after_keyword_overlap_ratio",
    "before_evidence_overlap_ratio",
    "after_evidence_overlap_ratio",
    "change_reason_auto",
    "retrieved_evidence",
    "manual_is_real_regression",
    "manual_reason",
    "manual_notes",
]

CONDITION_FIELDS = [
    "review_id",
    "question_id",
    "question",
    "difficulty",
    "generated_answer",
    "auto_quality",
    "possible_error_type",
    "retrieved_evidence",
    "condition_points_found",
    "manual_condition_complete",
    "missing_condition_points",
    "manual_notes",
]

MEDIUM_FIELDS = [
    "review_id",
    "question_id",
    "question",
    "question_type",
    "generated_answer",
    "auto_quality",
    "possible_error_type",
    "keyword_overlap_ratio",
    "evidence_overlap_ratio",
    "retrieved_evidence",
    "manual_answer_too_general",
    "manual_should_expand",
    "manual_notes",
]

UNKNOWN_FIELDS = [
    "review_id",
    "question_id",
    "question",
    "question_type",
    "difficulty",
    "generated_answer",
    "auto_quality",
    "keyword_overlap_ratio",
    "evidence_overlap_ratio",
    "retrieved_evidence",
    "manual_error_type",
    "manual_auto_metric_misjudged",
    "manual_notes",
]

ALL_CASE_FIELDS = [
    "review_id",
    "question_id",
    "question",
    "question_type",
    "difficulty",
    "v2_generated_answer",
    "v2_auto_quality",
    "v2_keyword_overlap_ratio",
    "v2_evidence_overlap_ratio",
    "v3_generated_answer",
    "v3_auto_quality",
    "v3_keyword_overlap_ratio",
    "v3_evidence_overlap_ratio",
    "recommended_review_priority",
    "priority_reason",
    "manual_final_correctness",
    "manual_final_notes",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_map(rows: list[dict[str, Any]], key: str = "question_id") -> dict[str, dict[str, Any]]:
    return {str(row.get(key, "")): row for row in rows}


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def citations_to_text(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    citations = record.get("citations", []) or []
    excerpts = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        title = item.get("document_title", "")
        chunk_id = item.get("chunk_id", "")
        excerpt = item.get("excerpt", "")
        if excerpt:
            excerpts.append(f"[{title} | {chunk_id}] {excerpt}")
    return "\n".join(excerpts)


def record_label(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return str(record.get("auto_quality_label", record.get("auto_quality", "")) or "")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def possible_error_for(
    v2_review: dict[str, Any],
    *,
    v1_review: dict[str, Any] | None,
    manual_error_rows: dict[str, dict[str, str]],
) -> str:
    question_id = str(v2_review.get("question_id", ""))
    if question_id in manual_error_rows:
        return str(manual_error_rows[question_id].get("possible_error_type", "") or "unknown")
    change_type = classify_change(v1_review, v2_review)
    return classify_possible_error(v2_review, change_type=change_type)


def condition_points_found(record: dict[str, Any]) -> str:
    answer = str(record.get("generated_answer", "") or "")
    markers = ["条件", "申请", "符合", "不得", "应当", "可以", "需要", "资格", "审核"]
    found = [marker for marker in markers if marker in answer]
    return "；".join(found)


def recommend_priority(
    v2_review: dict[str, Any],
    *,
    possible_error_type: str,
    v1_to_v2_change: str,
    v2_v3_comparison: dict[str, Any] | None,
) -> tuple[str, str]:
    reasons: list[str] = []
    v2_quality = record_label(v2_review)
    question_type = str(v2_review.get("question_type", "") or "")
    difficulty = str(v2_review.get("difficulty", "") or "")
    v2_v3_change = str((v2_v3_comparison or {}).get("change_type", "") or "")
    v3_quality = str((v2_v3_comparison or {}).get("v3_auto_quality", "") or "")

    if v2_quality == "needs_review":
        reasons.append("v2 auto_quality=needs_review")
    if v1_to_v2_change == "worsened":
        reasons.append("v1_to_v2 worsened")
    if v2_v3_change == "worsened":
        reasons.append("v2_to_v3 worsened")
    if possible_error_type == "condition_not_fully_covered":
        reasons.append("possible_error_type=condition_not_fully_covered")
    if question_type == "condition" and v2_quality == "needs_review":
        reasons.append("condition 且 v2 needs_review")
    if reasons:
        return "high", "；".join(reasons)

    if difficulty == "medium":
        reasons.append("difficulty=medium")
    if possible_error_type in {"answer_too_general", "evidence_partially_used"}:
        reasons.append(f"possible_error_type={possible_error_type}")
    if question_type in {"condition"} or difficulty == "medium":
        if v2_v3_change == "improved" and v3_quality == "needs_review":
            reasons.append("v3 目标分组改善但质量仍 needs_review")
    if reasons:
        return "medium", "；".join(reasons)
    return "low", "未触发 high/medium 规则"


def next_review_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:03d}"


def build_manual_review_package() -> dict[str, Any]:
    v1_reviews = as_map(load_jsonl(V1_REVIEW_JSONL))
    v2_answers = as_map(load_jsonl(V2_ANSWERS_JSONL))
    v2_reviews = load_jsonl(V2_REVIEW_JSONL)
    v2_review_map = as_map(v2_reviews)
    v3_answers = as_map(load_jsonl(V3_ANSWERS_JSONL))
    v3_reviews = as_map(load_jsonl(V3_REVIEW_JSONL))
    v2_v3_comparisons = as_map(load_jsonl(V2_V3_COMPARISON_JSONL))

    manual_rows = load_csv(V2_NEEDS_REVIEW_CSV) + load_csv(V2_WORSENED_CSV) + load_csv(V2_CONDITION_MEDIUM_CSV)
    manual_error_rows = as_map(manual_rows)

    enriched: list[dict[str, Any]] = []
    for v2_review in v2_reviews:
        qid = str(v2_review.get("question_id", ""))
        v1_review = v1_reviews.get(qid)
        possible_error_type = possible_error_for(
            v2_review,
            v1_review=v1_review,
            manual_error_rows=manual_error_rows,
        )
        change_type = classify_change(v1_review, v2_review)
        v2_v3 = v2_v3_comparisons.get(qid)
        priority, reason = recommend_priority(
            v2_review,
            possible_error_type=possible_error_type,
            v1_to_v2_change=change_type,
            v2_v3_comparison=v2_v3,
        )
        retrieved_evidence = citations_to_text(v2_answers.get(qid)) or str(
            manual_error_rows.get(qid, {}).get("retrieved_evidence", "")
            or manual_error_rows.get(qid, {}).get("evidence_text", "")
            or ""
        )
        enriched.append(
            {
                "question_id": qid,
                "v1_review": v1_review or {},
                "v2_review": v2_review,
                "v2_answer": v2_answers.get(qid, {}),
                "v3_review": v3_reviews.get(qid, {}),
                "v3_answer": v3_answers.get(qid, {}),
                "v2_v3_comparison": v2_v3 or {},
                "possible_error_type": possible_error_type,
                "v1_to_v2_change": change_type,
                "retrieved_evidence": retrieved_evidence,
                "recommended_review_priority": priority,
                "priority_reason": reason,
            }
        )

    needs_review_rows = []
    for index, item in enumerate(
        [item for item in enriched if record_label(item["v2_review"]) == "needs_review"], start=1
    ):
        v2 = item["v2_review"]
        needs_review_rows.append(
            {
                "review_id": next_review_id("NR", index),
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "question_type": v2.get("question_type", ""),
                "difficulty": v2.get("difficulty", ""),
                "v2_generated_answer": v2.get("generated_answer", ""),
                "v2_auto_quality": record_label(v2),
                "v2_keyword_overlap_ratio": v2.get("keyword_overlap_ratio", ""),
                "v2_evidence_overlap_ratio": v2.get("evidence_overlap_ratio", ""),
                "v2_possible_error_type": item["possible_error_type"],
                "retrieved_evidence": item["retrieved_evidence"],
                "expected_answer": v2.get("expected_answer", ""),
                "reference_answer": v2.get("expected_answer", ""),
                "manual_answer_correctness": "",
                "manual_evidence_sufficiency": "",
                "manual_error_type": "",
                "manual_should_fix": "",
                "manual_notes": "",
            }
        )

    worsened_rows = []
    v1_to_v2_worsened = [item for item in enriched if item["v1_to_v2_change"] == "worsened"]
    for item in v1_to_v2_worsened:
        v1 = item["v1_review"]
        v2 = item["v2_review"]
        worsened_rows.append(
            {
                "change_source": "v1_to_v2",
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "question_type": v2.get("question_type", ""),
                "difficulty": v2.get("difficulty", ""),
                "before_answer": v1.get("generated_answer", ""),
                "after_answer": v2.get("generated_answer", ""),
                "before_quality": record_label(v1),
                "after_quality": record_label(v2),
                "before_keyword_overlap_ratio": v1.get("keyword_overlap_ratio", ""),
                "after_keyword_overlap_ratio": v2.get("keyword_overlap_ratio", ""),
                "before_evidence_overlap_ratio": v1.get("evidence_overlap_ratio", ""),
                "after_evidence_overlap_ratio": v2.get("evidence_overlap_ratio", ""),
                "change_reason_auto": item["possible_error_type"],
                "retrieved_evidence": item["retrieved_evidence"],
                "manual_is_real_regression": "",
                "manual_reason": "",
                "manual_notes": "",
            }
        )
    v2_to_v3_worsened = [
        item for item in enriched if str(item["v2_v3_comparison"].get("change_type", "")) == "worsened"
    ]
    for item in v2_to_v3_worsened:
        v2 = item["v2_review"]
        v3 = item["v3_review"]
        cmp_row = item["v2_v3_comparison"]
        worsened_rows.append(
            {
                "change_source": "v2_to_v3",
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "question_type": v2.get("question_type", ""),
                "difficulty": v2.get("difficulty", ""),
                "before_answer": v2.get("generated_answer", ""),
                "after_answer": v3.get("generated_answer", ""),
                "before_quality": cmp_row.get("v2_auto_quality", record_label(v2)),
                "after_quality": cmp_row.get("v3_auto_quality", record_label(v3)),
                "before_keyword_overlap_ratio": cmp_row.get("v2_keyword_overlap_ratio", ""),
                "after_keyword_overlap_ratio": cmp_row.get("v3_keyword_overlap_ratio", ""),
                "before_evidence_overlap_ratio": cmp_row.get("v2_evidence_overlap_ratio", ""),
                "after_evidence_overlap_ratio": cmp_row.get("v3_evidence_overlap_ratio", ""),
                "change_reason_auto": "v2_to_v3_worsened",
                "retrieved_evidence": item["retrieved_evidence"],
                "manual_is_real_regression": "",
                "manual_reason": "",
                "manual_notes": "",
            }
        )
    for index, row in enumerate(worsened_rows, start=1):
        row["review_id"] = next_review_id("WR", index)

    condition_items = [
        item
        for item in enriched
        if str(item["v2_review"].get("question_type", "")) == "condition"
        or item["possible_error_type"] == "condition_not_fully_covered"
    ]
    condition_rows = []
    for index, item in enumerate(condition_items, start=1):
        v2 = item["v2_review"]
        condition_rows.append(
            {
                "review_id": next_review_id("CD", index),
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "difficulty": v2.get("difficulty", ""),
                "generated_answer": v2.get("generated_answer", ""),
                "auto_quality": record_label(v2),
                "possible_error_type": item["possible_error_type"],
                "retrieved_evidence": item["retrieved_evidence"],
                "condition_points_found": condition_points_found(v2),
                "manual_condition_complete": "",
                "missing_condition_points": "",
                "manual_notes": "",
            }
        )

    medium_rows = []
    for index, item in enumerate(
        [item for item in enriched if str(item["v2_review"].get("difficulty", "")) == "medium"], start=1
    ):
        v2 = item["v2_review"]
        medium_rows.append(
            {
                "review_id": next_review_id("MD", index),
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "question_type": v2.get("question_type", ""),
                "generated_answer": v2.get("generated_answer", ""),
                "auto_quality": record_label(v2),
                "possible_error_type": item["possible_error_type"],
                "keyword_overlap_ratio": v2.get("keyword_overlap_ratio", ""),
                "evidence_overlap_ratio": v2.get("evidence_overlap_ratio", ""),
                "retrieved_evidence": item["retrieved_evidence"],
                "manual_answer_too_general": "",
                "manual_should_expand": "",
                "manual_notes": "",
            }
        )

    unknown_rows = []
    for index, item in enumerate(
        [item for item in enriched if item["possible_error_type"] == "unknown"], start=1
    ):
        v2 = item["v2_review"]
        unknown_rows.append(
            {
                "review_id": next_review_id("UN", index),
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "question_type": v2.get("question_type", ""),
                "difficulty": v2.get("difficulty", ""),
                "generated_answer": v2.get("generated_answer", ""),
                "auto_quality": record_label(v2),
                "keyword_overlap_ratio": v2.get("keyword_overlap_ratio", ""),
                "evidence_overlap_ratio": v2.get("evidence_overlap_ratio", ""),
                "retrieved_evidence": item["retrieved_evidence"],
                "manual_error_type": "",
                "manual_auto_metric_misjudged": "",
                "manual_notes": "",
            }
        )

    all_case_rows = []
    for index, item in enumerate(enriched, start=1):
        v2 = item["v2_review"]
        v3 = item["v3_review"]
        all_case_rows.append(
            {
                "review_id": next_review_id("ALL", index),
                "question_id": item["question_id"],
                "question": v2.get("question", ""),
                "question_type": v2.get("question_type", ""),
                "difficulty": v2.get("difficulty", ""),
                "v2_generated_answer": v2.get("generated_answer", ""),
                "v2_auto_quality": record_label(v2),
                "v2_keyword_overlap_ratio": v2.get("keyword_overlap_ratio", ""),
                "v2_evidence_overlap_ratio": v2.get("evidence_overlap_ratio", ""),
                "v3_generated_answer": v3.get("generated_answer", ""),
                "v3_auto_quality": record_label(v3),
                "v3_keyword_overlap_ratio": v3.get("keyword_overlap_ratio", ""),
                "v3_evidence_overlap_ratio": v3.get("evidence_overlap_ratio", ""),
                "recommended_review_priority": item["recommended_review_priority"],
                "priority_reason": item["priority_reason"],
                "manual_final_correctness": "",
                "manual_final_notes": "",
            }
        )

    priority_counts = Counter(row["recommended_review_priority"] for row in all_case_rows)
    summary = {
        "total_cases": len(enriched),
        "needs_review_cases": len(needs_review_rows),
        "worsened_cases": len(worsened_rows),
        "condition_cases": len(condition_rows),
        "medium_cases": len(medium_rows),
        "unknown_error_cases": len(unknown_rows),
        "high_priority_cases": priority_counts.get("high", 0),
        "medium_priority_cases": priority_counts.get("medium", 0),
        "low_priority_cases": priority_counts.get("low", 0),
        "files_generated": {
            "manual_review_needs_review_cases_csv": str(NEEDS_REVIEW_OUTPUT),
            "manual_review_worsened_cases_csv": str(WORSENED_OUTPUT),
            "manual_review_condition_cases_csv": str(CONDITION_OUTPUT),
            "manual_review_medium_cases_csv": str(MEDIUM_OUTPUT),
            "manual_review_unknown_error_cases_csv": str(UNKNOWN_OUTPUT),
            "manual_review_sample_all_cases_csv": str(ALL_CASES_OUTPUT),
            "manual_review_package_summary_json": str(SUMMARY_OUTPUT),
        },
        "review_priority_rule": {
            "high": [
                "v2 auto_quality=needs_review",
                "v1->v2 或 v2->v3 worsened",
                "possible_error_type=condition_not_fully_covered",
                "question_type=condition 且 v2 auto_quality=needs_review",
            ],
            "medium": [
                "difficulty=medium",
                "possible_error_type=answer_too_general",
                "possible_error_type=evidence_partially_used",
                "v3 目标分组改善但整体质量仍 needs_review",
            ],
            "low": ["其他样本"],
        },
        "notes": [
            "本复核包只整理已有 v1/v2/v3、v2 误差分析和实验结果汇总，不做新实验。",
            "人工填写字段均留空，当前自动评价不能替代人工判断。",
            "未调用大模型 API，未做 Reranker，未做 Web 前端，未覆盖历史实验结果。",
        ],
        "source_files": {
            "v1_review_jsonl": str(V1_REVIEW_JSONL),
            "v2_answers_jsonl": str(V2_ANSWERS_JSONL),
            "v2_review_jsonl": str(V2_REVIEW_JSONL),
            "v2_needs_review_csv": str(V2_NEEDS_REVIEW_CSV),
            "v2_worsened_csv": str(V2_WORSENED_CSV),
            "v2_condition_medium_csv": str(V2_CONDITION_MEDIUM_CSV),
            "v2_error_report_json": str(V2_ERROR_REPORT_JSON),
            "v3_answers_jsonl": str(V3_ANSWERS_JSONL),
            "v3_review_jsonl": str(V3_REVIEW_JSONL),
            "v2_v3_comparison_jsonl": str(V2_V3_COMPARISON_JSONL),
            "v2_v3_comparison_report_json": str(V2_V3_COMPARISON_REPORT_JSON),
        },
    }
    return {
        "needs_review_rows": needs_review_rows,
        "worsened_rows": worsened_rows,
        "condition_rows": condition_rows,
        "medium_rows": medium_rows,
        "unknown_rows": unknown_rows,
        "all_case_rows": all_case_rows,
        "summary": summary,
    }


def write_manual_review_package(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    package = build_manual_review_package()
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "needs_review": output_root / NEEDS_REVIEW_OUTPUT.name,
        "worsened": output_root / WORSENED_OUTPUT.name,
        "condition": output_root / CONDITION_OUTPUT.name,
        "medium": output_root / MEDIUM_OUTPUT.name,
        "unknown": output_root / UNKNOWN_OUTPUT.name,
        "all_cases": output_root / ALL_CASES_OUTPUT.name,
        "summary": output_root / SUMMARY_OUTPUT.name,
    }
    write_csv(output_paths["needs_review"], package["needs_review_rows"], NEEDS_REVIEW_FIELDS)
    write_csv(output_paths["worsened"], package["worsened_rows"], WORSENED_FIELDS)
    write_csv(output_paths["condition"], package["condition_rows"], CONDITION_FIELDS)
    write_csv(output_paths["medium"], package["medium_rows"], MEDIUM_FIELDS)
    write_csv(output_paths["unknown"], package["unknown_rows"], UNKNOWN_FIELDS)
    write_csv(output_paths["all_cases"], package["all_case_rows"], ALL_CASE_FIELDS)

    summary = dict(package["summary"])
    summary["files_generated"] = {
        "manual_review_needs_review_cases_csv": str(output_paths["needs_review"]),
        "manual_review_worsened_cases_csv": str(output_paths["worsened"]),
        "manual_review_condition_cases_csv": str(output_paths["condition"]),
        "manual_review_medium_cases_csv": str(output_paths["medium"]),
        "manual_review_unknown_error_cases_csv": str(output_paths["unknown"]),
        "manual_review_sample_all_cases_csv": str(output_paths["all_cases"]),
        "manual_review_package_summary_json": str(output_paths["summary"]),
    }
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = write_manual_review_package()
    for path in summary["files_generated"].values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
