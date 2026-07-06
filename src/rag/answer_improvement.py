"""RAG answer quality diagnosis and v2 comparison helpers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.rag.answer_quality import (
    DEFAULT_CONFIG_PATH as DEFAULT_QUALITY_CONFIG_PATH,
    DEFAULT_QA_JSONL,
    DEFAULT_RAG_ANSWERS,
    DEFAULT_OUTPUT_ROOT as DEFAULT_V1_QUALITY_ROOT,
    evaluate_answer_quality_record,
    load_answer_quality_config,
    load_qa_questions,
    load_rag_answers,
    summarize_quality,
    write_review_csv,
)
from src.rag.evaluation import CachedRAGRetriever, EvaluationCombination, evaluate_questions
from src.retrieval.dense_index import load_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
DEFAULT_IMPROVEMENT_ROOT = EXPERIMENT_ROOT / "rag_answer_quality_improvement"
DEFAULT_V1_REVIEW_JSONL = DEFAULT_V1_QUALITY_ROOT / "rag_answer_quality_review_v1.jsonl"
DEFAULT_V1_REPORT_JSON = DEFAULT_V1_QUALITY_ROOT / "rag_answer_quality_report.json"

NEEDS_REVIEW_ANALYSIS_CSV = DEFAULT_IMPROVEMENT_ROOT / "needs_review_analysis_v1.csv"
V2_ANSWERS_JSONL = DEFAULT_IMPROVEMENT_ROOT / "rag_answers_real_hybrid_weighted_fixed_length_v2.jsonl"
V2_REVIEW_CSV = DEFAULT_IMPROVEMENT_ROOT / "rag_answer_quality_review_v2.csv"
V2_REVIEW_JSONL = DEFAULT_IMPROVEMENT_ROOT / "rag_answer_quality_review_v2.jsonl"
IMPROVEMENT_REPORT_JSON = DEFAULT_IMPROVEMENT_ROOT / "rag_answer_quality_improvement_report.json"

ALLOWED_DIAGNOSIS_LABELS = {
    "missing_key_condition",
    "missing_procedure_step",
    "missing_responsible_body",
    "answer_too_short",
    "evidence_not_fully_used",
    "multi_clause_answer_needed",
    "keyword_metric_too_strict",
    "other",
}

TYPE_DIAGNOSIS = {
    "condition": ("missing_key_condition", "优先保留包含条件、情形、应当、不得、申请等约束词的句子。"),
    "procedure": ("missing_procedure_step", "优先保留包含申请、提交、审核、审批、批准、办理、公示等流程词的句子。"),
    "responsible_body": ("missing_responsible_body", "优先保留学院、职能部门、辅导员、班主任等责任主体。"),
    "comparison": ("multi_clause_answer_needed", "允许组织多分句答案，避免只抽取其中一侧内容。"),
}

LABEL_SCORE = {"insufficient": 0, "needs_review": 1, "good": 2}


def load_quality_records(path: Path = DEFAULT_V1_REVIEW_JSONL) -> list[dict[str, Any]]:
    return load_jsonl(path)


def diagnose_needs_review(record: dict[str, Any]) -> tuple[str, str, str]:
    question_type = str(record.get("question_type", "") or "")
    keyword_overlap = float(record.get("keyword_overlap_ratio", 0.0) or 0.0)
    evidence_overlap = float(record.get("evidence_overlap_ratio", 0.0) or 0.0)
    answer_chars = int(record.get("answer_char_count", 0) or 0)
    if answer_chars < 180:
        return "answer_too_short", "答案长度偏短，可能未覆盖标准答案中的多个关键词。", "在长度上限内补充更多高分候选句。"
    if keyword_overlap >= 0.45 and record.get("source_policy_hit") and record.get("related_chunk_hit"):
        return "keyword_metric_too_strict", "引用和相关 chunk 命中，但关键词重合略低，可能属于自动指标偏严格。", "保留人工复核入口，不为单一指标硬调规则。"
    if question_type in TYPE_DIAGNOSIS:
        label, fix = TYPE_DIAGNOSIS[question_type]
        return label, f"{question_type} 类问题的关键句可能没有被优先抽取。", fix
    if evidence_overlap < 0.55:
        return "evidence_not_fully_used", "生成答案没有充分覆盖证据片段中的关键词。", "提高证据相关句和同制度片段的选择权重。"
    return "other", "自动指标显示需要复核，但无法归入明确规则类别。", "保留人工检查，并在后续轮次按样例再细化规则。"


def analyze_needs_review(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("auto_quality_label") != "needs_review":
            continue
        diagnosis_label, diagnosis_reason, suggested_fix = diagnose_needs_review(record)
        rows.append(
            {
                "question_id": record.get("question_id", ""),
                "question": record.get("question", ""),
                "question_type": record.get("question_type", ""),
                "difficulty": record.get("difficulty", ""),
                "expected_answer": record.get("expected_answer", ""),
                "generated_answer": record.get("generated_answer", ""),
                "keyword_overlap_ratio": record.get("keyword_overlap_ratio", 0.0),
                "evidence_overlap_ratio": record.get("evidence_overlap_ratio", 0.0),
                "possible_missing_keywords": record.get("possible_missing_keywords", []),
                "possible_unsupported_terms": record.get("possible_unsupported_terms", []),
                "cited_policy_titles": record.get("cited_policy_titles", []),
                "diagnosis_label": diagnosis_label,
                "diagnosis_reason": diagnosis_reason,
                "suggested_fix": suggested_fix,
            }
        )
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_analysis_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id",
        "question",
        "question_type",
        "difficulty",
        "expected_answer",
        "generated_answer",
        "keyword_overlap_ratio",
        "evidence_overlap_ratio",
        "possible_missing_keywords",
        "possible_unsupported_terms",
        "cited_policy_titles",
        "diagnosis_label",
        "diagnosis_reason",
        "suggested_fix",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def run_needs_review_analysis(
    *,
    v1_review_jsonl: Path = DEFAULT_V1_REVIEW_JSONL,
    output_csv: Path = NEEDS_REVIEW_ANALYSIS_CSV,
) -> list[dict[str, Any]]:
    rows = analyze_needs_review(load_quality_records(v1_review_jsonl))
    write_analysis_csv(output_csv, rows)
    return rows


def generate_v2_answers(
    *,
    qa_jsonl: Path = DEFAULT_QA_JSONL,
    output_path: Path = V2_ANSWERS_JSONL,
    retriever: Any | None = None,
) -> list[dict[str, Any]]:
    questions = load_qa_questions(qa_jsonl)
    combination = EvaluationCombination(method="real_hybrid_weighted", strategy="fixed_length", top_k=5)
    records, _summary = evaluate_questions(questions, combination, retriever=retriever or CachedRAGRetriever())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    return records


def evaluate_v2_quality(
    *,
    qa_jsonl: Path = DEFAULT_QA_JSONL,
    answers_jsonl: Path = V2_ANSWERS_JSONL,
    review_jsonl: Path = V2_REVIEW_JSONL,
    review_csv: Path = V2_REVIEW_CSV,
    config_path: Path = DEFAULT_QUALITY_CONFIG_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = load_answer_quality_config(config_path)
    questions = {row["question_id"]: row for row in load_qa_questions(qa_jsonl)}
    answers = load_rag_answers(answers_jsonl)
    records = [
        evaluate_answer_quality_record(questions[answer["question_id"]], answer, config)
        for answer in answers
        if answer.get("question_id") in questions
    ]
    review_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(review_jsonl, records)
    write_review_csv(review_csv, records)
    return records, summarize_quality(records)


def _avg(records: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0) or 0.0) for row in records) / len(records) if records else 0.0


def _distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    values: dict[str, int] = {}
    for record in records:
        label = str(record.get("auto_quality_label", "") or "unknown")
        values[label] = values.get(label, 0) + 1
    return dict(sorted(values.items()))


def _record_change(v1: dict[str, Any], v2: dict[str, Any]) -> str:
    v1_label = LABEL_SCORE.get(str(v1.get("auto_quality_label", "")), 0)
    v2_label = LABEL_SCORE.get(str(v2.get("auto_quality_label", "")), 0)
    if v2_label > v1_label:
        return "improved"
    if v2_label < v1_label:
        return "worsened"
    delta = float(v2.get("keyword_overlap_ratio", 0.0) or 0.0) - float(v1.get("keyword_overlap_ratio", 0.0) or 0.0)
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "worsened"
    return "unchanged"


def _group_comparison(v1_records: list[dict[str, Any]], v2_records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    v1_by_id = {row["question_id"]: row for row in v1_records}
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for v2 in v2_records:
        v1 = v1_by_id.get(v2.get("question_id"))
        if not v1:
            continue
        grouped[str(v2.get(field, "") or "unknown")].append((v1, v2))
    result: dict[str, dict[str, Any]] = {}
    for group, pairs in sorted(grouped.items()):
        changes = [_record_change(v1, v2) for v1, v2 in pairs]
        v1_values = [v1 for v1, _v2 in pairs]
        v2_values = [v2 for _v1, v2 in pairs]
        result[group] = {
            "question_count": len(pairs),
            "v1_average_keyword_overlap_ratio": _avg(v1_values, "keyword_overlap_ratio"),
            "v2_average_keyword_overlap_ratio": _avg(v2_values, "keyword_overlap_ratio"),
            "v1_average_evidence_overlap_ratio": _avg(v1_values, "evidence_overlap_ratio"),
            "v2_average_evidence_overlap_ratio": _avg(v2_values, "evidence_overlap_ratio"),
            "v1_auto_quality_distribution": _distribution(v1_values),
            "v2_auto_quality_distribution": _distribution(v2_values),
            "improved_question_count": changes.count("improved"),
            "worsened_question_count": changes.count("worsened"),
            "unchanged_question_count": changes.count("unchanged"),
        }
    return result


def compare_quality(v1_records: list[dict[str, Any]], v2_records: list[dict[str, Any]]) -> dict[str, Any]:
    v1_by_id = {row["question_id"]: row for row in v1_records}
    paired = [(v1_by_id[row["question_id"]], row) for row in v2_records if row.get("question_id") in v1_by_id]
    changes = [_record_change(v1, v2) for v1, v2 in paired]
    return {
        "v1_average_keyword_overlap_ratio": _avg(v1_records, "keyword_overlap_ratio"),
        "v2_average_keyword_overlap_ratio": _avg(v2_records, "keyword_overlap_ratio"),
        "v1_average_evidence_overlap_ratio": _avg(v1_records, "evidence_overlap_ratio"),
        "v2_average_evidence_overlap_ratio": _avg(v2_records, "evidence_overlap_ratio"),
        "v1_auto_quality_distribution": _distribution(v1_records),
        "v2_auto_quality_distribution": _distribution(v2_records),
        "v1_needs_review_count": _distribution(v1_records).get("needs_review", 0),
        "v2_needs_review_count": _distribution(v2_records).get("needs_review", 0),
        "improved_question_count": changes.count("improved"),
        "worsened_question_count": changes.count("worsened"),
        "unchanged_question_count": changes.count("unchanged"),
        "question_type": _group_comparison(v1_records, v2_records, "question_type"),
        "difficulty": _group_comparison(v1_records, v2_records, "difficulty"),
    }


def run_improvement_evaluation(
    *,
    output_root: Path = DEFAULT_IMPROVEMENT_ROOT,
    retriever: Any | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    needs_review_csv = output_root / NEEDS_REVIEW_ANALYSIS_CSV.name
    v2_answers = output_root / V2_ANSWERS_JSONL.name
    v2_review_jsonl = output_root / V2_REVIEW_JSONL.name
    v2_review_csv = output_root / V2_REVIEW_CSV.name
    report_json = output_root / IMPROVEMENT_REPORT_JSON.name

    v1_records = load_quality_records(DEFAULT_V1_REVIEW_JSONL)
    analysis_rows = analyze_needs_review(v1_records)
    write_analysis_csv(needs_review_csv, analysis_rows)
    generate_v2_answers(output_path=v2_answers, retriever=retriever)
    v2_records, v2_summary = evaluate_v2_quality(
        answers_jsonl=v2_answers,
        review_jsonl=v2_review_jsonl,
        review_csv=v2_review_csv,
    )
    comparison = compare_quality(v1_records, v2_records)
    report = {
        "stage": "rag_answer_quality_improvement_v2",
        "question_count": len(v2_records),
        "v1_quality_review_jsonl": str(DEFAULT_V1_REVIEW_JSONL),
        "v1_rag_answers_path": str(DEFAULT_RAG_ANSWERS),
        "output_root": str(output_root),
        "v1_summary": summarize_quality(v1_records),
        "v2_summary": v2_summary,
        "comparison": comparison,
        "needs_review_analysis_count": len(analysis_rows),
        "output_files": {
            "needs_review_analysis_csv": str(needs_review_csv),
            "v2_answers_jsonl": str(v2_answers),
            "v2_review_jsonl": str(v2_review_jsonl),
            "v2_review_csv": str(v2_review_csv),
            "report_json": str(report_json),
        },
        "modified_inputs": {"raw": False, "cleaned": False, "chunks": False, "qa": False},
        "overwrote_historical_outputs": False,
        "committable_external_outputs": False,
        "used_expected_answer_for_generation": False,
        "used_evidence_text_for_generation": False,
        "used_llm_api": False,
        "used_reranker": False,
        "used_web_frontend": False,
        "notes": "v2 generation uses only question text and retrieved chunks; QA labels/evidence are used only by evaluation.",
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
