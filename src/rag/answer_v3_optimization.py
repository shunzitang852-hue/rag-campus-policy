"""V3 targeted rule-based RAG answer optimization.

This module is intentionally separate from the v1/v2 answer generator so that
historical experiments remain reproducible.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.rag.answer_improvement import V2_REVIEW_JSONL
from src.rag.answer_quality import (
    DEFAULT_CONFIG_PATH as DEFAULT_QUALITY_CONFIG_PATH,
    DEFAULT_QA_JSONL,
    evaluate_answer_quality_record,
    load_answer_quality_config,
    load_qa_questions,
    summarize_quality,
    write_review_csv,
)
from src.rag.citation import build_citations
from src.rag.context_builder import ContextItem, build_context, compact_text
from src.rag.evaluation import CachedRAGRetriever, EvaluationCombination, source_policy_hit, related_chunk_hit
from src.retrieval.dense_index import load_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
DEFAULT_V3_OUTPUT_ROOT = EXPERIMENT_ROOT / "rag_answer_quality_v3_optimization"

V3_ANSWERS_JSONL = DEFAULT_V3_OUTPUT_ROOT / "rag_answers_real_hybrid_weighted_fixed_length_v3.jsonl"
V3_REVIEW_CSV = DEFAULT_V3_OUTPUT_ROOT / "rag_answer_quality_review_v3.csv"
V3_REVIEW_JSONL = DEFAULT_V3_OUTPUT_ROOT / "rag_answer_quality_review_v3.jsonl"
V3_REPORT_JSON = DEFAULT_V3_OUTPUT_ROOT / "rag_answer_quality_v3_report.json"
V2_V3_COMPARISON_CSV = DEFAULT_V3_OUTPUT_ROOT / "rag_answer_quality_comparison_v2_v3.csv"
V2_V3_COMPARISON_JSONL = DEFAULT_V3_OUTPUT_ROOT / "rag_answer_quality_comparison_v2_v3.jsonl"
V2_V3_COMPARISON_REPORT_JSON = DEFAULT_V3_OUTPUT_ROOT / "rag_answer_quality_comparison_v2_v3_report.json"

LABEL_SCORE = {"insufficient": 0, "needs_review": 1, "good": 2}

TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "condition": ("什么情况下", "哪些情况", "条件", "是否可以", "能否", "需要满足", "认定", "处分", "申请条件", "资格"),
    "procedure": ("流程", "步骤", "怎么办", "如何办理", "申请", "处理程序", "程序", "办理"),
    "responsible_body": ("谁负责", "哪个部门", "责任主体", "由谁", "归口", "负责"),
    "qualification": ("资格", "能不能", "是否具备", "是否符合", "认定条件", "评定"),
}

TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "condition": ("应当", "需要", "可以", "不得", "符合", "申请", "条件", "情形", "资格", "处分", "认定"),
    "procedure": ("申请", "提交", "审核", "审批", "批准", "办理", "公示", "程序", "流程", "处理"),
    "responsible_body": ("学院", "部门", "教务处", "学生处", "辅导员", "班主任", "负责", "归口", "委员会"),
    "qualification": ("条件", "资格", "评定", "成绩", "违纪", "处分", "排名", "要求", "符合", "认定"),
    "general": (),
}

RESULT_TERMS = ("处理", "结果", "资格", "认定", "批准", "办理", "解除", "取消", "给予", "不得", "可以")
SOURCE_TERMS = ("第", "章", "条", "规定", "办法", "细则", "制度")


def question_structure_type(question: str, question_type: str = "", difficulty: str = "") -> str:
    normalized_type = str(question_type or "").strip()
    if normalized_type in {"condition", "procedure", "responsible_body", "qualification"}:
        return f"{normalized_type}_template"
    for candidate, hints in TYPE_HINTS.items():
        if any(hint in question for hint in hints):
            return f"{candidate}_template"
    if difficulty == "medium":
        return "general_template"
    return "general_template"


def _query_terms(question: str) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", question or "")
    terms = set(chars)
    for size in (2, 3, 4):
        for start in range(0, max(0, len(chars) - size + 1)):
            terms.add("".join(chars[start : start + size]))
    for token in re.findall(r"[A-Za-z0-9_]{2,}", question or ""):
        terms.add(token.lower())
    return terms


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；;])", compact_text(text))
    return [part.strip() for part in parts if part.strip()]


def _template_key(answer_structure_type: str) -> str:
    return answer_structure_type.replace("_template", "")


def _sentence_score(sentence: str, question_terms: set[str], template_key: str, item: ContextItem) -> int:
    lowered = sentence.lower()
    score = sum(1 for term in question_terms if term and term in lowered)
    score += 3 * sum(1 for term in TYPE_TERMS.get(template_key, ()) if term and term in sentence)
    score += 2 * sum(1 for term in SOURCE_TERMS if term and term in sentence)
    if item.rank <= 2:
        score += 2
    return score


def _candidate_sentences(question: str, context: list[ContextItem], answer_structure_type: str) -> list[dict[str, Any]]:
    question_terms = _query_terms(question)
    template_key = _template_key(answer_structure_type)
    rows: list[dict[str, Any]] = []
    for item in context:
        for sentence_index, sentence in enumerate(_sentences(item.text)):
            rows.append(
                {
                    "sentence": sentence,
                    "score": _sentence_score(sentence, question_terms, template_key, item),
                    "chunk_id": item.chunk_id,
                    "document_title": item.document_title,
                    "rank": item.rank,
                    "sentence_index": sentence_index,
                }
            )
    rows.sort(key=lambda row: (-int(row["score"]), int(row["rank"]), int(row["sentence_index"]), len(str(row["sentence"]))))
    return rows


def _select_points(candidates: list[dict[str, Any]], *, min_points: int, max_points: int, max_chars: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sentences: set[str] = set()
    chars = 0
    for row in candidates:
        sentence = str(row["sentence"])
        if sentence in used_sentences:
            continue
        if selected and int(row["score"]) <= 0 and len(selected) >= min_points:
            break
        bounded = sentence if len(sentence) <= max_chars else sentence[: max_chars - 3].rstrip() + "..."
        selected.append({**row, "sentence": bounded})
        used_sentences.add(sentence)
        chars += len(bounded)
        if len(selected) >= max_points or chars >= max_chars * max_points:
            break
    return selected


def _sentences_with_terms(candidates: list[dict[str, Any]], terms: tuple[str, ...], limit: int = 2) -> list[dict[str, Any]]:
    rows = [row for row in candidates if any(term in str(row["sentence"]) for term in terms)]
    return _select_points(rows, min_points=0, max_points=limit, max_chars=180)


def _source_label(citations: list[dict[str, object]], context: list[ContextItem]) -> str:
    titles = [str(citation.get("document_title", "") or "") for citation in citations if citation.get("document_title")]
    if not titles:
        titles = [item.document_title for item in context if item.document_title]
    unique = []
    for title in titles:
        if title and title not in unique:
            unique.append(title)
    return "；".join(unique[:2]) if unique else "当前检索片段"


def _render_points(points: list[dict[str, Any]]) -> str:
    return "；".join(str(point["sentence"]) for point in points if point.get("sentence"))


def _template_answer(
    *,
    question: str,
    context: list[ContextItem],
    citations: list[dict[str, object]],
    question_type: str,
    difficulty: str,
    max_answer_chars: int,
) -> tuple[str, list[dict[str, Any]], list[str], bool, str]:
    answer_structure_type = question_structure_type(question, question_type, difficulty)
    candidates = _candidate_sentences(question, context, answer_structure_type)
    min_points = 2 if difficulty == "medium" else 1
    max_points = 4 if difficulty == "medium" else 3
    selected = _select_points(candidates, min_points=min_points, max_points=max_points, max_chars=180)
    evidence_insufficient = len(selected) < min_points
    source = _source_label(citations, context)
    rules = [f"answer_version=v3", f"structure={answer_structure_type}"]
    if difficulty == "medium":
        rules.append("medium_min_two_evidence_points")
    if evidence_insufficient:
        rules.append("evidence_insufficient")

    if answer_structure_type == "condition_template":
        condition_points = _sentences_with_terms(candidates, TYPE_TERMS["condition"], limit=3) or selected
        result_points = _sentences_with_terms(candidates, RESULT_TERMS, limit=2)
        rules.append("condition_complete_coverage")
        body = [
            f"适用条件：{_render_points(condition_points)}" if condition_points else "适用条件：当前检索证据不足以覆盖全部条件",
            f"处理结果：{_render_points(result_points)}" if result_points else "处理结果：当前检索证据未明确给出完整处理结果",
            f"补充依据：{_render_points(selected)}" if selected else "",
            f"依据来源：{source}",
        ]
    elif answer_structure_type == "procedure_template":
        procedure_points = _sentences_with_terms(candidates, TYPE_TERMS["procedure"], limit=4) or selected
        rules.append("procedure_step_coverage")
        body = [
            f"办理/处理步骤：{_render_points(procedure_points)}" if procedure_points else "办理/处理步骤：当前检索证据不足以形成完整流程",
            f"关键要求：{_render_points(selected[:2])}" if selected else "关键要求：当前检索证据不足",
            f"依据来源：{source}",
        ]
    elif answer_structure_type == "responsible_body_template":
        body_points = _sentences_with_terms(candidates, TYPE_TERMS["responsible_body"], limit=3) or selected
        evidence_insufficient = evidence_insufficient or not body_points
        rules.append("responsible_body_explicit")
        body = [
            f"责任主体：{_render_points(body_points)}" if body_points else "责任主体：当前检索证据不足以明确责任主体",
            f"补充依据：{_render_points(selected)}" if selected else "",
            f"依据来源：{source}",
        ]
    elif answer_structure_type == "qualification_template":
        qualification_points = _sentences_with_terms(candidates, TYPE_TERMS["qualification"], limit=3) or selected
        negative_points = _sentences_with_terms(candidates, ("不得", "不能", "取消", "不予", "不符合"), limit=2)
        rules.append("qualification_condition_split")
        body = [
            f"资格条件：{_render_points(qualification_points)}" if qualification_points else "资格条件：当前检索证据不足以覆盖资格条件",
            f"不符合条件的情形：{_render_points(negative_points)}" if negative_points else "不符合条件的情形：当前检索证据未明确列出",
            f"补充依据：{_render_points(selected)}" if selected else "",
            f"依据来源：{source}",
        ]
    else:
        rules.append("general_multi_sentence")
        body = [
            f"主要依据：{_render_points(selected)}" if selected else "主要依据：当前检索证据不足以确认答案",
            f"依据来源：{source}",
        ]

    if evidence_insufficient:
        body.append("注意：当前检索证据不足以覆盖全部条件或完整流程，以上仅为已检索片段可支持的内容。")
    answer = "根据当前检索到的制度片段，" + "；".join(part for part in body if part)
    if len(answer) > max_answer_chars:
        answer = answer[: max_answer_chars - 3].rstrip() + "..."
    return answer, selected, rules, evidence_insufficient, answer_structure_type


def generate_v3_answer_from_context(
    question: dict[str, Any],
    context: list[ContextItem],
    citations: list[dict[str, object]],
    *,
    max_answer_chars: int = 760,
) -> dict[str, Any]:
    question_text = str(question.get("question", "") or "")
    question_type = str(question.get("question_type", "") or "")
    difficulty = str(question.get("difficulty", "") or "")
    if not context or not citations:
        return {
            "generated_answer": "当前检索证据不足以确认答案。",
            "answer_version": "v3",
            "evidence_used": [],
            "improvement_rules_applied": ["answer_version=v3", "evidence_insufficient"],
            "evidence_insufficient": True,
            "answer_structure_type": question_structure_type(question_text, question_type, difficulty),
        }
    answer, evidence_points, rules, evidence_insufficient, answer_structure_type = _template_answer(
        question=question_text,
        context=context,
        citations=citations,
        question_type=question_type,
        difficulty=difficulty,
        max_answer_chars=max_answer_chars,
    )
    return {
        "generated_answer": answer,
        "answer_version": "v3",
        "evidence_used": [
            {
                "chunk_id": point.get("chunk_id", ""),
                "document_title": point.get("document_title", ""),
                "sentence": point.get("sentence", ""),
                "score": point.get("score", 0),
            }
            for point in evidence_points
        ],
        "improvement_rules_applied": rules,
        "evidence_insufficient": evidence_insufficient,
        "answer_structure_type": answer_structure_type,
    }


def generate_v3_answers(
    *,
    qa_jsonl: Path = DEFAULT_QA_JSONL,
    output_path: Path = V3_ANSWERS_JSONL,
    retriever: Any | None = None,
) -> list[dict[str, Any]]:
    questions = load_qa_questions(qa_jsonl)
    retriever = retriever or CachedRAGRetriever()
    combination = EvaluationCombination(method="real_hybrid_weighted", strategy="fixed_length", top_k=5)
    records: list[dict[str, Any]] = []
    for question in questions:
        raw_results = retriever(question["question"], combination.method, combination.strategy, combination.top_k)
        context = build_context(raw_results, top_k=combination.top_k, max_context_chars=4000, max_excerpt_chars=220)
        citation_objects = build_citations(context, max_citations=3)
        citations = [citation.to_dict() for citation in citation_objects]
        generated = generate_v3_answer_from_context(question, context, citations)
        retrieved_chunk_ids = [item.chunk_id for item in context]
        record = {
            "question_id": question["question_id"],
            "question": question["question"],
            "question_type": question.get("question_type", ""),
            "difficulty": question.get("difficulty", ""),
            "generated_answer": generated["generated_answer"],
            "answer_version": "v3",
            "evidence_used": generated["evidence_used"],
            "retrieval_method": combination.method,
            "method": combination.method,
            "requested_method": combination.method,
            "chunk_strategy": combination.strategy,
            "strategy": combination.strategy,
            "top_k": combination.top_k,
            "citations": citations,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "source_policy_titles": sorted({item.document_title for item in context if item.document_title}),
            "improvement_rules_applied": generated["improvement_rules_applied"],
            "evidence_insufficient": generated["evidence_insufficient"],
            "answer_structure_type": generated["answer_structure_type"],
            "has_answer": bool(str(generated["generated_answer"]).strip()),
            "has_citation": bool(citations),
            "source_policy_hit": source_policy_hit(question, citations),
            "related_chunk_hit": related_chunk_hit(question, combination.strategy, retrieved_chunk_ids),
        }
        records.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    return records


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _group_records(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(field, "") or "unknown")].append(row)
    result: dict[str, dict[str, Any]] = {}
    for group, rows in sorted(grouped.items()):
        summary = summarize_quality(rows)
        result[group] = {
            "question_count": len(rows),
            "average_keyword_overlap_ratio": summary["average_keyword_overlap_ratio"],
            "average_evidence_overlap_ratio": summary["average_evidence_overlap_ratio"],
            "auto_quality_distribution": summary["auto_quality_distribution"],
            "needs_review_count": summary["needs_review_count"],
        }
    return result


def evaluate_v3_quality(
    *,
    qa_jsonl: Path = DEFAULT_QA_JSONL,
    answers_jsonl: Path = V3_ANSWERS_JSONL,
    review_jsonl: Path = V3_REVIEW_JSONL,
    review_csv: Path = V3_REVIEW_CSV,
    report_json: Path = V3_REPORT_JSON,
    config_path: Path = DEFAULT_QUALITY_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_answer_quality_config(config_path)
    questions = {row["question_id"]: row for row in load_qa_questions(qa_jsonl)}
    answers = load_jsonl(answers_jsonl)
    records = []
    answer_by_id = {row["question_id"]: row for row in answers}
    for answer in answers:
        question_id = answer.get("question_id")
        if question_id in questions:
            quality = evaluate_answer_quality_record(questions[question_id], answer, config)
            quality["answer_version"] = answer.get("answer_version", "v3")
            quality["answer_structure_type"] = answer.get("answer_structure_type", "")
            quality["evidence_insufficient"] = bool(answer.get("evidence_insufficient", False))
            quality["improvement_rules_applied"] = answer.get("improvement_rules_applied", [])
            records.append(quality)
    summary = summarize_quality(records)
    report = {
        "stage": "rag_answer_quality_v3_evaluation",
        "total_questions": len(records),
        "average_keyword_overlap_ratio": summary["average_keyword_overlap_ratio"],
        "average_evidence_overlap_ratio": summary["average_evidence_overlap_ratio"],
        "auto_quality_distribution": summary["auto_quality_distribution"],
        "needs_review_count": summary["needs_review_count"],
        "question_type_distribution": _group_records(records, "question_type"),
        "difficulty_distribution": _group_records(records, "difficulty"),
        "condition_quality_summary": _group_records([row for row in records if row.get("question_type") == "condition"], "question_type").get("condition", {}),
        "medium_quality_summary": _group_records([row for row in records if row.get("difficulty") == "medium"], "difficulty").get("medium", {}),
        "evidence_insufficient_count": sum(1 for row in records if row.get("evidence_insufficient")),
        "answer_structure_type_distribution": _distribution([str(answer_by_id[row["question_id"]].get("answer_structure_type", "")) for row in records]),
        "output_files": {
            "answers_jsonl": str(answers_jsonl),
            "review_csv": str(review_csv),
            "review_jsonl": str(review_jsonl),
            "report_json": str(report_json),
        },
        "modified_inputs": {"v1_outputs": False, "v2_outputs": False, "v2_error_analysis": False, "raw": False, "cleaned": False, "chunks": False, "qa": False},
        "used_llm_api": False,
        "used_reranker": False,
        "used_web_frontend": False,
        "notes": "V3 answers are rule-based and use only retrieved chunks plus question metadata.",
    }
    review_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(review_jsonl, records)
    write_review_csv(review_csv, records)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _avg(records: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0.0) or 0.0) for row in records) / len(records) if records else 0.0


def _label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    return _distribution([str(row.get("auto_quality_label", "") or "unknown") for row in records])


def _change_type(v2: dict[str, Any], v3: dict[str, Any]) -> str:
    v2_label = LABEL_SCORE.get(str(v2.get("auto_quality_label", "")), 0)
    v3_label = LABEL_SCORE.get(str(v3.get("auto_quality_label", "")), 0)
    if v3_label > v2_label:
        return "improved"
    if v3_label < v2_label:
        return "worsened"
    delta = float(v3.get("keyword_overlap_ratio", 0.0) or 0.0) - float(v2.get("keyword_overlap_ratio", 0.0) or 0.0)
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "worsened"
    return "unchanged"


def _comparison_group(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "") or "unknown")].append(row)
    result: dict[str, dict[str, Any]] = {}
    for group, values in sorted(grouped.items()):
        result[group] = {
            "question_count": len(values),
            "v2_average_keyword_overlap_ratio": _avg(values, "v2_keyword_overlap_ratio"),
            "v3_average_keyword_overlap_ratio": _avg(values, "v3_keyword_overlap_ratio"),
            "v2_average_evidence_overlap_ratio": _avg(values, "v2_evidence_overlap_ratio"),
            "v3_average_evidence_overlap_ratio": _avg(values, "v3_evidence_overlap_ratio"),
            "change_distribution": _distribution([str(row["change_type"]) for row in values]),
        }
    return result


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field], ensure_ascii=False) if isinstance(row.get(field), (list, dict)) else row.get(field, "") for field in fields})


def compare_v2_v3_quality(
    *,
    v2_review_jsonl: Path = V2_REVIEW_JSONL,
    v3_review_jsonl: Path = V3_REVIEW_JSONL,
    output_csv: Path = V2_V3_COMPARISON_CSV,
    output_jsonl: Path = V2_V3_COMPARISON_JSONL,
    report_json: Path = V2_V3_COMPARISON_REPORT_JSON,
) -> dict[str, Any]:
    v2_records = {row["question_id"]: row for row in load_jsonl(v2_review_jsonl)}
    v3_records = load_jsonl(v3_review_jsonl)
    rows: list[dict[str, Any]] = []
    for v3 in v3_records:
        v2 = v2_records.get(v3.get("question_id"))
        if not v2:
            continue
        change = _change_type(v2, v3)
        rows.append(
            {
                "question_id": v3.get("question_id", ""),
                "question": v3.get("question", ""),
                "question_type": v3.get("question_type", ""),
                "difficulty": v3.get("difficulty", ""),
                "v2_auto_quality": v2.get("auto_quality_label", ""),
                "v3_auto_quality": v3.get("auto_quality_label", ""),
                "v2_keyword_overlap_ratio": v2.get("keyword_overlap_ratio", 0.0),
                "v3_keyword_overlap_ratio": v3.get("keyword_overlap_ratio", 0.0),
                "v2_evidence_overlap_ratio": v2.get("evidence_overlap_ratio", 0.0),
                "v3_evidence_overlap_ratio": v3.get("evidence_overlap_ratio", 0.0),
                "change_type": change,
                "v3_answer_structure_type": v3.get("answer_structure_type", ""),
                "v3_evidence_insufficient": v3.get("evidence_insufficient", False),
            }
        )
    report = {
        "stage": "rag_answer_quality_comparison_v2_v3",
        "v2_average_keyword_overlap_ratio": _avg(list(v2_records.values()), "keyword_overlap_ratio"),
        "v3_average_keyword_overlap_ratio": _avg(v3_records, "keyword_overlap_ratio"),
        "v2_average_evidence_overlap_ratio": _avg(list(v2_records.values()), "evidence_overlap_ratio"),
        "v3_average_evidence_overlap_ratio": _avg(v3_records, "evidence_overlap_ratio"),
        "v2_auto_quality_distribution": _label_distribution(list(v2_records.values())),
        "v3_auto_quality_distribution": _label_distribution(v3_records),
        "v2_needs_review_count": _label_distribution(list(v2_records.values())).get("needs_review", 0),
        "v3_needs_review_count": _label_distribution(v3_records).get("needs_review", 0),
        "improved_count": sum(1 for row in rows if row["change_type"] == "improved"),
        "worsened_count": sum(1 for row in rows if row["change_type"] == "worsened"),
        "unchanged_count": sum(1 for row in rows if row["change_type"] == "unchanged"),
        "condition_v2_vs_v3_summary": _comparison_group([row for row in rows if row.get("question_type") == "condition"], "question_type").get("condition", {}),
        "medium_v2_vs_v3_summary": _comparison_group([row for row in rows if row.get("difficulty") == "medium"], "difficulty").get("medium", {}),
        "question_type_change_summary": _comparison_group(rows, "question_type"),
        "difficulty_change_summary": _comparison_group(rows, "difficulty"),
        "risks_and_limitations": [
            "Automatic keyword overlap is not equivalent to final answer correctness.",
            "V3 uses rule-based templates only and may increase wording length without improving human usefulness.",
            "Condition and medium cases still require manual review before thesis-level conclusions.",
        ],
        "output_files": {
            "comparison_csv": str(output_csv),
            "comparison_jsonl": str(output_jsonl),
            "comparison_report_json": str(report_json),
        },
        "modified_inputs": {"v1_outputs": False, "v2_outputs": False, "v2_error_analysis": False, "raw": False, "cleaned": False, "chunks": False, "qa": False},
        "used_llm_api": False,
        "used_reranker": False,
        "used_web_frontend": False,
    }
    write_jsonl(output_jsonl, rows)
    write_comparison_csv(output_csv, rows)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
