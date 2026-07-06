"""Rule-based answer quality assistance for generated RAG answers."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.build_real_embedding_dense_retrieval import load_simple_yaml
from src.retrieval.dense_index import load_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_answer_quality.yaml"
DEFAULT_QA_JSONL = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset" / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
DEFAULT_RAG_ANSWERS = EXPERIMENT_ROOT / "rag_evaluation" / "rag_answers_real_hybrid_weighted_fixed_length.jsonl"
DEFAULT_RAG_EVALUATION_REPORT = EXPERIMENT_ROOT / "rag_evaluation" / "rag_evaluation_report.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "rag_answer_quality"
MANUAL_FIELDS = [
    "manual_correctness_score",
    "manual_completeness_score",
    "manual_groundedness_score",
    "manual_notes",
]


def load_answer_quality_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(path)
    for key in ("keyword_min_length", "keyword_max_length", "max_keywords"):
        if key in config:
            config[key] = int(config[key])
    if "good_keyword_overlap_threshold" in config:
        config["good_keyword_overlap_threshold"] = float(config["good_keyword_overlap_threshold"])
    return config


def load_qa_questions(path: Path = DEFAULT_QA_JSONL) -> list[dict[str, Any]]:
    return load_jsonl(path)


def load_rag_answers(path: Path = DEFAULT_RAG_ANSWERS) -> list[dict[str, Any]]:
    return load_jsonl(path)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def extract_keywords(
    text: str,
    *,
    min_length: int = 2,
    max_length: int = 4,
    max_keywords: int = 30,
) -> list[str]:
    compact = compact_text(text)
    cjk = re.findall(r"[\u4e00-\u9fff]", compact)
    counter: Counter[str] = Counter()
    for size in range(min_length, max_length + 1):
        for start in range(0, max(0, len(cjk) - size + 1)):
            token = "".join(cjk[start : start + size])
            if not _is_low_value_keyword(token):
                counter[token] += 1
    for token in re.findall(r"[A-Za-z0-9_]{2,}", text or ""):
        counter[token.lower()] += 1
    return [keyword for keyword, _count in counter.most_common(max_keywords)]


def _is_low_value_keyword(keyword: str) -> bool:
    stop_fragments = {"根据", "当前", "检索", "片段", "制度", "学生", "规定", "可以", "应当", "进行"}
    return keyword in stop_fragments


def _normalize_keywords(values: list[Any], config: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in keywords:
            keywords.append(text)
    return keywords[: int(config.get("max_keywords", 30))]


def expected_keywords(question: dict[str, Any], config: dict[str, Any]) -> list[str]:
    keywords = extract_keywords(
        str(question.get("expected_answer", "")),
        min_length=int(config.get("keyword_min_length", 2)),
        max_length=int(config.get("keyword_max_length", 4)),
        max_keywords=int(config.get("max_keywords", 30)),
    )
    return _normalize_keywords(keywords, config)


def evidence_keywords(question: dict[str, Any], config: dict[str, Any]) -> list[str]:
    provided = question.get("evidence_keywords", []) or []
    keywords = _normalize_keywords(list(provided), config)
    if keywords:
        return keywords
    return extract_keywords(
        str(question.get("evidence_text", "")),
        min_length=int(config.get("keyword_min_length", 2)),
        max_length=int(config.get("keyword_max_length", 4)),
        max_keywords=int(config.get("max_keywords", 30)),
    )


def generated_keywords(answer: dict[str, Any], config: dict[str, Any]) -> list[str]:
    return extract_keywords(
        str(answer.get("generated_answer", "")),
        min_length=int(config.get("keyword_min_length", 2)),
        max_length=int(config.get("keyword_max_length", 4)),
        max_keywords=int(config.get("max_keywords", 30)),
    )


def overlap_ratio(reference_keywords: list[str], target_text: str) -> float:
    if not reference_keywords:
        return 0.0
    target = str(target_text or "").lower()
    hits = sum(1 for keyword in reference_keywords if str(keyword).lower() in target)
    return hits / len(reference_keywords)


def missing_keywords(reference_keywords: list[str], target_text: str) -> list[str]:
    target = str(target_text or "").lower()
    return [keyword for keyword in reference_keywords if str(keyword).lower() not in target]


def unsupported_terms(answer: dict[str, Any], question: dict[str, Any], generated: list[str]) -> list[str]:
    support_text = " ".join(
        [
            str(question.get("expected_answer", "")),
            str(question.get("evidence_text", "")),
            " ".join(str(item) for item in question.get("evidence_keywords", []) or []),
            " ".join(str(citation.get("excerpt", "")) for citation in answer.get("citations", []) or []),
        ]
    ).lower()
    return [keyword for keyword in generated if keyword.lower() not in support_text]


def source_policy_hit(question: dict[str, Any], answer: dict[str, Any]) -> bool:
    source_title = str(question.get("source_policy_title", "") or "")
    if not source_title:
        return False
    return any(source_title in str(citation.get("document_title", "")) for citation in answer.get("citations", []) or [])


def related_chunk_hit(question: dict[str, Any], answer: dict[str, Any], strategy: str) -> bool:
    related = set(question.get(f"related_chunk_ids_{strategy}", []) or [])
    retrieved = set(answer.get("retrieved_chunk_ids", []) or [])
    return bool(related and related.intersection(retrieved))


def auto_quality_label(record: dict[str, Any], threshold: float) -> str:
    if not record["answer_non_empty"] or not record["has_citation"]:
        return "insufficient"
    if (
        record["source_policy_hit"]
        and record["related_chunk_hit"]
        and float(record["keyword_overlap_ratio"]) >= threshold
    ):
        return "good"
    return "needs_review"


def evaluate_answer_quality_record(
    question: dict[str, Any],
    answer: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or load_answer_quality_config()
    strategy = str(config.get("strategy", answer.get("strategy", "fixed_length")))
    generated_answer = str(answer.get("generated_answer", "") or "")
    exp_keywords = expected_keywords(question, config)
    ev_keywords = evidence_keywords(question, config)
    gen_keywords = generated_keywords(answer, config)
    citations = answer.get("citations", []) or []
    record: dict[str, Any] = {
        "question_id": question.get("question_id", ""),
        "question": question.get("question", ""),
        "question_type": question.get("question_type", ""),
        "difficulty": question.get("difficulty", ""),
        "expected_answer": question.get("expected_answer", ""),
        "generated_answer": generated_answer,
        "source_policy_title": question.get("source_policy_title", ""),
        "cited_policy_titles": sorted({str(citation.get("document_title", "")) for citation in citations if citation.get("document_title")}),
        "expected_answer_keywords": exp_keywords,
        "generated_answer_keywords": gen_keywords,
        "evidence_keywords": ev_keywords,
        "keyword_overlap_ratio": overlap_ratio(exp_keywords, generated_answer),
        "evidence_overlap_ratio": overlap_ratio(ev_keywords, generated_answer),
        "source_policy_hit": source_policy_hit(question, answer),
        "related_chunk_hit": related_chunk_hit(question, answer, strategy),
        "has_citation": bool(citations),
        "answer_non_empty": bool(generated_answer.strip()),
        "answer_char_count": len(generated_answer),
        "citation_count": len(citations),
        "possible_missing_keywords": missing_keywords(exp_keywords, generated_answer),
        "possible_unsupported_terms": unsupported_terms(answer, question, gen_keywords),
    }
    record["auto_quality_label"] = auto_quality_label(
        record,
        float(config.get("good_keyword_overlap_threshold", 0.5)),
    )
    for field in MANUAL_FIELDS:
        record[field] = ""
    return record


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quality_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(record.get("auto_quality_label", "")) for record in records)
    return dict(sorted(counter.items()))


def grouped_summary(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field, "") or "unknown")].append(record)
    return {
        group: {
            "question_count": len(values),
            "average_keyword_overlap_ratio": _avg([float(row["keyword_overlap_ratio"]) for row in values]),
            "source_policy_hit_rate": _rate(sum(1 for row in values if row["source_policy_hit"]), len(values)),
            "related_chunk_hit_rate": _rate(sum(1 for row in values if row["related_chunk_hit"]), len(values)),
            "auto_quality_distribution": quality_distribution(values),
        }
        for group, values in sorted(grouped.items())
    }


def summarize_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    question_count = len(records)
    distribution = quality_distribution(records)
    return {
        "question_count": question_count,
        "average_keyword_overlap_ratio": _avg([float(row["keyword_overlap_ratio"]) for row in records]),
        "average_evidence_overlap_ratio": _avg([float(row["evidence_overlap_ratio"]) for row in records]),
        "source_policy_hit_rate": _rate(sum(1 for row in records if row["source_policy_hit"]), question_count),
        "related_chunk_hit_rate": _rate(sum(1 for row in records if row["related_chunk_hit"]), question_count),
        "answer_non_empty_rate": _rate(sum(1 for row in records if row["answer_non_empty"]), question_count),
        "citation_coverage_rate": _rate(sum(1 for row in records if row["has_citation"]), question_count),
        "auto_quality_distribution": distribution,
        "needs_review_count": distribution.get("needs_review", 0),
        "insufficient_count": distribution.get("insufficient", 0),
        "question_type": grouped_summary(records, "question_type"),
        "difficulty": grouped_summary(records, "difficulty"),
        "notes": "Rule-based metrics are auxiliary review signals and are not equivalent to final answer correctness.",
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: _csv_value(record.get(field, "")) for field in fields})


def run_answer_quality_evaluation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path | None = None,
) -> dict[str, Any]:
    config = load_answer_quality_config(config_path)
    qa_path = Path(str(config.get("qa_jsonl_path", DEFAULT_QA_JSONL)))
    answers_path = Path(str(config.get("rag_answers_path", DEFAULT_RAG_ANSWERS)))
    output_root = output_root or Path(str(config.get("output_root", DEFAULT_OUTPUT_ROOT)))
    questions = {row["question_id"]: row for row in load_qa_questions(qa_path)}
    answers = load_rag_answers(answers_path)
    records = [
        evaluate_answer_quality_record(questions[answer["question_id"]], answer, config)
        for answer in answers
        if answer.get("question_id") in questions
    ]
    report = {
        "stage": "rag_answer_quality_evaluation",
        "qa_jsonl_path": str(qa_path),
        "rag_answers_path": str(answers_path),
        "output_root": str(output_root),
        "summary": summarize_quality(records),
        "modified_inputs": {"raw": False, "cleaned": False, "chunks": False, "qa": False},
        "used_llm_api": False,
        "used_reranker": False,
        "used_web_frontend": False,
        "overwrote_historical_retrieval_outputs": False,
        "notes": "expected_answer and evidence_text are used only for evaluation, not answer generation.",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    review_jsonl = output_root / str(config.get("review_jsonl_name", "rag_answer_quality_review_v1.jsonl"))
    review_csv = output_root / str(config.get("review_csv_name", "rag_answer_quality_review_v1.csv"))
    report_json = output_root / str(config.get("report_json_name", "rag_answer_quality_report.json"))
    write_jsonl(review_jsonl, records)
    write_review_csv(review_csv, records)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output_files"] = {
        "review_jsonl": str(review_jsonl),
        "review_csv": str(review_csv),
        "report_json": str(report_json),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
