from __future__ import annotations

from pathlib import Path

from src.rag.answer_generator import generate_answer
from src.rag.answer_improvement import (
    ALLOWED_DIAGNOSIS_LABELS,
    DEFAULT_RAG_ANSWERS,
    V2_ANSWERS_JSONL,
    analyze_needs_review,
    compare_quality,
    evaluate_v2_quality,
)
from src.rag.citation import build_citations
from src.rag.context_builder import build_context
from src.rag.pipeline import answer_query
from src.retrieval.demo import DemoResult
from src.retrieval.embedding_providers import ProviderUnavailable


def _quality_record(question_id: str = "Q1", label: str = "needs_review") -> dict:
    return {
        "question_id": question_id,
        "question": "学生申请解除处分需要满足什么条件？",
        "question_type": "condition",
        "difficulty": "medium",
        "expected_answer": "学生处分期满后可以申请解除处分，并提交申请材料。",
        "generated_answer": "学生可以提出申请。",
        "keyword_overlap_ratio": 0.2,
        "evidence_overlap_ratio": 0.4,
        "source_policy_hit": True,
        "related_chunk_hit": True,
        "answer_char_count": 8,
        "possible_missing_keywords": ["处分期满", "申请材料"],
        "possible_unsupported_terms": [],
        "cited_policy_titles": ["学生解除处分管理办法"],
        "auto_quality_label": label,
    }


def _result(text: str, *, method: str = "bm25", chunk_id: str = "chunk-1") -> DemoResult:
    return DemoResult(
        rank=1,
        chunk_id=chunk_id,
        document_title="学生解除处分管理办法",
        score=1.0,
        chunk_text=text,
        chunk_text_summary=text,
        strategy="fixed_length",
        retrieval_method=method,
    )


def test_needs_review_analysis_function_is_runnable() -> None:
    rows = analyze_needs_review([_quality_record(), _quality_record("Q2", "good")])
    assert len(rows) == 1
    assert rows[0]["question_id"] == "Q1"


def test_diagnosis_label_is_allowed() -> None:
    rows = analyze_needs_review([_quality_record()])
    assert rows[0]["diagnosis_label"] in ALLOWED_DIAGNOSIS_LABELS


def test_optimized_answer_generator_condition_question_non_empty() -> None:
    context = build_context(
        [_result("学生处分期满后，可以申请解除处分；申请时应当提交书面材料，并符合学院审核条件。")],
        top_k=1,
    )
    answer = generate_answer("申请解除处分需要满足什么条件？", context, build_citations(context))
    assert answer.answer
    assert answer.answer != "当前检索片段不足以确认答案。"


def test_optimized_answer_generator_procedure_keeps_process_terms() -> None:
    context = build_context(
        [_result("学生应当提交申请，经学院审核后办理；审核通过后按规定报相关部门审批。")],
        top_k=1,
    )
    answer = generate_answer("学生申请流程如何办理？", context, build_citations(context))
    assert any(term in answer.answer for term in ["申请", "提交", "审核", "审批", "办理"])


def test_optimized_answer_generator_still_generates_citations() -> None:
    response = answer_query(
        "学生申请解除处分需要满足什么条件？",
        method="bm25",
        strategy="fixed_length",
        top_k=1,
        retriever=lambda _query, method, _strategy, _top_k: [
            _result("学生处分期满后可以申请解除处分，并提交申请材料。", method=method)
        ],
    )
    assert response.citations
    assert response.citations[0]["chunk_id"] == "chunk-1"


def test_v2_evaluation_can_generate_comparison_report() -> None:
    v1 = [_quality_record("Q1", "needs_review")]
    v2 = [{**_quality_record("Q1", "good"), "keyword_overlap_ratio": 0.7, "evidence_overlap_ratio": 0.8}]
    comparison = compare_quality(v1, v2)
    assert comparison["improved_question_count"] == 1
    assert comparison["v2_needs_review_count"] == 0


def test_generation_does_not_use_expected_answer_or_evidence_text() -> None:
    response = answer_query(
        "学生申请解除处分需要满足什么条件？",
        method="bm25",
        strategy="fixed_length",
        top_k=1,
        retriever=lambda _query, method, _strategy, _top_k: [
            _result("检索片段只说明学生可以提交申请。", method=method)
        ],
    )
    serialized = str(response.to_dict())
    assert "expected_answer" not in serialized
    assert "evidence_text" not in serialized


def test_v1_files_are_not_overwritten_by_v2_path() -> None:
    assert V2_ANSWERS_JSONL != DEFAULT_RAG_ANSWERS
    assert "rag_answer_quality_improvement" in str(V2_ANSWERS_JSONL)


def test_answer_length_control_is_effective() -> None:
    long_text = "学生处分期满后可以申请解除处分；" * 20
    context = build_context([_result(long_text)], top_k=1)
    answer = generate_answer("申请解除处分需要满足什么条件？", context, build_citations(context), max_answer_chars=80)
    assert len(answer.answer) <= 80


def test_fallback_logic_still_available() -> None:
    calls: list[str] = []

    def retriever(_query: str, method: str, _strategy: str, _top_k: int):
        calls.append(method)
        if method == "real_hybrid_weighted":
            raise ProviderUnavailable("provider unavailable in unit test")
        return [_result("学生可以提交申请，并由学院审核。", method=method)]

    response = answer_query(
        "学生申请解除处分如何办理？",
        method="real_hybrid_weighted",
        strategy="fixed_length",
        top_k=1,
        config={"fallback_methods": ["bm25"], "max_answer_chars": 120},
        retriever=retriever,
    )
    assert response.retrieval_info["fallback_used"] is True
    assert calls == ["real_hybrid_weighted", "bm25"]


def test_evaluate_v2_quality_writes_custom_outputs(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    answers_path = tmp_path / "answers.jsonl"
    review_jsonl = tmp_path / "review.jsonl"
    review_csv = tmp_path / "review.csv"
    qa_path.write_text(
        (
            '{"question_id":"Q1","question":"学生申请解除处分需要满足什么条件？",'
            '"question_type":"condition","difficulty":"medium","expected_answer":"学生处分期满后可以申请解除处分。",'
            '"source_policy_title":"学生解除处分管理办法","evidence_text":"处分期满后，学生可以申请解除处分。",'
            '"evidence_keywords":["处分期满","申请解除处分"],"related_chunk_ids_fixed_length":["chunk-1"]}\n'
        ),
        encoding="utf-8",
    )
    answers_path.write_text(
        (
            '{"question_id":"Q1","question":"学生申请解除处分需要满足什么条件？",'
            '"generated_answer":"学生处分期满后可以申请解除处分。","strategy":"fixed_length",'
            '"citations":[{"document_title":"学生解除处分管理办法","chunk_id":"chunk-1","excerpt":"处分期满后，学生可以申请解除处分。"}],'
            '"retrieved_chunk_ids":["chunk-1"]}\n'
        ),
        encoding="utf-8",
    )
    records, summary = evaluate_v2_quality(
        qa_jsonl=qa_path,
        answers_jsonl=answers_path,
        review_jsonl=review_jsonl,
        review_csv=review_csv,
    )
    assert len(records) == 1
    assert summary["question_count"] == 1
    assert review_jsonl.exists()
    assert review_csv.exists()
