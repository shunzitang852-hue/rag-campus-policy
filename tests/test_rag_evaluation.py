from __future__ import annotations

from pathlib import Path

from src.rag.evaluation import (
    EvaluationCombination,
    build_answer_record,
    evaluate_questions,
    load_qa_questions,
    load_rag_evaluation_config,
    parse_combinations,
    run_batch_evaluation,
)
from src.rag.pipeline import RAGResponse


QA_PATH = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl")


def _response(query: str, **kwargs) -> RAGResponse:
    assert "expected_answer" not in kwargs
    assert "evidence_text" not in kwargs
    strategy = kwargs["strategy"]
    method = kwargs["method"]
    chunk_id = f"{strategy}-chunk-1"
    return RAGResponse(
        question=query,
        answer="根据当前检索到的制度片段，测试回答。",
        evidence=[],
        citations=[
            {
                "citation_id": "C1",
                "document_title": "测试制度",
                "chunk_id": chunk_id,
                "score": 1.0,
                "excerpt": "测试摘录",
                "strategy": strategy,
                "retrieval_method": method,
            }
        ],
        source_documents=["测试制度"],
        retrieval_info={
            "requested_method": method,
            "actual_method": method,
            "strategy": strategy,
            "top_k": kwargs["top_k"],
            "retrieved_chunk_ids": [chunk_id],
            "source_policy_titles": ["测试制度"],
            "fallback_used": False,
        },
        notes=[],
        answer_mode="extractive",
    )


def _sample_questions() -> list[dict]:
    return [
        {
            "question_id": "Q1",
            "question": "问题一？",
            "question_type": "fact_lookup",
            "difficulty": "easy",
            "source_policy_title": "测试制度",
            "expected_answer": "不能用于生成",
            "evidence_text": "不能用于生成",
            "related_chunk_ids_fixed_length": ["fixed_length-chunk-1"],
        },
        {
            "question_id": "Q2",
            "question": "问题二？",
            "question_type": "condition",
            "difficulty": "hard",
            "source_policy_title": "测试制度",
            "expected_answer": "不能用于生成",
            "evidence_text": "不能用于生成",
            "related_chunk_ids_fixed_length": ["fixed_length-chunk-1"],
        },
    ]


def test_rag_evaluation_config_can_be_loaded() -> None:
    config = load_rag_evaluation_config()
    assert config["default_method"] == "real_hybrid_weighted"
    assert parse_combinations(config)[0].strategy == "fixed_length"


def test_qa_testset_can_be_read() -> None:
    questions = load_qa_questions(QA_PATH)
    assert len(questions) == 40
    assert questions[0]["question_id"]


def test_single_question_can_generate_answer_and_citations() -> None:
    question = _sample_questions()[0]
    combo = EvaluationCombination("bm25", "fixed_length", 1)
    record = build_answer_record(question, _response(question["question"], method="bm25", strategy="fixed_length", top_k=1), combo)
    assert record["generated_answer"]
    assert record["citations"]


def test_evaluation_computes_answer_non_empty_rate() -> None:
    records, report = evaluate_questions(_sample_questions(), EvaluationCombination("bm25", "fixed_length", 1), answer_func=_response)
    assert records
    assert report["answer_non_empty_rate"] == 1.0


def test_evaluation_computes_citation_coverage_rate() -> None:
    _records, report = evaluate_questions(_sample_questions(), EvaluationCombination("bm25", "fixed_length", 1), answer_func=_response)
    assert report["citation_coverage_rate"] == 1.0


def test_evaluation_computes_related_chunk_hit_rate() -> None:
    _records, report = evaluate_questions(_sample_questions(), EvaluationCombination("bm25", "fixed_length", 1), answer_func=_response)
    assert report["related_chunk_hit_rate"] == 1.0


def test_grouped_metrics_by_question_type_and_difficulty() -> None:
    _records, report = evaluate_questions(_sample_questions(), EvaluationCombination("bm25", "fixed_length", 1), answer_func=_response)
    assert set(report["question_type"]) == {"condition", "fact_lookup"}
    assert set(report["difficulty"]) == {"easy", "hard"}


def test_batch_evaluation_returns_non_empty_report_for_small_sample(tmp_path: Path) -> None:
    config_path = tmp_path / "rag_eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "default_method: bm25",
                "default_strategy: fixed_length",
                "default_top_k: 1",
                "combinations:",
                "  - bm25|fixed_length|1",
            ]
        ),
        encoding="utf-8",
    )

    def answer_func(query: str, **kwargs) -> RAGResponse:
        return _response(query, **kwargs)

    report = run_batch_evaluation(
        config_path=config_path,
        output_root=tmp_path / "out",
        answer_func=answer_func,
        retriever=None,
    )
    assert report["default_report"]["question_count"] == 40
    assert (tmp_path / "out" / "rag_evaluation_report.json").exists()


def test_expected_answer_and_evidence_text_are_not_passed_to_generation() -> None:
    def strict_answer_func(query: str, **kwargs) -> RAGResponse:
        assert query
        assert "expected_answer" not in kwargs
        assert "evidence_text" not in kwargs
        return _response(query, **kwargs)

    records, _report = evaluate_questions(_sample_questions(), EvaluationCombination("bm25", "fixed_length", 1), answer_func=strict_answer_func)
    assert records[0]["question_id"] == "Q1"


def test_output_record_contains_required_fields() -> None:
    records, _report = evaluate_questions(_sample_questions()[:1], EvaluationCombination("bm25", "fixed_length", 1), answer_func=_response)
    record = records[0]
    assert record["question_id"]
    assert record["generated_answer"]
    assert record["citations"]
    assert record["retrieved_chunk_ids"]
