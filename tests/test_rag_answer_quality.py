from __future__ import annotations

from pathlib import Path

from src.rag.answer_quality import (
    MANUAL_FIELDS,
    evaluate_answer_quality_record,
    load_answer_quality_config,
    load_qa_questions,
    load_rag_answers,
    run_answer_quality_evaluation,
    summarize_quality,
)


QA_PATH = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl")
ANSWERS_PATH = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_answers_real_hybrid_weighted_fixed_length.jsonl")


def _question() -> dict:
    return {
        "question_id": "Q1",
        "question": "申请解除处分需要什么？",
        "question_type": "condition",
        "difficulty": "medium",
        "expected_answer": "学生处分期满后可以申请解除处分，并提交申请材料。",
        "source_policy_title": "学生解除处分管理办法",
        "evidence_text": "处分期满后，学生可以申请解除处分。",
        "evidence_keywords": ["处分期满", "申请解除处分"],
        "related_chunk_ids_fixed_length": ["chunk-1"],
    }


def _answer() -> dict:
    return {
        "question_id": "Q1",
        "question": "申请解除处分需要什么？",
        "generated_answer": "学生处分期满后，可以申请解除处分，并按要求提交申请材料。",
        "citations": [
            {
                "document_title": "学生解除处分管理办法",
                "chunk_id": "chunk-1",
                "score": 1.0,
                "excerpt": "处分期满后，学生可以申请解除处分。",
            }
        ],
        "retrieved_chunk_ids": ["chunk-1"],
        "strategy": "fixed_length",
    }


def test_rag_answer_quality_config_can_be_loaded() -> None:
    config = load_answer_quality_config()
    assert config["strategy"] == "fixed_length"
    assert config["max_keywords"] > 0


def test_qa_testset_can_be_read() -> None:
    assert len(load_qa_questions(QA_PATH)) == 40


def test_rag_answers_file_can_be_read() -> None:
    rows = load_rag_answers(ANSWERS_PATH)
    assert len(rows) == 40
    assert rows[0]["generated_answer"]


def test_single_quality_record_outputs_keyword_overlap_ratio() -> None:
    record = evaluate_answer_quality_record(_question(), _answer())
    assert "keyword_overlap_ratio" in record
    assert record["keyword_overlap_ratio"] > 0


def test_single_quality_record_outputs_auto_quality_label() -> None:
    record = evaluate_answer_quality_record(_question(), _answer())
    assert record["auto_quality_label"] in {"good", "needs_review", "insufficient"}


def test_manual_review_fields_exist() -> None:
    record = evaluate_answer_quality_record(_question(), _answer())
    for field in MANUAL_FIELDS:
        assert field in record
        assert record[field] == ""


def test_batch_quality_evaluation_handles_small_sample() -> None:
    records = [evaluate_answer_quality_record(_question(), _answer())]
    report = summarize_quality(records)
    assert report["question_count"] == 1
    assert report["auto_quality_distribution"]


def test_summary_report_contains_question_count() -> None:
    report = summarize_quality([evaluate_answer_quality_record(_question(), _answer())])
    assert report["question_count"] == 1


def test_summary_report_contains_auto_quality_distribution() -> None:
    report = summarize_quality([evaluate_answer_quality_record(_question(), _answer())])
    assert "auto_quality_distribution" in report


def test_expected_answer_and_evidence_text_are_used_only_for_evaluation(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    answers_path = tmp_path / "answers.jsonl"
    config_path = tmp_path / "config.yaml"
    qa_path.write_text(__import__("json").dumps(_question(), ensure_ascii=False) + "\n", encoding="utf-8")
    answers_path.write_text(__import__("json").dumps(_answer(), ensure_ascii=False) + "\n", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                f"qa_jsonl_path: {qa_path}",
                f"rag_answers_path: {answers_path}",
                f"output_root: {tmp_path / 'out'}",
                "strategy: fixed_length",
            ]
        ),
        encoding="utf-8",
    )
    report = run_answer_quality_evaluation(config_path=config_path)
    assert report["summary"]["question_count"] == 1
    output_text = (tmp_path / "out" / "rag_answer_quality_report.json").read_text(encoding="utf-8")
    assert "used only for evaluation" in output_text
