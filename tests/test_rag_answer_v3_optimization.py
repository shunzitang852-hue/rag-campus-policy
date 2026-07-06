from __future__ import annotations

import json
from pathlib import Path

from src.rag.answer_v3_optimization import (
    V2_REVIEW_JSONL,
    V3_ANSWERS_JSONL,
    V3_REVIEW_JSONL,
    compare_v2_v3_quality,
    generate_v3_answer_from_context,
    generate_v3_answers,
    question_structure_type,
)
from src.rag.context_builder import build_context
from src.retrieval.demo import DemoResult
from src.retrieval.dense_index import write_jsonl


def _question(question_type: str = "condition", difficulty: str = "medium") -> dict:
    return {
        "question_id": "Q1",
        "question": "学生申请解除处分需要满足什么条件？",
        "question_type": question_type,
        "difficulty": difficulty,
    }


def _context(text: str):
    return build_context(
        [
            DemoResult(
                rank=1,
                chunk_id="chunk-1",
                document_title="学生解除处分管理办法",
                score=1.0,
                chunk_text=text,
                chunk_text_summary=text,
                strategy="fixed_length",
                retrieval_method="bm25",
            )
        ],
        top_k=1,
    )


def _citations(context):
    return [
        {
            "document_title": item.document_title,
            "chunk_id": item.chunk_id,
            "excerpt": item.excerpt,
            "score": item.score,
            "strategy": item.strategy,
            "retrieval_method": item.retrieval_method,
        }
        for item in context
    ]


def test_v3_condition_template_triggered() -> None:
    assert question_structure_type("学生申请解除处分需要满足什么条件？", "condition", "medium") == "condition_template"


def test_v3_medium_logic_uses_multiple_evidence_points() -> None:
    context = _context("学生处分期满后可以申请解除处分。申请时应当提交书面材料。学院审核后办理。")
    answer = generate_v3_answer_from_context(_question("condition", "medium"), context, _citations(context))
    assert "medium_min_two_evidence_points" in answer["improvement_rules_applied"]
    assert len(answer["evidence_used"]) >= 2


def test_v3_procedure_template_triggered() -> None:
    context = _context("学生应当提交申请，经学院审核后办理；审核通过后按规定报相关部门审批。")
    answer = generate_v3_answer_from_context(_question("procedure", "medium"), context, _citations(context))
    assert answer["answer_structure_type"] == "procedure_template"
    assert "办理/处理步骤" in answer["generated_answer"]


def test_v3_responsible_body_template_triggered() -> None:
    context = _context("学院负责学生申请材料审核，相关部门负责审批。")
    answer = generate_v3_answer_from_context(_question("responsible_body", "easy"), context, _citations(context))
    assert answer["answer_structure_type"] == "responsible_body_template"
    assert "责任主体" in answer["generated_answer"]


def test_v3_qualification_template_triggered() -> None:
    context = _context("申请人应符合评定条件，存在违纪处分情形的不得参评。")
    answer = generate_v3_answer_from_context(_question("qualification", "medium"), context, _citations(context))
    assert answer["answer_structure_type"] == "qualification_template"
    assert "资格条件" in answer["generated_answer"]


def test_v3_evidence_insufficient_flag() -> None:
    answer = generate_v3_answer_from_context(_question(), [], [])
    assert answer["evidence_insufficient"] is True
    assert answer["answer_version"] == "v3"


def test_v3_output_fields_complete(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.jsonl"
    output_path = tmp_path / "answers_v3.jsonl"
    qa_path.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "question": "学生申请解除处分需要满足什么条件？",
                "question_type": "condition",
                "difficulty": "medium",
                "related_chunk_ids_fixed_length": ["chunk-1"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def retriever(_query: str, method: str, strategy: str, _top_k: int):
        return [
            DemoResult(
                rank=1,
                chunk_id="chunk-1",
                document_title="学生解除处分管理办法",
                score=1.0,
                chunk_text="学生处分期满后可以申请解除处分。申请时应当提交书面材料。学院审核后办理。",
                chunk_text_summary="学生处分期满后可以申请解除处分。",
                strategy=strategy,
                retrieval_method=method,
            )
        ]

    records = generate_v3_answers(qa_jsonl=qa_path, output_path=output_path, retriever=retriever)
    required = {
        "question_id",
        "question",
        "question_type",
        "difficulty",
        "generated_answer",
        "answer_version",
        "evidence_used",
        "retrieval_method",
        "chunk_strategy",
        "improvement_rules_applied",
        "evidence_insufficient",
        "answer_structure_type",
    }
    assert required.issubset(records[0])
    assert records[0]["answer_version"] == "v3"
    assert output_path.exists()


def test_v2_v3_comparison_report_fields_complete(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.jsonl"
    v3 = tmp_path / "v3.jsonl"
    output_csv = tmp_path / "comparison.csv"
    output_jsonl = tmp_path / "comparison.jsonl"
    report_json = tmp_path / "report.json"
    base = {
        "question_id": "Q1",
        "question": "问题",
        "question_type": "condition",
        "difficulty": "medium",
        "keyword_overlap_ratio": 0.4,
        "evidence_overlap_ratio": 0.8,
        "auto_quality_label": "needs_review",
    }
    write_jsonl(v2, [base])
    write_jsonl(v3, [{**base, "keyword_overlap_ratio": 0.7, "auto_quality_label": "good", "answer_structure_type": "condition_template"}])
    report = compare_v2_v3_quality(
        v2_review_jsonl=v2,
        v3_review_jsonl=v3,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        report_json=report_json,
    )
    required = {
        "v2_average_keyword_overlap_ratio",
        "v3_average_keyword_overlap_ratio",
        "v2_average_evidence_overlap_ratio",
        "v3_average_evidence_overlap_ratio",
        "v2_auto_quality_distribution",
        "v3_auto_quality_distribution",
        "v2_needs_review_count",
        "v3_needs_review_count",
        "improved_count",
        "worsened_count",
        "unchanged_count",
        "condition_v2_vs_v3_summary",
        "medium_v2_vs_v3_summary",
        "question_type_change_summary",
        "difficulty_change_summary",
        "risks_and_limitations",
    }
    assert required.issubset(report)
    assert report["improved_count"] == 1


def test_v3_paths_do_not_overwrite_v1_v2_outputs() -> None:
    assert "rag_answer_quality_v3_optimization" in str(V3_ANSWERS_JSONL)
    assert "rag_answer_quality_v3_optimization" in str(V3_REVIEW_JSONL)
    assert V3_REVIEW_JSONL != V2_REVIEW_JSONL
