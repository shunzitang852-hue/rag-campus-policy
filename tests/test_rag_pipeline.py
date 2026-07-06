from __future__ import annotations

import builtins

import pytest

from scripts.demo_rag_qa import main
from src.rag.answer_generator import INSUFFICIENT_MESSAGE, generate_answer
from src.rag.citation import build_citations
from src.rag.context_builder import build_context
from src.rag.pipeline import answer_query, load_rag_config
from src.retrieval.demo import DemoResult


QUERY = "学生申请解除处分需要满足什么条件？"


def _result(rank: int = 1, *, chunk_id: str = "chunk-1", score: float = 1.0) -> DemoResult:
    text = "学生申请解除处分，应当在处分期满后提出书面申请。学院审核通过后，按规定报相关部门审批。"
    return DemoResult(
        rank=rank,
        chunk_id=chunk_id,
        document_title="学生解除处分管理办法",
        score=score,
        chunk_text=text,
        chunk_text_summary=text,
        strategy="fixed_length",
        retrieval_method="bm25",
    )


def _stub_retriever(query: str, method: str, strategy: str, top_k: int):
    return [
        DemoResult(
            rank=index,
            chunk_id=f"{strategy}-{method}-{index}",
            document_title="学生解除处分管理办法",
            score=1.0 / index,
            chunk_text=f"学生解除处分第{index}条规定，申请材料和审批流程应按学院制度办理。",
            chunk_text_summary=f"学生解除处分第{index}条规定。",
            strategy=strategy,
            retrieval_method=method,
        )
        for index in range(1, top_k + 1)
    ]


def test_rag_config_can_be_loaded() -> None:
    config = load_rag_config()
    assert config["default_retrieval_method"] == "real_hybrid_weighted"
    assert config["default_strategy"] == "fixed_length"
    assert config["default_top_k"] == 5


def test_context_builder_accepts_retrieval_results_and_returns_context() -> None:
    context = build_context([_result(), _result(rank=2, chunk_id="chunk-1")], top_k=5, max_context_chars=2000)
    assert len(context) == 1
    assert context[0].chunk_id == "chunk-1"
    assert context[0].text


def test_citation_contains_required_fields() -> None:
    context = build_context([_result()], top_k=1, max_context_chars=2000)
    citations = build_citations(context)
    row = citations[0].to_dict()
    assert row["document_title"] == "学生解除处分管理办法"
    assert row["chunk_id"] == "chunk-1"
    assert row["excerpt"]


def test_answer_generator_returns_non_empty_answer_with_context() -> None:
    context = build_context([_result()], top_k=1, max_context_chars=2000)
    citations = build_citations(context)
    answer = generate_answer(QUERY, context, citations)
    assert answer.answer
    assert "申请" in answer.answer
    assert answer.evidence


def test_answer_generator_does_not_invent_without_context() -> None:
    answer = generate_answer(QUERY, [], [])
    assert answer.answer == INSUFFICIENT_MESSAGE
    assert answer.evidence == []


def test_pipeline_returns_answer_citations_and_retrieval_info_for_chinese_query() -> None:
    response = answer_query(QUERY, method="bm25", strategy="fixed_length", top_k=2, retriever=_stub_retriever)
    assert response.answer
    assert response.citations
    assert response.retrieval_info["actual_method"] == "bm25"
    assert response.retrieval_info["strategy"] == "fixed_length"


def test_pipeline_does_not_use_expected_answer_or_qa_evidence_text() -> None:
    response = answer_query(QUERY, method="bm25", strategy="fixed_length", top_k=1, retriever=_stub_retriever)
    serialized = str(response.to_dict())
    assert "expected_answer" not in serialized
    assert "evidence_text" not in serialized


def test_demo_rag_qa_can_be_imported_and_interactive_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: (_ for _ in ()).throw(EOFError))
    assert main(["--method", "bm25"]) == 0


def test_top_k_parameter_is_applied() -> None:
    response = answer_query(QUERY, method="bm25", strategy="fixed_length", top_k=2, retriever=_stub_retriever)
    assert response.retrieval_info["top_k"] == 2
    assert response.retrieval_info["context_count"] == 2


def test_method_and_strategy_parameters_are_applied() -> None:
    response = answer_query(QUERY, method="bm25", strategy="structure_aware", top_k=1, retriever=_stub_retriever)
    assert response.retrieval_info["requested_method"] == "bm25"
    assert response.citations[0]["strategy"] == "structure_aware"
    assert response.citations[0]["retrieval_method"] == "bm25"
