"""End-to-end extractive RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.build_real_embedding_dense_retrieval import load_simple_yaml
from src.retrieval import demo as retrieval_demo
from src.retrieval.embedding_providers import ProviderUnavailable

from .answer_generator import GeneratedAnswer, generate_answer
from .citation import Citation, build_citations
from .context_builder import ContextItem, build_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"
Retriever = Callable[[str, str, str, int], list[Any]]


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    evidence: list[dict[str, object]]
    citations: list[dict[str, object]]
    source_documents: list[str]
    retrieval_info: dict[str, object]
    notes: list[str]
    answer_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_rag_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(path)
    int_keys = {
        "default_top_k",
        "max_context_chars",
        "max_citations",
        "max_excerpt_chars",
        "max_answer_chars",
    }
    for key in int_keys:
        if key in config:
            config[key] = int(config[key])
    if "insufficient_score_threshold" in config:
        config["insufficient_score_threshold"] = float(config["insufficient_score_threshold"])
    return config


def _default_retriever(query: str, method: str, strategy: str, top_k: int) -> list[Any]:
    return retrieval_demo.retrieve(query, method=method, strategy=strategy, top_k=top_k)


def _attempt_retrieve(
    query: str,
    method: str,
    strategy: str,
    top_k: int,
    retriever: Retriever,
    fallback_methods: list[str],
) -> tuple[list[Any], str, list[str]]:
    attempted: list[str] = []
    errors: list[str] = []
    for candidate in [method, *[item for item in fallback_methods if item != method]]:
        if candidate in attempted:
            continue
        attempted.append(candidate)
        try:
            return retriever(query, candidate, strategy, top_k), candidate, errors
        except ProviderUnavailable as exc:
            errors.append(f"{candidate}: {exc}")
            continue
    return [], method, errors


def answer_query(
    query: str,
    *,
    method: str | None = None,
    strategy: str | None = None,
    top_k: int | None = None,
    answer_mode: str | None = None,
    config: dict[str, Any] | None = None,
    retriever: Retriever | None = None,
) -> RAGResponse:
    config = config or load_rag_config()
    query = retrieval_demo.validate_query(query)
    selected_method = method or str(config.get("default_retrieval_method", "real_hybrid_weighted"))
    selected_strategy = strategy or str(config.get("default_strategy", "fixed_length"))
    selected_top_k = int(top_k or config.get("default_top_k", 5))
    selected_answer_mode = answer_mode or str(config.get("answer_mode", "extractive"))
    retrieval_demo.validate_method(selected_method)
    retrieval_demo.validate_strategy(selected_strategy)
    if selected_top_k < 1:
        raise ValueError("top_k must be positive")

    retrieve_func = retriever or _default_retriever
    fallback_methods = [str(item) for item in config.get("fallback_methods", ["real_dense", "bm25"])]
    raw_results, actual_method, retrieval_errors = _attempt_retrieve(
        query,
        selected_method,
        selected_strategy,
        selected_top_k,
        retrieve_func,
        fallback_methods,
    )
    context: list[ContextItem] = build_context(
        raw_results,
        top_k=selected_top_k,
        max_context_chars=int(config.get("max_context_chars", 4000)),
        max_excerpt_chars=int(config.get("max_excerpt_chars", config.get("max_excerpt_chars", 220))),
    )
    citations: list[Citation] = build_citations(context, max_citations=int(config.get("max_citations", 3)))
    generated: GeneratedAnswer = generate_answer(
        query,
        context,
        citations,
        answer_mode=selected_answer_mode,
        max_answer_chars=int(config.get("max_answer_chars", 500)),
        insufficient_score_threshold=float(config.get("insufficient_score_threshold", 0.0)),
    )
    retrieval_info: dict[str, object] = {
        "requested_method": selected_method,
        "actual_method": actual_method,
        "strategy": selected_strategy,
        "top_k": selected_top_k,
        "result_count": len(raw_results),
        "context_count": len(context),
        "fallback_used": actual_method != selected_method,
        "fallback_errors": retrieval_errors,
    }
    notes = list(generated.notes)
    if actual_method != selected_method:
        notes.append(f"请求的检索方法 {selected_method} 不可用，已回退到 {actual_method}。")
    return RAGResponse(
        question=query,
        answer=generated.answer,
        evidence=generated.evidence,
        citations=[citation.to_dict() for citation in citations],
        source_documents=generated.source_documents,
        retrieval_info=retrieval_info,
        notes=notes,
        answer_mode=generated.answer_mode,
    )
