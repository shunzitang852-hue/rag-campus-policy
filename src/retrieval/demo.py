"""Retrieval-only QA demo helpers for the student handbook corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_real_embedding_dense_retrieval import CONFIG_PATH, configure_model_cache_environment, load_simple_yaml
from scripts.evaluate_real_embedding_dense_retrieval import search_with_provider
from src.retrieval.bm25 import build_index, search as bm25_search
from src.retrieval.dense_index import load_jsonl
from src.retrieval.embedding_providers import ProviderUnavailable, create_provider
from src.retrieval.hybrid import normalize_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
REAL_EMBEDDING_ROOT = EXPERIMENT_ROOT / "dense_retrieval_real_embedding" / "embeddings"
STRATEGIES = ("fixed_length", "article_based", "structure_aware")
METHODS = ("bm25", "real_dense", "real_hybrid_weighted")
DEFAULT_METHOD = "real_dense"
DEFAULT_STRATEGY = "fixed_length"
DEFAULT_TOP_K = 5
SUMMARY_CHAR_LIMIT = 300


@dataclass(frozen=True)
class DemoResult:
    rank: int
    chunk_id: str
    document_title: str
    score: float
    chunk_text: str
    chunk_text_summary: str
    strategy: str
    retrieval_method: str


def validate_query(query: str) -> str:
    value = (query or "").strip()
    if not value:
        raise ValueError("query must not be empty")
    return value


def validate_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    return strategy


def validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    return method


def chunk_path(strategy: str) -> Path:
    return CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"


def embedding_path(strategy: str) -> Path:
    return REAL_EMBEDDING_ROOT / f"{strategy}_real_embedding_embeddings.jsonl"


def load_chunks(strategy: str) -> list[dict[str, Any]]:
    validate_strategy(strategy)
    rows = load_jsonl(chunk_path(strategy))
    return [row for row in rows if not is_directory_like_chunk(row)]


def load_real_embedding_rows(strategy: str) -> list[dict[str, Any]]:
    validate_strategy(strategy)
    rows = load_jsonl(embedding_path(strategy))
    return [row for row in rows if not is_directory_like_chunk(row)]


def is_directory_like_chunk(chunk: dict[str, Any]) -> bool:
    text = (chunk.get("chunk_text") or "").strip()
    policy_title = chunk.get("policy_title") or chunk.get("metadata", {}).get("policy_title") or ""
    if not text:
        return True
    if "目录" in text[:80] and "第" not in text[:120]:
        return True
    if "目录" in str(policy_title) and len(text) < 200:
        return True
    return False


def summarize_text(text: str, limit: int = SUMMARY_CHAR_LIMIT) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _title_for_chunk(chunk: dict[str, Any]) -> str:
    return (
        chunk.get("policy_title")
        or chunk.get("metadata", {}).get("policy_title")
        or chunk.get("retrieved_document_title")
        or chunk.get("document_title")
        or ""
    )


def _chunk_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def _format_results(
    rows: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    strategy: str,
    method: str,
    score_field: str = "score",
) -> list[DemoResult]:
    results: list[DemoResult] = []
    for rank, row in enumerate(rows, start=1):
        chunk_id = row.get("retrieved_chunk_id") or row.get("chunk_id")
        chunk = chunks_by_id.get(chunk_id, row)
        chunk_text = chunk.get("chunk_text", row.get("chunk_text", ""))
        results.append(
            DemoResult(
                rank=rank,
                chunk_id=chunk_id,
                document_title=_title_for_chunk(chunk),
                score=float(row.get(score_field, row.get("score", 0.0))),
                chunk_text=chunk_text,
                chunk_text_summary=summarize_text(chunk_text),
                strategy=strategy,
                retrieval_method=method,
            )
        )
    return results


def retrieve_bm25(query: str, strategy: str, top_k: int = DEFAULT_TOP_K) -> list[DemoResult]:
    query = validate_query(query)
    chunks = load_chunks(strategy)
    chunks_by_id = _chunk_map(chunks)
    index = build_index(chunks)
    rows = bm25_search(query, index, top_k=top_k)
    return _format_results(rows, chunks_by_id, strategy, "bm25")


def _load_real_provider() -> Any:
    config = load_simple_yaml(CONFIG_PATH)
    configure_model_cache_environment(config)
    return create_provider(config)


def retrieve_real_dense(query: str, strategy: str, top_k: int = DEFAULT_TOP_K) -> list[DemoResult]:
    query = validate_query(query)
    rows = load_real_embedding_rows(strategy)
    chunks_by_id = _chunk_map(rows)
    provider = _load_real_provider()
    search_rows = search_with_provider(query, rows, provider, top_k=top_k)
    return _format_results(search_rows, chunks_by_id, strategy, "real_dense")


def retrieve_real_hybrid_weighted(query: str, strategy: str, top_k: int = DEFAULT_TOP_K) -> list[DemoResult]:
    query = validate_query(query)
    pool_size = max(top_k, 20)
    real_dense_rows = retrieve_real_dense(query, strategy, top_k=pool_size)
    bm25_rows = retrieve_bm25(query, strategy, top_k=pool_size)

    real_dense_scores = normalize_scores([row.score for row in real_dense_rows])
    bm25_scores = normalize_scores([row.score for row in bm25_rows])
    candidates: dict[str, dict[str, Any]] = {}
    for row, normalized_score in zip(real_dense_rows, real_dense_scores):
        candidates.setdefault(
            row.chunk_id,
            {
                "chunk_id": row.chunk_id,
                "document_title": row.document_title,
                "chunk_text": row.chunk_text,
                "real_dense_score": 0.0,
                "bm25_score": 0.0,
            },
        )["real_dense_score"] = normalized_score
    for row, normalized_score in zip(bm25_rows, bm25_scores):
        candidates.setdefault(
            row.chunk_id,
            {
                "chunk_id": row.chunk_id,
                "document_title": row.document_title,
                "chunk_text": row.chunk_text,
                "real_dense_score": 0.0,
                "bm25_score": 0.0,
            },
        )["bm25_score"] = normalized_score

    fused = []
    for item in candidates.values():
        item["score"] = 0.5 * item["real_dense_score"] + 0.5 * item["bm25_score"]
        fused.append(item)
    fused.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return _format_results(fused[:top_k], {item["chunk_id"]: item for item in fused}, strategy, "real_hybrid_weighted")


def retrieve(query: str, method: str = DEFAULT_METHOD, strategy: str = DEFAULT_STRATEGY, top_k: int = DEFAULT_TOP_K) -> list[DemoResult]:
    method = validate_method(method)
    strategy = validate_strategy(strategy)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if method == "bm25":
        return retrieve_bm25(query, strategy, top_k)
    if method == "real_dense":
        return retrieve_real_dense(query, strategy, top_k)
    if method == "real_hybrid_weighted":
        return retrieve_real_hybrid_weighted(query, strategy, top_k)
    raise ValueError(f"unsupported method: {method}")


def result_to_dict(result: DemoResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "document_title": result.document_title,
        "score": result.score,
        "chunk_text": result.chunk_text,
        "chunk_text_summary": result.chunk_text_summary,
        "strategy": result.strategy,
        "retrieval_method": result.retrieval_method,
    }
