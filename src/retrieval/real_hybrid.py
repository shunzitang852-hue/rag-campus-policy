"""Real embedding dense + BM25 hybrid fusion helpers."""

from __future__ import annotations

from typing import Any

from src.retrieval.hybrid import (
    evaluate_hybrid_strategy,
    fuse_rrf,
    fuse_weighted_score,
    group_rows_by_question,
    load_strategy_results,
    normalize_scores,
)


def _rename_dense_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["real_dense_score"] = item.pop("dense_score")
        converted.append(item)
    return converted


def fuse_real_weighted_score(
    question: dict[str, Any],
    strategy: str,
    real_dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    real_dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 10,
    equal_score_value: float = 0.0,
) -> list[dict[str, Any]]:
    rows = fuse_weighted_score(
        question=question,
        strategy=strategy,
        dense_rows=real_dense_rows,
        bm25_rows=bm25_rows,
        dense_weight=real_dense_weight,
        bm25_weight=bm25_weight,
        top_k=top_k,
        equal_score_value=equal_score_value,
    )
    return _rename_dense_score(rows)


def fuse_real_rrf(
    question: dict[str, Any],
    strategy: str,
    real_dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    rows = fuse_rrf(
        question=question,
        strategy=strategy,
        dense_rows=real_dense_rows,
        bm25_rows=bm25_rows,
        k=k,
        top_k=top_k,
    )
    return _rename_dense_score(rows)
