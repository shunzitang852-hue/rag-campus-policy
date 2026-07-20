"""Reusable real-embedding dense retrieval helpers."""

from __future__ import annotations

import math
from typing import Any

from src.retrieval.embedding import cosine_similarity
from src.retrieval.embedding_providers import validate_same_dimension


def validate_chunk_vector_alignment(
    chunks: list[dict[str, Any]],
    embedding_rows: list[dict[str, Any]],
    *,
    expected_dimension: int,
) -> dict[str, Any]:
    """Validate the immutable positional binding between chunks and vectors."""

    if len(chunks) != len(embedding_rows):
        raise ValueError(
            "chunk/vector count mismatch: "
            f"chunks={len(chunks)} vectors={len(embedding_rows)}"
        )
    for index, (chunk, row) in enumerate(zip(chunks, embedding_rows)):
        chunk_id = str(chunk.get("chunk_id", ""))
        vector_chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id or chunk_id != vector_chunk_id:
            raise ValueError(
                "chunk/vector order mismatch at index "
                f"{index}: chunk_id={chunk_id!r} "
                f"vector_chunk_id={vector_chunk_id!r}"
            )
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError(f"embedding is not a list for chunk_id={chunk_id}")
        if len(embedding) != expected_dimension:
            raise ValueError(
                "embedding dimension mismatch for "
                f"chunk_id={chunk_id}: expected={expected_dimension} "
                f"actual={len(embedding)}"
            )
        for dimension_index, value in enumerate(embedding):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "embedding contains a non-numeric value for "
                    f"chunk_id={chunk_id} dimension={dimension_index}"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(
                    "embedding contains a non-finite value for "
                    f"chunk_id={chunk_id} dimension={dimension_index}"
                )
    return {
        "chunk_count": len(chunks),
        "vector_count": len(embedding_rows),
        "embedding_dimension": expected_dimension,
        "order_binding": "same_index_and_chunk_id",
        "all_values_finite": True,
    }


def search_with_provider(
    query: str,
    embedding_rows: list[dict[str, Any]],
    provider: Any,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search precomputed rows with the existing cosine-similarity semantics.

    Python's stable sort preserves immutable vector-file order for equal scores.
    The explicit input position below documents that existing tie behavior.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_vectors = provider.encode([query])
    query_dimension = validate_same_dimension(query_vectors)
    query_vector = query_vectors[0]
    if any(not math.isfinite(float(value)) for value in query_vector):
        raise ValueError("query embedding contains non-finite values")

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for position, row in enumerate(embedding_rows):
        embedding = row["embedding"]
        if len(embedding) != query_dimension:
            raise ValueError(
                "query/vector dimension mismatch for "
                f"chunk_id={row.get('chunk_id')}: "
                f"query={query_dimension} vector={len(embedding)}"
            )
        score = cosine_similarity(query_vector, embedding)
        if not math.isfinite(score):
            raise ValueError(
                f"dense score is non-finite for chunk_id={row.get('chunk_id')}"
            )
        scored.append((score, position, row))
    scored.sort(key=lambda item: (-item[0], item[1]))

    results: list[dict[str, Any]] = []
    for rank, (score, _, row) in enumerate(scored[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "retrieved_chunk_id": row["chunk_id"],
                "retrieved_document_title": row["document_title"],
                "score": score,
                "chunk_text": row["chunk_text"],
            }
        )
    return results
