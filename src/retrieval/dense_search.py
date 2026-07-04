"""Dense search over lightweight JSONL embedding files."""

from __future__ import annotations

from typing import Any

from .embedding import cosine_similarity, embed_text


def search(query: str, embedding_rows: list[dict[str, Any]], dimension: int, top_k: int = 10) -> list[dict[str, Any]]:
    query_vector = embed_text(query, dimension)
    scored = []
    for row in embedding_rows:
        scored.append((cosine_similarity(query_vector, row["embedding"]), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, Any]] = []
    for rank, (score, row) in enumerate(scored[:top_k], start=1):
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
