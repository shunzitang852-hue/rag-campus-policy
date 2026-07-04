"""Build and load lightweight dense indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .embedding import embed_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_embedding_rows(chunks: list[dict[str, Any]], dimension: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "strategy": chunk.get("strategy") or chunk.get("chunk_strategy"),
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "chunk_text": chunk["chunk_text"],
                "embedding": embed_text(chunk["chunk_text"], dimension),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return rows


def write_index(index_path: Path, strategy: str, embeddings_path: Path, chunk_count: int, dimension: int) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": strategy,
        "index_type": "in_memory_jsonl_cosine",
        "embedding_method": "deterministic_hash_embedding",
        "embedding_dimension": dimension,
        "chunk_count": chunk_count,
        "embeddings_path": str(embeddings_path),
        "notes": "Lightweight index metadata; vectors are stored in JSONL and searched in memory.",
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
