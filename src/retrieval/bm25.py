"""Standard-library BM25 baseline for Chinese policy chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def tokenize(text: str) -> list[str]:
    """Return reproducible lexical tokens without third-party segmenters."""
    compact = re.sub(r"\s+", "", text or "")
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", compact)
    tokens = list(cjk_chars)
    tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    tokens.extend(re.findall(r"[A-Za-z0-9_]+", text or ""))
    return tokens


def build_index(chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> dict[str, Any]:
    documents = []
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for chunk in chunks:
        tokens = tokenize(chunk["chunk_text"])
        term_frequency = Counter(tokens)
        document_frequency.update(term_frequency.keys())
        total_length += len(tokens)
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "strategy": chunk.get("strategy") or chunk.get("chunk_strategy"),
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "chunk_text": chunk["chunk_text"],
                "metadata": chunk.get("metadata", {}),
                "term_frequency": dict(term_frequency),
                "token_count": len(tokens),
            }
        )
    document_count = len(documents)
    return {
        "index_type": "standard_library_bm25",
        "tokenizer": "cjk_char_bigram",
        "k1": k1,
        "b": b,
        "document_count": document_count,
        "average_document_length": (total_length / document_count) if document_count else 0.0,
        "document_frequency": dict(document_frequency),
        "documents": documents,
    }


def idf(term: str, index: dict[str, Any]) -> float:
    document_count = int(index["document_count"])
    df = int(index["document_frequency"].get(term, 0))
    return math.log(1 + (document_count - df + 0.5) / (df + 0.5))


def score_document(query_terms: list[str], document: dict[str, Any], index: dict[str, Any]) -> float:
    k1 = float(index["k1"])
    b = float(index["b"])
    avgdl = float(index["average_document_length"]) or 1.0
    doc_len = int(document["token_count"]) or 1
    tf = document["term_frequency"]
    score = 0.0
    for term in query_terms:
        freq = int(tf.get(term, 0))
        if freq == 0:
            continue
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
        score += idf(term, index) * numerator / denominator
    return score


def search(query: str, index: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    scored = []
    for document in index["documents"]:
        scored.append((score_document(query_terms, document, index), document))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, Any]] = []
    for rank, (score, document) in enumerate(scored[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "retrieved_chunk_id": document["chunk_id"],
                "retrieved_document_title": document["document_title"],
                "score": score,
                "chunk_text": document["chunk_text"],
            }
        )
    return results
