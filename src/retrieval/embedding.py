"""Deterministic hash embedding baseline.

This module intentionally avoids model downloads and third-party dependencies.
The resulting vectors are reproducible lexical hash vectors, not real semantic
embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re


DEFAULT_DIMENSION = 384


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    normalized = re.sub(r"\s+", "", text or "")
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    tokens.extend(cjk_chars)
    tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    tokens.extend(re.findall(r"[A-Za-z0-9_]+", text or ""))
    return tokens


def _hash_token(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dimension
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def embed_text(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    for token in tokenize(text):
        index, sign = _hash_token(token, dimension)
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right))
