"""Build bounded answer context from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_MAX_CONTEXT_CHARS = 4000
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class ContextItem:
    rank: int
    chunk_id: str
    document_title: str
    score: float
    text: str
    excerpt: str
    strategy: str
    retrieval_method: str


def _field(row: Any, name: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def compact_text(text: str) -> str:
    return " ".join((text or "").split())


def make_excerpt(text: str, limit: int = 220) -> str:
    compact = compact_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def is_directory_like(text: str, title: str = "") -> bool:
    sample = (text or "").strip()[:120]
    title = title or ""
    return "目录" in sample and "第" not in sample or ("目录" in title and len(text or "") < 200)


def build_context(
    retrieval_results: Iterable[Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_excerpt_chars: int = 220,
) -> list[ContextItem]:
    """Deduplicate chunks and keep a bounded list of context items."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be positive")

    items: list[ContextItem] = []
    seen_chunk_ids: set[str] = set()
    used_chars = 0
    for row in retrieval_results:
        if len(items) >= top_k:
            break
        chunk_id = str(_field(row, "chunk_id", "") or "")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        text = compact_text(str(_field(row, "chunk_text", _field(row, "text", "")) or ""))
        title = str(_field(row, "document_title", "") or "")
        if not text or is_directory_like(text, title):
            continue
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            break
        bounded_text = text[:remaining].rstrip()
        if not bounded_text:
            break
        seen_chunk_ids.add(chunk_id)
        used_chars += len(bounded_text)
        items.append(
            ContextItem(
                rank=int(_field(row, "rank", len(items) + 1) or len(items) + 1),
                chunk_id=chunk_id,
                document_title=title,
                score=float(_field(row, "score", 0.0) or 0.0),
                text=bounded_text,
                excerpt=make_excerpt(bounded_text, max_excerpt_chars),
                strategy=str(_field(row, "strategy", "") or ""),
                retrieval_method=str(_field(row, "retrieval_method", "") or ""),
            )
        )
    return items
