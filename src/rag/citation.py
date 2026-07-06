"""Citation formatting for extractive RAG answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .context_builder import ContextItem


@dataclass(frozen=True)
class Citation:
    citation_id: str
    document_title: str
    chunk_id: str
    score: float
    excerpt: str
    strategy: str
    retrieval_method: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_citations(context: list[ContextItem], *, max_citations: int = 3) -> list[Citation]:
    if max_citations < 1:
        raise ValueError("max_citations must be positive")
    citations: list[Citation] = []
    for index, item in enumerate(context[:max_citations], start=1):
        citations.append(
            Citation(
                citation_id=f"C{index}",
                document_title=item.document_title,
                chunk_id=item.chunk_id,
                score=item.score,
                excerpt=item.excerpt,
                strategy=item.strategy,
                retrieval_method=item.retrieval_method,
            )
        )
    return citations
