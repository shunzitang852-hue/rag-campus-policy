"""Data models for chunk generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChunkConfig:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunk_size: int
    merge_short_chunks: bool
    preserve_article_boundary: bool
    sentence_boundary_chars: str
    output_format: str
    version: str


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    document_id: str
    document_title: str
    policy_id: str
    policy_title: str
    chapter_title: str
    section_title: str
    article_number: str
    chunk_text: str
    chunk_char_count: int
    chunk_token_count: int
    source_page_start: int | None
    source_page_end: int | None
    source_file: str
    source_hash: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    previous_chunk_id: str
    next_chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["strategy"] = self.chunk_strategy
        row["char_count"] = self.chunk_char_count
        row["start_char"] = self.metadata.get("split_start_offset", 0)
        row["end_char"] = self.metadata.get("split_end_offset", self.chunk_char_count)
        return row
