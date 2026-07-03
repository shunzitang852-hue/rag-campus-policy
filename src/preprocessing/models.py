"""Data models for student handbook preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PolicyCatalogEntry:
    sequence: int
    policy_title: str
    issuer: str = ""
    document_number: str = ""
    policy_year: str = ""
    revision_status: str = ""
    policy_category: str = ""
    detected_in_body: str = ""
    title_match_status: str = ""
    verification_status: str = ""
    notes: str = ""


@dataclass
class SplitPolicy:
    document_id: str
    policy_id: str
    sequence: int
    policy_title_original: str
    policy_title_canonical: str
    title_match_status: str
    issuer: str
    document_number: str
    policy_year: str
    revision_status: str
    policy_category: str
    source_start_line: int
    source_end_line: int
    raw_text: str
    verification_status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructureRecord:
    record_id: str
    document_id: str
    policy_id: str
    sequence: int
    policy_title_original: str
    policy_title_canonical: str
    policy_category: str
    issuer: str
    document_number: str
    chapter_number: str = ""
    chapter_title: str = ""
    section_number: str = ""
    section_title: str = ""
    article_number: str = ""
    article_index: int | None = None
    clause_number: str = ""
    item_number: str = ""
    heading_path: list[str] = field(default_factory=list)
    content_raw: str = ""
    source_start_line: int = 0
    source_end_line: int = 0
    parse_status: str = ""
    verification_status: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Anomaly:
    anomaly_id: str
    policy_id: str
    anomaly_type: str
    severity: str
    source_start_line: int
    source_end_line: int
    original_text: str
    description: str
    automatic_action: str
    requires_manual_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
