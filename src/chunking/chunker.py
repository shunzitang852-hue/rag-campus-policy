"""Deterministic chunk strategies for parsed policy records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import Chunk, ChunkConfig


DOCUMENT_TITLE = "杭州电子科技大学信息工程学院学生手册"
SOURCE_FILE = r"F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt"
SOURCE_HASH = "83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189"
DOCUMENT_TITLE = "杭州电子科技大学信息工程学院学生手册"
SOURCE_FILE = r"F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt"
RAW_SOURCE_HASH = SOURCE_HASH


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


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(re.findall(r"[A-Za-z0-9_]+|[^\s\u4e00-\u9fff]", text))
    return cjk + other


def stable_chunk_id(strategy: str, document_id: str, policy_id: str, sequence: int, text: str, record_ids: list[str]) -> str:
    source = "|".join([strategy, document_id, policy_id, str(sequence), ",".join(record_ids), hashlib.sha1(text.encode("utf-8")).hexdigest()])
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"{strategy}_{policy_id}_{sequence:04d}_{digest}"


def natural_cut(text: str, max_len: int, boundaries: str) -> int:
    if len(text) <= max_len:
        return len(text)
    window = text[:max_len]
    positions = [window.rfind(ch) for ch in boundaries]
    cut = max(positions)
    if cut >= max_len * 0.55:
        return cut + 1
    return max_len


def split_text_segments(text: str, max_len: int, overlap: int, boundaries: str) -> list[tuple[int, int, str]]:
    if not text:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = start + natural_cut(text[start:], max_len, boundaries)
        if end <= start:
            end = min(len(text), start + max_len)
        segments.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1) if overlap > 0 else end
    return segments


def group_by_policy(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["policy_id"]].append(record)
    return dict(grouped)


def record_text(record: dict[str, Any]) -> str:
    return record.get("content_clean") or record.get("content_raw") or ""


def make_chunk(
    config: ChunkConfig,
    sequence: int,
    records: list[dict[str, Any]],
    text: str,
    split_offsets: tuple[int, int] | None = None,
) -> Chunk:
    first = records[0]
    last = records[-1]
    record_ids = [row["record_id"] for row in records]
    reference_number = first.get("document_number", "")
    year_clues = sorted(set(re.findall(r"(?:19|20)\d{2}", text)))
    metadata = {
        "source_record_ids": record_ids,
        "source_start_line": min(int(row["source_start_line"]) for row in records),
        "source_end_line": max(int(row["source_end_line"]) for row in records),
        "policy_id": first["policy_id"],
        "policy_title": first["policy_title_canonical"],
        "policy_category": first.get("policy_category", ""),
        "section_title": first.get("section_title", ""),
        "heading_path": first.get("heading_path", []),
        "parse_statuses": sorted(set(row.get("parse_status", "") for row in records)),
        "article_numbers": [item for item in sorted(set(row.get("article_number", "") for row in records)) if item],
        "has_reference_number": bool(reference_number.strip()),
        "reference_number": reference_number,
        "year_clues": year_clues,
        "raw_source_sha256": RAW_SOURCE_HASH,
        "cleaning_version": "cleaning-v1",
        "cleaned_source_path": SOURCE_FILE,
    }
    if split_offsets:
        metadata["split_start_offset"] = split_offsets[0]
        metadata["split_end_offset"] = split_offsets[1]
    return Chunk(
        chunk_id=stable_chunk_id(config.strategy, first["document_id"], first["policy_id"], sequence, text, record_ids),
        chunk_index=sequence,
        document_id=first["document_id"],
        document_title=DOCUMENT_TITLE,
        policy_id=first["policy_id"],
        policy_title=first["policy_title_canonical"],
        chapter_title=first.get("chapter_title", ""),
        section_title=first.get("section_title", ""),
        article_number=first.get("article_number", ""),
        chunk_text=text,
        chunk_char_count=len(text),
        chunk_token_count=estimate_tokens(text),
        source_page_start=None,
        source_page_end=None,
        source_file=SOURCE_FILE,
        source_hash=RAW_SOURCE_HASH,
        chunk_strategy=config.strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        previous_chunk_id="",
        next_chunk_id="",
        metadata=metadata,
    )


def link_chunks(chunks: list[Chunk]) -> list[Chunk]:
    for index, chunk in enumerate(chunks):
        chunk.previous_chunk_id = chunks[index - 1].chunk_id if index > 0 and chunks[index - 1].policy_id == chunk.policy_id else ""
        chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) and chunks[index + 1].policy_id == chunk.policy_id else ""
    return chunks


def chunks_from_records(config: ChunkConfig, records: list[dict[str, Any]], start_sequence: int = 1) -> list[Chunk]:
    text = "\n".join(record_text(row) for row in records if record_text(row)).strip()
    if not text:
        return []
    chunks: list[Chunk] = []
    if len(text) <= config.max_chunk_size:
        chunks.append(make_chunk(config, start_sequence, records, text))
        return chunks
    for offset_index, (start, end, segment) in enumerate(split_text_segments(text, config.max_chunk_size, config.chunk_overlap, config.sentence_boundary_chars), start=start_sequence):
        chunks.append(make_chunk(config, offset_index, records, segment.strip(), (start, end)))
    return chunks


def fixed_length_chunks(records: list[dict[str, Any]], config: ChunkConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    sequence = 1
    for policy_id, policy_records in group_by_policy(records).items():
        text = "\n".join(record_text(row) for row in policy_records if record_text(row)).strip()
        if not text:
            continue
        for start, end, segment in split_text_segments(text, config.chunk_size, config.chunk_overlap, config.sentence_boundary_chars):
            chunks.append(make_chunk(config, sequence, policy_records, segment.strip(), (start, end)))
            sequence += 1
    return link_chunks(chunks)


def article_groups(policy_records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in policy_records:
        if record.get("parse_status") == "article" and current:
            groups.append(current)
            current = [record]
        else:
            current.append(record)
    if current:
        groups.append(current)
    return groups


def merge_short_groups(groups: list[list[dict[str, Any]]], config: ChunkConfig) -> list[list[dict[str, Any]]]:
    if not config.merge_short_chunks:
        return groups
    merged: list[list[dict[str, Any]]] = []
    buffer: list[dict[str, Any]] = []
    for group in groups:
        candidate = buffer + group
        candidate_text = "\n".join(record_text(row) for row in candidate if record_text(row))
        same_policy = not buffer or buffer[0]["policy_id"] == group[0]["policy_id"]
        same_chapter = not buffer or buffer[-1].get("chapter_number", "") == group[0].get("chapter_number", "")
        if buffer and (not same_policy or not same_chapter or len(candidate_text) > config.max_chunk_size):
            merged.append(buffer)
            buffer = group
        elif len(candidate_text) < config.min_chunk_size:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = []
    if buffer:
        merged.append(buffer)
    return merged


def article_based_chunks(records: list[dict[str, Any]], config: ChunkConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    sequence = 1
    for policy_records in group_by_policy(records).values():
        groups = merge_short_groups(article_groups(policy_records), config)
        for group in groups:
            group_chunks = chunks_from_records(config, group, sequence)
            chunks.extend(group_chunks)
            sequence += len(group_chunks)
    return link_chunks(chunks)


def structure_groups(policy_records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_article = article_groups(policy_records)
    if len(by_article) > 1:
        return by_article
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key = None
    for record in policy_records:
        key = (record.get("chapter_number", ""), record.get("section_number", ""))
        if current and key != current_key and current_key != ("", ""):
            groups.append(current)
            current = [record]
        else:
            current.append(record)
        current_key = key
    if current:
        groups.append(current)
    return groups


def structure_aware_chunks(records: list[dict[str, Any]], config: ChunkConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    sequence = 1
    for policy_records in group_by_policy(records).values():
        groups = merge_short_groups(structure_groups(policy_records), config)
        for group in groups:
            group_chunks = chunks_from_records(config, group, sequence)
            chunks.extend(group_chunks)
            sequence += len(group_chunks)
    return link_chunks(chunks)


def generate_chunks(records: list[dict[str, Any]], config: ChunkConfig) -> list[Chunk]:
    if config.strategy == "fixed_length":
        return fixed_length_chunks(records, config)
    if config.strategy == "article_based":
        return article_based_chunks(records, config)
    if config.strategy == "structure_aware":
        return structure_aware_chunks(records, config)
    raise ValueError(f"Unsupported strategy: {config.strategy}")
