"""Conservative text cleaning for parsed policy records."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .policy_splitter import DEFAULT_INTERIM_ROOT, DATASETS_ROOT, safe_filename, write_jsonl
from .structure_parser import load_jsonl


DEFAULT_PROCESSED_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook"


def clean_text(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if cleaned != text:
        actions.append("normalize_newlines")

    replaced = cleaned.replace("\u3000", " ")
    if replaced != cleaned:
        actions.append("normalize_full_width_spaces")
    cleaned = replaced

    stripped_lines = [line.rstrip() for line in cleaned.split("\n")]
    if stripped_lines != cleaned.split("\n"):
        actions.append("strip_trailing_spaces")
    cleaned = "\n".join(stripped_lines)

    collapsed = re.sub(r"\n{3,}", "\n\n", cleaned)
    if collapsed != cleaned:
        actions.append("collapse_excess_blank_lines")
    cleaned = collapsed

    return cleaned, actions


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    content_clean, actions = clean_text(record.get("content_raw", ""))
    updated = dict(record)
    updated["content_clean"] = content_clean
    updated["cleaning_actions"] = actions
    return updated


def run_clean(
    articles_path: Path = DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_articles.jsonl",
    policies_path: Path = DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_policies.jsonl",
    output_root: Path = DEFAULT_PROCESSED_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    clean_policy_dir = output_root / "policies_clean"
    clean_policy_dir.mkdir(parents=True, exist_ok=True)

    article_rows = load_jsonl(articles_path)
    clean_articles = [clean_record(row) for row in article_rows]
    write_jsonl(output_root / "hdu_ies_student_handbook_articles_clean.jsonl", clean_articles)

    policies = load_jsonl(policies_path)
    full_clean_parts: list[str] = []
    action_counter: Counter[str] = Counter()
    raw_chars = 0
    clean_chars = 0
    for policy in policies:
        clean, actions = clean_text(policy["raw_text"])
        action_counter.update(actions)
        raw_chars += len(policy["raw_text"])
        clean_chars += len(clean)
        filename = f"{int(policy['sequence']):03d}_{safe_filename(policy['policy_title_canonical'])}.txt"
        (clean_policy_dir / filename).write_text(clean + "\n", encoding="utf-8", newline="\n")
        full_clean_parts.append(clean)

    full_clean = "\n\n".join(full_clean_parts)
    (output_root / "hdu_ies_student_handbook_full_clean.txt").write_text(
        full_clean + "\n", encoding="utf-8", newline="\n"
    )

    return {
        "article_records": len(clean_articles),
        "policy_count": len(policies),
        "raw_policy_characters": raw_chars,
        "clean_policy_characters": clean_chars,
        "cleaning_character_delta": clean_chars - raw_chars,
        "cleaning_actions": dict(action_counter),
    }
