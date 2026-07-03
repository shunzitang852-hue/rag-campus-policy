"""Conservative line-traceable structure parser for policy text."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Anomaly, SplitPolicy, StructureRecord
from .policy_splitter import DEFAULT_INTERIM_ROOT, DOCUMENT_ID, write_jsonl


CN_NUM = "零〇一二三四五六七八九十百千万两"
CHAPTER_RE = re.compile(rf"^(第[{CN_NUM}0-9]+章)\s*(.*)$")
SECTION_RE = re.compile(rf"^(第[{CN_NUM}0-9]+节)\s*(.*)$")
ARTICLE_RE = re.compile(rf"^(第[{CN_NUM}0-9]+条)\s*(.*)$")
CLAUSE_RE = re.compile(rf"^([（(][{CN_NUM}0-9]+[）)])\s*(.*)$")
ITEM_RE = re.compile(r"^([0-9]+[.、])\s*(.*)$")
CN_LIST_RE = re.compile(rf"^([{CN_NUM}]+[、.])\s*(.*)$")
PART_RE = re.compile(rf"^(第[{CN_NUM}0-9]+[编篇])\s*(.*)$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_policies(path: Path = DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_policies.jsonl") -> list[SplitPolicy]:
    return [SplitPolicy(**row) for row in load_jsonl(path)]


def _heading_path(chapter: str, section: str, article: str) -> list[str]:
    return [item for item in [chapter, section, article] if item]


def parse_policy(policy: SplitPolicy, anomaly_start: int = 1) -> tuple[list[StructureRecord], list[Anomaly]]:
    records: list[StructureRecord] = []
    anomalies: list[Anomaly] = []
    current_chapter_number = ""
    current_chapter_title = ""
    current_section_number = ""
    current_section_title = ""
    current_article_number = ""
    current_article_index: int | None = None
    seen_articles: Counter[str] = Counter()

    for offset, line in enumerate(policy.raw_text.splitlines()):
        source_line = policy.source_start_line + offset
        stripped = line.strip()
        if not stripped:
            continue

        chapter_number = current_chapter_number
        chapter_title = current_chapter_title
        section_number = current_section_number
        section_title = current_section_title
        article_number = current_article_number
        article_index = current_article_index
        clause_number = ""
        item_number = ""
        parse_status = "paragraph"
        notes = ""

        if PART_RE.match(stripped):
            match = PART_RE.match(stripped)
            parse_status = "part_heading"
            notes = "编或篇标题"
        elif CHAPTER_RE.match(stripped):
            match = CHAPTER_RE.match(stripped)
            current_chapter_number = match.group(1)
            current_chapter_title = match.group(2).strip()
            current_section_number = ""
            current_section_title = ""
            chapter_number = current_chapter_number
            chapter_title = current_chapter_title
            section_number = ""
            section_title = ""
            parse_status = "chapter_heading"
        elif SECTION_RE.match(stripped):
            match = SECTION_RE.match(stripped)
            current_section_number = match.group(1)
            current_section_title = match.group(2).strip()
            section_number = current_section_number
            section_title = current_section_title
            parse_status = "section_heading"
        elif ARTICLE_RE.match(stripped):
            match = ARTICLE_RE.match(stripped)
            current_article_number = match.group(1)
            current_article_index = (current_article_index or 0) + 1
            seen_articles[current_article_number] += 1
            article_number = current_article_number
            article_index = current_article_index
            parse_status = "article"
            if seen_articles[current_article_number] > 1:
                anomalies.append(
                    Anomaly(
                        anomaly_id=f"A{anomaly_start + len(anomalies):04d}",
                        policy_id=policy.policy_id,
                        anomaly_type="duplicate_article_number",
                        severity="medium",
                        source_start_line=source_line,
                        source_end_line=source_line,
                        original_text=stripped,
                        description=f"重复条款编号：{current_article_number}",
                        automatic_action="record_preserved",
                    )
                )
        elif CLAUSE_RE.match(stripped):
            match = CLAUSE_RE.match(stripped)
            clause_number = match.group(1)
            parse_status = "clause"
        elif ITEM_RE.match(stripped):
            match = ITEM_RE.match(stripped)
            item_number = match.group(1)
            parse_status = "item"
        elif CN_LIST_RE.match(stripped):
            match = CN_LIST_RE.match(stripped)
            item_number = match.group(1)
            parse_status = "cn_item"
        elif "　" in line or re.search(r"\s{3,}", line):
            parse_status = "possible_table"
            anomalies.append(
                Anomaly(
                    anomaly_id=f"A{anomaly_start + len(anomalies):04d}",
                    policy_id=policy.policy_id,
                    anomaly_type="possible_table",
                    severity="low",
                    source_start_line=source_line,
                    source_end_line=source_line,
                    original_text=stripped,
                    description="该行包含较多空白或全角空格，可能来自表格或排版对齐文本。",
                    automatic_action="record_preserved",
                    requires_manual_review=True,
                )
            )
        elif not current_chapter_number and not current_article_number and len(records) > 0:
            parse_status = "unclassified"
            anomalies.append(
                Anomaly(
                    anomaly_id=f"A{anomaly_start + len(anomalies):04d}",
                    policy_id=policy.policy_id,
                    anomaly_type="unclassified_text",
                    severity="low",
                    source_start_line=source_line,
                    source_end_line=source_line,
                    original_text=stripped,
                    description="该段落未识别到章节、条款、款项或列表编号。",
                    automatic_action="record_preserved",
                )
            )

        heading_label = " ".join(_heading_path(chapter_number, section_number, article_number))
        records.append(
            StructureRecord(
                record_id=f"{policy.policy_id}_R{len(records)+1:04d}",
                document_id=DOCUMENT_ID,
                policy_id=policy.policy_id,
                sequence=policy.sequence,
                policy_title_original=policy.policy_title_original,
                policy_title_canonical=policy.policy_title_canonical,
                policy_category=policy.policy_category,
                issuer=policy.issuer,
                document_number=policy.document_number,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                section_number=section_number,
                section_title=section_title,
                article_number=article_number,
                article_index=article_index,
                clause_number=clause_number,
                item_number=item_number,
                heading_path=_heading_path(chapter_number, section_number, article_number),
                content_raw=line,
                source_start_line=source_line,
                source_end_line=source_line,
                parse_status=parse_status,
                verification_status=policy.verification_status,
                notes=notes or heading_label,
            )
        )

    return records, anomalies


def parse_policies(policies: list[SplitPolicy]) -> tuple[list[StructureRecord], list[Anomaly]]:
    all_records: list[StructureRecord] = []
    all_anomalies: list[Anomaly] = []
    for policy in policies:
        records, anomalies = parse_policy(policy, anomaly_start=len(all_anomalies) + 1)
        all_records.extend(records)
        all_anomalies.extend(anomalies)
    return all_records, all_anomalies


def run_parse(
    policies_path: Path = DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_policies.jsonl",
    anomalies_path: Path = DEFAULT_INTERIM_ROOT / "parse_anomalies.jsonl",
    articles_path: Path = DEFAULT_INTERIM_ROOT / "hdu_ies_student_handbook_articles.jsonl",
) -> tuple[list[StructureRecord], list[Anomaly]]:
    policies = load_policies(policies_path)
    records, parse_anomalies = parse_policies(policies)
    existing = [Anomaly(**row) for row in load_jsonl(anomalies_path)] if anomalies_path.exists() else []
    for index, anomaly in enumerate(parse_anomalies, start=len(existing) + 1):
        anomaly.anomaly_id = f"A{index:04d}"
    combined = existing + parse_anomalies
    write_jsonl(articles_path, [record.to_dict() for record in records])
    write_jsonl(anomalies_path, [anomaly.to_dict() for anomaly in combined])
    return records, combined
