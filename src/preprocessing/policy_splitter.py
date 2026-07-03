"""Split the HDU IES student handbook into policy-level documents."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from .models import Anomaly, PolicyCatalogEntry, SplitPolicy


DOCUMENT_ID = "hdu_ies_student_handbook"
EXPECTED_RAW_SHA256 = "83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRADUATION_ROOT = PROJECT_ROOT.parent
DATASETS_ROOT = GRADUATION_ROOT / "datasets"
DEFAULT_RAW_PATH = DATASETS_ROOT / "raw" / "hdu_ies" / "student_handbook" / "hdu_ies_student_handbook_source.txt"
DEFAULT_CATALOG_PATH = DATASETS_ROOT / "metadata" / "hdu_ies" / "hdu_ies_student_handbook_document_catalog.csv"
DEFAULT_INTERIM_ROOT = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_title(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("〔", "[").replace("〕", "]")
    return re.sub(r"\s+", "", text)


def safe_filename(title: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", title).strip()
    safe = re.sub(r"\s+", "", safe)
    return safe[:120] or "untitled"


def read_catalog(path: Path) -> list[PolicyCatalogEntry]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries: list[PolicyCatalogEntry] = []
    for row in rows:
        entries.append(
            PolicyCatalogEntry(
                sequence=int(row["sequence"]),
                policy_title=row["policy_title"],
                issuer=row.get("issuer", ""),
                document_number=row.get("document_number", ""),
                policy_year=row.get("policy_year", ""),
                revision_status=row.get("revision_status", ""),
                policy_category=row.get("policy_category", ""),
                detected_in_body=row.get("detected_in_body", ""),
                title_match_status=row.get("title_match_status", ""),
                verification_status=row.get("verification_status", ""),
                notes=row.get("notes", ""),
            )
        )
    return entries


def read_raw_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _window_text(lines: list[str], index: int, width: int = 3) -> str:
    return "".join(line.strip() for line in lines[index : min(len(lines), index + width)])


def _matched_original_title(lines: list[str], start_index: int, canonical: str, width: int = 3) -> str:
    target = normalize_title(canonical)
    parts: list[str] = []
    for offset in range(width):
        line_index = start_index + offset
        if line_index >= len(lines):
            break
        parts.append(lines[line_index].strip())
        if target in normalize_title("".join(parts)):
            break
    return "".join(parts).strip()


def _target_start_offset(lines: list[str], index: int, target: str, width: int = 3) -> int | None:
    normalized_target = normalize_title(target)
    for offset in range(width - 1, -1, -1):
        if index + offset >= len(lines):
            continue
        candidate = "".join(line.strip() for line in lines[index + offset : min(len(lines), index + width)])
        if normalized_target in normalize_title(candidate):
            return offset
    return None


def find_title_match(
    lines: list[str],
    entry: PolicyCatalogEntry,
    search_start_index: int,
) -> tuple[int | None, str, str, str]:
    canonical_norm = normalize_title(entry.policy_title)
    fuzzy_markers = {
        14: "学生校解除处分管理办法",
    }
    fuzzy_marker = fuzzy_markers.get(entry.sequence, "")

    for index in range(max(0, search_start_index), len(lines)):
        window = _window_text(lines, index)
        normalized_window = normalize_title(window)
        if canonical_norm in normalized_window:
            offset = _target_start_offset(lines, index, entry.policy_title) or 0
            start_index = index + offset
            original = _matched_original_title(lines, start_index, entry.policy_title)
            return start_index, original, "exact", "normalized_title_window"
        if fuzzy_marker and normalize_title(fuzzy_marker) in normalized_window:
            offset = _target_start_offset(lines, index, fuzzy_marker) or 0
            start_index = index + offset
            original = _matched_original_title(lines, start_index, fuzzy_marker)
            return start_index, original, "fuzzy", f"controlled_marker:{fuzzy_marker}"

    return None, "", "not_found", "not_found"


def find_body_search_start(lines: list[str], catalog: list[PolicyCatalogEntry]) -> int:
    """Return a conservative index after the table-of-contents block."""
    if not catalog:
        return 0
    last_title = normalize_title(catalog[-1].policy_title)
    toc_index = 0
    for index, line in enumerate(lines):
        if normalize_title(line) == normalize_title("目 录"):
            toc_index = index
            break
    for index in range(toc_index + 1, len(lines)):
        normalized = normalize_title(line := lines[index])
        if last_title in normalized and ("\t" in line or re.search(r"\d+\s*$", line)):
            return index + 1
    return toc_index + 1


def split_policies(lines: list[str], catalog: list[PolicyCatalogEntry]) -> tuple[list[SplitPolicy], list[Anomaly]]:
    policies: list[SplitPolicy] = []
    anomalies: list[Anomaly] = []
    starts: list[tuple[PolicyCatalogEntry, int, str, str, str]] = []
    search_index = find_body_search_start(lines, catalog)

    for entry in catalog:
        index, original, match_status, match_basis = find_title_match(lines, entry, search_index)
        if index is None:
            anomalies.append(
                Anomaly(
                    anomaly_id=f"A{len(anomalies)+1:04d}",
                    policy_id=f"HDU_IES_SH_{entry.sequence:03d}",
                    anomaly_type="policy_boundary_uncertain",
                    severity="high",
                    source_start_line=0,
                    source_end_line=0,
                    original_text=entry.policy_title,
                    description="制度标题未在正文区定位，未执行该制度正文拆分。",
                    automatic_action="recorded_not_found",
                )
            )
            continue
        starts.append((entry, index, original, match_status, match_basis))
        search_index = index + 1

    for pos, (entry, start_index, original, match_status, match_basis) in enumerate(starts):
        end_index = starts[pos + 1][1] - 1 if pos + 1 < len(starts) else len(lines) - 1
        raw_text = "\n".join(lines[start_index : end_index + 1]).strip("\n")
        policy_id = f"HDU_IES_SH_{entry.sequence:03d}"
        notes = entry.notes
        verification_status = entry.verification_status or "pending_independent_verification"
        if match_status == "fuzzy":
            verification_status = "pending_manual_verification"
            fuzzy_note = (
                "目录标题与正文标题存在差异，正文疑似出现“学生校解除处分管理办法（试行）”；"
                "raw原文未修改，需结合纸质手册或其他正式版本人工核验。"
            )
            notes = f"{notes}；{fuzzy_note}" if notes else fuzzy_note
            anomalies.append(
                Anomaly(
                    anomaly_id=f"A{len(anomalies)+1:04d}",
                    policy_id=policy_id,
                    anomaly_type="title_mismatch",
                    severity="medium",
                    source_start_line=start_index + 1,
                    source_end_line=start_index + 1,
                    original_text=original,
                    description=f"目录规范标题为“{entry.policy_title}”，正文标题为“{original}”。",
                    automatic_action=f"title_match_status=fuzzy; match_basis={match_basis}",
                )
            )

        if not entry.document_number.strip():
            anomalies.append(
                Anomaly(
                    anomaly_id=f"A{len(anomalies)+1:04d}",
                    policy_id=policy_id,
                    anomaly_type="missing_document_number",
                    severity="low",
                    source_start_line=start_index + 1,
                    source_end_line=start_index + 1,
                    original_text=original,
                    description="目录元数据中未登记明确文号。",
                    automatic_action="left_document_number_empty",
                )
            )

        policies.append(
            SplitPolicy(
                document_id=DOCUMENT_ID,
                policy_id=policy_id,
                sequence=entry.sequence,
                policy_title_original=original or entry.policy_title,
                policy_title_canonical=entry.policy_title,
                title_match_status=match_status,
                issuer=entry.issuer,
                document_number=entry.document_number,
                policy_year=entry.policy_year,
                revision_status=entry.revision_status,
                policy_category=entry.policy_category,
                source_start_line=start_index + 1,
                source_end_line=end_index + 1,
                raw_text=raw_text,
                verification_status=verification_status,
                notes=notes,
            )
        )

    return policies, anomalies


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_policy_texts(policies: list[SplitPolicy], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for policy in policies:
        filename = f"{policy.sequence:03d}_{safe_filename(policy.policy_title_canonical)}.txt"
        (output_dir / filename).write_text(policy.raw_text + "\n", encoding="utf-8", newline="\n")


def run_split(
    raw_path: Path = DEFAULT_RAW_PATH,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    output_root: Path = DEFAULT_INTERIM_ROOT,
    expected_sha256: str = EXPECTED_RAW_SHA256,
) -> tuple[list[SplitPolicy], list[Anomaly]]:
    actual_hash = sha256_file(raw_path)
    if actual_hash != expected_sha256:
        raise RuntimeError(f"raw SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}")

    lines = read_raw_lines(raw_path)
    catalog = read_catalog(catalog_path)
    policies, anomalies = split_policies(lines, catalog)

    write_policy_texts(policies, output_root / "policies")
    write_jsonl(output_root / "hdu_ies_student_handbook_policies.jsonl", [p.to_dict() for p in policies])
    write_jsonl(output_root / "parse_anomalies.jsonl", [a.to_dict() for a in anomalies])
    return policies, anomalies
