"""Build an offline anonymous review sheet and import validated scores.

Only the anonymous CSV is read. No identity key, generation records or API is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .analyze_deepseek_manual_review import ReviewError, read_csv, validate_structure, validate_scores, parse_score
from .manifests import atomic_write_text


ROOT = Path(__file__).resolve().parents[3] / "experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness/deepseek"
CSV_NAME = "deepseek_cross_model_manual_review.csv"
HTML_NAME = "blind_review_side_by_side.html"
TEMPLATE = Path(__file__).with_name("blind_review_sheet.html")
SCORE_FIELDS = ("correctness", "faithfulness", "completeness", "hallucination")
EDITABLE = (*SCORE_FIELDS, "review_note")
SCHEMA = "anonymous-side-by-side-scores-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def experiment_directory(experiment_id: str, *, root: Path | None = None, csv_name: str = CSV_NAME) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment_id):
        raise ReviewError("invalid experiment_id")
    root = (root or ROOT).resolve()
    directory = (root / experiment_id).resolve()
    if directory.parent != root or not directory.is_dir():
        raise ReviewError("experiment directory missing or outside the permitted root")
    if Path(csv_name).name != csv_name:
        raise ReviewError("invalid CSV filename")
    source = directory / csv_name
    if not source.is_file() or source.resolve().parent != directory:
        raise ReviewError("anonymous CSV missing or outside experiment directory")
    return directory


def load_source(directory: Path, csv_name: str = CSV_NAME, *, allow_empty_answers: bool = False):
    path = directory / csv_name
    original = path.read_bytes()
    fields, rows = read_csv(path)
    evidence_field = validate_structure(fields, rows, directory.name, allow_empty_answers=allow_empty_answers)
    if path.read_bytes() != original:
        raise ReviewError("CSV changed during reading; retry when editing has stopped")
    return original, fields, rows, evidence_field


def build_sheet(experiment_id: str, *, root: Path | None = None, csv_name: str = CSV_NAME,
                review_id: str | None = None, allow_empty_answers: bool = False) -> dict:
    directory = experiment_directory(experiment_id, root=root, csv_name=csv_name)
    original, _fields, rows, evidence_field = load_source(directory, csv_name, allow_empty_answers=allow_empty_answers)
    pairs = {}
    for row in rows:
        qid = row["question_id"]
        item = pairs.setdefault(qid, {"question_id": qid, "question": row["question"], "evidence": row[evidence_field], "models": {}})
        # Explicit projection: no additional columns or identity metadata enter the HTML.
        values = {k: row[k] for k in EDITABLE}
        for field in SCORE_FIELDS:
            number, issue = parse_score(values[field], field)
            if issue is None:
                values[field] = str(number)
        item["models"][row["anonymous_model"]] = {"answer": row["answer"], **values}
    payload = {"schema_version": SCHEMA, "experiment_id": review_id or experiment_id,
               "source_csv_sha256": sha256(original), "questions": list(pairs.values())}
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__REVIEW_PAYLOAD__") != 1:
        raise ReviewError("invalid HTML template")
    html = template.replace("__REVIEW_PAYLOAD__", serialized)
    target = directory / HTML_NAME
    if target.resolve().parent != directory:
        raise ReviewError("HTML destination escapes experiment directory")
    atomic_write_text(target, html)
    if (directory / csv_name).read_bytes() != original:
        raise ReviewError("original CSV changed during HTML generation")
    return {"status": "created", "html": str(target), "questions": 40, "anonymous_rows": 80,
            "source_csv_sha256": sha256(original), "source_csv_modified": False, "api_calls": 0}


def load_score_export(path: Path) -> tuple[bytes, dict]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReviewError("duplicate JSON key in score export")
            result[key] = value
        return result
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError("invalid score-export JSON") from exc
    if not isinstance(data, dict):
        raise ReviewError("score export must be a JSON object")
    return raw, data


def validate_export(data: dict, rows: list[dict], experiment_id: str, source_hash: str) -> dict:
    allowed_top = {"schema_version", "experiment_id", "source_csv_sha256", "exported_at", "complete", "rows"}
    if set(data) - allowed_top:
        raise ReviewError("unexpected fields in score export")
    if data.get("schema_version") != SCHEMA or data.get("experiment_id") != experiment_id:
        raise ReviewError("score-export schema or experiment does not match")
    if data.get("source_csv_sha256") != source_hash:
        raise ReviewError("source CSV SHA-256 mismatch; this review sheet is stale")
    exported = data.get("rows")
    if not isinstance(exported, list) or len(exported) != 80:
        raise ReviewError("score export must contain exactly 80 rows")
    lookup = {}
    allowed_row = {"question_id", "anonymous_model", *EDITABLE}
    for r in exported:
        if not isinstance(r, dict) or set(r) != allowed_row:
            raise ReviewError("score row must contain only anonymous IDs, scores and review_note")
        if not isinstance(r["question_id"], str) or r["anonymous_model"] not in ("Model A", "Model B"):
            raise ReviewError("invalid question_id or anonymous_model")
        key = (r["question_id"], r["anonymous_model"])
        if key in lookup:
            raise ReviewError("duplicate question/anonymous_model in score export")
        if not isinstance(r["review_note"], str):
            raise ReviewError("review_note must be text")
        copy = dict(r)
        for field in SCORE_FIELDS:
            if isinstance(r[field], bool) or not isinstance(r[field], (str, int, float)):
                raise ReviewError(f"invalid {field} value")
            copy[field] = str(r[field])
        lookup[key] = copy
    expected = {(r["question_id"], r["anonymous_model"]) for r in rows}
    if set(lookup) != expected or len({r["question_id"] for r in exported}) != 40:
        raise ReviewError("score export must exactly match all 40 anonymous question pairs")
    validation = validate_scores(list(lookup.values()))
    if not validation["valid"]:
        issue = validation["issues"][0]
        raise ReviewError(f"manual review incomplete or invalid: {validation['issue_count']} cells; {issue['question_id']} / {issue['anonymous_model']} / {issue['field']}")
    return lookup


def import_scores(experiment_id: str, score_path: Path, replace_existing: bool = False, *,
                  root: Path | None = None, csv_name: str = CSV_NAME, review_id: str | None = None,
                  allow_empty_answers: bool = False) -> dict:
    directory = experiment_directory(experiment_id, root=root, csv_name=csv_name)
    source = directory / csv_name
    score_path = Path(score_path).resolve()
    if score_path == source.resolve():
        raise ReviewError("score export cannot be the source CSV")
    if "blind_key" in score_path.name.casefold():
        raise ReviewError("identity-key files are forbidden; choose the downloaded score export")
    score_bytes, data = load_score_export(score_path)
    lock = directory / ".blind_review_import.lock"
    try:
        lock_handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ReviewError("an import is already in progress; lock exists") from exc
    try:
        with lock_handle:
            lock_handle.write(str(os.getpid()))
        original, fields, rows, _evidence = load_source(directory, csv_name, allow_empty_answers=allow_empty_answers)
        original_hash = sha256(original)
        lookup = validate_export(data, rows, review_id or experiment_id, original_hash)
        updated = []
        for row in rows:
            scores = lookup[(row["question_id"], row["anonymous_model"])]
            result = dict(row)
            for field in EDITABLE:
                value = str(parse_score(scores[field], field)[0]) if field in SCORE_FIELDS else scores[field]
                if not replace_existing and row[field].strip():
                    equivalent = (parse_score(row[field], field)[0] == parse_score(value, field)[0]) if field in SCORE_FIELDS else row[field] == value
                    if not equivalent:
                        raise ReviewError(f"existing {field} conflicts; use --replace-existing-scores only to deliberately replace prior review")
                    # Preserve already-filled equivalent cells byte-for-byte in CSV values.
                    value = row[field]
                result[field] = value
            updated.append(result)
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updated)
        new_bytes = text.getvalue().encode("utf-8-sig")
        if updated == rows:
            new_bytes = original
        backup_root = directory / "manual_review_backups"
        if backup_root.resolve().parent != directory:
            raise ReviewError("backup directory escapes experiment directory")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = backup_root / (stamp + "_" + uuid.uuid4().hex[:8])
        backup.mkdir(parents=True, exist_ok=False)
        backup_csv = backup / (csv_name + ".bak")
        with backup_csv.open("xb") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        if backup_csv.read_bytes() != original:
            raise ReviewError("CSV backup verification failed; original unchanged")
        manifest = {"experiment_id": experiment_id, "source_csv": str(source), "original_sha256": original_hash,
                    "backup_csv": str(backup_csv), "backup_sha256": sha256(backup_csv.read_bytes()),
                    "score_export_sha256": sha256(score_bytes), "new_csv_sha256": sha256(new_bytes),
                    "created_at": datetime.now(timezone.utc).isoformat(), "rows": 80, "questions": 40,
                    "editable_fields": list(EDITABLE), "replace_existing_scores": replace_existing, "status": "prepared"}
        atomic_write_text(backup / "backup_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if source.read_bytes() != original:
            raise ReviewError("CSV changed before import; refused to overwrite it")
        if new_bytes != original:
            atomic_write_text(source, text.getvalue(), encoding="utf-8-sig")
        if source.read_bytes() != new_bytes:
            raise ReviewError("post-import verification failed; restore using verified backup")
        receipt = {**manifest, "status": "imported" if new_bytes != original else "unchanged",
                   "original_csv_changed": new_bytes != original, "api_calls": 0}
        atomic_write_text(backup / "import_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        return receipt
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--experiment-id", required=True)
    importer = commands.add_parser("import")
    importer.add_argument("--experiment-id", required=True)
    importer.add_argument("--scores", required=True, type=Path)
    importer.add_argument("--replace-existing-scores", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_sheet(args.experiment_id) if args.command == "build" else import_scores(args.experiment_id, args.scores, args.replace_existing_scores)
    except (ReviewError, OSError) as exc:
        print(json.dumps({"status": "FAIL FAST", "error": str(exc), "api_calls": 0}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
