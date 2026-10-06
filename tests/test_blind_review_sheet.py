"""Import safety tests use synthetic anonymous CSVs, never real scores."""
import csv
import hashlib
import json
import socket
from pathlib import Path
from urllib import request

import pytest

from src.experiments import blind_review_sheet as sheet


@pytest.fixture(autouse=True)
def offline_and_no_key(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network/API prohibited")
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if "blind_key" in path.name.casefold():
            raise AssertionError("identity key must never be read")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    directory = root / "synthetic_sheet"
    directory.mkdir(parents=True)
    monkeypatch.setattr(sheet, "ROOT", root)
    rows = []
    for i in range(40):
        evidence = f"synthetic evidence {i}"
        for label in ("Model B", "Model A"):
            rows.append({"experiment_id": directory.name, "question_id": f"SYN_{i:03d}", "anonymous_model": label,
                "question": f"question {i}", "context": evidence, "context_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
                "answer": f"answer {i} {label}", "reference_answer": "not exported",
                "correctness": "", "faithfulness": "", "completeness": "", "hallucination": "", "review_note": ""})
    source = directory / sheet.CSV_NAME
    write_rows(source, rows)
    exported = {"schema_version": sheet.SCHEMA, "experiment_id": directory.name,
        "source_csv_sha256": sheet.sha256(source.read_bytes()), "complete": True,
        "rows": [{"question_id": r["question_id"], "anonymous_model": r["anonymous_model"],
            "correctness": "2", "faithfulness": "1", "completeness": "0", "hallucination": "0", "review_note": "review\n备注,含逗号"} for r in rows]}
    score_path = directory / "blind_review_scores.json"
    score_path.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    return directory, source, score_path, rows, exported


def test_build_anonymous_offline_html_preserves_source(experiment):
    directory, source, _scores, _rows, _exported = experiment
    original = source.read_bytes()
    result = sheet.build_sheet(directory.name)
    html = (directory / sheet.HTML_NAME).read_text(encoding="utf-8")
    assert result["questions"] == 40 and result["anonymous_rows"] == 80
    assert source.read_bytes() == original
    assert "connect-src 'none'" in html
    assert "Model A" in html and "Model B" in html
    assert "not exported" not in html
    assert "requested_model" not in html and "actual_model" not in html
    assert "__REVIEW_PAYLOAD__" not in html
    assert "http://" not in html and "https://" not in html


def test_import_creates_verified_backup_and_updates_only_scores(experiment):
    directory, source, scores, rows, _exported = experiment
    original = source.read_bytes()
    result = sheet.import_scores(directory.name, scores)
    assert result["status"] == "imported" and result["api_calls"] == 0
    backup = Path(result["backup_csv"])
    assert backup.read_bytes() == original
    assert result["original_sha256"] == sheet.sha256(original)
    manifest = json.loads((backup.parent / "backup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backup_sha256"] == sheet.sha256(original)
    _fields, imported = sheet.read_csv(source)
    for old, new in zip(rows, imported):
        assert {k: v for k, v in old.items() if k not in sheet.EDITABLE} == {k: v for k, v in new.items() if k not in sheet.EDITABLE}
        assert new["correctness"] == "2" and new["faithfulness"] == "1"
        assert new["review_note"] == "review\n备注,含逗号"
    assert not (directory / ".blind_review_import.lock").exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra_model", "unknown_question", "correctness_range", "hallucination_range", "empty", "nan", "boolean", "extra_field", "wrong_experiment", "stale_hash"])
def test_invalid_import_never_modifies_source(experiment, mutation):
    directory, source, scores, _rows, exported = experiment
    original = source.read_bytes()
    if mutation == "missing": exported["rows"].pop()
    elif mutation == "duplicate": exported["rows"][1] = dict(exported["rows"][0])
    elif mutation == "extra_model": exported["rows"][0]["anonymous_model"] = "Model C"
    elif mutation == "unknown_question": exported["rows"][0]["question_id"] = "not-in-source"
    elif mutation == "correctness_range": exported["rows"][0]["correctness"] = "3"
    elif mutation == "hallucination_range": exported["rows"][0]["hallucination"] = "2"
    elif mutation == "empty": exported["rows"][0]["faithfulness"] = ""
    elif mutation == "nan": exported["rows"][0]["completeness"] = "NaN"
    elif mutation == "boolean": exported["rows"][0]["hallucination"] = True
    elif mutation == "extra_field": exported["rows"][0]["answer"] = "tamper"
    elif mutation == "wrong_experiment": exported["experiment_id"] = "different"
    else: exported["source_csv_sha256"] = "outdated"
    scores.write_text(json.dumps(exported), encoding="utf-8")
    with pytest.raises(sheet.ReviewError): sheet.import_scores(directory.name, scores)
    assert source.read_bytes() == original
    assert not (directory / "manual_review_backups").exists()
    assert not (directory / ".blind_review_import.lock").exists()


def test_backup_precedes_replace(experiment, monkeypatch):
    directory, source, scores, _rows, _exported = experiment
    original = source.read_bytes()
    write = sheet.atomic_write_text
    seen = []
    def guarded(path, text, **kwargs):
        if path == source:
            backups = list((directory / "manual_review_backups").glob("*/*.bak"))
            assert len(backups) == 1
            assert backups[0].read_bytes() == original
            manifest = backups[0].parent / "backup_manifest.json"
            assert manifest.exists()
            assert json.loads(manifest.read_text())["original_sha256"] == sheet.sha256(original)
            seen.append(True)
        return write(path, text, **kwargs)
    monkeypatch.setattr(sheet, "atomic_write_text", guarded)
    sheet.import_scores(directory.name, scores)
    assert seen == [True]


def test_backup_failure_does_not_write_csv(experiment, monkeypatch):
    directory, source, scores, _rows, _exported = experiment
    original = source.read_bytes()
    write = sheet.atomic_write_text
    def fail(path, *args, **kwargs):
        if path.name == "backup_manifest.json": raise OSError("simulated disk failure")
        return write(path, *args, **kwargs)
    monkeypatch.setattr(sheet, "atomic_write_text", fail)
    with pytest.raises(OSError): sheet.import_scores(directory.name, scores)
    assert source.read_bytes() == original


def test_source_changed_during_backup_refused(experiment, monkeypatch):
    directory, source, scores, _rows, _exported = experiment
    write = sheet.atomic_write_text
    changed = source.read_bytes() + b"\n"
    def concurrent_edit(path, *args, **kwargs):
        result = write(path, *args, **kwargs)
        if path.name == "backup_manifest.json": source.write_bytes(changed)
        return result
    monkeypatch.setattr(sheet, "atomic_write_text", concurrent_edit)
    with pytest.raises(sheet.ReviewError, match="changed before import"):
        sheet.import_scores(directory.name, scores)
    assert source.read_bytes() == changed


def test_existing_scores_protected_unless_explicit_replacement(experiment):
    directory, source, scores, rows, exported = experiment
    rows[0]["correctness"] = "1"
    write_rows(source, rows)
    original = source.read_bytes()
    exported["source_csv_sha256"] = sheet.sha256(original)
    scores.write_text(json.dumps(exported))
    with pytest.raises(sheet.ReviewError, match="conflicts"):
        sheet.import_scores(directory.name, scores)
    assert source.read_bytes() == original
    assert sheet.import_scores(directory.name, scores, replace_existing=True)["status"] == "imported"


def test_duplicate_json_keys_rejected(experiment):
    directory, source, scores, _rows, _exported = experiment
    original = source.read_bytes()
    scores.write_text('{"rows":[],"rows":[]}')
    with pytest.raises(sheet.ReviewError, match="duplicate"):
        sheet.import_scores(directory.name, scores)
    assert source.read_bytes() == original


def test_lock_does_not_get_deleted_by_second_importer(experiment):
    directory, source, scores, _rows, _exported = experiment
    original = source.read_bytes()
    lock = directory / ".blind_review_import.lock"
    lock.write_text("first importer")
    with pytest.raises(sheet.ReviewError, match="already"):
        sheet.import_scores(directory.name, scores)
    assert lock.read_text() == "first importer"
    assert source.read_bytes() == original


def test_key_path_refused_without_reading(experiment):
    directory, _source, _scores, _rows, _exported = experiment
    with pytest.raises(sheet.ReviewError, match="forbidden"):
        sheet.import_scores(directory.name, directory / "any_blind_key.json")


def test_script_injection_is_serialized_and_not_executed(experiment):
    directory, source, _scores, rows, _exported = experiment
    rows[0]["answer"] = '</script><script>alert("bad")</script>'
    write_rows(source, rows)
    sheet.build_sheet(directory.name)
    html = (directory / sheet.HTML_NAME).read_text(encoding="utf-8")
    assert '</script><script>alert("bad")</script>' not in html
    assert r'\u003c/script\u003e' in html


def test_identity_columns_are_never_embedded(experiment):
    directory, source, _scores, rows, _exported = experiment
    for r in rows: r["requested_model"] = "hidden-identity-must-not-leak"
    write_rows(source, rows)
    sheet.build_sheet(directory.name)
    assert "hidden-identity-must-not-leak" not in (directory / sheet.HTML_NAME).read_text(encoding="utf-8")


def test_output_paths_stay_inside_current_experiment(experiment):
    directory, _source, scores, _rows, _exported = experiment
    before = set(directory.parent.parent.rglob('*')) - set(directory.rglob('*'))
    sheet.build_sheet(directory.name)
    sheet.import_scores(directory.name, scores)
    assert before == set(directory.parent.parent.rglob('*')) - set(directory.rglob('*'))
    with pytest.raises(sheet.ReviewError): sheet.build_sheet("../escape")
