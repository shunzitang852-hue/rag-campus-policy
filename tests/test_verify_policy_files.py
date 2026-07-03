from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.verify_policy_files import inspect_pdf, iter_pdf_targets, sha256_file


def test_missing_file_reports_error(tmp_path: Path) -> None:
    result = inspect_pdf(tmp_path / "missing.pdf")
    assert result["exists"] is False
    assert "file_not_found" in result["errors"]


def test_non_pdf_file_reports_invalid_header(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("not a pdf", encoding="utf-8")
    result = inspect_pdf(sample)
    assert result["pdf_header"] == "false"
    assert "invalid_pdf_header" in result["errors"]


def test_empty_file_has_stable_sha256_and_invalid_header(tmp_path: Path) -> None:
    sample = tmp_path / "empty.pdf"
    sample.write_bytes(b"")
    result = inspect_pdf(sample)
    assert result["size_bytes"] == 0
    assert result["sha256"] == hashlib.sha256(b"").hexdigest()
    assert result["pdf_header"] == "false"


def test_pdf_header_check(tmp_path: Path) -> None:
    sample = tmp_path / "minimal.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    result = inspect_pdf(sample)
    assert result["pdf_header"] == "true"
    assert result["sha256"] == sha256_file(sample)


def test_sha256_result_is_stable(tmp_path: Path) -> None:
    sample = tmp_path / "stable.pdf"
    sample.write_bytes(b"%PDF-1.7\nstable")
    assert sha256_file(sample) == sha256_file(sample)


def test_directory_input_lists_only_pdf_files(tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    txt = tmp_path / "b.txt"
    pdf.write_bytes(b"%PDF-1.4\n")
    txt.write_text("ignore", encoding="utf-8")
    assert iter_pdf_targets(tmp_path) == [pdf]
