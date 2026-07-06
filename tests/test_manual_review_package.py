from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_manual_review_package import OUTPUT_ROOT, write_manual_review_package


SCRIPT_PATH = Path(r"F:\GraduationThesis\rag-campus-policy\scripts\build_manual_review_package.py")
GUIDE_PATH = Path(r"F:\GraduationThesis\rag-campus-policy\docs\results\manual_review_package_guide.md")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manual_review_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_manual_review_guide_exists() -> None:
    assert GUIDE_PATH.exists()


def test_running_script_creates_external_output_dir(tmp_path: Path) -> None:
    summary = write_manual_review_package(tmp_path)
    assert tmp_path.exists()
    assert Path(summary["files_generated"]["manual_review_package_summary_json"]).exists()


def test_six_csv_files_are_generated(tmp_path: Path) -> None:
    summary = write_manual_review_package(tmp_path)
    csv_keys = [
        "manual_review_needs_review_cases_csv",
        "manual_review_worsened_cases_csv",
        "manual_review_condition_cases_csv",
        "manual_review_medium_cases_csv",
        "manual_review_unknown_error_cases_csv",
        "manual_review_sample_all_cases_csv",
    ]
    for key in csv_keys:
        path = Path(summary["files_generated"][key])
        assert path.exists()
        assert _read_csv(path)


def test_summary_json_is_generated(tmp_path: Path) -> None:
    summary = write_manual_review_package(tmp_path)
    path = Path(summary["files_generated"]["manual_review_package_summary_json"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "total_cases" in data
    assert data["total_cases"] == 40


def test_csvs_include_manual_notes_field(tmp_path: Path) -> None:
    summary = write_manual_review_package(tmp_path)
    for key, path_text in summary["files_generated"].items():
        if not key.endswith("_csv"):
            continue
        rows = _read_csv(Path(path_text))
        assert rows
        assert "manual_notes" in rows[0] or "manual_final_notes" in rows[0]


def test_all_cases_contains_priority_fields(tmp_path: Path) -> None:
    summary = write_manual_review_package(tmp_path)
    rows = _read_csv(Path(summary["files_generated"]["manual_review_sample_all_cases_csv"]))
    assert "recommended_review_priority" in rows[0]
    assert "priority_reason" in rows[0]
    assert {row["recommended_review_priority"] for row in rows}.issubset({"high", "medium", "low"})


def test_summary_contains_priority_counts(tmp_path: Path) -> None:
    summary = write_manual_review_package(tmp_path)
    path = Path(summary["files_generated"]["manual_review_package_summary_json"])
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["high_priority_cases", "medium_priority_cases", "low_priority_cases"]:
        assert key in data
    assert data["high_priority_cases"] + data["medium_priority_cases"] + data["low_priority_cases"] == 40


def test_default_output_root_is_external_experiment_outputs() -> None:
    resolved = OUTPUT_ROOT.resolve()
    assert str(resolved).startswith(r"F:\GraduationThesis\experiment_outputs")
    assert "rag-campus-policy" not in str(resolved)
