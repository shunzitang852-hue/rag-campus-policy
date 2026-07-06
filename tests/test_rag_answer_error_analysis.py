from __future__ import annotations

import csv
import json
from pathlib import Path

from src.rag.answer_error_analysis import (
    ERROR_ANALYSIS_REPORT_JSON,
    MANUAL_CHECK_FIELDS,
    V2_REVIEW_JSONL,
    build_error_analysis,
    build_manual_check_row,
    classify_possible_error,
    run_error_analysis,
    write_manual_check_csv,
)


def _record(**overrides):
    row = {
        "question_id": "Q1",
        "question": "申请解除处分需要满足什么条件？",
        "question_type": "condition",
        "difficulty": "medium",
        "expected_answer": "处分期满后可以申请解除处分，并提交申请材料。",
        "generated_answer": "学生可以申请解除处分。",
        "auto_quality_label": "needs_review",
        "keyword_overlap_ratio": 0.3,
        "evidence_overlap_ratio": 0.9,
        "answer_char_count": 18,
        "possible_missing_keywords": ["处分期满", "申请材料", "审核"],
        "possible_unsupported_terms": [],
        "citations": [{"excerpt": "处分期满后，学生可以申请解除处分。"}],
    }
    row.update(overrides)
    return row


def test_error_classifier_marks_condition_not_fully_covered() -> None:
    assert classify_possible_error(_record()) == "condition_not_fully_covered"


def test_manual_row_marks_condition_and_medium_focus() -> None:
    row = build_manual_check_row(_record(), v1_record=_record(auto_quality_label="good", keyword_overlap_ratio=0.6))
    assert row["is_condition_focus"] is True
    assert row["is_medium_focus"] is True
    assert row["change_type"] == "worsened"


def test_manual_check_csv_has_complete_fields(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    write_manual_check_csv(path, [build_manual_check_row(_record())])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MANUAL_CHECK_FIELDS
        rows = list(reader)
    assert rows[0]["manual_review_result"] == ""
    assert rows[0]["manual_error_type"] == ""
    assert rows[0]["manual_notes"] == ""


def test_error_analysis_report_contains_required_fields() -> None:
    analysis = build_error_analysis()
    report = analysis["report"]
    required = {
        "total_questions",
        "needs_review_count",
        "worsened_count",
        "condition_case_count",
        "medium_case_count",
        "error_type_distribution",
        "question_type_error_distribution",
        "difficulty_error_distribution",
        "condition_error_summary",
        "medium_error_summary",
        "recommendations_for_v3",
    }
    assert required.issubset(report)
    assert report["total_questions"] == 40


def test_error_analysis_does_not_target_existing_v1_v2_outputs(tmp_path: Path) -> None:
    report = run_error_analysis(output_root=tmp_path)
    output_paths = set(report["output_files"].values())
    assert str(V2_REVIEW_JSONL) not in output_paths
    assert str(ERROR_ANALYSIS_REPORT_JSON) not in output_paths
    assert all(str(tmp_path) in value for value in output_paths)
    assert json.loads((tmp_path / "rag_answer_error_analysis_v2_report.json").read_text(encoding="utf-8"))[
        "modified_inputs"
    ]["v2_outputs"] is False
