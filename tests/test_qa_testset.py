import csv
import json
from pathlib import Path


QA_ROOT = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset")
CSV_PATH = QA_ROOT / "hdu_ies_student_handbook_qa_testset_v1.csv"
JSONL_PATH = QA_ROOT / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
REPORT_PATH = QA_ROOT / "hdu_ies_student_handbook_qa_testset_report.json"

ALLOWED_TYPES = {
    "fact_lookup",
    "condition",
    "procedure",
    "responsible_body",
    "time_limit",
    "qualification",
    "comparison",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def load_rows():
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_qa_testset_files_exist():
    assert CSV_PATH.exists()
    assert JSONL_PATH.exists()
    assert REPORT_PATH.exists()


def test_question_count_equals_40():
    assert len(load_rows()) == 40


def test_question_ids_are_unique():
    rows = load_rows()
    ids = [row["question_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_required_text_fields_are_present():
    for row in load_rows():
        assert row["question"].strip()
        assert row["expected_answer"].strip()
        assert row["source_policy_title"].strip()
        assert row["evidence_text"].strip()
        assert int(row["evidence_start_char"]) >= 0
        assert int(row["evidence_end_char"]) > int(row["evidence_start_char"])


def test_types_and_difficulties_are_allowed():
    for row in load_rows():
        assert row["question_type"] in ALLOWED_TYPES
        assert row["difficulty"] in ALLOWED_DIFFICULTIES


def test_covers_at_least_20_policies():
    policies = {row["source_policy_title"] for row in load_rows()}
    assert len(policies) >= 20


def test_each_question_links_to_chunks():
    for row in load_rows():
        assert (
            row["related_chunk_ids_fixed_length"]
            or row["related_chunk_ids_article_based"]
            or row["related_chunk_ids_structure_aware"]
        )


def test_each_question_links_all_three_chunk_strategies():
    for row in load_rows():
        assert row["related_chunk_ids_fixed_length"]
        assert row["related_chunk_ids_article_based"]
        assert row["related_chunk_ids_structure_aware"]


def test_report_acceptance_metrics():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["question_count"] == 40
    assert report["missing_evidence_count"] == 0
    assert report["duplicate_question_count"] == 0
    assert report["covered_policy_count"] >= 20
    assert report["missing_related_chunk_count"] == 0


def test_csv_parseable_and_row_count_matches_jsonl():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(load_rows())
