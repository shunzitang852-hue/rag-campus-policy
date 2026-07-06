from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_experiment_results_tables import OUTPUT_ROOT, build_results_summary, write_outputs


RESULTS_DIR = Path(r"F:\GraduationThesis\rag-campus-policy\docs\results")
INDEX_DOC = RESULTS_DIR / "experiment_results_index.md"
TABLES_DOC = RESULTS_DIR / "experiment_results_tables.md"
SUMMARY_DOC = RESULTS_DIR / "experiment_results_summary.md"


def test_results_index_exists() -> None:
    assert INDEX_DOC.exists()


def test_results_tables_exists() -> None:
    assert TABLES_DOC.exists()


def test_results_summary_exists() -> None:
    assert SUMMARY_DOC.exists()


def test_tables_document_contains_default_strategy() -> None:
    text = TABLES_DOC.read_text(encoding="utf-8")
    assert "real_hybrid_weighted + fixed_length + top_k=5 + RAG v2" in text


def test_tables_document_contains_v1_to_v2_metrics() -> None:
    text = TABLES_DOC.read_text(encoding="utf-8")
    assert "0.5616666666666668 -> 0.5925" in text
    assert "0.7741666666666667 -> 0.8012499999999999" in text
    assert "8 / 5 / 27" in text


def test_tables_document_contains_v2_to_v3_metrics() -> None:
    text = TABLES_DOC.read_text(encoding="utf-8")
    assert "0.5925 -> 0.5458333333333333" in text
    assert "0.8012499999999999 -> 0.8720833333333333" in text
    assert "6 / 8 / 26" in text


def test_tables_document_says_v3_does_not_replace_v2_default() -> None:
    text = TABLES_DOC.read_text(encoding="utf-8")
    assert "v3 不替代 v2 默认方案" in text


def test_export_outputs_json_and_csv_fields_complete(tmp_path: Path) -> None:
    summary = write_outputs(tmp_path)
    output_files = summary["output_files"]
    json_path = Path(output_files["experiment_results_summary_json"])
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    required = {
        "data_consistency",
        "chunk_strategy_results",
        "retrieval_strategy_results",
        "rag_v1_summary",
        "rag_v2_summary",
        "rag_v2_error_analysis_summary",
        "rag_v3_summary",
        "rag_v2_v3_comparison",
        "default_strategy_selection_rows",
        "limitations",
    }
    assert required.issubset(data)

    for key in [
        "chunk_strategy_results_csv",
        "rag_version_comparison_csv",
        "error_type_distribution_csv",
        "default_strategy_selection_csv",
    ]:
        path = Path(output_files[key])
        assert path.exists()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows


def test_default_output_root_is_external_experiment_outputs() -> None:
    resolved = OUTPUT_ROOT.resolve()
    assert str(resolved).startswith(r"F:\GraduationThesis\experiment_outputs")
    assert "rag-campus-policy" not in str(resolved)


def test_summary_marks_v3_as_non_default() -> None:
    summary = build_results_summary()
    rows = summary["default_strategy_selection_rows"]
    v3_rows = [row for row in rows if row["component"] == "rag_v3_role"]
    assert v3_rows
    assert v3_rows[0]["use_as_default"] == "no"
