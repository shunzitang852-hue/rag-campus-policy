from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.experiments.recorder import ExperimentRecorder, create_experiment_id
from src.experiments.registry import load_registry, rebuild_registry
from src.experiments.thesis_export import export_thesis_tables


def _input_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "data_source": tmp_path / "数据 source.txt",
        "chunk_file": tmp_path / "chunks.jsonl",
        "qa_file": tmp_path / "qa.jsonl",
    }
    for role, path in paths.items():
        path.write_text(f"{role}\n", encoding="utf-8")
    return paths


def _record_completed_run(
    experiment_root: Path,
    tmp_path: Path,
    *,
    experiment_id: str | None = None,
) -> ExperimentRecorder:
    recorder = ExperimentRecorder(
        task_type="retrieval",
        retrieval_method="bm25",
        chunk_strategy="fixed_length",
        top_k=5,
        embedding_model="cjk_char_bigram",
        qa_version="v1",
        random_seed=42,
        inputs=_input_files(tmp_path),
        experiment_root=experiment_root,
        repo_root=Path(__file__).resolve().parents[1],
        experiment_id=experiment_id,
    )
    with recorder:
        recorder.record_results(
            {
                "question_count": 1,
                "Recall@5": 0.75,
                "MRR@10": 0.625,
            },
            [{"question_id": "q1", "rank": 1, "is_relevant": True}],
        )
    return recorder


def test_experiment_id_is_unique_and_contains_required_dimensions() -> None:
    timestamp = datetime(2026, 7, 20, 8, 9, 10, tzinfo=timezone.utc)
    first = create_experiment_id(
        "retrieval",
        "real_hybrid_weighted",
        "fixed_length",
        5,
        timestamp=timestamp,
    )
    second = create_experiment_id(
        "retrieval",
        "real_hybrid_weighted",
        "fixed_length",
        5,
        timestamp=timestamp,
    )
    assert first != second
    assert "retrieval" in first
    assert "real-hybrid-weighted" in first
    assert "fixed-length" in first
    assert "topk5" in first


def test_recorder_generates_required_json_jsonl_and_csv(tmp_path: Path) -> None:
    root = tmp_path / "experiment_outputs"
    recorder = _record_completed_run(root, tmp_path)
    expected = {
        "config.json",
        "metrics.json",
        "per_question_results.jsonl",
        "summary.csv",
        "environment.json",
        "data_manifest.json",
        "git_commit.txt",
        "run.log",
    }
    assert {path.name for path in recorder.run_dir.iterdir()} == expected
    metrics = json.loads((recorder.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"
    assert metrics["metrics"]["Recall@5"] == 0.75
    rows = (recorder.run_dir / "per_question_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert json.loads(rows[0])["question_id"] == "q1"
    with (recorder.run_dir / "summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        assert list(csv.DictReader(handle))[0]["status"] == "completed"


def test_recorder_records_failed_status_and_error(tmp_path: Path) -> None:
    recorder = ExperimentRecorder(
        task_type="rag",
        retrieval_method="bm25",
        chunk_strategy="fixed_length",
        top_k=5,
        inputs=_input_files(tmp_path),
        experiment_root=tmp_path / "outputs",
    )
    with pytest.raises(ValueError, match="intentional"):
        with recorder:
            raise ValueError("intentional")
    metrics = json.loads((recorder.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["error"]["type"] == "ValueError"
    assert metrics["error"]["message"] == "intentional"


def test_existing_experiment_directory_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _record_completed_run(root, tmp_path, experiment_id="fixed-id")
    second = ExperimentRecorder(
        task_type="retrieval",
        retrieval_method="bm25",
        chunk_strategy="fixed_length",
        top_k=5,
        inputs=_input_files(tmp_path),
        experiment_root=root,
        experiment_id="fixed-id",
    )
    with pytest.raises(FileExistsError):
        second.start()


def test_registry_can_be_rebuilt_and_tolerates_missing_fields(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    recorder = _record_completed_run(root, tmp_path)
    incomplete = root / "runs" / "incomplete-run"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    registry_path = root / "experiment_registry.csv"
    rows = rebuild_registry(root, registry_path, include_legacy=False)
    assert len(rows) == 2
    assert registry_path.exists()
    loaded = load_registry(registry_path)
    completed = next(
        row for row in loaded if row["experiment_id"] == recorder.experiment_id
    )
    assert completed["legacy"] == "false"
    assert completed["has_per_question_results"] == "true"
    incomplete_row = next(
        row for row in loaded if row["experiment_id"] == "incomplete-run"
    )
    assert incomplete_row["metadata_missing"]


def test_legacy_result_can_be_registered_without_modification(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    evaluation = root / "bm25_retrieval" / "evaluation"
    evaluation.mkdir(parents=True)
    report_path = evaluation / "bm25_evaluation_report.json"
    report = {
        "stage": "bm25_retrieval_baseline",
        "top_k_values": [1, 3, 5, 10],
        "metrics_by_strategy": {
            "fixed_length": {
                "question_count": 2,
                "Recall@5": 0.5,
                "MRR@10": 0.25,
            }
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )
    per_question = evaluation / "bm25_search_results_fixed_length.jsonl"
    per_question.write_text('{"question_id":"q1"}\n', encoding="utf-8")
    before = report_path.read_bytes()
    rows = rebuild_registry(root, include_legacy=True)
    assert len(rows) == 1
    assert rows[0]["legacy"] == "true"
    assert rows[0]["retrieval_method"] == "bm25"
    assert rows[0]["has_per_question_results"] == "true"
    assert report_path.read_bytes() == before


def test_thesis_export_uses_persisted_metrics_and_leaves_missing_blank(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    recorder = _record_completed_run(root, tmp_path)
    registry_path = root / "experiment_registry.csv"
    rebuild_registry(root, registry_path, include_legacy=False)
    output_dir = tmp_path / "thesis tables"
    manifest = export_thesis_tables(registry_path, output_dir)
    with (
        output_dir / "retrieval_method_comparison.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == recorder.experiment_id
    assert rows[0]["Recall@5"] == "0.75"
    assert rows[0]["MRR@10"] == "0.625"
    assert rows[0]["average_top1_score"] == ""
    assert manifest["missing_cell_count"] > 0
    missing_text = (output_dir / "missing_data_report.md").read_text(
        encoding="utf-8"
    )
    assert "average_top1_score" in missing_text


def test_windows_style_values_are_safely_slugged() -> None:
    experiment_id = create_experiment_id(
        r"rag\evaluation",
        r"real_hybrid\weighted",
        r"fixed\length",
        5,
    )
    assert "\\" not in experiment_id
    assert "/" not in experiment_id
