from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_normalized_bm25_smoke import run_normalized_bm25_smoke
from src.experiments.thesis_export import export_thesis_tables


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_normalized_bm25_smoke_records_one_row_per_question(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cleaned.txt"
    source.write_text("奖学金申请条件。处分解除程序。", encoding="utf-8")
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks,
        [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "document_title": "学生手册",
                "policy_title": "奖学金办法",
                "chunk_text": "奖学金申请条件包括成绩优秀。",
                "chunk_strategy": "fixed_length",
            },
            {
                "chunk_id": "c2",
                "document_id": "d1",
                "document_title": "学生手册",
                "policy_title": "处分办法",
                "chunk_text": "学生可以按照程序申请解除处分。",
                "chunk_strategy": "fixed_length",
            },
        ],
    )
    qa = tmp_path / "qa.jsonl"
    _write_jsonl(
        qa,
        [
            {
                "question_id": "q1",
                "question": "奖学金申请条件是什么？",
                "question_type": "condition",
                "difficulty": "easy",
                "source_policy_title": "奖学金办法",
                "related_chunk_ids_fixed_length": ["c1"],
            },
            {
                "question_id": "q2",
                "question": "如何申请解除处分？",
                "question_type": "procedure",
                "difficulty": "medium",
                "source_policy_title": "处分办法",
                "related_chunk_ids_fixed_length": ["c2"],
            },
        ],
    )
    experiment_root = tmp_path / "outputs"
    result = run_normalized_bm25_smoke(
        experiment_root=experiment_root,
        data_source=source,
        raw_source=None,
        chunk_file=chunks,
        qa_file=qa,
        top_k=2,
        repo_root=Path(__file__).resolve().parents[1],
    )

    run_dir = Path(result["run_dir"])
    assert {path.name for path in run_dir.iterdir()} == {
        "config.json",
        "metrics.json",
        "per_question_results.jsonl",
        "summary.csv",
        "environment.json",
        "data_manifest.json",
        "git_commit.txt",
        "run.log",
    }
    per_question = [
        json.loads(line)
        for line in (run_dir / "per_question_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(per_question) == 2
    assert {row["question_id"] for row in per_question} == {"q1", "q2"}
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["question_count"] == 2
    assert metrics["metrics"]["per_question_result_count"] == 2
    assert metrics["metrics"]["top_k"] == 2
    assert "MRR" in metrics["metrics"]
    assert result["registry_row"]["legacy"] == "false"
    assert result["registry_row"]["top_k"] == "2"

    manifest = json.loads(
        (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    for role in ("data_source", "chunk_file", "qa_file"):
        assert manifest["inputs"][role]["sha256"] != "unknown"

    export_dir = tmp_path / "export"
    export_thesis_tables(
        experiment_root / "experiment_registry.csv",
        export_dir,
    )
    with (export_dir / "retrieval_method_comparison.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        export_rows = list(csv.DictReader(handle))
    assert any(
        row["experiment_id"] == result["experiment_id"]
        for row in export_rows
    )
