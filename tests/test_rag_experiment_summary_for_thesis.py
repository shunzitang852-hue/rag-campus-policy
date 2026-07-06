from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_rag_experiments_for_thesis import (
    HISTORICAL_OUTPUT_ROOTS,
    OUTPUT_JSON,
    OUTPUT_ROOT,
    build_summary,
    write_summary,
)


DOC_PATH = Path(r"F:\GraduationThesis\rag-campus-policy\docs\rag\rag_experiment_summary_for_thesis.md")


def test_summary_json_required_fields_complete(tmp_path: Path) -> None:
    output_path = tmp_path / "rag_experiment_summary_for_thesis.json"
    summary = write_summary(output_path)
    required = {
        "project_stage",
        "data_consistency",
        "chunking_summary",
        "retrieval_summary",
        "rag_v1_summary",
        "rag_v2_summary",
        "rag_v2_error_analysis_summary",
        "rag_v3_summary",
        "rag_v1_v2_comparison",
        "rag_v2_v3_comparison",
        "default_rag_answer_strategy",
        "main_findings",
        "limitations",
        "future_work",
    }
    assert required.issubset(summary)
    assert json.loads(output_path.read_text(encoding="utf-8"))["project_stage"]


def test_default_strategy_is_v2() -> None:
    summary = build_summary()
    assert summary["default_rag_answer_strategy"]["answer_version"] == "v2"
    assert summary["default_rag_answer_strategy"]["retrieval_method"] == "real_hybrid_weighted"
    assert summary["default_rag_answer_strategy"]["chunk_strategy"] == "fixed_length"


def test_v3_marked_as_targeted_experiment_version() -> None:
    summary = build_summary()
    assert summary["rag_v3_summary"]["experimental_role"] == "targeted_optimization_experiment"
    assert summary["rag_v3_summary"]["is_default_answer_strategy"] is False


def test_summary_output_does_not_target_existing_experiment_dirs() -> None:
    output_resolved = OUTPUT_ROOT.resolve()
    assert output_resolved not in {path.resolve() for path in HISTORICAL_OUTPUT_ROOTS}
    assert OUTPUT_JSON.parent == OUTPUT_ROOT
    assert all(not str(output_resolved).startswith(str(path.resolve()) + "\\") for path in HISTORICAL_OUTPUT_ROOTS)


def test_document_mentions_automatic_metric_limitations() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "自动评价指标局限性" in text
    assert "自动评价指标只能作为辅助" in text
    assert "不等于人工评价" in text


def test_document_says_v3_does_not_replace_v2_default() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "v3 不替代 v2 默认方案" in text
    assert "v3 是定向优化实验" in text
