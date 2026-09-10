from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments import normalized_rag as rag
from src.experiments.registry import rebuild_registry
from src.experiments.thesis_export import build_thesis_tables
from src.rag.answer_generator import GeneratedAnswer
from src.rag.citation import build_citations
from src.rag.context_builder import ContextItem


def _question(index: int = 1) -> dict[str, object]:
    return {
        "question_id": f"Q{index:03d}", "question": "申请需要什么条件？",
        "question_type": "condition", "difficulty": "easy",
        "expected_answer": "申请需要符合条件。", "evidence_text": "申请需要符合条件。",
        "evidence_keywords": ["申请", "条件"], "source_policy_title": "测试政策",
        "related_chunk_ids_fixed_length": [f"chunk-{index}"],
    }


def _context(index: int = 1) -> list[ContextItem]:
    return [ContextItem(1, f"chunk-{index}", "测试政策", 1.0, "申请需要符合条件。", "申请需要符合条件。", "fixed_length", "real_hybrid_weighted")]


@pytest.mark.parametrize("version,function_name", [
    ("extractive_v1", "generate_answer_v1"),
    ("extractive_v2", "generate_answer"),
])
def test_extractive_dispatch_v1_v2(monkeypatch: pytest.MonkeyPatch, version: str, function_name: str) -> None:
    called: list[str] = []
    def fake(*_args: object, **_kwargs: object) -> GeneratedAnswer:
        called.append(function_name)
        return GeneratedAnswer("answer", [], [], [])
    monkeypatch.setattr(rag, function_name, fake)
    answer, details = rag._extractive_answer(version, _question(), _context(), build_citations(_context()))
    assert answer == "answer"
    assert called == [function_name]
    assert details["legacy_version"] in {"v1", "v2"}


def test_extractive_dispatch_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "generate_v3_answer_from_context", lambda *_args, **_kwargs: {"generated_answer": "v3"})
    answer, _details = rag._extractive_answer("extractive_v3", _question(), _context(), build_citations(_context()))
    assert answer == "v3"


def test_all_versions_share_one_frozen_retrieval_config() -> None:
    assert rag.FROZEN_RETRIEVAL.method == "real_hybrid_weighted"
    assert rag.FROZEN_RETRIEVAL.strategy == "fixed_length"
    assert rag.FROZEN_RETRIEVAL.top_k == 5
    assert rag.FROZEN_RETRIEVAL.candidate_pool_k == 50


def test_qa_order_must_match_retrieval_order(tmp_path: Path) -> None:
    qa = tmp_path / "qa.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    results = tmp_path / "results.jsonl"
    qa.write_text("".join(json.dumps(_question(i)) + "\n" for i in range(1, 41)), encoding="utf-8")
    chunks.write_text('{"chunk_id":"unused"}\n', encoding="utf-8")
    results.write_text("".join(json.dumps({"question_id": f"Q{i:03d}"}) + "\n" for i in range(40, 0, -1)), encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        rag.load_frozen_inputs(qa_path=qa, chunk_path=chunks, retrieval_results_path=results)


def test_unified_per_question_schema_is_complete() -> None:
    record = rag._unified_record("extractive_v1", _question(), _context(), "申请需要符合条件。")
    assert rag.REQUIRED_PER_QUESTION_FIELDS <= set(record)
    assert record["generation_family"] == "extractive"
    assert record["provider"] is None


def test_extractive_recorder_metadata_and_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = {name: tmp_path / f"{name}.txt" for name in ("raw", "cleaned", "chunks", "qa", "retrieval")}
    for path in inputs.values():
        path.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(rag, "RAW_PATH", inputs["raw"])
    monkeypatch.setattr(rag, "CLEANED_PATH", inputs["cleaned"])
    monkeypatch.setattr(rag, "CHUNK_PATH", inputs["chunks"])
    monkeypatch.setattr(rag, "QA_PATH", inputs["qa"])
    monkeypatch.setattr(rag, "FINAL_RETRIEVAL_RESULTS", inputs["retrieval"])
    recorder = rag._recorder("extractive_v2", experiment_id="rag-test", experiment_root=tmp_path / "outputs")
    with recorder:
        recorder.record_results({"question_count": 1}, [{"question_id": "Q001"}])
    config = json.loads((recorder.run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["parameters"]["generation_version"] == "extractive_v2"
    assert config["parameters"]["external_api"] is False
    rows = rebuild_registry(tmp_path / "outputs", include_legacy=False)
    assert rows[0]["generation_family"] == "extractive"
    assert rows[0]["generation_version"] == "extractive_v2"
    assert rows[0]["external_api"] == "false"


def test_thesis_export_does_not_classify_rag_as_retrieval(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    config = tmp_path / "config.json"
    metrics.write_text(json.dumps({"metrics": {"question_count": 40}}), encoding="utf-8")
    config.write_text(json.dumps({"parameters": {"generation_version": "extractive_v1"}}), encoding="utf-8")
    row = {"experiment_id": "rag", "task_type": "rag_answer_quality", "metrics_path": str(metrics), "metric_selector": "metrics", "config_path": str(config), "legacy": "false"}
    tables, _missing = build_thesis_tables([row])
    assert not tables["retrieval_method_comparison"]
    assert tables["rag_version_comparison"][0]["rag_version"] == "extractive_v1"


def test_llm_dry_run_never_constructs_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "validate_frozen_retrieval", lambda: {})
    questions = [_question(i) for i in range(1, 41)]
    chunks = {f"chunk-{i}": {} for i in range(1, 41)}
    retrieval = {f"Q{i:03d}": {} for i in range(1, 41)}
    monkeypatch.setattr(rag, "load_frozen_inputs", lambda: (questions, chunks, retrieval))
    monkeypatch.setattr(rag, "build_frozen_context", lambda *_args: _context())
    monkeypatch.setattr(rag, "create_llm_provider", lambda *_args, **_kwargs: pytest.fail("provider constructed"))
    report = rag.dry_run_llm()
    assert report["actual_api_requests"] == 0
    assert report["external_api_called"] is False
    assert report["prompt_build_count"] == 40


def test_duplicate_protection_counts_only_incomplete_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "load_frozen_inputs", lambda: ([_question(i) for i in range(1, 41)], {}, {}))
    run_dir = tmp_path / "exp"
    run_dir.mkdir()
    (run_dir / "completed.jsonl").write_text(json.dumps({"question_id": "Q001", "generated_answer": "ok", "error": None}) + "\n", encoding="utf-8")
    plan = rag.llm_batch_plan("exp", staging_root=tmp_path)
    assert plan["already_completed"] == 1
    assert plan["requests_to_send"] == 39


def test_sanitized_failure_and_records_do_not_contain_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "top-secret-token"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    sanitized = rag._sanitize_error(RuntimeError(f"Bearer {secret}"))
    assert secret not in json.dumps(sanitized)
    recorder = rag._recorder("llm_grounded_v1", experiment_id="safe")
    assert secret not in json.dumps(recorder.config)
    assert recorder.config["parameters"]["retry"] == 0
