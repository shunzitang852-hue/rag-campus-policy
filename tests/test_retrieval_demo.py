from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from scripts.demo_retrieval_qa import main
from src.retrieval import demo
from src.retrieval.embedding_providers import ProviderUnavailable


QUERY = "学生申请解除处分需要满足什么条件？"


def test_demo_module_can_be_imported() -> None:
    assert demo.DEFAULT_METHOD == "real_dense"
    assert "bm25" in demo.METHODS


def test_can_read_all_chunk_strategies() -> None:
    for strategy in demo.STRATEGIES:
        rows = demo.load_chunks(strategy)
        assert rows
        assert all(row.get("chunk_id") for row in rows)


def test_bm25_demo_returns_chinese_query_results() -> None:
    results = demo.retrieve(QUERY, method="bm25", strategy="fixed_length", top_k=3)
    assert len(results) == 3
    assert all(result.retrieval_method == "bm25" for result in results)
    assert_result_fields(results[0])


def test_real_dense_demo_returns_chinese_query_results_or_skips() -> None:
    try:
        results = demo.retrieve(QUERY, method="real_dense", strategy="fixed_length", top_k=3)
    except ProviderUnavailable as exc:
        pytest.skip(f"real embedding provider unavailable: {exc}")
    assert len(results) == 3
    assert all(result.retrieval_method == "real_dense" for result in results)
    assert_result_fields(results[0])


def test_real_provider_uses_validated_local_snapshot_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model-snapshot"
    model_path.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        demo,
        "load_simple_yaml",
        lambda _: {"model_name": "remote/model", "local_model_path": str(model_path)},
    )
    monkeypatch.setattr(demo, "configure_model_cache_environment", lambda _: {})

    def fake_create_provider(config: dict[str, object]) -> object:
        captured.update(config)
        return object()

    monkeypatch.setattr(demo, "create_provider", fake_create_provider)

    demo._load_real_provider()

    assert captured["model_name"] == str(model_path)
    assert captured["local_files_only"] is True


def test_top_k_parameter_is_applied() -> None:
    results = demo.retrieve(QUERY, method="bm25", strategy="fixed_length", top_k=2)
    assert [result.rank for result in results] == [1, 2]
    assert len(results) == 2


@pytest.mark.parametrize("strategy", demo.STRATEGIES)
def test_strategy_parameter_is_applied(strategy: str) -> None:
    results = demo.retrieve(QUERY, method="bm25", strategy=strategy, top_k=1)
    assert len(results) == 1
    assert results[0].strategy == strategy


def test_empty_query_has_clear_error() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        demo.retrieve("", method="bm25", strategy="fixed_length", top_k=1)


def test_cli_without_query_enters_interactive_mode_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: (_ for _ in ()).throw(EOFError))
    assert main(["--method", "bm25"]) == 0


def assert_result_fields(result: demo.DemoResult) -> None:
    row = demo.result_to_dict(result)
    assert row["chunk_id"]
    assert row["document_title"]
    assert isinstance(row["score"], float)
    assert row["chunk_text"]
    assert len(row["chunk_text_summary"]) <= demo.SUMMARY_CHAR_LIMIT
