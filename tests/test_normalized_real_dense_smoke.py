from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from scripts.run_normalized_real_dense_smoke import (
    EXPECTED_EMBEDDING_DIMENSION,
    FORBIDDEN_OLD_MODEL_PATH,
    enforce_offline_environment,
    resolve_model_snapshot,
    run_normalized_real_dense_smoke,
)
from src.retrieval.embedding_providers import SentenceTransformersProvider
from src.retrieval.real_dense import (
    search_with_provider,
    validate_chunk_vector_alignment,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _make_model_snapshot(root: Path) -> Path:
    model_path = root / "models--BAAI--bge-small-zh-v1.5" / "snapshots" / "revision"
    required = [
        "config.json",
        "modules.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "1_Pooling/config.json",
    ]
    for relative in required:
        path = model_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    return model_path


class _FakeProvider:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for _ in texts:
            vector = [0.0] * EXPECTED_EMBEDDING_DIMENSION
            vector[0] = 1.0
            vectors.append(vector)
        return vectors


def _make_run_inputs(
    root: Path,
    *,
    count: int = 60,
) -> tuple[Path, Path, Path, Path, Path]:
    source = root / "cleaned.txt"
    source.write_text("规范化 real dense 测试数据。", encoding="utf-8")
    chunks_path = root / "chunks.jsonl"
    vectors_path = root / "vectors.jsonl"
    qa_path = root / "qa.jsonl"
    model_path = _make_model_snapshot(root / "model")

    chunks: list[dict] = []
    vectors: list[dict] = []
    for index in range(count):
        chunk_id = f"c{index:03d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": "d1",
                "document_title": "学生手册",
                "policy_title": "奖学金办法",
                "chunk_text": f"第 {index} 条规定",
                "chunk_strategy": "fixed_length",
            }
        )
        embedding = [0.0] * EXPECTED_EMBEDDING_DIMENSION
        embedding[0] = 1.0 - index / 1000.0
        vectors.append(
            {
                "chunk_id": chunk_id,
                "strategy": "fixed_length",
                "document_id": "d1",
                "document_title": "学生手册",
                "chunk_text": f"第 {index} 条规定",
                "embedding": embedding,
                "metadata": {"policy_title": "奖学金办法"},
            }
        )
    _write_jsonl(chunks_path, chunks)
    _write_jsonl(vectors_path, vectors)
    _write_jsonl(
        qa_path,
        [
            {
                "question_id": "q1",
                "question": "奖学金规定是什么？",
                "expected_answer": "第零条",
                "evidence_text": "第 0 条规定",
                "question_type": "fact_lookup",
                "difficulty": "easy",
                "source_policy_title": "奖学金办法",
                "related_chunk_ids_fixed_length": ["c000"],
            }
        ],
    )
    return source, chunks_path, vectors_path, qa_path, model_path


def test_offline_model_path_resolution_requires_complete_snapshot(
    tmp_path: Path,
) -> None:
    model_path = _make_model_snapshot(tmp_path)

    assert resolve_model_snapshot(model_path) == model_path.resolve()

    (model_path / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="snapshot is incomplete"):
        resolve_model_snapshot(model_path)


def test_obsolete_model_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden obsolete model path"):
        resolve_model_snapshot(FORBIDDEN_OLD_MODEL_PATH)


def test_offline_environment_is_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="require --offline"):
        enforce_offline_environment(offline=False)

    values = enforce_offline_environment(offline=True)
    assert values["HF_HUB_OFFLINE"] == "1"
    assert values["TRANSFORMERS_OFFLINE"] == "1"


def test_sentence_transformer_receives_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured["model_name"] = model_name
            captured.update(kwargs)

        def get_embedding_dimension(self) -> int:
            return EXPECTED_EMBEDDING_DIMENSION

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(
        "src.retrieval.embedding_providers.importlib.util.find_spec",
        lambda name: object(),
    )

    provider = SentenceTransformersProvider(
        "local-snapshot",
        local_files_only=True,
    )

    assert captured["model_name"] == "local-snapshot"
    assert captured["local_files_only"] is True
    assert provider.metadata()["local_files_only"] is True


def test_chunk_vector_count_mismatch_fails() -> None:
    chunks = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    vectors = [
        {
            "chunk_id": "c1",
            "embedding": [0.0] * EXPECTED_EMBEDDING_DIMENSION,
        }
    ]

    with pytest.raises(ValueError, match="count mismatch"):
        validate_chunk_vector_alignment(
            chunks,
            vectors,
            expected_dimension=EXPECTED_EMBEDDING_DIMENSION,
        )


def test_embedding_dimension_mismatch_fails() -> None:
    chunks = [{"chunk_id": "c1"}]
    vectors = [{"chunk_id": "c1", "embedding": [0.0] * 4}]

    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_chunk_vector_alignment(
            chunks,
            vectors,
            expected_dimension=EXPECTED_EMBEDDING_DIMENSION,
        )


def test_candidate_pool_cannot_be_smaller_than_top_k(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        run_normalized_real_dense_smoke(
            experiment_root=tmp_path / "outputs",
            top_k=5,
            candidate_pool_k=4,
            model_path=tmp_path / "model",
        )


def test_equal_scores_preserve_vector_file_order() -> None:
    provider = _FakeProvider()
    embedding = [0.0] * EXPECTED_EMBEDDING_DIMENSION
    rows = [
        {
            "chunk_id": "c2",
            "document_title": "doc",
            "chunk_text": "two",
            "embedding": embedding,
        },
        {
            "chunk_id": "c1",
            "document_title": "doc",
            "chunk_text": "one",
            "embedding": embedding,
        },
    ]

    first = search_with_provider("q", rows, provider, top_k=2)
    second = search_with_provider("q", rows, provider, top_k=2)

    assert [row["retrieved_chunk_id"] for row in first] == ["c2", "c1"]
    assert first == second


def test_normalized_run_records_top_five_and_top_fifty(
    tmp_path: Path,
) -> None:
    source, chunks, vectors, qa, model = _make_run_inputs(tmp_path)
    output = tmp_path / "outputs"

    result = run_normalized_real_dense_smoke(
        experiment_root=output,
        data_source=source,
        raw_source=None,
        chunk_file=chunks,
        vector_file=vectors,
        qa_file=qa,
        top_k=5,
        candidate_pool_k=50,
        model_path=model,
        offline=True,
        repo_root=Path(__file__).resolve().parents[1],
        provider_factory=lambda path: _FakeProvider(),
    )
    run_dir = Path(result["run_dir"])
    rows = [
        json.loads(line)
        for line in (run_dir / "per_question_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(rows) == 1
    assert len(rows[0]["retrieved_results"]) == 5
    assert len(rows[0]["candidate_pool_results"]) == 50
    assert rows[0]["retrieved_results"] == rows[0]["candidate_pool_results"][:5]
    assert [row["rank"] for row in rows[0]["candidate_pool_results"]] == list(
        range(1, 51)
    )
    assert all(
        "raw_dense_score" in row
        for row in rows[0]["candidate_pool_results"]
    )
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

    manifest = json.loads(
        (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    expected_vector_hash = hashlib.sha256(vectors.read_bytes()).hexdigest().upper()
    assert manifest["inputs"]["vector_file"]["sha256"] == expected_vector_hash
    assert manifest["inputs"]["model_snapshot"]["type"] == "directory"
    assert manifest["metadata"]["model"]["local_files_only"] is True
    assert (
        manifest["metadata"]["model"]["resolved_snapshot_path"]
        == str(model.resolve())
    )
    assert result["registry_row"]["candidate_pool_k"] == "50"
    assert result["registry_row"]["legacy"] == "false"


def test_normalized_run_never_overwrites_existing_run_directory(
    tmp_path: Path,
) -> None:
    source, chunks, vectors, qa, model = _make_run_inputs(tmp_path)
    output = tmp_path / "outputs"
    kwargs = {
        "experiment_root": output,
        "data_source": source,
        "raw_source": None,
        "chunk_file": chunks,
        "vector_file": vectors,
        "qa_file": qa,
        "top_k": 5,
        "candidate_pool_k": 50,
        "model_path": model,
        "offline": True,
        "repo_root": Path(__file__).resolve().parents[1],
        "provider_factory": lambda path: _FakeProvider(),
    }

    first = run_normalized_real_dense_smoke(**kwargs)
    first_metrics = (
        Path(first["run_dir"]) / "metrics.json"
    ).read_bytes()
    second = run_normalized_real_dense_smoke(**kwargs)

    assert first["experiment_id"] != second["experiment_id"]
    assert first["run_dir"] != second["run_dir"]
    assert (
        Path(first["run_dir"]) / "metrics.json"
    ).read_bytes() == first_metrics


def test_missing_model_is_recorded_as_failed(
    tmp_path: Path,
) -> None:
    source, chunks, vectors, qa, _ = _make_run_inputs(tmp_path)
    output = tmp_path / "outputs"

    with pytest.raises(RuntimeError, match="snapshot missing"):
        run_normalized_real_dense_smoke(
            experiment_root=output,
            data_source=source,
            raw_source=None,
            chunk_file=chunks,
            vector_file=vectors,
            qa_file=qa,
            model_path=tmp_path / "missing-model",
            offline=True,
            repo_root=Path(__file__).resolve().parents[1],
            provider_factory=lambda path: _FakeProvider(),
        )

    runs = list((output / "runs").iterdir())
    assert len(runs) == 1
    metrics = json.loads(
        (runs[0] / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["status"] == "failed"
    assert metrics["error"]["type"] == "RuntimeError"
