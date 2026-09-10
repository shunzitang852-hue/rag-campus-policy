from __future__ import annotations

import json
from pathlib import Path

from scripts.run_normalized_real_dense_smoke import EXPECTED_EMBEDDING_DIMENSION
from scripts.run_normalized_real_hybrid import run_normalized_real_hybrid


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class _FakeProvider:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vector = [0.0] * EXPECTED_EMBEDDING_DIMENSION
        vector[0] = 1.0
        return [list(vector) for _ in texts]


def _inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    source = root / "cleaned.txt"
    source.write_text("奖学金条件", encoding="utf-8")
    model = root / "model" / "snapshots" / "revision"
    for relative in ["config.json", "modules.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json"]:
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    chunks = root / "chunks.jsonl"
    vectors = root / "vectors.jsonl"
    chunk_rows = []
    vector_rows = []
    for index in range(8):
        chunk_id = f"c{index}"
        chunk_rows.append({"chunk_id": chunk_id, "document_id": "d1", "document_title": "学生手册", "policy_title": "奖学金办法", "chunk_strategy": "fixed_length", "chunk_text": f"奖学金条件 {index}"})
        embedding = [0.0] * EXPECTED_EMBEDDING_DIMENSION
        embedding[0] = 1.0 - index / 100.0
        vector_rows.append({"chunk_id": chunk_id, "document_title": "学生手册", "chunk_text": f"奖学金条件 {index}", "embedding": embedding})
    _write_jsonl(chunks, chunk_rows)
    _write_jsonl(vectors, vector_rows)
    qa = root / "qa.jsonl"
    _write_jsonl(qa, [{"question_id": "q1", "question": "奖学金条件", "question_type": "condition", "difficulty": "easy", "source_policy_title": "奖学金办法", "related_chunk_ids_fixed_length": ["c0"]}])
    return source, chunks, vectors, qa, model


def test_normalized_weighted_hybrid_records_top_k_and_candidate_pool(tmp_path: Path) -> None:
    source, chunks, vectors, qa, model = _inputs(tmp_path)
    result = run_normalized_real_hybrid(
        experiment_root=tmp_path / "outputs",
        data_source=source,
        raw_source=None,
        chunk_file=chunks,
        vector_file=vectors,
        qa_file=qa,
        chunk_strategy="fixed_length",
        fusion_method="weighted",
        top_k=3,
        candidate_pool_k=6,
        model_path=model,
        offline=True,
        repo_root=Path(__file__).resolve().parents[1],
        provider_factory=lambda path: _FakeProvider(),
    )
    row = json.loads((Path(result["run_dir"]) / "per_question_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(row["retrieved_results"]) == 3
    assert len(row["candidate_pool_results"]) == 6
    assert result["registry_row"]["retrieval_method"] == "real_hybrid_weighted"
    assert result["registry_row"]["candidate_pool_k"] == "6"
