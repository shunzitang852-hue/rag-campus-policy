"""Run one normalized real-embedding dense retrieval smoke experiment."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.manifests import path_manifest
from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT, ExperimentRecorder
from src.experiments.registry import rebuild_registry
from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH
from src.retrieval.dense_index import load_jsonl
from src.retrieval.embedding_providers import SentenceTransformersProvider
from src.retrieval.evaluation import (
    annotate_results,
    evaluate_strategy_at_k,
    relevant_ids_for_strategy,
)
from src.retrieval.real_dense import (
    search_with_provider,
    validate_chunk_vector_alignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
DATA_SOURCE = (
    DATASETS_ROOT
    / "interim"
    / "hdu_ies"
    / "student_handbook"
    / "cleaned"
    / "hdu_ies_student_handbook_cleaned.txt"
)
RAW_SOURCE = DEFAULT_RAW_PATH
CHUNK_ROOT = (
    DATASETS_ROOT
    / "processed"
    / "hdu_ies"
    / "student_handbook"
    / "chunks"
)
QA_JSONL = (
    DATASETS_ROOT
    / "processed"
    / "hdu_ies"
    / "student_handbook"
    / "qa_testset"
    / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
)
REAL_DENSE_LEGACY_ROOT = (
    PROJECT_ROOT.parent
    / "experiment_outputs"
    / "hdu_ies"
    / "student_handbook"
    / "dense_retrieval_real_embedding"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "models"
    / "models--BAAI--bge-small-zh-v1.5"
    / "snapshots"
    / "7999e1d3359715c523056ef9478215996d62a620"
)
FORBIDDEN_OLD_MODEL_PATH = (
    PROJECT_ROOT.parent / "models" / "BAAI" / "bge-small-zh-v1.5"
)
MODEL_ID = "BAAI/bge-small-zh-v1.5"
EXPECTED_EMBEDDING_DIMENSION = 512
DEFAULT_STRATEGY = "fixed_length"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL_K = 50
DEFAULT_RANDOM_SEED = 42
REQUIRED_MODEL_FILES = {
    "config.json",
    "modules.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
}
PROTOCOL_SNAPSHOT_FINGERPRINT = (
    "D0A8E83A6221E4B4F6347279D8DA7B6AB46DE0389DB20B4C376FB4E8D27D5314"
)


def _ensure_file(path: Path, role: str) -> None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{role} missing or empty: {path}")


def resolve_model_snapshot(model_path: Path) -> Path:
    """Resolve and validate the explicitly supplied local model snapshot."""

    resolved = Path(model_path).expanduser().resolve()
    if os.path.normcase(str(resolved)) == os.path.normcase(
        str(FORBIDDEN_OLD_MODEL_PATH.resolve())
    ):
        raise ValueError(f"forbidden obsolete model path: {resolved}")
    if not resolved.exists() or not resolved.is_dir():
        raise RuntimeError(f"local model snapshot missing: {resolved}")
    missing = sorted(
        relative
        for relative in REQUIRED_MODEL_FILES
        if not (resolved / Path(relative)).is_file()
    )
    if missing:
        raise RuntimeError(
            "local model snapshot is incomplete; missing: " + ", ".join(missing)
        )
    return resolved


def enforce_offline_environment(*, offline: bool) -> dict[str, str]:
    """Force Hugging Face libraries into offline-only mode."""

    if not offline:
        raise ValueError("normalized real_dense experiments require --offline")
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    os.environ.update(values)
    return values


def _first_relevant_rank(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row["is_relevant"]:
            return int(row["rank"])
    return None


def _policy_title(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    return str(
        chunk.get("policy_title")
        or (metadata.get("policy_title") if isinstance(metadata, dict) else "")
        or ""
    )


def run_normalized_real_dense_smoke(
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    data_source: Path = DATA_SOURCE,
    raw_source: Path | None = RAW_SOURCE,
    chunk_file: Path | None = None,
    vector_file: Path | None = None,
    qa_file: Path = QA_JSONL,
    chunk_strategy: str = DEFAULT_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_k: int = DEFAULT_CANDIDATE_POOL_K,
    random_seed: int = DEFAULT_RANDOM_SEED,
    model_path: Path = DEFAULT_MODEL_PATH,
    offline: bool = True,
    repo_root: Path = PROJECT_ROOT,
    provider_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Reuse precomputed vectors and the existing dense search implementation."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if candidate_pool_k < top_k:
        raise ValueError(
            "candidate_pool_k must be greater than or equal to top_k"
        )
    chunk_path = Path(chunk_file) if chunk_file else (
        CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{chunk_strategy}.jsonl"
    )
    vector_path = Path(vector_file) if vector_file else (
        REAL_DENSE_LEGACY_ROOT
        / "embeddings"
        / f"{chunk_strategy}_real_embedding_embeddings.jsonl"
    )
    resolved_model_path = Path(model_path).expanduser().resolve()
    snapshot_manifest = path_manifest(resolved_model_path)
    offline_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    required_inputs = {
        "data_source": Path(data_source),
        "chunk_file": chunk_path,
        "qa_file": Path(qa_file),
        "vector_file": vector_path,
        "model_snapshot": resolved_model_path,
    }
    if raw_source is not None:
        required_inputs["raw_source"] = Path(raw_source)

    recorder = ExperimentRecorder(
        task_type="retrieval",
        retrieval_method="real_dense",
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        embedding_model=MODEL_ID,
        qa_version="v1",
        random_seed=random_seed,
        inputs=required_inputs,
        parameters={
            "candidate_pool_k": candidate_pool_k,
            "metric_primary": f"Recall@{top_k}",
            "metric_secondary": f"MRR@{top_k}",
            "mrr_cutoff": top_k,
            "model_id": MODEL_ID,
            "resolved_snapshot_path": str(resolved_model_path),
            "snapshot_revision": resolved_model_path.name,
            "snapshot_fingerprint": snapshot_manifest["sha256"],
            "protocol_snapshot_fingerprint": PROTOCOL_SNAPSHOT_FINGERPRINT,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "local_files_only": True,
            "offline": offline,
            "dense_score": "cosine_similarity",
            "dense_tie_break": "score_desc_then_vector_file_order",
            "chunk_vector_binding": "same_index_and_chunk_id",
        },
        manifest_metadata={
            "model": {
                "model_id": MODEL_ID,
                "resolved_snapshot_path": str(resolved_model_path),
                "snapshot_revision": resolved_model_path.name,
                "snapshot_fingerprint": snapshot_manifest["sha256"],
                "protocol_snapshot_fingerprint": PROTOCOL_SNAPSHOT_FINGERPRINT,
                "snapshot_file_count": snapshot_manifest.get("file_count"),
                "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
                "local_files_only": True,
                "offline_environment": offline_environment,
            },
            "retrieval": {
                "top_k": top_k,
                "candidate_pool_k": candidate_pool_k,
                "chunk_vector_binding": "same_index_and_chunk_id",
            },
        },
        experiment_root=Path(experiment_root),
        repo_root=Path(repo_root),
        generation_script="scripts/run_normalized_real_dense_smoke.py",
    )

    registry_rows: list[dict[str, str]] = []
    try:
        with recorder:
            started = perf_counter()
            random.seed(random_seed)
            enforce_offline_environment(offline=offline)
            for role, path in required_inputs.items():
                if role != "model_snapshot":
                    _ensure_file(path, role)
            validated_model_path = resolve_model_snapshot(resolved_model_path)

            chunks = load_jsonl(chunk_path)
            embedding_rows = load_jsonl(vector_path)
            questions = load_jsonl(Path(qa_file))
            alignment = validate_chunk_vector_alignment(
                chunks,
                embedding_rows,
                expected_dimension=EXPECTED_EMBEDDING_DIMENSION,
            )
            actual_candidate_pool_k = min(candidate_pool_k, len(embedding_rows))
            if provider_factory is None:
                provider = SentenceTransformersProvider(
                    str(validated_model_path),
                    device="cpu",
                    local_files_only=True,
                )
            else:
                provider = provider_factory(validated_model_path)

            chunks_by_id = {
                str(chunk["chunk_id"]): chunk
                for chunk in chunks
            }
            annotated_final_rows: list[dict[str, Any]] = []
            per_question_rows: list[dict[str, Any]] = []
            for question in questions:
                candidate_rows = search_with_provider(
                    question["question"],
                    embedding_rows,
                    provider,
                    top_k=actual_candidate_pool_k,
                )
                annotated_candidates = annotate_results(
                    question,
                    chunk_strategy,
                    candidate_rows,
                )
                annotated_final = annotated_candidates[:top_k]
                annotated_final_rows.extend(annotated_final)
                source_policy_title = str(
                    question.get("source_policy_title", "")
                )
                candidate_results: list[dict[str, Any]] = []
                for row in annotated_candidates:
                    chunk = chunks_by_id.get(
                        str(row["retrieved_chunk_id"]),
                        {},
                    )
                    retrieved_policy_title = _policy_title(chunk)
                    candidate_results.append(
                        {
                            "rank": int(row["rank"]),
                            "chunk_id": row["retrieved_chunk_id"],
                            "retrieved_chunk_id": row[
                                "retrieved_chunk_id"
                            ],
                            "retrieved_document_title": row[
                                "retrieved_document_title"
                            ],
                            "retrieved_policy_title": retrieved_policy_title,
                            "raw_dense_score": float(row["score"]),
                            "is_relevant": bool(row["is_relevant"]),
                            "matched_by": row["matched_by"],
                            "source_policy_match": bool(
                                source_policy_title
                                and retrieved_policy_title
                                == source_policy_title
                            ),
                        }
                    )
                final_results = candidate_results[:top_k]
                first_rank = _first_relevant_rank(annotated_final)
                per_question_rows.append(
                    {
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "expected_answer": question.get(
                            "expected_answer",
                            "",
                        ),
                        "evidence_text": question.get("evidence_text", ""),
                        "question_type": question.get(
                            "question_type",
                            "unknown",
                        ),
                        "difficulty": question.get(
                            "difficulty",
                            "unknown",
                        ),
                        "source_policy_title": source_policy_title,
                        "relevant_chunk_ids": sorted(
                            relevant_ids_for_strategy(
                                question,
                                chunk_strategy,
                            )
                        ),
                        "chunk_strategy": chunk_strategy,
                        "retrieval_method": "real_dense",
                        "top_k": top_k,
                        "candidate_pool_k_requested": candidate_pool_k,
                        "candidate_pool_size": len(candidate_results),
                        "retrieved_results": final_results,
                        "candidate_pool_results": candidate_results,
                        "first_relevant_rank": first_rank,
                        f"hit_at_{top_k}": first_rank is not None,
                        "reciprocal_rank": (
                            1.0 / first_rank if first_rank else 0.0
                        ),
                        "source_policy_hit": any(
                            row["source_policy_match"]
                            for row in final_results
                        ),
                        "related_chunk_hit": first_rank is not None,
                    }
                )

            metrics = evaluate_strategy_at_k(
                chunk_strategy,
                annotated_final_rows,
                questions,
                top_k,
            )
            question_count = len(questions)
            metrics.update(
                {
                    "retrieval_method": "real_dense",
                    "model_id": MODEL_ID,
                    "resolved_snapshot_path": str(
                        validated_model_path
                    ),
                    "snapshot_revision": validated_model_path.name,
                    "snapshot_fingerprint": snapshot_manifest["sha256"],
                    "embedding_dimension": (
                        EXPECTED_EMBEDDING_DIMENSION
                    ),
                    "local_files_only": True,
                    "offline": True,
                    "random_seed": random_seed,
                    "chunk_count": len(chunks),
                    "vector_count": len(embedding_rows),
                    "candidate_pool_k": candidate_pool_k,
                    "actual_candidate_pool_k": (
                        actual_candidate_pool_k
                    ),
                    "per_question_result_count": len(
                        per_question_rows
                    ),
                    "chunk_vector_alignment": alignment,
                    "source_policy_hit_rate": (
                        sum(
                            1
                            for row in per_question_rows
                            if row["source_policy_hit"]
                        )
                        / question_count
                        if question_count
                        else 0.0
                    ),
                    "related_chunk_hit_rate": (
                        sum(
                            1
                            for row in per_question_rows
                            if row["related_chunk_hit"]
                        )
                        / question_count
                        if question_count
                        else 0.0
                    ),
                    "runtime_seconds": perf_counter() - started,
                }
            )
            recorder.record_results(
                metrics,
                per_question_rows,
                summary_rows=[
                    {
                        "experiment_id": recorder.experiment_id,
                        "status": "completed",
                        "retrieval_method": "real_dense",
                        "chunk_strategy": chunk_strategy,
                        "top_k": top_k,
                        "candidate_pool_k": candidate_pool_k,
                        "question_count": question_count,
                        f"Recall@{top_k}": metrics[
                            f"Recall@{top_k}"
                        ],
                        f"MRR@{top_k}": metrics[f"MRR@{top_k}"],
                        "source_policy_hit_rate": metrics[
                            "source_policy_hit_rate"
                        ],
                        "related_chunk_hit_rate": metrics[
                            "related_chunk_hit_rate"
                        ],
                        "runtime_seconds": metrics[
                            "runtime_seconds"
                        ],
                    }
                ],
            )
    finally:
        registry_rows = rebuild_registry(Path(experiment_root))

    registry_row = next(
        row
        for row in registry_rows
        if row["experiment_id"] == recorder.experiment_id
    )
    return {
        "experiment_id": recorder.experiment_id,
        "run_dir": str(recorder.run_dir),
        "metrics": metrics,
        "registry_row": registry_row,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--chunk-strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_K,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    result = run_normalized_real_dense_smoke(
        experiment_root=args.experiment_root,
        chunk_strategy=args.chunk_strategy,
        top_k=args.top_k,
        candidate_pool_k=args.candidate_pool_k,
        random_seed=args.random_seed,
        model_path=args.model_path,
        offline=args.offline,
    )
    metrics = result["metrics"]
    print(f"experiment_id: {result['experiment_id']}")
    print(f"run_dir: {result['run_dir']}")
    print(f"question_count: {metrics['question_count']}")
    print(
        "per_question_result_count: "
        f"{metrics['per_question_result_count']}"
    )
    print(f"Recall@5: {metrics.get('Recall@5')}")
    print(f"MRR@5: {metrics.get('MRR@5')}")
    print(
        "source_policy_hit_rate: "
        f"{metrics['source_policy_hit_rate']}"
    )
    print(
        "related_chunk_hit_rate: "
        f"{metrics['related_chunk_hit_rate']}"
    )
    print(f"registry_legacy: {result['registry_row']['legacy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
