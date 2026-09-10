"""Run one normalized real-embedding + BM25 hybrid retrieval experiment."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_normalized_bm25_smoke import (
    CHUNK_ROOT,
    DATA_SOURCE,
    QA_JSONL,
    RAW_SOURCE,
)
from scripts.run_normalized_real_dense_smoke import (
    DEFAULT_MODEL_PATH,
    EXPECTED_EMBEDDING_DIMENSION,
    MODEL_ID,
    PROTOCOL_SNAPSHOT_FINGERPRINT,
    REAL_DENSE_LEGACY_ROOT,
    enforce_offline_environment,
    resolve_model_snapshot,
)
from src.experiments.manifests import path_manifest
from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT, ExperimentRecorder
from src.experiments.registry import rebuild_registry
from src.retrieval.bm25 import build_index, search
from src.retrieval.dense_index import load_jsonl
from src.retrieval.embedding_providers import SentenceTransformersProvider
from src.retrieval.evaluation import evaluate_strategy_at_k, relevant_ids_for_strategy
from src.retrieval.real_dense import search_with_provider, validate_chunk_vector_alignment
from src.retrieval.real_hybrid import fuse_real_rrf, fuse_real_weighted_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL_K = 50
DEFAULT_RANDOM_SEED = 42
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_RRF_K = 60


def _ensure_file(path: Path, role: str) -> None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{role} missing or empty: {path}")


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


def run_normalized_real_hybrid(
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    data_source: Path = DATA_SOURCE,
    raw_source: Path | None = RAW_SOURCE,
    chunk_file: Path | None = None,
    vector_file: Path | None = None,
    qa_file: Path = QA_JSONL,
    chunk_strategy: str = "fixed_length",
    fusion_method: str = "weighted",
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_k: int = DEFAULT_CANDIDATE_POOL_K,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_k: int = DEFAULT_RRF_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    random_seed: int = DEFAULT_RANDOM_SEED,
    model_path: Path = DEFAULT_MODEL_PATH,
    offline: bool = True,
    repo_root: Path = PROJECT_ROOT,
    provider_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    if fusion_method not in {"weighted", "rrf"}:
        raise ValueError("fusion_method must be weighted or rrf")
    if top_k < 1 or candidate_pool_k < top_k:
        raise ValueError("candidate_pool_k must be greater than or equal to top_k")
    if fusion_method == "weighted" and abs(dense_weight + bm25_weight - 1.0) > 1e-12:
        raise ValueError("dense_weight and bm25_weight must sum to 1.0")

    chunk_path = Path(chunk_file) if chunk_file else CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{chunk_strategy}.jsonl"
    vector_path = Path(vector_file) if vector_file else REAL_DENSE_LEGACY_ROOT / "embeddings" / f"{chunk_strategy}_real_embedding_embeddings.jsonl"
    resolved_model_path = Path(model_path).expanduser().resolve()
    snapshot_manifest = path_manifest(resolved_model_path)
    required_inputs = {
        "data_source": Path(data_source),
        "chunk_file": chunk_path,
        "qa_file": Path(qa_file),
        "vector_file": vector_path,
        "model_snapshot": resolved_model_path,
    }
    if raw_source is not None:
        required_inputs["raw_source"] = Path(raw_source)

    retrieval_method = f"real_hybrid_{fusion_method}"
    hybrid_parameters = (
        {
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "normalization": "per_query_min_max",
            "equal_score_value": 0.0,
        }
        if fusion_method == "weighted"
        else {"rrf_k": rrf_k}
    )
    recorder = ExperimentRecorder(
        task_type="retrieval",
        retrieval_method=retrieval_method,
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        embedding_model=f"{MODEL_ID}+bm25",
        qa_version="v1",
        hybrid_parameters=hybrid_parameters,
        random_seed=random_seed,
        inputs=required_inputs,
        parameters={
            "candidate_pool_k": candidate_pool_k,
            "dense_candidate_pool_k": candidate_pool_k,
            "bm25_candidate_pool_k": candidate_pool_k,
            "fusion_candidate_pool_k": candidate_pool_k,
            "metric_primary": f"Recall@{top_k}",
            "metric_secondary": f"MRR@{top_k}",
            "mrr_cutoff": top_k,
            "model_id": MODEL_ID,
            "resolved_snapshot_path": str(resolved_model_path),
            "snapshot_fingerprint": snapshot_manifest["sha256"],
            "local_files_only": True,
            "offline": offline,
            "bm25_tokenizer": "cjk_char_bigram",
            "bm25_k1": k1,
            "bm25_b": b,
            "tie_break": "hybrid_desc_bm25_desc_dense_desc_chunk_id_asc",
        },
        manifest_metadata={
            "model": {
                "model_id": MODEL_ID,
                "resolved_snapshot_path": str(resolved_model_path),
                "snapshot_revision": resolved_model_path.name,
                "snapshot_fingerprint": snapshot_manifest["sha256"],
                "protocol_snapshot_fingerprint": PROTOCOL_SNAPSHOT_FINGERPRINT,
                "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
                "local_files_only": True,
            },
            "retrieval": {"top_k": top_k, "candidate_pool_k": candidate_pool_k},
        },
        experiment_root=Path(experiment_root),
        repo_root=Path(repo_root),
        generation_script="scripts/run_normalized_real_hybrid.py",
    )

    metrics: dict[str, Any] = {}
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
            vectors = load_jsonl(vector_path)
            questions = load_jsonl(Path(qa_file))
            validate_chunk_vector_alignment(chunks, vectors, expected_dimension=EXPECTED_EMBEDDING_DIMENSION)
            actual_pool_k = min(candidate_pool_k, len(chunks))
            provider = (
                SentenceTransformersProvider(str(validated_model_path), device="cpu", local_files_only=True)
                if provider_factory is None
                else provider_factory(validated_model_path)
            )
            bm25_index = build_index(chunks, k1=k1, b=b)
            chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
            annotated_final_rows: list[dict[str, Any]] = []
            per_question_rows: list[dict[str, Any]] = []
            for question in questions:
                dense_rows = search_with_provider(question["question"], vectors, provider, top_k=actual_pool_k)
                bm25_rows = search(question["question"], bm25_index, top_k=actual_pool_k)
                if fusion_method == "weighted":
                    fused = fuse_real_weighted_score(
                        question, chunk_strategy, dense_rows, bm25_rows,
                        real_dense_weight=dense_weight, bm25_weight=bm25_weight,
                        top_k=actual_pool_k, equal_score_value=0.0,
                    )
                else:
                    fused = fuse_real_rrf(
                        question, chunk_strategy, dense_rows, bm25_rows,
                        k=rrf_k, top_k=actual_pool_k,
                    )
                final_rows = fused[:top_k]
                annotated_final_rows.extend({**row, "score": row["hybrid_score"]} for row in final_rows)
                source_policy_title = str(question.get("source_policy_title", ""))
                candidate_results: list[dict[str, Any]] = []
                for row in fused:
                    chunk = chunks_by_id.get(str(row["retrieved_chunk_id"]), {})
                    policy_title = _policy_title(chunk)
                    candidate_results.append(
                        {
                            "rank": int(row["rank"]),
                            "retrieved_chunk_id": row["retrieved_chunk_id"],
                            "retrieved_document_title": row["retrieved_document_title"],
                            "retrieved_policy_title": policy_title,
                            "real_dense_score": float(row["real_dense_score"]),
                            "bm25_score": float(row["bm25_score"]),
                            "hybrid_score": float(row["hybrid_score"]),
                            "fusion_method": row["fusion_method"],
                            "is_relevant": bool(row["is_relevant"]),
                            "matched_by": row["matched_by"],
                            "source_policy_match": bool(source_policy_title and policy_title == source_policy_title),
                        }
                    )
                final_results = candidate_results[:top_k]
                first_rank = _first_relevant_rank(final_results)
                per_question_rows.append(
                    {
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "expected_answer": question.get("expected_answer", ""),
                        "evidence_text": question.get("evidence_text", ""),
                        "question_type": question.get("question_type", "unknown"),
                        "difficulty": question.get("difficulty", "unknown"),
                        "source_policy_title": source_policy_title,
                        "relevant_chunk_ids": sorted(relevant_ids_for_strategy(question, chunk_strategy)),
                        "chunk_strategy": chunk_strategy,
                        "retrieval_method": retrieval_method,
                        "top_k": top_k,
                        "candidate_pool_k_requested": candidate_pool_k,
                        "candidate_pool_size": len(candidate_results),
                        "retrieved_results": final_results,
                        "candidate_pool_results": candidate_results,
                        "first_relevant_rank": first_rank,
                        f"hit_at_{top_k}": first_rank is not None,
                        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                        "source_policy_hit": any(row["source_policy_match"] for row in final_results),
                        "related_chunk_hit": first_rank is not None,
                    }
                )
            metrics = evaluate_strategy_at_k(chunk_strategy, annotated_final_rows, questions, top_k)
            question_count = len(questions)
            metrics.update(
                {
                    "retrieval_method": retrieval_method,
                    "fusion_method": fusion_method,
                    "model_id": MODEL_ID,
                    "random_seed": random_seed,
                    "candidate_pool_k": candidate_pool_k,
                    "actual_candidate_pool_k": actual_pool_k,
                    "per_question_result_count": len(per_question_rows),
                    "source_policy_hit_rate": sum(row["source_policy_hit"] for row in per_question_rows) / question_count if question_count else 0.0,
                    "related_chunk_hit_rate": sum(row["related_chunk_hit"] for row in per_question_rows) / question_count if question_count else 0.0,
                    "runtime_seconds": perf_counter() - started,
                    **hybrid_parameters,
                }
            )
            recorder.record_results(metrics, per_question_rows)
    finally:
        registry_rows = rebuild_registry(Path(experiment_root))
    registry_row = next(row for row in registry_rows if row["experiment_id"] == recorder.experiment_id)
    return {"experiment_id": recorder.experiment_id, "run_dir": str(recorder.run_dir), "metrics": metrics, "registry_row": registry_row}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--chunk-strategy", default="fixed_length")
    parser.add_argument("--fusion-method", choices=("weighted", "rrf"), default="weighted")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-pool-k", type=int, default=DEFAULT_CANDIDATE_POOL_K)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    result = run_normalized_real_hybrid(
        experiment_root=args.experiment_root,
        chunk_strategy=args.chunk_strategy,
        fusion_method=args.fusion_method,
        top_k=args.top_k,
        candidate_pool_k=args.candidate_pool_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
        random_seed=args.random_seed,
        model_path=args.model_path,
        offline=args.offline,
    )
    print(f"experiment_id: {result['experiment_id']}")
    print(f"run_dir: {result['run_dir']}")
    print(f"Recall@5: {result['metrics'].get('Recall@5')}")
    print(f"MRR@5: {result['metrics'].get('MRR@5')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

