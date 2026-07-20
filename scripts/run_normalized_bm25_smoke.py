"""Run one normalized BM25 retrieval smoke experiment."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT, ExperimentRecorder
from src.experiments.registry import rebuild_registry
from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH
from src.retrieval.bm25 import build_index, search
from src.retrieval.dense_index import load_jsonl
from src.retrieval.evaluation import (
    annotate_results,
    evaluate_strategy_at_k,
    relevant_ids_for_strategy,
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
DEFAULT_STRATEGY = "fixed_length"
DEFAULT_TOP_K = 5
DEFAULT_RANDOM_SEED = 42
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def _ensure_inputs(paths: dict[str, Path]) -> None:
    for role, path in paths.items():
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{role} missing or empty: {path}")


def _first_relevant_rank(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row["is_relevant"]:
            return int(row["rank"])
    return None


def run_normalized_bm25_smoke(
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    data_source: Path = DATA_SOURCE,
    raw_source: Path | None = RAW_SOURCE,
    chunk_file: Path | None = None,
    qa_file: Path = QA_JSONL,
    chunk_strategy: str = DEFAULT_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    random_seed: int = DEFAULT_RANDOM_SEED,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Reuse the existing BM25 implementation and persist one normalized run."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    chunk_path = chunk_file or (
        CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{chunk_strategy}.jsonl"
    )
    required_inputs = {
        "data_source": Path(data_source),
        "chunk_file": Path(chunk_path),
        "qa_file": Path(qa_file),
    }
    extra_inputs: dict[str, Path] = {}
    if raw_source is not None:
        extra_inputs["raw_source"] = Path(raw_source)
    _ensure_inputs({**required_inputs, **extra_inputs})

    recorder = ExperimentRecorder(
        task_type="retrieval",
        retrieval_method="bm25",
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        embedding_model="not_applicable",
        qa_version="v1",
        random_seed=random_seed,
        inputs={**required_inputs, **extra_inputs},
        parameters={
            "tokenizer": "cjk_char_bigram",
            "k1": k1,
            "b": b,
            "metric_primary": f"Recall@{top_k}",
            "metric_secondary": "MRR",
            "mrr_cutoff": top_k,
        },
        experiment_root=Path(experiment_root),
        repo_root=Path(repo_root),
        generation_script="scripts/run_normalized_bm25_smoke.py",
    )

    with recorder:
        random.seed(random_seed)
        started = perf_counter()
        chunks = load_jsonl(Path(chunk_path))
        questions = load_jsonl(Path(qa_file))
        index = build_index(chunks, k1=k1, b=b)
        chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
        annotated_rows: list[dict[str, Any]] = []
        per_question_rows: list[dict[str, Any]] = []
        for question in questions:
            search_rows = search(question["question"], index, top_k=top_k)
            annotated = annotate_results(question, chunk_strategy, search_rows)
            annotated_rows.extend(annotated)
            first_rank = _first_relevant_rank(annotated)
            source_policy_title = str(question.get("source_policy_title", ""))
            retrieved_results: list[dict[str, Any]] = []
            source_policy_hit = False
            for row in annotated:
                chunk = chunks_by_id.get(str(row["retrieved_chunk_id"]), {})
                policy_title = str(
                    chunk.get("policy_title")
                    or chunk.get("metadata", {}).get("policy_title")
                    or ""
                )
                if source_policy_title and policy_title == source_policy_title:
                    source_policy_hit = True
                retrieved_results.append(
                    {
                        "rank": row["rank"],
                        "retrieved_chunk_id": row["retrieved_chunk_id"],
                        "retrieved_document_title": row[
                            "retrieved_document_title"
                        ],
                        "retrieved_policy_title": policy_title,
                        "score": row["score"],
                        "is_relevant": row["is_relevant"],
                        "matched_by": row["matched_by"],
                    }
                )
            per_question_rows.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "question_type": question.get("question_type", "unknown"),
                    "difficulty": question.get("difficulty", "unknown"),
                    "source_policy_title": source_policy_title,
                    "chunk_strategy": chunk_strategy,
                    "retrieval_method": "bm25",
                    "top_k": top_k,
                    "relevant_chunk_ids": sorted(
                        relevant_ids_for_strategy(question, chunk_strategy)
                    ),
                    "retrieved_results": retrieved_results,
                    "first_relevant_rank": first_rank,
                    f"hit_at_{top_k}": first_rank is not None,
                    "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                    "source_policy_hit": source_policy_hit,
                    "related_chunk_hit": first_rank is not None,
                }
            )

        metrics = evaluate_strategy_at_k(
            chunk_strategy,
            annotated_rows,
            questions,
            top_k,
        )
        question_count = len(questions)
        metrics.update(
            {
                "retrieval_method": "bm25",
                "tokenizer": "cjk_char_bigram",
                "k1": k1,
                "b": b,
                "random_seed": random_seed,
                "chunk_count": len(chunks),
                "per_question_result_count": len(per_question_rows),
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
                    "retrieval_method": "bm25",
                    "chunk_strategy": chunk_strategy,
                    "top_k": top_k,
                    "question_count": question_count,
                    "Recall@5": metrics.get("Recall@5", ""),
                    "MRR": metrics["MRR"],
                    "source_policy_hit_rate": metrics[
                        "source_policy_hit_rate"
                    ],
                    "related_chunk_hit_rate": metrics[
                        "related_chunk_hit_rate"
                    ],
                    "runtime_seconds": metrics["runtime_seconds"],
                }
            ],
        )

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
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--chunk-strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    args = parser.parse_args(argv)
    result = run_normalized_bm25_smoke(
        experiment_root=args.experiment_root,
        chunk_strategy=args.chunk_strategy,
        top_k=args.top_k,
        random_seed=args.random_seed,
        k1=args.k1,
        b=args.b,
    )
    metrics = result["metrics"]
    print(f"experiment_id: {result['experiment_id']}")
    print(f"run_dir: {result['run_dir']}")
    print(f"question_count: {metrics['question_count']}")
    print(f"per_question_result_count: {metrics['per_question_result_count']}")
    print(f"Recall@5: {metrics.get('Recall@5')}")
    print(f"MRR: {metrics['MRR']}")
    print(f"source_policy_hit_rate: {metrics['source_policy_hit_rate']}")
    print(f"related_chunk_hit_rate: {metrics['related_chunk_hit_rate']}")
    print(f"registry_legacy: {result['registry_row']['legacy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
