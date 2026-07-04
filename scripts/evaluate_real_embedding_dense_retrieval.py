"""Evaluate Dense Search with the configured real embedding provider."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_real_embedding_dense_retrieval import CONFIG_PATH, OUTPUT_ROOT, STRATEGIES, build_all
from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH, EXPECTED_RAW_SHA256, sha256_file
from src.retrieval.dense_index import load_jsonl, write_jsonl
from src.retrieval.embedding import cosine_similarity
from src.retrieval.embedding_providers import ProviderUnavailable, create_provider, validate_same_dimension
from src.retrieval.evaluation import annotate_results, evaluate_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
QA_JSONL = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset" / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
CLEANED_PATH = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook" / "cleaned" / "hdu_ies_student_handbook_cleaned.txt"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
TOP_K_VALUES = [1, 3, 5, 10]


def write_summary_csv(path: Path, metrics_by_strategy: dict[str, dict]) -> None:
    fields = [
        "strategy",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "average_top1_score",
        "average_relevant_rank",
        "no_hit_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for strategy, metrics in metrics_by_strategy.items():
            writer.writerow({field: metrics.get(field, strategy if field == "strategy" else "") for field in fields})


def search_with_provider(query: str, embedding_rows: list[dict[str, Any]], provider: Any, top_k: int = 10) -> list[dict[str, Any]]:
    query_vectors = provider.encode([query])
    validate_same_dimension(query_vectors)
    query_vector = query_vectors[0]
    scored = []
    for row in embedding_rows:
        scored.append((cosine_similarity(query_vector, row["embedding"]), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, Any]] = []
    for rank, (score, row) in enumerate(scored[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "retrieved_chunk_id": row["chunk_id"],
                "retrieved_document_title": row["document_title"],
                "score": score,
                "chunk_text": row["chunk_text"],
            }
        )
    return results


def write_report_only(output_root: Path, reason: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    evaluation_dir = output_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "dense_real_embedding_retrieval",
        "model_ran": False,
        "is_real_semantic_embedding": False,
        "official_experiment_ready": False,
        "embedding_provider": (config or {}).get("embedding_provider", "sentence_transformers"),
        "model_name": (config or {}).get("model_name", "BAAI/bge-small-zh-v1.5"),
        "model_cache_dir": (config or {}).get("model_cache_dir", str(PROJECT_ROOT.parent / "models")),
        "failure_reason": reason,
        "metrics_by_strategy": {},
        "raw_sha256": sha256_file(DEFAULT_RAW_PATH),
        "cleaned_sha256": sha256_file(CLEANED_PATH),
        "chunk_file_sha256": {
            strategy: sha256_file(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
            for strategy in STRATEGIES
        },
        "qa_jsonl_sha256": sha256_file(QA_JSONL),
        "notes": "Real embedding model was not run. No Reranker or RAG artifacts were generated.",
    }
    report_path = evaluation_dir / "real_embedding_dense_evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"real embedding model unavailable: {reason}")
    return report


def evaluate_all(output_root: Path = OUTPUT_ROOT, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if sha256_file(DEFAULT_RAW_PATH) != EXPECTED_RAW_SHA256:
        raise RuntimeError("raw source hash changed")
    if not CLEANED_PATH.exists() or CLEANED_PATH.stat().st_size == 0:
        raise RuntimeError("cleaned source missing or empty")

    from scripts.build_real_embedding_dense_retrieval import load_simple_yaml

    config = load_simple_yaml(config_path)
    build_summary = build_all(output_root, config_path)
    if not build_summary.get("model_ran"):
        return write_report_only(output_root, str(build_summary.get("reason", "provider unavailable")), config)

    try:
        provider = create_provider(config)
    except ProviderUnavailable as exc:
        return write_report_only(output_root, str(exc), config)

    questions = load_jsonl(QA_JSONL)
    evaluation_dir = output_root / "evaluation"
    all_metrics: dict[str, dict] = {}
    for strategy in STRATEGIES:
        embeddings_path = Path(build_summary["strategies"][strategy]["embeddings_path"])
        embedding_rows = load_jsonl(embeddings_path)
        annotated_rows = []
        for question in questions:
            results = search_with_provider(question["question"], embedding_rows, provider, top_k=10)
            annotated_rows.extend(annotate_results(question, strategy, results))
        results_path = evaluation_dir / f"real_embedding_dense_search_results_{strategy}.jsonl"
        write_jsonl(results_path, annotated_rows)
        all_metrics[strategy] = evaluate_strategy(strategy, annotated_rows, questions, TOP_K_VALUES)
        all_metrics[strategy]["results_path"] = str(results_path)
        print(f"{strategy}: Recall@5={all_metrics[strategy]['Recall@5']:.4f} MRR@10={all_metrics[strategy]['MRR@10']:.4f}")

    report = {
        "stage": "dense_real_embedding_retrieval",
        "model_ran": True,
        "is_real_semantic_embedding": True,
        "official_experiment_ready": True,
        "embedding_provider": build_summary["provider"],
        "question_count": len(questions),
        "top_k_values": TOP_K_VALUES,
        "metrics_by_strategy": all_metrics,
        "raw_sha256": sha256_file(DEFAULT_RAW_PATH),
        "cleaned_sha256": sha256_file(CLEANED_PATH),
        "chunk_file_sha256": {
            strategy: sha256_file(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
            for strategy in STRATEGIES
        },
        "qa_jsonl_sha256": sha256_file(QA_JSONL),
        "notes": "Real embedding Dense Search only. No BM25, Hybrid Search, Reranker, or RAG artifacts were generated.",
    }
    report_path = evaluation_dir / "real_embedding_dense_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(evaluation_dir / "real_embedding_dense_evaluation_summary.csv", all_metrics)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    evaluate_all(args.output_root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
