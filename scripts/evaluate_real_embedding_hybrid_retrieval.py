"""Evaluate Real Embedding Dense + BM25 Hybrid Search."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH, EXPECTED_RAW_SHA256, sha256_file
from src.retrieval.dense_index import load_jsonl, write_jsonl
from src.retrieval.real_hybrid import (
    evaluate_hybrid_strategy,
    fuse_real_rrf,
    fuse_real_weighted_score,
    load_strategy_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval_hybrid_real_embedding.yaml"
QA_JSONL = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset" / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
CLEANED_PATH = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook" / "cleaned" / "hdu_ies_student_handbook_cleaned.txt"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
REAL_DENSE_EVALUATION_DIR = EXPERIMENT_ROOT / "dense_retrieval_real_embedding" / "evaluation"
BM25_EVALUATION_DIR = EXPERIMENT_ROOT / "bm25_retrieval" / "evaluation"
HASH_HYBRID_EVALUATION_DIR = EXPERIMENT_ROOT / "hybrid_retrieval" / "evaluation"
OUTPUT_ROOT = EXPERIMENT_ROOT / "hybrid_retrieval_real_embedding"
STRATEGIES = ["fixed_length", "article_based", "structure_aware"]
FUSION_METHODS = ["weighted_score", "rrf"]
TOP_K_VALUES = [1, 3, 5, 10]
REAL_DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5
RRF_K = 60
TOP_K = 10
EQUAL_SCORE_VALUE = 0.0


def load_report_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("metrics_by_strategy", {})


def write_summary_csv(path: Path, metrics_by_strategy: dict[str, dict[str, dict]]) -> None:
    fields = [
        "strategy",
        "fusion_method",
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
        for strategy, metrics_by_method in metrics_by_strategy.items():
            for fusion_method, metrics in metrics_by_method.items():
                writer.writerow({field: metrics.get(field, "") for field in fields})


def ensure_input_files() -> None:
    if sha256_file(DEFAULT_RAW_PATH) != EXPECTED_RAW_SHA256:
        raise RuntimeError("raw source hash changed")
    if not CLEANED_PATH.exists() or CLEANED_PATH.stat().st_size == 0:
        raise RuntimeError("cleaned source missing or empty")
    for strategy in STRATEGIES:
        paths = [
            CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl",
            REAL_DENSE_EVALUATION_DIR / f"real_embedding_dense_search_results_{strategy}.jsonl",
            BM25_EVALUATION_DIR / f"bm25_search_results_{strategy}.jsonl",
        ]
        for path in paths:
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"required input missing or empty: {path}")


def evaluate_all(output_root: Path = OUTPUT_ROOT) -> dict[str, dict[str, dict]]:
    ensure_input_files()
    questions = load_jsonl(QA_JSONL)
    evaluation_dir = output_root / "evaluation"
    all_metrics: dict[str, dict[str, dict]] = {}
    for strategy in STRATEGIES:
        real_dense_by_question = load_strategy_results(REAL_DENSE_EVALUATION_DIR, "real_embedding_dense", strategy)
        bm25_by_question = load_strategy_results(BM25_EVALUATION_DIR, "bm25", strategy)
        all_metrics[strategy] = {}
        method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in FUSION_METHODS}
        for question in questions:
            question_id = question["question_id"]
            real_dense_rows = real_dense_by_question.get(question_id, [])
            bm25_rows = bm25_by_question.get(question_id, [])
            method_rows["weighted_score"].extend(
                fuse_real_weighted_score(
                    question,
                    strategy,
                    real_dense_rows,
                    bm25_rows,
                    real_dense_weight=REAL_DENSE_WEIGHT,
                    bm25_weight=BM25_WEIGHT,
                    top_k=TOP_K,
                    equal_score_value=EQUAL_SCORE_VALUE,
                )
            )
            method_rows["rrf"].extend(
                fuse_real_rrf(
                    question,
                    strategy,
                    real_dense_rows,
                    bm25_rows,
                    k=RRF_K,
                    top_k=TOP_K,
                )
            )

        for fusion_method, rows in method_rows.items():
            results_path = evaluation_dir / f"real_hybrid_search_results_{strategy}_{fusion_method}.jsonl"
            write_jsonl(results_path, rows)
            metrics = evaluate_hybrid_strategy(strategy, fusion_method, rows, questions)
            metrics["results_path"] = str(results_path)
            all_metrics[strategy][fusion_method] = metrics
            print(
                f"{strategy}/{fusion_method}: "
                f"Recall@5={metrics['Recall@5']:.4f} MRR@10={metrics['MRR@10']:.4f}"
            )

        combined_rows: list[dict[str, Any]] = []
        for fusion_method in FUSION_METHODS:
            combined_rows.extend(method_rows[fusion_method])
        write_jsonl(evaluation_dir / f"real_hybrid_search_results_{strategy}.jsonl", combined_rows)

    real_dense_metrics = load_report_metrics(REAL_DENSE_EVALUATION_DIR / "real_embedding_dense_evaluation_report.json")
    bm25_metrics = load_report_metrics(BM25_EVALUATION_DIR / "bm25_evaluation_report.json")
    hash_hybrid_metrics = load_report_metrics(HASH_HYBRID_EVALUATION_DIR / "hybrid_evaluation_report.json")
    report = {
        "stage": "real_embedding_hybrid_retrieval",
        "config_path": str(CONFIG_PATH),
        "fusion_methods": FUSION_METHODS,
        "weighted_score": {
            "real_dense_weight": REAL_DENSE_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "normalization": "per_query_min_max",
            "equal_score_value": EQUAL_SCORE_VALUE,
        },
        "rrf": {"k": RRF_K},
        "question_count": len(questions),
        "top_k": TOP_K,
        "top_k_values": TOP_K_VALUES,
        "metrics_by_strategy": all_metrics,
        "real_dense_baseline_comparison": real_dense_metrics,
        "bm25_baseline_comparison": bm25_metrics,
        "hash_hybrid_baseline_comparison": hash_hybrid_metrics,
        "raw_sha256": sha256_file(DEFAULT_RAW_PATH),
        "cleaned_sha256": sha256_file(CLEANED_PATH),
        "chunk_file_sha256": {
            strategy: sha256_file(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
            for strategy in STRATEGIES
        },
        "qa_jsonl_sha256": sha256_file(QA_JSONL),
        "modified_inputs": {
            "raw": False,
            "cleaned": False,
            "chunks": False,
            "qa": False,
        },
        "overwrote_historical_outputs": {
            "dense_hash": False,
            "bm25": False,
            "hash_hybrid": False,
            "real_embedding_dense": False,
        },
        "notes": "Real Embedding + BM25 Hybrid Search only. No BM25 rebuild, Hash Hybrid overwrite, Reranker, or RAG artifacts were generated.",
    }
    report_path = evaluation_dir / "real_hybrid_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(evaluation_dir / "real_hybrid_evaluation_summary.csv", all_metrics)
    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    evaluate_all(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
