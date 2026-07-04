"""Evaluate Hybrid Search from existing Dense Hash and BM25 result files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH, EXPECTED_RAW_SHA256, sha256_file
from src.retrieval.dense_index import load_jsonl, write_jsonl
from src.retrieval.hybrid import (
    evaluate_hybrid_strategy,
    fuse_rrf,
    fuse_weighted_score,
    load_strategy_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
EXPERIMENT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval_hybrid.yaml"
QA_JSONL = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset" / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
CLEANED_PATH = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook" / "cleaned" / "hdu_ies_student_handbook_cleaned.txt"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
DENSE_EVALUATION_DIR = EXPERIMENT_ROOT / "dense_retrieval" / "evaluation"
BM25_EVALUATION_DIR = EXPERIMENT_ROOT / "bm25_retrieval" / "evaluation"
OUTPUT_ROOT = EXPERIMENT_ROOT / "hybrid_retrieval"
STRATEGIES = ["fixed_length", "article_based", "structure_aware"]
FUSION_METHODS = ["weighted_score", "rrf"]
TOP_K_VALUES = [1, 3, 5, 10]
DENSE_WEIGHT = 0.4
BM25_WEIGHT = 0.6
RRF_K = 60
TOP_K = 10
EQUAL_SCORE_VALUE = 0.0


def load_report_metrics(path: Path) -> dict[str, dict]:
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
            DENSE_EVALUATION_DIR / f"dense_search_results_{strategy}.jsonl",
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
        dense_by_question = load_strategy_results(DENSE_EVALUATION_DIR, "dense", strategy)
        bm25_by_question = load_strategy_results(BM25_EVALUATION_DIR, "bm25", strategy)
        all_metrics[strategy] = {}
        method_rows: dict[str, list[dict]] = {method: [] for method in FUSION_METHODS}
        for question in questions:
            question_id = question["question_id"]
            dense_rows = dense_by_question.get(question_id, [])
            bm25_rows = bm25_by_question.get(question_id, [])
            method_rows["weighted_score"].extend(
                fuse_weighted_score(
                    question,
                    strategy,
                    dense_rows,
                    bm25_rows,
                    dense_weight=DENSE_WEIGHT,
                    bm25_weight=BM25_WEIGHT,
                    top_k=TOP_K,
                    equal_score_value=EQUAL_SCORE_VALUE,
                )
            )
            method_rows["rrf"].extend(
                fuse_rrf(
                    question,
                    strategy,
                    dense_rows,
                    bm25_rows,
                    k=RRF_K,
                    top_k=TOP_K,
                )
            )
        for fusion_method, rows in method_rows.items():
            results_path = evaluation_dir / f"hybrid_search_results_{strategy}_{fusion_method}.jsonl"
            write_jsonl(results_path, rows)
            metrics = evaluate_hybrid_strategy(strategy, fusion_method, rows, questions)
            metrics["results_path"] = str(results_path)
            all_metrics[strategy][fusion_method] = metrics
            print(
                f"{strategy}/{fusion_method}: "
                f"Recall@5={metrics['Recall@5']:.4f} MRR@10={metrics['MRR@10']:.4f}"
            )

        combined_rows: list[dict] = []
        for fusion_method in FUSION_METHODS:
            combined_rows.extend(method_rows[fusion_method])
        write_jsonl(evaluation_dir / f"hybrid_search_results_{strategy}.jsonl", combined_rows)

    dense_metrics = load_report_metrics(DENSE_EVALUATION_DIR / "dense_search_evaluation_report.json")
    bm25_metrics = load_report_metrics(BM25_EVALUATION_DIR / "bm25_evaluation_report.json")
    report = {
        "stage": "hybrid_retrieval_baseline",
        "config_path": str(CONFIG_PATH),
        "fusion_methods": FUSION_METHODS,
        "weighted_score": {
            "dense_weight": DENSE_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "normalization": "per_query_min_max",
            "equal_score_value": EQUAL_SCORE_VALUE,
        },
        "rrf": {"k": RRF_K},
        "question_count": len(questions),
        "top_k": TOP_K,
        "top_k_values": TOP_K_VALUES,
        "metrics_by_strategy": all_metrics,
        "dense_hash_baseline_comparison": dense_metrics,
        "bm25_baseline_comparison": bm25_metrics,
        "raw_sha256": sha256_file(DEFAULT_RAW_PATH),
        "cleaned_sha256": sha256_file(CLEANED_PATH),
        "chunk_file_sha256": {
            strategy: sha256_file(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
            for strategy in STRATEGIES
        },
        "qa_jsonl_sha256": sha256_file(QA_JSONL),
        "notes": "Hybrid Search reads existing Dense Hash and BM25 results only. No chunk regeneration, Reranker, or RAG artifacts were generated.",
    }
    report_path = evaluation_dir / "hybrid_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(evaluation_dir / "hybrid_evaluation_summary.csv", all_metrics)
    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    evaluate_all(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
