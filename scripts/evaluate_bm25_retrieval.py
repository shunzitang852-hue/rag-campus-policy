"""Evaluate BM25 retrieval baseline for all chunk strategies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_bm25_retrieval_baseline import B, K1, OUTPUT_ROOT, STRATEGIES, build_all
from src.retrieval.bm25 import search
from src.retrieval.dense_index import load_jsonl, write_jsonl
from src.retrieval.evaluation import annotate_results, evaluate_strategy
from src.preprocessing.policy_splitter import DEFAULT_RAW_PATH, EXPECTED_RAW_SHA256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
QA_JSONL = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset" / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
CLEANED_PATH = DATASETS_ROOT / "interim" / "hdu_ies" / "student_handbook" / "cleaned" / "hdu_ies_student_handbook_cleaned.txt"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
DENSE_REPORT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook" / "dense_retrieval" / "evaluation" / "dense_search_evaluation_report.json"
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


def load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_all(output_root: Path = OUTPUT_ROOT, k1: float = K1, b: float = B) -> dict[str, dict]:
    if sha256_file(DEFAULT_RAW_PATH) != EXPECTED_RAW_SHA256:
        raise RuntimeError("raw source hash changed")
    if not CLEANED_PATH.exists() or CLEANED_PATH.stat().st_size == 0:
        raise RuntimeError("cleaned source missing or empty")

    build_summary = build_all(output_root, k1, b)
    questions = load_jsonl(QA_JSONL)
    evaluation_dir = output_root / "evaluation"
    all_metrics: dict[str, dict] = {}
    for strategy in STRATEGIES:
        index = load_index(Path(build_summary[strategy]["index_path"]))
        annotated_rows = []
        for question in questions:
            results = search(question["question"], index, top_k=10)
            annotated_rows.extend(annotate_results(question, strategy, results))
        results_path = evaluation_dir / f"bm25_search_results_{strategy}.jsonl"
        write_jsonl(results_path, annotated_rows)
        all_metrics[strategy] = evaluate_strategy(strategy, annotated_rows, questions, TOP_K_VALUES)
        all_metrics[strategy]["results_path"] = str(results_path)
        print(f"{strategy}: Recall@5={all_metrics[strategy]['Recall@5']:.4f} MRR@10={all_metrics[strategy]['MRR@10']:.4f}")

    dense_metrics = {}
    if DENSE_REPORT.exists():
        dense_report = json.loads(DENSE_REPORT.read_text(encoding="utf-8"))
        dense_metrics = {
            strategy: {
                "Recall@5": metrics["Recall@5"],
                "MRR@10": metrics["MRR@10"],
            }
            for strategy, metrics in dense_report.get("metrics_by_strategy", {}).items()
        }

    report = {
        "stage": "bm25_retrieval_baseline",
        "tokenizer": "cjk_char_bigram",
        "tokenizer_limitations": "标准库实现，使用中文单字和相邻二元片段；不能像jieba等中文分词工具一样识别完整词语和专有名词。",
        "future_tokenizer_recommendation": "后续可替换为jieba、pkuseg或适合中文政策文本的分词工具后重新计算。",
        "k1": k1,
        "b": b,
        "question_count": len(questions),
        "top_k_values": TOP_K_VALUES,
        "metrics_by_strategy": all_metrics,
        "dense_hash_baseline_comparison": dense_metrics,
        "raw_sha256": sha256_file(DEFAULT_RAW_PATH),
        "cleaned_sha256": sha256_file(CLEANED_PATH),
        "chunk_file_sha256": {
            strategy: sha256_file(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
            for strategy in STRATEGIES
        },
        "qa_jsonl_sha256": sha256_file(QA_JSONL),
        "notes": "Only BM25 retrieval baseline was run. No Hybrid Search, Reranker, or RAG artifacts were generated.",
    }
    report_path = evaluation_dir / "bm25_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(evaluation_dir / "bm25_evaluation_summary.csv", all_metrics)
    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--k1", type=float, default=K1)
    parser.add_argument("--b", type=float, default=B)
    args = parser.parse_args()
    evaluate_all(args.output_root, args.k1, args.b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
