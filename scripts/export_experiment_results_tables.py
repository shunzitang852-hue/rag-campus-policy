from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
DATA_ROOT = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook")
EXPERIMENT_ROOT = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook")
OUTPUT_ROOT = EXPERIMENT_ROOT / "experiment_results_tables"

OUTPUT_JSON = OUTPUT_ROOT / "experiment_results_summary.json"
CHUNK_CSV = OUTPUT_ROOT / "chunk_strategy_results.csv"
RAG_COMPARISON_CSV = OUTPUT_ROOT / "rag_version_comparison.csv"
ERROR_TYPE_CSV = OUTPUT_ROOT / "error_type_distribution.csv"
DEFAULT_STRATEGY_CSV = OUTPUT_ROOT / "default_strategy_selection.csv"

RAW_PATH = Path(r"F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt")
CHUNK_REPORT_JSON = DATA_ROOT / "chunks" / "hdu_ies_student_handbook_chunking_report.json"
QA_REPORT_JSON = DATA_ROOT / "qa_testset" / "hdu_ies_student_handbook_qa_testset_report.json"
DENSE_REPORT_JSON = EXPERIMENT_ROOT / "dense_retrieval" / "evaluation" / "dense_search_evaluation_report.json"
BM25_REPORT_JSON = EXPERIMENT_ROOT / "bm25_retrieval" / "evaluation" / "bm25_evaluation_report.json"
HYBRID_REPORT_JSON = EXPERIMENT_ROOT / "hybrid_retrieval" / "evaluation" / "hybrid_evaluation_report.json"
REAL_DENSE_REPORT_JSON = (
    EXPERIMENT_ROOT
    / "dense_retrieval_real_embedding"
    / "evaluation"
    / "real_embedding_dense_evaluation_report.json"
)
REAL_HYBRID_REPORT_JSON = (
    EXPERIMENT_ROOT / "hybrid_retrieval_real_embedding" / "evaluation" / "real_hybrid_evaluation_report.json"
)
RAG_EVALUATION_REPORT_JSON = EXPERIMENT_ROOT / "rag_evaluation" / "rag_evaluation_report.json"
RAG_V1_REPORT_JSON = EXPERIMENT_ROOT / "rag_answer_quality" / "rag_answer_quality_report.json"
RAG_V2_REPORT_JSON = (
    EXPERIMENT_ROOT / "rag_answer_quality_improvement" / "rag_answer_quality_improvement_report.json"
)
RAG_V2_ERROR_REPORT_JSON = (
    EXPERIMENT_ROOT / "rag_answer_quality_error_analysis" / "rag_answer_error_analysis_v2_report.json"
)
RAG_V3_REPORT_JSON = (
    EXPERIMENT_ROOT / "rag_answer_quality_v3_optimization" / "rag_answer_quality_v3_report.json"
)
RAG_V2_V3_REPORT_JSON = (
    EXPERIMENT_ROOT / "rag_answer_quality_v3_optimization" / "rag_answer_quality_comparison_v2_v3_report.json"
)

HISTORICAL_OUTPUT_ROOTS = {
    EXPERIMENT_ROOT / "dense_retrieval",
    EXPERIMENT_ROOT / "bm25_retrieval",
    EXPERIMENT_ROOT / "hybrid_retrieval",
    EXPERIMENT_ROOT / "dense_retrieval_real_embedding",
    EXPERIMENT_ROOT / "hybrid_retrieval_real_embedding",
    EXPERIMENT_ROOT / "rag_evaluation",
    EXPERIMENT_ROOT / "rag_answer_quality",
    EXPERIMENT_ROOT / "rag_answer_quality_improvement",
    EXPERIMENT_ROOT / "rag_answer_quality_error_analysis",
    EXPERIMENT_ROOT / "rag_answer_quality_v3_optimization",
    EXPERIMENT_ROOT / "rag_experiment_summary_for_thesis",
}

NOT_FOUND = "未在当前结果中找到"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(data: dict[str, Any], key: str) -> Any:
    return data.get(key, NOT_FOUND)


def quality_distribution_text(distribution: dict[str, Any]) -> str:
    return ", ".join(f"{key}={val}" for key, val in distribution.items())


def retrieval_metric_rows() -> list[dict[str, Any]]:
    reports = [
        ("Dense Hash", "dense_hash", load_json(DENSE_REPORT_JSON), False),
        ("BM25", "bm25", load_json(BM25_REPORT_JSON), False),
        ("Hybrid weighted", "hybrid_weighted", load_json(HYBRID_REPORT_JSON), "weighted_score"),
        ("Hybrid rrf", "hybrid_rrf", load_json(HYBRID_REPORT_JSON), "rrf"),
        ("Real Dense", "real_dense", load_json(REAL_DENSE_REPORT_JSON), False),
        ("Real Hybrid weighted", "real_hybrid_weighted", load_json(REAL_HYBRID_REPORT_JSON), "weighted_score"),
        ("Real Hybrid rrf", "real_hybrid_rrf", load_json(REAL_HYBRID_REPORT_JSON), "rrf"),
    ]
    rows: list[dict[str, Any]] = []
    for label, key, report, method in reports:
        for strategy, metrics in report["metrics_by_strategy"].items():
            row_metrics = metrics[method] if method else metrics
            rows.append(
                {
                    "retrieval_method": key,
                    "label": label,
                    "chunk_strategy": strategy,
                    "Recall@1": value(row_metrics, "Recall@1"),
                    "Recall@3": value(row_metrics, "Recall@3"),
                    "Recall@5": value(row_metrics, "Recall@5"),
                    "Recall@10": value(row_metrics, "Recall@10"),
                    "MRR@10": value(row_metrics, "MRR@10"),
                    "no_hit_count": value(row_metrics, "no_hit_count"),
                    "question_count": value(row_metrics, "question_count"),
                }
            )
    return rows


def build_results_summary() -> dict[str, Any]:
    chunk_report = load_json(CHUNK_REPORT_JSON)
    qa_report = load_json(QA_REPORT_JSON)
    dense_report = load_json(DENSE_REPORT_JSON)
    rag_eval = load_json(RAG_EVALUATION_REPORT_JSON)
    rag_v1 = load_json(RAG_V1_REPORT_JSON)
    rag_v2 = load_json(RAG_V2_REPORT_JSON)
    rag_v2_error = load_json(RAG_V2_ERROR_REPORT_JSON)
    rag_v3 = load_json(RAG_V3_REPORT_JSON)
    rag_v2_v3 = load_json(RAG_V2_V3_REPORT_JSON)

    chunk_rows = []
    for strategy, metrics in chunk_report["strategies"].items():
        chunk_rows.append(
            {
                "strategy": strategy,
                "chunk_count": metrics["chunk_count"],
                "average_char_count": metrics["average_char_count"],
                "body_coverage_ratio": metrics["body_coverage_ratio"],
                "duplicate_coverage_ratio": metrics["duplicate_coverage_ratio"],
                "uncovered_source_record_count": metrics["uncovered_source_record_count"],
                "empty_chunk_count": metrics["empty_chunk_count"],
                "duplicate_chunk_count": metrics["duplicate_chunk_count"],
                "feature_and_applicability": {
                    "fixed_length": "固定窗口基线；当前检索表现最稳，适合作为默认 RAG 输入。",
                    "article_based": "更贴近条款结构，片段更短；适合结构化制度阅读对比。",
                    "structure_aware": "保留解析结构信息；适合后续结构规则扩展。",
                }.get(strategy, NOT_FOUND),
                "chunks_path": metrics["chunks_path"],
            }
        )

    rag_comparison_rows = [
        {
            "version": "v1",
            "question_count": rag_v1["summary"]["question_count"],
            "average_keyword_overlap_ratio": rag_v1["summary"]["average_keyword_overlap_ratio"],
            "average_evidence_overlap_ratio": rag_v1["summary"]["average_evidence_overlap_ratio"],
            "auto_quality_distribution": quality_distribution_text(rag_v1["summary"]["auto_quality_distribution"]),
            "needs_review_count": rag_v1["summary"]["needs_review_count"],
            "comparison_role": "baseline",
        },
        {
            "version": "v2",
            "question_count": rag_v2["v2_summary"]["question_count"],
            "average_keyword_overlap_ratio": rag_v2["v2_summary"]["average_keyword_overlap_ratio"],
            "average_evidence_overlap_ratio": rag_v2["v2_summary"]["average_evidence_overlap_ratio"],
            "auto_quality_distribution": quality_distribution_text(rag_v2["v2_summary"]["auto_quality_distribution"]),
            "needs_review_count": rag_v2["v2_summary"]["needs_review_count"],
            "comparison_role": "current_default_answer_strategy",
        },
        {
            "version": "v3",
            "question_count": rag_v3["total_questions"],
            "average_keyword_overlap_ratio": rag_v3["average_keyword_overlap_ratio"],
            "average_evidence_overlap_ratio": rag_v3["average_evidence_overlap_ratio"],
            "auto_quality_distribution": quality_distribution_text(rag_v3["auto_quality_distribution"]),
            "needs_review_count": rag_v3["needs_review_count"],
            "comparison_role": "targeted_optimization_experiment",
        },
    ]

    error_type_rows = [
        {"error_type": error_type, "count": count, "source": "rag_v2_error_analysis"}
        for error_type, count in rag_v2_error["error_type_distribution"].items()
    ]

    default_strategy_rows = [
        {
            "component": "retrieval_input",
            "selected_option": "real_hybrid_weighted + fixed_length + top_k=5",
            "basis": "Real Hybrid weighted + fixed_length 的 Recall@5=1.0、MRR@10=1.0；RAG v1/v2/v3 均基于该组合。",
            "use_as_default": "yes",
            "needs_manual_review": "no",
        },
        {
            "component": "rag_answer_version",
            "selected_option": "RAG v2",
            "basis": "v2 相比 v1 关键词重合率 0.5616666666666668 -> 0.5925，证据重合率 0.7741666666666667 -> 0.8012499999999999，needs_review 19 -> 17。",
            "use_as_default": "yes",
            "needs_manual_review": "yes",
        },
        {
            "component": "rag_v3_role",
            "selected_option": "定向优化实验版本",
            "basis": "v3 改善 condition/medium 目标分组，但整体 keyword overlap 和 needs_review 不如 v2。",
            "use_as_default": "no",
            "needs_manual_review": "yes",
        },
    ]

    return {
        "data_consistency": {
            "raw_path": str(RAW_PATH),
            "raw_sha256": qa_report["raw_sha256"],
            "cleaned_path": chunk_report["input_file"],
            "cleaned_sha256": qa_report["cleaned_sha256"],
            "qa_jsonl_path": qa_report["jsonl_path"],
            "qa_jsonl_sha256": dense_report["qa_jsonl_sha256"],
            "chunk_file_sha256": dense_report["chunk_file_sha256"],
            "hashes_unchanged_in_later_experiments": True,
        },
        "chunk_strategy_results": chunk_rows,
        "retrieval_strategy_results": retrieval_metric_rows(),
        "rag_v1_summary": {
            "retrieval_input": "real_hybrid_weighted + fixed_length + top_k=5",
            "summary": rag_v1["summary"],
            "output_files": rag_v1["output_files"],
            "main_conclusion": "v1 是自动评价基线；来源命中和引用覆盖较好，但 needs_review=19。",
        },
        "rag_v2_summary": {
            "retrieval_input": "real_hybrid_weighted + fixed_length + top_k=5",
            "summary": rag_v2["v2_summary"],
            "output_files": rag_v2["output_files"],
            "v1_v2_comparison": rag_v2["comparison"],
            "main_conclusion": "v2 相比 v1 整体自动指标小幅改善，是当前默认 RAG 回答方案。",
        },
        "rag_v2_error_analysis_summary": {
            "summary": {
                key: rag_v2_error[key]
                for key in [
                    "total_questions",
                    "needs_review_count",
                    "worsened_count",
                    "condition_case_count",
                    "medium_case_count",
                    "error_type_distribution",
                    "condition_error_summary",
                    "medium_error_summary",
                    "manual_review_status",
                ]
            },
            "output_files": rag_v2_error["output_files"],
            "main_conclusion": "condition 条件覆盖不足和 medium 回答过泛是后续重点；人工复核待完成。",
        },
        "rag_v3_summary": {
            "retrieval_input": "real_hybrid_weighted + fixed_length + top_k=5",
            "summary": {
                "total_questions": rag_v3["total_questions"],
                "average_keyword_overlap_ratio": rag_v3["average_keyword_overlap_ratio"],
                "average_evidence_overlap_ratio": rag_v3["average_evidence_overlap_ratio"],
                "auto_quality_distribution": rag_v3["auto_quality_distribution"],
                "needs_review_count": rag_v3["needs_review_count"],
                "evidence_insufficient_count": rag_v3["evidence_insufficient_count"],
            },
            "output_files": rag_v3["output_files"],
            "main_conclusion": "v3 是 condition/medium 定向优化实验版本，不替代 v2 默认方案。",
        },
        "rag_v2_v3_comparison": {
            "summary": {
                key: rag_v2_v3[key]
                for key in [
                    "v2_average_keyword_overlap_ratio",
                    "v3_average_keyword_overlap_ratio",
                    "v2_average_evidence_overlap_ratio",
                    "v3_average_evidence_overlap_ratio",
                    "v2_needs_review_count",
                    "v3_needs_review_count",
                    "improved_count",
                    "worsened_count",
                    "unchanged_count",
                    "condition_v2_vs_v3_summary",
                    "medium_v2_vs_v3_summary",
                ]
            },
            "output_files": rag_v2_v3["output_files"],
            "main_conclusion": "v3 对目标分组有改善，但整体自动指标不如 v2。",
        },
        "rag_batch_evaluation": {
            "default_report": rag_eval["default_report"],
            "answer_paths": rag_eval["answer_paths"],
        },
        "rag_version_comparison_rows": rag_comparison_rows,
        "error_type_distribution_rows": error_type_rows,
        "default_strategy_selection_rows": default_strategy_rows,
        "limitations": [
            "自动评价指标只能作为辅助，不等同于人工评价。",
            "needs_review、worsened、unknown 结果保留，尚未完成人工复核。",
            "未调用大模型 API，未做 Reranker，未做 Web 前端。",
        ],
        "output_files": {
            "experiment_results_summary_json": str(OUTPUT_JSON),
            "chunk_strategy_results_csv": str(CHUNK_CSV),
            "rag_version_comparison_csv": str(RAG_COMPARISON_CSV),
            "error_type_distribution_csv": str(ERROR_TYPE_CSV),
            "default_strategy_selection_csv": str(DEFAULT_STRATEGY_CSV),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    summary = build_results_summary()
    output_root.mkdir(parents=True, exist_ok=True)

    output_json = output_root / OUTPUT_JSON.name
    chunk_csv = output_root / CHUNK_CSV.name
    rag_comparison_csv = output_root / RAG_COMPARISON_CSV.name
    error_type_csv = output_root / ERROR_TYPE_CSV.name
    default_strategy_csv = output_root / DEFAULT_STRATEGY_CSV.name

    summary["output_files"] = {
        "experiment_results_summary_json": str(output_json),
        "chunk_strategy_results_csv": str(chunk_csv),
        "rag_version_comparison_csv": str(rag_comparison_csv),
        "error_type_distribution_csv": str(error_type_csv),
        "default_strategy_selection_csv": str(default_strategy_csv),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(
        chunk_csv,
        summary["chunk_strategy_results"],
        [
            "strategy",
            "chunk_count",
            "average_char_count",
            "body_coverage_ratio",
            "duplicate_coverage_ratio",
            "uncovered_source_record_count",
            "empty_chunk_count",
            "duplicate_chunk_count",
            "feature_and_applicability",
            "chunks_path",
        ],
    )
    write_csv(
        rag_comparison_csv,
        summary["rag_version_comparison_rows"],
        [
            "version",
            "question_count",
            "average_keyword_overlap_ratio",
            "average_evidence_overlap_ratio",
            "auto_quality_distribution",
            "needs_review_count",
            "comparison_role",
        ],
    )
    write_csv(error_type_csv, summary["error_type_distribution_rows"], ["error_type", "count", "source"])
    write_csv(
        default_strategy_csv,
        summary["default_strategy_selection_rows"],
        ["component", "selected_option", "basis", "use_as_default", "needs_manual_review"],
    )
    return summary


def main() -> None:
    summary = write_outputs()
    for path in summary["output_files"].values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
