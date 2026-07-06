from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
DATA_ROOT = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook")
EXPERIMENT_ROOT = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook")
OUTPUT_ROOT = EXPERIMENT_ROOT / "rag_experiment_summary_for_thesis"
OUTPUT_JSON = OUTPUT_ROOT / "rag_experiment_summary_for_thesis.json"

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
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "question_count",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "no_hit_count",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def summarize_retrieval_report(report: dict[str, Any], nested_methods: bool = False) -> dict[str, Any]:
    metrics_by_strategy = report["metrics_by_strategy"]
    if nested_methods:
        metrics = {
            strategy: {method: compact_metrics(values) for method, values in methods.items()}
            for strategy, methods in metrics_by_strategy.items()
        }
    else:
        metrics = {strategy: compact_metrics(values) for strategy, values in metrics_by_strategy.items()}
    return {
        "stage": report["stage"],
        "question_count": report["question_count"],
        "metrics_by_strategy": metrics,
        "raw_sha256": report.get("raw_sha256"),
        "cleaned_sha256": report.get("cleaned_sha256"),
        "chunk_file_sha256": report.get("chunk_file_sha256"),
        "qa_jsonl_sha256": report.get("qa_jsonl_sha256"),
        "notes": report.get("notes"),
    }


def build_chunking_summary(chunk_report: dict[str, Any]) -> dict[str, Any]:
    strategies = {}
    for name, metrics in chunk_report["strategies"].items():
        strategies[name] = {
            "chunk_count": metrics["chunk_count"],
            "average_char_count": metrics["average_char_count"],
            "body_coverage_ratio": metrics["body_coverage_ratio"],
            "duplicate_coverage_ratio": metrics["duplicate_coverage_ratio"],
            "empty_chunk_count": metrics["empty_chunk_count"],
            "covered_policy_count": metrics["covered_policy_count"],
            "missing_policy_count": metrics["missing_policy_count"],
            "chunks_path": metrics["chunks_path"],
        }
    return {
        "stage": chunk_report["stage"],
        "input_file": chunk_report["input_file"],
        "output_dir": chunk_report["output_dir"],
        "strategy_order": chunk_report["strategy_order"],
        "policy_count": chunk_report["policy_count"],
        "strategies": strategies,
        "coverage_note": "覆盖率按唯一结构记录计算，重复覆盖率单独记录。",
    }


def build_summary() -> dict[str, Any]:
    chunk_report = load_json(CHUNK_REPORT_JSON)
    qa_report = load_json(QA_REPORT_JSON)
    dense = load_json(DENSE_REPORT_JSON)
    bm25 = load_json(BM25_REPORT_JSON)
    hybrid = load_json(HYBRID_REPORT_JSON)
    real_dense = load_json(REAL_DENSE_REPORT_JSON)
    real_hybrid = load_json(REAL_HYBRID_REPORT_JSON)
    rag_eval = load_json(RAG_EVALUATION_REPORT_JSON)
    rag_v1 = load_json(RAG_V1_REPORT_JSON)
    rag_v2 = load_json(RAG_V2_REPORT_JSON)
    rag_v2_errors = load_json(RAG_V2_ERROR_REPORT_JSON)
    rag_v3 = load_json(RAG_V3_REPORT_JSON)
    rag_v2_v3 = load_json(RAG_V2_V3_REPORT_JSON)

    retrieval_summary = {
        "dense_hash": {
            **summarize_retrieval_report(dense),
            "embedding_method": dense["embedding_method"],
            "official_experiment_ready": dense["official_experiment_ready"],
        },
        "bm25": {
            **summarize_retrieval_report(bm25),
            "tokenizer": bm25["tokenizer"],
        },
        "hybrid_hash": summarize_retrieval_report(hybrid, nested_methods=True),
        "real_dense": {
            **summarize_retrieval_report(real_dense),
            "provider": real_dense["provider"],
            "model_name": real_dense["model_name"],
            "embedding_dimension": real_dense["embedding_dimension"],
            "official_experiment_ready": real_dense["official_experiment_ready"],
        },
        "real_hybrid": summarize_retrieval_report(real_hybrid, nested_methods=True),
        "recommended_rag_retrieval_input": {
            "method": "real_hybrid_weighted",
            "chunk_strategy": "fixed_length",
            "top_k": 5,
            "basis": "Real Hybrid weighted_score 在 fixed_length 上 Recall@5=1.0、MRR@10=1.0，且后续 RAG v1/v2/v3 均基于该输入组合生成或评估。",
        },
    }

    return {
        "project_stage": "论文实验章节阶段性汇总",
        "data_consistency": {
            "raw_sha256": qa_report["raw_sha256"],
            "cleaned_sha256": qa_report["cleaned_sha256"],
            "qa_jsonl_sha256": dense["qa_jsonl_sha256"],
            "chunk_file_sha256": real_hybrid.get("chunk_file_sha256"),
            "unchanged_artifacts_statement": "raw、cleaned、chunk、QA hash 在 Dense/BM25/Hybrid/Real Dense/Real Hybrid/RAG v1/v2/v3 后续实验中保持不变；本汇总仅新增文档和汇总 JSON。",
        },
        "chunking_summary": build_chunking_summary(chunk_report),
        "retrieval_summary": retrieval_summary,
        "rag_v1_summary": {
            "stage": rag_v1["stage"],
            "summary": rag_v1["summary"],
            "used_llm_api": rag_v1["used_llm_api"],
            "used_reranker": rag_v1["used_reranker"],
            "used_web_frontend": rag_v1["used_web_frontend"],
        },
        "rag_v2_summary": {
            "stage": rag_v2["stage"],
            "summary": rag_v2["v2_summary"],
            "used_llm_api": rag_v2["used_llm_api"],
            "used_reranker": rag_v2["used_reranker"],
            "used_web_frontend": rag_v2["used_web_frontend"],
            "used_expected_answer_for_generation": rag_v2["used_expected_answer_for_generation"],
            "used_evidence_text_for_generation": rag_v2["used_evidence_text_for_generation"],
        },
        "rag_v2_error_analysis_summary": {
            key: rag_v2_errors[key]
            for key in [
                "stage",
                "total_questions",
                "needs_review_count",
                "worsened_count",
                "condition_case_count",
                "medium_case_count",
                "error_type_distribution",
                "condition_error_summary",
                "medium_error_summary",
                "recommendations_for_v3",
                "manual_review_status",
            ]
        },
        "rag_v3_summary": {
            "stage": rag_v3["stage"],
            "experimental_role": "targeted_optimization_experiment",
            "is_default_answer_strategy": False,
            "total_questions": rag_v3["total_questions"],
            "average_keyword_overlap_ratio": rag_v3["average_keyword_overlap_ratio"],
            "average_evidence_overlap_ratio": rag_v3["average_evidence_overlap_ratio"],
            "auto_quality_distribution": rag_v3["auto_quality_distribution"],
            "needs_review_count": rag_v3["needs_review_count"],
            "condition_quality_summary": rag_v3["condition_quality_summary"],
            "medium_quality_summary": rag_v3["medium_quality_summary"],
            "used_llm_api": rag_v3["used_llm_api"],
            "used_reranker": rag_v3["used_reranker"],
            "used_web_frontend": rag_v3["used_web_frontend"],
        },
        "rag_v1_v2_comparison": rag_v2["comparison"],
        "rag_v2_v3_comparison": {
            key: rag_v2_v3[key]
            for key in [
                "stage",
                "v2_average_keyword_overlap_ratio",
                "v3_average_keyword_overlap_ratio",
                "v2_average_evidence_overlap_ratio",
                "v3_average_evidence_overlap_ratio",
                "v2_auto_quality_distribution",
                "v3_auto_quality_distribution",
                "v2_needs_review_count",
                "v3_needs_review_count",
                "improved_count",
                "worsened_count",
                "unchanged_count",
                "condition_v2_vs_v3_summary",
                "medium_v2_vs_v3_summary",
                "risks_and_limitations",
            ]
        },
        "default_rag_answer_strategy": {
            "answer_version": "v2",
            "retrieval_method": "real_hybrid_weighted",
            "chunk_strategy": "fixed_length",
            "top_k": 5,
            "basis": "v2 相比 v1 在关键词重合率、证据重合率、good 数量和 needs_review 数量上整体更稳；v3 虽改善目标分组，但整体 needs_review 增加，因此不替代 v2。",
        },
        "main_findings": [
            "fixed_length 在当前 QA 集上是检索阶段最稳定的 chunk 策略。",
            "Real Hybrid weighted_score + fixed_length 是当前推荐的 RAG 检索输入。",
            "RAG v2 是当前默认回答方案；v3 是 condition/medium 定向优化实验版本。",
            "所有自动质量指标只能作为辅助复核信号，不等同于人工正确性评价。",
        ],
        "limitations": [
            "尚未完成正式人工复核。",
            "尚未引入 Reranker。",
            "尚未使用 LLM Judge。",
            "尚未调用真实大模型 API 生成对照答案。",
            "当前 RAG 回答仍为规则型抽取/模板生成，不能代表完整生成式大模型能力。",
        ],
        "future_work": [
            "对 v2/v3 improved、worsened、needs_review 样例进行人工复核。",
            "开展 Reranker 对照实验。",
            "引入 LLM Judge 或真实大模型生成结果作为扩展对比。",
            "将本汇总改写为毕业论文实验章节正文。",
        ],
        "source_reports": {
            "chunk_report": str(CHUNK_REPORT_JSON),
            "qa_report": str(QA_REPORT_JSON),
            "rag_batch_evaluation": str(RAG_EVALUATION_REPORT_JSON),
            "rag_v1_quality": str(RAG_V1_REPORT_JSON),
            "rag_v2_quality": str(RAG_V2_REPORT_JSON),
            "rag_v2_error_analysis": str(RAG_V2_ERROR_REPORT_JSON),
            "rag_v3_quality": str(RAG_V3_REPORT_JSON),
            "rag_v2_v3_comparison": str(RAG_V2_V3_REPORT_JSON),
        },
        "output_files": {"summary_json": str(OUTPUT_JSON)},
        "historical_outputs_overwritten": False,
    }


def write_summary(output_path: Path = OUTPUT_JSON) -> dict[str, Any]:
    summary = build_summary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = write_summary()
    print(f"Wrote {summary['output_files']['summary_json']}")


if __name__ == "__main__":
    main()
