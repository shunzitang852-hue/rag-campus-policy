# 实验结果索引

本索引只记录当前项目中已有实验结果的位置、对应脚本、对应文档和论文引用状态，便于后续人工撰写论文时查阅。本轮未做新实验，未调用大模型 API，未做 Reranker，未做 Web 前端，未覆盖历史实验输出。

| 类别 | 项目内脚本 | 项目内文档 | 外部输出路径 | 是否可用于论文引用 | 是否需要人工复核 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 数据清洗与结构解析 | `scripts/clean_hdu_ies_student_handbook.py`; `scripts/parse_policy_structure.py` | `docs/data/hdu_ies_student_handbook_cleaning_report.md`; `docs/data/hdu_ies_student_handbook_parsing_report.md` | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook` | 是 | 是 | raw、cleaned、结构化结果是后续实验固定输入；政策真实性仍需人工确认。 |
| QA 测试集 | `scripts/build_student_handbook_qa_testset.py` | `docs/data/hdu_ies_student_handbook_qa_testset_report.md` | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset` | 是 | 是 | 40 个问题覆盖 32 项制度；标准答案与证据仍需人工复核。 |
| Chunk 策略 | `scripts/generate_student_handbook_chunks.py` | `docs/data/hdu_ies_student_handbook_chunking_design.md`; `docs/data/hdu_ies_student_handbook_chunking_report.md` | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks` | 是 | 部分需要 | fixed_length、article_based、structure_aware 三种策略均覆盖政策正文。 |
| Dense Hash | `scripts/evaluate_dense_retrieval.py` | `docs/retrieval/dense_retrieval_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval` | 仅可作为流程基线 | 否 | deterministic hash embedding，不是真实语义模型。 |
| BM25 | `scripts/evaluate_bm25_retrieval.py` | `docs/retrieval/bm25_retrieval_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval` | 是 | 否 | 标准库 BM25，中文切词能力有限。 |
| Hybrid Hash | `scripts/evaluate_hybrid_retrieval.py` | `docs/retrieval/hybrid_retrieval_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval` | 是，需注明 hash dense 限制 | 否 | 融合 Dense Hash 与 BM25，不能等同真实语义 Hybrid。 |
| Real Dense | `scripts/evaluate_real_embedding_dense_retrieval.py` | `docs/retrieval/real_embedding_retrieval_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding` | 是 | 否 | 使用 `BAAI/bge-small-zh-v1.5`。 |
| Real Hybrid | `scripts/evaluate_real_embedding_hybrid_retrieval.py` | `docs/retrieval/real_embedding_hybrid_retrieval_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding` | 是 | 否 | 当前推荐 RAG 检索输入为 `real_hybrid_weighted + fixed_length + top_k=5`。 |
| RAG v1 批量评价 | `scripts/evaluate_rag_qa.py`; `scripts/evaluate_rag_answer_quality.py` | `docs/rag/rag_evaluation_report.md`; `docs/rag/rag_answer_quality_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation`; `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality` | 是，标注为自动评价 | 是 | v1 是基线回答生成与自动质量评估。 |
| RAG v2 优化 | `scripts/evaluate_rag_answer_quality_improvement.py` | `docs/rag/rag_answer_quality_improvement_report.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement` | 是，标注为自动评价 | 是 | v2 是当前默认 RAG 回答方案。 |
| RAG v2 误差分析 | `scripts/analyze_rag_answer_errors_v2.py` | `docs/rag/rag_answer_error_analysis_v2.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_error_analysis` | 是，作为误差分析材料 | 是 | condition 与 medium 是后续重点，人工复核状态为待完成。 |
| RAG v3 定向优化 | `scripts/generate_rag_answers_v3.py`; `scripts/review_rag_answer_quality_v3.py`; `scripts/compare_rag_answer_quality_v2_v3.py` | `docs/rag/rag_answer_quality_v3_optimization.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_v3_optimization` | 是，作为定向优化实验 | 是 | v3 不替代 v2 默认方案。 |
| 论文实验汇总 | `scripts/summarize_rag_experiments_for_thesis.py` | `docs/rag/rag_experiment_summary_for_thesis.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_experiment_summary_for_thesis` | 是，作为综合汇总 | 是 | 汇总已有实验，不新增实验。 |
| 本轮结果表导出 | `scripts/export_experiment_results_tables.py` | `docs/results/experiment_results_tables.md`; `docs/results/experiment_results_summary.md` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\experiment_results_tables` | 是，作为查阅表 | 是 | 导出 JSON/CSV 结果表，不提交外部输出。 |

## 固定边界

- 当前结果均来自已有文件、报告和外部实验输出。
- 自动评价指标只作为辅助，不等于人工评价。
- `needs_review`、`worsened`、`unknown` 均保留，不做美化。
- 找不到的指标统一标记为“未在当前结果中找到”。
