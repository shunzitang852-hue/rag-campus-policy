# 实验资产审计

## 1. 审计范围与结论

- 固定论文题目：《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》
- 审计日期：2026-07-20
- 仓库：`F:\GraduationThesis\rag-campus-policy`
- 审计依据：仓库代码、`configs`、现有 JSON/JSONL/CSV、Git 状态和实际文件系统；无法确认的字段统一写为 `unknown`。
- 审计前 Git HEAD：`e91c9246c2c5c7b967e805ceb38229640437839b`
- 审计开始时工作区：clean。

检查结果：

1. 仓库内不存在 `F:\GraduationThesis\rag-campus-policy\experiment_outputs`，因此不存在可与外部目录竞争的项目内实验结果源。
2. 现有实验结果实际位于 `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook`。审计前共 91 个历史文件，未移动、删除或覆盖。
3. `F:\GraduationThesis\experiment_outputs` 是多个课题/数据对象可共用的上层输出容器；本项目的唯一主输出目录确定为：

   `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook`

4. 新规范目录放在主输出目录的 `runs\<experiment_id>`；总账放在主输出目录的 `experiment_registry.csv`；论文表格放在主输出目录的 `thesis_exports\<timestamp>`。
5. 仓库内不存在 `thesis_documents`。外部 `F:\GraduationThesis\thesis_documents\hdu_ies` 存在但为空，当前不承担实验输出、实验总账或论文表格来源职责。
6. `scripts`、`configs`、`src`、`docs`、`tests` 均存在。现有脚本和配置普遍把数据指向 `F:\GraduationThesis\datasets`，把实验结果指向上述外部主输出目录。

## 2. 历史实验登记结论

历史结果已被扫描为 27 条 `legacy=true` 实验记录：

- 检索实验 21 条：5 个检索家族，展开到 Chunk 策略和融合方法。
- RAG 与答案分析 6 条：批量评估、v1/v2/v3 质量、v2 误差分析、v2/v3 对比。

历史记录可由脚本、配置、结果报告和逐题文件部分复现，但没有统一保存 Python/依赖快照、Git commit、工作区干净状态、精确开始时间和随机种子，因此统一标记 `reproducible=partial`，不补造缺失值。

## 3. 实验产物元数据表

表中“路径”是一个产物家族；第 4 节列出家族内每个实际文件。`Top K=10` 表示检索结果保存到前 10 名并计算 `Recall@1/3/5/10`；RAG 使用 `top_k=5`。

| ID | 路径 | 生成脚本 | 实验类型 | Chunk 策略 | 检索方法 | 模型 | Top K | QA 版本 | 指标字段 | 逐题结果 | 可复现 | 缺失参数或版本信息 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENSE-HASH | `dense_retrieval` | `build_dense_retrieval_baseline.py`; `evaluate_dense_retrieval.py` | retrieval | fixed_length; article_based; structure_aware | dense_hash | deterministic_hash_embedding, 384 维；非真实语义模型 | 10 | v1 | Recall@1/3/5/10, MRR@10, average_top1_score, average_relevant_rank, no_hit_count, 分组 Recall@5 | yes，3 个 search JSONL | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子 |
| BM25 | `bm25_retrieval` | `build_bm25_retrieval_baseline.py`; `evaluate_bm25_retrieval.py` | retrieval | fixed_length; article_based; structure_aware | bm25 | cjk_char_bigram，k1=1.5，b=0.75 | 10 | v1 | Recall@1/3/5/10, MRR@10, average_top1_score, average_relevant_rank, no_hit_count, 分组 Recall@5 | yes，3 个 search JSONL | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子 |
| HYBRID-HASH | `hybrid_retrieval` | `evaluate_hybrid_retrieval.py` | retrieval | fixed_length; article_based; structure_aware | hybrid_weighted_score; hybrid_rrf | Dense Hash + BM25 | 10 | v1 | Recall@1/3/5/10, MRR@10, average_top1_score, average_relevant_rank, no_hit_count | yes，6 个方法 JSONL；另有 3 个合并 JSONL | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子；已有 weighted=0.4/0.6、RRF k=60 |
| REAL-DENSE | `dense_retrieval_real_embedding` | `build_real_embedding_dense_retrieval.py`; `evaluate_real_embedding_dense_retrieval.py` | retrieval | fixed_length; article_based; structure_aware | real_dense | sentence_transformers；BAAI/bge-small-zh-v1.5；512 维 | 10 | v1 | Recall@1/3/5/10, MRR@10, average_top1_score, average_relevant_rank, no_hit_count | yes，3 个 search JSONL | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子；保留一次 unavailable manifest |
| REAL-HYBRID | `hybrid_retrieval_real_embedding` | `evaluate_real_embedding_hybrid_retrieval.py` | retrieval | fixed_length; article_based; structure_aware | real_hybrid_weighted_score; real_hybrid_rrf | BAAI/bge-small-zh-v1.5 + BM25 | 10 | v1 | Recall@1/3/5/10, MRR@10, average_top1_score, average_relevant_rank, no_hit_count | yes，6 个方法 JSONL；另有 3 个合并 JSONL | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子；已有 weighted=0.5/0.5、RRF k=60；配置中的 0.6/0.4 和 0.7/0.3 仅为 optional，未发现对应结果 |
| RAG-V1-BATCH | `rag_evaluation` | `evaluate_rag_qa.py` | rag_batch_evaluation | fixed_length | real_hybrid_weighted | 检索为 BAAI/bge-small-zh-v1.5 + BM25；生成是 rule-based extractive | 5 | v1 | answer_non_empty_rate, citation_coverage_rate, average_citation_count, average_answer_chars, source_policy_hit_rate, related_chunk_hit_rate, no_answer/no_citation/warning/fallback count | yes，answers JSONL | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子、报告内数据/Chunk/QA 哈希 |
| RAG-V1-QUALITY | `rag_answer_quality` | `evaluate_rag_answer_quality.py` | rag_answer_quality | fixed_length | real_hybrid_weighted | rule_based_extractive_v1 | 5 | v1 | question_count, average_keyword_overlap_ratio, average_evidence_overlap_ratio, source_policy_hit_rate, related_chunk_hit_rate, auto_quality_distribution 等 | yes，review JSONL/CSV | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子、报告内数据/Chunk/QA 哈希 |
| RAG-V2-QUALITY | `rag_answer_quality_improvement` | `analyze_rag_answer_quality.py`; `evaluate_rag_answer_quality_improvement.py` | rag_answer_quality | fixed_length | real_hybrid_weighted | rule_based_extractive_v2 | 5 | v1 | v1_summary, v2_summary, comparison, overlap、质量分布、improved/worsened/unchanged | yes，v2 answers/review JSONL 和 CSV | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子、统一 config snapshot、报告内数据哈希 |
| RAG-V2-ERROR | `rag_answer_quality_error_analysis` | `analyze_rag_answer_errors_v2.py` | error_analysis | fixed_length | real_hybrid_weighted | rule_based_extractive_v2 | 5 | v1 | needs_review_count, worsened_count, condition/medium count, error_type/question_type/difficulty distribution | yes，3 个逐题人工复核 CSV | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子、config 版本、数据哈希 |
| RAG-V3 | `rag_answer_quality_v3_optimization` | `generate_rag_answers_v3.py`; `review_rag_answer_quality_v3.py` | rag_answer_quality | fixed_length | real_hybrid_weighted | rule_based_extractive_v3 | 5 | v1 | total_questions, average_keyword/evidence_overlap_ratio, auto_quality_distribution, needs_review_count, 分组质量、结构类型分布 | yes，answers/review JSONL 和 CSV | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子、统一 config snapshot、数据哈希 |
| RAG-V2-V3 | `rag_answer_quality_v3_optimization` | `compare_rag_answer_quality_v2_v3.py` | rag_answer_quality_comparison | fixed_length | real_hybrid_weighted | rule_based_extractive_v2_vs_v3 | 5 | v1 | v2/v3 overlap、质量分布、needs_review、improved/worsened/unchanged、类型/难度分组 | yes，comparison JSONL/CSV | partial | Python/依赖、Git commit/dirty、精确开始时间、随机种子、统一 config snapshot、数据哈希 |
| RESULTS-TABLES | `experiment_results_tables` | `export_experiment_results_tables.py` | derived_thesis_table | 混合 | 混合 | unknown | 混合 | v1 | Chunk、检索、RAG 版本、错误类型、默认方案汇总字段 | no；来源文件含逐题结果 | partial | 属于派生表而非独立实验；无独立运行环境/Git 快照 |
| THESIS-SUMMARY | `rag_experiment_summary_for_thesis` | `summarize_rag_experiments_for_thesis.py` | derived_summary | 混合 | 混合 | unknown | 混合 | v1 | data_consistency, chunking/retrieval/RAG v1-v3 summary, limitations | no；来源文件含逐题结果 | partial | 属于派生汇总而非独立实验；无独立运行环境/Git 快照 |
| MANUAL-REVIEW | `manual_review_package` | `build_manual_review_package.py` | manual_review_package | fixed_length | real_hybrid_weighted | v1/v2/v3 规则式回答 | 5 | v1 | 各复核分组数量、priority 数量；CSV 保留人工字段 | yes，6 个逐题 CSV | partial | 人工字段尚未填写；非独立模型实验；无独立运行环境/Git 快照 |

## 4. 实际文件清单

下列 91 个文件是审计前已经存在的历史结果。每个文件继承第 3 节对应 ID 的元数据。

### DENSE-HASH（12）

- `dense_retrieval\dense_embedding_index_summary.json`
- `dense_retrieval\embeddings\article_based_embeddings.jsonl`
- `dense_retrieval\embeddings\fixed_length_embeddings.jsonl`
- `dense_retrieval\embeddings\structure_aware_embeddings.jsonl`
- `dense_retrieval\evaluation\dense_search_evaluation_report.json`
- `dense_retrieval\evaluation\dense_search_evaluation_summary.csv`
- `dense_retrieval\evaluation\dense_search_results_article_based.jsonl`
- `dense_retrieval\evaluation\dense_search_results_fixed_length.jsonl`
- `dense_retrieval\evaluation\dense_search_results_structure_aware.jsonl`
- `dense_retrieval\indexes\article_based_index.json`
- `dense_retrieval\indexes\fixed_length_index.json`
- `dense_retrieval\indexes\structure_aware_index.json`

### BM25（9）

- `bm25_retrieval\bm25_index_summary.json`
- `bm25_retrieval\evaluation\bm25_evaluation_report.json`
- `bm25_retrieval\evaluation\bm25_evaluation_summary.csv`
- `bm25_retrieval\evaluation\bm25_search_results_article_based.jsonl`
- `bm25_retrieval\evaluation\bm25_search_results_fixed_length.jsonl`
- `bm25_retrieval\evaluation\bm25_search_results_structure_aware.jsonl`
- `bm25_retrieval\indexes\article_based_bm25_index.json`
- `bm25_retrieval\indexes\fixed_length_bm25_index.json`
- `bm25_retrieval\indexes\structure_aware_bm25_index.json`

### HYBRID-HASH（11）

- `hybrid_retrieval\evaluation\hybrid_evaluation_report.json`
- `hybrid_retrieval\evaluation\hybrid_evaluation_summary.csv`
- `hybrid_retrieval\evaluation\hybrid_search_results_article_based.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_article_based_rrf.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_article_based_weighted_score.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_fixed_length.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_fixed_length_rrf.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_fixed_length_weighted_score.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_structure_aware.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_structure_aware_rrf.jsonl`
- `hybrid_retrieval\evaluation\hybrid_search_results_structure_aware_weighted_score.jsonl`

### REAL-DENSE（13）

- `dense_retrieval_real_embedding\real_embedding_index_summary.json`
- `dense_retrieval_real_embedding\real_embedding_unavailable_manifest.json`
- `dense_retrieval_real_embedding\embeddings\article_based_real_embedding_embeddings.jsonl`
- `dense_retrieval_real_embedding\embeddings\fixed_length_real_embedding_embeddings.jsonl`
- `dense_retrieval_real_embedding\embeddings\structure_aware_real_embedding_embeddings.jsonl`
- `dense_retrieval_real_embedding\evaluation\real_embedding_dense_evaluation_report.json`
- `dense_retrieval_real_embedding\evaluation\real_embedding_dense_evaluation_summary.csv`
- `dense_retrieval_real_embedding\evaluation\real_embedding_dense_search_results_article_based.jsonl`
- `dense_retrieval_real_embedding\evaluation\real_embedding_dense_search_results_fixed_length.jsonl`
- `dense_retrieval_real_embedding\evaluation\real_embedding_dense_search_results_structure_aware.jsonl`
- `dense_retrieval_real_embedding\indexes\article_based_real_embedding_index.json`
- `dense_retrieval_real_embedding\indexes\fixed_length_real_embedding_index.json`
- `dense_retrieval_real_embedding\indexes\structure_aware_real_embedding_index.json`

### REAL-HYBRID（11）

- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_evaluation_report.json`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_evaluation_summary.csv`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_article_based.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_article_based_rrf.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_article_based_weighted_score.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_fixed_length.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_fixed_length_rrf.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_fixed_length_weighted_score.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_structure_aware.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_structure_aware_rrf.jsonl`
- `hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_structure_aware_weighted_score.jsonl`

### RAG-V1-BATCH（3）

- `rag_evaluation\rag_answers_real_hybrid_weighted_fixed_length.jsonl`
- `rag_evaluation\rag_evaluation_report.json`
- `rag_evaluation\rag_evaluation_summary.csv`

### RAG-V1-QUALITY（3）

- `rag_answer_quality\rag_answer_quality_report.json`
- `rag_answer_quality\rag_answer_quality_review_v1.csv`
- `rag_answer_quality\rag_answer_quality_review_v1.jsonl`

### RAG-V2-QUALITY（5）

- `rag_answer_quality_improvement\needs_review_analysis_v1.csv`
- `rag_answer_quality_improvement\rag_answers_real_hybrid_weighted_fixed_length_v2.jsonl`
- `rag_answer_quality_improvement\rag_answer_quality_improvement_report.json`
- `rag_answer_quality_improvement\rag_answer_quality_review_v2.csv`
- `rag_answer_quality_improvement\rag_answer_quality_review_v2.jsonl`

### RAG-V2-ERROR（4）

- `rag_answer_quality_error_analysis\rag_answer_error_analysis_v2_report.json`
- `rag_answer_quality_error_analysis\rag_v2_condition_medium_focus_cases.csv`
- `rag_answer_quality_error_analysis\rag_v2_needs_review_manual_check.csv`
- `rag_answer_quality_error_analysis\rag_v2_worsened_cases_manual_check.csv`

### RAG-V3 与 RAG-V2-V3（7）

- `rag_answer_quality_v3_optimization\rag_answers_real_hybrid_weighted_fixed_length_v3.jsonl`
- `rag_answer_quality_v3_optimization\rag_answer_quality_comparison_v2_v3.csv`
- `rag_answer_quality_v3_optimization\rag_answer_quality_comparison_v2_v3.jsonl`
- `rag_answer_quality_v3_optimization\rag_answer_quality_comparison_v2_v3_report.json`
- `rag_answer_quality_v3_optimization\rag_answer_quality_review_v3.csv`
- `rag_answer_quality_v3_optimization\rag_answer_quality_review_v3.jsonl`
- `rag_answer_quality_v3_optimization\rag_answer_quality_v3_report.json`

### RESULTS-TABLES（5）

- `experiment_results_tables\chunk_strategy_results.csv`
- `experiment_results_tables\default_strategy_selection.csv`
- `experiment_results_tables\error_type_distribution.csv`
- `experiment_results_tables\experiment_results_summary.json`
- `experiment_results_tables\rag_version_comparison.csv`

### THESIS-SUMMARY（1）

- `rag_experiment_summary_for_thesis\rag_experiment_summary_for_thesis.json`

### MANUAL-REVIEW（7）

- `manual_review_package\manual_review_condition_cases.csv`
- `manual_review_package\manual_review_medium_cases.csv`
- `manual_review_package\manual_review_needs_review_cases.csv`
- `manual_review_package\manual_review_package_summary.json`
- `manual_review_package\manual_review_sample_all_cases.csv`
- `manual_review_package\manual_review_unknown_error_cases.csv`
- `manual_review_package\manual_review_worsened_cases.csv`

## 5. 数据输入与哈希现状

现有检索报告已保存 raw、cleaned、三种 Chunk 和 QA 的 SHA-256。实际 Chunk 和 QA 位于：

- `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks`
- `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset`

RAG v1/v2/v3 的部分后期报告只引用上游结果路径，没有重复保存全部数据哈希。这不影响读取现有数字，但意味着旧 RAG 结果仍缺少完整、单次运行级的 `data_manifest.json`。

## 6. 不应被误解为正式结果的资产

- `dense_retrieval` 的 deterministic hash embedding 是流程基线，不是正式语义模型结果。
- `dense_retrieval_real_embedding\real_embedding_unavailable_manifest.json` 记录一次真实模型不可用事件，不是成功检索指标。
- `retrieval_hybrid_real_embedding.yaml` 中 0.6/0.4、0.7/0.3 是可选参数，不代表这些权重已产生实验结果。
- `manual_review_package` 中人工评分字段为空，不能作为已经完成人工评估的证据。
- v3 是定向优化实验，不替代当前默认的 `real_hybrid_weighted + fixed_length + top_k=5 + RAG v2`。
