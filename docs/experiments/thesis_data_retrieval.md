# 论文实验数据调取与导出

## 数据来源规则

论文表格只能读取：

1. `experiment_registry.csv` 中已登记的实验；
2. 规范运行目录的 `metrics.json`；
3. `legacy=true` 行明确指向的旧报告。

缺失值留空，并写入 `missing_data_report.md`。脚本不估算、不插值、不根据配置中的可选参数制造结果。

## 自动导出

```powershell
.\.venv\Scripts\python.exe scripts\export_thesis_tables.py
```

脚本先重建总账，再在以下位置创建全新的时间戳目录：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\thesis_exports\<timestamp>`

每次导出均创建新目录；目标目录非空时拒绝覆盖。

生成五组 CSV 和 Markdown：

- `chunk_strategy_comparison.csv/.md`
- `retrieval_method_comparison.csv/.md`
- `hybrid_weight_comparison.csv/.md`
- `rag_version_comparison.csv/.md`
- `error_type_statistics.csv/.md`

同时生成：

- `missing_data_report.md`
- `thesis_tables_manifest.json`

指定输出目录：

```powershell
.\.venv\Scripts\python.exe scripts\export_thesis_tables.py `
  --output-dir F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\thesis_exports\manual_20260720
```

该目录必须不存在或为空。

## 从逐题结果开展误差分析

先查实验详情：

```powershell
.\.venv\Scripts\python.exe scripts\list_experiments.py `
  --experiment-id legacy-rag-v2-error-analysis
```

总账中的 `per_question_results_path` 指向逐题 JSONL 或 CSV。分析时：

1. 以 `question_id` 连接 QA、检索结果和回答质量记录；
2. 使用 `question_type`、`difficulty`、`chunk_strategy`、`retrieval_method` 分组；
3. 保留 `auto_quality_label` 与人工字段的区别；
4. 人工字段为空时只能报告“待复核”，不能报告人工正确率；
5. 错误类型统计优先读取持久化的 `error_type_distribution`，不从论文文字反推。

现有人工复核入口：

- `rag_answer_quality_error_analysis\rag_v2_needs_review_manual_check.csv`
- `rag_answer_quality_error_analysis\rag_v2_worsened_cases_manual_check.csv`
- `rag_answer_quality_error_analysis\rag_v2_condition_medium_focus_cases.csv`
- `manual_review_package\manual_review_sample_all_cases.csv`

## 追溯 Git 与数据哈希

规范实验：

- `git_commit.txt`：运行时 commit；
- `run.log`：运行开始时工作区是否干净；
- `environment.json`：Python、平台和依赖版本；
- `data_manifest.json`：数据、Chunk、QA 路径和 SHA-256；
- `config.json`：算法参数和随机种子。

Legacy 实验：

- 总账的 `generation_script`、`config_path`、`metrics_path`、`per_question_results_path` 用于路径追溯；
- 检索报告内已有 raw、cleaned、Chunk、QA 哈希；
- 旧结果没有单次运行级 Git/环境快照，相关字段保持 `unknown`。

## 使用边界

- Dense Hash 是流程基线，不作为真实语义模型结论。
- 自动关键词/证据重合不等于人工正确性。
- v3 是定向优化实验，不替代 v2 默认方案。
- 当前默认方案表述保持为：`real_hybrid_weighted + fixed_length + top_k=5 + RAG v2`。
