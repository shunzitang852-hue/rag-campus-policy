# RAG 答案质量优化设计

## 目标

本轮优化面向第一版 extractive/template-based RAG Demo 的答案质量辅助评估结果，不引入大模型 API、Reranker、Web 前端、新依赖或新模型下载。优化范围限定在规则型答案生成、needs_review 问题分析、v2 批量评估与对比报告。

## 输入与边界

- QA 测试集：`F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`
- v1 RAG 回答：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_answers_real_hybrid_weighted_fixed_length.jsonl`
- v1 质量评估：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality`
- v2 输出目录：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement`

生成阶段只使用问题文本和检索返回的 Chunk 内容；`expected_answer` 与 `evidence_text` 只用于评估与诊断，不进入答案生成。

## 规则调整

`configs/rag_answer_generation.yaml` 集中保存答案生成参数。当前 v2 保留 500 字答案上限，避免为了压缩长度导致关键词覆盖下降；调用方仍可传入更短 `max_answer_chars`，用于单次回答长度控制。

答案生成器新增以下规则：

- 从问题中识别条款锚点，例如“第十条”，并提高同锚点句子的优先级。
- 根据问题文本推断 `condition`、`procedure`、`responsible_body`、`time_limit`、`qualification`、`comparison` 等类型。
- 对不同问题类型设置关键词加权，例如条件类优先“应当、需要、可以、不得、符合、申请、条件”，流程类优先“申请、提交、审核、审批、批准、办理、公示”。
- 同一制度标题的候选句获得轻量加权，减少跨文档片段误选。
- 仍然只抽取检索片段中的原文句子，并保留 citation。

## needs_review 诊断

`src/rag/answer_improvement.py` 提供 `analyze_needs_review` 和 `diagnose_needs_review`，将 v1 的 `needs_review` 样例归入以下标签：

- `missing_key_condition`
- `missing_procedure_step`
- `missing_responsible_body`
- `answer_too_short`
- `evidence_not_fully_used`
- `multi_clause_answer_needed`
- `keyword_metric_too_strict`
- `other`

诊断结果输出到：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\needs_review_analysis_v1.csv`

## 评估流程

```powershell
.\.venv\Scripts\python.exe scripts\analyze_rag_answer_quality.py
.\.venv\Scripts\python.exe scripts\evaluate_rag_answer_quality_improvement.py
```

v2 使用默认组合 `real_hybrid_weighted + fixed_length + top_k=5`，生成回答后再运行规则型质量评估，并输出 v1/v2 对比报告。

## 风险与后续方向

本轮规则优化改善了整体自动指标，但 `condition` 与 `medium` 分组仍有回退。原因可能是条件类标准答案包含多项约束，单纯按句子打分仍可能漏掉分散在不同句中的条件。后续可在不引入大模型的前提下，针对条件类增加“多条件合并”规则，但需要继续通过 v1/v2 对比验证，避免硬调。
