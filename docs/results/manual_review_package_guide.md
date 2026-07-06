# RAG 人工复核辅助包说明

本文档说明 `manual_review_package` 中各 CSV 文件的用途和填写方式。本说明只服务于人工复核，不写论文正文，不伪造人工复核结论。当前所有自动评价结果均为待人工复核材料，不能替代人工判断。

## 1. 人工复核包目的

人工复核包用于把已有 RAG v1/v2/v3、v2 误差分析和实验结果汇总中的样本整理成可填写表格，方便后续逐条判断：

- 生成答案是否正确；
- 检索证据是否足够；
- 自动评价是否误判；
- possible_error_type 是否合理；
- v2 或 v3 的规则调整是否造成真实回退；
- 哪些样本可作为论文中的典型案例。

本轮不做新实验，不调用大模型 API，不做 Reranker，不做 Web 前端，不覆盖历史实验结果。

## 2. CSV 文件与复核对象

| 文件 | 复核对象 | 建议用途 |
| --- | --- | --- |
| `manual_review_needs_review_cases.csv` | v2 中 `auto_quality=needs_review` 的样本 | 优先判断当前默认方案 v2 中哪些答案确实需要修正。 |
| `manual_review_worsened_cases.csv` | v1->v2 或 v2->v3 中自动指标变差的样本 | 判断自动指标回退是否是真实质量回退。 |
| `manual_review_condition_cases.csv` | `question_type=condition` 或 `possible_error_type=condition_not_fully_covered` 的样本 | 检查条件覆盖是否完整。 |
| `manual_review_medium_cases.csv` | `difficulty=medium` 的样本 | 检查中等难度问题是否回答过泛或证据使用不足。 |
| `manual_review_unknown_error_cases.csv` | `possible_error_type=unknown` 的样本 | 归因自动规则无法判断的问题，识别自动评价盲区。 |
| `manual_review_sample_all_cases.csv` | 全部 40 个问题总览 | 按 high / medium / low 优先级进行抽样或全量复核。 |

外部输出目录为：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\manual_review_package`

该目录不提交进 Git。

## 3. 字段说明

### 通用字段

| 字段 | 含义 |
| --- | --- |
| `review_id` | 复核表内部编号，便于沟通和追踪。 |
| `question_id` | QA 样本编号。 |
| `question` | 问题文本。 |
| `question_type` | 问题类型，如 condition、fact_lookup、procedure 等。 |
| `difficulty` | 难度标签。 |
| `generated_answer` / `v2_generated_answer` / `v3_generated_answer` | 对应版本生成的回答。 |
| `auto_quality` / `v2_auto_quality` / `v3_auto_quality` | 自动评价标签，不等于人工结论。 |
| `keyword_overlap_ratio` | 自动关键词重合率。 |
| `evidence_overlap_ratio` | 自动证据重合率。 |
| `retrieved_evidence` | 检索引用片段或已有证据文本。 |
| `possible_error_type` | 自动粗分类结果，只作为复核线索。 |

### 人工填写字段

| 字段 | 建议填写方式 |
| --- | --- |
| `manual_answer_correctness` | 可填 `correct`、`partially_correct`、`incorrect`、`uncertain`。 |
| `manual_evidence_sufficiency` | 可填 `sufficient`、`partially_sufficient`、`insufficient`、`irrelevant`。 |
| `manual_error_type` | 可填 `retrieval_insufficient`、`evidence_partially_used`、`condition_incomplete`、`answer_too_general`、`auto_metric_misjudged`、`other`。 |
| `manual_should_fix` | 可填 `yes`、`no`、`optional`。 |
| `manual_is_real_regression` | 可填 `yes`、`no`、`uncertain`，用于判断 worsened 是否是真实回退。 |
| `manual_condition_complete` | 可填 `yes`、`no`、`uncertain`。 |
| `missing_condition_points` | 如果条件不完整，填写遗漏的条件点。 |
| `manual_answer_too_general` | 可填 `yes`、`no`、`uncertain`。 |
| `manual_should_expand` | 可填 `yes`、`no`、`optional`。 |
| `manual_auto_metric_misjudged` | 可填 `yes`、`no`、`uncertain`。 |
| `manual_notes` / `manual_final_notes` | 填写复核依据、备注或后续处理建议。 |

## 4. 推荐复核顺序

建议按以下顺序复核：

1. 先看 `manual_review_sample_all_cases.csv` 中 `recommended_review_priority=high` 的样本。
2. 再看 `manual_review_needs_review_cases.csv`，确认 v2 默认方案中真正需要修正的答案。
3. 接着看 `manual_review_worsened_cases.csv`，区分真实回退和自动指标误判。
4. 再看 `manual_review_condition_cases.csv` 与 `manual_review_medium_cases.csv`，检查 v3 定向优化依据是否合理。
5. 最后看 `manual_review_unknown_error_cases.csv`，补充新的人工误差类型。

## 5. 如何区分常见问题

### 检索证据不足

表现：`retrieved_evidence` 中没有答案所需制度条款，或引用到了明显不相关的制度。

建议填写：

- `manual_evidence_sufficiency=insufficient`
- `manual_error_type=retrieval_insufficient`
- `manual_should_fix=yes`

### 证据足够但回答使用不充分

表现：证据片段中包含答案要点，但生成答案只使用其中一部分，遗漏了关键句或关键条件。

建议填写：

- `manual_evidence_sufficiency=sufficient`
- `manual_error_type=evidence_partially_used`
- `manual_should_fix=yes`

### 条件覆盖不完整

表现：问题询问“条件、情形、资格、能否、是否可以”等，答案只列出部分条件，遗漏限制条件、例外情形或审核前提。

建议填写：

- `manual_condition_complete=no`
- `manual_error_type=condition_incomplete`
- 在 `missing_condition_points` 中列出遗漏点。

### 回答过泛

表现：答案方向大致正确，但没有具体条款、具体要求或可执行信息。

建议填写：

- `manual_answer_too_general=yes`
- `manual_error_type=answer_too_general`
- `manual_should_expand=yes`

### 自动评价误判

表现：关键词重合率或证据重合率偏低，但人工判断答案语义正确；或指标偏高但答案实际不完整。

建议填写：

- `manual_auto_metric_misjudged=yes`
- `manual_error_type=auto_metric_misjudged`
- 在 `manual_notes` 中说明误判原因。

## 6. 可用于论文的内容

在人工复核前，可用于论文的内容包括：

- 自动评价指标的统计结果；
- v1/v2/v3 的自动指标对比；
- needs_review、worsened、unknown 的数量；
- 自动误差分析中提出的候选问题类型；
- 人工复核包的设计方法和字段设计。

这些内容必须明确标注为“自动评价结果”或“人工复核辅助材料”。

## 7. 必须等人工复核后才能写进论文的内容

以下内容必须等待人工填写 CSV 后才能作为论文结论：

- 某个答案是否最终正确；
- 某个 worsened 是否是真实质量回退；
- 某个 possible_error_type 是否最终成立；
- v3 对具体样本是否真的优于 v2；
- 自动评价是否误判的比例；
- 典型案例的最终人工归因。

## 8. 重要边界

- 当前自动评价不能替代人工判断。
- 当前 CSV 中人工字段均为空，不代表已完成评审。
- 不得把 `needs_review` 自动标签直接等同于“答案错误”。
- 不得把 `good` 自动标签直接等同于“人工正确”。
- 不得把 v3 的局部分组提升表述为全面替代 v2。
