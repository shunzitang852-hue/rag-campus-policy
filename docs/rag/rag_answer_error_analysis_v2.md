# RAG v2 答案误差分析与人工复核辅助报告

## 一、本轮实验目的

本轮实验进入 RAG 答案生成质量的深化分析阶段，目标不是继续扩大数据准备工作，而是围绕 RAG v2 的质量评估结果，识别仍需人工复核的问题、v1 到 v2 变差的问题，以及在 v2 中表现相对薄弱的 `condition` 类型和 `medium` 难度问题。该分析用于形成后续 v3 定向优化依据，并为论文实验章节提供可解释的误差归因材料。

本轮不调用大模型 API，不使用 Reranker，不构建 Web 前端，不修改 raw、cleaned、Chunk、QA 测试集，也不覆盖已有 v1、v2 实验结果。

## 二、数据来源

本轮分析读取以下既有结果：

- v1 质量评估：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality\rag_answer_quality_review_v1.jsonl`
- v2 质量评估：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\rag_answer_quality_review_v2.jsonl`
- QA 测试集：`F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`

新增输出目录为：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_error_analysis`

其中包含 needs_review 人工复核表、v1 到 v2 变差样例人工复核表、condition 与 medium 重点问题表，以及 JSON 汇总报告。

## 三、v2 整体表现回顾

上一轮 v2 规则优化后，整体自动评价指标相较 v1 有小幅提升：

- `average_keyword_overlap_ratio`：由 `0.5616666666666668` 提升到 `0.5925`
- `average_evidence_overlap_ratio`：由 `0.7741666666666667` 提升到 `0.8012499999999999`
- 自动质量分布：由 `good=21, needs_review=19` 变为 `good=23, needs_review=17`
- `needs_review_count`：由 `19` 降至 `17`

这说明 v2 的规则优化对整体样本有效，但提升幅度有限，且并非所有问题类型均改善。因此，v2 不能被表述为“全面解决答案质量问题”，更适合作为从通用规则优化进入定向误差修正的中间版本。

## 四、needs_review 问题分析

v2 中仍有 `17` 个问题被自动评价标记为 `needs_review`。这些问题已经输出到：

`rag_v2_needs_review_manual_check.csv`

该表保留问题、标准答案、证据文本、生成答案、关键词重合率、证据重合率、v1/v2 自动质量、变化类型和自动粗分类结果，同时预留人工复核字段。当前人工复核尚未完成，相关字段保持为空，状态记为“待人工复核”。

自动粗分类显示，v2 的问题主要集中在答案表达过泛、证据使用不充分、条件覆盖不完整以及部分无法由规则直接判断的样例。由于这些分类仅基于规则信号，不能替代人工正确性判断。

## 五、worsened 问题分析

从 v1 到 v2，共有 `5` 个问题被标记为 `worsened`。这些问题输出到：

`rag_v2_worsened_cases_manual_check.csv`

这类样例对后续 v3 特别重要，因为它们说明 v2 的规则加权虽然改善了总体平均指标，但也可能改变部分问题的句子选择顺序，导致个别问题的关键词覆盖下降。v3 优化时应把这些 worsened 样例作为回归检查样例，避免为了提升某一类问题而牺牲已有有效样例。

## 六、condition 类型问题回退原因分析

v2 中 `condition` 类型共有 `6` 个问题，其中自动质量分布为 `good=2, needs_review=4`。自动误差粗分类中，`condition_not_fully_covered` 为 `5` 个，说明条件类问题的主要风险不是检索完全失败，而是答案未能完整覆盖多个条件或约束。

条件类问题往往需要同时回答“适用情形、限制条件、申请要求、审核前提”等多个要素。当前 v2 仍以句子级抽取和关键词加权为主，可能优先选中包含条件词的单句，但未能把分散在多个句子中的条件完整合并。因此，condition 类型是 v3 的首要优化对象。

人工复核结论目前尚未形成，以上分析属于自动辅助归因，状态为“待人工复核”。

## 七、medium 难度问题回退原因分析

v2 中 `medium` 难度共有 `16` 个问题，其中自动质量分布为 `good=6, needs_review=10`，变化类型分布为 `improved=3, unchanged=11, worsened=2`。自动误差类型中，`answer_too_general` 和 `condition_not_fully_covered` 各占 `5` 个，是 medium 样例的主要关注点。

medium 难度问题通常比 easy 问题包含更多限定条件，但又未达到 hard 问题那种明确多跳或复杂比较形式，因此容易出现答案“方向正确但不够完整”的情况。这类问题适合作为 v3 中答案完整性规则的重点验证集合。

## 八、自动评价指标的局限性

当前质量评估主要依赖关键词重合率、证据重合率、来源制度命中和相关 chunk 命中等规则指标。这些指标具备可复现和可解释的优势，但存在以下局限：

- 关键词重合不等同于语义正确，可能低估同义表达或概括性表达。
- 证据覆盖率高不代表答案组织完整，尤其是条件类和流程类问题。
- 自动指标无法判断答案是否适合真实学生问答场景。
- 规则分类只能提供人工复核线索，不能替代最终人工判断。

因此，本轮生成的人工复核表只作为辅助材料，不能伪装为已完成的人工评审结论。

## 九、v3 定向优化方向

v3 应优先围绕以下方向展开：

1. condition 类问题：增加多条件合并和条件完整性检查，避免只抽取单个条件句。
2. medium 难度问题：单独建立复核集合，区分答案遗漏、表达过泛和指标偏严格。
3. v2 worsened 问题：建立回归样例，v3 每次规则调整后单独比较。
4. 证据覆盖率高但仍 needs_review 的问题：判断是自动评价局限，还是答案组织不充分。
5. 答案过短或条件表达不完整的问题：在不引入生成式大模型的前提下，增加候选句补全规则。

这些方向仍应保持 extractive/template-based 的边界，即答案只能来自检索片段，不得引入无来源内容。

## 十、对论文实验章节的意义

本轮误差分析把 RAG v2 的整体提升与局部回退同时记录下来，有助于论文实验章节呈现更真实的系统迭代过程。相比只报告平均指标，误差分析能够说明不同问题类型、不同难度层级下的系统表现差异，并为后续 v3 优化提供明确依据。

从论文写作角度，本轮材料可用于支撑以下内容：

- RAG 答案生成模块的规则优化过程；
- 自动评价指标与人工复核之间的关系；
- condition 与 medium 样例的误差归因；
- 后续优化不是任意调参，而是基于可复现误差分析的定向改进。

截至本报告，人工复核尚未完成，相关结论均应表述为“自动辅助分析结果，待人工复核”。
