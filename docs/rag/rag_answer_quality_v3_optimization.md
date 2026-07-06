# RAG v3 定向答案优化实验记录

## 一、本轮实验目的

本轮实验基于 v2 误差分析结果，尝试在不调用大模型 API、不引入 Reranker、不构建 Web 前端的前提下，对规则型 RAG 答案生成器进行 v3 定向优化。优化重点不是全面替换 v2，而是验证针对 `condition` 类型和 `medium` 难度问题的结构化规则是否能改善局部问题。

本轮输出位于：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_v3_optimization`

## 二、v2 误差分析结论回顾

v2 误差分析指出两个主要问题：

1. `condition` 类型问题容易出现条件覆盖不完整，答案可能只抽取单个条件句，而未覆盖多个限制条件或处理结果。
2. `medium` 难度问题容易出现回答过泛、答案过短或证据没有充分转化为答案的情况。

因此，v3 优先围绕条件类模板、medium 多证据点覆盖、流程类结构化、责任主体明确化和资格条件拆分开展实验。

## 三、v3 定向优化策略

v3 新增独立模块 `src/rag/answer_v3_optimization.py`，不改动 v1/v2 答案生成逻辑。v3 以 `answer_version=v3` 标识，并输出 `answer_structure_type`、`evidence_used`、`improvement_rules_applied` 和 `evidence_insufficient` 等字段。

v3 仍使用 `real_hybrid_weighted + fixed_length + top_k=5`。生成阶段只使用问题文本、问题类型、难度标签和检索返回的 Chunk 内容，不使用 `expected_answer` 或 `evidence_text` 参与答案生成。

## 四、condition 类型问题优化方法

当 `question_type=condition` 或问题中出现“条件、是否可以、能否、需要满足、认定、处分、申请条件、资格”等表达时，v3 启用 `condition_template`。模板包括：

- 适用条件；
- 处理结果；
- 补充依据；
- 依据来源。

该设计用于缓解 v2 中条件句覆盖不完整的问题。自动对比结果显示，condition 分组的关键词重合率由 v2 的 `0.37777777777777777` 提升到 v3 的 `0.39999999999999997`，证据重合率保持 `0.9166666666666666`，变化分布为 `improved=2, unchanged=4`，未出现 worsened。

## 五、medium 难度问题优化方法

对于 `difficulty=medium` 的问题，v3 要求尽量保留两个以上证据点，并将 `medium_min_two_evidence_points` 写入规则记录。该策略避免只返回一句过泛答案。

自动对比结果显示，medium 分组的关键词重合率由 `0.4375` 提升到 `0.48125`，证据重合率由 `0.884375` 提升到 `0.9125`，变化分布为 `improved=4, unchanged=10, worsened=2`。这说明 medium 定向策略有一定效果，但仍存在少量回退样例。

## 六、procedure / responsible_body / qualification 结构化策略

v3 对其他问题类型也增加了结构化输出：

- `procedure_template`：包含办理/处理步骤、关键要求、依据来源；
- `responsible_body_template`：突出责任主体和补充依据；
- `qualification_template`：区分资格条件、不符合条件的情形和补充依据。

自动评估显示，qualification 分组从 v2 的 `0.54` 提升到 v3 的 `0.6599999999999999`，但 responsible_body 分组从 `0.5444444444444444` 降至 `0.39444444444444443`，说明责任主体模板仍需进一步细化，不能据此认为 v3 全面优于 v2。

## 七、v3 自动评估结果

v3 自动质量评估结果如下：

- total_questions：`40`
- average_keyword_overlap_ratio：`0.5458333333333333`
- average_evidence_overlap_ratio：`0.8720833333333333`
- auto_quality_distribution：`good=20, needs_review=20`
- needs_review_count：`20`
- evidence_insufficient_count：`0`
- answer_structure_type_distribution：`condition_template=7, general_template=16, procedure_template=6, qualification_template=5, responsible_body_template=6`

从整体指标看，v3 的证据重合率高于 v2，但关键词重合率和 needs_review 数量不如 v2。

## 八、v2 与 v3 对比结果

v2/v3 对比结果如下：

| 指标 | v2 | v3 |
| --- | ---: | ---: |
| average_keyword_overlap_ratio | 0.5925 | 0.5458333333333333 |
| average_evidence_overlap_ratio | 0.8012499999999999 | 0.8720833333333333 |
| good | 23 | 20 |
| needs_review | 17 | 20 |

问题级变化：

- improved_count：`6`
- worsened_count：`8`
- unchanged_count：`26`

局部分组方面，condition 和 medium 均有提升；但 comparison、responsible_body、easy、hard 等非目标分组存在回退。这说明 v3 是一次定向结构化实验，而不是可直接替代 v2 的全面升级版本。

## 九、效果提升与回退问题

v3 的主要提升体现在：

- condition 分组关键词重合率小幅提升；
- medium 分组关键词和证据重合率均提升；
- qualification 分组提升明显；
- 总体证据重合率提升。

主要回退体现在：

- 总体关键词重合率下降；
- needs_review_count 从 v2 的 `17` 增加到 v3 的 `20`；
- responsible_body 和 comparison 分组回退；
- easy 和 hard 难度问题整体不如 v2。

因此，v3 的实验结论应表述为：结构化模板能改善目标分组的部分问题，但当前规则不适合作为默认全面替代方案。

## 十、自动评价指标局限性

本轮评估仍是自动辅助评价。关键词重合率偏低不必然表示答案错误，证据重合率偏高也不必然表示答案组织更好。v3 的结构化模板会引入“适用条件、处理结果、依据来源”等组织性文本，这可能改变关键词分布，从而影响自动指标。因此，最终答案质量仍需要人工复核或更高层次的评估方法验证。

## 十一、对论文实验章节的意义

本轮实验为论文实验章节提供了一个重要材料：规则型 RAG 优化不一定表现为所有指标同步提升。v3 说明了定向结构化策略可以改善特定问题类型，但也可能牺牲其他类别的自动指标。这有助于论文讨论检索增强问答系统中“可解释规则优化”和“整体指标稳定性”之间的权衡。

## 十二、后续工作

后续建议：

1. 对 v3 improved 和 worsened 样例进行人工复核，区分自动指标回退和真实质量回退；
2. 将 v3 的 condition/medium 规则作为可选策略，而不是直接替换 v2 默认生成器；
3. 针对 responsible_body 分组重新设计责任主体抽取规则；
4. 在论文允许范围内，可进一步设计 Reranker 对照实验或 LLM Judge 辅助评价，但本轮尚未实施；
5. 保持 v2 作为当前更稳健的默认版本，v3 作为定向优化实验版本记录。

截至本报告，未完成正式人工复核，不能伪造人工复核结论。
