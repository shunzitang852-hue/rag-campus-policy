# RAG 实验阶段性汇总与论文实验章节材料

## 1. 实验阶段概述

本阶段面向本科毕业论文《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》的实验章节，汇总已经完成的数据处理、Chunk 切分、检索策略对比、RAG 回答质量评估、v2 误差分析与 v3 定向优化实验结果。本文档仅整理既有实验结果，不进行新的模型训练，不调用大模型 API，不引入 Reranker，不构建 Web 前端，也不覆盖 Dense、BM25、Hybrid、Real Dense、Real Hybrid、RAG v1、RAG v2 或 RAG v3 的历史输出。

当前实验对象为杭州电子科技大学信息工程学院学生手册相关制度文本，测试集包含 40 个问答样例，覆盖 32 项制度。实验主线可以分为三层：第一层是数据清洗与 Chunk 生成，第二层是检索策略实验，第三层是基于检索结果的规则型 RAG 回答生成与质量评估。

## 2. 数据集与预处理一致性说明

本阶段后续实验保持同一批输入数据不变，以保证 Dense/BM25/Hybrid/RAG 各实验之间具有可比性。已记录的关键 hash 如下：

| 文件类型 | SHA-256 |
| --- | --- |
| raw 文件 | `83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189` |
| cleaned 文件 | `13C309934F6413A02D2D4CAFFD3CFD737FB9316B8C4ED285C4443853FDADC2CB` |
| QA JSONL | `4ECCD47E5FD6FEF7F5BBB77F13FC7646A9EE674961C3BE87A42CA0D7BE2D9CC5` |

三种 Chunk 文件在后续检索与 RAG 实验中保持不变：`fixed_length`、`article_based`、`structure_aware`。既有记录中的 chunk 文件 hash 分别为：`fixed_length=B89EF648A39BAA1E332FA63F6BF76FF6AFDC8B52C8245F233E68999BDDFADC36`，`article_based=8B96C8D166FE05B6443CE35A939AB9CE86FEDE2DBBDCC491FE2514B40EF25803`，`structure_aware=78304E60CA6BCC9E2F7544F66D6059525C449E6A24B1BF4A8D2917297AF06ADF`。因此，后续实验中的 raw / cleaned / chunk / QA hash 均作为固定输入边界，不把指标变化归因于数据重新生成。

## 3. Chunk 策略设计与结果

本阶段比较了三种 Chunk 策略。`fixed_length` 作为长度控制基线，按固定窗口切分；`article_based` 尽量围绕制度条款或文章结构组织片段；`structure_aware` 在结构解析基础上保留层级信息。三种策略均覆盖 32 项制度，唯一结构记录覆盖率均为 1.0，说明制度正文没有因切分策略而丢失。

| 策略 | chunk 数 | 平均长度 | 覆盖率 | 重复覆盖率 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed_length | 244 | 452.594262295082 | 1.0 | 1.0 | 固定长度基线，重复覆盖较高，适合作为稳定检索输入 |
| article_based | 358 | 251.68435754189943 | 1.0 | 0.21330275229357798 | 更贴近条款结构，片段更短 |
| structure_aware | 358 | 251.68435754189943 | 1.0 | 0.21330275229357798 | 保留结构信息，与 article_based 在当前数据上统计结果一致 |

覆盖率按唯一解析结构记录计算，重复覆盖率单独记录，避免用重叠片段字符数直接累加造成覆盖率被高估。

## 4. 检索策略实验汇总

检索实验依次完成 Dense Hash、BM25、Hash Hybrid、Real Dense 和 Real Hybrid。Dense Hash 使用确定性 hash embedding，只能作为流程验证基线，不应被表述为真实语义检索模型。Real Dense 使用 `sentence_transformers` 与 `BAAI/bge-small-zh-v1.5`，embedding 维度为 512。Real Hybrid 在 Real Dense 与 BM25 结果上进行加权融合和 RRF 融合。

| 检索方案 | 策略 | Recall@5 | MRR@10 | 备注 |
| --- | --- | ---: | ---: | --- |
| Dense Hash | fixed_length | 0.900 | 0.8021 | 非真实语义模型，仅作流程基线 |
| Dense Hash | article_based / structure_aware | 0.600 | 0.3947 | 两者当前结果一致 |
| BM25 | fixed_length | 0.950 | 0.8767 | 标准库 BM25，中文分词仍较粗 |
| BM25 | article_based / structure_aware | 0.675 | 0.5779 | 两者当前结果一致 |
| Hash Hybrid weighted | fixed_length | 0.950 | 0.8750 | 融合 Dense Hash 与 BM25 |
| Hash Hybrid weighted | article_based / structure_aware | 0.725 | 0.5590 | 两者当前结果一致 |
| Real Dense | fixed_length | 1.000 | 0.9833 | 使用真实中文 embedding |
| Real Dense | article_based / structure_aware | 0.750 | 0.7136 | 两者当前结果一致 |
| Real Hybrid weighted | fixed_length | 1.000 | 1.0000 | 当前推荐 RAG 检索输入 |
| Real Hybrid weighted | article_based / structure_aware | 0.750 | 0.6562 | 两者当前结果一致 |

从已有检索指标看，`real_hybrid_weighted + fixed_length` 在当前 40 题测试集上达到 Recall@5=1.0 和 MRR@10=1.0，且后续 RAG v1/v2/v3 均围绕该输入组合展开，因此当前推荐将其作为 RAG 输入检索方案。

## 5. RAG 回答生成实验设计

RAG 回答生成阶段使用检索返回的 Chunk 内容进行规则型抽取或模板化回答，不调用大模型 API。生成阶段只使用问题文本和检索片段，`expected_answer` 与 `evidence_text` 仅用于后续自动评价，不参与答案生成。该设计保证了当前实验可离线复现，但也意味着回答能力主要受检索片段质量、规则抽取逻辑和模板设计影响。

当前自动评价指标包括关键词重合率、证据重合率、来源制度命中、相关 chunk 命中、引用覆盖和自动质量标签等。自动评价指标局限性必须明确：这些指标只能作为辅助复核线索，不等于人工评价，也不能直接证明答案在真实问答场景中完全正确。

## 6. RAG v1 基线结果

RAG v1 是基线回答生成版本，基于 `real_hybrid_weighted + fixed_length + top_k=5` 生成规则型答案。v1 质量评估结果如下：

| 指标 | v1 |
| --- | ---: |
| question_count | 40 |
| average_keyword_overlap_ratio | 0.5616666666666668 |
| average_evidence_overlap_ratio | 0.7741666666666667 |
| source_policy_hit_rate | 1.0 |
| related_chunk_hit_rate | 1.0 |
| citation_coverage_rate | 1.0 |
| auto_quality_distribution | good=21, needs_review=19 |
| needs_review_count | 19 |

v1 能够保持来源命中和引用覆盖，但仍有 19 个问题需要复核，说明仅有高检索召回并不等于回答表达已经充分。

## 7. RAG v2 通用优化结果

RAG v2 在不改变检索输入、不覆盖 v1 输出的前提下，对规则型答案生成进行通用质量优化。v1 与 v2 对比如下：

| 指标 | v1 | v2 |
| --- | ---: | ---: |
| average_keyword_overlap_ratio | 0.5616666666666668 | 0.5925 |
| average_evidence_overlap_ratio | 0.7741666666666667 | 0.8012499999999999 |
| good | 21 | 23 |
| needs_review | 19 | 17 |

问题级变化为 improved=8、worsened=5、unchanged=27。整体看，v2 在关键词重合率、证据重合率和 needs_review 数量上均较 v1 有小幅改善。因此，在当前已有自动指标下，v2 整体优于 v1，可作为当前默认 RAG 回答方案。

## 8. RAG v2 误差分析

v2 误差分析读取 v1/v2 质量评估结果和 QA 测试集，生成 needs_review 人工复核表、worsened 样例表和 condition/medium 重点样例表。汇总结果如下：

| 指标 | 数值 |
| --- | ---: |
| total_questions | 40 |
| needs_review_count | 17 |
| worsened_count | 5 |
| condition_case_count | 6 |
| medium_case_count | 16 |

误差类型分布如下：

| error_type | count |
| --- | ---: |
| answer_too_general | 8 |
| condition_not_fully_covered | 5 |
| evidence_missing | 4 |
| evidence_partially_used | 6 |
| unknown | 17 |

误差分析显示，`condition` 类问题的主要风险是条件覆盖不完整，`medium` 难度问题的主要风险是回答过泛或证据使用不充分。这些结论仍是自动辅助分析，人工复核尚未完成，不能表述为最终人工评审结论。

## 9. RAG v3 定向优化实验

RAG v3 基于 v2 误差分析进行定向优化，重点处理 `condition` 类型和 `medium` 难度问题，并加入若干结构化模板。v3 是定向优化实验，不是默认回答方案。v3 输出使用独立目录和 `answer_version=v3` 标识，不改动 v1/v2 历史结果。

v3 总体评估结果如下：

| 指标 | v3 |
| --- | ---: |
| total_questions | 40 |
| average_keyword_overlap_ratio | 0.5458333333333333 |
| average_evidence_overlap_ratio | 0.8720833333333333 |
| good | 20 |
| needs_review | 20 |

v3 提高了整体证据重合率，但关键词重合率低于 v2，needs_review 数量也从 v2 的 17 增至 20，因此不能作为 v2 的全面替代版本。

## 10. v1 / v2 / v3 综合对比

| 指标 | v1 | v2 | v3 |
| --- | ---: | ---: | ---: |
| average_keyword_overlap_ratio | 0.5616666666666668 | 0.5925 | 0.5458333333333333 |
| average_evidence_overlap_ratio | 0.7741666666666667 | 0.8012499999999999 | 0.8720833333333333 |
| good | 21 | 23 | 20 |
| needs_review | 19 | 17 | 20 |

v2 相比 v1：improved=8、worsened=5、unchanged=27。v3 相比 v2：improved=6、worsened=8、unchanged=26。

目标分组中，condition 的关键词重合率由 v2 的 0.37777777777777777 提升到 v3 的 0.39999999999999997，变化为 improved=2、unchanged=4。medium 的关键词重合率由 0.4375 提升到 0.48125，证据重合率由 0.884375 提升到 0.9125，变化为 improved=4、unchanged=10、worsened=2。这说明 v3 对目标分组有局部改善，但整体自动指标不如 v2。

## 11. 当前默认方案选择依据

当前默认 RAG 回答方案建议使用 v2，检索输入建议使用 `real_hybrid_weighted + fixed_length + top_k=5`。

选择依据如下：第一，检索阶段 `real_hybrid_weighted + fixed_length` 在当前测试集上 Recall@5 和 MRR@10 均为 1.0，是已有结果中最稳定的 RAG 输入组合。第二，回答生成阶段 v2 相比 v1 在整体关键词重合率、证据重合率、good 数量和 needs_review 数量上均有改善。第三，v3 虽然对 condition 和 medium 目标分组有改善，但整体 `average_keyword_overlap_ratio` 下降，`needs_review_count` 增加，且 worsened 数量高于 improved 数量。

因此，v3 不替代 v2 默认方案。更稳妥的表述是：v2 是当前默认 RAG 回答方案，v3 是定向优化实验版本，可为后续人工复核和进一步规则设计提供依据。

## 12. 实验结论

已有实验表明，在当前学生手册 40 题 QA 测试集上，固定长度 Chunk 与 Real Hybrid 检索组合表现最稳定，适合作为当前 RAG 输入。RAG 回答生成方面，v2 的通用规则优化相较 v1 有小幅但一致的整体提升，因此可作为当前默认版本。v3 证明定向结构化规则能够改善 condition 和 medium 分组的部分样例，但同时带来整体指标回退，因此应作为实验记录和后续优化依据，而非默认替代方案。

上述结论均基于已有自动评价结果。由于自动评价指标只能作为辅助，最终论文表述中应避免夸大“答案正确率”或“系统已完全可用”，更适合表述为“在规则指标下表现改善”“仍需人工复核确认”。

## 13. 局限性与后续工作

当前局限性主要包括：尚未完成人工复核；自动指标不能充分判断语义正确性和真实可用性；当前回答生成仍是规则型抽取/模板生成；尚未引入 Reranker；尚未使用 LLM Judge；尚未调用真实大模型 API 进行生成式对照；尚未建设 Web 前端。

后续工作建议包括：对 v2/v3 的 needs_review、improved 和 worsened 样例进行人工复核；将本汇总改写为论文实验章节正文；设计 Reranker 对照实验；引入 LLM Judge 或真实大模型生成结果作为扩展评价；在保持历史结果不可覆盖的前提下继续完善检索和回答质量评估。
