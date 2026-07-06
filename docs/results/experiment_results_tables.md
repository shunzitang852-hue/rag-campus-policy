# 实验结果汇总表

本文档集中整理当前已有实验数据、指标和结论依据，只作为论文写作查阅表，不写论文正文，不生成论文章节草稿。

## 表1：数据版本与 hash 校验表

| 数据项 | 路径 | SHA-256 | 后续实验是否保持不变 |
| --- | --- | --- | --- |
| raw 文件 | `F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt` | `83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189` | 是 |
| cleaned 文件 | `F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt` | `13C309934F6413A02D2D4CAFFD3CFD737FB9316B8C4ED285C4443853FDADC2CB` | 是 |
| QA JSONL | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl` | `4ECCD47E5FD6FEF7F5BBB77F13FC7646A9EE674961C3BE87A42CA0D7BE2D9CC5` | 是 |
| fixed_length chunk | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_fixed_length.jsonl` | `B89EF648A39BAA1E332FA63F6BF76FF6AFDC8B52C8245F233E68999BDDFADC36` | 是 |
| article_based chunk | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_article_based.jsonl` | `8B96C8D166FE05B6443CE35A939AB9CE86FEDE2DBBDCC491FE2514B40EF25803` | 是 |
| structure_aware chunk | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_structure_aware.jsonl` | `78304E60CA6BCC9E2F7544F66D6059525C449E6A24B1BF4A8D2917297AF06ADF` | 是 |

## 表2：Chunk 策略结果对比表

| 策略 | chunk 数量 | 平均长度 | 覆盖率 | 重复覆盖率 | 未覆盖结构数 | 空 chunk | 重复 chunk | 特点和适用性 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed_length | 244 | 452.594262295082 | 1.0 | 1.0 | 0 | 0 | 0 | 固定窗口基线；当前检索表现最稳，适合作为默认 RAG 输入。 |
| article_based | 358 | 251.68435754189943 | 1.0 | 0.21330275229357798 | 0 | 0 | 0 | 更贴近条款结构，片段更短；适合结构化制度阅读对比。 |
| structure_aware | 358 | 251.68435754189943 | 1.0 | 0.21330275229357798 | 0 | 0 | 0 | 保留解析结构信息；适合后续结构规则扩展。 |

## 表3：检索策略结果汇总表

| 检索方案 | chunk 策略 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense Hash | fixed_length | 0.725 | 0.875 | 0.900 | 0.925 | 0.8021 | 流程基线，不是真实语义模型。 |
| Dense Hash | article_based | 0.250 | 0.500 | 0.600 | 0.750 | 0.3947 | 流程基线。 |
| Dense Hash | structure_aware | 0.250 | 0.500 | 0.600 | 0.750 | 0.3947 | 流程基线。 |
| BM25 | fixed_length | 0.800 | 0.950 | 0.950 | 1.000 | 0.8767 | 标准库 BM25。 |
| BM25 | article_based | 0.525 | 0.625 | 0.675 | 0.700 | 0.5779 | 中文切词能力有限。 |
| BM25 | structure_aware | 0.525 | 0.625 | 0.675 | 0.700 | 0.5779 | 中文切词能力有限。 |
| Hybrid weighted | fixed_length | 0.800 | 0.950 | 0.950 | 0.950 | 0.8750 | 融合 Dense Hash 与 BM25。 |
| Hybrid weighted | article_based | 0.450 | 0.650 | 0.725 | 0.750 | 0.5590 | 融合 Dense Hash 与 BM25。 |
| Hybrid weighted | structure_aware | 0.450 | 0.650 | 0.725 | 0.750 | 0.5590 | 融合 Dense Hash 与 BM25。 |
| Real Dense | fixed_length | 0.975 | 1.000 | 1.000 | 1.000 | 0.9833 | 使用 `BAAI/bge-small-zh-v1.5`。 |
| Real Dense | article_based | 0.675 | 0.750 | 0.750 | 0.800 | 0.7136 | 使用真实中文 embedding。 |
| Real Dense | structure_aware | 0.675 | 0.750 | 0.750 | 0.800 | 0.7136 | 使用真实中文 embedding。 |
| Real Hybrid weighted | fixed_length | 1.000 | 1.000 | 1.000 | 1.000 | 1.0000 | 当前推荐 RAG 检索输入。 |
| Real Hybrid weighted | article_based | 0.600 | 0.725 | 0.750 | 0.750 | 0.6562 | 目标策略下不如 fixed_length。 |
| Real Hybrid weighted | structure_aware | 0.600 | 0.725 | 0.750 | 0.750 | 0.6562 | 目标策略下不如 fixed_length。 |

当前推荐用于 RAG 输入的方案：`real_hybrid_weighted + fixed_length + top_k=5`。推荐依据：该组合在当前结果中 Recall@5=1.0、MRR@10=1.0，且 RAG v1/v2/v3 均基于该检索输入进行回答生成或对比。

## 表4：RAG v1 与 v2 对比表

| 指标 | v1 | v2 | 变化 |
| --- | ---: | ---: | --- |
| 输入检索方案 | `real_hybrid_weighted + fixed_length + top_k=5` | `real_hybrid_weighted + fixed_length + top_k=5` | 不变 |
| question_count | 40 | 40 | 不变 |
| average_keyword_overlap_ratio | 0.5616666666666668 | 0.5925 | 0.5616666666666668 -> 0.5925 |
| average_evidence_overlap_ratio | 0.7741666666666667 | 0.8012499999999999 | 0.7741666666666667 -> 0.8012499999999999 |
| auto_quality_distribution | good=21, needs_review=19 | good=23, needs_review=17 | good +2，needs_review -2 |
| needs_review_count | 19 | 17 | 19 -> 17 |
| improved / worsened / unchanged | 未在当前结果中找到 | 8 / 5 / 27 | 自动评价对比 |
| 输出文件路径 | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement` | v2 独立输出 |

主要结论：v2 在整体自动评价指标上优于 v1，可作为当前默认 RAG 回答方案，但仍需人工复核。

## 表5：v2 误差类型分布表

| error_type | count |
| --- | ---: |
| answer_too_general | 8 |
| condition_not_fully_covered | 5 |
| evidence_missing | 4 |
| evidence_partially_used | 6 |
| unknown | 17 |

v2 误差分析补充指标：total_questions=40，needs_review_count=17，worsened_count=5，condition_case_count=6，medium_case_count=16。condition 摘要：6 条，其中 condition_not_fully_covered=5，人工复核状态为“待人工复核”。medium 摘要：16 条，其中 answer_too_general=5、condition_not_fully_covered=5，人工复核状态为“待人工复核”。

## 表6：RAG v2 与 v3 对比表

| 指标 | v2 | v3 | 变化 |
| --- | ---: | ---: | --- |
| 输入检索方案 | `real_hybrid_weighted + fixed_length + top_k=5` | `real_hybrid_weighted + fixed_length + top_k=5` | 不变 |
| question_count | 40 | 40 | 不变 |
| average_keyword_overlap_ratio | 0.5925 | 0.5458333333333333 | 0.5925 -> 0.5458333333333333 |
| average_evidence_overlap_ratio | 0.8012499999999999 | 0.8720833333333333 | 0.8012499999999999 -> 0.8720833333333333 |
| auto_quality_distribution | good=23, needs_review=17 | good=20, needs_review=20 | good -3，needs_review +3 |
| needs_review_count | 17 | 20 | 17 -> 20 |
| evidence_insufficient_count | 未在当前结果中找到 | 0 | v3 无 evidence_insufficient |
| improved / worsened / unchanged | 未在当前结果中找到 | 6 / 8 / 26 | 自动评价对比 |
| 输出文件路径 | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement` | `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_v3_optimization` | v3 独立输出 |

主要结论：v3 对目标分组有局部改善，但整体自动指标不如 v2，因此 v3 不替代 v2 默认方案。

## 表7：condition / medium 目标分组对比表

| 分组 | v2 keyword | v3 keyword | v2 evidence | v3 evidence | improved | worsened | unchanged | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| condition | 0.37777777777777777 | 0.39999999999999997 | 0.9166666666666666 | 0.9166666666666666 | 2 | 0 | 4 | 目标分组关键词指标小幅改善。 |
| medium | 0.4375 | 0.48125 | 0.884375 | 0.9125 | 4 | 2 | 10 | 目标分组关键词与证据指标均改善，但仍有 worsened。 |

## 表8：当前默认方案选择依据表

| 组件 | 当前选择 | 依据 | 是否默认 | 是否需人工复核 |
| --- | --- | --- | --- | --- |
| 检索输入 | `real_hybrid_weighted + fixed_length + top_k=5` | Real Hybrid weighted + fixed_length 的 Recall@5=1.0、MRR@10=1.0；RAG v1/v2/v3 均基于该组合。 | 是 | 否 |
| 回答版本 | RAG v2 | v2 相比 v1 关键词重合率 0.5616666666666668 -> 0.5925，证据重合率 0.7741666666666667 -> 0.8012499999999999，needs_review 19 -> 17。 | 是 | 是 |
| v3 角色 | 定向优化实验版本 | v3 改善 condition/medium 目标分组，但整体 keyword overlap 和 needs_review 不如 v2。 | 否 | 是 |
| 总体默认方案 | `real_hybrid_weighted + fixed_length + top_k=5 + RAG v2` | 检索端最稳，回答端 v2 整体自动指标优于 v1 且稳于 v3。 | 是 | 是 |

## 表9：实验局限性与后续验证需求表

| 局限性 / 后续需求 | 当前状态 | 说明 |
| --- | --- | --- |
| 自动评价指标局限 | 存在 | 关键词/证据重合率只能作为辅助，不等于人工评价。 |
| 人工复核 | 待完成 | needs_review、worsened、unknown 需要人工核验。 |
| Reranker | 未实施 | 当前检索未加入重排序模型。 |
| LLM Judge | 未实施 | 尚未用大模型裁判做质量对照。 |
| 大模型 API 生成答案 | 未调用 | 当前回答为规则型抽取/模板生成。 |
| Web 前端 | 未实施 | 本轮不做系统前端。 |
| v3 默认替代 | 不采用 | v3 是定向优化实验版本，不替代 v2 默认方案。 |
