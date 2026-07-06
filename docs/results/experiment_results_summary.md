# 实验结果摘要

本摘要只整理当前已有实验结果，不写论文正文，不生成论文章节草稿。所有结论均基于当前已保存的自动评价结果和外部实验报告；人工复核尚未完成。

## 数据与 Chunk

- raw 文件：`F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt`，SHA-256 为 `83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189`。
- cleaned 文件：`F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt`，SHA-256 为 `13C309934F6413A02D2D4CAFFD3CFD737FB9316B8C4ED285C4443853FDADC2CB`。
- QA JSONL hash：`4ECCD47E5FD6FEF7F5BBB77F13FC7646A9EE674961C3BE87A42CA0D7BE2D9CC5`。
- Chunk hash 在后续 Dense/BM25/Hybrid/Real Dense/Real Hybrid/RAG 实验中保持不变。
- fixed_length：244 chunks，平均 452.59 字，覆盖率 1.0，重复覆盖率 1.0，未覆盖结构数 0，空 chunk 0，重复 chunk 0。
- article_based：358 chunks，平均 251.68 字，覆盖率 1.0，重复覆盖率 0.2133，未覆盖结构数 0，空 chunk 0，重复 chunk 0。
- structure_aware：358 chunks，平均 251.68 字，覆盖率 1.0，重复覆盖率 0.2133，未覆盖结构数 0，空 chunk 0，重复 chunk 0。

## 检索结果

- Dense Hash 仅作为流程基线，不是真实语义模型；fixed_length 下 Recall@5=0.900，MRR@10=0.8021。
- BM25 在 fixed_length 下 Recall@5=0.950，MRR@10=0.8767。
- Hash Hybrid weighted 在 fixed_length 下 Recall@5=0.950，MRR@10=0.8750。
- Real Dense 使用 `BAAI/bge-small-zh-v1.5`，fixed_length 下 Recall@5=1.000，MRR@10=0.9833。
- Real Hybrid weighted 在 fixed_length 下 Recall@5=1.000，MRR@10=1.0000，是当前推荐 RAG 检索输入。

## RAG v1 / v2

- RAG v1 输入检索方案：`real_hybrid_weighted + fixed_length + top_k=5`。
- v1 自动评价：question_count=40，average_keyword_overlap_ratio=0.5616666666666668，average_evidence_overlap_ratio=0.7741666666666667，auto_quality_distribution 为 good=21、needs_review=19。
- RAG v2 输入检索方案同 v1。
- v2 自动评价：question_count=40，average_keyword_overlap_ratio=0.5925，average_evidence_overlap_ratio=0.8012499999999999，auto_quality_distribution 为 good=23、needs_review=17。
- v1 -> v2：improved / worsened / unchanged 为 8 / 5 / 27。
- 结论：在当前自动评价指标下，v2 整体优于 v1，可作为当前默认 RAG 回答方案，但仍需人工复核。

## v2 误差分析

- total_questions=40，needs_review_count=17，worsened_count=5。
- condition_case_count=6，medium_case_count=16。
- error_type_distribution：answer_too_general=8，condition_not_fully_covered=5，evidence_missing=4，evidence_partially_used=6，unknown=17。
- condition 摘要：6 条，其中 condition_not_fully_covered=5，人工复核状态为“待人工复核”。
- medium 摘要：16 条，其中 answer_too_general=5、condition_not_fully_covered=5，人工复核状态为“待人工复核”。
- 结论：condition 条件覆盖不足、medium 难度回答过泛，是后续优化重点。

## RAG v3 与 v2 对比

- v3 输入检索方案：`real_hybrid_weighted + fixed_length + top_k=5`。
- v3 自动评价：total_questions=40，average_keyword_overlap_ratio=0.5458333333333333，average_evidence_overlap_ratio=0.8720833333333333，auto_quality_distribution 为 good=20、needs_review=20，evidence_insufficient_count=0。
- v2 -> v3：keyword overlap 为 0.5925 -> 0.5458333333333333，evidence overlap 为 0.8012499999999999 -> 0.8720833333333333，needs_review 为 17 -> 20，improved / worsened / unchanged 为 6 / 8 / 26。
- condition 分组：关键词 0.37777777777777777 -> 0.39999999999999997，improved=2，unchanged=4。
- medium 分组：关键词 0.4375 -> 0.48125，证据 0.884375 -> 0.9125，improved=4，unchanged=10，worsened=2。
- 结论：v3 对 condition / medium 目标分组有改善，但整体自动指标不如 v2，因此 v3 是定向优化实验版本，不替代 v2 默认方案。

## 当前默认方案

当前默认实验方案为：`real_hybrid_weighted + fixed_length + top_k=5 + RAG v2`。

选择依据：检索端 Real Hybrid weighted + fixed_length 的 Recall@5=1.0、MRR@10=1.0；回答端 v2 相比 v1 整体自动指标改善，并且比 v3 更适合作为默认稳定版本。

## 边界与后续

- 自动评价指标只能作为辅助，不等于人工评价。
- 当前没有调用大模型 API。
- 当前没有做 Reranker。
- 当前没有做 Web 前端。
- 下一步由人工基于这些结果撰写论文正文，或继续进行人工复核 / Reranker / LLM Judge 扩展实验。
