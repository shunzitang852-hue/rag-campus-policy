# RAG 答案质量优化报告

## 执行范围

本轮只进行 RAG 答案质量问题分析和规则型答案生成优化。未修改 raw、cleaned、Chunk、QA 测试集文件，未覆盖 Dense、BM25、Hybrid、Real Dense、Real Hybrid 和 v1 RAG 历史结果，未调用大模型 API，未做 Reranker，未做 Web 前端。

## 输出文件

- needs_review 分析：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\needs_review_analysis_v1.csv`
- v2 RAG 回答：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\rag_answers_real_hybrid_weighted_fixed_length_v2.jsonl`
- v2 质量评估 CSV：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\rag_answer_quality_review_v2.csv`
- v2 质量评估 JSONL：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\rag_answer_quality_review_v2.jsonl`
- v1/v2 对比报告：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality_improvement\rag_answer_quality_improvement_report.json`

## 总体指标

| 指标 | v1 | v2 |
| --- | ---: | ---: |
| question_count | 40 | 40 |
| average_keyword_overlap_ratio | 0.5616666666666668 | 0.5925 |
| average_evidence_overlap_ratio | 0.7741666666666667 | 0.8012499999999999 |
| source_policy_hit_rate | 1.0 | 1.0 |
| related_chunk_hit_rate | 1.0 | 1.0 |
| citation_coverage_rate | 1.0 | 1.0 |
| insufficient_count | 0 | 0 |
| needs_review_count | 19 | 17 |

v2 自动质量分布为 `good=23, needs_review=17`，相较 v1 的 `good=21, needs_review=19` 有小幅改善。

## 问题变化

- improved_question_count：8
- worsened_question_count：5
- unchanged_question_count：27

## question_type 分组概览

| question_type | v1 keyword avg | v2 keyword avg | v1 分布 | v2 分布 |
| --- | ---: | ---: | --- | --- |
| comparison | 0.49333333333333335 | 0.6466666666666667 | good=2, needs_review=3 | good=3, needs_review=2 |
| condition | 0.5055555555555555 | 0.37777777777777777 | good=4, needs_review=2 | good=2, needs_review=4 |
| fact_lookup | 0.6666666666666666 | 0.7888888888888889 | good=4, needs_review=2 | good=5, needs_review=1 |
| procedure | 0.5055555555555555 | 0.5777777777777778 | good=3, needs_review=3 | good=4, needs_review=2 |
| qualification | 0.54 | 0.54 | good=2, needs_review=3 | good=2, needs_review=3 |
| responsible_body | 0.5333333333333333 | 0.5444444444444444 | good=3, needs_review=3 | good=3, needs_review=3 |
| time_limit | 0.6722222222222222 | 0.6722222222222222 | good=3, needs_review=3 | good=4, needs_review=2 |

condition 类回退最明显，说明条件类问题后续需要更细的多条件合并规则，而不是继续提高单句关键词权重。

## difficulty 分组概览

| difficulty | v1 keyword avg | v2 keyword avg | v1 分布 | v2 分布 |
| --- | ---: | ---: | --- | --- |
| easy | 0.6233333333333333 | 0.7033333333333334 | good=6, needs_review=4 | good=7, needs_review=3 |
| hard | 0.6071428571428571 | 0.6904761904761905 | good=8, needs_review=6 | good=10, needs_review=4 |
| medium | 0.48333333333333334 | 0.4375 | good=7, needs_review=9 | good=6, needs_review=10 |

medium 难度问题略有回退，后续需要结合具体样例人工复核。

## 结论

本轮 v2 规则优化在整体平均关键词重合率、证据重合率和 needs_review 数量上取得改善，同时保持引用覆盖率、来源制度命中率、相关 chunk 命中率均为 1.0。由于 condition 和 medium 分组仍存在回退，本轮不继续硬调规则，保留当前可解释的轻量优化作为第一轮结果。
