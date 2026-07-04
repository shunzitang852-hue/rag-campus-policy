# BM25 检索基线报告

## 一、实验范围

本阶段完成 BM25 关键词检索基线。当前实现使用标准库 BM25 和 `cjk_char_bigram` 分词策略。

本阶段没有执行 Hybrid Search、Reranker 或 RAG，也没有重新生成 Embedding 或修改 Dense Search 结果。

## 二、输出文件

BM25 索引：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\indexes\fixed_length_bm25_index.json
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\indexes\article_based_bm25_index.json
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\indexes\structure_aware_bm25_index.json
```

检索结果：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation\bm25_search_results_fixed_length.jsonl
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation\bm25_search_results_article_based.jsonl
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation\bm25_search_results_structure_aware.jsonl
```

评估报告：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation\bm25_evaluation_report.json
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation\bm25_evaluation_summary.csv
```

## 三、总体指标

| strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | no_hit_count |
| -- | --: | --: | --: | --: | --: | --: |
| fixed_length | 0.800 | 0.950 | 0.950 | 1.000 | 0.8767 | 0 |
| article_based | 0.525 | 0.625 | 0.675 | 0.700 | 0.5779 | 12 |
| structure_aware | 0.525 | 0.625 | 0.675 | 0.700 | 0.5779 | 12 |

## 四、分组 Recall@5

question_type 分组：

| strategy | fact_lookup | condition | procedure | responsible_body | time_limit | qualification | comparison |
| -- | --: | --: | --: | --: | --: | --: | --: |
| fixed_length | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 |
| article_based | 0.833 | 0.500 | 0.500 | 0.833 | 0.667 | 0.600 | 0.800 |
| structure_aware | 0.833 | 0.500 | 0.500 | 0.833 | 0.667 | 0.600 | 0.800 |

difficulty 分组：

| strategy | easy | medium | hard |
| -- | --: | --: | --: |
| fixed_length | 1.000 | 0.875 | 1.000 |
| article_based | 0.800 | 0.5625 | 0.7143 |
| structure_aware | 0.800 | 0.5625 | 0.7143 |

## 五、与 Dense Hash 基线简要对比

| strategy | BM25 Recall@5 | Dense Hash Recall@5 | BM25 MRR@10 | Dense Hash MRR@10 |
| -- | --: | --: | --: | --: |
| fixed_length | 0.950 | 0.900 | 0.8767 | 0.8021 |
| article_based | 0.675 | 0.600 | 0.5779 | 0.3947 |
| structure_aware | 0.675 | 0.600 | 0.5779 | 0.3947 |

说明：Dense Hash 不是正式语义 embedding，因此这里只做流程级对比，不作为最终模型优劣结论。

## 六、边界确认

raw、cleaned、Chunk、QA 测试集和 Dense Search 输出均未在本阶段修改。本阶段未生成 Hybrid Search、Reranker 或 RAG 产物。
