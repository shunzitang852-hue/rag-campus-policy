# Dense Search 检索基线报告

## 一、实验性质

本阶段完成 Embedding 与 Dense Search 检索基线流程验证。

当前 embedding 方法为 `deterministic_hash_embedding`，不是正式语义 embedding。当前结果不能作为论文最终语义检索实验结论，只能作为可复现的流程基线。

## 二、输出文件

Embedding 输出：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\embeddings\fixed_length_embeddings.jsonl
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\embeddings\article_based_embeddings.jsonl
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\embeddings\structure_aware_embeddings.jsonl
```

索引输出：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\indexes\fixed_length_index.json
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\indexes\article_based_index.json
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\indexes\structure_aware_index.json
```

检索结果：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\evaluation\dense_search_results_fixed_length.jsonl
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\evaluation\dense_search_results_article_based.jsonl
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\evaluation\dense_search_results_structure_aware.jsonl
```

评估报告：

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\evaluation\dense_search_evaluation_report.json
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\evaluation\dense_search_evaluation_summary.csv
```

## 三、总体指标

| strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | no_hit_count |
| -- | --: | --: | --: | --: | --: | --: |
| fixed_length | 0.725 | 0.875 | 0.900 | 0.925 | 0.8021 | 3 |
| article_based | 0.250 | 0.500 | 0.600 | 0.750 | 0.3947 | 10 |
| structure_aware | 0.250 | 0.500 | 0.600 | 0.750 | 0.3947 | 10 |

## 四、分组 Recall@5

question_type 分组：

| strategy | fact_lookup | condition | procedure | responsible_body | time_limit | qualification | comparison |
| -- | --: | --: | --: | --: | --: | --: | --: |
| fixed_length | 1.000 | 0.667 | 0.833 | 1.000 | 1.000 | 0.800 | 1.000 |
| article_based | 1.000 | 0.500 | 0.500 | 0.500 | 0.667 | 0.400 | 0.600 |
| structure_aware | 1.000 | 0.500 | 0.500 | 0.500 | 0.667 | 0.400 | 0.600 |

difficulty 分组：

| strategy | easy | medium | hard |
| -- | --: | --: | --: |
| fixed_length | 1.000 | 0.8125 | 0.9286 |
| article_based | 0.800 | 0.4375 | 0.6429 |
| structure_aware | 0.800 | 0.4375 | 0.6429 |

## 五、解释限制

1. 当前 embedding 是 deterministic hash vector，不是真实中文语义模型。
2. fixed_length 的相关 chunk 标注继承了前序 Chunk 元数据粒度，相关集合更宽，指标不可直接等同于严格人工相关性评价。
3. 本结果主要证明 Dense Search 评估管线可运行，正式论文实验应替换真实中文 embedding 模型后重新计算。

## 六、边界确认

本阶段未运行 BM25、Hybrid Search、Reranker 或 RAG。raw、cleaned、Chunk 和 QA 测试集文件均只读使用。
