# 杭电信工学生手册问答测试集报告

本报告由 `scripts/build_student_handbook_qa_testset.py` 生成外部数据后配套记录。统计口径以外部 JSON 报告为准：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_report.json
```

## 一、输出文件

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.csv
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl
```

## 二、验收指标

| 指标 | 要求 | 本次结果 |
| -- | --: | --: |
| question_count | 40 | 40 |
| missing_evidence_count | 0 | 0 |
| duplicate_question_count | 0 | 0 |
| covered_policy_count | 不少于20 | 32 |
| missing_related_chunk_count | 0 | 0 |
| missing_all_strategy_chunk_count | 优先为0 | 0 |

## 三、分布统计

问题类型分布：

| question_type | 数量 |
| -- | --: |
| fact_lookup | 6 |
| condition | 6 |
| procedure | 6 |
| responsible_body | 6 |
| time_limit | 6 |
| qualification | 5 |
| comparison | 5 |

难度分布：

| difficulty | 数量 |
| -- | --: |
| easy | 10 |
| medium | 16 |
| hard | 14 |

平均证据文本长度：116.0 字符。

## 四、覆盖情况

第一版测试集覆盖 32 项制度。每条问题均保留 `evidence_text`，并关联 `fixed_length`、`article_based` 和 `structure_aware` 三种 Chunk 策略下的 chunk_id。

## 五、边界说明

本阶段没有执行 Embedding、向量库、BM25、Dense Search、Hybrid Search、Reranker 或 RAG。

raw、cleaned 和 Chunk 文件仅用于读取和校验，不在本阶段修改。
