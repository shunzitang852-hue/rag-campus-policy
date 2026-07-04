# 杭电信工学生手册 Chunk 切分报告

## 一、阶段范围

本阶段仅完成杭州电子科技大学信息工程学院学生手册 cleaned 正文的 Chunk 切分、统计和自动化验收。

本阶段没有生成 Embedding、向量库、BM25、Dense Search、Hybrid Search、Reranker 或 RAG 产物。

## 二、输入文件

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt
```

raw 文件 SHA-256 复核结果：

```text
83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189
```

## 三、输出文件

输出目录：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks
```

主要输出文件：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_fixed_length.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_article_based.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_structure_aware.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunking_report.json
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunking_anomalies.csv
```

## 四、策略说明

| 策略 | 说明 | 主要参数 |
| -- | -- | -- |
| fixed_length | 在制度边界内按固定长度切分，用作后续检索实验对照组。 | chunk_size=500, overlap=100 |
| article_based | 优先按条款、自然段或条目组织 Chunk，过短片段可合并，避免破坏制度边界。 | max_chars=900, overlap=80 |
| structure_aware | 优先保留制度标题、章节、条款等结构信息，每个 Chunk 可追溯到 document_id 和制度标题。 | max_chars=900, overlap=80 |

## 五、统计结果

| 策略 | chunk_count | avg_chars | max_chars | covered_char_ratio | duplicate_coverage_ratio | empty_chunk_count | covered_policy_count | missing_policy_count | anomaly_count |
| -- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| fixed_length | 244 | 452.59 | 500 | 1.0 | 1.0 | 0 | 32 | 0 | 0 |
| article_based | 358 | 251.68 | 900 | 1.0 | 0.21330275229357798 | 0 | 32 | 0 | 0 |
| structure_aware | 358 | 251.68 | 900 | 1.0 | 0.21330275229357798 | 0 | 32 | 0 | 0 |

## 六、质量验收结论

| 验收项 | 结果 |
| -- | -- |
| empty_chunk_count 是否为 0 | 通过 |
| covered_policy_count 是否为 32 | 通过 |
| missing_policy_count 是否为 0 | 通过 |
| Chunk 文本是否非空 | 通过 |
| Chunk 是否保留制度标题追溯字段 | 通过 |
| 是否修改 raw 文件 | 否 |
| 是否修改 cleaned 文件 | 否 |
| 是否生成 Embedding、向量库、BM25、Hybrid、RAG 产物 | 否 |

## 七、异常项

自动验收未发现空 Chunk、超长 Chunk、缺失制度覆盖或跨制度 Chunk。

已知标题差异仍沿用清洗阶段记录：`杭州电子科技大学信息工程学院学生解除处分管理办法` 的目录标题与正文标题存在差异，本阶段只继承记录，不改写正文。

## 八、测试

测试命令：

```text
python -m pytest tests/test_chunking.py tests/test_acceptance_artifacts.py
```

测试结果以本次提交前实际运行为准。
