# 杭电信工学生手册问答测试集设计

## 一、阶段范围

本阶段构建第一版“杭电信工学生手册问答测试集”，用于后续评估不同 Chunk 策略和检索策略的 Recall@K 效果。

本阶段只生成测试问题集，不执行 Embedding、向量库、BM25、Dense Search、Hybrid Search、Reranker 或 RAG。

## 二、输入文件

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt
F:\GraduationThesis\datasets\metadata\hdu_ies\hdu_ies_student_handbook_document_catalog.csv
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks
```

测试问题以 cleaned 正文和前序结构化记录为依据，Chunk 关联依据三种策略 JSONL 中的 `metadata.source_record_ids`。

## 三、输出文件

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.csv
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_report.json
```

外部测试集数据文件不进入 Git。

## 四、字段设计

每条问题包含：

```text
question_id
question
question_type
expected_answer
source_policy_title
evidence_text
evidence_start_char
evidence_end_char
evidence_keywords
related_chunk_ids_fixed_length
related_chunk_ids_article_based
related_chunk_ids_structure_aware
difficulty
answer_granularity
notes
```

## 五、问题类型

允许的问题类型包括：

```text
fact_lookup
condition
procedure
responsible_body
time_limit
qualification
comparison
```

## 六、质量约束

1. 第一版固定生成 40 条问题。
2. 每条问题必须有非空 `evidence_text`。
3. `evidence_text` 来自 cleaned 正文或其结构化记录，不改写原文。
4. 每条问题至少关联一种 Chunk 策略下的 chunk_id。
5. 优先要求三种 Chunk 策略均能关联 chunk_id。
6. 覆盖制度数量不少于 20 项。
7. 不将目录标题作为证据。
8. 不推断学生手册版本年份。
9. 不编造未识别文号。

## 七、人工复核说明

本测试集用于检索实验前的数据准备。`expected_answer` 采用证据文本本身，后续论文实验前应由项目负责人人工复核问题表述、答案粒度和证据相关性。
