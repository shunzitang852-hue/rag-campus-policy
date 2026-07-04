# Dense Search 检索基线设计

## 一、阶段范围

本阶段构建 Embedding 与 Dense Search 检索基线，用于验证三种 Chunk 策略的检索评估流程。

本阶段不执行 BM25、Hybrid Search、Reranker 或 RAG 问答生成。

## 二、输入文件

三种 Chunk 文件：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_fixed_length.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_article_based.jsonl
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_structure_aware.jsonl
```

问答测试集：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl
```

## 三、Embedding 方法

当前实现使用 `deterministic_hash_embedding`：

1. 使用中文字符、中文二元片段和英文数字 token。
2. 通过 SHA-256 将 token 映射到固定维度向量。
3. 使用余弦相似度进行 Dense Search。
4. 不下载模型，不安装新依赖。

重要说明：当前方法不是正式语义 embedding，不能作为论文最终语义检索实验结论。它的用途是验证索引、检索、标注命中和 Recall@K 评估流程。后续正式实验应替换为适合中文的本地 embedding 模型，例如 BGE、text2vec 或其他 sentence-transformers 中文模型。

## 四、输出目录

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval
```

输出分为：

```text
embeddings\
indexes\
evaluation\
```

外部 embedding、索引和评估结果文件不进入 Git。

## 五、评估指标

每种 Chunk 策略计算：

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR@10
average_top1_score
average_relevant_rank
no_hit_count
question_type 分组 Recall@5
difficulty 分组 Recall@5
```

`is_relevant` 判断依据问答测试集中的 `related_chunk_ids_*` 字段。检索 query 只使用 `question` 字段，不拼接 `expected_answer` 或 `evidence_text`。

## 六、边界说明

1. 不修改 raw 文件。
2. 不修改 cleaned 文件。
3. 不修改已有 Chunk 文件。
4. 不修改 QA 测试集文件。
5. 不生成 BM25、Hybrid Search、Reranker 或 RAG 产物。
