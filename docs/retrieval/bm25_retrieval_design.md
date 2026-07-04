# BM25 检索基线设计

## 一、阶段范围

本阶段构建 BM25 关键词检索基线，用于评估 `fixed_length`、`article_based` 和 `structure_aware` 三种 Chunk 策略在关键词检索下的 Recall@K 和 MRR@10 表现。

本阶段不执行 Hybrid Search、Reranker 或 RAG，也不重新生成 Embedding 或修改 Dense Search 结果。

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

检索 query 只使用 `question` 字段，不拼接 `expected_answer` 或 `evidence_text`。

## 三、BM25 实现

当前实现使用 Python 标准库，不安装新依赖。

分词方法为 `cjk_char_bigram`：

1. 提取中文单字 token。
2. 提取相邻中文二元片段 token。
3. 保留英文、数字和下划线连续片段。

局限性：该方法不能像 `jieba`、`pkuseg` 等中文分词工具一样识别完整词语、专有名词和政策术语边界。后续正式实验可替换为更适合中文政策文本的分词工具后重新计算。

BM25 参数：

```text
k1 = 1.5
b = 0.75
```

## 四、输出目录

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval
```

主要输出：

```text
indexes\
evaluation\
```

外部 BM25 索引、检索结果和评估报告不进入 Git。

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

`is_relevant` 依据问答测试集中的 `related_chunk_ids_*` 字段判断，不人工临时修改相关性。

## 六、边界说明

1. 不修改 raw 文件。
2. 不修改 cleaned 文件。
3. 不修改 Chunk 文件。
4. 不修改 QA 测试集文件。
5. 不修改 Dense Search 结果。
6. 不生成 Hybrid Search、Reranker 或 RAG 产物。
