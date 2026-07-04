# 杭电信工学生手册 Chunk 切分设计

## 一、设计目标

本设计面向杭州电子科技大学信息工程学院学生手册 cleaned 正文，为后续检索实验准备可比较的 Chunk 数据。

本阶段只做 Chunk 切分，不做 Embedding、向量库、BM25、Dense Search、Hybrid Search、Reranker 或 RAG。

## 二、输入与来源

正式输入文件：

```text
F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt
```

为保留制度标题、条款编号、来源行号和文号等结构信息，生成脚本同时读取前序结构化记录：

```text
F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\hdu_ies_student_handbook_articles_clean.jsonl
```

Chunk 的 `source_file` 字段统一指向 cleaned 正文文件。

## 三、输出契约

每个 Chunk 至少包含以下字段：

```text
chunk_id
strategy
document_id
document_title
chunk_index
chunk_text
char_count
start_char
end_char
source_file
metadata
```

为兼容项目内部已有解析流程，输出中同时保留 `policy_id`、`policy_title`、`chunk_strategy`、`chunk_char_count`、`source_hash` 等字段。

`metadata` 至少包含：

```text
policy_title
section_title
article_numbers
has_reference_number
reference_number
year_clues
cleaning_version
cleaned_source_path
```

## 四、切分策略

### fixed_length

固定长度切分策略，用于后续检索实验的基础对照组。该策略在制度边界内按长度窗口切分，默认 `chunk_size=500`，`overlap=100`。

### article_based

条款优先策略。该策略优先按制度条款、自然段或条目切分，过短片段可在同一制度内合并，避免破坏条款编号。默认 `max_chars=900`。

### structure_aware

结构感知策略。该策略优先保留制度标题、章节、条款结构，每个 Chunk 均可追溯到 `document_id`、制度标题、来源结构记录和 cleaned 文件路径。默认 `max_chars=900`。

## 五、质量指标

每种策略统计以下指标：

```text
chunk_count
min_chars
max_chars
avg_chars
empty_chunk_count
overlong_chunk_count
covered_char_ratio
duplicate_coverage_ratio
covered_policy_count
missing_policy_count
missing_policy_titles
anomaly_count
```

覆盖率按前序结构化记录 ID 去重计算，overlap 导致的重复覆盖单独记录为 `duplicate_coverage_ratio`。

## 六、验收标准

1. `empty_chunk_count` 必须为 0。
2. `covered_policy_count` 应为 32。
3. `missing_policy_count` 应为 0。
4. `article_based` 和 `structure_aware` 不应破坏制度标题追溯。
5. Chunk 文本不得为空白。
6. Chunk 文本不得只包含目录标题。
7. 不得修改 raw 文件。
8. 不得修改 cleaned 文件。
9. 不得生成 Embedding、向量库、BM25、Hybrid 或 RAG 产物。
