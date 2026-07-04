# 检索问答 Demo

本 Demo 用于阶段性展示学生手册语料上的检索问答能力。系统只根据用户输入的问题返回相关制度片段，不调用大语言模型，不生成最终自然语言答案，不执行 Reranker，也不改写任何实验结果文件。

## 支持范围

- 默认检索方法：`real_dense`
- 默认 Chunk 策略：`fixed_length`
- 默认返回数量：`top_k = 5`
- 支持中文自然语言问题
- 输出固定包含：`rank`、`chunk_id`、`document_title`、`score`、`chunk_text` 摘要、`strategy`、`retrieval_method`

支持的 Chunk 策略：

- `fixed_length`
- `article_based`
- `structure_aware`

支持的检索方法：

- `bm25`：运行时基于所选 Chunk 文件构建内存 BM25 索引。
- `real_dense`：读取已有真实中文 Embedding 向量文件，并对新 query 动态编码检索。
- `real_hybrid_weighted`：运行时分别计算 `real_dense` 与 `bm25` 候选结果，在 query 内归一化后按 0.5/0.5 加权融合。

`real_hybrid_weighted` 不读取固定 QA 测试集结果来回答新问题，也不会覆盖历史 Real Hybrid 实验输出。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\demo_retrieval_qa.py --query "学生申请解除处分需要满足什么条件？" --method real_dense --strategy fixed_length --top-k 5
```

如果不传 `--query`，脚本会进入交互模式：

```powershell
.\.venv\Scripts\python.exe scripts\demo_retrieval_qa.py --method bm25 --strategy structure_aware --top-k 3
```

交互模式下输入 `exit` 或 `quit` 退出。

## 输出说明

示例输出结构：

```text
问题：学生申请解除处分需要满足什么条件？
检索方法：real_dense
Chunk 策略：fixed_length
Top K：5
说明：当前为检索结果展示，不是大模型生成答案。

[1] 杭州电子科技大学信息工程学院学生解除处分管理办法
chunk_id: ...
score: ...
strategy: fixed_length
retrieval_method: real_dense
摘要：...
```

摘要会压缩空白并限制在 300 字以内，避免控制台输出过长。

## 边界

- 不调用大模型 API。
- 不生成无法溯源的答案。
- 不使用 `expected_answer` 或 `evidence_text`。
- 不重新生成 raw、cleaned、Chunk 或 QA 测试集。
- 不覆盖 Dense Hash、BM25、旧 Hybrid、Real Dense、Real Hybrid 历史实验结果。
- 不下载新模型，不安装新依赖。
