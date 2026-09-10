# 开题答辩演示指南

## 1. 演示边界

当前系统已打通“问题 -> Top-K 检索结果 -> 有界上下文 -> 抽取式模板答案 -> 引用”的可运行链路。回答模块不调用大模型 API，也没有生成式 Prompt；答辩中应明确表述为“可复现的抽取式 RAG 基线”，不要宣称已经完成 LLM 生成式 RAG。

## 2. 环境准备

在 `F:\GraduationThesis\rag-campus-policy` 下使用 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONIOENCODING = "utf-8"
```

真实向量演示固定读取以下本地模型快照，并设置 `local_files_only=true`，演示期间不依赖 Hugging Face 网络：

```text
F:\GraduationThesis\models\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620
```

## 3. 答辩前验收

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\audit_final_experiment_matrix.py
```

测试应全部通过；实验矩阵审计应显示 `missing=0`。

## 4. 检索演示

```powershell
.\.venv\Scripts\python.exe scripts\demo_retrieval_qa.py --query "学生申请国家奖学金需要满足什么条件？" --method real_hybrid_weighted --strategy fixed_length --top-k 5
```

预期输出包含查询、方法、Chunk 策略，以及 5 条带 `chunk_id`、来源标题、分数和文本摘要的检索结果。

若本地向量模型不可用，可临时切换到不依赖模型的 BM25：

```powershell
.\.venv\Scripts\python.exe scripts\demo_retrieval_qa.py --query "学生申请国家奖学金需要满足什么条件？" --method bm25 --strategy fixed_length --top-k 5
```

## 5. RAG 回答演示

```powershell
.\.venv\Scripts\python.exe scripts\demo_rag_qa.py --query "学生申请国家奖学金需要满足什么条件？" --method real_hybrid_weighted --strategy fixed_length --top-k 5
```

预期输出包含抽取式答案、引用来源、`chunk_id`、检索分数和实际使用的方法。真实向量不可用时脚本会回退到 `real_dense`，再回退到 `bm25`，并在输出中显示实际方法。

## 6. 常见问题

- 中文乱码：先执行 `$env:PYTHONIOENCODING = "utf-8"`；必要时再执行 `chcp 65001`。
- 模型目录不存在：检查 `configs/retrieval_dense_real_embedding.yaml` 中的 `local_model_path`，或使用 BM25 演示。
- 数据文件不存在：确认 `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook` 未被移动。
- 不要在答辩前临时重跑全部实验；先运行矩阵审计，只补缺失项。
