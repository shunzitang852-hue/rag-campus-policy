# Extractive RAG Demo Report

## Current Capability

The demo supports retrieval-augmented extractive answers over student handbook chunks.

Supported retrieval methods:

- `bm25`
- `real_dense`
- `real_hybrid_weighted`

Supported chunk strategies:

- `fixed_length`
- `article_based`
- `structure_aware`

Default parameters:

- `method`: `real_hybrid_weighted`
- `strategy`: `fixed_length`
- `top_k`: `5`
- `answer_mode`: `extractive`
- `max_context_chars`: `4000`

## Example Command

```powershell
.\.venv\Scripts\python.exe scripts\demo_rag_qa.py --query "学生申请解除处分需要满足什么条件？" --method real_hybrid_weighted --strategy fixed_length --top-k 5
```

## Output Structure

The command prints:

- generated answer
- retrieval evidence
- source documents
- `chunk_id`
- retrieval method and strategy
- notes

## Limitations

This is a first-version controlled RAG demo. It uses template-based extraction and does not call a large language model, paid API, reranker, web frontend, or new model download.
