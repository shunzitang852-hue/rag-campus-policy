# Extractive RAG Design

This stage adds the first controlled RAG answer module for the `hdu_ies` student handbook corpus.

## Scope

- Input: retrieved Chunk results from `src.retrieval.demo`.
- Output: a short extractive answer, citations, source documents, retrieval metadata, and notes.
- Default retrieval method: `real_hybrid_weighted`.
- Fallback order: `real_dense`, then `bm25` when a real embedding provider is unavailable.
- Answer mode: `extractive`.

The module does not call an LLM API, does not run a reranker, does not rebuild raw/cleaned/chunk/QA data, and does not overwrite historical retrieval experiment outputs.

## Modules

- `context_builder.py`: removes duplicate `chunk_id` values, skips directory-like chunks, preserves document titles and chunk IDs, and bounds total context length.
- `citation.py`: converts context items to citation records with `citation_id`, `document_title`, `chunk_id`, `score`, `excerpt`, `strategy`, and `retrieval_method`.
- `answer_generator.py`: selects short evidence-bearing sentences from retrieved chunks and wraps them in a controlled template.
- `pipeline.py`: validates query/method/strategy/top-k, calls retrieval, applies fallback, builds context/citations, and returns a structured response.
- `scripts/demo_rag_qa.py`: command-line and interactive demo entry point.

## Data Boundary

The generator only receives retrieved Chunk text. It does not read `expected_answer` or `evidence_text` from the QA test set, so answer text remains independent from evaluation labels.

## Reliability Notes

If retrieval returns no usable context, or the configured minimum top score is not met, the answer is:

```text
当前检索片段不足以确认答案。
```

Every successful answer includes citations and a note that the result is generated from retrieved policy fragments rather than an official final school interpretation.
