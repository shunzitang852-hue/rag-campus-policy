# Real Embedding Hybrid Retrieval Design

## Scope

This stage evaluates Hybrid Search using the existing real Chinese embedding Dense Search results and the existing BM25 results. It does not rebuild BM25, regenerate chunks, overwrite the older Dense Hash + BM25 Hybrid outputs, run a reranker, or generate RAG answers.

## Inputs

- Real Embedding Dense evaluation: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\evaluation`
- BM25 evaluation: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation`
- QA test set: `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`

## Fusion

`weighted_score` applies per-query min-max normalization to real dense scores and BM25 scores, then computes:

```text
hybrid_score = 0.5 * real_dense_normalized_score + 0.5 * bm25_normalized_score
```

If one score list is constant for a query, normalized scores are set to `0.0` to avoid division by zero. Missing scores from either retrieval source contribute `0.0`.

`rrf` uses Reciprocal Rank Fusion:

```text
hybrid_score = sum(1 / (60 + rank_i))
```

## Output Fields

Each result row contains:

- `question_id`
- `question`
- `strategy`
- `fusion_method`
- `rank`
- `retrieved_chunk_id`
- `retrieved_document_title`
- `real_dense_score`
- `bm25_score`
- `hybrid_score`
- `is_relevant`
- `matched_by`

Relevance is computed only from `related_chunk_ids_{strategy}` in the QA test set. `expected_answer` and `evidence_text` are not used for retrieval.
