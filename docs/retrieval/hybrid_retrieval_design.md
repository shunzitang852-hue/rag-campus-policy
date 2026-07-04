# Hybrid Search Design

## Scope

This stage builds a reproducible Hybrid Search baseline for the `hdu_ies` student handbook corpus. It reads the existing Dense Hash and BM25 top-10 evaluation outputs and fuses their candidates. It does not regenerate chunks, replace embeddings, run a reranker, or generate RAG answers.

## Inputs

- QA test set: `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`
- Dense result files: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval\evaluation\dense_search_results_{strategy}.jsonl`
- BM25 result files: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation\bm25_search_results_{strategy}.jsonl`
- Chunk strategies: `fixed_length`, `article_based`, `structure_aware`

## Fusion Methods

`weighted_score` performs per-query min-max normalization separately for Dense and BM25 scores, then computes:

```text
hybrid_score = 0.4 * dense_normalized_score + 0.6 * bm25_normalized_score
```

If all scores for one retrieval source are equal within a query, the normalized values are set to `0.0` to avoid division by zero. If a candidate is missing from one retrieval source, that source contributes `0.0`.

`rrf` uses Reciprocal Rank Fusion:

```text
hybrid_score = sum(1 / (60 + rank_i))
```

Only retrieval sources that returned the candidate contribute to the sum.

## Output Schema

Each Hybrid result row contains:

- `question_id`
- `question`
- `strategy`
- `fusion_method`
- `rank`
- `retrieved_chunk_id`
- `retrieved_document_title`
- `dense_score`
- `bm25_score`
- `hybrid_score`
- `is_relevant`
- `matched_by`

`is_relevant` is computed only from the QA test set's `related_chunk_ids_{strategy}` fields. `expected_answer` and `evidence_text` are not used as retrieval queries or relevance inputs.

## Evaluation

For every strategy and fusion method, the script computes:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR@10
- average_top1_score
- average_relevant_rank
- no_hit_count
- question_type grouped Recall@5
- difficulty grouped Recall@5

External outputs are written under `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval`.
