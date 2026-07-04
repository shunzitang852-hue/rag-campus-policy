# Real Embedding Dense Retrieval Design

## Scope

This stage adds a configurable real Chinese embedding interface for Dense Search. It preserves the existing deterministic hash baseline and does not modify raw, cleaned, chunk, QA, Dense Hash, BM25, or Hybrid result files.

No Reranker or RAG generation is included in this stage.

## Providers

The config file is `configs/retrieval_dense_real_embedding.yaml`.

Supported provider names:

- `deterministic_hash`: existing reproducible hash vector baseline, kept for interface validation only.
- `sentence_transformers`: real embedding provider for local or downloadable sentence-transformers models.
- `local_model_placeholder`: explicit placeholder for future local-model integration; it fails with a controlled `ProviderUnavailable` error instead of silently producing fake vectors.

The preferred real model is `BAAI/bge-small-zh-v1.5`, cached under `F:\GraduationThesis\models`. The active run uses `sentence-transformers==5.6.0` with CPU `torch` and keeps HuggingFace cache files outside the Git repository.

## Data Flow

`scripts/build_real_embedding_dense_retrieval.py`:

1. Loads the real embedding config.
2. Creates the configured provider.
3. Encodes each chunk strategy in batches.
4. Writes embeddings and lightweight index metadata to `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding`.

Before loading any model, the build script sets:

- `HF_HOME=F:\GraduationThesis\cache\huggingface`
- `TRANSFORMERS_CACHE=F:\GraduationThesis\cache\huggingface\transformers`
- `SENTENCE_TRANSFORMERS_HOME=F:\GraduationThesis\models\sentence_transformers`

`scripts/evaluate_real_embedding_dense_retrieval.py`:

1. Verifies the raw and cleaned source boundaries.
2. Builds embeddings if the provider is available.
3. Searches with only the QA `question` field.
4. Computes Recall@1, Recall@3, Recall@5, Recall@10, MRR@10, average top-1 score, average relevant rank, no-hit count, question-type Recall@5, and difficulty Recall@5.

`expected_answer` and `evidence_text` are not used for retrieval.

## Fallback

If the real model cannot run, the evaluation script writes `real_embedding_dense_evaluation_report.json` with:

- `model_ran: false`
- the configured provider and model name
- the failure reason
- input file hashes
- an empty `metrics_by_strategy`

In that fallback mode, no real embedding metrics are claimed.
