# Real Embedding Dense Retrieval Report

## Current Run

The real Chinese embedding provider did not run in the current environment.

Reason: `sentence-transformers` is not installed in `.venv`, and no local Chinese embedding model was found under `F:\GraduationThesis\models` or `F:\GraduationThesis\cache`.

## Implemented Artifacts

- Config: `configs/retrieval_dense_real_embedding.yaml`
- Provider interface: `src/retrieval/embedding_providers.py`
- Build script: `scripts/build_real_embedding_dense_retrieval.py`
- Evaluation script: `scripts/evaluate_real_embedding_dense_retrieval.py`
- Tests: `tests/test_real_embedding_retrieval.py`

## External Report

The fallback report path is:

```text
F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\evaluation\real_embedding_dense_evaluation_report.json
```

Because the configured real model did not run, this report intentionally contains no real Dense Search metrics.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_real_embedding_dense_retrieval.py
.\.venv\Scripts\python.exe -m pytest tests\test_real_embedding_retrieval.py
.\.venv\Scripts\python.exe -m pytest
```

## Boundary

This stage did not modify raw files, cleaned files, chunk files, QA files, Dense Hash outputs, BM25 outputs, or Hybrid outputs. It did not create Reranker or RAG artifacts.
