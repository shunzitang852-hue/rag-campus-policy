# Real Embedding Dense Retrieval Report

## Current Run

The real Chinese embedding model ran successfully.

- Provider: `sentence_transformers`
- Model: `BAAI/bge-small-zh-v1.5`
- Embedding dimension: `512`
- Model cache path: `F:\GraduationThesis\models`
- HF cache path: `F:\GraduationThesis\cache\huggingface`
- Output root: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding`

The run used CPU PyTorch and did not use CUDA.

## Metrics

| strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_length | 0.975 | 1.000 | 1.000 | 1.000 | 0.9833 |
| article_based | 0.675 | 0.750 | 0.750 | 0.800 | 0.7136 |
| structure_aware | 0.675 | 0.750 | 0.750 | 0.800 | 0.7136 |

Chunk counts:

| strategy | chunks |
| --- | ---: |
| fixed_length | 244 |
| article_based | 358 |
| structure_aware | 358 |

## Recall@5 Comparison

| strategy | Dense Hash | BM25 | Hybrid weighted | Real Embedding Dense |
| --- | ---: | ---: | ---: | ---: |
| fixed_length | 0.900 | 0.950 | 0.950 | 1.000 |
| article_based | 0.600 | 0.675 | 0.725 | 0.750 |
| structure_aware | 0.600 | 0.675 | 0.725 | 0.750 |

## External Artifacts

- Embeddings: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\embeddings`
- Indexes: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\indexes`
- Evaluation: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\evaluation`
- Report: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\evaluation\real_embedding_dense_evaluation_report.json`

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_real_embedding_dense_retrieval.py
.\.venv\Scripts\python.exe -m pytest tests\test_real_embedding_retrieval.py
.\.venv\Scripts\python.exe -m pytest
```

The scripts set these cache directories before loading the model:

```text
HF_HOME=F:\GraduationThesis\cache\huggingface
TRANSFORMERS_CACHE=F:\GraduationThesis\cache\huggingface\transformers
SENTENCE_TRANSFORMERS_HOME=F:\GraduationThesis\models\sentence_transformers
```

## Boundary

This stage did not modify raw files, cleaned files, chunk files, QA files, Dense Hash outputs, BM25 outputs, or Hybrid outputs. It did not create BM25, Hybrid, Reranker, or RAG artifacts.
