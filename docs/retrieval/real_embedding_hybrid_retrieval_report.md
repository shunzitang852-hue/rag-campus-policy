# Real Embedding Hybrid Retrieval Report

## Current Run

This stage fused real Chinese embedding Dense Search results from `BAAI/bge-small-zh-v1.5` with the existing BM25 results.

- Output root: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding`
- Weighted fusion: real dense `0.5`, BM25 `0.5`
- RRF parameter: `k = 60`
- Question count: `40`

## weighted_score

| strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_length | 1.000 | 1.000 | 1.000 | 1.000 | 1.0000 |
| article_based | 0.600 | 0.725 | 0.750 | 0.750 | 0.6562 |
| structure_aware | 0.600 | 0.725 | 0.750 | 0.750 | 0.6562 |

## rrf

| strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_length | 0.900 | 1.000 | 1.000 | 1.000 | 0.9458 |
| article_based | 0.550 | 0.675 | 0.750 | 0.750 | 0.6287 |
| structure_aware | 0.550 | 0.675 | 0.750 | 0.750 | 0.6287 |

## Recall@5 Comparison

| strategy | Dense Hash | BM25 | Hash Hybrid weighted | Real Embedding Dense | Real Hybrid weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_length | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 |
| article_based | 0.600 | 0.675 | 0.725 | 0.750 | 0.750 |
| structure_aware | 0.600 | 0.675 | 0.725 | 0.750 | 0.750 |

## Grouped Recall@5

For `fixed_length`, both fusion methods reach `1.0` Recall@5 in every question type and difficulty group.

For `article_based` and `structure_aware`, both fusion methods share these grouped Recall@5 values:

- question_type: fact_lookup `1.0`, condition `0.6667`, procedure `0.5`, responsible_body `0.8333`, time_limit `0.6667`, qualification `0.8`, comparison `0.8`
- difficulty: easy `0.9`, medium `0.6875`, hard `0.7143`

## Outputs

- Results: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding\evaluation\real_hybrid_search_results_{strategy}.jsonl`
- Method-specific results: `real_hybrid_search_results_{strategy}_{fusion_method}.jsonl`
- Report: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding\evaluation\real_hybrid_evaluation_report.json`
- Summary: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding\evaluation\real_hybrid_evaluation_summary.csv`

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_real_embedding_hybrid_retrieval.py
.\.venv\Scripts\python.exe -m pytest tests\test_real_embedding_hybrid_retrieval.py
.\.venv\Scripts\python.exe -m pytest
```

## Boundary

This stage did not modify raw files, cleaned files, chunk files, QA files, Dense Hash outputs, BM25 outputs, old Hash Hybrid outputs, or Real Embedding Dense outputs. It did not create Reranker or RAG artifacts.
