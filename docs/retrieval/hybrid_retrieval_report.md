# Hybrid Search Report

## Stage Boundary

This stage validates Hybrid Search fusion on top of the existing Dense Hash and BM25 result files. It does not modify raw files, cleaned files, chunk files, QA files, Dense results, or BM25 results. It does not create Reranker or RAG artifacts.

## Configuration

- Config: `configs/retrieval_hybrid.yaml`
- Fusion methods: `weighted_score`, `rrf`
- Weighted score weights: Dense `0.4`, BM25 `0.6`
- RRF parameter: `k = 60`
- Per-query output count: top 10

## Outputs

- Result files: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval\evaluation\hybrid_search_results_{strategy}.jsonl`
- Method-specific result files: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval\evaluation\hybrid_search_results_{strategy}_{fusion_method}.jsonl`
- Evaluation report: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval\evaluation\hybrid_evaluation_report.json`
- Evaluation summary: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval\evaluation\hybrid_evaluation_summary.csv`

## Reproduce

```powershell
python scripts/evaluate_hybrid_retrieval.py
python -m pytest tests/test_hybrid_retrieval.py tests/test_acceptance_artifacts.py
```

## Notes

Dense Search still uses deterministic hash embeddings, so the Hybrid baseline is a workflow and evaluation baseline rather than a final semantic-retrieval conclusion. BM25 remains the stronger lexical baseline in the current result set.
