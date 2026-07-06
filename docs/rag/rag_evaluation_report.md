# RAG Batch Evaluation Report

## Default Evaluation

Default combination:

- method: `real_hybrid_weighted`
- strategy: `fixed_length`
- top_k: `5`
- answer_mode: `extractive`

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag_qa.py
```

Outputs:

- `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_answers_real_hybrid_weighted_fixed_length.jsonl`
- `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_evaluation_report.json`
- `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_evaluation_summary.csv`

## Boundary Statement

The evaluation writes external result files only. Those files are not committed to Git. The committed project files only contain reusable configuration, code, tests, and documentation.

No large language model API, reranker, Web frontend, new dependency installation, or new model download is part of this stage.

## Latest Default Metrics

The latest default run produced:

- `question_count`: `40`
- `answer_non_empty_rate`: `1.0`
- `citation_coverage_rate`: `1.0`
- `average_citation_count`: `3.0`
- `average_answer_chars`: `421.825`
- `source_policy_hit_rate`: `1.0`
- `related_chunk_hit_rate`: `1.0`
- `no_answer_count`: `0`
- `no_citation_count`: `0`
- `unsupported_answer_warning_count`: `0`
- `fallback_count`: `0`

Question type groups all reached answer/citation/related-chunk rates of `1.0`: `fact_lookup`, `condition`, `procedure`, `responsible_body`, `time_limit`, `qualification`, and `comparison`.

Difficulty groups all reached answer/citation/related-chunk rates of `1.0`: `easy`, `medium`, and `hard`.
