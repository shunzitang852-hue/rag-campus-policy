# RAG Answer Quality Report

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag_answer_quality.py
```

Outputs:

- `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality\rag_answer_quality_review_v1.csv`
- `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality\rag_answer_quality_review_v1.jsonl`
- `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_answer_quality\rag_answer_quality_report.json`

The generated CSV is intended for manual review. Rule-based metrics are auxiliary signals and should not be reported as final accuracy.

No large language model API, reranker, Web frontend, new dependency installation, or new model download is part of this stage.

## Latest Metrics

The latest run produced:

- `question_count`: `40`
- `average_keyword_overlap_ratio`: `0.5616666666666668`
- `average_evidence_overlap_ratio`: `0.7741666666666667`
- `source_policy_hit_rate`: `1.0`
- `related_chunk_hit_rate`: `1.0`
- `answer_non_empty_rate`: `1.0`
- `citation_coverage_rate`: `1.0`
- `auto_quality_distribution`: `good=21`, `needs_review=19`
- `needs_review_count`: `19`
- `insufficient_count`: `0`

Question type overview:

- `fact_lookup`: `6` questions, average keyword overlap `0.6666666666666666`, `good=4`, `needs_review=2`
- `condition`: `6` questions, average keyword overlap `0.5055555555555555`, `good=4`, `needs_review=2`
- `procedure`: `6` questions, average keyword overlap `0.5055555555555555`, `good=3`, `needs_review=3`
- `responsible_body`: `6` questions, average keyword overlap `0.5333333333333333`, `good=3`, `needs_review=3`
- `time_limit`: `6` questions, average keyword overlap `0.6722222222222222`, `good=3`, `needs_review=3`
- `qualification`: `5` questions, average keyword overlap `0.54`, `good=2`, `needs_review=3`
- `comparison`: `5` questions, average keyword overlap `0.49333333333333335`, `good=2`, `needs_review=3`

Difficulty overview:

- `easy`: `10` questions, average keyword overlap `0.6233333333333333`, `good=6`, `needs_review=4`
- `medium`: `16` questions, average keyword overlap `0.48333333333333334`, `good=7`, `needs_review=9`
- `hard`: `14` questions, average keyword overlap `0.6071428571428571`, `good=8`, `needs_review=6`

All question type and difficulty groups have `source_policy_hit_rate=1.0` and `related_chunk_hit_rate=1.0`.
