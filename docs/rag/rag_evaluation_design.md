# RAG Batch Evaluation Design

This stage evaluates the current extractive/template-based RAG demo with the existing 40-question QA test set.

## Scope

- Reads the existing QA JSONL file as evaluation input.
- Sends only `question` text to the RAG pipeline.
- Uses `expected_answer`, `evidence_text`, `source_policy_title`, and related chunk IDs only for scoring.
- Writes generated answer results and reports under `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation`.

The evaluator does not modify raw, cleaned, Chunk, or QA files. It does not overwrite Dense Hash, BM25, old Hybrid, Real Dense, or Real Hybrid outputs.

## Metrics

The default report includes:

- `question_count`
- `answer_non_empty_rate`
- `citation_coverage_rate`
- `average_citation_count`
- `average_answer_chars`
- `no_answer_count`
- `no_citation_count`
- `source_policy_hit_rate`
- `related_chunk_hit_rate`
- `unsupported_answer_warning_count`
- `fallback_count`
- grouped metrics by `question_type`
- grouped metrics by `difficulty`

`source_policy_hit_rate` and `related_chunk_hit_rate` are approximate automatic indicators. They do not prove the generated answer is fully correct.
