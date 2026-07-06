# RAG Answer Quality Design

This module provides first-version rule-based quality assistance for generated RAG answers.

## Inputs

- QA test set: `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl`
- RAG batch answers: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_answers_real_hybrid_weighted_fixed_length.jsonl`
- RAG batch report: `F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\rag_evaluation\rag_evaluation_report.json`

`expected_answer` and `evidence_text` are used only for evaluation. They are not passed into answer generation.

## Rule Metrics

The evaluator computes:

- keyword overlap between `expected_answer` and `generated_answer`
- evidence overlap between evidence keywords/text and `generated_answer`
- source policy title hit in citations
- related chunk hit from retrieved chunk IDs
- missing expected-answer keywords
- possible unsupported generated terms
- `auto_quality_label`: `good`, `needs_review`, or `insufficient`

These metrics are review aids. They are not equivalent to final answer correctness.

## Manual Review Fields

The CSV reserves empty columns for:

- `manual_correctness_score`
- `manual_completeness_score`
- `manual_groundedness_score`
- `manual_notes`

Suggested manual scores use `0`, `1`, and `2`:

- correctness: `0` clearly wrong, `1` partly correct, `2` correct
- completeness: `0` misses core requirements, `1` basically covers but incomplete, `2` relatively complete
- groundedness: `0` unsupported or contradicts evidence, `1` partly supported, `2` mainly supported
