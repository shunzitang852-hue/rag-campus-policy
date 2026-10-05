# DeepSeek matched experiment

The runner consumes persisted top-5 retrieval results. It never recomputes retrieval during generation. Existing historical runs are read-only, and outputs stay under `experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness/deepseek/<experiment_id>`.

Use the configured process environment `DEEPSEEK_API_KEY`; never pass credentials as command-line arguments. First run `python scripts/check_deepseek_model_identity.py`. Use the experiment ID it reports for all later commands:

```powershell
python scripts/run_deepseek_cross_model.py --experiment-id <id> --stage prepare
python scripts/run_deepseek_cross_model.py --experiment-id <id> --stage smoke1
python scripts/run_deepseek_cross_model.py --experiment-id <id> --stage smoke3
python scripts/run_deepseek_cross_model.py --experiment-id <id> --stage full
```

Proceed only after the preceding gate passes. Identical API-returned model IDs stop the comparison. Two unknown identities permit only an API-configuration robustness claim. One unknown identity stops until resolved.

Each request saves requested and actual model separately, exact context/prompt hashes, response metadata, usage and timing. The first three questions are reused in the full set of 40, for at most 80 generation requests. The append-only recorder flushes/fsyncs each result, and a pending intent prevents automatic duplicate calls after an interruption. Failed calls receive no automatic retries. This runner assumes one process per experiment ID; do not launch concurrent runs for the same ID.

Frozen file integrity and deterministic rendering are validated before generation. The historical baseline is included in strict comparisons only if saved full contexts establish exact equality; matching chunk IDs is insufficient.

The existing rule-based evaluator provides keyword overlap, evidence keyword overlap and good/needs_review/insufficient labels. Other semantic metrics stay blank. The 80-row manual review sheet has anonymous labels and empty scoring fields; keep the separate blind key and unblinded result files away from reviewers.

The per-experiment registry registers the two persisted metric groups without rebuilding or modifying the historical global registry. `git_state.json` records whether the source checkout was clean; local changes are retained and no commit/push is performed.
