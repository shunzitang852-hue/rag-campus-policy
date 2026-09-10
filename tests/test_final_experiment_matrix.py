from __future__ import annotations

import json

from src.experiments.final_matrix import assess_matrix, final_retrieval_specs


def _row(method: str, strategy: str, parameters: dict | None = None) -> dict[str, str]:
    return {
        "experiment_id": f"run-{method}-{strategy}",
        "task_type": "retrieval",
        "retrieval_method": method,
        "chunk_strategy": strategy,
        "top_k": "5",
        "status": "completed",
        "legacy": "false",
        "git_worktree_clean": "true",
        "reproducible": "full",
        "has_per_question_results": "true",
        "hybrid_parameters": json.dumps(parameters or {}),
    }


def test_main_matrix_has_twelve_frozen_combinations() -> None:
    specs = final_retrieval_specs()
    assert len(specs) == 12
    assert len({(item.chunk_strategy, item.retrieval_method) for item in specs}) == 12


def test_dirty_legacy_or_partial_rows_do_not_fill_matrix() -> None:
    row = _row("bm25", "fixed_length")
    for field, invalid in [
        ("legacy", "true"),
        ("git_worktree_clean", "false"),
        ("reproducible", "partial"),
    ]:
        candidate = dict(row)
        candidate[field] = invalid
        status = assess_matrix([candidate])
        fixed_bm25 = next(item for item in status if item["spec_key"] == "main_fixed_length_bm25")
        assert fixed_bm25["status"] == "missing"


def test_weight_sweep_requires_exact_canonical_weights() -> None:
    rows = [
        _row(
            "real_hybrid_weighted",
            "fixed_length",
            {"dense_weight": 0.3, "bm25_weight": 0.7},
        )
    ]
    status = assess_matrix(rows, include_weight_sweep=True)
    target = next(item for item in status if item["spec_key"].endswith("dw0.3_bw0.7"))
    other = next(item for item in status if item["spec_key"].endswith("dw0.4_bw0.6"))
    assert target["status"] == "completed"
    assert other["status"] == "missing"

