"""Frozen final retrieval matrix and registry completeness checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


CHUNK_STRATEGIES = ("fixed_length", "article_based", "structure_aware")
RETRIEVAL_METHODS = (
    "bm25",
    "real_dense",
    "real_hybrid_weighted",
    "real_hybrid_rrf",
)
WEIGHT_SWEEP = ((0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3))


@dataclass(frozen=True)
class ExperimentSpec:
    phase: str
    chunk_strategy: str
    retrieval_method: str
    top_k: int = 5
    dense_weight: float | None = None
    bm25_weight: float | None = None
    rrf_k: int | None = None

    @property
    def key(self) -> str:
        suffix = ""
        if self.dense_weight is not None:
            suffix = f"_dw{self.dense_weight:.1f}_bw{self.bm25_weight:.1f}"
        if self.rrf_k is not None:
            suffix = f"_rrfk{self.rrf_k}"
        return f"{self.phase}_{self.chunk_strategy}_{self.retrieval_method}{suffix}"


def final_retrieval_specs(*, include_weight_sweep: bool = False) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for chunk_strategy in CHUNK_STRATEGIES:
        for retrieval_method in RETRIEVAL_METHODS:
            kwargs: dict[str, Any] = {}
            if retrieval_method == "real_hybrid_weighted":
                kwargs.update(dense_weight=0.5, bm25_weight=0.5)
            elif retrieval_method == "real_hybrid_rrf":
                kwargs.update(rrf_k=60)
            specs.append(
                ExperimentSpec(
                    phase="main",
                    chunk_strategy=chunk_strategy,
                    retrieval_method=retrieval_method,
                    **kwargs,
                )
            )
    if include_weight_sweep:
        for dense_weight, bm25_weight in WEIGHT_SWEEP:
            specs.append(
                ExperimentSpec(
                    phase="weight_sweep",
                    chunk_strategy="fixed_length",
                    retrieval_method="real_hybrid_weighted",
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                )
            )
    return specs


def _parameters(row: Mapping[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("hybrid_parameters", "") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _float_equal(left: Any, right: float | None) -> bool:
    if right is None:
        return True
    try:
        return abs(float(left) - right) < 1e-12
    except (TypeError, ValueError):
        return False


def row_satisfies_spec(row: Mapping[str, str], spec: ExperimentSpec) -> bool:
    if not (
        row.get("task_type") == "retrieval"
        and row.get("retrieval_method") == spec.retrieval_method
        and row.get("chunk_strategy") == spec.chunk_strategy
        and row.get("top_k") == str(spec.top_k)
        and row.get("status") == "completed"
        and row.get("legacy") == "false"
        and row.get("git_worktree_clean") == "true"
        and row.get("reproducible") == "full"
        and row.get("has_per_question_results") == "true"
    ):
        return False
    parameters = _parameters(row)
    if spec.dense_weight is not None:
        if not _float_equal(parameters.get("dense_weight"), spec.dense_weight):
            return False
        if not _float_equal(parameters.get("bm25_weight"), spec.bm25_weight):
            return False
    if spec.rrf_k is not None:
        rrf_value = parameters.get("rrf_k", parameters.get("k"))
        if not _float_equal(rrf_value, float(spec.rrf_k)):
            return False
    return True


def assess_matrix(
    rows: Iterable[Mapping[str, str]],
    *,
    include_weight_sweep: bool = False,
) -> list[dict[str, Any]]:
    registry_rows = list(rows)
    assessments: list[dict[str, Any]] = []
    for spec in final_retrieval_specs(include_weight_sweep=include_weight_sweep):
        matches = [row for row in registry_rows if row_satisfies_spec(row, spec)]
        assessments.append(
            {
                "spec_key": spec.key,
                "phase": spec.phase,
                "chunk_strategy": spec.chunk_strategy,
                "retrieval_method": spec.retrieval_method,
                "top_k": spec.top_k,
                "dense_weight": spec.dense_weight,
                "bm25_weight": spec.bm25_weight,
                "rrf_k": spec.rrf_k,
                "status": "completed" if matches else "missing",
                "matching_experiment_id": matches[0].get("experiment_id", "") if matches else "",
            }
        )
    return assessments

