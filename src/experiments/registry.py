"""Rebuild the experiment ledger from normalized runs and immutable legacy outputs."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifests import atomic_write_csv
from .recorder import DEFAULT_EXPERIMENT_ROOT


REGISTRY_FIELDS = [
    "experiment_id",
    "status",
    "task_type",
    "retrieval_method",
    "chunk_strategy",
    "embedding_model",
    "top_k",
    "qa_version",
    "hybrid_parameters",
    "random_seed",
    "started_at",
    "completed_at",
    "git_commit",
    "git_worktree_clean",
    "metric_fields",
    "has_per_question_results",
    "reproducible",
    "metadata_missing",
    "legacy",
    "generation_script",
    "config_path",
    "metrics_path",
    "per_question_results_path",
    "run_dir",
    "source_artifact",
    "metric_selector",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _json_text(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _metric_fields(metrics: Mapping[str, Any]) -> str:
    return "|".join(sorted(str(key) for key in metrics))


def _missing_normalized_metadata(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    environment: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> list[str]:
    missing: list[str] = []
    for field in (
        "task_type",
        "retrieval_method",
        "chunk_strategy",
        "top_k",
        "qa_version",
        "random_seed",
        "started_at",
    ):
        if config.get(field) in (None, "", "unknown"):
            missing.append(field)
    if metrics.get("completed_at") in (None, "", "unknown"):
        missing.append("completed_at")
    if environment.get("python_version") in (None, "", "unknown"):
        missing.append("python_version")
    if not environment.get("dependencies"):
        missing.append("dependency_versions")
    if git_state.get("git_commit") in (None, "", "unknown"):
        missing.append("git_commit")
    if git_state.get("git_worktree_clean") in (None, "", "unknown"):
        missing.append("git_worktree_clean")
    inputs = data_manifest.get("inputs", {})
    for role in ("data_source", "chunk_file", "qa_file"):
        item = inputs.get(role, {})
        if item.get("path") in (None, "", "unknown"):
            missing.append(f"{role}_path")
        if item.get("sha256") in (None, "", "unknown"):
            missing.append(f"{role}_sha256")
    return missing


def _scan_run(run_dir: Path) -> dict[str, str]:
    config = _read_json(run_dir / "config.json")
    metrics_document = _read_json(run_dir / "metrics.json")
    metrics = metrics_document.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    environment = _read_json(run_dir / "environment.json")
    data_manifest = _read_json(run_dir / "data_manifest.json")
    commit_path = run_dir / "git_commit.txt"
    commit = (
        commit_path.read_text(encoding="utf-8").strip()
        if commit_path.exists()
        else "unknown"
    )
    log_text = (
        (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "run.log").exists()
        else ""
    )
    clean_match = re.search(r"git_worktree_clean=(true|false|unknown)", log_text, re.I)
    git_state = {
        "git_commit": commit or "unknown",
        "git_worktree_clean": (
            clean_match.group(1).lower() if clean_match else "unknown"
        ),
    }
    missing = _missing_normalized_metadata(
        config,
        metrics_document,
        environment,
        data_manifest,
        git_state,
    )
    per_question = run_dir / "per_question_results.jsonl"
    return _registry_row(
        experiment_id=str(config.get("experiment_id") or run_dir.name),
        status=str(metrics_document.get("status") or "unknown"),
        task_type=str(config.get("task_type") or "unknown"),
        retrieval_method=str(config.get("retrieval_method") or "unknown"),
        chunk_strategy=str(config.get("chunk_strategy") or "unknown"),
        embedding_model=str(config.get("embedding_model") or "unknown"),
        top_k=config.get("top_k"),
        qa_version=str(config.get("qa_version") or "unknown"),
        hybrid_parameters=config.get("hybrid_parameters", {}),
        random_seed=config.get("random_seed"),
        started_at=str(config.get("started_at") or "unknown"),
        completed_at=str(metrics_document.get("completed_at") or "unknown"),
        git_commit=commit or "unknown",
        git_worktree_clean=git_state["git_worktree_clean"],
        metric_fields=_metric_fields(metrics),
        has_per_question_results=per_question.exists(),
        reproducible=(
            "full"
            if not missing and git_state["git_worktree_clean"] == "true"
            else "partial"
        ),
        metadata_missing=missing,
        legacy=False,
        generation_script=str(config.get("generation_script") or "unknown"),
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.json",
        per_question_results_path=per_question,
        run_dir=run_dir,
        source_artifact=run_dir,
        metric_selector="metrics",
    )


def _registry_row(**values: Any) -> dict[str, str]:
    row = {field: "" for field in REGISTRY_FIELDS}
    for field, value in values.items():
        if field not in row:
            continue
        if field in {"hybrid_parameters", "metadata_missing"}:
            row[field] = _json_text(value)
        elif isinstance(value, bool):
            row[field] = str(value).lower()
        elif value is None:
            row[field] = ""
        else:
            row[field] = str(value)
    return row


def _legacy_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()


def _legacy_missing(report: Mapping[str, Any], *, seed_known: bool = False) -> list[str]:
    missing = [
        "python_version",
        "dependency_versions",
        "git_commit",
        "git_worktree_clean",
        "run_started_at",
    ]
    if not seed_known:
        missing.append("random_seed")
    if not report.get("raw_sha256") and not report.get("cleaned_sha256"):
        missing.extend(["data_source_path", "data_source_sha256"])
    if not report.get("chunk_file_sha256"):
        missing.append("chunk_file_sha256")
    if not report.get("qa_jsonl_sha256"):
        missing.append("qa_file_sha256")
    return missing


def _legacy_id(*parts: Any) -> str:
    text = "-".join(str(part) for part in parts if part not in (None, ""))
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return f"legacy-{slug}"


def _retrieval_legacy_rows(root: Path) -> list[dict[str, str]]:
    specs = [
        {
            "report": root / "dense_retrieval" / "evaluation" / "dense_search_evaluation_report.json",
            "method": "dense_hash",
            "model": "deterministic_hash_embedding",
            "script": "scripts/evaluate_dense_retrieval.py",
            "config": "configs/retrieval_dense.yaml",
            "prefix": "dense_search_results",
            "nested": False,
        },
        {
            "report": root / "bm25_retrieval" / "evaluation" / "bm25_evaluation_report.json",
            "method": "bm25",
            "model": "cjk_char_bigram",
            "script": "scripts/evaluate_bm25_retrieval.py",
            "config": "configs/retrieval_bm25.yaml",
            "prefix": "bm25_search_results",
            "nested": False,
        },
        {
            "report": root / "hybrid_retrieval" / "evaluation" / "hybrid_evaluation_report.json",
            "method": "hybrid",
            "model": "deterministic_hash_embedding+bm25",
            "script": "scripts/evaluate_hybrid_retrieval.py",
            "config": "configs/retrieval_hybrid.yaml",
            "prefix": "hybrid_search_results",
            "nested": True,
        },
        {
            "report": root / "dense_retrieval_real_embedding" / "evaluation" / "real_embedding_dense_evaluation_report.json",
            "method": "real_dense",
            "model": "BAAI/bge-small-zh-v1.5",
            "script": "scripts/evaluate_real_embedding_dense_retrieval.py",
            "config": "configs/retrieval_dense_real_embedding.yaml",
            "prefix": "real_embedding_dense_search_results",
            "nested": False,
        },
        {
            "report": root / "hybrid_retrieval_real_embedding" / "evaluation" / "real_hybrid_evaluation_report.json",
            "method": "real_hybrid",
            "model": "BAAI/bge-small-zh-v1.5+bm25",
            "script": "scripts/evaluate_real_embedding_hybrid_retrieval.py",
            "config": "configs/retrieval_hybrid_real_embedding.yaml",
            "prefix": "real_hybrid_search_results",
            "nested": True,
        },
    ]
    project_root = Path(__file__).resolve().parents[2]
    rows: list[dict[str, str]] = []
    for spec in specs:
        report_path = Path(spec["report"])
        if not report_path.exists():
            continue
        report = _read_json(report_path)
        metrics_by_strategy = report.get("metrics_by_strategy", {})
        if not isinstance(metrics_by_strategy, dict):
            continue
        for strategy, value in metrics_by_strategy.items():
            method_metrics = value if isinstance(value, dict) else {}
            if spec["nested"]:
                items = method_metrics.items()
            else:
                items = [(spec["method"], method_metrics)]
            for method_name, metrics in items:
                if not isinstance(metrics, dict):
                    continue
                retrieval_method = (
                    f"{spec['method']}_{method_name}"
                    if spec["nested"]
                    else str(spec["method"])
                )
                per_question = report_path.parent / (
                    f"{spec['prefix']}_{strategy}_{method_name}.jsonl"
                    if spec["nested"]
                    else f"{spec['prefix']}_{strategy}.jsonl"
                )
                hybrid_parameters: dict[str, Any] = {}
                if spec["nested"]:
                    if method_name == "weighted_score":
                        hybrid_parameters = dict(report.get("weighted_score", {}))
                    elif method_name == "rrf":
                        hybrid_parameters = dict(report.get("rrf", {}))
                top_k = report.get("top_k")
                if top_k in (None, ""):
                    values = report.get("top_k_values", [])
                    top_k = max(values) if values else "unknown"
                missing = _legacy_missing(report)
                rows.append(
                    _registry_row(
                        experiment_id=_legacy_id(
                            "retrieval",
                            retrieval_method,
                            strategy,
                            f"topk{top_k}",
                        ),
                        status="completed",
                        task_type="retrieval",
                        retrieval_method=retrieval_method,
                        chunk_strategy=strategy,
                        embedding_model=spec["model"],
                        top_k=top_k,
                        qa_version="v1",
                        hybrid_parameters=hybrid_parameters,
                        random_seed="unknown",
                        started_at="unknown",
                        completed_at=_legacy_timestamp(report_path),
                        git_commit="unknown",
                        git_worktree_clean="unknown",
                        metric_fields=_metric_fields(metrics),
                        has_per_question_results=per_question.exists(),
                        reproducible="partial",
                        metadata_missing=missing,
                        legacy=True,
                        generation_script=spec["script"],
                        config_path=project_root / str(spec["config"]),
                        metrics_path=report_path,
                        per_question_results_path=per_question if per_question.exists() else "",
                        run_dir=report_path.parent.parent,
                        source_artifact=report_path,
                        metric_selector=(
                            f"metrics_by_strategy.{strategy}.{method_name}"
                            if spec["nested"]
                            else f"metrics_by_strategy.{strategy}"
                        ),
                    )
                )
    return rows


def _rag_legacy_rows(root: Path) -> list[dict[str, str]]:
    project_root = Path(__file__).resolve().parents[2]
    specs = [
        {
            "id": ("rag", "v1", "batch"),
            "report": root / "rag_evaluation" / "rag_evaluation_report.json",
            "task": "rag_batch_evaluation",
            "method": "real_hybrid_weighted",
            "strategy": "fixed_length",
            "model": "BAAI/bge-small-zh-v1.5+bm25",
            "top_k": 5,
            "qa": "v1",
            "script": "scripts/evaluate_rag_qa.py",
            "config": "configs/rag_evaluation.yaml",
            "selector": "default_report",
            "per_question": root / "rag_evaluation" / "rag_answers_real_hybrid_weighted_fixed_length.jsonl",
        },
        {
            "id": ("rag", "v1", "quality"),
            "report": root / "rag_answer_quality" / "rag_answer_quality_report.json",
            "task": "rag_answer_quality",
            "method": "real_hybrid_weighted",
            "strategy": "fixed_length",
            "model": "rule_based_extractive_v1",
            "top_k": 5,
            "qa": "v1",
            "script": "scripts/evaluate_rag_answer_quality.py",
            "config": "configs/rag_answer_quality.yaml",
            "selector": "summary",
            "per_question": root / "rag_answer_quality" / "rag_answer_quality_review_v1.jsonl",
        },
        {
            "id": ("rag", "v2", "quality"),
            "report": root / "rag_answer_quality_improvement" / "rag_answer_quality_improvement_report.json",
            "task": "rag_answer_quality",
            "method": "real_hybrid_weighted",
            "strategy": "fixed_length",
            "model": "rule_based_extractive_v2",
            "top_k": 5,
            "qa": "v1",
            "script": "scripts/evaluate_rag_answer_quality_improvement.py",
            "config": "configs/rag_answer_generation.yaml",
            "selector": "v2_summary",
            "per_question": root / "rag_answer_quality_improvement" / "rag_answer_quality_review_v2.jsonl",
        },
        {
            "id": ("rag", "v2", "error-analysis"),
            "report": root / "rag_answer_quality_error_analysis" / "rag_answer_error_analysis_v2_report.json",
            "task": "error_analysis",
            "method": "real_hybrid_weighted",
            "strategy": "fixed_length",
            "model": "rule_based_extractive_v2",
            "top_k": 5,
            "qa": "v1",
            "script": "scripts/analyze_rag_answer_errors_v2.py",
            "config": "unknown",
            "selector": "",
            "per_question": root / "rag_answer_quality_error_analysis" / "rag_v2_needs_review_manual_check.csv",
        },
        {
            "id": ("rag", "v3", "quality"),
            "report": root / "rag_answer_quality_v3_optimization" / "rag_answer_quality_v3_report.json",
            "task": "rag_answer_quality",
            "method": "real_hybrid_weighted",
            "strategy": "fixed_length",
            "model": "rule_based_extractive_v3",
            "top_k": 5,
            "qa": "v1",
            "script": "scripts/review_rag_answer_quality_v3.py",
            "config": "unknown",
            "selector": "",
            "per_question": root / "rag_answer_quality_v3_optimization" / "rag_answer_quality_review_v3.jsonl",
        },
        {
            "id": ("rag", "v2-v3", "comparison"),
            "report": root / "rag_answer_quality_v3_optimization" / "rag_answer_quality_comparison_v2_v3_report.json",
            "task": "rag_answer_quality_comparison",
            "method": "real_hybrid_weighted",
            "strategy": "fixed_length",
            "model": "rule_based_extractive_v2_vs_v3",
            "top_k": 5,
            "qa": "v1",
            "script": "scripts/compare_rag_answer_quality_v2_v3.py",
            "config": "unknown",
            "selector": "",
            "per_question": root / "rag_answer_quality_v3_optimization" / "rag_answer_quality_comparison_v2_v3.jsonl",
        },
    ]
    rows: list[dict[str, str]] = []
    for spec in specs:
        report_path = Path(spec["report"])
        if not report_path.exists():
            continue
        report = _read_json(report_path)
        selected = _select_mapping(report, str(spec["selector"]))
        missing = _legacy_missing(report)
        if spec["config"] == "unknown":
            missing.append("config_version")
        rows.append(
            _registry_row(
                experiment_id=_legacy_id(*spec["id"]),
                status="completed",
                task_type=spec["task"],
                retrieval_method=spec["method"],
                chunk_strategy=spec["strategy"],
                embedding_model=spec["model"],
                top_k=spec["top_k"],
                qa_version=spec["qa"],
                hybrid_parameters={"real_dense_weight": 0.5, "bm25_weight": 0.5},
                random_seed="unknown",
                started_at="unknown",
                completed_at=_legacy_timestamp(report_path),
                git_commit="unknown",
                git_worktree_clean="unknown",
                metric_fields=_metric_fields(selected),
                has_per_question_results=Path(spec["per_question"]).exists(),
                reproducible="partial",
                metadata_missing=missing,
                legacy=True,
                generation_script=spec["script"],
                config_path=(
                    project_root / str(spec["config"])
                    if spec["config"] != "unknown"
                    else "unknown"
                ),
                metrics_path=report_path,
                per_question_results_path=spec["per_question"],
                run_dir=report_path.parent,
                source_artifact=report_path,
                metric_selector=spec["selector"],
            )
        )
    return rows


def _select_mapping(document: Mapping[str, Any], selector: str) -> dict[str, Any]:
    current: Any = document
    if selector:
        for part in selector.split("."):
            if not isinstance(current, dict):
                return {}
            current = current.get(part, {})
    return current if isinstance(current, dict) else {}


def discover_legacy_experiments(experiment_root: Path) -> list[dict[str, str]]:
    root = Path(experiment_root)
    rows = _retrieval_legacy_rows(root)
    rows.extend(_rag_legacy_rows(root))
    return rows


def scan_registry(
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    include_legacy: bool = True,
) -> list[dict[str, str]]:
    root = Path(experiment_root)
    rows: list[dict[str, str]] = []
    runs_root = root / "runs"
    if runs_root.exists():
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            rows.append(_scan_run(run_dir))
    if include_legacy:
        rows.extend(discover_legacy_experiments(root))
    rows.sort(
        key=lambda row: (
            row.get("completed_at", ""),
            row.get("experiment_id", ""),
        ),
        reverse=True,
    )
    return rows


def rebuild_registry(
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    registry_path: Path | None = None,
    *,
    include_legacy: bool = True,
) -> list[dict[str, str]]:
    root = Path(experiment_root)
    destination = Path(registry_path) if registry_path else root / "experiment_registry.csv"
    rows = scan_registry(root, include_legacy=include_legacy)
    atomic_write_csv(destination, rows, REGISTRY_FIELDS)
    return rows


def load_registry(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_registry(
    rows: Iterable[dict[str, str]],
    *,
    task_type: str | None = None,
    retrieval_method: str | None = None,
    chunk_strategy: str | None = None,
    experiment_id: str | None = None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (task_type is None or row.get("task_type") == task_type)
        and (
            retrieval_method is None
            or row.get("retrieval_method") == retrieval_method
        )
        and (
            chunk_strategy is None
            or row.get("chunk_strategy") == chunk_strategy
        )
        and (
            experiment_id is None
            or row.get("experiment_id") == experiment_id
        )
    ]
