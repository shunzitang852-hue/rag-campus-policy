"""Export thesis tables strictly from registered, persisted experiment results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .manifests import atomic_write_csv, atomic_write_json, atomic_write_text
from .recorder import DEFAULT_EXPERIMENT_ROOT
from .registry import load_registry


TABLE_SPECS: dict[str, list[str]] = {
    "chunk_strategy_comparison": [
        "experiment_id",
        "retrieval_method",
        "chunk_strategy",
        "top_k",
        "question_count",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "MRR",
        "MRR@10",
        "legacy",
    ],
    "retrieval_method_comparison": [
        "experiment_id",
        "retrieval_method",
        "chunk_strategy",
        "embedding_model",
        "top_k",
        "Recall@5",
        "MRR",
        "MRR@10",
        "average_top1_score",
        "no_hit_count",
        "legacy",
    ],
    "hybrid_weight_comparison": [
        "experiment_id",
        "retrieval_method",
        "chunk_strategy",
        "top_k",
        "dense_weight",
        "real_dense_weight",
        "bm25_weight",
        "rrf_k",
        "normalization",
        "Recall@5",
        "MRR",
        "MRR@10",
        "legacy",
    ],
    "rag_version_comparison": [
        "experiment_id",
        "rag_version",
        "retrieval_method",
        "chunk_strategy",
        "top_k",
        "question_count",
        "average_keyword_overlap_ratio",
        "average_evidence_overlap_ratio",
        "needs_review_count",
        "good_count",
        "legacy",
    ],
    "error_type_statistics": [
        "experiment_id",
        "rag_version",
        "error_type",
        "count",
        "legacy",
    ],
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _select(document: Mapping[str, Any], selector: str) -> dict[str, Any]:
    current: Any = document
    for part in (selector or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict):
            return {}
        current = current.get(part, {})
    return current if isinstance(current, dict) else {}


def _metrics_for(row: Mapping[str, str]) -> dict[str, Any]:
    path_text = row.get("metrics_path", "")
    if not path_text:
        return {}
    document = _read_json(Path(path_text))
    return _select(document, row.get("metric_selector", ""))


def _value(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key, "")
    return "" if value in (None, "unknown") else value


def _json_mapping(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text) if text else {}
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _rag_version(row: Mapping[str, str], config: Mapping[str, Any]) -> str:
    parameters = config.get("parameters", {})
    if isinstance(parameters, dict) and parameters.get("generation_version"):
        return str(parameters["generation_version"])
    if isinstance(parameters, dict) and parameters.get("rag_version"):
        return str(parameters["rag_version"])
    combined = " ".join(
        [
            row.get("experiment_id", ""),
            row.get("embedding_model", ""),
            row.get("generation_script", ""),
        ]
    ).lower()
    for version in ("v3", "v2", "v1"):
        if version in combined:
            return version
    return ""


def _config_for(row: Mapping[str, str]) -> dict[str, Any]:
    path_text = row.get("config_path", "")
    if not path_text or path_text == "unknown":
        return {}
    path = Path(path_text)
    return _read_json(path) if path.suffix.lower() == ".json" else {}


def build_thesis_tables(
    registry_rows: Sequence[dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    tables = {name: [] for name in TABLE_SPECS}
    missing: list[dict[str, str]] = []
    for row in registry_rows:
        metrics = _metrics_for(row)
        config = _config_for(row)
        task = row.get("task_type", "")
        if task == "retrieval":
            common = {
                "experiment_id": row.get("experiment_id", ""),
                "retrieval_method": row.get("retrieval_method", ""),
                "chunk_strategy": row.get("chunk_strategy", ""),
                "embedding_model": row.get("embedding_model", ""),
                "top_k": row.get("top_k", ""),
                "question_count": _value(metrics, "question_count"),
                "Recall@1": _value(metrics, "Recall@1"),
                "Recall@3": _value(metrics, "Recall@3"),
                "Recall@5": _value(metrics, "Recall@5"),
                "Recall@10": _value(metrics, "Recall@10"),
                "MRR": (
                    _value(metrics, "MRR")
                    or _value(metrics, "MRR@5")
                    or _value(metrics, "MRR@10")
                ),
                "MRR@10": _value(metrics, "MRR@10"),
                "average_top1_score": _value(metrics, "average_top1_score"),
                "no_hit_count": _value(metrics, "no_hit_count"),
                "legacy": row.get("legacy", ""),
            }
            tables["chunk_strategy_comparison"].append(common)
            tables["retrieval_method_comparison"].append(common)
            if "hybrid" in row.get("retrieval_method", ""):
                parameters = _json_mapping(row.get("hybrid_parameters", ""))
                tables["hybrid_weight_comparison"].append(
                    {
                        **common,
                        "dense_weight": _value(parameters, "dense_weight"),
                        "real_dense_weight": _value(
                            parameters, "real_dense_weight"
                        ),
                        "bm25_weight": _value(parameters, "bm25_weight"),
                        "rrf_k": _value(parameters, "k"),
                        "normalization": _value(parameters, "normalization"),
                    }
                )
        if task == "rag_answer_quality":
            distribution = metrics.get("auto_quality_distribution", {})
            if not isinstance(distribution, dict):
                distribution = {}
            tables["rag_version_comparison"].append(
                {
                    "experiment_id": row.get("experiment_id", ""),
                    "rag_version": _rag_version(row, config),
                    "retrieval_method": row.get("retrieval_method", ""),
                    "chunk_strategy": row.get("chunk_strategy", ""),
                    "top_k": row.get("top_k", ""),
                    "question_count": _value(metrics, "question_count")
                    or _value(metrics, "total_questions"),
                    "average_keyword_overlap_ratio": _value(
                        metrics, "average_keyword_overlap_ratio"
                    ),
                    "average_evidence_overlap_ratio": _value(
                        metrics, "average_evidence_overlap_ratio"
                    ),
                    "needs_review_count": _value(metrics, "needs_review_count")
                    or _value(distribution, "needs_review"),
                    "good_count": _value(distribution, "good"),
                    "legacy": row.get("legacy", ""),
                }
            )
        if task == "error_analysis":
            distribution = metrics.get("error_type_distribution", {})
            if isinstance(distribution, dict):
                version = _rag_version(row, config) or "v2"
                for error_type, count in distribution.items():
                    tables["error_type_statistics"].append(
                        {
                            "experiment_id": row.get("experiment_id", ""),
                            "rag_version": version,
                            "error_type": error_type,
                            "count": count,
                            "legacy": row.get("legacy", ""),
                        }
                    )

    for table_name, rows in tables.items():
        fields = TABLE_SPECS[table_name]
        rows.sort(
            key=lambda item: tuple(str(item.get(field, "")) for field in fields)
        )
        for index, table_row in enumerate(rows, start=1):
            for field in fields:
                if table_row.get(field) in (None, "", "unknown"):
                    missing.append(
                        {
                            "table": table_name,
                            "row": str(index),
                            "experiment_id": str(
                                table_row.get("experiment_id", "")
                            ),
                            "field": field,
                            "reason": "not present in registered experiment output",
                        }
                    )
    return tables, missing


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> str:
    def cell(value: Any) -> str:
        if value in (None, "unknown"):
            return ""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(field, "")) for field in fields) + " |")
    if not rows:
        lines.append("| " + " | ".join("" for _ in fields) + " |")
    return "\n".join(lines) + "\n"


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def export_thesis_tables(
    registry_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir)
    rows = load_registry(registry_path)
    tables, missing = build_thesis_tables(rows)
    output_files: dict[str, str] = {}
    for table_name, table_rows in tables.items():
        fields = TABLE_SPECS[table_name]
        csv_path = output_dir / f"{table_name}.csv"
        markdown_path = output_dir / f"{table_name}.md"
        atomic_write_csv(csv_path, table_rows, fields)
        atomic_write_text(
            markdown_path,
            f"# {table_name}\n\n"
            "本表仅使用实验总账所指向的持久化真实结果；空白表示源结果缺失。\n\n"
            + _markdown_table(table_rows, fields),
        )
        output_files[f"{table_name}_csv"] = str(csv_path)
        output_files[f"{table_name}_markdown"] = str(markdown_path)

    missing_path = output_dir / "missing_data_report.md"
    missing_lines = [
        "# 论文表格缺失数据报告",
        "",
        f"- 总账路径：`{registry_path}`",
        f"- 缺失单元格数：{len(missing)}",
        "- 处理规则：保留空白，不估算、不补造。",
        "",
        _markdown_table(
            missing,
            ["table", "row", "experiment_id", "field", "reason"],
        ).rstrip(),
        "",
    ]
    atomic_write_text(missing_path, "\n".join(missing_lines))
    output_files["missing_data_report"] = str(missing_path)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path),
        "registry_row_count": len(rows),
        "table_row_counts": {
            name: len(table_rows) for name, table_rows in tables.items()
        },
        "missing_cell_count": len(missing),
        "source_policy": "persisted registered results only; no estimates",
        "output_files": output_files,
    }
    manifest["output_files"]["manifest"] = str(
        output_dir / "thesis_tables_manifest.json"
    )
    atomic_write_json(output_dir / "thesis_tables_manifest.json", manifest)
    return manifest


def default_export_dir(
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    timestamp: datetime | None = None,
) -> Path:
    current = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (
        Path(experiment_root)
        / "thesis_exports"
        / current.strftime("%Y%m%dT%H%M%S%fZ")
    )
