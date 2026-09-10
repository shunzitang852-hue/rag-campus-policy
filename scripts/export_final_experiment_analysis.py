"""Export paper-ready final tables, SVG figures, and per-question error analysis."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.final_matrix import assess_matrix
from src.experiments.manifests import atomic_write_csv, atomic_write_json, atomic_write_text
from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT
from src.experiments.registry import load_registry, rebuild_registry


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _registry_index(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["experiment_id"]: row for row in rows}


def _metrics(row: dict[str, str]) -> dict[str, Any]:
    document = _read_json(row["metrics_path"])
    value: Any = document
    for part in (row.get("metric_selector") or "").split("."):
        if part:
            value = value.get(part, {}) if isinstance(value, dict) else {}
    return value if isinstance(value, dict) else {}


def classify_question(row: dict[str, Any]) -> str:
    rank = row.get("first_relevant_rank")
    if rank == 1:
        return "success_rank1"
    if isinstance(rank, int) and rank > 1:
        return "ranking_not_first"
    pool = row.get("candidate_pool_results", [])
    if any(item.get("is_relevant") for item in pool[5:]):
        return "relevant_below_cutoff"
    if row.get("source_policy_hit"):
        return "same_policy_wrong_chunk"
    return "candidate_recall_failure"


def _md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _svg_bar(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1180, 620
    left, top, plot_w, plot_h = 85, 70, 1040, 430
    group_w = plot_w / len(rows)
    bars = []
    labels = []
    for index, row in enumerate(rows):
        x = left + index * group_w
        for offset, field, color in [(0.16, "Recall@5", "#2F5597"), (0.52, "MRR@5", "#ED7D31")]:
            value = float(row[field])
            bw = group_w * 0.28
            bh = value * plot_h
            bars.append(f'<rect x="{x + group_w*offset:.1f}" y="{top+plot_h-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}"/>')
        label = f"{row['chunk_strategy']}\n{row['retrieval_method']}"
        labels.append(f'<text x="{x+group_w/2:.1f}" y="{top+plot_h+18}" text-anchor="end" transform="rotate(-45 {x+group_w/2:.1f} {top+plot_h+18})" font-size="11">{html.escape(label)}</text>')
    ticks = []
    for value in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - value * plot_h
        ticks.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#D9D9D9"/><text x="{left-10}" y="{y+4}" text-anchor="end" font-size="12">{value:.2f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial, Microsoft YaHei, sans-serif" fill="#222"><text x="{left}" y="34" font-size="20" font-weight="bold">最终检索矩阵：Recall@5 与 MRR@5</text>{''.join(ticks)}{''.join(bars)}{''.join(labels)}<rect x="850" y="28" width="14" height="14" fill="#2F5597"/><text x="870" y="40" font-size="12">Recall@5</text><rect x="950" y="28" width="14" height="14" fill="#ED7D31"/><text x="970" y="40" font-size="12">MRR@5</text></g></svg>'''
    atomic_write_text(path, svg)


def _svg_weight(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 800, 500
    left, top, plot_w, plot_h = 80, 65, 650, 340
    points_recall, points_mrr = [], []
    for index, row in enumerate(rows):
        x = left + index * plot_w / (len(rows) - 1)
        points_recall.append((x, top + plot_h - float(row["Recall@5"]) * plot_h))
        points_mrr.append((x, top + plot_h - float(row["MRR@5"]) * plot_h))
    def points(values: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in values)
    ticks = []
    for value in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - value * plot_h
        ticks.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#D9D9D9"/><text x="{left-10}" y="{y+4}" text-anchor="end" font-size="12">{value:.2f}</text>')
    xlabels = "".join(f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-size="12">{rows[i]["dense_weight"]:.1f}/{rows[i]["bm25_weight"]:.1f}</text>' for i,(x,_) in enumerate(points_recall))
    circles = "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#2F5597"/>' for x,y in points_recall) + "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#ED7D31"/>' for x,y in points_mrr)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial, Microsoft YaHei, sans-serif" fill="#222"><text x="{left}" y="32" font-size="20" font-weight="bold">Fixed-length Hybrid 权重敏感性</text>{''.join(ticks)}<polyline points="{points(points_recall)}" fill="none" stroke="#2F5597" stroke-width="3"/><polyline points="{points(points_mrr)}" fill="none" stroke="#ED7D31" stroke-width="3"/>{circles}{xlabels}<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="13">dense_weight / bm25_weight</text><rect x="535" y="25" width="14" height="14" fill="#2F5597"/><text x="555" y="37" font-size="12">Recall@5</text><rect x="635" y="25" width="14" height="14" fill="#ED7D31"/><text x="655" y="37" font-size="12">MRR@5</text></g></svg>'''
    atomic_write_text(path, svg)


def export_analysis(output_dir: Path) -> dict[str, Any]:
    registry_rows = rebuild_registry(DEFAULT_EXPERIMENT_ROOT)
    index = _registry_index(registry_rows)
    assessment = assess_matrix(registry_rows, include_weight_sweep=True)
    missing = [row for row in assessment if row["status"] == "missing"]
    if missing:
        raise RuntimeError(f"final matrix is incomplete: {len(missing)} missing")
    output_dir.mkdir(parents=True, exist_ok=False)

    final_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for spec in assessment:
        registry_row = index[spec["matching_experiment_id"]]
        metrics = _metrics(registry_row)
        result_row = {
            **{key: spec[key] for key in ("phase", "chunk_strategy", "retrieval_method", "top_k", "dense_weight", "bm25_weight", "rrf_k")},
            "experiment_id": registry_row["experiment_id"],
            "Recall@1": metrics.get("Recall@1", ""),
            "Recall@3": metrics.get("Recall@3", ""),
            "Recall@5": metrics.get("Recall@5", ""),
            "MRR@5": metrics.get("MRR@5", metrics.get("MRR", "")),
            "source_policy_hit_rate": metrics.get("source_policy_hit_rate", ""),
            "related_chunk_hit_rate": metrics.get("related_chunk_hit_rate", ""),
            "runtime_seconds": metrics.get("runtime_seconds", ""),
        }
        if spec["phase"] == "main":
            final_rows.append(result_row)
            for question in _read_jsonl(registry_row["per_question_results_path"]):
                detail_rows.append(
                    {
                        "question_id": question["question_id"],
                        "question": question.get("question", ""),
                        "question_type": question.get("question_type", "unknown"),
                        "difficulty": question.get("difficulty", "unknown"),
                        "chunk_strategy": spec["chunk_strategy"],
                        "retrieval_method": spec["retrieval_method"],
                        "first_relevant_rank": question.get("first_relevant_rank", ""),
                        "hit_at_5": bool(question.get("hit_at_5")),
                        "source_policy_hit": bool(question.get("source_policy_hit")),
                        "error_type": classify_question(question),
                        "experiment_id": registry_row["experiment_id"],
                    }
                )
        else:
            weight_rows.append(result_row)
    weight_rows.sort(key=lambda row: float(row["dense_weight"]))

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["chunk_strategy"], row["retrieval_method"], row["question_type"], row["difficulty"])].append(row)
    error_summary = []
    for key, rows in sorted(grouped.items()):
        hits = [row for row in rows if row["hit_at_5"]]
        error_summary.append(
            {
                "chunk_strategy": key[0], "retrieval_method": key[1], "question_type": key[2], "difficulty": key[3],
                "question_count": len(rows), "hit_count": len(hits), "miss_count": len(rows)-len(hits),
                "Recall@5": len(hits)/len(rows),
                "MRR@5": sum((1/int(row["first_relevant_rank"])) if row["first_relevant_rank"] not in ("", None) else 0 for row in rows)/len(rows),
            }
        )
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        by_question[row["question_id"]].append(row)
    hard_questions = []
    for question_id, rows in by_question.items():
        counts = Counter(row["error_type"] for row in rows)
        hard_questions.append(
            {
                "question_id": question_id,
                "question": rows[0]["question"],
                "question_type": rows[0]["question_type"],
                "difficulty": rows[0]["difficulty"],
                "matrix_run_count": len(rows),
                "miss_count": sum(not row["hit_at_5"] for row in rows),
                "rank1_count": counts["success_rank1"],
                "dominant_error_type": counts.most_common(1)[0][0],
            }
        )
    hard_questions.sort(key=lambda row: (-row["miss_count"], row["rank1_count"], row["question_id"]))

    atomic_write_csv(output_dir / "final_retrieval_matrix.csv", final_rows)
    atomic_write_csv(output_dir / "hybrid_weight_sweep.csv", weight_rows)
    atomic_write_csv(output_dir / "per_question_error_analysis.csv", detail_rows)
    atomic_write_csv(output_dir / "error_summary_by_group.csv", error_summary)
    atomic_write_csv(output_dir / "hard_questions.csv", hard_questions)
    _svg_bar(final_rows, output_dir / "final_retrieval_matrix.svg")
    _svg_weight(weight_rows, output_dir / "hybrid_weight_sweep.svg")

    best = sorted(final_rows, key=lambda row: (-float(row["Recall@5"]), -float(row["MRR@5"]), float(row["runtime_seconds"] or 1e99)))[0]
    chapter = (
        "# 实验结果与误差分析（自动整理稿）\n\n"
        "## 实验设置\n\n最终检索实验固定 QA v1（40 题）、top_k=5、随机种子 42。主指标为 Recall@5，次指标为 MRR@5。真实 Dense 使用本地 BAAI/bge-small-zh-v1.5，Hybrid 候选池深度为 50。\n\n"
        "## 检索结果\n\n" + _md_table(final_rows, ["chunk_strategy", "retrieval_method", "Recall@5", "MRR@5", "source_policy_hit_rate", "runtime_seconds"]) + "\n"
        f"按预注册排序规则，当前首选组合为 `{best['retrieval_method']} + {best['chunk_strategy']}`，Recall@5={float(best['Recall@5']):.3f}，MRR@5={float(best['MRR@5']):.3f}。\n\n"
        "## Hybrid 权重实验\n\n" + _md_table(weight_rows, ["dense_weight", "bm25_weight", "Recall@5", "MRR@5", "runtime_seconds"]) + "\n"
        "## 逐题误差分析\n\n误差按五类登记：首位命中、命中但未排第一、相关块位于 Top 5 之后、命中同一政策但未命中标注块、候选池召回失败。完整逐题结果见 `per_question_error_analysis.csv`，困难问题排序见 `hard_questions.csv`。\n\n"
        "## 有效性限制\n\n自动命中依赖既有相关 Chunk 标注；同一政策内的等价证据可能被严格 Chunk 命中口径判为错误。回答质量的自动重合指标不等于人工正确率，人工复核字段未完成前不报告人工准确率。\n"
    )
    atomic_write_text(output_dir / "experiment_chapter_draft.md", chapter)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(DEFAULT_EXPERIMENT_ROOT / "experiment_registry.csv"),
        "registry_row_count": len(registry_rows),
        "main_matrix_count": len(final_rows),
        "weight_sweep_count": len(weight_rows),
        "per_question_row_count": len(detail_rows),
        "missing_count": 0,
        "best_retrieval": best,
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    atomic_write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or DEFAULT_EXPERIMENT_ROOT / "final_analysis" / stamp
    manifest = export_analysis(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

