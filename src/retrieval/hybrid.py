"""Hybrid retrieval fusion utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.retrieval.dense_index import load_jsonl
from src.retrieval.evaluation import relevant_ids_for_strategy


def normalize_scores(scores: list[float], equal_score_value: float = 0.0) -> list[float]:
    """Normalize one query's scores with min-max scaling."""
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [equal_score_value for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _index_by_chunk(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["retrieved_chunk_id"]: row for row in rows}


def _normalized_score_map(rows: list[dict[str, Any]], equal_score_value: float = 0.0) -> dict[str, float]:
    scores = [float(row.get("score", 0.0)) for row in rows]
    normalized = normalize_scores(scores, equal_score_value=equal_score_value)
    return {
        row["retrieved_chunk_id"]: normalized_score
        for row, normalized_score in zip(rows, normalized)
    }


def fuse_weighted_score(
    question: dict[str, Any],
    strategy: str,
    dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    dense_weight: float = 0.4,
    bm25_weight: float = 0.6,
    top_k: int = 10,
    equal_score_value: float = 0.0,
) -> list[dict[str, Any]]:
    dense_by_chunk = _index_by_chunk(dense_rows)
    bm25_by_chunk = _index_by_chunk(bm25_rows)
    dense_norm = _normalized_score_map(dense_rows, equal_score_value=equal_score_value)
    bm25_norm = _normalized_score_map(bm25_rows, equal_score_value=equal_score_value)
    return _build_fused_rows(
        question=question,
        strategy=strategy,
        fusion_method="weighted_score",
        candidate_ids=set(dense_by_chunk) | set(bm25_by_chunk),
        dense_by_chunk=dense_by_chunk,
        bm25_by_chunk=bm25_by_chunk,
        score_fn=lambda chunk_id: (
            dense_weight * dense_norm.get(chunk_id, 0.0)
            + bm25_weight * bm25_norm.get(chunk_id, 0.0)
        ),
        top_k=top_k,
    )


def fuse_rrf(
    question: dict[str, Any],
    strategy: str,
    dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    dense_by_chunk = _index_by_chunk(dense_rows)
    bm25_by_chunk = _index_by_chunk(bm25_rows)

    def score(chunk_id: str) -> float:
        value = 0.0
        if chunk_id in dense_by_chunk:
            value += 1.0 / (k + int(dense_by_chunk[chunk_id]["rank"]))
        if chunk_id in bm25_by_chunk:
            value += 1.0 / (k + int(bm25_by_chunk[chunk_id]["rank"]))
        return value

    return _build_fused_rows(
        question=question,
        strategy=strategy,
        fusion_method="rrf",
        candidate_ids=set(dense_by_chunk) | set(bm25_by_chunk),
        dense_by_chunk=dense_by_chunk,
        bm25_by_chunk=bm25_by_chunk,
        score_fn=score,
        top_k=top_k,
    )


def _build_fused_rows(
    question: dict[str, Any],
    strategy: str,
    fusion_method: str,
    candidate_ids: set[str],
    dense_by_chunk: dict[str, dict[str, Any]],
    bm25_by_chunk: dict[str, dict[str, Any]],
    score_fn: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    relevant_ids = relevant_ids_for_strategy(question, strategy)
    scored = []
    for chunk_id in candidate_ids:
        dense_row = dense_by_chunk.get(chunk_id)
        bm25_row = bm25_by_chunk.get(chunk_id)
        source_row = dense_row or bm25_row or {}
        dense_score = float(dense_row.get("score", 0.0)) if dense_row else 0.0
        bm25_score = float(bm25_row.get("score", 0.0)) if bm25_row else 0.0
        hybrid_score = float(score_fn(chunk_id))
        scored.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "strategy": strategy,
                "fusion_method": fusion_method,
                "retrieved_chunk_id": chunk_id,
                "retrieved_document_title": source_row.get("retrieved_document_title", ""),
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score,
                "is_relevant": chunk_id in relevant_ids,
                "matched_by": "related_chunk_ids" if chunk_id in relevant_ids else "",
            }
        )
    scored.sort(
        key=lambda row: (
            -row["hybrid_score"],
            -row["bm25_score"],
            -row["dense_score"],
            row["retrieved_chunk_id"],
        )
    )
    for rank, row in enumerate(scored[:top_k], start=1):
        row["rank"] = rank
    return scored[:top_k]


def group_rows_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["rank"]))
    return dict(grouped)


def _first_relevant_rank(rows: list[dict[str, Any]], limit: int = 10) -> int | None:
    for row in rows[:limit]:
        if row["is_relevant"]:
            return int(row["rank"])
    return None


def recall_at_k(rows_by_question: dict[str, list[dict[str, Any]]], k: int, question_count: int) -> float:
    if question_count == 0:
        return 0.0
    hits = sum(1 for rows in rows_by_question.values() if _first_relevant_rank(rows, k) is not None)
    return hits / question_count


def mrr_at_10(rows_by_question: dict[str, list[dict[str, Any]]], question_count: int) -> float:
    if question_count == 0:
        return 0.0
    total = 0.0
    for rows in rows_by_question.values():
        rank = _first_relevant_rank(rows, 10)
        if rank:
            total += 1.0 / rank
    return total / question_count


def evaluate_hybrid_strategy(
    strategy: str,
    fusion_method: str,
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_question = group_rows_by_question(rows)
    question_by_id = {question["question_id"]: question for question in questions}
    question_count = len(questions)
    first_ranks = {
        question["question_id"]: _first_relevant_rank(rows_by_question.get(question["question_id"], []), 10)
        for question in questions
    }
    top1_scores = [
        values[0]["hybrid_score"]
        for values in rows_by_question.values()
        if values
    ]
    relevant_ranks = [rank for rank in first_ranks.values() if rank is not None]
    metrics: dict[str, Any] = {
        "strategy": strategy,
        "fusion_method": fusion_method,
        "question_count": question_count,
        "Recall@1": recall_at_k(rows_by_question, 1, question_count),
        "Recall@3": recall_at_k(rows_by_question, 3, question_count),
        "Recall@5": recall_at_k(rows_by_question, 5, question_count),
        "Recall@10": recall_at_k(rows_by_question, 10, question_count),
        "MRR@10": mrr_at_10(rows_by_question, question_count),
        "average_top1_score": sum(top1_scores) / len(top1_scores) if top1_scores else 0.0,
        "average_relevant_rank": sum(relevant_ranks) / len(relevant_ranks) if relevant_ranks else None,
        "no_hit_count": sum(1 for rank in first_ranks.values() if rank is None),
        "question_type_recall_at_5": {},
        "difficulty_recall_at_5": {},
    }
    for group_field, output_key in [
        ("question_type", "question_type_recall_at_5"),
        ("difficulty", "difficulty_recall_at_5"),
    ]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        for question in questions:
            grouped[question[group_field]][question["question_id"]] = rows_by_question.get(question["question_id"], [])
        metrics[output_key] = {
            group: recall_at_k(group_rows, 5, len(group_rows))
            for group, group_rows in grouped.items()
        }
    return metrics


def load_strategy_results(evaluation_dir: Path, prefix: str, strategy: str) -> dict[str, list[dict[str, Any]]]:
    path = evaluation_dir / f"{prefix}_search_results_{strategy}.jsonl"
    rows = load_jsonl(path)
    grouped = group_rows_by_question(rows)
    return grouped
