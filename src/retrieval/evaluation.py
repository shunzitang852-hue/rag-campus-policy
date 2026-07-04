"""Evaluation metrics for dense retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def relevant_ids_for_strategy(question: dict[str, Any], strategy: str) -> set[str]:
    key = f"related_chunk_ids_{strategy}"
    value = question.get(key, [])
    if isinstance(value, str):
        return {item for item in value.split(";") if item}
    return set(value or [])


def annotate_results(question: dict[str, Any], strategy: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant_ids = relevant_ids_for_strategy(question, strategy)
    annotated = []
    for result in results:
        is_relevant = result["retrieved_chunk_id"] in relevant_ids
        annotated.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "strategy": strategy,
                "rank": result["rank"],
                "retrieved_chunk_id": result["retrieved_chunk_id"],
                "retrieved_document_title": result["retrieved_document_title"],
                "score": result["score"],
                "is_relevant": is_relevant,
                "matched_by": "related_chunk_ids" if is_relevant else "",
            }
        )
    return annotated


def _first_relevant_rank(rows: list[dict[str, Any]], limit: int = 10) -> int | None:
    for row in rows[:limit]:
        if row["is_relevant"]:
            return int(row["rank"])
    return None


def recall_at_k(rows_by_question: dict[str, list[dict[str, Any]]], k: int) -> float:
    if not rows_by_question:
        return 0.0
    hits = sum(1 for rows in rows_by_question.values() if _first_relevant_rank(rows, k) is not None)
    return hits / len(rows_by_question)


def mrr_at_10(rows_by_question: dict[str, list[dict[str, Any]]]) -> float:
    if not rows_by_question:
        return 0.0
    total = 0.0
    for rows in rows_by_question.values():
        rank = _first_relevant_rank(rows, 10)
        if rank:
            total += 1.0 / rank
    return total / len(rows_by_question)


def evaluate_strategy(
    strategy: str,
    annotated_rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    top_k_values: list[int],
) -> dict[str, Any]:
    rows_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated_rows:
        rows_by_question[row["question_id"]].append(row)
    for rows in rows_by_question.values():
        rows.sort(key=lambda item: int(item["rank"]))

    first_ranks = {
        question_id: _first_relevant_rank(rows, 10)
        for question_id, rows in rows_by_question.items()
    }
    question_by_id = {question["question_id"]: question for question in questions}
    top1_scores = [
        rows[0]["score"]
        for rows in rows_by_question.values()
        if rows
    ]
    relevant_ranks = [rank for rank in first_ranks.values() if rank is not None]
    metrics: dict[str, Any] = {
        "strategy": strategy,
        "question_count": len(questions),
        "Recall@1": recall_at_k(rows_by_question, 1),
        "Recall@3": recall_at_k(rows_by_question, 3),
        "Recall@5": recall_at_k(rows_by_question, 5),
        "Recall@10": recall_at_k(rows_by_question, 10),
        "MRR@10": mrr_at_10(rows_by_question),
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
        for question_id, rows in rows_by_question.items():
            group = question_by_id[question_id][group_field]
            grouped[group][question_id] = rows
        metrics[output_key] = {group: recall_at_k(rows, 5) for group, rows in grouped.items()}
    return metrics
