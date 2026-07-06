"""Batch evaluation for extractive RAG QA outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.build_real_embedding_dense_retrieval import CONFIG_PATH, configure_model_cache_environment, load_simple_yaml
from scripts.evaluate_real_embedding_dense_retrieval import search_with_provider
from src.rag.answer_generator import INSUFFICIENT_MESSAGE
from src.rag.pipeline import RAGResponse, answer_query
from src.retrieval import demo as retrieval_demo
from src.retrieval.bm25 import build_index, search as bm25_search
from src.retrieval.demo import DemoResult
from src.retrieval.dense_index import load_jsonl, write_jsonl
from src.retrieval.embedding_providers import create_provider
from src.retrieval.hybrid import normalize_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_evaluation.yaml"
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
DEFAULT_QA_JSONL = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "qa_testset" / "hdu_ies_student_handbook_qa_testset_v1.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook" / "rag_evaluation"


AnswerFunction = Callable[..., RAGResponse]


@dataclass(frozen=True)
class EvaluationCombination:
    method: str
    strategy: str
    top_k: int

    @property
    def slug(self) -> str:
        return f"{self.method}_{self.strategy}"


class CachedRAGRetriever:
    """Reusable retriever for batch evaluation so the real embedding model loads once."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[dict[str, Any]]] = {}
        self._bm25_indexes: dict[str, dict[str, Any]] = {}
        self._embedding_rows: dict[str, list[dict[str, Any]]] = {}
        self._provider: Any | None = None

    def __call__(self, query: str, method: str, strategy: str, top_k: int) -> list[DemoResult]:
        if method == "bm25":
            return self.retrieve_bm25(query, strategy, top_k)
        if method == "real_dense":
            return self.retrieve_real_dense(query, strategy, top_k)
        if method == "real_hybrid_weighted":
            return self.retrieve_real_hybrid_weighted(query, strategy, top_k)
        raise ValueError(f"unsupported method: {method}")

    def chunks(self, strategy: str) -> list[dict[str, Any]]:
        if strategy not in self._chunks:
            self._chunks[strategy] = retrieval_demo.load_chunks(strategy)
        return self._chunks[strategy]

    def bm25_index(self, strategy: str) -> dict[str, Any]:
        if strategy not in self._bm25_indexes:
            self._bm25_indexes[strategy] = build_index(self.chunks(strategy))
        return self._bm25_indexes[strategy]

    def embedding_rows(self, strategy: str) -> list[dict[str, Any]]:
        if strategy not in self._embedding_rows:
            self._embedding_rows[strategy] = retrieval_demo.load_real_embedding_rows(strategy)
        return self._embedding_rows[strategy]

    def provider(self) -> Any:
        if self._provider is None:
            config = load_simple_yaml(CONFIG_PATH)
            configure_model_cache_environment(config)
            self._provider = create_provider(config)
        return self._provider

    def retrieve_bm25(self, query: str, strategy: str, top_k: int) -> list[DemoResult]:
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks(strategy)}
        rows = bm25_search(query, self.bm25_index(strategy), top_k=top_k)
        return _format_results(rows, chunks_by_id, strategy, "bm25")

    def retrieve_real_dense(self, query: str, strategy: str, top_k: int) -> list[DemoResult]:
        rows = self.embedding_rows(strategy)
        chunks_by_id = {row["chunk_id"]: row for row in rows}
        search_rows = search_with_provider(query, rows, self.provider(), top_k=top_k)
        return _format_results(search_rows, chunks_by_id, strategy, "real_dense")

    def retrieve_real_hybrid_weighted(self, query: str, strategy: str, top_k: int) -> list[DemoResult]:
        pool_size = max(top_k, 20)
        dense_rows = self.retrieve_real_dense(query, strategy, pool_size)
        bm25_rows = self.retrieve_bm25(query, strategy, pool_size)
        dense_scores = normalize_scores([row.score for row in dense_rows])
        bm25_scores = normalize_scores([row.score for row in bm25_rows])
        candidates: dict[str, dict[str, Any]] = {}
        for row, score in zip(dense_rows, dense_scores):
            candidates.setdefault(row.chunk_id, _candidate_from_result(row))["real_dense_score"] = score
        for row, score in zip(bm25_rows, bm25_scores):
            candidates.setdefault(row.chunk_id, _candidate_from_result(row))["bm25_score"] = score
        fused = []
        for item in candidates.values():
            item["score"] = 0.5 * item.get("real_dense_score", 0.0) + 0.5 * item.get("bm25_score", 0.0)
            fused.append(item)
        fused.sort(key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
        return _format_results(fused[:top_k], {item["chunk_id"]: item for item in fused}, strategy, "real_hybrid_weighted")


def _candidate_from_result(row: DemoResult) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
        "document_title": row.document_title,
        "chunk_text": row.chunk_text,
        "real_dense_score": 0.0,
        "bm25_score": 0.0,
    }


def _format_results(rows: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, Any]], strategy: str, method: str) -> list[DemoResult]:
    results: list[DemoResult] = []
    for rank, row in enumerate(rows, start=1):
        chunk_id = row.get("retrieved_chunk_id") or row.get("chunk_id")
        chunk = chunks_by_id.get(chunk_id, row)
        chunk_text = chunk.get("chunk_text", row.get("chunk_text", ""))
        title = (
            chunk.get("policy_title")
            or chunk.get("metadata", {}).get("policy_title")
            or chunk.get("retrieved_document_title")
            or chunk.get("document_title")
            or ""
        )
        results.append(
            DemoResult(
                rank=rank,
                chunk_id=str(chunk_id),
                document_title=str(title),
                score=float(row.get("score", 0.0) or 0.0),
                chunk_text=str(chunk_text),
                chunk_text_summary=retrieval_demo.summarize_text(str(chunk_text)),
                strategy=strategy,
                retrieval_method=method,
            )
        )
    return results


def load_rag_evaluation_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(path)
    for key in ("default_top_k",):
        if key in config:
            config[key] = int(config[key])
    return config


def parse_combinations(config: dict[str, Any], include_optional: bool = False) -> list[EvaluationCombination]:
    raw_values = list(config.get("combinations", []))
    if include_optional:
        raw_values.extend(config.get("optional_combinations", []))
    if not raw_values:
        raw_values = [
            f"{config.get('default_method', 'real_hybrid_weighted')}|"
            f"{config.get('default_strategy', 'fixed_length')}|"
            f"{config.get('default_top_k', 5)}"
        ]
    combinations: list[EvaluationCombination] = []
    for value in raw_values:
        method, strategy, top_k = str(value).split("|")
        combinations.append(EvaluationCombination(method=method, strategy=strategy, top_k=int(top_k)))
    return combinations


def load_qa_questions(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def related_ids_for_strategy(question: dict[str, Any], strategy: str) -> set[str]:
    return set(question.get(f"related_chunk_ids_{strategy}", []) or [])


def source_policy_hit(question: dict[str, Any], citations: list[dict[str, Any]]) -> bool:
    source_title = str(question.get("source_policy_title", "") or "")
    if not source_title:
        return False
    return any(source_title in str(citation.get("document_title", "")) for citation in citations)


def related_chunk_hit(question: dict[str, Any], strategy: str, retrieved_chunk_ids: list[str]) -> bool:
    related = related_ids_for_strategy(question, strategy)
    return bool(related and related.intersection(retrieved_chunk_ids))


def build_answer_record(question: dict[str, Any], response: RAGResponse, combination: EvaluationCombination) -> dict[str, Any]:
    retrieval_info = response.retrieval_info
    retrieved_chunk_ids = [str(item) for item in retrieval_info.get("retrieved_chunk_ids", [])]
    citations = response.citations
    warning = ""
    if response.answer == INSUFFICIENT_MESSAGE:
        warning = "insufficient_retrieved_context"
    elif not citations:
        warning = "missing_citation"
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "generated_answer": response.answer,
        "method": retrieval_info.get("actual_method", combination.method),
        "requested_method": combination.method,
        "strategy": combination.strategy,
        "top_k": combination.top_k,
        "citations": citations,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "source_policy_titles": list(retrieval_info.get("source_policy_titles", response.source_documents)),
        "answer_mode": response.answer_mode,
        "has_answer": bool(str(response.answer).strip()),
        "has_citation": bool(citations),
        "used_fallback": bool(retrieval_info.get("fallback_used", False)),
        "warning": warning,
        "question_type": question.get("question_type", ""),
        "difficulty": question.get("difficulty", ""),
        "source_policy_hit": source_policy_hit(question, citations),
        "related_chunk_hit": related_chunk_hit(question, combination.strategy, retrieved_chunk_ids),
    }


def evaluate_questions(
    questions: list[dict[str, Any]],
    combination: EvaluationCombination,
    *,
    answer_func: AnswerFunction = answer_query,
    retriever: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question in questions:
        response = answer_func(
            question["question"],
            method=combination.method,
            strategy=combination.strategy,
            top_k=combination.top_k,
            answer_mode="extractive",
            retriever=retriever,
        )
        records.append(build_answer_record(question, response, combination))
    return records, summarize_records(records, combination)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_records(records: list[dict[str, Any]], combination: EvaluationCombination) -> dict[str, Any]:
    question_count = len(records)
    answer_count = sum(1 for row in records if row["has_answer"])
    citation_count = sum(1 for row in records if row["has_citation"])
    fallback_count = sum(1 for row in records if row["used_fallback"])
    unsupported_count = sum(1 for row in records if row["warning"])
    metrics: dict[str, Any] = {
        "method": combination.method,
        "strategy": combination.strategy,
        "top_k": combination.top_k,
        "question_count": question_count,
        "answer_non_empty_rate": _rate(answer_count, question_count),
        "citation_coverage_rate": _rate(citation_count, question_count),
        "average_citation_count": (
            sum(len(row["citations"]) for row in records) / question_count if question_count else 0.0
        ),
        "average_answer_chars": (
            sum(len(str(row["generated_answer"])) for row in records) / question_count if question_count else 0.0
        ),
        "no_answer_count": question_count - answer_count,
        "no_citation_count": question_count - citation_count,
        "source_policy_hit_rate": _rate(sum(1 for row in records if row["source_policy_hit"]), question_count),
        "related_chunk_hit_rate": _rate(sum(1 for row in records if row["related_chunk_hit"]), question_count),
        "unsupported_answer_warning_count": unsupported_count,
        "fallback_count": fallback_count,
        "question_type": grouped_metrics(records, "question_type"),
        "difficulty": grouped_metrics(records, "difficulty"),
        "notes": "source_policy_hit_rate and related_chunk_hit_rate are approximate automatic indicators, not proof of full answer correctness.",
    }
    return metrics


def grouped_metrics(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(field, "") or "unknown")].append(row)
    return {
        group: {
            "question_count": len(values),
            "answer_non_empty_rate": _rate(sum(1 for row in values if row["has_answer"]), len(values)),
            "citation_coverage_rate": _rate(sum(1 for row in values if row["has_citation"]), len(values)),
            "related_chunk_hit_rate": _rate(sum(1 for row in values if row["related_chunk_hit"]), len(values)),
        }
        for group, values in sorted(grouped.items())
    }


def write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "strategy",
        "top_k",
        "question_count",
        "answer_non_empty_rate",
        "citation_coverage_rate",
        "average_citation_count",
        "average_answer_chars",
        "source_policy_hit_rate",
        "related_chunk_hit_rate",
        "no_answer_count",
        "no_citation_count",
        "unsupported_answer_warning_count",
        "fallback_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for report in reports:
            writer.writerow({field: report.get(field, "") for field in fields})


def run_batch_evaluation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    include_optional: bool = False,
    output_root: Path | None = None,
    answer_func: AnswerFunction = answer_query,
    retriever: Any | None = None,
) -> dict[str, Any]:
    config = load_rag_evaluation_config(config_path)
    qa_path = Path(str(config.get("qa_jsonl_path", DEFAULT_QA_JSONL)))
    output_root = output_root or Path(str(config.get("output_root", DEFAULT_OUTPUT_ROOT)))
    questions = load_qa_questions(qa_path)
    combinations = parse_combinations(config, include_optional=include_optional)
    retriever = retriever if retriever is not None else CachedRAGRetriever()

    reports: list[dict[str, Any]] = []
    answer_paths: dict[str, str] = {}
    for combination in combinations:
        records, report = evaluate_questions(questions, combination, answer_func=answer_func, retriever=retriever)
        answer_path = output_root / f"rag_answers_{combination.slug}.jsonl"
        write_jsonl(answer_path, records)
        reports.append(report)
        answer_paths[combination.slug] = str(answer_path)

    overall = {
        "stage": "rag_batch_evaluation",
        "qa_jsonl_path": str(qa_path),
        "output_root": str(output_root),
        "answer_paths": answer_paths,
        "evaluated_combinations": [report["method"] + "+" + report["strategy"] for report in reports],
        "default_report": reports[0] if reports else {},
        "reports": reports,
        "modified_inputs": {"raw": False, "cleaned": False, "chunks": False, "qa": False},
        "overwrote_historical_retrieval_outputs": False,
        "used_llm_api": False,
        "used_reranker": False,
        "used_web_frontend": False,
        "notes": "QA labels are used only for automatic scoring. RAG generation receives question text only.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "rag_evaluation_report.json"
    report_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(output_root / "rag_evaluation_summary.csv", reports)
    return overall
