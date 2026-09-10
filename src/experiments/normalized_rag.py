"""Normalized RAG experiments over the frozen final retrieval results."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.build_real_embedding_dense_retrieval import load_simple_yaml
from src.experiments.manifests import atomic_write_json, atomic_write_jsonl
from src.experiments.recorder import DEFAULT_EXPERIMENT_ROOT, ExperimentRecorder
from src.rag.answer_generator import generate_answer
from src.rag.answer_generator_v1 import generate_answer_v1
from src.rag.answer_quality import evaluate_answer_quality_record, load_answer_quality_config, summarize_quality
from src.rag.answer_v3_optimization import generate_v3_answer_from_context
from src.rag.citation import build_citations
from src.rag.context_builder import ContextItem
from src.rag.llm_generator import generate_llm_answer
from src.rag.llm_provider import create_llm_provider, load_llm_config
from src.rag.prompt_builder import PROMPT_VERSION, build_policy_prompt
from src.retrieval.dense_index import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
QA_PATH = DATASETS_ROOT / "processed/hdu_ies/student_handbook/qa_testset/hdu_ies_student_handbook_qa_testset_v1.jsonl"
CHUNK_PATH = DATASETS_ROOT / "processed/hdu_ies/student_handbook/chunks/hdu_ies_student_handbook_chunks_fixed_length.jsonl"
CLEANED_PATH = DATASETS_ROOT / "interim/hdu_ies/student_handbook/cleaned/hdu_ies_student_handbook_cleaned.txt"
RAW_PATH = DATASETS_ROOT / "raw/hdu_ies/student_handbook/hdu_ies_student_handbook_source.txt"
FINAL_RETRIEVAL_RUN_ID = "20260910T035306268526Z_retrieval_real-hybrid-weighted_fixed-length_topk5_2b110eac"
FINAL_RETRIEVAL_RUN = DEFAULT_EXPERIMENT_ROOT / "runs" / FINAL_RETRIEVAL_RUN_ID
FINAL_RETRIEVAL_RESULTS = FINAL_RETRIEVAL_RUN / "per_question_results.jsonl"
FINAL_RETRIEVAL_CONFIG = FINAL_RETRIEVAL_RUN / "config.json"
GENERATION_VERSIONS = ("extractive_v1", "extractive_v2", "extractive_v3", "llm_grounded_v1")
REQUIRED_PER_QUESTION_FIELDS = {
    "question_id", "question", "retrieval_method", "retrieved_chunk_ids",
    "retrieved_policy_titles", "generated_answer", "reference_answer",
    "source_policy_hit", "relevant_chunk_hit", "automatic_metrics",
    "citations", "error", "generation_family", "generation_version",
    "provider", "model", "prompt_version",
}


@dataclass(frozen=True)
class FrozenRetrieval:
    method: str = "real_hybrid_weighted"
    strategy: str = "fixed_length"
    top_k: int = 5
    candidate_pool_k: int = 50
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    normalization: str = "per_query_min_max"
    equal_score_value: float = 0.0
    random_seed: int = 42
    dense_model: str = "BAAI/bge-small-zh-v1.5"


FROZEN_RETRIEVAL = FrozenRetrieval()


def generation_family(version: str) -> str:
    if version not in GENERATION_VERSIONS:
        raise ValueError(f"unsupported generation version: {version}")
    return "llm" if version == "llm_grounded_v1" else "extractive"


def validate_frozen_retrieval(run_dir: Path = FINAL_RETRIEVAL_RUN) -> dict[str, Any]:
    config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    params = config.get("parameters", {})
    hybrid = config.get("hybrid_parameters", {})
    expected = FROZEN_RETRIEVAL
    checks = {
        "retrieval_method": (config.get("retrieval_method"), expected.method),
        "chunk_strategy": (config.get("chunk_strategy"), expected.strategy),
        "top_k": (config.get("top_k"), expected.top_k),
        "candidate_pool_k": (params.get("candidate_pool_k"), expected.candidate_pool_k),
        "dense_candidate_pool_k": (params.get("dense_candidate_pool_k"), expected.candidate_pool_k),
        "bm25_candidate_pool_k": (params.get("bm25_candidate_pool_k"), expected.candidate_pool_k),
        "fusion_candidate_pool_k": (params.get("fusion_candidate_pool_k"), expected.candidate_pool_k),
        "dense_weight": (hybrid.get("dense_weight"), expected.dense_weight),
        "bm25_weight": (hybrid.get("bm25_weight"), expected.bm25_weight),
        "normalization": (hybrid.get("normalization"), expected.normalization),
        "equal_score_value": (hybrid.get("equal_score_value"), expected.equal_score_value),
        "random_seed": (config.get("random_seed"), expected.random_seed),
        "model_id": (params.get("model_id"), expected.dense_model),
    }
    mismatches = {key: {"actual": actual, "expected": wanted} for key, (actual, wanted) in checks.items() if actual != wanted}
    if mismatches:
        raise ValueError("frozen retrieval configuration mismatch: " + json.dumps(mismatches, ensure_ascii=False))
    return config


def load_frozen_inputs(
    *, qa_path: Path = QA_PATH, chunk_path: Path = CHUNK_PATH,
    retrieval_results_path: Path = FINAL_RETRIEVAL_RESULTS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = load_jsonl(Path(qa_path))
    if len(questions) != 40 or len({row.get("question_id") for row in questions}) != 40:
        raise ValueError("frozen QA dataset must contain 40 unique questions")
    chunks = {str(row["chunk_id"]): row for row in load_jsonl(Path(chunk_path))}
    retrieval_rows = load_jsonl(Path(retrieval_results_path))
    retrieval_by_id = {str(row["question_id"]): row for row in retrieval_rows}
    if [row.get("question_id") for row in questions] != [row.get("question_id") for row in retrieval_rows]:
        raise ValueError("frozen retrieval results do not preserve the 40-question QA order")
    return questions, chunks, retrieval_by_id


def build_frozen_context(retrieval_row: Mapping[str, Any], chunks: Mapping[str, dict[str, Any]]) -> list[ContextItem]:
    context: list[ContextItem] = []
    for result in list(retrieval_row.get("retrieved_results", []))[: FROZEN_RETRIEVAL.top_k]:
        chunk_id = str(result.get("retrieved_chunk_id", ""))
        chunk = chunks.get(chunk_id)
        if not chunk:
            raise ValueError(f"retrieved chunk is missing from frozen chunk file: {chunk_id}")
        text = str(chunk.get("chunk_text", ""))
        title = str(chunk.get("policy_title") or chunk.get("metadata", {}).get("policy_title") or result.get("retrieved_policy_title", ""))
        score = float(result.get("hybrid_score", 0.0) or 0.0)
        context.append(ContextItem(int(result.get("rank", len(context) + 1)), chunk_id, title, score, text, text[:220], FROZEN_RETRIEVAL.strategy, FROZEN_RETRIEVAL.method))
    if len(context) != FROZEN_RETRIEVAL.top_k:
        raise ValueError("every question must have exactly five frozen retrieval results")
    return context


def _extractive_answer(version: str, question: dict[str, Any], context: list[ContextItem], citation_objects: list[Any]) -> tuple[str, dict[str, Any]]:
    if version == "extractive_v1":
        generated = generate_answer_v1(question["question"], context, citation_objects)
        return generated.answer, {"legacy_version": "v1"}
    if version == "extractive_v2":
        generated = generate_answer(question["question"], context, citation_objects)
        return generated.answer, {"legacy_version": "v2"}
    if version == "extractive_v3":
        details = generate_v3_answer_from_context(question, context, [item.to_dict() for item in citation_objects])
        return str(details["generated_answer"]), details
    raise ValueError(f"not an extractive version: {version}")


def _unified_record(
    version: str, question: dict[str, Any], context: list[ContextItem], answer: str,
    details: Mapping[str, Any] | None = None, *, provider: str | None = None,
    model: str | None = None, prompt_version: str | None = None, error: dict[str, str] | None = None,
) -> dict[str, Any]:
    citations = [item.to_dict() for item in build_citations(context, max_citations=3)]
    base = {
        "question_id": question["question_id"], "question": question["question"],
        "retrieval_method": FROZEN_RETRIEVAL.method,
        "retrieved_chunk_ids": [item.chunk_id for item in context],
        "retrieved_policy_titles": list(dict.fromkeys(item.document_title for item in context if item.document_title)),
        "generated_answer": answer, "reference_answer": question.get("expected_answer", ""),
        "citations": citations, "error": error,
        "generation_family": generation_family(version), "generation_version": version,
        "provider": provider, "model": model, "prompt_version": prompt_version,
        "question_type": question.get("question_type", ""), "difficulty": question.get("difficulty", ""),
        "strategy": FROZEN_RETRIEVAL.strategy, "top_k": FROZEN_RETRIEVAL.top_k,
    }
    quality_input = {**base, "citations": citations}
    quality = evaluate_answer_quality_record(question, quality_input, load_answer_quality_config())
    automatic = {key: quality[key] for key in (
        "keyword_overlap_ratio", "evidence_overlap_ratio", "answer_non_empty",
        "has_citation", "answer_char_count", "citation_count", "auto_quality_label",
        "possible_missing_keywords", "possible_unsupported_terms",
    )}
    base.update({"source_policy_hit": quality["source_policy_hit"], "relevant_chunk_hit": quality["related_chunk_hit"], "automatic_metrics": automatic})
    base.update({key: quality[key] for key in ("keyword_overlap_ratio", "evidence_overlap_ratio", "answer_non_empty", "has_citation", "auto_quality_label")})
    if details:
        base["generation_details"] = dict(details)
    return base


def build_records(version: str, *, provider: Any | None = None) -> list[dict[str, Any]]:
    validate_frozen_retrieval()
    questions, chunks, retrieval = load_frozen_inputs()
    records: list[dict[str, Any]] = []
    for question in questions:
        context = build_frozen_context(retrieval[question["question_id"]], chunks)
        citation_objects = build_citations(context, max_citations=3)
        if generation_family(version) == "extractive":
            answer, details = _extractive_answer(version, question, context, citation_objects)
            record = _unified_record(version, question, context, answer, details)
        else:
            generated = generate_llm_answer(question["question"], context, provider)
            record = _unified_record(version, question, context, generated.answer, provider=generated.provider, model=generated.model, prompt_version=generated.prompt_version)
        missing = REQUIRED_PER_QUESTION_FIELDS.difference(record)
        if missing:
            raise ValueError(f"incomplete per-question schema: {sorted(missing)}")
        records.append(record)
    return records


def dry_run_llm() -> dict[str, Any]:
    validate_frozen_retrieval()
    questions, chunks, retrieval = load_frozen_inputs()
    prompt_count = 0
    context_count = 0
    for question in questions:
        context = build_frozen_context(retrieval[question["question_id"]], chunks)
        prompt = build_policy_prompt(question["question"], context)
        if not prompt.system_prompt or not prompt.user_prompt:
            raise ValueError(f"empty prompt for {question['question_id']}")
        prompt_count += 1
        context_count += len(context)
    return {
        "generation_version": "llm_grounded_v1", "qa_total": len(questions),
        "prompt_build_count": prompt_count, "retrieved_context_count": context_count,
        "requests_to_send": len(questions), "requests_sent": 0,
        "actual_api_requests": 0, "external_api_called": False, "status": "passed",
    }


def _recorder(version: str, *, experiment_id: str | None = None, experiment_root: Path = DEFAULT_EXPERIMENT_ROOT) -> ExperimentRecorder:
    family = generation_family(version)
    llm_config = load_llm_config() if family == "llm" else {}
    params = {
        "experiment_type": "rag_answer_quality", "generation_family": family,
        "generation_version": version, "legacy_version": version.removeprefix("extractive_") if family == "extractive" else None,
        "provider": "deepseek_openai_compatible" if family == "llm" else None,
        "model": llm_config.get("model") if family == "llm" else None,
        "prompt_version": PROMPT_VERSION if family == "llm" else None,
        "thinking": "disabled" if family == "llm" else None,
        "temperature": float(llm_config.get("temperature", 0.0)) if family == "llm" else None,
        "max_tokens": int(llm_config.get("max_tokens", 512)) if family == "llm" else None,
        "external_api": family == "llm", "candidate_pool_k": 50,
        "dense_candidate_pool_k": 50, "bm25_candidate_pool_k": 50, "fusion_candidate_pool_k": 50,
        "retrieval_source_experiment_id": FINAL_RETRIEVAL_RUN_ID, "retry": 0,
    }
    return ExperimentRecorder(
        task_type="rag_answer_quality", retrieval_method=FROZEN_RETRIEVAL.method,
        chunk_strategy=FROZEN_RETRIEVAL.strategy, top_k=FROZEN_RETRIEVAL.top_k,
        embedding_model=FROZEN_RETRIEVAL.dense_model + "+bm25", qa_version="v1",
        hybrid_parameters={"dense_weight": 0.5, "bm25_weight": 0.5, "normalization": "per_query_min_max", "equal_score_value": 0.0},
        random_seed=42, inputs={"data_source": CLEANED_PATH, "raw_source": RAW_PATH, "chunk_file": CHUNK_PATH, "qa_file": QA_PATH, "retrieval_results": FINAL_RETRIEVAL_RESULTS},
        parameters=params, experiment_root=Path(experiment_root), repo_root=PROJECT_ROOT,
        experiment_id=experiment_id, generation_script="scripts/run_normalized_rag.py",
        manifest_metadata={"retrieval": {"source_experiment_id": FINAL_RETRIEVAL_RUN_ID, "top_k": 5, "candidate_pool_k": 50}},
    )


def run_extractive(version: str, *, experiment_root: Path = DEFAULT_EXPERIMENT_ROOT, experiment_id: str | None = None) -> ExperimentRecorder:
    if generation_family(version) != "extractive":
        raise ValueError("run_extractive accepts extractive versions only")
    records = build_records(version)
    quality_rows = []
    for row in records:
        quality_rows.append({**row, **row["automatic_metrics"], "related_chunk_hit": row["relevant_chunk_hit"]})
    metrics = summarize_quality(quality_rows)
    metrics.update({"generation_family": "extractive", "generation_version": version, "external_api": False, "retrieval_source_experiment_id": FINAL_RETRIEVAL_RUN_ID})
    recorder = _recorder(version, experiment_id=experiment_id, experiment_root=experiment_root)
    with recorder:
        recorder.record_results(metrics, records)
    return recorder


def _sanitize_error(exc: BaseException) -> dict[str, str]:
    message = str(exc)
    for name in ("DEEPSEEK_API_KEY", "LLM_API_KEY"):
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message, flags=re.I)
    return {"type": type(exc).__name__, "message": message[:1000]}


def llm_batch_plan(experiment_id: str, *, staging_root: Path | None = None, retry_failed: bool = False) -> dict[str, Any]:
    questions, _chunks, _retrieval = load_frozen_inputs()
    root = Path(staging_root or (DEFAULT_EXPERIMENT_ROOT / "llm_batch_staging")) / experiment_id
    completed_path = root / "completed.jsonl"
    failures_path = root / "failures.jsonl"
    completed = load_jsonl(completed_path) if completed_path.exists() else []
    failures = load_jsonl(failures_path) if failures_path.exists() else []
    completed_ids = {str(row.get("question_id")) for row in completed if not row.get("error") and row.get("generated_answer")}
    failed_ids = {str(row.get("question_id")) for row in failures}
    protected_ids = completed_ids | (set() if retry_failed else failed_ids)
    return {"total_questions": len(questions), "already_completed": len(completed_ids), "already_failed": len(failed_ids), "requests_to_send": len(questions) - len(protected_ids), "completed_ids": completed_ids, "failed_ids": failed_ids, "protected_ids": protected_ids, "retry_failed": retry_failed, "staging_dir": root}


def run_llm_batch(experiment_id: str, *, execute_external_api: bool = False, staging_root: Path | None = None, retry_failed: bool = False, provider_factory: Callable[..., Any] = create_llm_provider) -> dict[str, Any]:
    if not execute_external_api:
        return dry_run_llm()
    plan = llm_batch_plan(experiment_id, staging_root=staging_root, retry_failed=retry_failed)
    root = Path(plan["staging_dir"])
    root.mkdir(parents=True, exist_ok=True)
    questions, chunks, retrieval = load_frozen_inputs()
    completed_path, failures_path = root / "completed.jsonl", root / "failures.jsonl"
    completed = load_jsonl(completed_path) if completed_path.exists() else []
    failures = load_jsonl(failures_path) if failures_path.exists() else []
    completed_ids = set(plan["completed_ids"])
    protected_ids = set(plan["protected_ids"])
    provider = provider_factory(load_llm_config())
    requests_sent = 0
    for question in questions:
        if question["question_id"] in protected_ids:
            continue
        context = build_frozen_context(retrieval[question["question_id"]], chunks)
        requests_sent += 1
        try:
            generated = generate_llm_answer(question["question"], context, provider)
            completed.append(_unified_record("llm_grounded_v1", question, context, generated.answer, provider=generated.provider, model=generated.model, prompt_version=generated.prompt_version))
            completed_ids.add(question["question_id"])
            protected_ids.add(question["question_id"])
            atomic_write_jsonl(completed_path, completed)
        except BaseException as exc:
            failures.append({"question_id": question["question_id"], "error": _sanitize_error(exc)})
            atomic_write_jsonl(failures_path, failures)
    state = {"experiment_id": experiment_id, "total_questions": 40, "already_completed": len(completed_ids), "requests_sent_this_run": requests_sent, "retry_failed": retry_failed, "automatic_retry": 0, "failures": len(failures)}
    atomic_write_json(root / "state.json", state)
    return state
