"""Matched DeepSeek generation from immutable persisted evidence, with resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_deepseek_model_identity import MODELS, OUTPUT_ROOT, health_check, identity_gate
from src.experiments.normalized_rag import (validate_frozen_retrieval, load_frozen_inputs,
    build_frozen_context, QA_PATH, CHUNK_PATH, FINAL_RETRIEVAL_RESULTS, FINAL_RETRIEVAL_CONFIG,
    FINAL_RETRIEVAL_RUN_ID, PROJECT_ROOT)
from src.experiments.manifests import atomic_write_json, atomic_write_jsonl, build_data_manifest, collect_environment, collect_git_state
from src.rag.prompt_builder import build_policy_prompt, PROMPT_VERSION
from src.rag.context_builder import ContextItem
from src.rag.citation import build_citations
from src.rag.answer_quality import evaluate_answer_quality_record, load_answer_quality_config, summarize_quality

LEGACY_ID = "20260910T083640797201Z_rag-answer-quality-llm_real-hybrid-weighted_fixed-length_topk5_4ff30120"
LEGACY_DIR = OUTPUT_ROOT.parents[1] / "runs" / LEGACY_ID


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_rows():
    validate_frozen_retrieval()
    questions, chunks, retrieval = load_frozen_inputs()
    rows = []
    for question in questions:
        context = build_frozen_context(retrieval[question["question_id"]], chunks)
        prompt = build_policy_prompt(question["question"], context)
        rendered = prompt.user_prompt[len("【政策上下文】\n"):].split("\n\n【用户问题】\n", 1)[0]
        rows.append({"question_id": question["question_id"], "question": question["question"],
                     "evaluation_reference": question, "chunks": [asdict(c) for c in context],
                     "context": rendered, "context_sha256": sha(rendered),
                     "system_prompt": prompt.system_prompt, "user_prompt": prompt.user_prompt,
                     "prompt_sha256": sha(prompt.debug_text()), "prompt_version": PROMPT_VERSION})
    return rows


def validate_evidence(rows):
    if len(rows) != 40 or len({r["question_id"] for r in rows}) != 40:
        raise ValueError("evidence must have exactly 40 unique questions")
    for r in rows:
        context = [ContextItem(**c) for c in r["chunks"]]
        prompt = build_policy_prompt(r["question"], context)
        rendered = prompt.user_prompt[len("【政策上下文】\n"):].split("\n\n【用户问题】\n", 1)[0]
        if (len(context) != 5 or r["context"] != rendered or sha(rendered) != r["context_sha256"]
            or r["system_prompt"] != prompt.system_prompt or r["user_prompt"] != prompt.user_prompt
            or sha(prompt.debug_text()) != r["prompt_sha256"] or r["prompt_version"] != PROMPT_VERSION):
            raise ValueError("frozen evidence integrity failure")


def audit_legacy(rows):
    history = read_rows(LEGACY_DIR / "per_question_results.jsonl")
    by_id = {r["question_id"]: r for r in history}
    fields = {k: sum(k in r for r in history) for k in
              ("question_id", "retrieved_chunk_ids", "retrieval_scores", "context", "prompt", "context_sha256")}
    orders_match = len(history) == 40 and all(
        by_id.get(r["question_id"], {}).get("retrieved_chunk_ids") == [c["chunk_id"] for c in r["chunks"]] for r in rows)
    contexts_proven = len(history) == 40 and all(
        by_id.get(r["question_id"], {}).get("context") == r["context"] for r in rows)
    return {"experiment_id": LEGACY_ID, "question_count": len(history), "saved_field_counts": fields,
            "top5_chunk_order_matches": orders_match, "verbatim_context_match_proven": contexts_proven,
            "strict_paired_comparison_included": contexts_proven,
            "note": "IDs/order and excerpts cannot prove full historical request context. Legacy remains unchanged."}


def prepare(output):
    target = output / "frozen_evidence_40.jsonl"
    if target.exists():
        rows = read_rows(target)
        validate_evidence(rows)
        manifest = json.loads((output / "frozen_evidence_manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["frozen_evidence_sha256"]:
            raise ValueError("frozen evidence file hash mismatch")
        return rows
    rows = freeze_rows()
    if rows != freeze_rows():
        raise ValueError("evidence is not deterministic")
    validate_evidence(rows)
    atomic_write_jsonl(target, rows)
    manifest = build_data_manifest({"questions": QA_PATH, "chunks": CHUNK_PATH,
        "retrieval_results": FINAL_RETRIEVAL_RESULTS, "retrieval_config": FINAL_RETRIEVAL_CONFIG})
    manifest.update({"retrieval_source_experiment_id": FINAL_RETRIEVAL_RUN_ID, "question_count": 40,
        "frozen_evidence_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "context_hash_encoding": "SHA-256 of UTF-8 exact rendered policy context, including ordered chunk metadata",
        "parameters": {"retrieval": "real_hybrid_weighted", "chunk": "fixed_length", "top_k": 5,
        "dense_model": "BAAI/bge-small-zh-v1.5", "dense_weight": 0.5, "bm25_weight": 0.5,
        "candidate_pool": 50, "normalization": "per_query_min_max", "seed": 42}, "deterministic": True})
    atomic_write_json(output / "frozen_evidence_manifest.json", manifest)
    atomic_write_json(output / "legacy_context_audit.json", audit_legacy(rows))
    atomic_write_json(output / "legacy_before_manifest.json", build_data_manifest({"legacy": LEGACY_DIR}))
    atomic_write_json(output / "environment.json", collect_environment())
    atomic_write_json(output / "git_state.json", collect_git_state(PROJECT_ROOT))
    atomic_write_json(output / "evaluation_config.json", load_answer_quality_config())
    atomic_write_json(output / "config.json", {"models": MODELS, "provider": "deepseek", "base_url": "https://api.deepseek.com",
        "prompt_version": PROMPT_VERSION, "temperature": 0.0, "max_tokens": 512,
        "thinking_requested": "disabled", "seed": 42, "automatic_retry": 0,
        "generation_retrieval_calls": 0, "smoke_results_reused_in_formal_40": True,
        "max_generation_requests": 80, "recorder": "append-only generation_records.jsonl with per-request flush/fsync"})
    return rows


def record_key(r):
    return (r["experiment_id"], r["question_id"], r["requested_model"])


def run_subset(output, evidence, api_key, client=health_check):
    path = output / "generation_records.jsonl"
    records = read_rows(path)
    protected = {record_key(r) for r in records}
    experiment_id = output.name
    expected_identities = {r["requested_model"]: r["actual_model"] for r in read_rows(output / "health_checks.jsonl")}
    for e in evidence:
        for model in MODELS:
            if (experiment_id, e["question_id"], model) in protected:
                continue
            if len(records) >= 80:
                raise ValueError("80 generation request limit reached")
            if any(r["status"] != "completed" for r in records):
                raise ValueError("failed or interrupted record present; stop without automatic retries")
            pending = {"experiment_id": experiment_id, "question_id": e["question_id"], "provider": "deepseek",
                "requested_model": model, "actual_model": "unknown", "prompt_version": PROMPT_VERSION,
                "temperature": 0.0, "max_tokens": 512, "thinking_requested": "disabled", "thinking_effective": "unknown",
                "context_sha256": e["context_sha256"], "prompt_sha256": e["prompt_sha256"], "answer": "",
                "latency_ms": None, "input_tokens": None, "output_tokens": None, "total_tokens": None,
                "timestamp": None, "status": "started", "error": None}
            # Persist intent before the API call: uncertain interrupted requests are never repeated.
            atomic_write_json(output / "pending_request.json", pending)
            result = client(model, api_key, messages=[{"role": "system", "content": e["system_prompt"]},
                {"role": "user", "content": e["user_prompt"]}], max_tokens=512)
            usage = result["response_metadata"].get("usage", {})
            record = {**pending, **result, "input_tokens": usage.get("prompt_tokens"),
                      "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
                      "status": "completed" if result["response_success"] else "failed"}
            identity_ok = result["actual_model"] == expected_identities[model]
            if not identity_ok:
                record.update(status="failed", error="model identity changed since health check")
            if result.get("finish_reason") != "stop":
                record.update(status="failed", error="completion did not finish normally")
            serialized = json.dumps(record, ensure_ascii=False).replace(api_key, "[REDACTED]")
            with path.open("a", encoding="utf-8") as f:
                f.write(serialized + "\n")
                f.flush()
                os.fsync(f.fileno())
            (output / "pending_request.json").unlink()
            records.append(json.loads(serialized))
            print(json.dumps({"question_id": e["question_id"], "model": model,
                              "status": record["status"], "latency_ms": record["latency_ms"]}), flush=True)
            if record["status"] != "completed":
                raise ValueError("generation failed; stopped")
    return records


def check_pairs(records, evidence, experiment_id):
    expected = {(experiment_id, e["question_id"], m): e for e in evidence for m in MODELS}
    selected = [r for r in records if record_key(r) in expected]
    if len(selected) != len(expected) or len({record_key(r) for r in selected}) != len(expected):
        return False
    return all(r["status"] == "completed" and r["answer"] and r["actual_model"]
               and r["context_sha256"] == expected[record_key(r)]["context_sha256"]
               and r["prompt_sha256"] == expected[record_key(r)]["prompt_sha256"] for r in selected)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def export_results(output, evidence, records):
    refs = {e["question_id"]: e for e in evidence}
    per_question = []
    config = json.loads((output / "evaluation_config.json").read_text(encoding="utf-8"))
    quality_by_model = {m: [] for m in MODELS}
    for r in records:
        e = refs[r["question_id"]]
        context = [ContextItem(**c) for c in e["chunks"]]
        quality = evaluate_answer_quality_record(e["evaluation_reference"], {
            "generated_answer": r["answer"], "strategy": "fixed_length",
            "citations": [c.to_dict() for c in build_citations(context, max_citations=3)],
            "retrieved_chunk_ids": [c.chunk_id for c in context]}, config)
        quality_by_model[r["requested_model"]].append(quality)
        per_question.append({**r, "keyword_overlap_ratio": quality["keyword_overlap_ratio"],
            "evidence_overlap_ratio": quality["evidence_overlap_ratio"],
            "possible_missing_keywords": quality["possible_missing_keywords"],
            "possible_unsupported_terms": quality["possible_unsupported_terms"],
            "auto_quality_label": quality["auto_quality_label"], "answer_correctness": "",
            "evidence_coverage": "", "key_point_coverage": "", "faithfulness": "",
            "completeness": "", "hallucination_indicator": "", "Good/Bad": ""})
    summaries = []
    for model, quality in quality_by_model.items():
        summary = summarize_quality(quality)
        summaries.append({"experiment_id": output.name, "requested_model": model,
            "actual_models": sorted({r["actual_model"] for r in records if r["requested_model"] == model}),
            "successful_questions": sum(r["status"] == "completed" for r in records if r["requested_model"] == model),
            **summary, "answer_correctness": "", "evidence_coverage": "", "key_point_coverage": "",
            "faithfulness": "", "completeness": "", "hallucination_indicator": "", "Good/Bad": ""})
    write_csv(output / "deepseek_cross_model_per_question.csv", per_question)
    write_csv(output / "deepseek_cross_model_summary.csv", summaries)
    atomic_write_json(output / "metrics.json", summaries)
    gate = identity_gate(read_rows(output / "health_checks.jsonl"))
    title = "不同 API 模型配置下的稳健性验证" if gate == "unknown_config_comparison_only" else "DeepSeek matched cross-model experiment"
    identity_note = ("API 无法确认实际底层模型身份；本实验仅为不同 API 模型配置下的稳健性验证，不能称为严格的不同底层模型比较。"
        if gate == "unknown_config_comparison_only" else
        "API model identifiers differ. This identifies API-reported models, not an independent audit of backend weights.")
    report = ["# " + title, "",
        "Experiment ID: " + output.name, "",
        "All models use exactly the same frozen context and policy_grounded_qa_v1 prompt, temperature=0, max_tokens=512.",
        "Thinking was requested disabled; effective mode is unknown unless explicitly confirmed by response metadata.",
        identity_note, "",
        "| Requested model | Successful | Keyword overlap | Evidence keyword overlap | good | needs_review | insufficient |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        d = s["auto_quality_distribution"]
        report.append(f"| {s['requested_model']} | {s['successful_questions']} | {s['average_keyword_overlap_ratio']:.6f} | {s['average_evidence_overlap_ratio']:.6f} | {d.get('good', 0)} | {d.get('needs_review', 0)} | {d.get('insufficient', 0)} |")
    report.extend(["", "Evaluation uses the existing rule-based evaluator without model-specific adjustments.",
        "Keyword and evidence keyword overlap are auxiliary signals, not semantic correctness or evidence coverage.",
        "possible_unsupported_terms is a lexical review signal, not a hallucination rate.",
        "Unimplemented correctness, evidence coverage, key-point coverage, faithfulness, completeness, hallucination indicator and binary Good/Bad metrics are blank.",
        "The existing good/needs_review/insufficient label is retained verbatim; needs_review is not converted to Bad.",
        "Citations are generated by the program from the same frozen top-5 context; they are not model-authored citations.",
        "Historical Legacy Baseline lacks saved full context/prompt/hash and is excluded from strict paired comparisons.",
        "Smoke requests are reused in the 80 records; completed experiment/question/model keys are never called again.",
        "Manual review fields are intentionally blank. Share the blind review CSV without the separate key or unblinded result files.", ""])
    (output / "deepseek_cross_model_summary.md").write_text("\n".join(report), encoding="utf-8")
    write_csv(output / "experiment_registry.csv", [{"experiment_id": output.name,
        "requested_model": s["requested_model"], "status": "completed", "question_count": s["successful_questions"],
        "metrics_path": str(output / "metrics.json"), "metric_selector": s["requested_model"],
        "per_question_results_path": str(output / "generation_records.jsonl"),
        "frozen_evidence_path": str(output / "frozen_evidence_40.jsonl"), "legacy": False} for s in summaries])
    rng = random.Random(42)
    models = list(MODELS)
    rng.shuffle(models)
    mapping = dict(zip(("Model A", "Model B"), models))
    manual = []
    for e in evidence:
        labels = list(mapping)
        rng.shuffle(labels)
        for label in labels:
            r = next(r for r in records if r["question_id"] == e["question_id"] and r["requested_model"] == mapping[label])
            manual.append({"experiment_id": output.name, "question_id": e["question_id"], "anonymous_model": label,
                "question": e["question"], "context": e["context"], "context_sha256": e["context_sha256"],
                "reference_answer": e["evaluation_reference"].get("expected_answer", ""), "answer": r["answer"],
                "correctness": "", "faithfulness": "", "completeness": "", "hallucination": "", "review_note": ""})
    write_csv(output / "deepseek_cross_model_manual_review.csv", manual)
    atomic_write_json(output / "deepseek_cross_model_blind_key.json", {"seed": 42, "mapping": mapping})
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke1", "smoke3", "full"), required=True)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in (".", "..", LEGACY_ID):
        raise ValueError("invalid experiment ID")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY: missing; current process did not inherit the variable")
        return 1
    output = OUTPUT_ROOT / args.experiment_id
    health = read_rows(output / "health_checks.jsonl")
    if identity_gate(health) not in ("distinct_actual_models", "unknown_config_comparison_only"):
        raise ValueError("model identity gate did not pass")
    metadata = json.loads((output / "experiment_metadata.json").read_text(encoding="utf-8"))
    evidence = prepare(output)
    metadata.update(frozen_evidence_established=True, legacy_context_audit=audit_legacy(evidence))
    atomic_write_json(output / "experiment_metadata.json", metadata)
    pending = output / "pending_request.json"
    if pending.exists():
        intent = json.loads(pending.read_text(encoding="utf-8"))
        if record_key(intent) not in {record_key(r) for r in read_rows(output / "generation_records.jsonl")}:
            raise ValueError("interrupted API call has unknown outcome; refusing duplicate request")
        pending.unlink()
    records = read_rows(output / "generation_records.jsonl")
    for r in records:
        if r["status"] != "completed":
            raise ValueError("prior failed request requires investigation")
    if args.stage == "prepare":
        print(json.dumps({"frozen_count": len(evidence), "legacy_audit": metadata["legacy_context_audit"]}))
        return 0
    prerequisite = {"smoke3": "smoke1_pass", "full": "smoke3_pass"}.get(args.stage)
    if prerequisite and not metadata.get(prerequisite):
        raise ValueError("previous smoke gate did not pass")
    if args.stage == "full" and not check_pairs(records, evidence[:3], output.name):
        raise ValueError("persisted smoke records failed integrity check")
    subset = evidence[:1] if args.stage == "smoke1" else evidence[:3] if args.stage == "smoke3" else evidence
    records = run_subset(output, subset, api_key)
    passed = check_pairs(records, subset, output.name)
    metadata.update({args.stage + "_pass": passed, "generation_request_count": len(records),
                     "context_hash_pass": passed, "status": "completed" if args.stage == "full" and passed else args.stage})
    if args.stage == "full" and passed:
        metadata["summary"] = export_results(output, evidence, records)
        metadata["manual_review_rows"] = 80
    before = json.loads((output / "legacy_before_manifest.json").read_text(encoding="utf-8"))
    after = build_data_manifest({"legacy": LEGACY_DIR})
    metadata["legacy_unchanged"] = before == after
    if before != after:
        raise ValueError("legacy files changed")
    atomic_write_json(output / "experiment_metadata.json", metadata)
    print(json.dumps({"stage": args.stage, "pass": passed, "generation_requests_total": len(records),
                      "legacy_unchanged": before == after}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
