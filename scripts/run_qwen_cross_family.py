"""Staged, resumable Qwen generation from byte-identical locked DeepSeek evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import check_tokenpony_model_identity as gateway
from scripts.check_tokenpony_reasoning import normal_answer, thinking_off
from scripts.run_deepseek_cross_model import read_rows, validate_evidence, write_csv
from src.experiments.manifests import atomic_write_json, atomic_write_jsonl, collect_environment, collect_git_state
from src.rag.answer_quality import evaluate_answer_quality_record, summarize_quality
from src.rag.context_builder import ContextItem
from src.rag.citation import build_citations

OUTPUT_ROOT = gateway.ROOT / "qwen"
SOURCE = gateway.SOURCE
MODELS = ("deepseek-flash", "deepseek-v4-pro", gateway.MODEL)
DISPLAY_NAMES = dict(zip(MODELS, ("DeepSeek-Flash", "DeepSeek-V4-Pro", "Qwen3.8-Max")))


def output_directory(experiment_id):
    if not re.fullmatch(r"[0-9]{8}T[0-9]+Z_qwen38max_tokenpony_cross_family", experiment_id):
        raise ValueError("invalid formal Qwen experiment ID")
    path = (OUTPUT_ROOT / experiment_id).resolve()
    if path.parent != OUTPUT_ROOT.resolve():
        raise ValueError("Qwen output escapes independent root")
    return path


def protected_snapshot():
    paths = {"deepseek": SOURCE}
    for eid in ("20261006T064519132741Z_qwen_cross_model", "20261006T065148355806Z_qwen_tokenpony_cross_family"):
        paths[eid] = OUTPUT_ROOT / eid
    return {name: gateway.snapshot(path) for name, path in paths.items()}


def load_evidence(source=SOURCE):
    evidence_path = source / "frozen_evidence_40.jsonl"
    manifest_path = source / "frozen_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if digest != manifest["frozen_evidence_sha256"]:
        raise ValueError("Frozen Evidence SHA-256 mismatch")
    rows = read_rows(evidence_path)
    validate_evidence(rows)  # Prompt reconstruction only; never retrieval/freeze_rows.
    references = read_rows(source / "generation_records.jsonl")
    lookup = {(r["question_id"], r["requested_model"]): r for r in references}
    if len(references) != 80 or len(lookup) != 80:
        raise ValueError("DeepSeek reference must contain 80 unique completed records")
    for e in rows:
        for model in MODELS[:2]:
            r = lookup.get((e["question_id"], model), {})
            if (r.get("status") != "completed" or r.get("context_sha256") != e["context_sha256"]
                or r.get("prompt_sha256") != e["prompt_sha256"]):
                raise ValueError("Context/Prompt hash differs from DeepSeek")
    return rows, manifest, references


def selection(diagnostics):
    # Evaluate saved responses, without sending another identity/health request.
    result = json.loads((diagnostics / "compatibility_result.json").read_text(encoding="utf-8"))
    if not result["protected_unchanged"]:
        raise ValueError("diagnostics preservation did not pass")
    for row in result["tests"]:
        if row.get("http_status") == 200 and row.get("actual_model") != gateway.MODEL:
            raise ValueError("diagnostics identity mismatch")
        if row.get("test") in ("A", "B") and thinking_off(row):
            return {"request_options": row["request_options"], "thinking_requested": False,
                "thinking_effective": False, "thinking_control_method":
                "reasoning_effort_none" if row["test"] == "A" else "enable_thinking_false",
                "temperature": 0.0, "max_tokens": 512, "prompt_version": "policy_grounded_qa_v1"}
    chosen = result.get("selected_config")
    if chosen and chosen["thinking_control_method"] == "disable_unavailable":
        return chosen
    raise ValueError("thinking compatibility gate did not pass")


def prepare(output, diagnostics, key_env):
    evidence, manifest, references = load_evidence()
    selected = selection(diagnostics)
    config = {"provider": gateway.PROVIDER, "base_url": gateway.BASE_URL,
        "base_url_type": gateway.BASE_URL_TYPE, "requested_model": gateway.MODEL,
        "api_key_env": key_env, "base_url_env": "TOKENPONY_BASE_URL",
        "seed": 42, "timeout_seconds": 60, "automatic_retry": 0,
        "generation_retrieval_calls": 0, "source_experiment": gateway.SOURCE_ID,
        "diagnostics_dir": str(diagnostics), "source_manifest_sha256":
        hashlib.sha256((SOURCE / "frozen_evidence_manifest.json").read_bytes()).hexdigest(),
        "frozen_evidence_sha256": manifest["frozen_evidence_sha256"], **selected}
    if output.exists():
        existing = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("resume configuration differs; no request allowed")
        before = json.loads((output / "protected_before_sha256.json").read_text(encoding="utf-8"))
        if before != protected_snapshot():
            raise ValueError("locked reference assets changed since prepare")
        for name in ("frozen_evidence_40.jsonl", "frozen_evidence_manifest.json", "evaluation_config.json"):
            if (output / name).read_bytes() != (SOURCE / name).read_bytes():
                raise ValueError("copied frozen/evaluation input changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / "protected_before_sha256.json", protected_snapshot())
        atomic_write_json(output / "config.json", config)
        for name in ("frozen_evidence_40.jsonl", "frozen_evidence_manifest.json", "evaluation_config.json"):
            with (output / name).open("xb") as handle:
                handle.write((SOURCE / name).read_bytes())
        atomic_write_json(output / "environment.json", collect_environment())
        atomic_write_json(output / "git_state.json", collect_git_state(Path(__file__).resolve().parents[1]))
        atomic_write_json(output / "experiment_metadata.json", {"experiment_id": output.name,
            "status": "prepared", "model_identity_pass": True, "thinking_config_determined": True,
            "frozen_evidence_hash_pass": True, "context_hash_pass": True, "prompt_hash_pass": True,
            "question_count": 40, "generation_request_count": 0, "smoke_results_reused": True})
    return evidence, references, config


TERMINAL_STATUSES = {"completed", "truncated", "api_error", "generation_error"}


def load_generation_records(output):
    """Apply hash-bound status provenance without rewriting historical raw lines."""
    path = output / "generation_records.jsonl"
    raw_lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    records = [json.loads(line) for line in raw_lines if line.strip()]
    override_path = output / "generation_status_overrides.json"
    if not override_path.exists():
        return records
    overrides = json.loads(override_path.read_text(encoding="utf-8"))["overrides"]
    applied = set()
    for change in overrides:
        key = (change["experiment_id"], change["question_id"], change["requested_model"])
        matches = [(i, line) for i, line in enumerate(raw_lines) if line.strip()
                   and tuple(json.loads(line)[k] for k in ("experiment_id", "question_id", "requested_model")) == key]
        if len(matches) != 1 or key in applied:
            raise ValueError("status override has duplicate/missing raw attempt")
        _, line = matches[0]
        row = next(r for r in records if (r["experiment_id"], r["question_id"], r["requested_model"]) == key)
        if (hashlib.sha256(line).hexdigest() != change["source_record_sha256"]
            or row["status"] != change["recorded_status"] or change["recorded_status"] != "failed"
            or change["effective_status"] != "truncated" or row.get("finish_reason") != "length"
            or row.get("http_status") != 200 or row.get("actual_model") != gateway.MODEL):
            raise ValueError("status override raw record integrity failure")
        row["recorded_status"] = row["status"]
        row["status"] = "truncated"
        applied.add(key)
    return records


def reliability(records):
    attempted = len(records)
    counts = {s + "_count": sum(r["status"] == s for r in records) for s in TERMINAL_STATUSES}
    return {"attempted_count": attempted, **counts,
            "completion_rate": counts["completed_count"] / attempted if attempted else 0.0,
            "truncation_rate": counts["truncated_count"] / attempted if attempted else 0.0}


def classify_attempt(row, config):
    if row.get("provider") != gateway.PROVIDER or row.get("requested_model") != gateway.MODEL:
        return "integrity_error"
    if row.get("http_status") in (401, 403):
        return "authentication_error"
    if row.get("http_status") == 200 and row.get("actual_model") != gateway.MODEL:
        return "identity_error"
    if row.get("http_status") != 200:
        return "api_error"
    if config["thinking_effective"] is False and row.get("thinking_effective") is not False:
        return "configuration_error"
    if row.get("finish_reason") == "length":
        return "truncated"
    return "completed" if normal_answer(row) else "generation_error"


def validate_records(records, evidence, experiment_id, config, *, complete=False):
    refs = {e["question_id"]: e for e in evidence}
    keys = set()
    for row in records:
        key = (row["experiment_id"], row["question_id"], row["requested_model"])
        e = refs.get(row["question_id"])
        if (key in keys or key[0] != experiment_id or key[2] != gateway.MODEL or e is None
            or row.get("status") not in TERMINAL_STATUSES
            or classify_attempt(row, config) != row["status"]
            or row.get("provider") != gateway.PROVIDER or row.get("base_url_type") != gateway.BASE_URL_TYPE
            or row.get("context_sha256") != e["context_sha256"] or row.get("prompt_sha256") != e["prompt_sha256"]
            or any(row.get(k) != config[k] for k in ("temperature", "max_tokens", "prompt_version", "thinking_control_method"))
            or row.get("thinking_requested") is not False):
            raise ValueError("persisted record failed status/identity/hash/config integrity gate")
        keys.add(key)
    if complete and len(keys) != 40:
        raise ValueError("formal run must contain exactly 40 unique attempted questions")


def run_subset(output, subset, evidence, config, api_key, client=gateway.completion):
    path = output / "generation_records.jsonl"
    records = load_generation_records(output)
    validate_records(records, evidence, output.name, config)
    attempted = {r["question_id"] for r in records}
    pending_path = output / "pending_request.json"
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if pending["question_id"] not in attempted:
            raise ValueError("interrupted request has unknown outcome; refusing duplicate charge")
        pending_path.unlink()
    for e in subset:
        if e["question_id"] in attempted:
            continue
        pending = {"experiment_id": output.name, "question_id": e["question_id"],
            "provider": gateway.PROVIDER, "base_url_type": gateway.BASE_URL_TYPE,
            "requested_model": gateway.MODEL, "prompt_version": config["prompt_version"],
            "temperature": config["temperature"], "max_tokens": config["max_tokens"],
            "thinking_requested": False, "thinking_control_method": config["thinking_control_method"],
            "context_sha256": e["context_sha256"], "prompt_sha256": e["prompt_sha256"],
            "status": "started", "timestamp": datetime.now(timezone.utc).isoformat()}
        atomic_write_json(pending_path, pending)
        result = client(api_key, config["base_url"], messages=[
            {"role": "system", "content": e["system_prompt"]},
            {"role": "user", "content": e["user_prompt"]}], max_tokens=512,
            request_options=config["request_options"])
        row = {"answer": "", "finish_reason": None, **pending, **result}
        if (config["request_options"] == {"enable_thinking": False}
            and not row.get("reasoning_content_present") and row.get("reasoning_tokens") in (None, 0)):
            row["thinking_effective"] = False
        row["status"] = classify_attempt(row, config)
        fatal = row["status"] not in TERMINAL_STATUSES
        if row["status"] == "truncated":
            row["error"] = None
        elif row["status"] != "completed" and not row.get("error"):
            row["error"] = row["status"]
        serialized = json.dumps(row, ensure_ascii=False).replace(api_key, "[REDACTED]")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        pending_path.unlink()
        records.append(json.loads(serialized))
        print(json.dumps({"question_id": e["question_id"], "status": row["status"],
            "actual_model": row["actual_model"], "finish_reason": row["finish_reason"],
            "reasoning_tokens": row["reasoning_tokens"], "latency_ms": row["latency_ms"]}), flush=True)
        if fatal:
            raise ValueError("critical generation identity/authentication/configuration gate failed; stopped")
        attempted.add(e["question_id"])
    validate_records(records, evidence, output.name, config)
    return records


def blind_assets(output, evidence, records, references):
    # Fresh mapping, randomized independently of the original DeepSeek key.
    rng = random.Random(42)
    models = [gateway.MODEL, "deepseek-flash"]
    rng.shuffle(models)
    mapping = dict(zip(("Model A", "Model B"), models))
    lookup = {(r["question_id"], r["requested_model"]): r for r in [*records, *references]}
    rows = []
    for e in evidence:
        labels = list(mapping)
        rng.shuffle(labels)
        for label in labels:
            rows.append({"experiment_id": output.name, "question_id": e["question_id"],
                "anonymous_model": label, "question": e["question"], "context": e["context"],
                "context_sha256": e["context_sha256"], "prompt_sha256": e["prompt_sha256"],
                "answer": lookup[(e["question_id"], mapping[label])]["answer"],
                "correctness": "", "faithfulness": "", "completeness": "", "hallucination": "", "review_note": ""})
    manual_path = output / "qwen_deepseek_manual_review.csv"
    key_path = output / "qwen_deepseek_blind_key.json"
    key_data = {"seed": 42, "mapping": mapping, "independent_of_previous_mapping": True}
    if manual_path.exists():
        if read_csv_rows(manual_path) != rows:
            raise ValueError("existing manual review differs; refusing to overwrite scores")
    else:
        write_csv(manual_path, rows)
    if key_path.exists():
        if json.loads(key_path.read_text(encoding="utf-8")) != key_data:
            raise ValueError("existing blind mapping differs; refusing overwrite")
    else:
        atomic_write_json(key_path, key_data)
    return rows


def export_results(output, evidence, records, references):
    from src.experiments.blind_review_sheet import build_sheet
    if (len(records) != 40 or {r["question_id"] for r in records} != {e["question_id"] for e in evidence}
        or any(r.get("status") not in TERMINAL_STATUSES for r in records)):
        raise ValueError("cannot export formal metrics/review before 40/40 attempted")
    evaluation = json.loads((output / "evaluation_config.json").read_text(encoding="utf-8"))
    refs = {e["question_id"]: e for e in evidence}
    qwen_quality = []
    qwen_per = []
    for row in records:
        e = refs[row["question_id"]]
        context = [ContextItem(**c) for c in e["chunks"]]
        quality = evaluate_answer_quality_record(e["evaluation_reference"], {
            "generated_answer": row.get("answer") or "", "strategy": "fixed_length",
            "citations": [c.to_dict() for c in build_citations(context, max_citations=3)],
            "retrieved_chunk_ids": [c.chunk_id for c in context]}, evaluation)
        qwen_quality.append(quality)
        qwen_per.append({**row, "recorded_status": row.get("recorded_status", row["status"]),
                        **{k: quality[k] for k in ("keyword_overlap_ratio", "evidence_overlap_ratio",
            "possible_missing_keywords", "possible_unsupported_terms", "auto_quality_label")}})
    write_csv(output / "qwen_cross_family_per_question.csv", qwen_per)
    all_per = [*read_csv_rows(SOURCE / "deepseek_cross_model_per_question.csv"), *qwen_per]
    completed_by_model = {model: {r["question_id"] for r in [*references, *records]
        if r["requested_model"] == model and r["status"] == "completed"} for model in MODELS}
    common_ids = sorted(set.intersection(*completed_by_model.values()))
    common_set = set(common_ids)
    atomic_write_json(output / "analysis_sets.json", {"primary": {"name": "all-attempted analysis",
        "question_ids": [e["question_id"] for e in evidence], "N": 40},
        "secondary": {"name": "completed-only sensitivity analysis", "question_ids": common_ids,
            "N": len(common_ids), "excluded_question_ids": sorted(set(refs) - common_set)}})
    summary = []
    for model in MODELS:
        model_per = [r for r in all_per if r["requested_model"] == model]
        model_records = [r for r in [*references, *records] if r["requested_model"] == model]
        completed_common = [r for r in model_per if r["question_id"] in common_set]
        rates = reliability(model_records)
        summary.append({"model": DISPLAY_NAMES[model], "requested_model": model,
            "provider": gateway.PROVIDER if model == gateway.MODEL else "deepseek", "N": len(model_per),
            **rates, "attempted": rates["attempted_count"], "completed": rates["completed_count"],
            "truncated": rates["truncated_count"], "completed_common_N": len(completed_common),
            "keyword_overlap_ratio": statistics.mean(float(r["keyword_overlap_ratio"]) for r in model_per),
            "evidence_overlap_ratio": statistics.mean(float(r["evidence_overlap_ratio"]) for r in model_per),
            "good": sum(r["auto_quality_label"] == "good" for r in model_per),
            "needs_review": sum(r["auto_quality_label"] == "needs_review" for r in model_per),
            "insufficient": sum(r["auto_quality_label"] == "insufficient" for r in model_per),
            "keyword_overlap_ratio_all40": statistics.mean(float(r["keyword_overlap_ratio"]) for r in model_per),
            "evidence_overlap_ratio_all40": statistics.mean(float(r["evidence_overlap_ratio"]) for r in model_per),
            "good_all40": sum(r["auto_quality_label"] == "good" for r in model_per),
            "needs_review_all40": sum(r["auto_quality_label"] == "needs_review" for r in model_per),
            "keyword_overlap_ratio_completed_common": statistics.mean(float(r["keyword_overlap_ratio"]) for r in completed_common) if completed_common else None,
            "evidence_overlap_ratio_completed_common": statistics.mean(float(r["evidence_overlap_ratio"]) for r in completed_common) if completed_common else None,
            "good_completed_common": sum(r["auto_quality_label"] == "good" for r in completed_common),
            "needs_review_completed_common": sum(r["auto_quality_label"] == "needs_review" for r in completed_common),
            **{"mean_" + k: statistics.mean(r[k] for r in model_records) if all(isinstance(r.get(k), (int, float)) for r in model_records) else None
               for k in ("latency_ms", "input_tokens", "output_tokens", "reasoning_tokens")}})
    write_csv(output / "cross_family_three_model_auto_summary.csv", summary)
    combined = []
    all_lookup = {(r["question_id"], r["requested_model"]): r for r in all_per}
    for e in evidence:
        combined_row = {"question_id": e["question_id"], "question": e["question"],
            "context_sha256": e["context_sha256"], "prompt_sha256": e["prompt_sha256"],
            "in_completed_common_set": e["question_id"] in common_set}
        for prefix, model in zip(("flash", "v4_pro", "qwen"), MODELS):
            row = all_lookup[(e["question_id"], model)]
            combined_row.update({prefix + "_" + name: row.get(source, "") for name, source in (
                ("answer", "answer"), ("keyword_overlap", "keyword_overlap_ratio"),
                ("evidence_overlap", "evidence_overlap_ratio"), ("quality_label", "auto_quality_label"),
                ("generation_status", "status"), ("finish_reason", "finish_reason"))})
        combined.append(combined_row)
    write_csv(output / "cross_family_three_model_per_question.csv", combined)
    atomic_write_json(output / "generation_records_effective.jsonl.manifest.json", {
        "raw_records": "generation_records.jsonl", "status_view": "generation_status_overrides.json",
        "note": "Historical raw attempts are immutable; classification override does not change answer or usage."})
    atomic_write_jsonl(output / "generation_records_effective.jsonl", records)
    atomic_write_json(output / "generation_reliability.json", reliability(records))
    atomic_write_json(output / "metrics.json", {"qwen": summarize_quality(qwen_quality),
        "qwen_reliability": reliability(records), "three_model_summary": summary})
    blind_assets(output, evidence, records, references)
    review_id = "review_" + hashlib.sha256(output.name.encode()).hexdigest()[:16]
    sheet = build_sheet(output.name, root=OUTPUT_ROOT, csv_name="qwen_deepseek_manual_review.csv",
                        review_id=review_id, allow_empty_answers=True)
    atomic_write_json(output / "blind_review_sheet_manifest.json", sheet)
    atomic_write_json(output / "manual_statistics_pipeline_config.json", {
        "status": "awaiting_manual_scores", "execute_statistics_automatically": False,
        "review_id": review_id, "score_definitions": {"correctness": [0, 1, 2], "faithfulness": [0, 1, 2],
        "completeness": [0, 1, 2], "hallucination": [0, 1]},
        "manual_csv": "qwen_deepseek_manual_review.csv", "blind_key": "qwen_deepseek_blind_key.json"})
    pipeline = json.loads((output / "manual_statistics_pipeline_config.json").read_text(encoding="utf-8"))
    pipeline["analysis_sets"] = {"all40": [e["question_id"] for e in evidence],
        "completed_common" + str(len(common_ids)): common_ids}
    pipeline["reuse_same_manual_scores"] = True
    atomic_write_json(output / "manual_statistics_pipeline_config.json", pipeline)
    write_csv(output / "experiment_registry.csv", [{"experiment_id": output.name,
        "provider": gateway.PROVIDER, "requested_model": gateway.MODEL, "actual_model": gateway.MODEL,
        "status": "completed_with_truncation" if reliability(records)["truncated_count"] else "completed",
        "question_count": 40, "legacy": False,
        "metrics_path": str(output / "metrics.json"), "per_question_results_path": str(output / "generation_records.jsonl"),
        "frozen_evidence_path": str(output / "frozen_evidence_40.jsonl")}])
    return summary


def read_csv_rows(path):
    import csv
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke1", "smoke3", "full"), required=True)
    args = parser.parse_args()
    key, _, key_env = gateway.provider_environment()
    if not key:
        print("API key: missing")
        return 1
    output = output_directory(args.experiment_id)
    evidence, references, config = prepare(output, args.diagnostics_dir, key_env)
    metadata = json.loads((output / "experiment_metadata.json").read_text(encoding="utf-8"))
    try:
        prerequisite = {"smoke3": "smoke1_pass", "full": "smoke3_pass"}.get(args.stage)
        if prerequisite and not metadata.get(prerequisite):
            raise ValueError("prior smoke gate did not pass")
        prior = load_generation_records(output)
        validate_records(prior, evidence, output.name, config)
        if prerequisite:
            n = 1 if args.stage == "smoke3" else 3
            if not {e["question_id"] for e in evidence[:n]} <= {r["question_id"] for r in prior}:
                raise ValueError("persisted smoke questions missing")
        if args.stage != "prepare":
            subset = evidence[:1] if args.stage == "smoke1" else evidence[:3] if args.stage == "smoke3" else evidence
            records = run_subset(output, subset, evidence, config, key)
            if args.stage == "full":
                validate_records(records, evidence, output.name, config, complete=True)
                if not metadata.get("exports_complete"):
                    metadata["summary"] = export_results(output, evidence, records, references)
                    metadata["exports_complete"] = True
                    metadata["manual_review_rows"] = 80
            metadata.update({args.stage + "_pass": True, "generation_request_count": len(records),
                "status": ("completed_with_truncation" if reliability(records)["truncated_count"] else "completed") if args.stage == "full" else args.stage,
                "actual_models": sorted({r["actual_model"] for r in records if isinstance(r.get("actual_model"), str)})})
            metadata.pop("stopped_stage", None)
            metadata.pop("stop_reason", None)
    except Exception as exc:
        metadata.update(status="stopped", stopped_stage=args.stage, stop_reason=str(exc).replace(key, "[REDACTED]"))
        raise
    finally:
        after = protected_snapshot()
        before = json.loads((output / "protected_before_sha256.json").read_text(encoding="utf-8"))
        atomic_write_json(output / "protected_after_sha256.json", after)
        metadata["protected_unchanged"] = before == after
        records = load_generation_records(output)
        metadata["generation_request_count"] = len(records)
        metadata["success_count"] = sum(r["status"] == "completed" for r in records)
        metadata.update(reliability(records))
        metadata["failure_count"] = sum(r["status"] in ("api_error", "generation_error") for r in records)
        atomic_write_json(output / "experiment_metadata.json", metadata)
        if before != after:
            raise ValueError("protected DeepSeek/health attempts changed")
    print(json.dumps({"stage": args.stage, "status": metadata["status"], "generation_calls": len(records),
                      "protected_unchanged": metadata["protected_unchanged"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
