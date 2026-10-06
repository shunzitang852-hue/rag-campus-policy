"""Offline frozen-input, compatibility, resume, and blind-review safety tests."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_qwen_cross_family as runner
from scripts import check_tokenpony_model_identity as gateway
from scripts.check_tokenpony_reasoning import thinking_off, allow_test_b
from src.experiments import blind_review_sheet as sheet

SECRET = "unit-qwen-secret"


@pytest.fixture
def inputs():
    # Only read existing locked evidence; never reconstruct retrieval.
    return runner.load_evidence()


@pytest.fixture
def config():
    return {"base_url": gateway.BASE_URL, "request_options": {"enable_thinking": False},
            "thinking_requested": False, "thinking_effective": False,
            "thinking_control_method": "enable_thinking_false", "temperature": 0.0,
            "max_tokens": 512, "prompt_version": "policy_grounded_qa_v1"}


def fake_completion(*args, **kwargs):
    return {"provider": "tokenpony", "base_url_type": gateway.BASE_URL_TYPE,
        "requested_model": gateway.MODEL, "actual_model": gateway.MODEL,
        "http_status": 200, "answer": "依据政策回答。", "finish_reason": "stop",
        "response_success": True, "reasoning_content_present": False,
        "reasoning_tokens": None, "thinking_effective": "unknown", "thinking_requested": False,
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        "latency_ms": 1, "timestamp": "test", "error": None}


@pytest.mark.parametrize("options", [{"reasoning_effort": "none"}, {"enable_thinking": False}])
def test_authorized_options_are_top_level(options):
    from tests.test_tokenpony_model_identity import response
    calls = []
    def opener(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        assert all(body[k] == v for k, v in options.items())
        assert "extra_body" not in body
        return response(tokens=0)
    gateway.completion(SECRET, gateway.BASE_URL, opener=opener, request_options=options)
    assert len(calls) == 1


def test_conflicting_options_rejected_before_request():
    with pytest.raises(ValueError, match="conflicting"):
        gateway.completion(SECRET, gateway.BASE_URL,
            request_options={"reasoning_effort": "none", "enable_thinking": False})


def test_test_b_absence_and_null_token_count_is_not_forced_zero():
    row = fake_completion()
    row["request_options"] = {"enable_thinking": False}
    assert thinking_off(row)
    assert row["reasoning_tokens"] is None
    row["reasoning_content_present"] = True
    assert not thinking_off(row)


@pytest.mark.parametrize("status,thinking,allowed", [(400, "unknown", True), (200, True, True),
    (200, "unknown", False), (500, "unknown", False), (None, "unknown", False)])
def test_test_b_is_bounded(status, thinking, allowed):
    assert allow_test_b({"http_status": status, "thinking_effective": thinking}) == allowed


def test_frozen_hash_and_80_deepseek_pair_hashes(inputs):
    evidence, manifest, references = inputs
    assert len(evidence) == 40 and len(references) == 80
    assert hashlib.sha256((runner.SOURCE / "frozen_evidence_40.jsonl").read_bytes()).hexdigest() == manifest["frozen_evidence_sha256"]


@pytest.mark.parametrize("tamper", ["file", "context", "prompt"])
def test_frozen_tamper_fail_fast(tmp_path, inputs, tamper):
    for name in ("frozen_evidence_40.jsonl", "frozen_evidence_manifest.json", "generation_records.jsonl"):
        (tmp_path / name).write_bytes((runner.SOURCE / name).read_bytes())
    if tamper == "file":
        with (tmp_path / "frozen_evidence_40.jsonl").open("a") as handle:
            handle.write("\n")
    else:
        rows = runner.read_rows(tmp_path / "generation_records.jsonl")
        rows[0][tamper + "_sha256"] = "tampered"
        (tmp_path / "generation_records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="SHA-256|Context/Prompt"):
        runner.load_evidence(tmp_path)


def test_resume_reuses_smoke_and_preserves_all_sources(tmp_path, inputs, config, capsys):
    evidence, _, _ = inputs
    before = runner.protected_snapshot()
    calls = []
    def client(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["max_tokens"] == 512
        assert kwargs["request_options"] == {"enable_thinking": False}
        return fake_completion()
    runner.run_subset(tmp_path, evidence[:1], evidence, config, SECRET, client)
    runner.run_subset(tmp_path, evidence[:3], evidence, config, SECRET, client)
    final = runner.run_subset(tmp_path, evidence, evidence, config, SECRET, client)
    resumed = runner.run_subset(tmp_path, evidence, evidence, config, SECRET, client)
    assert len(calls) == 40 and len(final) == 40 and resumed == final
    runner.validate_records(final, evidence, tmp_path.name, config, complete=True)
    assert before == runner.protected_snapshot()
    assert SECRET not in (tmp_path / "generation_records.jsonl").read_text(encoding="utf-8")
    assert SECRET not in capsys.readouterr().out
    assert all(r["reasoning_tokens"] is None and r["thinking_effective"] is False for r in final)
    for sent, e in zip(calls, evidence):
        assert sent["messages"] == [{"role": "system", "content": e["system_prompt"]}, {"role": "user", "content": e["user_prompt"]}]


@pytest.mark.parametrize("change", [{"actual_model": "qwen-plus"},
    {"reasoning_content_present": True, "thinking_effective": True}, {"http_status": 401}])
def test_failures_stop_and_never_retry(tmp_path, inputs, config, change):
    evidence = inputs[0]
    calls = []
    def client(*args, **kwargs):
        calls.append(1)
        return {**fake_completion(), **change}
    with pytest.raises(ValueError, match="critical generation"):
        runner.run_subset(tmp_path, evidence[:3], evidence, config, SECRET, client)
    with pytest.raises(ValueError, match="persisted record"):
        runner.run_subset(tmp_path, evidence[:3], evidence, config, SECRET, client)
    assert len(calls) == 1


def test_pending_unknown_request_blocks_duplicate_charge(tmp_path, inputs, config):
    (tmp_path / "pending_request.json").write_text(json.dumps({"question_id": inputs[0][0]["question_id"]}))
    with pytest.raises(ValueError, match="unknown outcome"):
        runner.run_subset(tmp_path, inputs[0][:1], inputs[0], config, SECRET,
                          lambda *a, **k: pytest.fail("must not call"))


def test_incomplete_40_is_rejected(tmp_path, inputs, config):
    rows = runner.run_subset(tmp_path, inputs[0][:3], inputs[0], config, SECRET, fake_completion)
    with pytest.raises(ValueError, match="40 unique"):
        runner.validate_records(rows, inputs[0], tmp_path.name, config, complete=True)


def test_no_retrieval_is_called(tmp_path, inputs, config, monkeypatch):
    import scripts.run_deepseek_cross_model as deepseek
    monkeypatch.setattr(deepseek, "freeze_rows", lambda: pytest.fail("must not freeze/retrieve"))
    monkeypatch.setattr(deepseek, "load_frozen_inputs", lambda: pytest.fail("must not retrieve"))
    evidence = runner.load_evidence()[0]
    runner.run_subset(tmp_path, evidence[:1], evidence, config, SECRET, fake_completion)


def test_new_output_root_and_invalid_ids():
    name = "20261006T000000Z_qwen38max_tokenpony_cross_family"
    assert runner.output_directory(name).parent == runner.OUTPUT_ROOT.resolve()
    for bad in (gateway.SOURCE_ID, "../bad", "20261006T065148355806Z_qwen_tokenpony_cross_family"):
        with pytest.raises(ValueError):
            runner.output_directory(bad)


def test_blind_80_rows_deterministic_no_identity_in_html(tmp_path, inputs, config):
    evidence, _, references = inputs
    out = tmp_path / "independent_experiment"
    out.mkdir()
    records = runner.run_subset(out, evidence, evidence, config, SECRET, fake_completion)
    first = runner.blind_assets(out, evidence, records, references)
    key_bytes = (out / "qwen_deepseek_blind_key.json").read_bytes()
    second = runner.blind_assets(out, evidence, records, references)
    assert first == second and len(first) == 80
    assert (out / "qwen_deepseek_blind_key.json").read_bytes() == key_bytes
    assert all(all(r[k] == "" for k in sheet.EDITABLE) for r in first)
    original = (out / "qwen_deepseek_manual_review.csv").read_bytes()
    result = sheet.build_sheet(out.name, root=tmp_path, csv_name="qwen_deepseek_manual_review.csv", review_id="review_opaque")
    html = Path(result["html"]).read_text(encoding="utf-8")
    assert "qwen" not in html.lower() and "deepseek" not in html.lower() and "tokenpony" not in html.lower()
    assert "actual_model" not in html and "requested_model" not in html
    assert original == (out / "qwen_deepseek_manual_review.csv").read_bytes()
    scores = {"schema_version": sheet.SCHEMA, "experiment_id": "review_opaque", "source_csv_sha256": sheet.sha256(original),
        "rows": [{"question_id": r["question_id"], "anonymous_model": r["anonymous_model"],
                  "correctness": "1", "faithfulness": "1", "completeness": "1", "hallucination": "0", "review_note": "synthetic test"} for r in first]}
    score_path = out / "scores.json"
    score_path.write_text(json.dumps(scores))
    imported = sheet.import_scores(out.name, score_path, root=tmp_path, csv_name="qwen_deepseek_manual_review.csv", review_id="review_opaque")
    assert imported["status"] == "imported"
    assert Path(imported["backup_csv"]).read_bytes() == original
    with pytest.raises(ValueError, match="overwrite scores"):
        runner.blind_assets(out, evidence, records, references)


def test_saved_test_b_overrides_overly_strict_initial_selection(tmp_path):
    row = {**fake_completion(), "test": "B", "request_options": {"enable_thinking": False}}
    (tmp_path / "compatibility_result.json").write_text(json.dumps({
        "protected_unchanged": True, "tests": [row],
        "selected_config": {"thinking_control_method": "disable_unavailable"}}))
    chosen = runner.selection(tmp_path)
    assert chosen["thinking_control_method"] == "enable_thinking_false"
    assert chosen["thinking_effective"] is False and chosen["max_tokens"] == 512


def test_export_reproduces_reference_evaluator_and_all_assets(tmp_path, inputs, config, monkeypatch):
    evidence, _, references = inputs
    out = tmp_path / "synthetic_export"
    out.mkdir()
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    # Reuse Flash text only in an explicitly synthetic offline test directory.
    flash = {r["question_id"]: r for r in references if r["requested_model"] == "deepseek-flash"}
    calls = iter(evidence)
    def client(*args, **kwargs):
        e = next(calls)
        return {**fake_completion(), "answer": flash[e["question_id"]]["answer"]}
    records = runner.run_subset(out, evidence, evidence, config, SECRET, client)
    (out / "evaluation_config.json").write_bytes((runner.SOURCE / "evaluation_config.json").read_bytes())
    summary = runner.export_results(out, evidence, records, references)
    assert len(summary) == 3 and all(r["N"] == 40 for r in summary)
    for field in ("keyword_overlap_ratio", "evidence_overlap_ratio", "good", "needs_review"):
        assert summary[0][field] == summary[2][field]
    combined = runner.read_csv_rows(out / "cross_family_three_model_per_question.csv")
    assert len(combined) == 40
    assert len({r["question_id"] for r in combined}) == 40
    manual = runner.read_csv_rows(out / "qwen_deepseek_manual_review.csv")
    assert len(manual) == 80 and all(all(r[k] == "" for k in sheet.EDITABLE) for r in manual)
    assert json.loads((out / "manual_statistics_pipeline_config.json").read_text())["execute_statistics_automatically"] is False
    assert SECRET not in (out / "generation_records.jsonl").read_text(encoding="utf-8")


def test_incomplete_run_never_exports_fake_metrics_or_review(tmp_path, inputs, config):
    evidence, _, references = inputs
    records = runner.run_subset(tmp_path, evidence[:3], evidence, config, SECRET, fake_completion)
    with pytest.raises(ValueError, match="40/40"):
        runner.export_results(tmp_path, evidence, records, references)
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "qwen_deepseek_manual_review.csv").exists()


def test_truncation_resume_partial_answer_dual_analysis_and_blind(tmp_path, inputs, config, monkeypatch):
    evidence, _, references = inputs
    out = tmp_path / "synthetic_truncated"
    out.mkdir()
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    calls = []
    def client(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["max_tokens"] == 512 and kwargs["request_options"] == {"enable_thinking": False}
        if len(calls) == 23:
            return {**fake_completion(), "answer": "真实部分答案，不补全。", "finish_reason": "length", "output_tokens": 512}
        return fake_completion()
    first = runner.run_subset(out, evidence[:23], evidence, config, SECRET, client)
    assert first[22]["status"] == "truncated"
    assert first[22]["answer"] == "真实部分答案，不补全。"
    raw_prefix = (out / "generation_records.jsonl").read_bytes()
    rows = runner.run_subset(out, evidence, evidence, config, SECRET, client)
    resumed = runner.run_subset(out, evidence, evidence, config, SECRET, client)
    assert len(calls) == 40 and resumed == rows
    assert (out / "generation_records.jsonl").read_bytes().startswith(raw_prefix)
    runner.validate_records(rows, evidence, out.name, config, complete=True)
    counts = runner.reliability(rows)
    assert counts["completed_count"] == 39 and counts["truncated_count"] == 1
    assert counts["api_error_count"] == 0 and counts["completion_rate"] == 0.975
    (out / "evaluation_config.json").write_bytes((runner.SOURCE / "evaluation_config.json").read_bytes())
    rows[22]["recorded_status"] = "failed"  # Old immutable status appears on only one derived row.
    summaries = runner.export_results(out, evidence, rows, references)
    assert all(s["N"] == 40 and s["completed_common_N"] == 39 for s in summaries)
    sets = json.loads((out / "analysis_sets.json").read_text(encoding="utf-8"))
    assert sets["secondary"]["excluded_question_ids"] == [evidence[22]["question_id"]]
    per = runner.read_csv_rows(out / "cross_family_three_model_per_question.csv")
    q23 = next(r for r in per if r["question_id"] == evidence[22]["question_id"])
    assert q23["qwen_generation_status"] == "truncated"
    assert q23["qwen_answer"] == "真实部分答案，不补全。"
    manual = runner.read_csv_rows(out / "qwen_deepseek_manual_review.csv")
    assert len(manual) == 80 and sum(r["question_id"] == evidence[22]["question_id"] for r in manual) == 2
    assert any(r["answer"] == "真实部分答案，不补全。" for r in manual)
    html = (out / sheet.HTML_NAME).read_text(encoding="utf-8")
    for forbidden in ("truncated", "finish_reason", "qwen", "deepseek", "tokenpony"):
        assert forbidden not in html.lower()
    assert all(all(r[k] == "" for k in sheet.EDITABLE) for r in manual)


def test_legacy_failed_truncation_status_view_preserves_raw_attempt(tmp_path, inputs, config):
    evidence = inputs[0]
    rows = runner.run_subset(tmp_path, evidence[:1], evidence, config, SECRET, fake_completion)
    rows[0].update(status="failed", finish_reason="length", output_tokens=512, answer="部分原答案")
    raw = (json.dumps(rows[0], ensure_ascii=False) + "\n").encode()
    path = tmp_path / "generation_records.jsonl"
    path.write_bytes(raw)
    override = {"overrides": [{"experiment_id": tmp_path.name, "question_id": evidence[0]["question_id"],
        "requested_model": gateway.MODEL, "source_record_sha256": hashlib.sha256(raw).hexdigest(),
        "recorded_status": "failed", "effective_status": "truncated"}]}
    (tmp_path / "generation_status_overrides.json").write_text(json.dumps(override))
    effective = runner.run_subset(tmp_path, evidence[:1], evidence, config, SECRET,
        lambda *a, **k: pytest.fail("existing truncated attempt must not be requested"))
    assert effective[0]["status"] == "truncated" and effective[0]["recorded_status"] == "failed"
    assert path.read_bytes() == raw
    path.write_bytes(raw + b" ")
    # Changing the actual protected line, not an unrelated blank line, must fail.
    path.write_bytes(raw.replace(b'"failed"', b'"changed"'))
    with pytest.raises(ValueError, match="integrity"):
        runner.load_generation_records(tmp_path)


@pytest.mark.parametrize("change,status", [({"answer": "", "response_success": False}, "generation_error"),
    ({"http_status": 500, "response_success": False, "actual_model": None, "error": "HTTP 500"}, "api_error")])
def test_nonfatal_errors_continue_and_are_not_retried(tmp_path, inputs, config, change, status):
    evidence = inputs[0]
    calls = []
    def client(*args, **kwargs):
        calls.append(1)
        return {**fake_completion(), **change} if len(calls) == 1 else fake_completion()
    rows = runner.run_subset(tmp_path, evidence[:3], evidence, config, SECRET, client)
    assert rows[0]["status"] == status and len(rows) == 3
    runner.run_subset(tmp_path, evidence[:3], evidence, config, SECRET, client)
    assert len(calls) == 3
