import io
import json
from pathlib import Path

import pytest

from scripts.check_deepseek_model_identity import health_check, identity_gate, MODELS
from scripts.run_deepseek_cross_model import (freeze_rows, validate_evidence, run_subset,
    check_pairs, LEGACY_ID, sha)


class Response(io.BytesIO):
    status = 200


def test_actual_model_is_from_response_and_secret_is_redacted():
    secret = "dummy-unit-test-secret"
    def opener(request, timeout):
        assert request.get_header("Authorization") == "Bearer " + secret
        payload = json.loads(request.data)
        assert payload["thinking"] == {"type": "disabled"}
        return Response(json.dumps({"model": "different-returned-model", "id": secret,
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}).encode())
    row = health_check(MODELS[0], secret, opener)
    assert row["actual_model"] == "different-returned-model"
    assert row["requested_model"] == MODELS[0]
    assert secret not in json.dumps(row)
    assert row["thinking_effective"] == "unknown"


def test_missing_model_stays_unknown_and_errors_do_not_leak():
    row = health_check(MODELS[0], "dummy-secret", lambda *a, **k: Response(
        b'{"choices": [{"message": {"content": "OK"}}]}'))
    assert row["actual_model"] == "unknown"
    def error(*a, **k):
        raise RuntimeError("Bearer dummy-secret")
    failed = health_check(MODELS[0], "dummy-secret", error)
    assert failed["error"] == "RuntimeError"
    assert "dummy-secret" not in json.dumps(failed)


@pytest.mark.parametrize("actual,success,gate", [
    (("one", "one"), True, "same_actual_model_stop"),
    (("one", "two"), True, "distinct_actual_models"),
    (("unknown", "unknown"), True, "unknown_config_comparison_only"),
    (("one", "unknown"), True, "identity_inconclusive"),
    (("one", "two"), False, "health_check_failed")])
def test_identity_gate(actual, success, gate):
    assert identity_gate([{"actual_model": a, "response_success": success} for a in actual]) == gate


def test_frozen_evidence_deterministic_and_tamper_rejected():
    rows = freeze_rows()
    assert rows == freeze_rows()
    validate_evidence(rows)
    assert len(rows) == 40
    assert all(sha(r["context"]) == r["context_sha256"] for r in rows)
    rows[0]["chunks"][0]["text"] += "tampered"
    with pytest.raises(ValueError, match="integrity"):
        validate_evidence(rows)


def test_resume_no_duplicate_calls_and_same_context(tmp_path, monkeypatch, capsys):
    output = tmp_path / "new_experiment"
    output.mkdir()
    health = [{"requested_model": m, "actual_model": m} for m in MODELS]
    (output / "health_checks.jsonl").write_text("\n".join(json.dumps(r) for r in health))
    evidence = freeze_rows()[:1]
    calls = []
    secret = "dummy-not-on-disk"
    def client(model, api_key, **kwargs):
        calls.append((model, kwargs))
        return {"requested_model": model, "actual_model": model, "response_success": True,
            "response_metadata": {"usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
            "answer": "answer", "finish_reason": "stop", "latency_ms": 1, "timestamp": "test"}
    first = run_subset(output, evidence, secret, client)
    second = run_subset(output, evidence, secret, client)
    assert len(calls) == 2
    assert first == second
    assert calls[0][1] == calls[1][1]
    assert check_pairs(second, evidence, output.name)
    assert first[0]["context_sha256"] == first[1]["context_sha256"]
    assert secret not in (output / "generation_records.jsonl").read_text()
    assert secret not in capsys.readouterr().out
    assert not (output / "pending_request.json").exists()


def test_generation_does_not_retrieve(tmp_path, monkeypatch):
    import scripts.run_deepseek_cross_model as runner
    evidence = freeze_rows()[:1]
    def forbidden(*a, **k):
        raise AssertionError("generation must not reconstruct evidence or retrieve")
    monkeypatch.setattr(runner, "freeze_rows", forbidden)
    monkeypatch.setattr(runner, "load_frozen_inputs", forbidden)
    output = tmp_path / "new_experiment"
    output.mkdir()
    (output / "health_checks.jsonl").write_text("\n".join(json.dumps({"requested_model": m, "actual_model": m}) for m in MODELS))
    def client(model, key, **kwargs):
        return {"response_metadata": {}, "actual_model": model, "response_success": True,
                "answer": "OK", "finish_reason": "stop", "latency_ms": 1}
    assert len(run_subset(output, evidence, "dummy-secret", client)) == 2


def test_legacy_id_is_rejected_before_any_write(monkeypatch):
    import scripts.run_deepseek_cross_model as runner
    monkeypatch.setattr("sys.argv", ["runner", "--experiment-id", LEGACY_ID, "--stage", "prepare"])
    with pytest.raises(ValueError, match="invalid experiment"):
        runner.main()


def test_failed_response_is_not_automatically_retried(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "health_checks.jsonl").write_text("\n".join(json.dumps({"requested_model": m, "actual_model": m}) for m in MODELS))
    calls = []
    def fail(model, key, **kwargs):
        calls.append(model)
        return {"actual_model": "unknown", "response_success": False, "response_metadata": {}, "error": "HTTP 500", "latency_ms": 1}
    evidence = freeze_rows()[:1]
    with pytest.raises(ValueError, match="failed"):
        run_subset(output, evidence, "dummy-key", fail)
    with pytest.raises(ValueError, match="failed"):
        run_subset(output, evidence, "dummy-key", fail)
    assert len(calls) == 1
