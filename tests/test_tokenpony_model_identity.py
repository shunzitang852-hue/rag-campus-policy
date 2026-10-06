"""Offline health-gate tests: no paid API calls or retrieval."""
import io
import json
from urllib.error import HTTPError

import pytest

from scripts import check_tokenpony_model_identity as gate


class Response(io.BytesIO):
    status = 200


def response(model=gate.MODEL, content="OK", reasoning="", tokens=None, finish="stop"):
    return Response(json.dumps({"model": model,
        "choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 32, "total_tokens": 37,
                  "completion_tokens_details": {"reasoning_tokens": tokens}}}).encode())


def test_provider_environment_prefers_tokenpony(monkeypatch):
    values = {"TOKENPONY_API_KEY": "unit-new-key", "DASHSCOPE_API_KEY": "unit-old-key",
              "TOKENPONY_BASE_URL": gate.BASE_URL}
    monkeypatch.setattr(gate, "environment_value", lambda name: values.get(name, ""))
    assert gate.provider_environment() == ("unit-new-key", gate.BASE_URL, "TOKENPONY_API_KEY")


def test_legacy_key_compatibility_and_explicit_base(monkeypatch):
    monkeypatch.setattr(gate, "environment_value", lambda name: "unit-old-key" if name == "DASHSCOPE_API_KEY" else "")
    assert gate.provider_environment() == ("unit-old-key", gate.BASE_URL, "DASHSCOPE_API_KEY")


def test_missing_key_and_wrong_base(monkeypatch):
    monkeypatch.setattr(gate, "environment_value", lambda name: "")
    assert gate.provider_environment()[0] == ""
    monkeypatch.setattr(gate, "environment_value", lambda name: "https://example.invalid" if name == "TOKENPONY_BASE_URL" else "dummy")
    with pytest.raises(ValueError, match="authorized gateway"):
        gate.provider_environment()


def test_payload_pins_model_without_tools_or_fake_thinking():
    calls = []
    def opener(request, timeout):
        calls.append(json.loads(request.data))
        assert request.full_url == gate.BASE_URL + "/chat/completions"
        assert request.get_header("Authorization") == "Bearer unit-secret"
        assert timeout == 60
        return response()
    row = gate.completion("unit-secret", gate.BASE_URL, opener=opener)
    assert len(calls) == 1
    assert set(calls[0]) == {"model", "messages", "temperature", "max_tokens"}
    assert calls[0]["model"] == "qwen3.8-max"
    assert calls[0]["temperature"] == 0.0
    assert row["provider"] == "tokenpony"
    assert row["base_url_type"] == "third_party_openai_compatible"
    assert gate.identity_pass(row)


@pytest.mark.parametrize("actual", ["qwen3.8-max-0902", "qwen-plus", None])
def test_no_fallback_or_inferred_identity(actual):
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        return response(model=actual)
    row = gate.completion("unit-secret", gate.BASE_URL, opener=opener)
    assert row["actual_model"] == actual
    assert not gate.model_identity_pass(row)
    assert len(calls) == 1


def test_reasoning_budget_exhaustion_blocks_continuation():
    row = gate.completion("unit-secret", gate.BASE_URL,
        opener=lambda *a, **k: response(content="", reasoning="thinking", tokens=32, finish="length"))
    assert gate.model_identity_pass(row)
    assert not gate.identity_pass(row)
    assert row["thinking_requested"] is False
    assert row["thinking_effective"] is True
    assert row["reasoning_content_present"] is True
    assert row["reasoning_tokens"] == 32


def test_no_reasoning_does_not_prove_disabled():
    row = gate.completion("unit-secret", gate.BASE_URL, opener=lambda *a, **k: response(tokens=None))
    assert row["thinking_effective"] == "unknown"


def test_secret_redaction_and_no_automatic_retry(tmp_path):
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        body = io.BytesIO(b'{"error":{"message":"unit-secret rejected","code":"invalid_key"}}')
        raise HTTPError(gate.BASE_URL, 401, "unauthorized", {}, body)
    row = gate.completion("unit-secret", gate.BASE_URL, opener=opener)
    serialized = json.dumps(row)
    path = tmp_path / "record.json"
    path.write_text(serialized)
    assert "unit-secret" not in path.read_text()
    assert row["http_status"] == 401 and not gate.identity_pass(row)
    assert len(calls) == 1


def test_exception_text_never_saved():
    def opener(*args, **kwargs):
        raise RuntimeError("authorization: Bearer unit-secret")
    row = gate.completion("unit-secret", gate.BASE_URL, opener=opener)
    assert row["error"] == "RuntimeError"
    assert "unit-secret" not in json.dumps(row)


def test_unexpected_tools_block_completion():
    result = {"model": gate.MODEL, "choices": [{"message": {"content": "OK", "tool_calls": [{"id": "one"}]}}]}
    row = gate.completion("unit-secret", gate.BASE_URL, opener=lambda *a, **k: Response(json.dumps(result).encode()))
    assert not gate.identity_pass(row)
    assert "tool" in row["error"]
