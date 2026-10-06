"""TokenPony identity gate using the existing OpenAI-compatible client.

This gateway's returned model identifier is not proof of a weight snapshot.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_qwen_model_identity import environment_value, snapshot, ROOT, SOURCE, SOURCE_ID
from src.rag.llm_provider import OpenAICompatibleProvider

MODEL = "qwen3.8-max"
BASE_URL = "https://api.tokenpony.cn/v1"
PROVIDER = "tokenpony"
BASE_URL_TYPE = "third_party_openai_compatible"


def provider_environment():
    key = environment_value("TOKENPONY_API_KEY")
    key_env = "TOKENPONY_API_KEY"
    if not key:
        key = environment_value("DASHSCOPE_API_KEY")
        key_env = "DASHSCOPE_API_KEY"
    base = environment_value("TOKENPONY_BASE_URL") or BASE_URL
    if base.rstrip("/") != BASE_URL:
        raise ValueError("TOKENPONY_BASE_URL differs from the authorized gateway; stopped")
    return key, base.rstrip("/"), key_env


def model_identity_pass(row):
    return row.get("http_status") == 200 and row.get("actual_model") == MODEL


def identity_pass(row):
    """Continuation needs both the identity and a usable health answer."""
    return model_identity_pass(row) and row.get("response_success") is True


def completion(api_key, base_url, *, messages=None, max_tokens=32, request_options=None, opener=urlopen):
    options = dict(request_options or {})
    if options not in ({}, {"reasoning_effort": "none"}, {"enable_thinking": False}):
        raise ValueError("unsupported or conflicting thinking options")
    client = OpenAICompatibleProvider(base_url=base_url, api_key=api_key,
        model_name=MODEL, temperature=0.0, max_tokens=max_tokens, request_options=options)
    payload = client.build_payload("Reply briefly.", "Reply OK.")
    if messages is not None:
        payload["messages"] = messages
    # Only the two explicitly authorized compatibility options are permitted.
    request = Request(client.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    row = dict(provider=PROVIDER, base_url_type=BASE_URL_TYPE, requested_model=MODEL,
        actual_model=None, http_status=None, latency_ms=None, input_tokens=None,
        output_tokens=None, reasoning_tokens=None, total_tokens=None,
        response_success=False, reasoning_content_present=False, thinking_requested=False,
        thinking_effective="unknown", timestamp=datetime.now(timezone.utc).isoformat(), error=None)
    start = time.perf_counter()
    try:
        with opener(request, timeout=60) as response:
            row["http_status"] = response.status
            result = json.loads(response.read().decode("utf-8"))
        row["actual_model"] = result.get("model")
        usage = result.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens", usage.get("reasoning_tokens"))
        row.update(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                   reasoning_tokens=reasoning_tokens, total_tokens=usage.get("total_tokens"))
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        row["answer"] = message.get("content")
        row["finish_reason"] = choice.get("finish_reason")
        row["reasoning_content_present"] = bool(message.get("reasoning_content"))
        row["response_success"] = row["http_status"] == 200 and isinstance(row["answer"], str) and bool(row["answer"].strip())
        if row["reasoning_content_present"] or (isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0):
            row["thinking_effective"] = True
        elif reasoning_tokens == 0 and not row["reasoning_content_present"]:
            row["thinking_effective"] = False
        if message.get("tool_calls") or message.get("function_call"):
            row.update(response_success=False, error="unexpected tool/function call; stopped")
    except HTTPError as exc:
        row["http_status"] = exc.code
        row["error"] = "HTTP " + str(exc.code)
        try:
            body = json.loads(exc.read().decode("utf-8"))
            error = body.get("error", body)
            if isinstance(error, dict):
                safe = {k: error[k] for k in ("code", "type", "message") if isinstance(error.get(k), (str, int))}
                if safe:
                    row["error"] = safe
        except Exception:
            pass
    except Exception as exc:
        row["error"] = type(exc).__name__
    row["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return json.loads(json.dumps(row, ensure_ascii=False).replace(api_key, "[REDACTED]"))


def main():
    key, base, key_env = provider_environment()
    print("provider: tokenpony\nbase_url: " + base)
    print(key_env + (": configured" if key else ": missing"))
    if not key:
        return 1
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_qwen_tokenpony_cross_family"
    output = ROOT / "qwen" / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot(SOURCE)
    previous = ROOT / "qwen" / "20261006T064519132741Z_qwen_cross_model"
    previous_before = snapshot(previous)
    def save(name, data):
        text = json.dumps(data, ensure_ascii=False, indent=2).replace(key, "[REDACTED]")
        (output / name).write_text(text + "\n", encoding="utf-8")
    save("deepseek_before_sha256.json", before)
    save("previous_attempt_before_sha256.json", previous_before)
    row = completion(key, base)
    save("health_check.json", row)
    after = snapshot(SOURCE)
    previous_after = snapshot(previous)
    save("deepseek_after_sha256.json", after)
    save("previous_attempt_after_sha256.json", previous_after)
    metadata = dict(experiment_id=experiment_id, provider=PROVIDER, base_url=base,
        base_url_type=BASE_URL_TYPE, api_key_env=key_env, api_key_configured=True,
        requested_model=MODEL, actual_model=row["actual_model"], health_pass=identity_pass(row),
        model_identity_pass=model_identity_pass(row),
        status="health_pass" if identity_pass(row) else "stopped_at_health_gate",
        health_request_count=1, generation_request_count=0, source_experiment=SOURCE_ID,
        deepseek_unchanged=before == after, previous_attempt_unchanged=previous_before == previous_after,
        output_dir=str(output), thinking_requested=False,
        thinking_effective=row["thinking_effective"], thinking_parameter_sent=False,
        thinking_disable_support="unverified; no undocumented parameter sent")
    save("experiment_metadata.json", metadata)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    if before != after or previous_before != previous_after:
        raise ValueError("protected assets changed")
    return 0 if identity_pass(row) else 2


if __name__ == "__main__":
    raise SystemExit(main())
