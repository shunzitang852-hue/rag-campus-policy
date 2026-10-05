"""Minimal, secret-safe DeepSeek identity gate; no automatic retries."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODELS = ("deepseek-flash", "deepseek-v4-pro")
OUTPUT_ROOT = Path("F:/GraduationThesis/experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness/deepseek")


def identity_gate(rows):
    if len(rows) != 2 or not all(r["response_success"] for r in rows):
        return "health_check_failed"
    actual = [r["actual_model"] for r in rows]
    if actual == ["unknown", "unknown"]:
        return "unknown_config_comparison_only"
    if "unknown" in actual:
        return "identity_inconclusive"
    if not all(isinstance(a, str) and a for a in actual):
        return "identity_inconclusive"
    return "same_actual_model_stop" if actual[0] == actual[1] else "distinct_actual_models"


def health_check(model, api_key, opener=urlopen, *, messages=None, max_tokens=8):
    payload = {"model": model, "messages": messages or [{"role": "user", "content": "Reply OK."}],
               "temperature": 0.0, "max_tokens": max_tokens, "thinking": {"type": "disabled"}}
    request = Request("https://api.deepseek.com/chat/completions",
                      data=json.dumps(payload).encode(),
                      headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                      method="POST")
    row = {"provider": "deepseek", "requested_model": model, "actual_model": "unknown",
           "http_status": None, "response_success": False, "latency_ms": None,
           "timestamp": datetime.now(timezone.utc).isoformat(), "thinking_requested": "disabled",
           "thinking_effective": "unknown", "error": None, "response_metadata": {}}
    started = time.perf_counter()
    try:
        with opener(request, timeout=60) as response:
            row["http_status"] = response.status
            result = json.loads(response.read().decode("utf-8"))
        # Preserve the API's identity verbatim; never substitute the requested alias.
        row["actual_model"] = result["model"] if "model" in result else "unknown"
        row["response_metadata"] = {k: result[k] for k in
            ("id", "object", "created", "model", "system_fingerprint", "service_tier", "usage") if k in result}
        message = result.get("choices", [{}])[0].get("message", {})
        row["finish_reason"] = result.get("choices", [{}])[0].get("finish_reason")
        row["answer"] = message.get("content")
        row["response_success"] = 200 <= row["http_status"] < 300 and bool(row["answer"])
        row["reasoning_content_present"] = bool(message.get("reasoning_content"))
        # Absence of reasoning text alone does not establish effective thinking mode.
        if result.get("thinking", {}).get("type") in ("enabled", "disabled"):
            row["thinking_effective"] = result["thinking"]["type"]
    except HTTPError as exc:
        row["http_status"] = exc.code
        row["error"] = "HTTP " + str(exc.code)
    except Exception as exc:
        # Exception text or raw error bodies may contain credentials. Store type only.
        row["error"] = type(exc).__name__
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    # Defense in depth against a service unexpectedly echoing the credential.
    return json.loads(json.dumps(row, ensure_ascii=False).replace(api_key, "[REDACTED]"))


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY: missing; current process did not inherit the variable")
        return 1
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_deepseek_cross_model"
    output = OUTPUT_ROOT / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for model in MODELS:
        row = health_check(model, api_key)
        rows.append(row)
        with (output / "health_checks.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    metadata = {"experiment_id": experiment_id, "api_key_configured": True,
                "gate": identity_gate(rows), "health_request_count": 2,
                "generation_request_count": 0, "output_dir": str(output)}
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
