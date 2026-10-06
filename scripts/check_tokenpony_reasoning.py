"""Bounded reasoning compatibility checks, separately from locked health attempts."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import check_tokenpony_model_identity as gateway

MESSAGES = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply exactly: OK"}]


def normal_answer(row):
    return (gateway.identity_pass(row) and row.get("finish_reason") == "stop"
            and isinstance(row.get("answer"), str) and bool(row["answer"].strip()))


def thinking_off(row):
    explicit_b_no_reasoning = (row.get("request_options") == {"enable_thinking": False}
        and not row.get("reasoning_content_present") and row.get("reasoning_tokens") in (None, 0))
    return normal_answer(row) and (row.get("thinking_effective") is False or explicit_b_no_reasoning)


def allow_test_b(row):
    status = row.get("http_status")
    return ((isinstance(status, int) and 400 <= status < 500) or
            (status == 200 and row.get("thinking_effective") is True))


def main():
    key, base, key_env = gateway.provider_environment()
    if not key:
        print("API key: missing")
        return 1
    output = gateway.ROOT / "qwen" / "diagnostics" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_reasoning_compatibility")
    output.mkdir(parents=True, exist_ok=False)
    locked = {"deepseek": gateway.SOURCE,
              "dashscope_attempt": gateway.ROOT / "qwen" / "20261006T064519132741Z_qwen_cross_model",
              "tokenpony_attempt": gateway.ROOT / "qwen" / "20261006T065148355806Z_qwen_tokenpony_cross_family"}
    before = {name: gateway.snapshot(path) for name, path in locked.items()}
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2).replace(key, "[REDACTED]") + "\n", encoding="utf-8")
    save("protected_before_sha256.json", before)
    result = {"provider": gateway.PROVIDER, "requested_model": gateway.MODEL, "api_key_env": key_env,
              "prior_identity_gate_reused": True, "output_dir": str(output), "tests": [], "selected_config": None}
    try:
        for name, options in (("A", {"reasoning_effort": "none"}), ("B", {"enable_thinking": False})):
            row = gateway.completion(key, base, messages=MESSAGES, max_tokens=128, request_options=options)
            row.update(test=name, request_options=options, max_tokens=128)
            result["tests"].append(row)
            save("test_" + name.lower() + ".json", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            # Never reinterpret identity drift, including on error responses.
            if row.get("http_status") == 200 and row.get("actual_model") != gateway.MODEL:
                result["stop_reason"] = "model identity drift"
                break
            if thinking_off(row):
                result["selected_config"] = {"request_options": options,
                    "thinking_requested": False, "thinking_effective": False,
                    "thinking_control_method": "reasoning_effort_none" if name == "A" else "enable_thinking_false",
                    "temperature": 0.0, "max_tokens": 512, "prompt_version": "policy_grounded_qa_v1"}
                break
            if name == "A" and not allow_test_b(row):
                result["stop_reason"] = "Test A failed without an authorized Test B condition"
                break
        if not result["selected_config"] and "stop_reason" not in result:
            # Both authorized disable methods failed. The one budget check is authorized.
            row = gateway.completion(key, base, messages=MESSAGES, max_tokens=512)
            row.update(test="reasoning_enabled_budget512", request_options={}, max_tokens=512)
            result["tests"].append(row)
            save("budget512.json", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if normal_answer(row):
                result["selected_config"] = {"request_options": {}, "thinking_requested": False,
                    "thinking_effective": row["thinking_effective"], "thinking_control_method": "disable_unavailable",
                    "temperature": 0.0, "max_tokens": 512, "prompt_version": "policy_grounded_qa_v1"}
            else:
                result["stop_reason"] = "512-token health failed; no formal requests allowed"
    finally:
        after = {name: gateway.snapshot(path) for name, path in locked.items()}
        save("protected_after_sha256.json", after)
        result["protected_unchanged"] = before == after
        save("compatibility_result.json", result)
    if not result["protected_unchanged"]:
        raise ValueError("protected artifacts changed")
    print(json.dumps({"diagnostics": str(output), "selected_config": result["selected_config"],
                      "protected_unchanged": result["protected_unchanged"]}), flush=True)
    return 0 if result["selected_config"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
