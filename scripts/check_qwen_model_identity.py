"""Fixed DashScope snapshot health gate; no fallback, tools or automatic retries."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.llm_provider import OpenAICompatibleProvider

MODEL = "qwen3.8-max-0902"
ROOT = Path("F:/GraduationThesis/experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness")
SOURCE_ID = "20261005T124451501115Z_deepseek_cross_model"
SOURCE = ROOT / "deepseek" / SOURCE_ID


def environment_value(name):
    value = os.environ.get(name, "")
    if not value and os.name == "nt":
        import winreg
        for hive, path in ((winreg.HKEY_CURRENT_USER, "Environment"),
                           (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, path) as key:
                    value = winreg.QueryValueEx(key, name)[0]
                if value:
                    break
            except OSError:
                pass
    return value


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def health_check(api_key, base_url, opener=urlopen):
    client = OpenAICompatibleProvider(base_url=base_url, api_key=api_key,
        model_name=MODEL, temperature=0.0, max_tokens=8)
    # No undocumented reasoning parameter: requested off, effective unknown.
    request = Request(client.endpoint,
        data=json.dumps(client.build_payload("Reply briefly.", "Reply OK.")).encode(),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    row = dict(provider="dashscope", requested_model=MODEL, actual_model=None,
        http_status=None, latency_ms=None, input_tokens=None, output_tokens=None,
        total_tokens=None, response_success=False, thinking_requested=False,
        thinking_effective="unknown", timestamp=datetime.now(timezone.utc).isoformat(), error=None)
    started = time.perf_counter()
    try:
        with opener(request, timeout=60) as response:
            row["http_status"] = response.status
            result = json.loads(response.read().decode("utf-8"))
        row["actual_model"] = result.get("model")
        usage = result.get("usage") or {}
        row.update(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                   total_tokens=usage.get("total_tokens"))
        choices = result.get("choices") or [{}]
        row["answer"] = choices[0].get("message", {}).get("content")
        row["finish_reason"] = choices[0].get("finish_reason")
        row["response_success"] = 200 <= response.status < 300 and bool(row["answer"])
    except HTTPError as exc:
        row["http_status"] = exc.code
        # Capture only provider error code/message, never headers or raw bodies.
        try:
            error = json.loads(exc.read().decode("utf-8")).get("error", {})
            row["error"] = {k: error[k] for k in ("code", "type", "message") if k in error}
        except Exception:
            row["error"] = "HTTP " + str(exc.code)
    except Exception as exc:
        row["error"] = type(exc).__name__
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return json.loads(json.dumps(row, ensure_ascii=False).replace(api_key, "[REDACTED]"))


def main():
    key, base = (environment_value(n) for n in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"))
    for name, value in (("DASHSCOPE_API_KEY", key), ("DASHSCOPE_BASE_URL", base)):
        print(name + (": configured" if value else ": missing"))
    if not key or not base:
        return 1
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_qwen_cross_model"
    output = ROOT / "qwen" / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot(SOURCE)
    (output / "deepseek_before_sha256.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    evidence = SOURCE / "frozen_evidence_40.jsonl"
    manifest = SOURCE / "frozen_evidence_manifest.json"
    source_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if evidence_hash != source_manifest["frozen_evidence_sha256"] or len(rows) != 40 or len({r["question_id"] for r in rows}) != 40:
        raise ValueError("frozen evidence integrity failed; no request sent")
    record = health_check(key, base)
    (output / "health_checks.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    after = snapshot(SOURCE)
    passed = record["response_success"] and record["actual_model"] == MODEL
    metadata = dict(experiment_id=experiment_id, provider="dashscope", requested_model=MODEL,
        actual_model=record["actual_model"], health_pass=passed,
        status="health_pass" if passed else "stopped_at_health_gate",
        health_request_count=1, generation_request_count=0, source_experiment=SOURCE_ID,
        question_count=len(rows), frozen_evidence_sha256=evidence_hash,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        frozen_evidence_hash_pass=True, deepseek_unchanged=before == after, output_dir=str(output))
    (output / "deepseek_after_sha256.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    print(json.dumps(metadata, ensure_ascii=False))
    if before != after:
        raise ValueError("DeepSeek preservation check failed")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
