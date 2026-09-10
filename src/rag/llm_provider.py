"""Minimal provider interface for real OpenAI-compatible chat generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.build_real_embedding_dense_retrieval import load_simple_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_llm.yaml"


class LLMProviderError(RuntimeError):
    """Base error for LLM provider configuration or generation failures."""


class LLMProviderUnavailable(LLMProviderError):
    """Raised when the configured real LLM cannot be called."""


@dataclass(frozen=True)
class LLMCompletion:
    answer: str
    provider: str
    model: str


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        """Generate one real completion."""


def _config_text(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key, "")
    if isinstance(value, list):
        return ""
    return str(value or "").strip()


def load_llm_config(path: Path = DEFAULT_LLM_CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(path)
    for key in ("timeout_seconds", "max_tokens"):
        if key in config and not isinstance(config[key], list):
            config[key] = int(config[key])
    if "temperature" in config and not isinstance(config["temperature"], list):
        config["temperature"] = float(config["temperature"])
    return config


class OpenAICompatibleProvider:
    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 512,
        request_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not base_url:
            raise LLMProviderUnavailable("LLM base URL is not configured")
        if not api_key:
            raise LLMProviderUnavailable("LLM API key is not configured")
        if not model_name:
            raise LLMProviderUnavailable("LLM model is not configured")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_options = dict(request_options or {})

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def build_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        reserved = {"model", "messages", "temperature", "max_tokens"}
        conflicting = reserved.intersection(self.request_options)
        if conflicting:
            raise LLMProviderError(
                "request_options cannot override core fields: " + ", ".join(sorted(conflicting))
            )
        payload.update(self.request_options)
        return payload

    def generate(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        payload = self.build_payload(system_prompt, user_prompt)
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured HTTPS endpoint
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMProviderError(f"LLM request failed with HTTP status {exc.code}") from exc
        except URLError as exc:
            raise LLMProviderError("LLM endpoint is unreachable") from exc
        except (OSError, TimeoutError) as exc:
            raise LLMProviderError("LLM request could not be completed") from exc
        try:
            answer = str(result["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM response does not contain a chat completion") from exc
        if not answer:
            raise LLMProviderError("LLM returned an empty answer")
        return LLMCompletion(answer=answer, provider=self.provider_name, model=self.model_name)


def create_llm_provider(
    config: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LLMProvider:
    config = dict(config or load_llm_config())
    environ = environ or os.environ
    provider_name = _config_text(config, "provider") or "openai_compatible"
    if provider_name != "openai_compatible":
        raise LLMProviderUnavailable(f"unsupported LLM provider: {provider_name}")

    base_url_env = _config_text(config, "base_url_env") or "LLM_BASE_URL"
    model_env = _config_text(config, "model_env") or "LLM_MODEL"
    api_key_env = _config_text(config, "api_key_env") or "LLM_API_KEY"
    base_url = str(environ.get(base_url_env, "") or _config_text(config, "base_url"))
    model_name = str(environ.get(model_env, "") or _config_text(config, "model"))
    api_key = str(environ.get(api_key_env, ""))
    request_options_text = _config_text(config, "request_options_json")
    try:
        request_options = json.loads(request_options_text) if request_options_text else {}
    except json.JSONDecodeError as exc:
        raise LLMProviderUnavailable("request_options_json is not valid JSON") from exc
    if not isinstance(request_options, dict):
        raise LLMProviderUnavailable("request_options_json must contain a JSON object")

    missing = []
    if not base_url:
        missing.append(base_url_env)
    if not model_name:
        missing.append(model_env)
    if not api_key:
        missing.append(api_key_env)
    if missing:
        raise LLMProviderUnavailable("missing LLM configuration: " + ", ".join(missing))

    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        timeout_seconds=int(config.get("timeout_seconds", 60)),
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 512)),
        request_options=request_options,
    )
