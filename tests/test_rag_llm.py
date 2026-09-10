from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.answer_generator import INSUFFICIENT_MESSAGE
from src.rag.context_builder import ContextItem
from src.rag.llm_generator import generate_llm_answer
from src.rag.llm_provider import (
    LLMCompletion,
    LLMProviderError,
    LLMProviderUnavailable,
    OpenAICompatibleProvider,
    create_llm_provider,
)
from src.rag.pipeline import answer_query
from src.rag.prompt_builder import PROMPT_VERSION, build_policy_prompt
from src.retrieval.demo import DemoResult


QUESTION = "学生申请解除处分需要满足什么条件？"
SECRET = "test-secret-must-not-appear"


def _context() -> list[ContextItem]:
    return [
        ContextItem(
            rank=1,
            chunk_id="chunk-real-1",
            document_title="学生解除处分管理办法",
            score=0.98,
            text="处分期满并符合规定条件的学生，可以申请解除处分。",
            excerpt="处分期满并符合规定条件的学生，可以申请解除处分。",
            strategy="fixed_length",
            retrieval_method="bm25",
        )
    ]


def _retriever(query: str, method: str, strategy: str, top_k: int) -> list[DemoResult]:
    assert query == QUESTION
    return [
        DemoResult(
            rank=1,
            chunk_id="chunk-real-1",
            document_title="学生解除处分管理办法",
            score=0.98,
            chunk_text="处分期满并符合规定条件的学生，可以申请解除处分。",
            chunk_text_summary="处分期满并符合规定条件的学生，可以申请解除处分。",
            strategy=strategy,
            retrieval_method=method,
        )
    ][:top_k]


class FakeProvider:
    provider_name = "fake_for_unit_test"
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        self.calls.append((system_prompt, user_prompt))
        return LLMCompletion(
            answer="模型回答中提到虚构政策，但程序引用不得采用它。",
            provider=self.provider_name,
            model=self.model_name,
        )


class FailingProvider(FakeProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMCompletion:
        raise LLMProviderError("provider failed safely")


def test_prompt_contains_question_context_and_policy_metadata() -> None:
    prompt = build_policy_prompt(QUESTION, _context())
    assert prompt.prompt_version == PROMPT_VERSION
    assert QUESTION in prompt.user_prompt
    assert "chunk-real-1" in prompt.user_prompt
    assert "学生解除处分管理办法" in prompt.user_prompt
    assert "只能依据" in prompt.system_prompt


def test_prompt_marks_empty_context_explicitly() -> None:
    prompt = build_policy_prompt(QUESTION, [])
    assert "无可用政策片段" in prompt.user_prompt
    assert QUESTION in prompt.user_prompt


def test_llm_generator_calls_provider_with_versioned_prompt() -> None:
    provider = FakeProvider()
    result = generate_llm_answer(QUESTION, _context(), provider)
    assert provider.calls
    assert result.answer
    assert result.provider == "fake_for_unit_test"
    assert result.model == "fake-model"
    assert result.prompt_version == PROMPT_VERSION


def test_llm_generator_does_not_call_provider_without_context() -> None:
    provider = FakeProvider()
    result = generate_llm_answer(QUESTION, [], provider)
    assert result.answer == INSUFFICIENT_MESSAGE
    assert provider.calls == []


def test_llm_provider_error_is_propagated() -> None:
    with pytest.raises(LLMProviderError, match="failed safely"):
        generate_llm_answer(QUESTION, _context(), FailingProvider())


def test_missing_provider_configuration_does_not_leak_secret() -> None:
    config = {
        "provider": "openai_compatible",
        "base_url_env": "LLM_BASE_URL",
        "model_env": "LLM_MODEL",
        "api_key_env": "DEEPSEEK_API_KEY",
    }
    with pytest.raises(LLMProviderUnavailable) as error:
        create_llm_provider(config, environ={"DEEPSEEK_API_KEY": SECRET})
    assert SECRET not in str(error.value)
    assert "LLM_BASE_URL" in str(error.value)
    assert "LLM_MODEL" in str(error.value)


def test_deepseek_endpoint_and_non_thinking_payload_are_exact() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com",
        api_key=SECRET,
        model_name="deepseek-v4-flash",
        temperature=0.0,
        max_tokens=512,
        request_options={"thinking": {"type": "disabled"}},
    )
    payload = provider.build_payload("system", "user")
    assert provider.endpoint == "https://api.deepseek.com/chat/completions"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 512
    assert SECRET not in str(payload)


def test_full_chat_completion_endpoint_is_not_duplicated() -> None:
    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com/chat/completions",
        api_key=SECRET,
        model_name="deepseek-v4-flash",
    )
    assert provider.endpoint == "https://api.deepseek.com/chat/completions"


def test_pipeline_llm_mode_uses_prompt_provider_and_retrieval_citations() -> None:
    provider = FakeProvider()
    response = answer_query(
        QUESTION,
        method="bm25",
        strategy="fixed_length",
        top_k=1,
        generator="llm",
        retriever=_retriever,
        llm_provider=provider,
        include_prompt=True,
    )
    assert response.answer_mode == "llm"
    assert response.generation_info["llm_provider"] == "fake_for_unit_test"
    assert response.generation_info["llm_model"] == "fake-model"
    assert response.generation_info["prompt_version"] == PROMPT_VERSION
    assert QUESTION in str(response.generation_info["prompt"])
    assert response.citations[0]["document_title"] == "学生解除处分管理办法"
    assert response.citations[0]["chunk_id"] == "chunk-real-1"
    assert "虚构政策" not in response.source_documents


def test_pipeline_extractive_mode_remains_default() -> None:
    response = answer_query(
        QUESTION,
        method="bm25",
        strategy="fixed_length",
        top_k=1,
        retriever=_retriever,
    )
    assert response.answer_mode == "extractive"
    assert response.generation_info["generator"] == "extractive"
    assert response.generation_info["llm_provider"] == "not_used"
