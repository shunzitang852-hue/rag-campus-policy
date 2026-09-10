"""Generation adapter that sends a versioned policy prompt to a real LLM."""

from __future__ import annotations

from dataclasses import dataclass

from .answer_generator import INSUFFICIENT_MESSAGE
from .context_builder import ContextItem
from .llm_provider import LLMProvider
from .prompt_builder import PROMPT_VERSION, PromptBundle, build_policy_prompt


@dataclass(frozen=True)
class LLMGeneratedAnswer:
    answer: str
    provider: str
    model: str
    prompt_version: str
    notes: list[str]
    prompt: PromptBundle | None = None


def generate_llm_answer(
    question: str,
    context: list[ContextItem],
    provider: LLMProvider | None,
) -> LLMGeneratedAnswer:
    if not context:
        return LLMGeneratedAnswer(
            answer=INSUFFICIENT_MESSAGE,
            provider="not_called",
            model="not_called",
            prompt_version=PROMPT_VERSION,
            notes=["未检索到可用政策上下文，因此未调用 LLM。"],
        )
    if provider is None:
        raise ValueError("provider is required when context is available")
    prompt = build_policy_prompt(question, context)
    completion = provider.generate(prompt.system_prompt, prompt.user_prompt)
    return LLMGeneratedAnswer(
        answer=completion.answer,
        provider=completion.provider,
        model=completion.model,
        prompt_version=prompt.prompt_version,
        notes=["回答由真实 LLM 生成；正式引用由程序根据检索 Chunk 元数据生成。"],
        prompt=prompt,
    )
