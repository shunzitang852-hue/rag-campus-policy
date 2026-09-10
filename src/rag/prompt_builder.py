"""Versioned prompts for grounded university-policy question answering."""

from __future__ import annotations

from dataclasses import dataclass

from .context_builder import ContextItem


PROMPT_VERSION = "policy_grounded_qa_v1"
SYSTEM_INSTRUCTION = """你是高校政策问答助手。请严格遵守以下要求：
1. 只能依据用户消息中提供的政策上下文作答，不得使用上下文之外的政策知识。
2. 如果上下文不足以确认答案，明确回答“无法从给定政策材料确认”，不要猜测或补充。
3. 回答应简洁、准确，并完整保留必要条件、限制、时间、金额、比例和程序步骤。
4. 不得修改政策事实，不得把相似政策的规定混为一谈。
5. 不要自行编造政策名称、条款号或引用。正式引用由程序根据检索结果生成。
"""


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    prompt_version: str = PROMPT_VERSION

    def debug_text(self) -> str:
        return f"[SYSTEM]\n{self.system_prompt}\n\n[USER]\n{self.user_prompt}"


def build_policy_prompt(question: str, context: list[ContextItem]) -> PromptBundle:
    """Build a traceable prompt from the question and retrieved context."""

    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    blocks: list[str] = []
    for index, item in enumerate(context, start=1):
        blocks.append(
            "\n".join(
                [
                    f"--- CHUNK {index} ---",
                    f"chunk_id: {item.chunk_id}",
                    f"policy_title: {item.document_title}",
                    f"retrieval_score: {item.score:.6f}",
                    "content:",
                    item.text,
                ]
            )
        )
    rendered_context = "\n\n".join(blocks) if blocks else "（无可用政策片段）"
    user_prompt = (
        "【政策上下文】\n"
        f"{rendered_context}\n\n"
        "【用户问题】\n"
        f"{question}\n\n"
        "请直接给出基于上述政策材料的回答。"
    )
    return PromptBundle(system_prompt=SYSTEM_INSTRUCTION, user_prompt=user_prompt)
