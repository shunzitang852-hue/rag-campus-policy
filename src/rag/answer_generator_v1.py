"""Frozen compatibility copy of the historical extractive RAG v1 generator.

The selection logic is intentionally kept equivalent to ``answer_generator.py``
at commit 5acb30b.  It exists separately so later v2 tuning cannot silently
change the v1 baseline.
"""

from __future__ import annotations

import re

from .answer_generator import DISCLAIMER, INSUFFICIENT_MESSAGE, GeneratedAnswer
from .citation import Citation
from .context_builder import ContextItem, compact_text


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        if len(token) > 1:
            terms.add(token.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", query)
    terms.update(chinese)
    for size in (2, 3, 4):
        for start in range(0, max(0, len(chinese) - size + 1)):
            terms.add("".join(chinese[start : start + size]))
    return terms


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；;])", compact_text(text))
    return [part.strip() for part in parts if part.strip()]


def _select_answer_sentences(query: str, context: list[ContextItem], limit: int) -> list[str]:
    terms = _query_terms(query)
    candidates: list[tuple[int, int, int, str]] = []
    for item_index, item in enumerate(context):
        for sentence in _sentences(item.text):
            lowered = sentence.lower()
            hits = sum(1 for term in terms if term and term in lowered)
            candidates.append((-hits, item_index, min(len(sentence), 9999), sentence))
    candidates.sort()
    selected: list[str] = []
    used: set[str] = set()
    char_count = 0
    for neg_hits, _item_index, _length, sentence in candidates:
        if sentence in used:
            continue
        if selected and neg_hits == 0:
            break
        next_len = len(sentence) + (1 if selected else 0)
        if char_count + next_len > limit:
            continue
        used.add(sentence)
        selected.append(sentence)
        char_count += next_len
        if len(selected) >= 3:
            break
    if not selected and context:
        selected.append(context[0].excerpt)
    return selected


def generate_answer_v1(
    query: str,
    context: list[ContextItem],
    citations: list[Citation],
    *,
    answer_mode: str = "extractive",
    max_answer_chars: int = 500,
    insufficient_score_threshold: float = 0.0,
) -> GeneratedAnswer:
    if answer_mode != "extractive":
        raise ValueError(f"unsupported answer_mode: {answer_mode}")
    notes = [DISCLAIMER, "答案只使用检索返回的 Chunk 内容，不使用 QA 测试集标签或人工证据字段。"]
    evidence = [citation.to_dict() for citation in citations]
    sources = sorted({citation.document_title for citation in citations if citation.document_title})
    if not context or not citations:
        return GeneratedAnswer(INSUFFICIENT_MESSAGE, [], [], notes, answer_mode)
    if context[0].score <= insufficient_score_threshold:
        return GeneratedAnswer(INSUFFICIENT_MESSAGE, evidence, sources, notes, answer_mode)
    selected = _select_answer_sentences(query, context, max_answer_chars)
    answer = "根据当前检索到的制度片段，" + "".join(selected)
    if len(answer) > max_answer_chars:
        answer = answer[: max(0, max_answer_chars - 3)].rstrip() + "..."
    return GeneratedAnswer(answer, evidence, sources, notes, answer_mode)
