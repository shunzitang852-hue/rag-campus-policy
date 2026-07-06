"""Template-based answer generation from retrieved context only."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.build_real_embedding_dense_retrieval import load_simple_yaml

from .citation import Citation
from .context_builder import ContextItem, compact_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_answer_generation.yaml"

INSUFFICIENT_MESSAGE = "当前检索片段不足以确认答案。"
DISCLAIMER = "当前回答由检索到的制度片段组织生成，不是学校官方最终解释。"

QUESTION_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "condition": ("应当", "需要", "可以", "不得", "符合", "申请", "条件", "情形"),
    "procedure": ("申请", "提交", "审核", "审批", "批准", "办理", "公示", "程序"),
    "responsible_body": ("学院", "学工部", "教务处", "学生处", "辅导员", "班主任", "相关部门"),
    "time_limit": ("日", "天", "周", "月", "学期", "期限", "期满", "按时", "学年"),
    "qualification": ("条件", "资格", "评定", "成绩", "违纪", "处分", "排名", "要求"),
    "comparison": ("区别", "分别", "一是", "二是", "不同", "相比", "比较"),
    "fact_lookup": (),
}

QUESTION_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "condition": ("条件", "情形", "什么情况", "何种", "需要"),
    "procedure": ("程序", "流程", "步骤", "如何", "怎么", "办理", "申请"),
    "responsible_body": ("谁", "主体", "责任", "负责", "部门", "机构"),
    "time_limit": ("时间", "期限", "何时", "多久", "多少天", "学期", "学年"),
    "qualification": ("资格", "要求", "评定", "排名", "成绩", "限制"),
    "comparison": ("区别", "不同", "比较", "相比", "分别"),
}


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    evidence: list[dict[str, object]]
    source_documents: list[str]
    notes: list[str]
    answer_mode: str = "extractive"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_answer_generation_config(path: Path = DEFAULT_GENERATION_CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(path) if path.exists() else {}
    int_keys = {
        "max_answer_chars",
        "max_sentences",
        "max_sentence_chars",
        "min_sentence_score",
        "same_document_bonus",
        "article_anchor_bonus",
        "type_keyword_bonus",
        "question_term_bonus",
        "fallback_excerpt_chars",
    }
    for key in int_keys:
        if key in config:
            config[key] = int(config[key])
    return config


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


def _article_anchors(query: str) -> set[str]:
    return set(re.findall(r"第[一二三四五六七八九十百0-9]+条", query or ""))


def _infer_question_type(query: str) -> str:
    for question_type, hints in QUESTION_TYPE_HINTS.items():
        if any(hint and hint in query for hint in hints):
            return question_type
    return "fact_lookup"


def _type_terms(query: str) -> tuple[str, ...]:
    return QUESTION_TYPE_TERMS.get(_infer_question_type(query), ())


def _truncate_sentence(sentence: str, max_chars: int) -> str:
    if max_chars < 4 or len(sentence) <= max_chars:
        return sentence
    return sentence[: max(0, max_chars - 3)].rstrip() + "..."


def _sentence_score(
    sentence: str,
    terms: set[str],
    type_terms: tuple[str, ...],
    anchors: set[str],
    item: ContextItem,
    primary_document_title: str,
    config: dict[str, Any],
) -> tuple[int, int]:
    lowered = sentence.lower()
    question_hits = sum(1 for term in terms if term and term in lowered)
    type_hits = sum(1 for term in type_terms if term and term in sentence)
    anchor_hits = sum(1 for anchor in anchors if anchor and anchor in sentence)
    score = question_hits * int(config.get("question_term_bonus", 1))
    score += type_hits * int(config.get("type_keyword_bonus", 2))
    score += anchor_hits * int(config.get("article_anchor_bonus", 6))
    if primary_document_title and item.document_title == primary_document_title:
        score += int(config.get("same_document_bonus", 2))
    return score, min(len(sentence), 9999)


def _select_answer_sentences(query: str, context: list[ContextItem], limit: int, config: dict[str, Any]) -> list[str]:
    terms = _query_terms(query)
    type_terms = _type_terms(query)
    anchors = _article_anchors(query)
    primary_document_title = context[0].document_title if context else ""
    candidates: list[tuple[int, int, int, int, str]] = []
    for item_index, item in enumerate(context):
        for sentence_index, sentence in enumerate(_sentences(item.text)):
            score, length = _sentence_score(sentence, terms, type_terms, anchors, item, primary_document_title, config)
            candidates.append((-score, item_index, sentence_index, length, sentence))
    candidates.sort()

    selected: list[str] = []
    used: set[str] = set()
    char_count = 0
    max_sentences = int(config.get("max_sentences", 4))
    max_sentence_chars = int(config.get("max_sentence_chars", 180))
    min_sentence_score = int(config.get("min_sentence_score", 1))
    for neg_score, _item_index, _sentence_index, _length, raw_sentence in candidates:
        if raw_sentence in used:
            continue
        if -neg_score < min_sentence_score and selected:
            break
        sentence = _truncate_sentence(raw_sentence, max_sentence_chars)
        next_len = len(sentence) + (1 if selected else 0)
        if char_count + next_len > limit:
            continue
        used.add(raw_sentence)
        selected.append(sentence)
        char_count += next_len
        if len(selected) >= max_sentences:
            break
    if not selected and context:
        selected.append(context[0].excerpt[: int(config.get("fallback_excerpt_chars", 260))])
    return selected


def generate_answer(
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
    notes = [DISCLAIMER, "答案只使用检索返回的 Chunk 内容，不使用 QA 测试集标准答案或人工证据字段。"]
    if not context or not citations:
        return GeneratedAnswer(
            answer=INSUFFICIENT_MESSAGE,
            evidence=[],
            source_documents=[],
            notes=notes,
            answer_mode=answer_mode,
        )
    if context[0].score <= insufficient_score_threshold:
        return GeneratedAnswer(
            answer=INSUFFICIENT_MESSAGE,
            evidence=[citation.to_dict() for citation in citations],
            source_documents=sorted({citation.document_title for citation in citations if citation.document_title}),
            notes=notes,
            answer_mode=answer_mode,
        )

    generation_config = load_answer_generation_config()
    effective_max_answer_chars = min(
        int(max_answer_chars),
        int(generation_config.get("max_answer_chars", max_answer_chars)),
    )
    prefix = "根据当前检索到的制度片段，"
    selected = _select_answer_sentences(
        query,
        context,
        max(0, effective_max_answer_chars - len(prefix)),
        generation_config,
    )
    answer = prefix + "".join(selected)
    if len(answer) > effective_max_answer_chars:
        answer = answer[: max(0, effective_max_answer_chars - 3)].rstrip() + "..."
    return GeneratedAnswer(
        answer=answer,
        evidence=[citation.to_dict() for citation in citations],
        source_documents=sorted({citation.document_title for citation in citations if citation.document_title}),
        notes=notes,
        answer_mode=answer_mode,
    )
