"""Extractive RAG pipeline for the student handbook retrieval demo."""

from .answer_generator import GeneratedAnswer, generate_answer
from .citation import Citation, build_citations
from .context_builder import ContextItem, build_context
from .pipeline import RAGResponse, answer_query, load_rag_config

__all__ = [
    "Citation",
    "ContextItem",
    "GeneratedAnswer",
    "RAGResponse",
    "answer_query",
    "build_citations",
    "build_context",
    "generate_answer",
    "load_rag_config",
]
