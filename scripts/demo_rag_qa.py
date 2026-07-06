"""Command-line extractive RAG QA demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.pipeline import answer_query, load_rag_config
from src.retrieval.demo import METHODS, STRATEGIES


def print_response(response) -> None:
    print(f"问题：{response.question}")
    print()
    print("生成答案：")
    print(response.answer)
    print()
    print("检索依据：")
    if not response.citations:
        print("- 当前没有可引用的检索片段。")
    for citation in response.citations:
        print(f"[{citation['citation_id']}] {citation['document_title']}")
        print(f"chunk_id: {citation['chunk_id']}")
        print(f"score: {float(citation['score']):.6f}")
        print(f"excerpt: {citation['excerpt']}")
    print()
    print("来源制度：")
    if response.source_documents:
        for title in response.source_documents:
            print(f"- {title}")
    else:
        print("- 无")
    print()
    print("检索信息：")
    for key, value in response.retrieval_info.items():
        print(f"{key}: {value}")
    print()
    print("注意事项：")
    for note in response.notes:
        print(f"- {note}")


def run_once(query: str, method: str, strategy: str, top_k: int, answer_mode: str) -> int:
    response = answer_query(query, method=method, strategy=strategy, top_k=top_k, answer_mode=answer_mode)
    print_response(response)
    return 0


def interactive_loop(method: str, strategy: str, top_k: int, answer_mode: str) -> int:
    print("RAG QA Demo: 输入问题后回车；输入 exit 或 quit 退出。")
    while True:
        try:
            query = input("问题> ").strip()
        except EOFError:
            return 0
        if query.lower() in {"exit", "quit"}:
            return 0
        if not query:
            print("query must not be empty")
            continue
        run_once(query, method, strategy, top_k, answer_mode)
    return 0


def main(argv: list[str] | None = None) -> int:
    config = load_rag_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="", help="Chinese natural-language question.")
    parser.add_argument("--method", choices=METHODS, default=str(config.get("default_retrieval_method", "real_hybrid_weighted")))
    parser.add_argument("--strategy", choices=STRATEGIES, default=str(config.get("default_strategy", "fixed_length")))
    parser.add_argument("--top-k", type=int, default=int(config.get("default_top_k", 5)))
    parser.add_argument("--answer-mode", choices=["extractive"], default=str(config.get("answer_mode", "extractive")))
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.query.strip():
        return interactive_loop(args.method, args.strategy, args.top_k, args.answer_mode)
    return run_once(args.query.strip(), args.method, args.strategy, args.top_k, args.answer_mode)


if __name__ == "__main__":
    raise SystemExit(main())
