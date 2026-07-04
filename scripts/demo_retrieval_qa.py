"""Command-line retrieval-only QA demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.demo import DEFAULT_METHOD, DEFAULT_STRATEGY, DEFAULT_TOP_K, METHODS, STRATEGIES, retrieve
from src.retrieval.embedding_providers import ProviderUnavailable


DISCLAIMER = "说明：当前为检索结果展示，不是大模型生成答案。"


def print_results(query: str, method: str, strategy: str, top_k: int) -> None:
    results = retrieve(query, method=method, strategy=strategy, top_k=top_k)
    print(f"问题：{query}")
    print(f"检索方法：{method}")
    print(f"Chunk 策略：{strategy}")
    print(f"Top K：{top_k}")
    print(DISCLAIMER)
    print()
    for result in results:
        print(f"[{result.rank}] {result.document_title}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"score: {result.score:.6f}")
        print(f"strategy: {result.strategy}")
        print(f"retrieval_method: {result.retrieval_method}")
        print(f"摘要：{result.chunk_text_summary}")
        print()


def interactive_loop(method: str, strategy: str, top_k: int) -> int:
    print(DISCLAIMER)
    print("输入问题后回车开始检索；输入 exit 或 quit 退出。")
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
        try:
            print_results(query, method, strategy, top_k)
        except ProviderUnavailable as exc:
            print(f"real embedding provider unavailable: {exc}")
            return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="", help="Chinese natural-language question.")
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.query.strip():
        return interactive_loop(args.method, args.strategy, args.top_k)
    try:
        print_results(args.query.strip(), args.method, args.strategy, args.top_k)
    except ProviderUnavailable as exc:
        print(f"real embedding provider unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
