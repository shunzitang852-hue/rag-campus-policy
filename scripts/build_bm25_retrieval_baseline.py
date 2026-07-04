"""Build BM25 index files for all student handbook chunk strategies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.bm25 import build_index
from src.retrieval.dense_index import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
OUTPUT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook" / "bm25_retrieval"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
STRATEGIES = ["fixed_length", "article_based", "structure_aware"]
K1 = 1.5
B = 0.75


def build_all(output_root: Path = OUTPUT_ROOT, k1: float = K1, b: float = B) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    indexes_dir = output_root / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES:
        chunks_path = CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
        chunks = load_jsonl(chunks_path)
        index = build_index(chunks, k1=k1, b=b)
        index_path = indexes_dir / f"{strategy}_bm25_index.json"
        index_path.write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
        summary[strategy] = {
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "document_count": index["document_count"],
            "average_document_length": index["average_document_length"],
            "k1": k1,
            "b": b,
            "tokenizer": index["tokenizer"],
        }
        print(f"{strategy}: documents={index['document_count']} index={index_path}")
    (output_root / "bm25_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--k1", type=float, default=K1)
    parser.add_argument("--b", type=float, default=B)
    args = parser.parse_args()
    build_all(args.output_root, args.k1, args.b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
