"""Build deterministic embedding files and lightweight dense indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.dense_index import build_embedding_rows, load_jsonl, write_index, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
OUTPUT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook" / "dense_retrieval"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
STRATEGIES = ["fixed_length", "article_based", "structure_aware"]
DIMENSION = 384


def build_all(output_root: Path = OUTPUT_ROOT, dimension: int = DIMENSION) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    embeddings_dir = output_root / "embeddings"
    indexes_dir = output_root / "indexes"
    for strategy in STRATEGIES:
        chunks_path = CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
        chunks = load_jsonl(chunks_path)
        embedding_rows = build_embedding_rows(chunks, dimension)
        embeddings_path = embeddings_dir / f"{strategy}_embeddings.jsonl"
        index_path = indexes_dir / f"{strategy}_index.json"
        write_jsonl(embeddings_path, embedding_rows)
        write_index(index_path, strategy, embeddings_path, len(embedding_rows), dimension)
        summary[strategy] = {
            "chunks_path": str(chunks_path),
            "embeddings_path": str(embeddings_path),
            "index_path": str(index_path),
            "chunk_count": len(embedding_rows),
            "embedding_dimension": dimension,
        }
        print(f"{strategy}: embeddings={len(embedding_rows)} path={embeddings_path}")
    (output_root / "dense_embedding_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    args = parser.parse_args()
    build_all(args.output_root, args.dimension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
