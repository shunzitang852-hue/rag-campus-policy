"""Build dense embeddings with a configurable real embedding provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.dense_index import load_jsonl, write_jsonl
from src.retrieval.embedding_providers import ProviderUnavailable, create_provider, encode_batches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"
OUTPUT_ROOT = PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook" / "dense_retrieval_real_embedding"
CHUNK_ROOT = DATASETS_ROOT / "processed" / "hdu_ies" / "student_handbook" / "chunks"
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval_dense_real_embedding.yaml"
STRATEGIES = ["fixed_length", "article_based", "structure_aware"]


def load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            config.setdefault(current_key, []).append(stripped[2:].strip())
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if value == "":
                config[key] = []
            else:
                config[key] = value
    return config


def _chunk_to_embedding_row(chunk: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "strategy": chunk.get("strategy") or chunk.get("chunk_strategy"),
        "document_id": chunk["document_id"],
        "document_title": chunk["document_title"],
        "chunk_text": chunk["chunk_text"],
        "embedding": embedding,
        "metadata": chunk.get("metadata", {}),
    }


def write_index(index_path: Path, strategy: str, embeddings_path: Path, chunk_count: int, provider_metadata: dict[str, Any], dimension: int) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": strategy,
        "index_type": "in_memory_jsonl_cosine",
        "embedding_provider": provider_metadata,
        "embedding_dimension": dimension,
        "chunk_count": chunk_count,
        "embeddings_path": str(embeddings_path),
        "notes": "Real embedding experiment index metadata; vectors are stored in JSONL and searched in memory.",
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_unavailable_manifest(output_root: Path, config: dict[str, Any], reason: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "real_embedding_unavailable_manifest.json"
    payload = {
        "stage": "dense_real_embedding_build",
        "model_ran": False,
        "embedding_provider": config.get("embedding_provider"),
        "model_name": config.get("model_name"),
        "model_cache_dir": config.get("model_cache_dir"),
        "reason": reason,
        "notes": "No embeddings or indexes were generated because the configured real embedding provider is unavailable.",
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_all(output_root: Path = OUTPUT_ROOT, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(config_path)
    batch_size = int(config.get("batch_size", 32))
    try:
        provider = create_provider(config)
    except ProviderUnavailable as exc:
        manifest_path = write_unavailable_manifest(output_root, config, str(exc))
        return {
            "model_ran": False,
            "reason": str(exc),
            "manifest_path": str(manifest_path),
            "provider": config.get("embedding_provider"),
            "model_name": config.get("model_name"),
        }

    summary: dict[str, Any] = {
        "model_ran": True,
        "provider": provider.metadata(),
        "strategies": {},
    }
    embeddings_dir = output_root / "embeddings"
    indexes_dir = output_root / "indexes"
    for strategy in STRATEGIES:
        chunks_path = CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
        chunks = load_jsonl(chunks_path)
        texts = [chunk["chunk_text"] for chunk in chunks]
        vectors, dimension = encode_batches(provider, texts, batch_size=batch_size)
        rows = [
            _chunk_to_embedding_row(chunk, vector)
            for chunk, vector in zip(chunks, vectors)
        ]
        embeddings_path = embeddings_dir / f"{strategy}_real_embedding_embeddings.jsonl"
        index_path = indexes_dir / f"{strategy}_real_embedding_index.json"
        write_jsonl(embeddings_path, rows)
        write_index(index_path, strategy, embeddings_path, len(rows), provider.metadata(), dimension)
        summary["strategies"][strategy] = {
            "chunks_path": str(chunks_path),
            "embeddings_path": str(embeddings_path),
            "index_path": str(index_path),
            "chunk_count": len(rows),
            "embedding_dimension": dimension,
        }
        print(f"{strategy}: embeddings={len(rows)} dimension={dimension} path={embeddings_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "real_embedding_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    build_all(args.output_root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
