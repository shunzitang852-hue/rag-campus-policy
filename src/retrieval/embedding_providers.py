"""Embedding provider interface for real-model dense retrieval experiments."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.retrieval.embedding import DEFAULT_DIMENSION, embed_text


class ProviderUnavailable(RuntimeError):
    """Raised when a configured embedding provider cannot run locally."""


class EmbeddingProvider(Protocol):
    name: str
    is_real_semantic_embedding: bool

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into same-dimension vectors."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable provider metadata."""


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@dataclass
class DeterministicHashProvider:
    dimension: int = DEFAULT_DIMENSION
    name: str = "deterministic_hash"
    is_real_semantic_embedding: bool = False

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(text, self.dimension) for text in texts]

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "embedding_dimension": self.dimension,
            "is_real_semantic_embedding": self.is_real_semantic_embedding,
            "notes": "Deterministic hash fallback for interface validation only.",
        }


@dataclass
class LocalModelPlaceholderProvider:
    model_path: Path
    name: str = "local_model_placeholder"
    is_real_semantic_embedding: bool = False

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise ProviderUnavailable(
            f"local_model_placeholder does not implement inference; expected model path: {self.model_path}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model_path": str(self.model_path),
            "is_real_semantic_embedding": self.is_real_semantic_embedding,
            "notes": "Placeholder used to keep the real-model interface configurable.",
        }


class SentenceTransformersProvider:
    name = "sentence_transformers"
    is_real_semantic_embedding = True

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        device: str | None = None,
        *,
        local_files_only: bool = False,
    ) -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise ProviderUnavailable("sentence-transformers is not installed in the active Python environment")
        from sentence_transformers import SentenceTransformer  # type: ignore

        kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_folder"] = str(cache_dir)
        if device:
            kwargs["device"] = device
        kwargs["local_files_only"] = local_files_only
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.local_files_only = local_files_only
        self.model = SentenceTransformer(model_name, **kwargs)
        dimension_method = getattr(
            self.model,
            "get_embedding_dimension",
            getattr(self.model, "get_sentence_embedding_dimension", lambda: None),
        )
        dimension = dimension_method()
        self.dimension = int(dimension) if dimension else None

    def encode(self, texts: list[str]) -> list[list[float]]:
        raw_vectors = self.model.encode(
            texts,
            batch_size=min(32, max(1, len(texts))),
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        vectors: list[list[float]] = []
        for vector in raw_vectors:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            vectors.append([float(value) for value in vector])
        return vectors

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "device": self.device,
            "local_files_only": self.local_files_only,
            "embedding_dimension": self.dimension,
            "is_real_semantic_embedding": self.is_real_semantic_embedding,
        }


def create_provider(config: dict[str, Any]) -> EmbeddingProvider:
    provider_name = str(config.get("embedding_provider", "deterministic_hash"))
    if provider_name == "deterministic_hash":
        dimension = config.get("embedding_dimension")
        if dimension in (None, "auto"):
            dimension = DEFAULT_DIMENSION
        return DeterministicHashProvider(dimension=int(dimension))
    if provider_name == "sentence_transformers":
        local_files_only = config.get("local_files_only", False)
        if isinstance(local_files_only, str):
            local_files_only = local_files_only.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        return SentenceTransformersProvider(
            model_name=str(config.get("model_name", "BAAI/bge-small-zh-v1.5")),
            cache_dir=Path(str(config["model_cache_dir"])) if config.get("model_cache_dir") else None,
            device=str(config["device"]) if config.get("device") else None,
            local_files_only=bool(local_files_only),
        )
    if provider_name == "local_model_placeholder":
        return LocalModelPlaceholderProvider(Path(str(config.get("local_model_path", ""))))
    raise ProviderUnavailable(f"unsupported embedding provider: {provider_name}")


def validate_same_dimension(vectors: list[list[float]]) -> int:
    if not vectors:
        raise ValueError("no vectors returned by provider")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent embedding dimensions: {sorted(dimensions)}")
    return dimensions.pop()


def encode_batches(provider: EmbeddingProvider, texts: list[str], batch_size: int) -> tuple[list[list[float]], int]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(provider.encode(batch))
    dimension = validate_same_dimension(vectors)
    return [_normalize(vector) for vector in vectors], dimension
