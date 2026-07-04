"""Configuration loading for chunk generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ChunkConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "chunking.yaml"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace("\\n", "\n")
    try:
        return int(value)
    except ValueError:
        return value


def load_chunk_config_yaml(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the small project-owned YAML subset used by configs/chunking.yaml."""
    result: dict[str, Any] = {}
    strategies: dict[str, dict[str, Any]] = {}
    current_strategy: str | None = None
    in_strategies = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0:
            key, value = line.split(":", 1)
            if key == "strategies":
                in_strategies = True
                continue
            result[key] = _parse_scalar(value)
        elif in_strategies and indent == 2 and line.endswith(":"):
            current_strategy = line[:-1]
            strategies[current_strategy] = {}
        elif in_strategies and indent == 4 and current_strategy:
            key, value = line.split(":", 1)
            strategies[current_strategy][key] = _parse_scalar(value)

    result["strategies"] = strategies
    return result


def get_strategy_config(strategy: str, path: Path = DEFAULT_CONFIG_PATH) -> ChunkConfig:
    raw = load_chunk_config_yaml(path)
    if strategy not in raw["strategies"]:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
    item = raw["strategies"][strategy]
    return ChunkConfig(
        strategy=strategy,
        chunk_size=int(item["chunk_size"]),
        chunk_overlap=int(item["chunk_overlap"]),
        min_chunk_size=int(item["min_chunk_size"]),
        max_chunk_size=int(item["max_chunk_size"]),
        merge_short_chunks=bool(item["merge_short_chunks"]),
        preserve_article_boundary=bool(item["preserve_article_boundary"]),
        sentence_boundary_chars=str(raw["sentence_boundary_chars"]),
        output_format=str(raw["output_format"]),
        version=str(raw["version"]),
    )

