"""Atomic file helpers and reproducibility manifests."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text with an atomic replace in the destination directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=False) + "\n"
        for row in rows
    )
    atomic_write_text(Path(path), text)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(str(key))
        fieldnames = seen
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def path_manifest(path: Path) -> dict[str, Any]:
    """Return a file or directory manifest without modifying the input."""

    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "type": "unknown",
        "sha256": "unknown",
        "size_bytes": None,
    }
    if not path.exists():
        return result
    if path.is_file():
        result.update(
            {
                "type": "file",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        return result
    if path.is_dir():
        digest = hashlib.sha256()
        total_size = 0
        file_count = 0
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix()
            child_hash = sha256_file(child)
            size = child.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(child_hash.encode("ascii"))
            digest.update(b"\0")
            total_size += size
            file_count += 1
        result.update(
            {
                "type": "directory",
                "sha256": digest.hexdigest().upper(),
                "size_bytes": total_size,
                "file_count": file_count,
            }
        )
    return result


def build_data_manifest(inputs: Mapping[str, Path | str | None]) -> dict[str, Any]:
    return {
        "algorithm": "SHA-256",
        "inputs": {
            role: (
                path_manifest(Path(value))
                if value not in (None, "")
                else {
                    "path": "unknown",
                    "exists": False,
                    "type": "unknown",
                    "sha256": "unknown",
                    "size_bytes": None,
                }
            )
            for role, value in inputs.items()
        },
    }


def collect_environment() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    try:
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            if name:
                dependencies[str(name)] = distribution.version
    except Exception as exc:  # pragma: no cover - defensive environment fallback
        dependencies = {"collection_error": f"{type(exc).__name__}: {exc}"}
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_name": os.name,
        "dependencies": dict(sorted(dependencies.items(), key=lambda item: item[0].lower())),
    }


def collect_git_state(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)

    def run(*args: str) -> tuple[int, str]:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return completed.returncode, completed.stdout.strip()

    commit_code, commit = run("rev-parse", "HEAD")
    status_code, status = run("status", "--porcelain=v1")
    return {
        "git_commit": commit if commit_code == 0 and commit else "unknown",
        "git_worktree_clean": status == "" if status_code == 0 else "unknown",
        "git_status_porcelain": status.splitlines() if status_code == 0 else [],
    }
