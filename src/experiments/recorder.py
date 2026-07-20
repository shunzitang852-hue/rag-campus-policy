"""Create immutable, self-describing experiment run directories."""

from __future__ import annotations

import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .manifests import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    build_data_manifest,
    collect_environment,
    collect_git_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT.parent / "experiment_outputs" / "hdu_ies" / "student_handbook"
)
REQUIRED_INPUT_ROLES = ("data_source", "chunk_file", "qa_file")


def _slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value if value not in (None, "") else fallback).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def create_experiment_id(
    task_type: str,
    retrieval_method: str,
    chunk_strategy: str,
    top_k: int | str | None,
    *,
    timestamp: datetime | None = None,
    nonce: str | None = None,
) -> str:
    """Build an informative ID with microseconds and a collision-resistant suffix."""

    current = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = current.strftime("%Y%m%dT%H%M%S%fZ")
    top_k_slug = f"topk{_slug(top_k)}"
    suffix = _slug(nonce or uuid.uuid4().hex[:8])
    return "_".join(
        [
            stamp,
            _slug(task_type),
            _slug(retrieval_method),
            _slug(chunk_strategy),
            top_k_slug,
            suffix,
        ]
    )


class ExperimentRecorder:
    """Context manager for the normalized ``runs/<experiment_id>`` contract."""

    def __init__(
        self,
        *,
        task_type: str,
        retrieval_method: str,
        chunk_strategy: str,
        top_k: int | None,
        embedding_model: str = "unknown",
        qa_version: str = "unknown",
        hybrid_parameters: Mapping[str, Any] | None = None,
        random_seed: int | None = None,
        inputs: Mapping[str, Path | str | None] | None = None,
        parameters: Mapping[str, Any] | None = None,
        experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
        repo_root: Path = PROJECT_ROOT,
        experiment_id: str | None = None,
        generation_script: str = "unknown",
        manifest_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.experiment_root = Path(experiment_root)
        self.runs_root = self.experiment_root / "runs"
        self.repo_root = Path(repo_root)
        self.experiment_id = experiment_id or create_experiment_id(
            task_type,
            retrieval_method,
            chunk_strategy,
            top_k,
        )
        self.run_dir = self.runs_root / self.experiment_id
        self.config: dict[str, Any] = {
            "schema_version": "experiment-run-v1",
            "experiment_id": self.experiment_id,
            "task_type": task_type or "unknown",
            "retrieval_method": retrieval_method or "unknown",
            "chunk_strategy": chunk_strategy or "unknown",
            "embedding_model": embedding_model or "unknown",
            "top_k": top_k,
            "qa_version": qa_version or "unknown",
            "hybrid_parameters": dict(hybrid_parameters or {}),
            "random_seed": random_seed if random_seed is not None else "unknown",
            "parameters": dict(parameters or {}),
            "generation_script": generation_script or "unknown",
        }
        supplied_inputs = dict(inputs or {})
        self.inputs = {
            role: supplied_inputs.get(role)
            for role in REQUIRED_INPUT_ROLES
        }
        self.inputs.update(
            {
                key: value
                for key, value in supplied_inputs.items()
                if key not in self.inputs
            }
        )
        self.manifest_metadata = dict(manifest_metadata or {})
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self._started = False
        self._finalized = False

    def start(self) -> "ExperimentRecorder":
        if self._started:
            return self
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=False, exist_ok=False)
        self.started_at = datetime.now(timezone.utc)
        git_state = collect_git_state(self.repo_root)
        config = {
            **self.config,
            "started_at": self.started_at.isoformat(),
        }
        atomic_write_json(self.run_dir / "config.json", config)
        atomic_write_json(self.run_dir / "environment.json", collect_environment())
        data_manifest = build_data_manifest(self.inputs)
        if self.manifest_metadata:
            data_manifest["metadata"] = self.manifest_metadata
        atomic_write_json(self.run_dir / "data_manifest.json", data_manifest)
        atomic_write_text(
            self.run_dir / "git_commit.txt",
            f"{git_state['git_commit']}\n",
        )
        atomic_write_json(
            self.run_dir / "metrics.json",
            {"status": "running", "metrics": {}, "error": None},
        )
        atomic_write_jsonl(self.run_dir / "per_question_results.jsonl", [])
        atomic_write_csv(self.run_dir / "summary.csv", [], ["status"])
        atomic_write_text(
            self.run_dir / "run.log",
            (
                f"{self.started_at.isoformat()} status=running\n"
                f"git_commit={git_state['git_commit']}\n"
                f"git_worktree_clean={git_state['git_worktree_clean']}\n"
            ),
        )
        self._started = True
        return self

    def record_results(
        self,
        metrics: Mapping[str, Any],
        per_question_results: Iterable[Mapping[str, Any]],
        *,
        summary_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not self._started:
            self.start()
        if self._finalized:
            raise RuntimeError("experiment run is already finalized")
        rows = [dict(row) for row in per_question_results]
        self.completed_at = datetime.now(timezone.utc)
        payload = {
            "status": "completed",
            "started_at": self.started_at.isoformat() if self.started_at else "unknown",
            "completed_at": self.completed_at.isoformat(),
            "metrics": dict(metrics),
            "error": None,
        }
        atomic_write_json(self.run_dir / "metrics.json", payload)
        atomic_write_jsonl(self.run_dir / "per_question_results.jsonl", rows)
        if summary_rows is None:
            summary_rows = [{"status": "completed", **_scalar_metrics(metrics)}]
        atomic_write_csv(self.run_dir / "summary.csv", list(summary_rows))
        self._append_log(
            f"{self.completed_at.isoformat()} status=completed per_question_count={len(rows)}"
        )
        self._finalized = True

    def record_failure(self, error: BaseException) -> None:
        if not self._started:
            self.start()
        if self._finalized:
            return
        self.completed_at = datetime.now(timezone.utc)
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        atomic_write_json(
            self.run_dir / "metrics.json",
            {
                "status": "failed",
                "started_at": self.started_at.isoformat() if self.started_at else "unknown",
                "completed_at": self.completed_at.isoformat(),
                "metrics": {},
                "error": error_payload,
            },
        )
        atomic_write_csv(
            self.run_dir / "summary.csv",
            [
                {
                    "status": "failed",
                    "error_type": error_payload["type"],
                    "error_message": error_payload["message"],
                }
            ],
        )
        self._append_log(
            f"{self.completed_at.isoformat()} status=failed "
            f"error={error_payload['type']}: {error_payload['message']}"
        )
        self._finalized = True

    def _append_log(self, message: str) -> None:
        path = self.run_dir / "run.log"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        atomic_write_text(path, existing + message + "\n")

    def __enter__(self) -> "ExperimentRecorder":
        return self.start()

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is not None:
            self.record_failure(exc)
            return False
        if not self._finalized:
            self.record_failure(
                RuntimeError("experiment exited without record_results()")
            )
        return False


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
