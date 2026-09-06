"""Small helpers for portable, privacy-safe DaT experiment provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent


def current_git_commit(root: str | Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(root), check=True,
            capture_output=True, text=True,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[: int(length)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: str | Path, root: str | Path = REPO_ROOT) -> str:
    """Return a repository-relative path and reject machine-specific paths."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must be inside repository root: {path}") from exc


def resolve_portable_path(value: str | Path, root: str | Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root).resolve() / path


def research_valid(*, max_train_batches: int | None, max_val_batches: int | None, debug: bool = False) -> bool:
    return not bool(debug) and (max_train_batches in (None, 0)) and (max_val_batches in (None, 0))
