"""Replayable source-only evidence shared by job builders and runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from a_share_research.contracts import ContractError

ALLOWED_ROOTS = {"configs", "docs", "patches", "scripts", "src", "tests"}
ALLOWED_ROOT_FILES = {".gitignore", "AGENTS.md", "CLAUDE.md", "README.md", "pyproject.toml"}
FORBIDDEN_PARTS = {
    "artifacts",
    "checkpoints",
    "data",
    "logs",
    "predictions",
    "results",
    "runs",
    "token",
    "weights",
}
RUNTIME_EXCLUDED_ROOTS = {".git", ".pytest_cache", ".ruff_cache", ".venv"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_path(relative: Path) -> bool:
    return (
        relative.parts[0] in RUNTIME_EXCLUDED_ROOTS
        or "__pycache__" in relative.parts
        or relative.suffix in {".pyc", ".pyo"}
    )


def source_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    violations: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _runtime_path(relative):
            continue
        if relative.parts[0].lower() in FORBIDDEN_PARTS:
            violations.append(relative.as_posix())
            continue
        if path.is_symlink():
            raise ContractError(f"source tree contains a symlink: {relative}")
        if not path.is_file():
            continue
        if len(relative.parts) == 1:
            if relative.name in ALLOWED_ROOT_FILES:
                files.append(path)
        elif relative.parts[0] in ALLOWED_ROOTS:
            files.append(path)
    if violations:
        raise ContractError(f"forbidden source paths detected: {sorted(violations)[:10]}")
    return tuple(sorted(files))


def source_manifest_payload(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    return {
        "root_name": root.name,
        "algorithm": "sha256",
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for path in source_files(root)
        ],
    }


def verify_source_manifest(path: Path, root: Path) -> str:
    """Require the receipt to describe the exact source tree now executing."""
    try:
        declared = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("source manifest is unreadable") from error
    actual = source_manifest_payload(root)
    if declared != actual:
        raise ContractError("source manifest does not match the executing source tree")
    return _sha256(path)
