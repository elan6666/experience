"""Content and permission checks for detached upstream worktrees."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

RUNTIME_DIRECTORY_NAMES = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
)
RUNTIME_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _excluded(relative: Path) -> bool:
    return any(part in RUNTIME_DIRECTORY_NAMES for part in relative.parts) or (
        relative.suffix in RUNTIME_FILE_SUFFIXES
    )


def worktree_content_sha256(root: Path) -> str:
    """Hash paths, symlink targets and file bytes outside Git/runtime caches."""

    root = root.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if _excluded(relative):
            continue
        relative_bytes = relative.as_posix().encode("utf-8")
        if path.is_symlink():
            digest.update(b"L\0" + relative_bytes + b"\0")
            digest.update(os.readlink(path).encode("utf-8") + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + relative_bytes + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        elif path.is_dir():
            digest.update(b"D\0" + relative_bytes + b"\0")
    return digest.hexdigest()


def git_status_porcelain(root: Path) -> tuple[str, ...]:
    """Return tracked and untracked changes using porcelain-v1 records."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def set_tree_read_only(root: Path) -> None:
    """Remove write bits from the checkout root and every descendant."""

    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            continue
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def writable_paths(root: Path) -> tuple[str, ...]:
    """List checkout-relative paths that still have any write permission bit."""

    writable: list[str] = []
    candidates = [root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())]
    for path in candidates:
        if path.is_symlink():
            continue
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            writable.append("." if path == root else path.relative_to(root).as_posix())
    return tuple(writable)
