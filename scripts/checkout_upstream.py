#!/usr/bin/env python3
"""Create one detached, integrity-checked, read-only upstream checkout."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from a_share_research.provenance import (
    EMPTY_STDERR_SHA256,
    RECEIPT_SCHEMA_VERSION,
    assert_registry,
    error_digest,
    git_status_porcelain,
    set_tree_read_only,
    worktree_content_sha256,
    writable_paths,
    write_receipt,
)

APPROVED_PREFIX = Path("/data/yilangliu/a_share_research")
CHECKOUT_STATES = {
    "READY_FOR_SERVER_SMOKE",
    "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED",
}


def _approved(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != APPROVED_PREFIX and APPROVED_PREFIX not in resolved.parents:
        raise ValueError(f"refusing non-server path: {resolved}")
    return resolved


def _run(*argv: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _base(model: str, status: str, stage: str, digest: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "checkout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "status": status,
        "stage": stage,
        "stderr_digest": digest,
        "command": _command(),
        "provenance_status": None,
        "commit": None,
        "license_status": None,
        "git_status": None,
        "worktree_content_sha256": None,
        "read_only_verified": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--registry", type=Path, default=Path("configs/upstreams.lock.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkout-root", type=Path)
    source.add_argument(
        "--existing-checkout",
        type=Path,
        help="Re-audit a previously detached checkout without network access.",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = _approved(args.receipt)
    stage = "load_registry"
    receipt = _base(args.model, "UPSTREAM_FAIL", stage, EMPTY_STDERR_SHA256)
    try:
        document: dict[str, Any] = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
        assert_registry(document)
        if args.model not in document["upstreams"]:
            raise ValueError(f"unknown model: {args.model}")
        entry = document["upstreams"][args.model]
        receipt.update(
            {
                "provenance_status": entry["provenance_status"],
                "commit": entry["commit"],
                "license_status": entry["license_status"],
            }
        )
        if entry["provenance_status"] not in CHECKOUT_STATES:
            stage = "license_gate"
            receipt["status"] = "BLOCKED"
            raise PermissionError(f"checkout gate is {entry['provenance_status']}")

        stage = "select_checkout"
        if args.existing_checkout is not None:
            destination = _approved(args.existing_checkout)
            if not destination.is_dir():
                raise FileNotFoundError(f"existing checkout is not a directory: {destination}")
        else:
            stage = "clone_detached_commit"
            assert args.checkout_root is not None
            checkout_root = _approved(args.checkout_root)
            destination = checkout_root / f"{args.model}@{entry['commit'][:12]}"
            if destination.exists():
                raise FileExistsError(f"destination already exists: {destination}")
            checkout_root.mkdir(parents=True, exist_ok=True)
            _run(
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                entry["repository_url"],
                str(destination),
            )
            _run("git", "checkout", "--detach", entry["commit"], cwd=destination)
        if destination == receipt_path or destination in receipt_path.parents:
            raise ValueError("receipt must be outside the read-only checkout")
        resolved_commit = _run("git", "rev-parse", "HEAD", cwd=destination)
        if resolved_commit != entry["commit"]:
            raise RuntimeError(f"commit mismatch: {resolved_commit} != {entry['commit']}")

        stage = "verify_clean_content"
        git_status = git_status_porcelain(destination)
        if git_status:
            raise RuntimeError(f"checkout has tracked or untracked changes: {git_status}")
        content_hash = worktree_content_sha256(destination)

        stage = "enforce_read_only"
        set_tree_read_only(destination)
        still_writable = writable_paths(destination)
        if still_writable:
            raise RuntimeError(f"checkout still has writable paths: {still_writable}")
        git_status_after = git_status_porcelain(destination)
        content_hash_after = worktree_content_sha256(destination)
        if git_status_after != git_status or content_hash_after != content_hash:
            raise RuntimeError("checkout changed while enforcing read-only permissions")

        receipt.update(
            {
                "status": (
                    "PASS_WITH_WARNING"
                    if entry["license_review_required"]
                    else "PASS"
                ),
                "stage": "complete",
                "stderr_digest": EMPTY_STDERR_SHA256,
                "commit": resolved_commit,
                "git_tree": _run("git", "rev-parse", "HEAD^{tree}", cwd=destination),
                "git_status": list(git_status_after),
                "worktree_content_sha256": content_hash_after,
                "checkout": str(destination),
                "read_only_verified": True,
                "license_review_required": entry["license_review_required"],
            }
        )
        stage = "write_receipt"
        write_receipt(receipt_path, receipt)
        return 0
    except Exception as error:
        if receipt["status"] in {"PASS", "PASS_WITH_WARNING"}:
            receipt["status"] = "UPSTREAM_FAIL"
        receipt["stage"] = stage
        receipt["stderr_digest"] = error_digest(error)
        write_receipt(receipt_path, receipt)
        print(f"checkout failed at {stage}; see receipt digest", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
