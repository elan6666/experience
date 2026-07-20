#!/usr/bin/env python3
"""Audit pinned remote refs and emit a schema-v2 server receipt."""

from __future__ import annotations

import argparse
import hashlib
import os
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
    checkout_candidates,
    error_digest,
    write_receipt,
)

APPROVED_PREFIX = Path("/data/yilangliu/a_share_research")


def _approved(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != APPROVED_PREFIX and APPROVED_PREFIX not in resolved.parents:
        raise ValueError(f"refusing non-server path: {resolved}")
    return resolved


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _head(repository: str, branch: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", repository, f"refs/heads/{branch}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    line = completed.stdout.strip()
    if not line:
        raise RuntimeError(f"empty remote ref for branch {branch}")
    return line.split("\t", 1)[0]


def _base(status: str, stage: str, stderr_digest: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "registry_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "__registry__",
        "status": status,
        "stage": stage,
        "stderr_digest": stderr_digest,
        "command": _command(),
        "provenance_status": "REGISTRY_AUDIT",
        "registry_sha256": None,
        "checkout_candidates": None,
        "upstreams": None,
        "failed_model": None,
        "error_kind": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/upstreams.lock.yaml"))
    parser.add_argument(
        "--model",
        action="append",
        help="Audit only the named upstream; repeat for multiple models.",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = _approved(args.receipt)
    stage = "load_registry"
    registry_sha256: str | None = None
    candidates: list[str] | None = None
    rows: list[dict[str, Any]] = []
    current_model: str | None = None
    try:
        registry_bytes = args.registry.read_bytes()
        registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
        document: dict[str, Any] = yaml.safe_load(registry_bytes)
        assert_registry(document)
        candidates = list(checkout_candidates(document))
        stage = "query_remote_heads"
        selected_names = args.model or sorted(document["upstreams"])
        unknown = sorted(set(selected_names) - set(document["upstreams"]))
        if unknown:
            raise ValueError(f"unknown upstream models: {unknown}")
        for name in selected_names:
            entry = document["upstreams"][name]
            current_model = name
            head = _head(entry["repository_url"], entry.get("default_branch", "main"))
            rows.append(
                {
                    "model": name,
                    "pinned_commit": entry["commit"],
                    "remote_head": head,
                    "pinned_is_current_head": head == entry["commit"],
                    "license_spdx": entry["license_spdx"],
                    "license_status": entry["license_status"],
                    "license_review_required": entry["license_review_required"],
                    "provenance_status": entry["provenance_status"],
                }
            )
        receipt = _base("PASS", "complete", EMPTY_STDERR_SHA256)
        receipt.update(
            {
                "registry_sha256": registry_sha256,
                "checkout_candidates": candidates,
                "upstreams": rows,
            }
        )
        stage = "write_receipt"
        write_receipt(receipt_path, receipt)
        return 0
    except Exception as error:
        receipt = _base("UPSTREAM_FAIL", stage, error_digest(error))
        receipt.update(
            {
                "registry_sha256": registry_sha256,
                "checkout_candidates": candidates,
                "upstreams": rows,
                "failed_model": current_model,
                "error_kind": type(error).__name__,
            }
        )
        write_receipt(receipt_path, receipt)
        print(f"registry audit failed at {stage}; see receipt digest", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
