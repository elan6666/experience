"""Structured receipt schema shared by server-only provenance scripts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA_VERSION = 2
EMPTY_STDERR_SHA256 = hashlib.sha256(b"").hexdigest()
COMMON_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "created_at_utc",
        "model",
        "status",
        "stage",
        "stderr_digest",
        "command",
        "provenance_status",
    }
)
TYPE_REQUIRED_FIELDS = {
    "registry_audit": frozenset({"registry_sha256", "upstreams", "checkout_candidates"}),
    "checkout": frozenset(
        {
            "commit",
            "license_status",
            "git_status",
            "worktree_content_sha256",
            "read_only_verified",
        }
    ),
    "environment": frozenset(
        {
            "environment_mode",
            "requirements_sha256",
            "resolved_lock_sha256",
            "resolved_lock_path",
            "python_version",
            "torch_version",
            "cuda_version",
        }
    ),
    "smoke": frozenset(
        {
            "commit",
            "license_status",
            "source_tree_hash_before",
            "source_tree_hash_after",
            "worktree_content_sha256_before",
            "worktree_content_sha256_after",
            "git_status_before",
            "git_status_after",
            "python_version",
            "torch_version",
            "cuda_version",
            "gpu_name",
            "cuda_visible_devices",
            "torch_current_device",
            "physical_gpu_requested",
            "physical_gpu_evidence",
            "environment_receipt_sha256",
            "resolved_lock_sha256",
            "sys_executable",
            "output_shape",
        }
    ),
}
SUCCESS_STATES = frozenset({"PASS", "PASS_WITH_WARNING"})
FAILURE_STATES = frozenset({"BLOCKED", "UPSTREAM_FAIL", "ENV_FAIL", "SMOKE_FAIL"})


def error_digest(error: BaseException | str) -> str:
    """Hash an error without persisting potentially sensitive raw stderr."""

    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        text = error.stderr if isinstance(error.stderr, str) else repr(error.stderr)
    else:
        text = str(error)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Validate key presence and success-state evidence."""

    errors: list[str] = []
    receipt_type = receipt.get("receipt_type")
    required = COMMON_REQUIRED_FIELDS | TYPE_REQUIRED_FIELDS.get(str(receipt_type), frozenset())
    missing = sorted(required - set(receipt))
    if missing:
        errors.append(f"missing receipt fields: {missing}")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RECEIPT_SCHEMA_VERSION}")
    status = receipt.get("status")
    if status not in SUCCESS_STATES | FAILURE_STATES:
        errors.append(f"unsupported receipt status: {status!r}")
    if not receipt.get("stage"):
        errors.append("stage must be non-empty")
    stderr_digest = receipt.get("stderr_digest")
    if not isinstance(stderr_digest, str) or len(stderr_digest) != 64:
        errors.append("stderr_digest must be a SHA-256 hex digest")
    if status in SUCCESS_STATES:
        for field in sorted(required - {"stderr_digest"}):
            if receipt.get(field) is None:
                errors.append(f"successful receipt field cannot be null: {field}")
        for field in (
            "registry_sha256",
            "requirements_sha256",
            "resolved_lock_sha256",
            "worktree_content_sha256",
            "worktree_content_sha256_before",
            "worktree_content_sha256_after",
        ):
            value = receipt.get(field)
            if value is not None and (not isinstance(value, str) or len(value) != 64):
                errors.append(f"{field} must be a SHA-256 hex digest")
        if receipt_type == "checkout":
            if receipt.get("git_status") != []:
                errors.append("successful checkout git_status must be empty")
            if receipt.get("read_only_verified") is not True:
                errors.append("successful checkout must verify read-only permissions")
        if receipt_type == "smoke":
            if receipt.get("git_status_before") != [] or receipt.get("git_status_after") != []:
                errors.append("successful smoke git status must be empty before and after")
            if receipt.get("source_tree_hash_before") != receipt.get("source_tree_hash_after"):
                errors.append("successful smoke Git tree hashes must match")
            if receipt.get("worktree_content_sha256_before") != receipt.get(
                "worktree_content_sha256_after"
            ):
                errors.append("successful smoke worktree hashes must match")
            evidence = receipt.get("physical_gpu_evidence")
            selected = evidence.get("selected", {}) if isinstance(evidence, Mapping) else {}
            for field in ("physical_index", "uuid", "pci_bus_id", "name"):
                if field not in selected:
                    errors.append(f"physical_gpu_evidence.selected.{field} is required")
            environment_sha = receipt.get("environment_receipt_sha256")
            if not isinstance(environment_sha, str) or len(environment_sha) != 64:
                errors.append("environment_receipt_sha256 must be a SHA-256 hex digest")
            resolved_sha = receipt.get("resolved_lock_sha256")
            if not isinstance(resolved_sha, str) or len(resolved_sha) != 64:
                errors.append("resolved_lock_sha256 must be a SHA-256 hex digest")
        if receipt_type == "environment" and receipt.get("environment_mode") == "compat-smoke":
            if not str(receipt.get("cuda_version", "")).startswith("12.8"):
                errors.append("compat-smoke environment must report CUDA 12.8")
    return errors


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically write a validated receipt."""

    errors = validate_receipt(receipt)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
