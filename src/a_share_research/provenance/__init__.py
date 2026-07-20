"""Upstream provenance contracts.

This package contains no network or model execution at import time. Network and
server checks live in ``scripts/`` and consume these pure validation helpers.
"""

from .integrity import (
    git_status_porcelain,
    set_tree_read_only,
    worktree_content_sha256,
    writable_paths,
)
from .receipts import (
    EMPTY_STDERR_SHA256,
    RECEIPT_SCHEMA_VERSION,
    error_digest,
    validate_receipt,
    write_receipt,
)
from .registry import RegistryError, assert_registry, checkout_candidates, validate_registry

__all__ = [
    "EMPTY_STDERR_SHA256",
    "RECEIPT_SCHEMA_VERSION",
    "RegistryError",
    "assert_registry",
    "checkout_candidates",
    "error_digest",
    "git_status_porcelain",
    "set_tree_read_only",
    "validate_receipt",
    "validate_registry",
    "worktree_content_sha256",
    "writable_paths",
    "write_receipt",
]
