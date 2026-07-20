"""Read-only audit of reusable server datasets before any incremental fetch."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExistingDataset:
    logical_name: str
    path: Path
    exists: bool
    manifest_sha256: str | None
    verified: bool = False
    verification_reasons: tuple[str, ...] = ()


DEFAULT_CANDIDATES = {
    "historical_csi300": "data/processed/historical_csi300",
    "historical_star50": "data/processed/historical_star50",
    "tech32": "data/processed/tech32_open_to_open_v2",
    "tech100": "data/processed/tech100",
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_manifest(path: Path) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ("manifest is unreadable or invalid JSON",)
    if not isinstance(payload, dict) or not payload.get("schema_version"):
        reasons.append("manifest lacks schema_version")
    hash_values = tuple(
        value
        for key, value in payload.items()
        if "hash" in str(key).lower() and isinstance(value, str)
    )
    if not any(re.fullmatch(r"[0-9a-f]{64}", value) for value in hash_values):
        reasons.append("manifest lacks SHA-256 source/content evidence")
    return not reasons, tuple(reasons)


def audit_existing_server_data(
    research_root: Path,
    candidates: dict[str, str] | None = None,
) -> tuple[ExistingDataset, ...]:
    """Inspect only manifests; do not trust or rewrite legacy materializations."""
    found: list[ExistingDataset] = []
    for logical_name, relative in sorted((candidates or DEFAULT_CANDIDATES).items()):
        path = (research_root / relative).resolve()
        manifests = tuple(sorted(path.glob("**/*manifest*.json"))) if path.exists() else ()
        digest = _file_hash(manifests[-1]) if manifests else None
        verified, reasons = _audit_manifest(manifests[-1]) if manifests else (
            False,
            ("no manifest found",),
        )
        found.append(
            ExistingDataset(
                logical_name,
                path,
                path.exists(),
                digest,
                verified,
                reasons,
            )
        )
    return tuple(found)


def missing_datasets(inventory: tuple[ExistingDataset, ...]) -> tuple[str, ...]:
    """Only these logical datasets may be scheduled for incremental acquisition."""
    return tuple(
        item.logical_name
        for item in inventory
        if not item.exists or not item.manifest_sha256 or not item.verified
    )
