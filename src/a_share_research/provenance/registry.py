"""Pure validation for the upstream lock registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPECTED_BASELINES = frozenset({"ridge", "lightgbm"})
EXPECTED_UPSTREAMS = frozenset({"itransformer", "fact", "timepro", "timexer", "s4m"})
ALLOWED_PROVENANCE_STATES = frozenset(
    {
        "READY",
        "READY_FOR_SERVER_SMOKE",
        "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED",
        "BLOCKED_LICENSE",
        "BLOCKED",
    }
)
REQUIRED_BASELINE_FIELDS = frozenset(
    {
        "kind",
        "implementation",
        "package",
        "package_url",
        "source_url",
        "documentation_url",
        "license_spdx",
        "provenance_status",
        "execution_device",
        "native_semantics",
        "project_boundary",
    }
)
REQUIRED_UPSTREAM_FIELDS = frozenset(
    {
        "display_name",
        "venue",
        "paper_title",
        "paper_url",
        "repository_url",
        "commit",
        "license_spdx",
        "license_status",
        "license_review_required",
        "provenance_status",
        "official_entrypoint",
        "native_semantics",
    }
)


class RegistryError(ValueError):
    """Raised when a provenance registry violates the frozen contract."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_registry(document: Mapping[str, Any]) -> list[str]:
    """Return deterministic validation errors without performing I/O."""

    errors: list[str] = []
    baselines = _mapping(document.get("baselines"))
    upstreams = _mapping(document.get("upstreams"))

    if set(baselines) != EXPECTED_BASELINES:
        errors.append(
            f"baselines must be exactly {sorted(EXPECTED_BASELINES)}, got {sorted(baselines)}"
        )
    if set(upstreams) != EXPECTED_UPSTREAMS:
        errors.append(
            f"upstreams must be exactly {sorted(EXPECTED_UPSTREAMS)}, got {sorted(upstreams)}"
        )

    for name, raw in sorted(baselines.items()):
        entry = _mapping(raw)
        missing = sorted(REQUIRED_BASELINE_FIELDS - set(entry))
        if missing:
            errors.append(f"{name}: missing baseline fields {missing}")
            continue
        if entry.get("kind") != "algorithm_baseline":
            errors.append(f"{name}: kind must be algorithm_baseline")
        if entry.get("provenance_status") != "READY":
            errors.append(f"{name}: baseline provenance_status must be READY")
        semantics = _mapping(entry.get("native_semantics"))
        for field in ("input", "output", "objective", "optimizer", "scheduler", "inference"):
            if not semantics.get(field):
                errors.append(f"{name}: native_semantics.{field} is required")

    for name, raw in sorted(upstreams.items()):
        entry = _mapping(raw)
        missing = sorted(REQUIRED_UPSTREAM_FIELDS - set(entry))
        if missing:
            errors.append(f"{name}: missing fields {missing}")
            continue

        commit = entry.get("commit")
        if not isinstance(commit, str) or len(commit) != 40:
            errors.append(f"{name}: commit must be a full 40-character SHA")

        state = entry.get("provenance_status")
        if state not in ALLOWED_PROVENANCE_STATES:
            errors.append(f"{name}: unsupported provenance_status {state!r}")

        license_spdx = entry.get("license_spdx")
        authorization = entry.get("license_authorization")
        authorized = isinstance(authorization, str) and authorization.strip() != ""
        if license_spdx == "NOASSERTION" and state != "BLOCKED_LICENSE" and not authorized:
            errors.append(
                f"{name}: NOASSERTION license must be BLOCKED_LICENSE"
                " or carry license_authorization"
            )
        if state == "BLOCKED_LICENSE" and license_spdx != "NOASSERTION":
            errors.append(f"{name}: BLOCKED_LICENSE must have license_spdx NOASSERTION")
        review_required = entry.get("license_review_required")
        if not isinstance(review_required, bool):
            errors.append(f"{name}: license_review_required must be boolean")
        if state == "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED" and not review_required:
            errors.append(f"{name}: review-required state must set license_review_required")
        if entry.get("license_status") == "MIT_WITH_ATTRIBUTION_AMBIGUITY":
            if state != "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED" or not review_required:
                errors.append(f"{name}: ambiguous MIT attribution must require review")

        semantics = _mapping(entry.get("native_semantics"))
        for field in ("architecture", "input", "output", "loss", "optimizer", "inference"):
            if not semantics.get(field):
                errors.append(f"{name}: native_semantics.{field} is required")

    gate = _mapping(document.get("server_gate"))
    declared_blocked = set(gate.get("blocked_entries", []))
    actual_blocked = {
        name
        for name, raw in upstreams.items()
        if _mapping(raw).get("provenance_status") == "BLOCKED_LICENSE"
    }
    if declared_blocked != actual_blocked:
        errors.append(
            "server_gate.blocked_entries must exactly match BLOCKED_LICENSE entries: "
            f"expected {sorted(actual_blocked)}, got {sorted(declared_blocked)}"
        )
    declared_review = set(gate.get("review_entries", []))
    actual_review = {
        name
        for name, raw in upstreams.items()
        if _mapping(raw).get("provenance_status")
        == "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED"
    }
    if declared_review != actual_review:
        errors.append(
            "server_gate.review_entries must exactly match review-required entries: "
            f"expected {sorted(actual_review)}, got {sorted(declared_review)}"
        )

    return errors


def assert_registry(document: Mapping[str, Any]) -> None:
    """Raise ``RegistryError`` when the registry is invalid."""

    errors = validate_registry(document)
    if errors:
        raise RegistryError("\n".join(errors))


def checkout_candidates(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only upstreams whose license and provenance permit a server smoke."""

    assert_registry(document)
    upstreams = _mapping(document["upstreams"])
    return tuple(
        name
        for name in sorted(upstreams)
        if _mapping(upstreams[name]).get("provenance_status")
        in {"READY_FOR_SERVER_SMOKE", "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED"}
    )
