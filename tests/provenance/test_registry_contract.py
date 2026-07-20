from __future__ import annotations

from pathlib import Path

import yaml

from a_share_research.provenance import checkout_candidates, validate_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _baseline() -> dict[str, object]:
    return {
        "kind": "algorithm_baseline",
        "implementation": "package.Estimator",
        "package": "package==1.0",
        "package_url": "https://example.invalid/package",
        "source_url": "https://example.invalid/source",
        "documentation_url": "https://example.invalid/docs",
        "license_spdx": "MIT",
        "provenance_status": "READY",
        "execution_device": "cpu",
        "native_semantics": {
            "input": "input",
            "output": "output",
            "objective": "objective",
            "optimizer": "optimizer",
            "scheduler": "scheduler",
            "inference": "inference",
        },
        "project_boundary": "boundary",
    }


def _entry(status: str, license_spdx: str, license_status: str) -> dict[str, object]:
    return {
        "display_name": "model",
        "venue": "venue",
        "paper_title": "paper",
        "paper_url": "https://example.invalid/paper",
        "repository_url": "https://example.invalid/repository.git",
        "commit": "a" * 40,
        "license_spdx": license_spdx,
        "license_status": license_status,
        "license_review_required": status == "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED",
        "provenance_status": status,
        "official_entrypoint": "run.py",
        "native_semantics": {
            "architecture": "architecture",
            "input": "input",
            "output": "output",
            "loss": "loss",
            "optimizer": "optimizer",
            "inference": "inference",
        },
    }


def _document() -> dict[str, object]:
    return {
        "baselines": {"ridge": _baseline(), "lightgbm": _baseline()},
        "upstreams": {
            "itransformer": _entry("READY_FOR_SERVER_SMOKE", "MIT", "MIT_CLEAR"),
            "fact": _entry(
                "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED",
                "MIT",
                "MIT_WITH_ATTRIBUTION_AMBIGUITY",
            ),
            "timepro": _entry("BLOCKED_LICENSE", "NOASSERTION", "NO_REPOSITORY_LICENSE"),
            "timexer": _entry("BLOCKED_LICENSE", "NOASSERTION", "NO_REPOSITORY_LICENSE"),
            "s4m": _entry("BLOCKED_LICENSE", "NOASSERTION", "NO_REPOSITORY_LICENSE"),
        },
        "server_gate": {
            "blocked_entries": ["timepro", "timexer", "s4m"],
            "review_entries": ["fact"],
        },
    }


def test_real_lock_file_satisfies_registry_contract() -> None:
    lock_path = PROJECT_ROOT / "configs" / "upstreams.lock.yaml"
    document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert validate_registry(document) == []
    assert checkout_candidates(document) == ("fact", "itransformer", "s4m", "timepro", "timexer")


def test_noassertion_cannot_pass_checkout_gate() -> None:
    document = _document()
    document["upstreams"]["timepro"]["provenance_status"] = "READY_FOR_SERVER_SMOKE"
    errors = validate_registry(document)
    assert any("NOASSERTION license must be BLOCKED_LICENSE" in error for error in errors)


def test_baseline_semantics_are_fully_validated() -> None:
    document = _document()
    del document["baselines"]["ridge"]["native_semantics"]["inference"]
    errors = validate_registry(document)
    assert any("ridge: native_semantics.inference is required" in error for error in errors)


def test_fact_attribution_ambiguity_requires_review_state() -> None:
    document = _document()
    document["upstreams"]["fact"]["license_review_required"] = False
    errors = validate_registry(document)
    assert any("ambiguous MIT attribution must require review" in error for error in errors)


def test_noassertion_with_authorization_can_pass_checkout_gate() -> None:
    document = _document()
    upstream = document["upstreams"]["timepro"]
    upstream["license_authorization"] = (
        "USER_ASSERTED_AUTHOR_PERMISSION_2026-07-20"
    )
    upstream["provenance_status"] = "READY_FOR_SERVER_SMOKE_REVIEW_REQUIRED"
    upstream["license_review_required"] = True
    document["server_gate"]["blocked_entries"] = ["timexer", "s4m"]
    document["server_gate"]["review_entries"] = ["fact", "timepro"]
    errors = validate_registry(document)
    assert not any("NOASSERTION" in error for error in errors)
    assert "timepro" in checkout_candidates(document)
    assert "timexer" not in checkout_candidates(document)


def test_noassertion_authorized_but_still_blocked_is_allowed() -> None:
    document = _document()
    document["upstreams"]["timepro"]["license_authorization"] = (
        "USER_ASSERTED_AUTHOR_PERMISSION_2026-07-20"
    )
    errors = validate_registry(document)
    assert not any("NOASSERTION" in error for error in errors)
    assert "timepro" not in checkout_candidates(document)
