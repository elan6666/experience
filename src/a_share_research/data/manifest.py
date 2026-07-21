"""Versioned D0 dataset evidence and per-universe gate status."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import ClassVar

from a_share_research.contracts import CanonicalModel, ContractError, canonical_hash
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class UniverseGate(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "d0_universe_gate"

    universe: UniverseClass
    status: ResultState
    membership_coverage: float
    core_coverage: float
    duplicate_keys: int
    pit_violations: int
    label_boundary_violations: int
    feature_schema_violations: int = 0
    warnings: tuple[str, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.universe, UniverseClass) or not isinstance(self.status, ResultState):
            raise ContractError("typed universe and result status are required")
        for name in ("membership_coverage", "core_coverage"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ContractError(f"{name} must be in [0, 1]")
        for name in (
            "duplicate_keys",
            "pit_violations",
            "label_boundary_violations",
            "feature_schema_violations",
        ):
            if getattr(self, name) < 0:
                raise ContractError(f"{name} cannot be negative")
        if any(
            (
                self.duplicate_keys,
                self.pit_violations,
                self.label_boundary_violations,
                self.feature_schema_violations,
            )
        ):
            if self.status not in {ResultState.INVALID_DATA, ResultState.BLOCKED}:
                raise ContractError("hard D0 violations cannot pass the universe gate")
        allowed = {
            ResultState.PASS,
            ResultState.PASS_WITH_WARNING,
            ResultState.EXPLORATORY_ONLY,
            ResultState.BLOCKED,
            ResultState.INVALID_DATA,
        }
        if self.status not in allowed:
            raise ContractError("D0 universe gate has a non-data result state")
        if self.universe in {UniverseClass.TECH32, UniverseClass.TECH90}:
            if self.status in {
                ResultState.PASS,
                ResultState.PASS_WITH_WARNING,
                ResultState.VALID_NEGATIVE,
            }:
                raise ContractError(
                    "2026 selected technology lists cannot receive a formal pass state"
                )


@dataclass(frozen=True)
class D0Manifest(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "d0_manifest"

    dataset_id: str
    created_at_utc: datetime
    cutoff_date: date
    raw_snapshot_hashes: dict[str, str]
    canonical_table_hashes: dict[str, str]
    security_master_hash: str
    trading_calendar_hash: str
    feature_schema_hash: str
    market_state_hash: str
    universe_gates: tuple[UniverseGate, ...]
    provider_transport_notice: str
    protocol_version: str = "v1"

    def validate(self) -> None:
        if not self.dataset_id or not self.protocol_version:
            raise ContractError("dataset_id and protocol_version are required")
        if self.created_at_utc.tzinfo is None or self.created_at_utc.utcoffset() is None:
            raise ContractError("created_at_utc must be timezone-aware")
        hashes = {
            **self.raw_snapshot_hashes,
            **self.canonical_table_hashes,
            "security_master": self.security_master_hash,
            "trading_calendar": self.trading_calendar_hash,
            "feature_schema": self.feature_schema_hash,
            "market_state": self.market_state_hash,
        }
        if not self.raw_snapshot_hashes or not self.canonical_table_hashes or any(
            not _SHA256.fullmatch(value) for value in hashes.values()
        ):
            raise ContractError("D0 evidence hashes must all be SHA-256")
        if {gate.universe for gate in self.universe_gates} != set(UniverseClass):
            raise ContractError("D0 manifest must contain exactly the four universe gates")
        if len(self.universe_gates) != len(UniverseClass):
            raise ContractError("D0 manifest contains duplicate universe gates")
        if "plain HTTP" not in self.provider_transport_notice:
            raise ContractError("provider plain HTTP limitation must remain visible")

    @property
    def content_hash(self) -> str:
        """Stable data identity excludes only receipt creation time."""
        payload = self.to_dict()
        payload.pop("created_at_utc")
        return canonical_hash(payload)
