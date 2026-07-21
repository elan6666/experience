"""Reproducible run manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from a_share_research.contracts.base import CanonicalModel, ContractError
from a_share_research.protocol.splits import Partition, Purpose, UniverseClass
from a_share_research.quality.states import ResultState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunManifest(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "run_manifest"

    run_id: str
    model: str
    universe: UniverseClass
    information_set: str
    split: Partition
    purpose: Purpose
    data_hash: str
    asset_registry_hash: str
    execution_calendar_manifest_hash: str
    feature_schema_hash: str
    market_state_hash: str
    config_hash: str
    code_hash: str
    upstream_commit: str
    seed: int
    status: ResultState
    started_at: datetime
    completed_at: datetime | None
    prediction_hash: str | None = None
    formal_feature_manifest_hash: str | None = None
    protocol_open_receipt_hash: str | None = None
    deviations: tuple[str, ...] = ()
    protocol_version: str = "v1"

    def validate(self) -> None:
        if not isinstance(self.universe, UniverseClass):
            raise ContractError("universe must use UniverseClass")
        if not isinstance(self.split, Partition):
            raise ContractError("split must use Partition")
        if not isinstance(self.purpose, Purpose):
            raise ContractError("purpose must use Purpose")
        if not isinstance(self.status, ResultState):
            raise ContractError("status must use ResultState")
        for name in ("run_id", "model", "information_set"):
            if not getattr(self, name):
                raise ContractError(f"{name} is required")
        for name in (
            "data_hash",
            "asset_registry_hash",
            "execution_calendar_manifest_hash",
            "feature_schema_hash",
            "market_state_hash",
            "config_hash",
            "code_hash",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be a lowercase SHA-256")
        if self.prediction_hash is not None and not _SHA256.fullmatch(self.prediction_hash):
            raise ContractError("prediction_hash must be a lowercase SHA-256")
        for name in ("formal_feature_manifest_hash", "protocol_open_receipt_hash"):
            value = getattr(self, name)
            if value is not None and not _SHA256.fullmatch(value):
                raise ContractError(f"{name} must be a lowercase SHA-256")
        if not self.upstream_commit:
            raise ContractError("upstream_commit or explicit internal provenance is required")
        for name in ("started_at", "completed_at"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ContractError(f"{name} must be timezone-aware")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ContractError("completed_at cannot precede started_at")
        if not self.protocol_version:
            raise ContractError("protocol_version is required")
        compatible_partitions = {
            Purpose.FIT: {Partition.TRAIN},
            Purpose.SELECT: {Partition.TRAIN, Partition.VALIDATION},
            Purpose.LEGACY_REPORT: {Partition.LEGACY_VIEWED},
            Purpose.FINAL_EVALUATION: {Partition.FUTURE_UNSEEN},
        }[self.purpose]
        if self.split not in compatible_partitions:
            raise ContractError("purpose is incompatible with manifest partition")
        if self.split is Partition.LEGACY_VIEWED and self.status in {
            ResultState.PASS,
            ResultState.PASS_WITH_WARNING,
            ResultState.VALID_NEGATIVE,
        }:
            raise ContractError("LEGACY_VIEWED result cannot carry a rankable state")
        if self.universe in {UniverseClass.TECH32, UniverseClass.TECH90} and self.status in {
            ResultState.PASS,
            ResultState.PASS_WITH_WARNING,
            ResultState.VALID_NEGATIVE,
        }:
            raise ContractError("selected technology universes must be EXPLORATORY_ONLY")
        candidate_status = self.status in {
            ResultState.PASS,
            ResultState.PASS_WITH_WARNING,
            ResultState.VALID_NEGATIVE,
        }
        formal_context = (
            candidate_status
            and self.split is not Partition.LEGACY_VIEWED
            and self.universe not in {UniverseClass.TECH32, UniverseClass.TECH90}
        )
        if formal_context and self.purpose not in {Purpose.SELECT, Purpose.FINAL_EVALUATION}:
            raise ContractError("rankable manifest has an incompatible purpose")
        if formal_context and self.formal_feature_manifest_hash is None:
            raise ContractError("formal rankable run requires D0 feature eligibility receipt")
        if formal_context and self.split is Partition.FUTURE_UNSEEN:
            if self.protocol_open_receipt_hash is None:
                raise ContractError("future rankable run requires protocol opening receipt")
