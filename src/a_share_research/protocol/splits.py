"""Frozen research partitions and fail-closed access policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import ClassVar

from a_share_research.contracts.base import CanonicalModel, ContractError


class Partition(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    LEGACY_VIEWED = "LEGACY_VIEWED"
    FUTURE_UNSEEN = "FUTURE_UNSEEN"


class UniverseClass(str, Enum):
    CSI300 = "CSI300"
    STAR50 = "STAR50"
    TECH32 = "TECH32"
    TECH90 = "TECH90"


class Purpose(str, Enum):
    FIT = "FIT"
    SELECT = "SELECT"
    LEGACY_REPORT = "LEGACY_REPORT"
    FINAL_EVALUATION = "FINAL_EVALUATION"


@dataclass(frozen=True)
class SplitWindow(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "split_window"

    partition: Partition
    start: date
    end: date | None

    def validate(self) -> None:
        if not isinstance(self.partition, Partition):
            raise ContractError("partition must use Partition")
        if self.end is not None and self.end < self.start:
            raise ContractError("split end cannot precede start")

    def contains(self, value: date) -> bool:
        return value >= self.start and (self.end is None or value <= self.end)


@dataclass(frozen=True)
class ProtocolSpec(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "protocol_spec"

    windows: tuple[SplitWindow, ...]
    future_opened: bool = False
    protocol_open_receipt_hash: str | None = None
    protocol_version: str = "v1"

    def validate(self) -> None:
        if not self.protocol_version:
            raise ContractError("protocol_version is required")
        if self.future_opened:
            if self.protocol_open_receipt_hash is None or not re.fullmatch(
                r"[0-9a-f]{64}", self.protocol_open_receipt_hash
            ):
                raise ContractError("opened future protocol requires a SHA-256 receipt")
        elif self.protocol_open_receipt_hash is not None:
            raise ContractError("closed protocol cannot carry an opening receipt")
        required = set(Partition)
        actual = {window.partition for window in self.windows}
        if actual != required or len(self.windows) != len(required):
            raise ContractError("protocol must define each partition exactly once")
        ordered = sorted(self.windows, key=lambda window: window.start)
        for window in ordered:
            window.validate()
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end is None or previous.end >= current.start:
                raise ContractError("split windows overlap or are open before the final partition")

    @classmethod
    def research_v1(cls, future_start: date = date(2026, 7, 18)) -> ProtocolSpec:
        return cls(
            windows=(
                SplitWindow(Partition.TRAIN, date(2019, 1, 1), date(2024, 12, 31)),
                SplitWindow(Partition.VALIDATION, date(2025, 1, 1), date(2025, 12, 31)),
                SplitWindow(Partition.LEGACY_VIEWED, date(2026, 1, 1), date(2026, 7, 17)),
                SplitWindow(Partition.FUTURE_UNSEEN, future_start, None),
            )
        )

    def partition_for(self, value: date) -> Partition:
        self.validate()
        for window in self.windows:
            if window.contains(value):
                return window.partition
        raise ContractError(f"date is outside the registered protocol: {value}")

    def assert_access(self, value: date, purpose: Purpose) -> None:
        partition = self.partition_for(value)
        allowed = {
            Purpose.FIT: {Partition.TRAIN},
            Purpose.SELECT: {Partition.TRAIN, Partition.VALIDATION},
            Purpose.LEGACY_REPORT: {Partition.LEGACY_VIEWED},
            Purpose.FINAL_EVALUATION: {Partition.FUTURE_UNSEEN} if self.future_opened else set(),
        }[purpose]
        if partition not in allowed:
            raise ContractError(f"{partition.value} is forbidden for {purpose.value}")

    def open_future(self, protocol_open_receipt_hash: str) -> ProtocolSpec:
        """Return a receipt-bound opened protocol; the original remains closed."""
        return ProtocolSpec(
            windows=self.windows,
            future_opened=True,
            protocol_open_receipt_hash=protocol_open_receipt_hash,
            protocol_version=self.protocol_version,
        )

    def assert_manifest_opening(self, manifest: object) -> None:
        """Cross-check a future RunManifest against this exact opened protocol."""
        if getattr(manifest, "protocol_version", None) != self.protocol_version:
            raise ContractError("manifest and protocol versions differ")
        split = getattr(manifest, "split", None)
        if split is not Partition.FUTURE_UNSEEN:
            return
        if not self.future_opened:
            raise ContractError("future manifest cannot rank against a closed protocol")
        if getattr(manifest, "protocol_open_receipt_hash", None) != self.protocol_open_receipt_hash:
            raise ContractError("manifest opening receipt does not match ProtocolSpec")
