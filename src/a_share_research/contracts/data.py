"""Canonical point-in-time data rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import ClassVar

from a_share_research.contracts.base import (
    CanonicalModel,
    ContractError,
    canonical_hash,
    require_finite,
    require_nonnegative,
)

_TS_CODE = re.compile(r"^[0-9]{6}\.(SH|SZ|BJ)$")


class FeatureGroup(str, Enum):
    CORE = "CORE"
    FINANCIAL = "FINANCIAL"
    VALUATION = "VALUATION"
    MARKET_STATE = "MARKET_STATE"


@dataclass(frozen=True)
class FormalFeatureManifest(CanonicalModel):
    """D0-anchored receipt proving every formal input was eligible."""

    SCHEMA_NAME: ClassVar[str] = "formal_feature_manifest"

    dataset_id: str
    d0_manifest_hash: str
    feature_eligibility: dict[str, bool]

    def validate(self) -> None:
        if not self.dataset_id or not self.feature_eligibility:
            raise ContractError("dataset_id and feature eligibility are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.d0_manifest_hash):
            raise ContractError("d0_manifest_hash must be SHA-256")
        if any(not name for name in self.feature_eligibility):
            raise ContractError("feature names cannot be empty")
        if any(type(value) is not bool for value in self.feature_eligibility.values()):
            raise ContractError("formal feature eligibility values must be booleans")

    def require_formal_eligible(self) -> str:
        ineligible = sorted(
            name for name, eligible in self.feature_eligibility.items() if not eligible
        )
        if ineligible:
            raise ContractError(f"formal feature manifest contains ineligible inputs: {ineligible}")
        return self.stable_hash()


def _validate_ts_code(ts_code: str) -> None:
    if not _TS_CODE.fullmatch(ts_code):
        raise ContractError(f"invalid permanent security identity: {ts_code!r}")


@dataclass(frozen=True)
class SecurityMaster(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "security_master"

    ts_code: str
    list_date: date
    delist_date: date | None
    board: str
    industry: str
    identity_version: str = "ts_code-v1"

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.delist_date is not None and self.delist_date < self.list_date:
            raise ContractError("delist_date cannot precede list_date")
        if not self.board or not self.industry or not self.identity_version:
            raise ContractError("board, industry and identity_version are required")


@dataclass(frozen=True)
class UniverseMembership(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "universe_membership"

    asof_date: date
    ts_code: str
    universe: str
    effective_from: date
    effective_to: date | None
    source: str

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.asof_date < self.effective_from:
            raise ContractError("membership used before effective_from")
        if self.effective_to is not None:
            if self.effective_to < self.effective_from:
                raise ContractError("effective_to cannot precede effective_from")
            if self.asof_date > self.effective_to:
                raise ContractError("membership used after effective_to")
        if not self.universe or not self.source:
            raise ContractError("universe and source are required")


@dataclass(frozen=True)
class DailyMarket(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "market_daily"

    trade_date: date
    ts_code: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover: float | None
    adj_factor: float
    up_limit: float | None
    down_limit: float | None
    suspended: bool
    st_state: bool

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        for name in ("open", "high", "low", "close", "volume", "amount", "adj_factor"):
            require_nonnegative(getattr(self, name), name)
        if self.low > min(self.open, self.close, self.high):
            raise ContractError("low is inconsistent with OHLC")
        if self.high < max(self.open, self.close, self.low):
            raise ContractError("high is inconsistent with OHLC")
        for name in ("turnover", "up_limit", "down_limit"):
            value = getattr(self, name)
            if value is not None:
                require_nonnegative(value, name)
        if self.up_limit is not None and self.down_limit is not None:
            if self.up_limit < self.down_limit:
                raise ContractError("up_limit cannot be below down_limit")


@dataclass(frozen=True)
class PITFeature(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "pit_feature"

    asof_date: date
    ts_code: str
    feature_name: str
    feature_group: FeatureGroup
    value: float | None
    source_date: date
    announce_time: datetime | None
    availability_time: datetime
    signal_cutoff_time: datetime
    missing_flag: bool
    source: str
    formal_eligible: bool = True

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if not self.feature_name or not self.source:
            raise ContractError("feature_name and source are required")
        for name in ("availability_time", "signal_cutoff_time"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ContractError(f"{name} must be timezone-aware")
        if self.source_date > self.signal_cutoff_time.date():
            raise ContractError("future source_date contaminates PIT feature")
        if self.asof_date != self.signal_cutoff_time.date():
            raise ContractError("asof_date must match signal_cutoff_time date")
        if self.availability_time > self.signal_cutoff_time:
            raise ContractError("feature became available after the exact signal cutoff")
        if self.announce_time is not None:
            if self.announce_time.tzinfo is None or self.announce_time.utcoffset() is None:
                raise ContractError("announce_time must be timezone-aware")
            if self.announce_time > self.availability_time:
                raise ContractError("availability cannot precede announcement")
        if not isinstance(self.feature_group, FeatureGroup):
            raise ContractError("feature_group must use FeatureGroup")
        formal_announcement_required = self.feature_group in {
            FeatureGroup.FINANCIAL,
            FeatureGroup.VALUATION,
        }
        if self.formal_eligible and formal_announcement_required and self.announce_time is None:
            raise ContractError("formal financial or valuation feature requires announce_time")
        if self.missing_flag != (self.value is None):
            raise ContractError("missing_flag must independently describe this feature value")
        if self.value is not None:
            require_finite(self.value, "feature value")


@dataclass(frozen=True)
class MarketState(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "market_state"

    asof_date: date
    feature_name: str
    value: float
    source_universe: str = "CSI300"
    source_hash: str = ""

    def validate(self) -> None:
        if not self.feature_name:
            raise ContractError("market-state feature_name is required")
        if self.source_universe != "CSI300":
            raise ContractError("shared market state must be sourced from CSI300")
        require_finite(self.value, "market-state value")
        if self.source_hash and len(self.source_hash) != 64:
            raise ContractError("source_hash must be SHA-256 when supplied")


@dataclass(frozen=True)
class Label(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "label"

    signal_date: date
    ts_code: str
    horizon: int
    entry_date: date
    exit_date: date
    open_to_open_return: float
    benchmark_return: float
    trading_calendar: tuple[date, ...]
    trading_calendar_hash: str

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if self.horizon not in {1, 5, 20}:
            raise ContractError("horizon must be one of 1, 5 or 20 trading days")
        if self.entry_date <= self.signal_date:
            raise ContractError("T+1 entry must occur after signal_date")
        if self.exit_date <= self.entry_date:
            raise ContractError("exit_date must occur after entry_date")
        if tuple(sorted(set(self.trading_calendar))) != self.trading_calendar:
            raise ContractError("trading_calendar must be unique and increasing")
        if self.trading_calendar_hash != canonical_hash(self.trading_calendar):
            raise ContractError("trading_calendar_hash does not match calendar evidence")
        try:
            signal_index = self.trading_calendar.index(self.signal_date)
        except ValueError as error:
            raise ContractError("signal_date is absent from trading calendar evidence") from error
        entry_index = signal_index + 1
        exit_index = entry_index + self.horizon
        if exit_index >= len(self.trading_calendar):
            raise ContractError("trading calendar evidence does not cover label exit")
        if self.entry_date != self.trading_calendar[entry_index]:
            raise ContractError("entry_date must be the next trading day")
        if self.exit_date != self.trading_calendar[exit_index]:
            raise ContractError("exit_date must be horizon trading days after entry")
        require_finite(self.open_to_open_return, "open_to_open_return")
        require_finite(self.benchmark_return, "benchmark_return")

    @property
    def relative_return(self) -> float:
        return self.open_to_open_return - self.benchmark_return


@dataclass(frozen=True)
class Eligibility(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "eligibility"

    signal_date: date
    ts_code: str
    universe: str
    member: bool
    observed: bool
    tradable: bool
    complete: bool

    def validate(self) -> None:
        _validate_ts_code(self.ts_code)
        if not self.universe:
            raise ContractError("universe is required")
        if self.tradable and not (self.member and self.observed):
            raise ContractError("tradable cannot be true outside observed membership")
        if self.complete and not self.observed:
            raise ContractError("complete cannot be true for an unobserved row")
