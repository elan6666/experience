"""T+1 open-to-open labels for 1/5/20 trading-day horizons."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

from a_share_research.contracts import CanonicalModel, ContractError, Label, canonical_hash


@dataclass(frozen=True)
class CompactLabel(CanonicalModel):
    """Calendar-normalized label row suitable for a large canonical panel."""

    SCHEMA_NAME: ClassVar[str] = "compact_label"

    signal_date: date
    ts_code: str
    horizon: int
    entry_date: date
    exit_date: date
    open_to_open_return: float
    benchmark_return: float
    trading_calendar_hash: str
    signal_index: int
    entry_index: int
    exit_index: int

    def validate(self) -> None:
        if self.horizon not in {1, 5, 20}:
            raise ContractError("unsupported compact label horizon")
        if self.entry_index != self.signal_index + 1:
            raise ContractError("compact label entry is not T+1")
        if self.exit_index != self.entry_index + self.horizon:
            raise ContractError("compact label exit index does not match horizon")
        if not self.signal_date < self.entry_date < self.exit_date:
            raise ContractError("compact label dates are not strictly increasing")
        if len(self.trading_calendar_hash) != 64:
            raise ContractError("compact label calendar hash must be SHA-256")
        if not math.isfinite(self.open_to_open_return) or not math.isfinite(
            self.benchmark_return
        ):
            raise ContractError("compact label returns must be finite")

    def verify_calendar(self, trading_calendar: tuple[date, ...]) -> None:
        if canonical_hash(trading_calendar) != self.trading_calendar_hash:
            raise ContractError("compact label calendar receipt mismatch")
        if self.exit_index >= len(trading_calendar):
            raise ContractError("compact label exits beyond calendar")
        if trading_calendar[self.signal_index] != self.signal_date:
            raise ContractError("compact label signal index mismatch")
        if trading_calendar[self.entry_index] != self.entry_date:
            raise ContractError("compact label entry index mismatch")
        if trading_calendar[self.exit_index] != self.exit_date:
            raise ContractError("compact label exit index mismatch")


def _log_return(start: float, end: float) -> float:
    if start <= 0 or end <= 0:
        raise ContractError("label opens must be strictly positive")
    return math.log(end / start)


def build_open_to_open_labels(
    *,
    ts_code: str,
    signal_dates: tuple[date, ...],
    trading_calendar: tuple[date, ...],
    opens: Mapping[date, float],
    benchmark_opens: Mapping[date, float],
    horizons: tuple[int, ...] = (1, 5, 20),
) -> tuple[Label, ...]:
    """Generate only labels fully covered by the supplied immutable calendar."""
    if tuple(sorted(set(trading_calendar))) != trading_calendar:
        raise ContractError("trading calendar must be unique and increasing")
    calendar_hash = canonical_hash(trading_calendar)
    labels: list[Label] = []
    for signal_date in signal_dates:
        if signal_date not in trading_calendar:
            raise ContractError("signal date is absent from trading calendar")
        signal_index = trading_calendar.index(signal_date)
        entry_index = signal_index + 1
        for horizon in horizons:
            if horizon not in {1, 5, 20}:
                raise ContractError("unsupported D0 label horizon")
            exit_index = entry_index + horizon
            if exit_index >= len(trading_calendar):
                continue
            entry_date = trading_calendar[entry_index]
            exit_date = trading_calendar[exit_index]
            if any(day not in opens for day in (entry_date, exit_date)):
                continue
            if any(day not in benchmark_opens for day in (entry_date, exit_date)):
                continue
            labels.append(
                Label(
                    signal_date=signal_date,
                    ts_code=ts_code,
                    horizon=horizon,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    open_to_open_return=_log_return(opens[entry_date], opens[exit_date]),
                    benchmark_return=_log_return(
                        benchmark_opens[entry_date], benchmark_opens[exit_date]
                    ),
                    trading_calendar=trading_calendar,
                    trading_calendar_hash=calendar_hash,
                )
            )
    return tuple(labels)


def compact_label(label: Label) -> CompactLabel:
    signal_index = label.trading_calendar.index(label.signal_date)
    row = CompactLabel(
        signal_date=label.signal_date,
        ts_code=label.ts_code,
        horizon=label.horizon,
        entry_date=label.entry_date,
        exit_date=label.exit_date,
        open_to_open_return=label.open_to_open_return,
        benchmark_return=label.benchmark_return,
        trading_calendar_hash=label.trading_calendar_hash,
        signal_index=signal_index,
        entry_index=signal_index + 1,
        exit_index=signal_index + 1 + label.horizon,
    )
    row.verify_calendar(label.trading_calendar)
    return row


def build_compact_open_labels(
    *,
    ts_code: str,
    signal_index: int,
    trading_calendar: tuple[date, ...],
    trading_calendar_hash: str,
    opens: Mapping[date, float],
    benchmark_opens: Mapping[date, float],
    horizons: tuple[int, ...] = (1, 5, 20),
) -> tuple[CompactLabel, ...]:
    """Build large-panel labels without copying the calendar into every row."""
    if len(trading_calendar_hash) != 64:
        raise ContractError("precomputed trading calendar hash must be SHA-256")
    if signal_index < 0 or signal_index >= len(trading_calendar):
        raise ContractError("signal index is outside the trading calendar")
    rows: list[CompactLabel] = []
    entry_index = signal_index + 1
    for horizon in horizons:
        if horizon not in {1, 5, 20}:
            raise ContractError("unsupported D0 label horizon")
        exit_index = entry_index + horizon
        if exit_index >= len(trading_calendar):
            continue
        entry_date = trading_calendar[entry_index]
        exit_date = trading_calendar[exit_index]
        if entry_date not in opens or exit_date not in opens:
            continue
        if entry_date not in benchmark_opens or exit_date not in benchmark_opens:
            continue
        rows.append(
            CompactLabel(
                signal_date=trading_calendar[signal_index],
                ts_code=ts_code,
                horizon=horizon,
                entry_date=entry_date,
                exit_date=exit_date,
                open_to_open_return=_log_return(opens[entry_date], opens[exit_date]),
                benchmark_return=_log_return(
                    benchmark_opens[entry_date], benchmark_opens[exit_date]
                ),
                trading_calendar_hash=trading_calendar_hash,
                signal_index=signal_index,
                entry_index=entry_index,
                exit_index=exit_index,
            )
        )
    return tuple(rows)
