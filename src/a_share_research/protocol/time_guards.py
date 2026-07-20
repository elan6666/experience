"""PIT availability, purge and future-perturbation guards."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from a_share_research.contracts.base import ContractError, canonical_hash, canonical_json


def assert_no_future_availability(
    *,
    source_date: date,
    announce_time: datetime | None,
    availability_time: datetime,
    signal_cutoff_time: datetime,
    formal_announcement_required: bool = False,
) -> None:
    for name, value in (
        ("availability_time", availability_time),
        ("signal_cutoff_time", signal_cutoff_time),
    ):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ContractError(f"{name} must be timezone-aware")
    if source_date > signal_cutoff_time.date():
        raise ContractError("source_date is in the future")
    if availability_time > signal_cutoff_time:
        raise ContractError("data is unavailable at exact signal cutoff")
    if formal_announcement_required and announce_time is None:
        raise ContractError("formal feature requires announcement evidence")
    if announce_time is not None:
        if announce_time.tzinfo is None or announce_time.utcoffset() is None:
            raise ContractError("announce_time must be timezone-aware")
        if announce_time > availability_time:
            raise ContractError("availability cannot precede announcement")


def purged_training_dates(
    trading_dates: Sequence[date],
    validation_start: date,
    horizon_steps: int,
) -> tuple[date, ...]:
    """Remove training signals whose future label touches validation."""
    if horizon_steps < 1:
        raise ContractError("horizon_steps must be positive")
    if list(trading_dates) != sorted(set(trading_dates)):
        raise ContractError("trading_dates must be unique and increasing")
    try:
        boundary_index = trading_dates.index(validation_start)
    except ValueError as error:
        raise ContractError("validation_start must be a registered trading date") from error
    # The label enters at T+1 and exits ``horizon_steps`` trading days later.
    # Therefore signal + 1 + horizon must be strictly before validation.
    last_allowed_index = boundary_index - horizon_steps - 2
    return tuple(trading_dates[: max(0, last_allowed_index + 1)])


def embargoed_dates(
    trading_dates: Sequence[date],
    evaluation_end: date,
    embargo_steps: int,
) -> tuple[date, ...]:
    """Identify post-evaluation dates forbidden to the next fitting fold."""
    if embargo_steps < 0:
        raise ContractError("embargo_steps cannot be negative")
    if list(trading_dates) != sorted(set(trading_dates)):
        raise ContractError("trading_dates must be unique and increasing")
    try:
        end_index = trading_dates.index(evaluation_end)
    except ValueError as error:
        raise ContractError("evaluation_end must be a registered trading date") from error
    return tuple(trading_dates[end_index + 1 : end_index + 1 + embargo_steps])


def _availability_time(row: Any) -> datetime:
    value = getattr(row, "availability_time", None)
    if value is None and isinstance(row, dict):
        value = row.get("availability_time")
    if not isinstance(value, datetime):
        raise ContractError("row has no exact availability_time")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError("row availability_time must be timezone-aware")
    return value


def hash_rows_asof(rows: Iterable[Any], cutoff: datetime) -> str:
    """Hash causally available rows in stable content order."""
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ContractError("cutoff must be timezone-aware")
    past = [
        row.to_dict() if hasattr(row, "to_dict") else row
        for row in rows
        if _availability_time(row) <= cutoff
    ]
    return canonical_hash(sorted(past, key=canonical_json))
