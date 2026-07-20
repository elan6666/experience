"""Normalize provider rows into canonical, unadjusted D0 market objects."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from a_share_research.contracts import ContractError, DailyMarket, SecurityMaster


def parse_provider_date(value: object) -> date:
    text = str(value).replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ContractError(f"invalid provider date: {value!r}")
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def _float(row: Mapping[str, object], name: str, *, optional: bool = False) -> float | None:
    value = row.get(name)
    if value in (None, ""):
        if optional:
            return None
        raise ContractError(f"required market field is missing: {name}")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"market field is not numeric: {name}") from error


def normalize_security_master(row: Mapping[str, object]) -> SecurityMaster:
    list_date = parse_provider_date(row.get("list_date"))
    delist_value = row.get("delist_date")
    return SecurityMaster(
        ts_code=str(row.get("ts_code", "")),
        list_date=list_date,
        delist_date=parse_provider_date(delist_value) if delist_value not in (None, "") else None,
        board=str(row.get("market") or row.get("board") or "UNKNOWN"),
        industry=str(row.get("industry") or "UNKNOWN"),
    )


def normalize_daily_market(
    daily: Mapping[str, object],
    *,
    daily_basic: Mapping[str, object] | None,
    adjustment: Mapping[str, object],
    limits: Mapping[str, object] | None,
    suspended: bool,
    st_state: bool,
) -> DailyMarket:
    """Preserve raw OHLC for labels/execution; adj_factor is evidence, not a future adjustment."""
    return DailyMarket(
        trade_date=parse_provider_date(daily.get("trade_date")),
        ts_code=str(daily.get("ts_code", "")),
        open=float(_float(daily, "open")),
        high=float(_float(daily, "high")),
        low=float(_float(daily, "low")),
        close=float(_float(daily, "close")),
        volume=float(_float(daily, "vol")),
        amount=float(_float(daily, "amount")),
        turnover=(
            _float(daily_basic, "turnover_rate", optional=True)
            if daily_basic is not None
            else None
        ),
        adj_factor=float(_float(adjustment, "adj_factor")),
        up_limit=_float(limits, "up_limit", optional=True) if limits is not None else None,
        down_limit=_float(limits, "down_limit", optional=True) if limits is not None else None,
        suspended=suspended,
        st_state=st_state,
    )

