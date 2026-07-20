"""PIT financial/valuation factors with conservative availability and null truth."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime

from a_share_research.contracts import ContractError, PITFeature
from a_share_research.features.availability import exact_or_next_trade_availability, signal_cutoff
from a_share_research.features.builders import build_feature_row
from a_share_research.features.schema import InformationClass, d0_features


def _numeric(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ContractError("factor source value is not numeric") from error


def build_fundamental_features(
    *,
    asof_date: date,
    ts_code: str,
    trading_dates: tuple[date, ...],
    daily_basic: Mapping[str, object] | None,
    daily_basic_source_date: date | None,
    financial: Mapping[str, object] | None,
    financial_announcement_date: date | None,
    financial_announcement_time: datetime | None,
    daily_basic_publish_time: datetime | None,
    industry_id: float | None = None,
    industry_effective_date: date | None = None,
    industry_source_date: date | None = None,
) -> tuple[PITFeature, ...]:
    """Absent or not-yet-announced values remain individually missing."""
    cutoff = signal_cutoff(asof_date)
    rows: list[PITFeature] = []
    fundamental_definitions = (
        item for item in d0_features() if item.information_class is InformationClass.F
    )
    for definition in fundamental_definitions:
        if definition.name == "industry_id":
            industry_known = industry_id is not None and industry_effective_date is not None
            effective_date = industry_effective_date or asof_date
            source_date = industry_source_date or effective_date
            available = signal_cutoff(effective_date)
            rows.append(
                build_feature_row(
                    definition,
                    asof_date=asof_date,
                    ts_code=ts_code,
                    value=(
                        industry_id
                        if industry_known and effective_date <= asof_date
                        else None
                    ),
                    source_date=min(source_date, asof_date),
                    announce_time=available if industry_known else None,
                    availability_time=available if industry_known else cutoff,
                    signal_cutoff_time=cutoff,
                    source="index_member_all_sw_l1_pit_v1",
                    formal_eligible=industry_known and effective_date <= asof_date,
                )
            )
            continue
        if definition.endpoint == "daily_basic":
            source_row = daily_basic
            source_date = daily_basic_source_date or asof_date
            exact_time = daily_basic_publish_time
            known_source = source_row is not None
            formal_source = known_source and daily_basic_source_date is not None
        else:
            source_row = financial
            source_date = financial_announcement_date or asof_date
            exact_time = financial_announcement_time
            known_source = source_row is not None and financial_announcement_date is not None
            formal_source = known_source
        if known_source:
            available = exact_or_next_trade_availability(
                source_date=source_date,
                trading_dates=trading_dates,
                exact_time=exact_time,
            )
            announcement = exact_time or available
        else:
            available = cutoff
            announcement = None
        formal = formal_source
        if available > cutoff or not known_source:
            value = None
            formal = False
            available = cutoff
            announcement = None
        else:
            value = _numeric(source_row.get(definition.source_field)) if source_row else None
        rows.append(
            build_feature_row(
                definition,
                asof_date=asof_date,
                ts_code=ts_code,
                value=value,
                source_date=min(source_date, asof_date),
                announce_time=announcement,
                availability_time=available,
                signal_cutoff_time=cutoff,
                source=definition.endpoint,
                formal_eligible=formal,
            )
        )
    return tuple(rows)
