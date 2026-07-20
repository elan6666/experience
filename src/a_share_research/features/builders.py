"""Build raw PIT feature rows and one missing marker per feature."""

from __future__ import annotations

from datetime import date, datetime

from a_share_research.contracts import ContractError, PITFeature
from a_share_research.features.schema import FeatureDefinition


def build_feature_row(
    definition: FeatureDefinition,
    *,
    asof_date: date,
    ts_code: str,
    value: float | None,
    source_date: date,
    announce_time: datetime | None,
    availability_time: datetime,
    signal_cutoff_time: datetime,
    source: str,
    formal_eligible: bool = True,
) -> PITFeature:
    """No fill or transform is allowed here; ``None`` is retained as truth."""
    return PITFeature(
        asof_date=asof_date,
        ts_code=ts_code,
        feature_name=definition.name,
        feature_group=definition.contract_group,
        value=value,
        source_date=source_date,
        announce_time=announce_time,
        availability_time=availability_time,
        signal_cutoff_time=signal_cutoff_time,
        missing_flag=value is None,
        source=source,
        formal_eligible=formal_eligible,
    )


def per_feature_missing(
    feature_rows: tuple[PITFeature, ...],
    *,
    expected_features: tuple[str, ...],
) -> dict[str, bool]:
    by_name: dict[str, PITFeature] = {}
    for row in feature_rows:
        if row.feature_name in by_name:
            raise ContractError("duplicate feature row for one asset/date")
        by_name[row.feature_name] = row
    unknown = set(by_name) - set(expected_features)
    if unknown:
        raise ContractError(f"unexpected D0 features: {sorted(unknown)}")
    return {
        name: name not in by_name or by_name[name].missing_flag
        for name in expected_features
    }
