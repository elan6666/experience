"""Fail-closed D0 quality gates and compact audit summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from a_share_research.contracts import Label, PITFeature, UniverseMembership
from a_share_research.data.labels import CompactLabel
from a_share_research.data.manifest import UniverseGate
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState


def duplicate_count(rows: Iterable[object], key_fields: tuple[str, ...]) -> int:
    counts = Counter(tuple(getattr(row, field) for field in key_fields) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def pit_violation_count(rows: Iterable[PITFeature]) -> int:
    violations = 0
    for row in rows:
        if row.availability_time > row.signal_cutoff_time:
            violations += 1
        if row.source_date > row.signal_cutoff_time.date():
            violations += 1
        if row.announce_time is not None and row.announce_time > row.availability_time:
            violations += 1
        if row.missing_flag != (row.value is None):
            violations += 1
    return violations


def label_boundary_violation_count(rows: Iterable[Label | CompactLabel]) -> int:
    violations = 0
    for row in rows:
        if isinstance(row, CompactLabel):
            if row.entry_index != row.signal_index + 1:
                violations += 1
            if row.exit_index != row.entry_index + row.horizon:
                violations += 1
            continue
        try:
            signal_index = row.trading_calendar.index(row.signal_date)
        except ValueError:
            violations += 1
            continue
        if signal_index + 1 + row.horizon >= len(row.trading_calendar):
            violations += 1
            continue
        if row.entry_date != row.trading_calendar[signal_index + 1]:
            violations += 1
        if row.exit_date != row.trading_calendar[signal_index + 1 + row.horizon]:
            violations += 1
    return violations


def assess_universe_gate(
    *,
    universe: UniverseClass,
    memberships: tuple[UniverseMembership, ...],
    features: tuple[PITFeature, ...],
    labels: tuple[Label | CompactLabel, ...],
    expected_member_dates: int,
    expected_core_values: int,
    star50_history_complete: bool = True,
    expected_feature_names: tuple[str, ...] | None = None,
) -> UniverseGate:
    membership_duplicates = duplicate_count(memberships, ("asof_date", "ts_code", "universe"))
    feature_duplicates = duplicate_count(features, ("asof_date", "ts_code", "feature_name"))
    label_duplicates = duplicate_count(labels, ("signal_date", "ts_code", "horizon"))
    duplicates = membership_duplicates + feature_duplicates + label_duplicates
    pit_violations = pit_violation_count(features)
    label_violations = label_boundary_violation_count(labels)
    feature_schema_violations = 0
    if expected_feature_names is not None:
        expected = set(expected_feature_names)
        by_key: dict[tuple[object, object], set[str]] = {}
        for row in features:
            by_key.setdefault((row.asof_date, row.ts_code), set()).add(row.feature_name)
        # Features are materialized only on registered signal dates.  Daily
        # membership rows outside those dates must not be mistaken for missing
        # feature panels.
        for actual in by_key.values():
            feature_schema_violations += len(expected.symmetric_difference(actual))
    core_present = sum(
        row.feature_group.value == "CORE" and not row.missing_flag for row in features
    )
    membership_dates = len({row.asof_date for row in memberships})
    membership_coverage = min(1.0, membership_dates / max(expected_member_dates, 1))
    core_coverage = min(1.0, core_present / max(expected_core_values, 1))
    warnings: list[str] = []
    if duplicates or pit_violations or label_violations or feature_schema_violations:
        status = ResultState.INVALID_DATA
    elif universe is UniverseClass.STAR50 and not star50_history_complete:
        status = ResultState.BLOCKED
        warnings.append("official STAR50 historical membership is incomplete")
    elif membership_coverage < 1.0 or core_coverage < 0.995:
        status = ResultState.BLOCKED
        warnings.append("membership/core coverage is below the formal D0 threshold")
    elif universe in {UniverseClass.TECH32, UniverseClass.TECH90}:
        status = ResultState.EXPLORATORY_ONLY
        warnings.append("2026-selected universe; conditional-selection bias cannot be removed")
    else:
        status = ResultState.PASS
    return UniverseGate(
        universe=universe,
        status=status,
        membership_coverage=membership_coverage,
        core_coverage=core_coverage,
        duplicate_keys=duplicates,
        pit_violations=pit_violations,
        label_boundary_violations=label_violations,
        feature_schema_violations=feature_schema_violations,
        warnings=tuple(warnings),
    )
