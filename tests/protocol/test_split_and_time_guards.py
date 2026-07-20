"""Temporal leakage guards; execute only on the approved server."""

from datetime import date, datetime, timedelta, timezone

import pytest

from a_share_research.contracts import ContractError, FeatureGroup, PITFeature
from a_share_research.protocol import (
    ExperimentRegistry,
    ProtocolSpec,
    Purpose,
    RegisteredExperiment,
    embargoed_dates,
    hash_rows_asof,
    purged_training_dates,
)

HASH = "b" * 64


def test_viewed_legacy_interval_cannot_fit_or_select() -> None:
    protocol = ProtocolSpec.research_v1()
    viewed_date = date(2026, 3, 2)
    with pytest.raises(ContractError, match="forbidden"):
        protocol.assert_access(viewed_date, Purpose.FIT)
    with pytest.raises(ContractError, match="forbidden"):
        protocol.assert_access(viewed_date, Purpose.SELECT)
    protocol.assert_access(viewed_date, Purpose.LEGACY_REPORT)


def test_future_interval_is_closed_until_protocol_is_explicitly_opened() -> None:
    protocol = ProtocolSpec.research_v1()
    with pytest.raises(ContractError, match="forbidden"):
        protocol.assert_access(date(2026, 7, 20), Purpose.FINAL_EVALUATION)
    opened = protocol.open_future("f" * 64)
    opened.assert_access(date(2026, 7, 20), Purpose.FINAL_EVALUATION)


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_purge_removes_training_labels_that_cross_validation(horizon: int) -> None:
    start = date(2024, 11, 1)
    trading_dates = tuple(start + timedelta(days=offset) for offset in range(90))
    validation_start = date(2025, 1, 1)
    purged = purged_training_dates(trading_dates, validation_start, horizon)
    boundary = trading_dates.index(validation_start)
    assert trading_dates.index(purged[-1]) + 1 + horizon < boundary


@pytest.mark.parametrize("embargo", [1, 5, 20])
def test_embargo_identifies_exactly_the_next_trading_steps(embargo: int) -> None:
    start = date(2025, 1, 1)
    trading_dates = tuple(start + timedelta(days=offset) for offset in range(40))
    blocked = embargoed_dates(trading_dates, trading_dates[5], embargo)
    assert blocked == trading_dates[6 : 6 + embargo]


def test_future_perturbation_cannot_change_past_content_hash() -> None:
    past = PITFeature(
        asof_date=date(2025, 1, 2),
        ts_code="000001.SZ",
        feature_name="pe_ttm",
        feature_group=FeatureGroup.VALUATION,
        value=10.0,
        source_date=date(2025, 1, 2),
        announce_time=datetime(2025, 1, 2, 8, tzinfo=timezone.utc),
        availability_time=datetime(2025, 1, 2, 8, tzinfo=timezone.utc),
        signal_cutoff_time=datetime(2025, 1, 2, 16, tzinfo=timezone.utc),
        missing_flag=False,
        source="synthetic",
    )
    future_a = PITFeature(
        asof_date=date(2025, 1, 3),
        ts_code="000001.SZ",
        feature_name="pe_ttm",
        feature_group=FeatureGroup.VALUATION,
        value=11.0,
        source_date=date(2025, 1, 3),
        announce_time=datetime(2025, 1, 3, 8, tzinfo=timezone.utc),
        availability_time=datetime(2025, 1, 3, 8, tzinfo=timezone.utc),
        signal_cutoff_time=datetime(2025, 1, 3, 16, tzinfo=timezone.utc),
        missing_flag=False,
        source="synthetic",
    )
    future_b = PITFeature(
        asof_date=date(2025, 1, 3),
        ts_code="000001.SZ",
        feature_name="pe_ttm",
        feature_group=FeatureGroup.VALUATION,
        value=-999.0,
        source_date=date(2025, 1, 3),
        announce_time=datetime(2025, 1, 3, 8, tzinfo=timezone.utc),
        availability_time=datetime(2025, 1, 3, 8, tzinfo=timezone.utc),
        signal_cutoff_time=datetime(2025, 1, 3, 16, tzinfo=timezone.utc),
        missing_flag=False,
        source="synthetic-perturbed",
    )
    cutoff = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    assert hash_rows_asof((past, future_a), cutoff) == hash_rows_asof((past, future_b), cutoff)
    assert hash_rows_asof((past, future_a), cutoff) == hash_rows_asof((future_a, past), cutoff)


def test_sealed_registry_rejects_new_or_changed_experiments() -> None:
    registry = ExperimentRegistry()
    registry.register(RegisteredExperiment("ridge-csi300-a0", "Ridge", "CSI300", "A0", HASH))
    receipt = registry.seal()
    assert len(receipt) == 64
    registry.assert_registered("ridge-csi300-a0", HASH)
    with pytest.raises(ContractError, match="sealed"):
        registry.register(
            RegisteredExperiment("fact-csi300-a0", "FACT", "CSI300", "A0", HASH)
        )
    with pytest.raises(ContractError, match="not in the sealed registry"):
        registry.assert_registered("ridge-csi300-a0", "c" * 64)


def test_registry_public_snapshot_is_immutable_and_private_tamper_is_detected() -> None:
    registry = ExperimentRegistry()
    registered = RegisteredExperiment("ridge-csi300-a0", "Ridge", "CSI300", "A0", HASH)
    registry.register(registered)
    registry.seal()
    with pytest.raises(TypeError):
        registry.experiments["injected"] = registered  # type: ignore[index]
    registry._experiments["injected"] = registered
    with pytest.raises(ContractError, match="was mutated"):
        registry.assert_registered("ridge-csi300-a0", HASH)


def test_exact_intraday_availability_after_cutoff_is_excluded() -> None:
    cutoff = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    with pytest.raises(ContractError, match="exact signal cutoff"):
        PITFeature(
            asof_date=date(2025, 1, 2),
            ts_code="000001.SZ",
            feature_name="momentum_20d",
            feature_group=FeatureGroup.CORE,
            value=0.1,
            source_date=date(2025, 1, 2),
            announce_time=None,
            availability_time=datetime(2025, 1, 2, 16, 1, tzinfo=timezone.utc),
            signal_cutoff_time=cutoff,
            missing_flag=False,
            source="synthetic",
        )
