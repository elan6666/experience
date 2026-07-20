from datetime import date, datetime, timezone

import pytest

from a_share_research.contracts import (
    ContractError,
    CoverageState,
    FormalFeatureManifest,
    Label,
    PredictionFrame,
    PredictionRecord,
    RunManifest,
    canonical_hash,
)
from a_share_research.evaluation import (
    EvaluationFrequency,
    OutcomeMode,
    SupportMode,
    evaluate_formal_predictions,
    evaluate_predictions,
    freeze_common_support,
)
from a_share_research.protocol import Partition, ProtocolSpec, Purpose, UniverseClass
from a_share_research.quality import ResultState

CALENDAR = (
    date(2025, 1, 2),
    date(2025, 1, 3),
    date(2025, 1, 6),
    date(2025, 1, 7),
    date(2025, 1, 8),
    date(2025, 1, 9),
    date(2025, 1, 10),
    date(2025, 1, 13),
    date(2025, 1, 14),
)


def _label(signal_date: date, code: str, value: float, benchmark: float = 0.0) -> Label:
    index = CALENDAR.index(signal_date)
    return Label(
        signal_date=signal_date,
        ts_code=code,
        horizon=5,
        entry_date=CALENDAR[index + 1],
        exit_date=CALENDAR[index + 6],
        open_to_open_return=value,
        benchmark_return=benchmark,
        trading_calendar=CALENDAR,
        trading_calendar_hash=canonical_hash(CALENDAR),
    )


def _frame(run_id: str, rows: list[tuple[date, str, float | None]]) -> PredictionFrame:
    return PredictionFrame(
        run_id=run_id,
        records=tuple(
            PredictionRecord(
                signal_date=signal_date,
                ts_code=code,
                score=score,
                coverage_state=(
                    CoverageState.SCORED
                    if score is not None
                    else CoverageState.INSUFFICIENT_HISTORY
                ),
            )
            for signal_date, code, score in rows
        ),
    )


def test_rank_ic_errors_and_monotonic_groups_are_hand_calculable() -> None:
    signal_date = CALENDAR[0]
    codes = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")
    frame = _frame(
        "perfect",
        [(signal_date, code, float(index)) for index, code in enumerate(codes)],
    )
    labels = tuple(
        _label(signal_date, code, float(index)) for index, code in enumerate(codes)
    )
    keys = {(signal_date, code) for code in codes}
    scorecard = evaluate_predictions(
        predictions=frame,
        labels=labels,
        frequency=EvaluationFrequency.WEEKLY,
        support=SupportMode.COMMON,
        outcome=OutcomeMode.ABSOLUTE,
        eligible_keys=keys,
        common_support_keys=keys,
        group_count=2,
    )
    assert scorecard.rank_ic == pytest.approx(1.0)
    assert scorecard.mae == 0
    assert scorecard.rmse == 0
    assert scorecard.sign_accuracy == 1
    assert scorecard.group_returns == pytest.approx((0.5, 2.5))
    assert scorecard.monotone_fraction == 1


def test_constant_cross_section_is_explicitly_excluded_from_rank_ic() -> None:
    signal_date = CALENDAR[0]
    frame = _frame(
        "constant",
        [(signal_date, "000001.SZ", 1.0), (signal_date, "000002.SZ", 1.0)],
    )
    labels = (
        _label(signal_date, "000001.SZ", -1.0),
        _label(signal_date, "000002.SZ", 1.0),
    )
    keys = {(row.signal_date, row.ts_code) for row in labels}
    scorecard = evaluate_predictions(
        predictions=frame,
        labels=labels,
        frequency=EvaluationFrequency.WEEKLY,
        support=SupportMode.NATIVE,
        outcome=OutcomeMode.ABSOLUTE,
        eligible_keys=keys,
    )
    assert scorecard.rank_ic is None
    assert scorecard.icir is None
    assert scorecard.excluded_constant_dates == 1


def test_common_support_is_frozen_intersection_and_required() -> None:
    signal_date = CALENDAR[0]
    keys = {(signal_date, "000001.SZ"), (signal_date, "000002.SZ")}
    first = _frame(
        "first",
        [(signal_date, "000001.SZ", 1.0), (signal_date, "000002.SZ", 2.0)],
    )
    second = _frame(
        "second",
        [(signal_date, "000001.SZ", 1.0), (signal_date, "000002.SZ", None)],
    )
    assert freeze_common_support({"a": first, "b": second}, keys) == {
        (signal_date, "000001.SZ")
    }
    with pytest.raises(ContractError, match="frozen common support"):
        evaluate_predictions(
            predictions=first,
            labels=(_label(signal_date, "000001.SZ", 1.0),),
            frequency=EvaluationFrequency.WEEKLY,
            support=SupportMode.COMMON,
            outcome=OutcomeMode.ABSOLUTE,
            eligible_keys=keys,
        )


def test_formal_entry_point_requires_registered_prediction_hash() -> None:
    signal_date = CALENDAR[0]
    frame = _frame(
        "registered",
        [(signal_date, "000001.SZ", 1.0), (signal_date, "000002.SZ", 2.0)],
    )
    feature_manifest = FormalFeatureManifest(
        dataset_id="d0",
        d0_manifest_hash="a" * 64,
        feature_eligibility={"ret_1d": True},
    )
    manifest = RunManifest(
        run_id=frame.run_id,
        model="Ridge",
        universe=UniverseClass.CSI300,
        information_set="A0",
        split=Partition.VALIDATION,
        purpose=Purpose.SELECT,
        data_hash="a" * 64,
        asset_registry_hash="b" * 64,
        execution_calendar_manifest_hash="c" * 64,
        feature_schema_hash="d" * 64,
        market_state_hash="e" * 64,
        config_hash="f" * 64,
        code_hash="1" * 64,
        upstream_commit="internal-ridge-v1",
        seed=1,
        status=ResultState.PASS,
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        prediction_hash=frame.stable_hash(),
        formal_feature_manifest_hash=feature_manifest.stable_hash(),
    )
    labels = (
        _label(signal_date, "000001.SZ", 1.0),
        _label(signal_date, "000002.SZ", 2.0),
    )
    keys = {(row.signal_date, row.ts_code) for row in labels}
    result = evaluate_formal_predictions(
        manifest=manifest,
        protocol=ProtocolSpec.research_v1(),
        feature_manifest=feature_manifest,
        predictions=frame,
        labels=labels,
        frequency=EvaluationFrequency.WEEKLY,
        support=SupportMode.COMMON,
        outcome=OutcomeMode.ABSOLUTE,
        eligible_keys=keys,
        common_support_keys=keys,
        group_count=2,
    )
    assert result.rank_ic == pytest.approx(1.0)
