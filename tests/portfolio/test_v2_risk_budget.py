"""Synthetic B1 contracts; execute only on the approved server."""

from datetime import date

import pytest

from a_share_research.contracts import ContractError, MarketState
from a_share_research.data.market_state import SharedMarketState
from a_share_research.portfolio import (
    RiskBudgetPolicy,
    TargetFrame,
    TargetWeight,
    apply_shared_risk_budget,
    build_shared_risk_budget_schedule,
)


def _state() -> tuple[SharedMarketState, tuple[date, ...]]:
    dates = tuple(date(2025, month, 3) for month in range(1, 5))
    rows = tuple(
        MarketState(
            asof_date=signal_date,
            feature_name="risk_score_input",
            value=value,
            source_universe="CSI300",
            source_hash=str(index) * 64,
        )
        for index, (signal_date, value) in enumerate(
            zip(dates, (0.05, 0.15, 0.25, 0.35)), start=1
        )
    )
    return SharedMarketState(rows, "a" * 64, "b" * 64), dates


def _policy(state: SharedMarketState) -> RiskBudgetPolicy:
    return RiskBudgetPolicy(
        version="synthetic-v1",
        market_state_hash=state.stable_hash,
        calibration_data_hash="c" * 64,
        calibrated_through=date(2025, 12, 31),
        feature_weights={"risk_score_input": 1.0},
        full_if_score_at_most=0.1,
        sixty_if_score_at_most=0.2,
        thirty_if_score_at_most=0.3,
    )


def test_one_shared_schedule_proves_all_four_budget_branches() -> None:
    state, dates = _state()
    schedule = build_shared_risk_budget_schedule(
        shared_state=state,
        policy=_policy(state),
        signal_dates=dates,
    )
    replay = build_shared_risk_budget_schedule(
        shared_state=state,
        policy=_policy(state),
        signal_dates=dates,
    )
    assert tuple(point.equity_budget for point in schedule.points) == (1.0, 0.6, 0.3, 0.0)
    assert all(point.market_state_hash == state.stable_hash for point in schedule.points)
    assert schedule.stable_hash() == replay.stable_hash()


def test_same_schedule_scales_any_consumer_without_coverage_dependent_budget() -> None:
    state, dates = _state()
    schedule = build_shared_risk_budget_schedule(
        shared_state=state,
        policy=_policy(state),
        signal_dates=dates,
    )
    always_full = TargetFrame(
        run_id="frozen-b0-model-pool",
        targets=tuple(TargetWeight(day, "000001.SZ", 1.0) for day in dates),
        cash_weight_by_date={day.isoformat(): 0.0 for day in dates},
    )
    csi = apply_shared_risk_budget(
        always_full_targets=always_full,
        schedule=schedule,
        output_run_id="b1-csi300-ridge",
    )
    tech = apply_shared_risk_budget(
        always_full_targets=always_full,
        schedule=schedule,
        output_run_id="b1-tech100-fact",
    )
    assert csi.cash_weight_by_date == tech.cash_weight_by_date
    assert tuple(csi.cash_weight_by_date.values()) == pytest.approx((0.0, 0.4, 0.7, 1.0))
    assert not any(target.signal_date == dates[-1] for target in csi.targets)
    assert sum(target.weight for target in csi.targets if target.signal_date == dates[1]) == 0.6


def test_b1_fails_closed_on_non_full_input_or_2026_calibration() -> None:
    state, dates = _state()
    schedule = build_shared_risk_budget_schedule(
        shared_state=state,
        policy=_policy(state),
        signal_dates=dates,
    )
    not_full = TargetFrame(
        run_id="invalid-b0",
        targets=tuple(TargetWeight(day, "000001.SZ", 0.5) for day in dates),
        cash_weight_by_date={day.isoformat(): 0.5 for day in dates},
    )
    with pytest.raises(ContractError, match="always-full"):
        apply_shared_risk_budget(
            always_full_targets=not_full,
            schedule=schedule,
            output_run_id="invalid-b1",
        )
    with pytest.raises(ContractError, match="2026"):
        RiskBudgetPolicy(
            version="leaky",
            market_state_hash=state.stable_hash,
            calibration_data_hash="c" * 64,
            calibrated_through=date(2026, 1, 1),
            feature_weights={"risk_score_input": 1.0},
            full_if_score_at_most=0.1,
            sixty_if_score_at_most=0.2,
            thirty_if_score_at_most=0.3,
        )


def test_frozen_policy_can_score_an_appended_prospective_state_table() -> None:
    calibration_state, dates = _state()
    policy = _policy(calibration_state)
    prospective_day = date(2026, 1, 5)
    appended = SharedMarketState(
        calibration_state.rows
        + (
            MarketState(
                asof_date=prospective_day,
                feature_name="risk_score_input",
                value=0.25,
                source_universe="CSI300",
                source_hash="f" * 64,
            ),
        ),
        calibration_state.source_membership_hash,
        "d" * 64,
    )
    schedule = build_shared_risk_budget_schedule(
        shared_state=appended,
        policy=policy,
        signal_dates=(prospective_day,),
    )
    assert schedule.market_state_hash == appended.stable_hash
    assert schedule.policy_hash == policy.stable_hash()
    assert schedule.points[0].equity_budget == 0.3
    assert prospective_day not in dates


def test_risk_policy_rejects_an_all_zero_state_score() -> None:
    state, _ = _state()
    with pytest.raises(ContractError, match="non-zero"):
        RiskBudgetPolicy(
            version="degenerate",
            market_state_hash=state.stable_hash,
            calibration_data_hash="c" * 64,
            calibrated_through=date(2025, 12, 31),
            feature_weights={"risk_score_input": 0.0},
            full_if_score_at_most=0.1,
            sixty_if_score_at_most=0.2,
            thirty_if_score_at_most=0.3,
        )
