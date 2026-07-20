from datetime import date

import pytest

from a_share_research.baselines import (
    IndexReference,
    IndexReferencePoint,
    IndexReturnKind,
    MomentumObservation,
    cash_baseline,
    eligible_equal_weight,
    momentum_prediction_frame,
    top_fraction_targets,
)
from a_share_research.baselines.contracts import BaselineKind
from a_share_research.contracts import ContractError, CoverageState, Eligibility


def _eligibility(signal_date: date) -> tuple[Eligibility, ...]:
    return tuple(
        Eligibility(
            signal_date=signal_date,
            ts_code=code,
            universe="CSI300",
            member=True,
            observed=True,
            tradable=tradable,
            complete=complete,
        )
        for code, tradable, complete in (
            ("000001.SZ", True, True),
            ("000002.SZ", True, True),
            ("000003.SZ", False, True),
            ("000004.SZ", True, False),
        )
    )


def test_equal_weight_cash_and_momentum_use_same_eligibility() -> None:
    signal_date = date(2025, 1, 2)
    eligibility = _eligibility(signal_date)
    equal_weight = eligible_equal_weight(run_id="ew", eligibility=eligibility)
    assert [target.weight for target in equal_weight.targets] == pytest.approx([0.5, 0.5])
    assert {target.ts_code for target in equal_weight.targets} == {
        "000001.SZ",
        "000002.SZ",
    }
    cash = cash_baseline(run_id="cash", signal_dates=(signal_date,))
    assert cash.targets == ()
    assert cash.cash_weight_by_date == {signal_date.isoformat(): 1.0}
    frame = momentum_prediction_frame(
        run_id="mom",
        eligibility=eligibility,
        observations=tuple(
            MomentumObservation(
                signal_date,
                code,
                date(2024, 12, 2),
                signal_date,
                value,
                "a" * 64,
            )
            for code, value in (
                ("000001.SZ", 0.1),
                ("000002.SZ", 0.2),
                ("000003.SZ", 99.0),
                ("000004.SZ", 99.0),
            )
        ),
    )
    assert [
        record.ts_code
        for record in frame.records
        if record.coverage_state is CoverageState.SCORED
    ] == ["000001.SZ", "000002.SZ"]


def test_dynamic_top_fraction_uses_ceil_of_scored_eligible_universe() -> None:
    signal_date = date(2025, 1, 2)
    eligibility = tuple(
        Eligibility(signal_date, f"{index:06d}.SZ", "CSI300", True, True, True, True)
        for index in range(1, 22)
    )
    frame = momentum_prediction_frame(
        run_id="mom",
        eligibility=eligibility,
        observations=tuple(
            MomentumObservation(
                signal_date,
                row.ts_code,
                date(2024, 12, 2),
                signal_date,
                float(index),
                "a" * 64,
            )
            for index, row in enumerate(eligibility)
        ),
    )
    targets = top_fraction_targets(
        predictions=frame,
        eligibility=eligibility,
        fraction=0.10,
    )
    assert len(targets.targets) == 3
    assert sum(target.weight for target in targets.targets) == pytest.approx(1.0)


def test_official_index_is_explicitly_a_reporting_reference() -> None:
    reference = IndexReference(
        baseline=BaselineKind.OFFICIAL_INDEX,
        index_code="000300.SH",
        return_kind=IndexReturnKind.PRICE_RETURN,
        source_hash="a" * 64,
        points=(IndexReferencePoint(date(2025, 1, 2), 5, 0.01),),
    )
    assert reference.return_kind is IndexReturnKind.PRICE_RETURN


def test_momentum_observation_rejects_post_signal_input() -> None:
    with pytest.raises(ContractError, match="post-signal"):
        MomentumObservation(
            date(2025, 1, 2),
            "000001.SZ",
            date(2024, 12, 2),
            date(2025, 1, 3),
            0.1,
            "a" * 64,
        )
