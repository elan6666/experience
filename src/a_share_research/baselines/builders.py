"""Simple baselines using the same frozen eligibility and target contracts."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

from a_share_research.contracts import (
    ContractError,
    CoverageState,
    Eligibility,
    PredictionFrame,
    PredictionRecord,
)
from a_share_research.portfolio.intents import TargetFrame, TargetWeight

from .contracts import MomentumObservation


def _tradable_by_date(eligibility: tuple[Eligibility, ...]) -> dict[date, tuple[str, ...]]:
    grouped: dict[date, list[str]] = defaultdict(list)
    seen: set[tuple[date, str]] = set()
    for row in eligibility:
        row.validate()
        key = (row.signal_date, row.ts_code)
        if key in seen:
            raise ContractError("duplicate eligibility row")
        seen.add(key)
        if row.tradable and row.complete:
            grouped[row.signal_date].append(row.ts_code)
    return {signal_date: tuple(sorted(codes)) for signal_date, codes in grouped.items()}


def eligible_equal_weight(
    *, run_id: str, eligibility: tuple[Eligibility, ...]
) -> TargetFrame:
    grouped = _tradable_by_date(eligibility)
    dates = sorted({row.signal_date for row in eligibility})
    targets: list[TargetWeight] = []
    cash: dict[str, float] = {}
    for signal_date in dates:
        codes = grouped.get(signal_date, ())
        if not codes:
            cash[signal_date.isoformat()] = 1.0
            continue
        weight = 1 / len(codes)
        targets.extend(TargetWeight(signal_date, code, weight) for code in codes)
        cash[signal_date.isoformat()] = 0.0
    return TargetFrame(run_id=run_id, targets=tuple(targets), cash_weight_by_date=cash)


def cash_baseline(*, run_id: str, signal_dates: tuple[date, ...]) -> TargetFrame:
    if not signal_dates or tuple(sorted(set(signal_dates))) != signal_dates:
        raise ContractError("cash baseline dates must be unique and increasing")
    return TargetFrame(
        run_id=run_id,
        targets=(),
        cash_weight_by_date={signal_date.isoformat(): 1.0 for signal_date in signal_dates},
    )


def momentum_prediction_frame(
    *,
    run_id: str,
    eligibility: tuple[Eligibility, ...],
    observations: tuple[MomentumObservation, ...],
) -> PredictionFrame:
    values: dict[tuple[date, str], float] = {}
    for observation in observations:
        observation.validate()
        key = (observation.signal_date, observation.ts_code)
        if key in values:
            raise ContractError("duplicate momentum observation")
        values[key] = observation.return_value
    records: list[PredictionRecord] = []
    for row in sorted(eligibility, key=lambda item: (item.signal_date, item.ts_code)):
        row.validate()
        value = values.get((row.signal_date, row.ts_code))
        scored = row.tradable and row.complete and value is not None
        records.append(
            PredictionRecord(
                signal_date=row.signal_date,
                ts_code=row.ts_code,
                score=value if scored else None,
                coverage_state=(
                    CoverageState.SCORED if scored else CoverageState.INSUFFICIENT_HISTORY
                ),
            )
        )
    return PredictionFrame(run_id=run_id, records=tuple(records))


def top_fraction_targets(
    *,
    predictions: PredictionFrame,
    eligibility: tuple[Eligibility, ...],
    fraction: float = 0.10,
) -> TargetFrame:
    """Convert frozen scores to dynamic ceil(N*fraction), equal-weight targets."""
    if not 0 < fraction <= 1:
        raise ContractError("top fraction must be in (0, 1]")
    predictions.validate()
    eligible = {
        (row.signal_date, row.ts_code)
        for row in eligibility
        if row.tradable and row.complete
    }
    dates = sorted({row.signal_date for row in eligibility})
    scored: dict[date, list[tuple[float, str]]] = defaultdict(list)
    for record in predictions.records:
        key = (record.signal_date, record.ts_code)
        if key in eligible and record.coverage_state is CoverageState.SCORED:
            if record.score is None:
                raise ContractError("scored prediction lacks score")
            scored[record.signal_date].append((record.score, record.ts_code))
    targets: list[TargetWeight] = []
    cash: dict[str, float] = {}
    for signal_date in dates:
        candidates = sorted(scored.get(signal_date, ()), key=lambda item: (-item[0], item[1]))
        if not candidates:
            cash[signal_date.isoformat()] = 1.0
            continue
        count = max(1, math.ceil(len(candidates) * fraction))
        selected = candidates[:count]
        targets.extend(
            TargetWeight(signal_date, code, 1 / count) for _, code in selected
        )
        cash[signal_date.isoformat()] = 0.0
    return TargetFrame(
        run_id=predictions.run_id,
        targets=tuple(targets),
        cash_weight_by_date=cash,
    )
