"""Deterministic, complete PredictionFrame export independent of author loaders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from a_share_research.adapters.common.identity import CausalAssetMaster
from a_share_research.adapters.common.types import AdapterContractError
from a_share_research.contracts import (
    AssetRegistry,
    CoverageState,
    MaskBundle,
    PredictionFrame,
    PredictionRecord,
)


@dataclass(frozen=True)
class PredictionBatch:
    """One author-loader batch; final partial batches are first-class evidence."""

    signal_dates: tuple[date, ...]
    scores: tuple[tuple[float | None, ...], ...]

    def __post_init__(self) -> None:
        if not self.signal_dates or len(self.signal_dates) != len(self.scores):
            raise AdapterContractError("prediction batch date/score row counts differ")


def export_prediction_batches(
    *,
    run_id: str,
    evaluation_registry: AssetRegistry,
    model_master: CausalAssetMaster,
    expected_dates: tuple[date, ...],
    masks: tuple[MaskBundle, ...],
    history_ready: tuple[tuple[bool, ...], ...],
    batches: Iterable[PredictionBatch],
) -> PredictionFrame:
    """Export all expected date/assets or fail; never hide a dropped final batch."""
    if not run_id or not expected_dates:
        raise AdapterContractError("run_id and expected dates are required")
    if expected_dates != tuple(sorted(set(expected_dates))):
        raise AdapterContractError("expected dates must be unique and increasing")
    if len(masks) != len(expected_dates) or len(history_ready) != len(expected_dates):
        raise AdapterContractError("mask/history rows must match expected dates")
    slot_count = len(evaluation_registry.asset_ids)
    batch_scores: dict[date, tuple[float | None, ...]] = {}
    for batch in batches:
        for signal_date, scores in zip(batch.signal_dates, batch.scores, strict=True):
            if signal_date in batch_scores:
                raise AdapterContractError(f"duplicate prediction date: {signal_date}")
            if len(scores) != len(model_master.asset_ids):
                raise AdapterContractError("model score width does not match its causal master")
            if any(score is not None and not math.isfinite(score) for score in scores):
                raise AdapterContractError("model emitted a non-finite score")
            batch_scores[signal_date] = scores
    if set(batch_scores) != set(expected_dates):
        missing = sorted(set(expected_dates) - set(batch_scores))
        extra = sorted(set(batch_scores) - set(expected_dates))
        raise AdapterContractError(
            f"incomplete prediction batches; missing={missing}, extra={extra}"
        )

    records: list[PredictionRecord] = []
    for date_index, signal_date in enumerate(expected_dates):
        bundle = masks[date_index]
        if bundle.signal_date != signal_date or bundle.asset_ids != evaluation_registry.asset_ids:
            raise AdapterContractError("evaluation mask identity/date mismatch")
        if len(history_ready[date_index]) != slot_count:
            raise AdapterContractError("history-ready width does not match evaluation registry")
        model_scores = batch_scores[signal_date]
        for slot, ts_code in enumerate(evaluation_registry.asset_ids):
            score: float | None = None
            if not bundle.member[slot]:
                state = CoverageState.NOT_MEMBER
            elif not bundle.observed[slot]:
                state = CoverageState.NOT_OBSERVED
            elif not model_master.supports(ts_code):
                state = CoverageState.MODEL_UNSUPPORTED
            elif not history_ready[date_index][slot]:
                state = CoverageState.INSUFFICIENT_HISTORY
            else:
                model_slot = model_master.slot(ts_code)
                score = model_scores[model_slot]
                if score is None:
                    raise AdapterContractError("eligible supported asset has no model score")
                state = CoverageState.SCORED
            records.append(PredictionRecord(signal_date, ts_code, score, state))
    return PredictionFrame(run_id=run_id, records=tuple(records))
