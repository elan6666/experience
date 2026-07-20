"""Shared causal fold checks, exports and diagnostics for tabular models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import ClassVar

from a_share_research.contracts import (
    CanonicalModel,
    ContractError,
    CoverageState,
    PredictionFrame,
    PredictionRecord,
    RunManifest,
)
from a_share_research.models.tabular.samples import TabularSample
from a_share_research.quality.states import ResultState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TabularDiagnostics(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "tabular_diagnostics"

    model: str
    information_set: str
    config_hash: str
    layout_hash: str
    gate_hash: str
    preprocessing_hash: str
    fit_data_hash: str
    fold_id: str
    training_start: date
    training_end: date
    validation_start: date | None
    validation_end: date | None
    n_train: int
    n_validation: int
    n_prediction: int
    n_scored: int
    seed: int
    feature_importance: tuple[tuple[str, float], ...]
    fit_warnings: tuple[str, ...] = ()
    best_iteration: int | None = None

    def validate(self) -> None:
        if not self.model or not self.information_set or not self.fold_id:
            raise ContractError("diagnostic model, information_set and fold_id are required")
        if self.information_set not in {"A0", "A1", "A2", "A3"}:
            raise ContractError("diagnostic information_set must be A0-A3")
        for name in (
            "config_hash",
            "layout_hash",
            "gate_hash",
            "preprocessing_hash",
            "fit_data_hash",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if self.training_end < self.training_start:
            raise ContractError("training diagnostic window is invalid")
        if (self.validation_start is None) != (self.validation_end is None):
            raise ContractError("validation start/end must be supplied together")
        if self.validation_start is not None:
            if self.validation_start <= self.training_end:
                raise ContractError("validation must follow the training cutoff")
            if self.validation_end is not None and self.validation_end < self.validation_start:
                raise ContractError("validation diagnostic window is invalid")
        if any(
            type(value) is not int
            for value in (
                self.n_train,
                self.n_validation,
                self.n_prediction,
                self.n_scored,
                self.seed,
            )
        ):
            raise ContractError("diagnostic counts and seed must be integers")
        if min(self.n_train, self.n_validation, self.n_prediction, self.n_scored) < 0:
            raise ContractError("diagnostic counts cannot be negative")
        if self.n_train < 1 or self.n_scored > self.n_prediction:
            raise ContractError("diagnostic sample counts are inconsistent")
        if any(not name for name, _ in self.feature_importance):
            raise ContractError("feature importance names cannot be empty")
        if len({name for name, _ in self.feature_importance}) != len(self.feature_importance):
            raise ContractError("feature importance names must be unique")
        if any(not math.isfinite(value) for _, value in self.feature_importance):
            raise ContractError("feature importances must be finite")
        if self.best_iteration is not None:
            if type(self.best_iteration) is not int or self.best_iteration < 1:
                raise ContractError("best_iteration must be a positive integer")


@dataclass(frozen=True)
class TabularModelResult:
    predictions: PredictionFrame
    diagnostics: TabularDiagnostics

    def __post_init__(self) -> None:
        self.predictions.validate()
        self.diagnostics.validate()
        if self.diagnostics.n_prediction != len(self.predictions.records):
            raise ContractError("diagnostics and PredictionFrame row counts disagree")


def validate_causal_fold(
    training: tuple[TabularSample, ...],
    validation: tuple[TabularSample, ...],
    prediction: tuple[TabularSample, ...],
    *,
    fit_end: date,
    validation_end: date | None,
) -> None:
    if not training:
        raise ContractError("training fold cannot be empty")
    if not prediction:
        raise ContractError("prediction fold cannot be empty")
    if any(sample.signal_date > fit_end for sample in training):
        raise ContractError("training fold crosses its declared cutoff")
    if validation:
        if validation_end is None:
            raise ContractError("validation rows require a declared validation cutoff")
        if any(not fit_end < sample.signal_date <= validation_end for sample in validation):
            raise ContractError("validation must be strictly after train and within its cutoff")
    elif validation_end is not None:
        raise ContractError("validation cutoff supplied without validation rows")
    if any(sample.signal_date <= fit_end for sample in prediction):
        raise ContractError("walk-forward prediction must follow the training cutoff")
    for name, samples in (
        ("training", training),
        ("validation", validation),
        ("prediction", prediction),
    ):
        keys = {(sample.signal_date, sample.ts_code) for sample in samples}
        if len(keys) != len(samples):
            raise ContractError(f"duplicate asset/date key in {name} fold")


def prediction_frame_from_scores(
    run_id: str,
    samples: tuple[TabularSample, ...],
    scores: tuple[float, ...],
) -> PredictionFrame:
    if not run_id or not samples:
        raise ContractError("run_id and prediction samples are required")
    expected_scores = sum(
        sample.coverage_state is CoverageState.SCORED for sample in samples
    )
    if len(scores) != expected_scores:
        raise ContractError("score count disagrees with scoreable prediction rows")
    iterator = iter(scores)
    records: list[PredictionRecord] = []
    for sample in samples:
        coverage = sample.coverage_state
        score = float(next(iterator)) if coverage is CoverageState.SCORED else None
        records.append(PredictionRecord(sample.signal_date, sample.ts_code, score, coverage))
    return PredictionFrame(run_id=run_id, records=tuple(records))


def complete_run_manifest(
    draft: RunManifest,
    result: TabularModelResult,
    *,
    status: ResultState,
    completed_at: datetime,
) -> RunManifest:
    """Bind model output to caller-supplied D0/protocol evidence without inventing hashes."""
    predictions = result.predictions
    diagnostics = result.diagnostics
    if draft.run_id != predictions.run_id:
        raise ContractError("RunManifest and PredictionFrame run_id differ")
    identity_checks = {
        "model": (draft.model, diagnostics.model),
        "information_set": (draft.information_set, diagnostics.information_set),
        "config_hash": (draft.config_hash, diagnostics.config_hash),
        "feature_schema_hash": (draft.feature_schema_hash, diagnostics.layout_hash),
        "data_hash": (draft.data_hash, diagnostics.fit_data_hash),
        "seed": (draft.seed, diagnostics.seed),
    }
    mismatches = sorted(
        name for name, (manifest_value, result_value) in identity_checks.items()
        if manifest_value != result_value
    )
    if mismatches:
        raise ContractError(f"RunManifest disagrees with model result: {mismatches}")
    if draft.completed_at is not None or draft.prediction_hash is not None:
        raise ContractError("only an incomplete manifest draft may be finalized")
    if not isinstance(status, ResultState):
        raise ContractError("status must use ResultState")
    return replace(
        draft,
        status=status,
        completed_at=completed_at,
        prediction_hash=predictions.stable_hash(),
    )
