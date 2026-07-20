"""Package-native LightGBM adapter; early stopping is explicit and validation-only."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Mapping

from a_share_research.contracts import ContractError, CoverageState, canonical_hash
from a_share_research.models.tabular.common import (
    TabularDiagnostics,
    TabularModelResult,
    prediction_frame_from_scores,
    validate_causal_fold,
)
from a_share_research.models.tabular.layout import FeatureGate, FeatureLayout
from a_share_research.models.tabular.preprocessing import (
    PreprocessingConfig,
    TrainOnlyPreprocessor,
)
from a_share_research.models.tabular.samples import TabularSample, require_training_targets


@dataclass(frozen=True)
class LightGBMConfig:
    objective: str = "regression"
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int | None = None
    seed: int = 20260719
    n_jobs: int = 1
    version: str = "lightgbm-v1"

    def __post_init__(self) -> None:
        integer_fields = (
            "n_estimators", "num_leaves", "max_depth", "min_child_samples", "seed", "n_jobs"
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ContractError("LightGBM count, depth, seed and n_jobs fields must be integers")
        numeric_fields = (
            "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"LightGBM {name} must be numeric")
            if not math.isfinite(value):
                raise ContractError(f"LightGBM {name} must be finite")
        if self.early_stopping_rounds is not None:
            if type(self.early_stopping_rounds) is not int:
                raise ContractError("early_stopping_rounds must be an integer")
        if not self.objective or not self.version:
            raise ContractError("LightGBM objective and version are required")
        if self.n_estimators < 1 or self.learning_rate <= 0 or self.num_leaves < 2:
            raise ContractError("invalid LightGBM capacity configuration")
        if self.max_depth == 0 or self.min_child_samples < 1:
            raise ContractError("invalid LightGBM tree constraints")
        for name in ("subsample", "colsample_bytree"):
            if not 0 < getattr(self, name) <= 1:
                raise ContractError(f"{name} must be in (0, 1]")
        if min(self.reg_alpha, self.reg_lambda) < 0:
            raise ContractError("LightGBM regularization must be non-negative")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ContractError("early_stopping_rounds must be positive when enabled")
        if self.n_jobs == 0:
            raise ContractError("n_jobs cannot be zero")

    def stable_hash(self) -> str:
        return canonical_hash(self.__dict__)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LightGBMConfig:
        expected = {
            "objective", "n_estimators", "learning_rate", "num_leaves", "max_depth",
            "min_child_samples", "subsample", "colsample_bytree", "reg_alpha",
            "reg_lambda", "early_stopping_rounds", "seed", "n_jobs",
        }
        supplied = set(payload) - {
            "model",
            "adapter",
            "config_version",
            "selection_policy",
        }
        if supplied != expected or payload.get("model") != "LightGBM":
            raise ContractError("LightGBM config does not match its frozen schema")
        if payload.get("config_version") != "lightgbm-v1":
            raise ContractError("unsupported LightGBM config version")
        early_stopping = payload["early_stopping_rounds"]
        return cls(
            objective=str(payload["objective"]),
            n_estimators=int(payload["n_estimators"]),
            learning_rate=float(payload["learning_rate"]),
            num_leaves=int(payload["num_leaves"]),
            max_depth=int(payload["max_depth"]),
            min_child_samples=int(payload["min_child_samples"]),
            subsample=float(payload["subsample"]),
            colsample_bytree=float(payload["colsample_bytree"]),
            reg_alpha=float(payload["reg_alpha"]),
            reg_lambda=float(payload["reg_lambda"]),
            early_stopping_rounds=(
                None if early_stopping is None else int(early_stopping)
            ),
            seed=int(payload["seed"]),
            n_jobs=int(payload["n_jobs"]),
        )


class LightGBMAdapter:
    model_name = "LightGBM"

    def __init__(
        self,
        layout: FeatureLayout,
        gate: FeatureGate,
        *,
        model_config: LightGBMConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.layout = layout
        self.gate = gate
        self.model_config = model_config or LightGBMConfig()
        self.preprocessor = TrainOnlyPreprocessor(layout, gate, preprocessing_config)
        self.model: object | None = None

    def fit_predict(
        self,
        *,
        run_id: str,
        training: tuple[TabularSample, ...],
        validation: tuple[TabularSample, ...],
        prediction: tuple[TabularSample, ...],
        fit_end: date,
        validation_end: date | None,
        fit_data_hash: str,
        fold_id: str,
    ) -> TabularModelResult:
        validate_causal_fold(
            training,
            validation,
            prediction,
            fit_end=fit_end,
            validation_end=validation_end,
        )
        targets = require_training_targets(training)
        validation_targets = require_training_targets(validation) if validation else ()
        if self.model_config.early_stopping_rounds is not None and not validation:
            raise ContractError("explicit LightGBM early stopping requires validation rows")
        state = self.preprocessor.fit(
            training,
            fit_end=fit_end,
            fit_data_hash=fit_data_hash,
            fold_id=fold_id,
        )
        x_train = self.preprocessor.transform(training)
        x_validation = self.preprocessor.transform(validation)
        scoreable = tuple(
            sample for sample in prediction if sample.coverage_state is CoverageState.SCORED
        )
        x_prediction = self.preprocessor.transform(scoreable)
        # Current LightGBM routes tuple-backed matrices ambiguously through
        # scipy.  Materialize explicit dense arrays at this package boundary.
        import numpy as np

        x_train_native = np.asarray(x_train, dtype=float)
        x_validation_native = np.asarray(x_validation, dtype=float)
        x_prediction_native = np.asarray(x_prediction, dtype=float)
        targets_native = np.asarray(targets, dtype=float)
        validation_targets_native = np.asarray(validation_targets, dtype=float)

        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(
            objective=self.model_config.objective,
            n_estimators=self.model_config.n_estimators,
            learning_rate=self.model_config.learning_rate,
            num_leaves=self.model_config.num_leaves,
            max_depth=self.model_config.max_depth,
            min_child_samples=self.model_config.min_child_samples,
            subsample=self.model_config.subsample,
            colsample_bytree=self.model_config.colsample_bytree,
            reg_alpha=self.model_config.reg_alpha,
            reg_lambda=self.model_config.reg_lambda,
            random_state=self.model_config.seed,
            n_jobs=self.model_config.n_jobs,
            verbosity=-1,
        )
        fit_kwargs: dict[str, object] = {}
        if validation:
            fit_kwargs["eval_set"] = [
                (x_validation_native, validation_targets_native)
            ]
        if self.model_config.early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.model_config.early_stopping_rounds, verbose=False)
            ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit(x_train_native, targets_native, **fit_kwargs)
            raw_scores = (
                self.model.predict(x_prediction_native) if len(x_prediction_native) else ()
            )
        scores = tuple(float(value) for value in raw_scores)
        frame = prediction_frame_from_scores(run_id, prediction, scores)
        importance = tuple(
            (name, float(value))
            for name, value in zip(self.layout.columns, self.model.feature_importances_)
        )
        best_iteration = getattr(self.model, "best_iteration_", None)
        if best_iteration is not None and int(best_iteration) < 1:
            best_iteration = None
        diagnostics = TabularDiagnostics(
            model=self.model_name,
            information_set=self.gate.information_set.value,
            config_hash=self.model_config.stable_hash(),
            layout_hash=self.layout.stable_hash(),
            gate_hash=self.gate.stable_hash(),
            preprocessing_hash=state.stable_hash(),
            fit_data_hash=state.fit_data_hash,
            fold_id=state.fold_id,
            training_start=min(sample.signal_date for sample in training),
            training_end=fit_end,
            validation_start=min((sample.signal_date for sample in validation), default=None),
            validation_end=validation_end,
            n_train=len(training),
            n_validation=len(validation),
            n_prediction=len(prediction),
            n_scored=len(scoreable),
            seed=self.model_config.seed,
            feature_importance=importance,
            fit_warnings=tuple(str(item.message) for item in caught),
            best_iteration=int(best_iteration) if best_iteration is not None else None,
        )
        return TabularModelResult(frame, diagnostics)
