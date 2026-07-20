"""scikit-learn Ridge adapter with frozen causal preprocessing."""

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
class RidgeConfig:
    alpha: float = 1.0
    fit_intercept: bool = True
    solver: str = "auto"
    seed: int = 20260719
    version: str = "ridge-v1"

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ContractError("Ridge alpha must be numeric")
        if not math.isfinite(self.alpha):
            raise ContractError("Ridge alpha must be finite")
        if type(self.seed) is not int:
            raise ContractError("Ridge seed must be an integer")
        if self.alpha < 0 or not self.solver or not self.version:
            raise ContractError("invalid Ridge configuration")
        if type(self.fit_intercept) is not bool:
            raise ContractError("fit_intercept must be boolean")

    def stable_hash(self) -> str:
        return canonical_hash(
            {
                "alpha": self.alpha,
                "fit_intercept": self.fit_intercept,
                "solver": self.solver,
                "seed": self.seed,
                "version": self.version,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RidgeConfig:
        expected = {"alpha", "fit_intercept", "solver", "seed"}
        supplied = set(payload) - {
            "model",
            "adapter",
            "config_version",
            "selection_policy",
        }
        if supplied != expected or payload.get("model") != "Ridge":
            raise ContractError("Ridge config does not match its frozen schema")
        if payload.get("config_version") != "ridge-v1":
            raise ContractError("unsupported Ridge config version")
        return cls(
            alpha=float(payload["alpha"]),
            fit_intercept=payload["fit_intercept"],  # type: ignore[arg-type]
            solver=str(payload["solver"]),
            seed=int(payload["seed"]),
        )


class RidgeAdapter:
    model_name = "Ridge"

    def __init__(
        self,
        layout: FeatureLayout,
        gate: FeatureGate,
        *,
        model_config: RidgeConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.layout = layout
        self.gate = gate
        self.model_config = model_config or RidgeConfig()
        self.preprocessor = TrainOnlyPreprocessor(layout, gate, preprocessing_config)
        self.model: object | None = None

    def fit_predict(
        self,
        *,
        run_id: str,
        training: tuple[TabularSample, ...],
        prediction: tuple[TabularSample, ...],
        fit_end: date,
        fit_data_hash: str,
        fold_id: str,
    ) -> TabularModelResult:
        validate_causal_fold(
            training,
            (),
            prediction,
            fit_end=fit_end,
            validation_end=None,
        )
        targets = require_training_targets(training)
        state = self.preprocessor.fit(
            training,
            fit_end=fit_end,
            fit_data_hash=fit_data_hash,
            fold_id=fold_id,
        )
        x_train = self.preprocessor.transform(training)
        scoreable = tuple(
            sample for sample in prediction if sample.coverage_state is CoverageState.SCORED
        )
        x_prediction = self.preprocessor.transform(scoreable)

        from sklearn.linear_model import Ridge

        self.model = Ridge(
            alpha=self.model_config.alpha,
            fit_intercept=self.model_config.fit_intercept,
            solver=self.model_config.solver,
            random_state=self.model_config.seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit(x_train, targets)
            raw_scores = self.model.predict(x_prediction) if x_prediction else ()
        scores = tuple(float(value) for value in raw_scores)
        frame = prediction_frame_from_scores(run_id, prediction, scores)
        coefficients = tuple(
            (name, float(value))
            for name, value in zip(self.layout.columns, self.model.coef_)
        )
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
            validation_start=None,
            validation_end=None,
            n_train=len(training),
            n_validation=0,
            n_prediction=len(prediction),
            n_scored=len(scoreable),
            seed=self.model_config.seed,
            feature_importance=coefficients,
            fit_warnings=tuple(str(item.message) for item in caught),
        )
        return TabularModelResult(frame, diagnostics)
