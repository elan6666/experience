"""Training-fold-only tabular preprocessing with replayable fitted state."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

from a_share_research.contracts import CanonicalModel, ContractError, CoverageState, canonical_hash
from a_share_research.models.tabular.layout import FeatureGate, FeatureLayout
from a_share_research.models.tabular.samples import TabularSample

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PreprocessingConfig:
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    neutralize: bool = True
    size_column: str = "total_mv"
    industry_column: str = "industry_id"
    version: str = "train-only-preprocessing-v1"

    def __post_init__(self) -> None:
        for name in ("winsor_lower", "winsor_upper"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ContractError(f"{name} must be finite")
        if not 0.0 <= self.winsor_lower < self.winsor_upper <= 1.0:
            raise ContractError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
        if type(self.neutralize) is not bool:
            raise ContractError("neutralize must be boolean")
        if not self.size_column or not self.industry_column or not self.version:
            raise ContractError("preprocessing config names and version are required")

    def stable_hash(self) -> str:
        return canonical_hash(
            {
                "winsor_lower": self.winsor_lower,
                "winsor_upper": self.winsor_upper,
                "neutralize": self.neutralize,
                "size_column": self.size_column,
                "industry_column": self.industry_column,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class ColumnTransform(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "tabular_column_transform"

    name: str
    lower: float
    upper: float
    impute: float
    mean: float
    scale: float
    binary_indicator: bool

    def validate(self) -> None:
        if not self.name:
            raise ContractError("column transform name is required")
        for name in ("lower", "upper", "impute", "mean", "scale"):
            if not math.isfinite(getattr(self, name)):
                raise ContractError(f"{name} must be finite")
        if self.lower > self.upper:
            raise ContractError("winsor lower cannot exceed upper")
        if self.scale <= 0:
            raise ContractError("scale must be positive")
        if type(self.binary_indicator) is not bool:
            raise ContractError("binary_indicator must be boolean")


@dataclass(frozen=True)
class NeutralizationCoefficient(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "tabular_neutralization_coefficient"

    feature_name: str
    coefficients: tuple[float, ...]

    def validate(self) -> None:
        if not self.feature_name or not self.coefficients:
            raise ContractError("neutralization feature and coefficients are required")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ContractError("neutralization coefficients must be finite")


@dataclass(frozen=True)
class PreprocessingState(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "tabular_preprocessing_state"

    layout_hash: str
    gate_hash: str
    config_hash: str
    fit_data_hash: str
    fold_id: str
    fit_start: date
    fit_end: date
    sample_count: int
    columns: tuple[ColumnTransform, ...]
    industry_categories: tuple[float, ...]
    neutralization: tuple[NeutralizationCoefficient, ...]

    def validate(self) -> None:
        for name in ("layout_hash", "gate_hash", "config_hash", "fit_data_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if not self.fold_id:
            raise ContractError("preprocessing fold_id is required")
        if type(self.sample_count) is not int:
            raise ContractError("preprocessing sample_count must be an integer")
        if self.fit_end < self.fit_start or self.sample_count < 1:
            raise ContractError("invalid preprocessing fit window")
        if not self.columns or len({column.name for column in self.columns}) != len(self.columns):
            raise ContractError("preprocessing columns must be non-empty and unique")
        if tuple(sorted(set(self.industry_categories))) != self.industry_categories:
            raise ContractError("industry categories must be unique and increasing")
        feature_names = [item.feature_name for item in self.neutralization]
        if len(feature_names) != len(set(feature_names)):
            raise ContractError("neutralization features must be unique")
        expected_width = 2 + max(len(self.industry_categories) - 1, 0)
        if any(len(item.coefficients) != expected_width for item in self.neutralization):
            raise ContractError("neutralization coefficient width disagrees with exposures")


class TrainOnlyPreprocessor:
    """Fit on one declared training fold, then replay without refitting."""

    def __init__(
        self,
        layout: FeatureLayout,
        gate: FeatureGate,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self.layout = layout
        self.gate = gate
        self.config = config or PreprocessingConfig()
        self.state: PreprocessingState | None = None

    def fit(
        self,
        samples: tuple[TabularSample, ...],
        *,
        fit_end: date,
        fit_data_hash: str,
        fold_id: str,
    ) -> PreprocessingState:
        if not samples:
            raise ContractError("cannot fit preprocessing without training rows")
        if any(sample.signal_date > fit_end for sample in samples):
            raise ContractError("preprocessing fit contains rows after declared training cutoff")
        if any(sample.coverage_state is not CoverageState.SCORED for sample in samples):
            raise ContractError("preprocessing fit accepts only scoreable rows")
        if not _SHA256.fullmatch(fit_data_hash) or not fold_id:
            raise ContractError("fit_data_hash and fold_id must bind preprocessing to its fold")

        import numpy as np

        raw = [sample.vector(self.layout, self.gate) for sample in samples]
        width = len(self.layout.columns)
        if any(len(row) != width for row in raw):
            raise ContractError("feature vector width disagrees with frozen layout")

        base = np.empty((len(raw), width), dtype=float)
        bounds: list[tuple[float, float, float, bool]] = []
        missing_columns = set(self.layout.fundamental_missing)
        for index, name in enumerate(self.layout.columns):
            values = np.asarray(
                [float(row[index]) for row in raw if row[index] is not None], dtype=float
            )
            if values.size and not np.isfinite(values).all():
                raise ContractError(f"non-finite training value in {name}")
            binary = name in missing_columns
            if binary:
                lower, upper, impute = 0.0, 1.0, 0.0
            elif name == self.config.industry_column and values.size:
                unique, counts = np.unique(values, return_counts=True)
                lower = float(values.min())
                upper = float(values.max())
                impute = float(unique[int(np.argmax(counts))])
            elif values.size:
                lower = float(np.quantile(values, self.config.winsor_lower))
                upper = float(np.quantile(values, self.config.winsor_upper))
                impute = float(np.median(values))
            else:
                lower = upper = impute = 0.0
            bounds.append((lower, upper, impute, binary))
            base[:, index] = [
                min(max(impute if value is None else float(value), lower), upper)
                for value in (row[index] for row in raw)
            ]

        categories: tuple[float, ...] = ()
        coefficient_by_name: dict[str, tuple[float, ...]] = {}
        if self._neutralization_is_active():
            industry_index = self.layout.columns.index(self.config.industry_column)
            size_index = self.layout.columns.index(self.config.size_column)
            categories = tuple(sorted(set(float(value) for value in base[:, industry_index])))
            design = self._design_matrix(base[:, industry_index], base[:, size_index], categories)
            target_names = tuple(
                name
                for name in self.layout.core + self.layout.fundamental
                if name not in {self.config.industry_column, self.config.size_column}
            )
            for name in target_names:
                index = self.layout.columns.index(name)
                coefficients = np.linalg.lstsq(design, base[:, index], rcond=None)[0]
                base[:, index] = base[:, index] - design @ coefficients
                coefficient_by_name[name] = tuple(float(value) for value in coefficients)

        transforms: list[ColumnTransform] = []
        for index, name in enumerate(self.layout.columns):
            lower, upper, impute, binary = bounds[index]
            if binary:
                mean, scale = 0.0, 1.0
            else:
                mean = float(base[:, index].mean())
                scale = float(base[:, index].std())
                if scale <= 1e-12:
                    scale = 1.0
            transforms.append(
                ColumnTransform(name, lower, upper, impute, mean, scale, binary)
            )

        self.state = PreprocessingState(
            layout_hash=self.layout.stable_hash(),
            gate_hash=self.gate.stable_hash(),
            config_hash=self.config.stable_hash(),
            fit_data_hash=fit_data_hash,
            fold_id=fold_id,
            fit_start=min(sample.signal_date for sample in samples),
            fit_end=fit_end,
            sample_count=len(samples),
            columns=tuple(transforms),
            industry_categories=categories,
            neutralization=tuple(
                NeutralizationCoefficient(name, coefficient_by_name[name])
                for name in self.layout.columns
                if name in coefficient_by_name
            ),
        )
        return self.state

    def transform(self, samples: tuple[TabularSample, ...]) -> tuple[tuple[float, ...], ...]:
        if self.state is None:
            raise ContractError("preprocessor must be fitted before transform")
        self.state.validate()
        if self.state.layout_hash != self.layout.stable_hash():
            raise ContractError("preprocessing state layout hash mismatch")
        if self.state.gate_hash != self.gate.stable_hash():
            raise ContractError("preprocessing state gate hash mismatch")
        if self.state.config_hash != self.config.stable_hash():
            raise ContractError("preprocessing state config hash mismatch")
        if tuple(column.name for column in self.state.columns) != self.layout.columns:
            raise ContractError("preprocessing state column order mismatch")
        if any(
            item.feature_name not in self.layout.columns
            for item in self.state.neutralization
        ):
            raise ContractError("preprocessing state contains an unknown neutralization feature")

        import numpy as np

        scoreable = tuple(
            sample for sample in samples if sample.coverage_state is CoverageState.SCORED
        )
        if not scoreable:
            return ()
        raw = [sample.vector(self.layout, self.gate) for sample in scoreable]
        matrix = np.empty((len(raw), len(self.layout.columns)), dtype=float)
        for index, column in enumerate(self.state.columns):
            matrix[:, index] = [
                min(
                    max(column.impute if row[index] is None else float(row[index]), column.lower),
                    column.upper,
                )
                for row in raw
            ]

        if self.state.neutralization:
            industry_index = self.layout.columns.index(self.config.industry_column)
            size_index = self.layout.columns.index(self.config.size_column)
            design = self._design_matrix(
                matrix[:, industry_index],
                matrix[:, size_index],
                self.state.industry_categories,
            )
            for item in self.state.neutralization:
                index = self.layout.columns.index(item.feature_name)
                matrix[:, index] = matrix[:, index] - design @ np.asarray(item.coefficients)

        for index, column in enumerate(self.state.columns):
            if not column.binary_indicator:
                matrix[:, index] = (matrix[:, index] - column.mean) / column.scale
        if not np.isfinite(matrix).all():
            raise ContractError("preprocessing produced non-finite values")
        return tuple(tuple(float(value) for value in row) for row in matrix)

    def _neutralization_is_active(self) -> bool:
        return (
            self.config.neutralize
            and self.gate.f_enabled
            and self.config.industry_column in self.layout.fundamental
            and self.config.size_column in self.layout.fundamental
        )

    @staticmethod
    def _design_matrix(industry: object, size: object, categories: tuple[float, ...]):
        import numpy as np

        industry_values = np.asarray(industry, dtype=float)
        size_values = np.log1p(np.maximum(np.asarray(size, dtype=float), 0.0))
        columns = [np.ones(industry_values.shape[0]), size_values]
        columns.extend((industry_values == category).astype(float) for category in categories[1:])
        return np.column_stack(columns)
