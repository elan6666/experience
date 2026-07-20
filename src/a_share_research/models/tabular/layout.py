"""Stable Core/F/F-missing/S layout and causal information gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from a_share_research.contracts import ContractError, canonical_hash
from a_share_research.features.schema import InformationClass, d0_features


class InformationSet(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"

    @property
    def enables_f(self) -> bool:
        return self in {InformationSet.A1, InformationSet.A3}

    @property
    def enables_s(self) -> bool:
        return self in {InformationSet.A2, InformationSet.A3}


@dataclass(frozen=True)
class FeatureGate:
    information_set: InformationSet

    def __post_init__(self) -> None:
        if not isinstance(self.information_set, InformationSet):
            raise ContractError("information_set must use InformationSet")

    @property
    def f_enabled(self) -> bool:
        return self.information_set.enables_f

    @property
    def s_enabled(self) -> bool:
        return self.information_set.enables_s

    def stable_hash(self) -> str:
        return canonical_hash(
            {
                "information_set": self.information_set.value,
                "f_enabled": self.f_enabled,
                "s_enabled": self.s_enabled,
            }
        )


@dataclass(frozen=True)
class FeatureLayout:
    """Union layout; its schema hash is intentionally independent of A0-A3."""

    core: tuple[str, ...]
    fundamental: tuple[str, ...]
    market_state: tuple[str, ...]
    missing_suffix: str = "__missing"
    version: str = "tabular-layout-v1"

    def __post_init__(self) -> None:
        groups = self.core + self.fundamental + self.market_state
        if not all(groups) or len(groups) != len(set(groups)):
            raise ContractError("feature groups must be non-empty, unique and disjoint")
        if not self.missing_suffix or not self.version:
            raise ContractError("layout suffix and version are required")

    @property
    def fundamental_missing(self) -> tuple[str, ...]:
        return tuple(f"{name}{self.missing_suffix}" for name in self.fundamental)

    @property
    def columns(self) -> tuple[str, ...]:
        return self.core + self.fundamental + self.fundamental_missing + self.market_state

    def stable_hash(self) -> str:
        return canonical_hash({"version": self.version, "columns": self.columns})

    def vectorize(
        self,
        values: Mapping[str, float | None],
        missing_flags: Mapping[str, bool],
        gate: FeatureGate,
    ) -> tuple[float | None, ...]:
        expected = set(self.core + self.fundamental + self.market_state)
        if set(values) != expected:
            missing_names = sorted(expected - set(values))
            unknown_names = sorted(set(values) - expected)
            raise ContractError(
                "tabular feature payload must match frozen layout; "
                f"missing={missing_names}, unknown={unknown_names}"
            )
        if set(missing_flags) != expected:
            raise ContractError("one independent missing flag is required for every D0 feature")
        for name in expected:
            if type(missing_flags[name]) is not bool:
                raise ContractError(f"missing flag for {name} must be boolean")
            if missing_flags[name] != (values[name] is None):
                raise ContractError(f"missing flag for {name} disagrees with its value")
        missing_core = sorted(name for name in self.core if missing_flags[name])
        if missing_core:
            raise ContractError(
                f"scoreable tabular row cannot impute mandatory Core inputs: {missing_core}"
            )
        missing_state = sorted(name for name in self.market_state if missing_flags[name])
        if gate.s_enabled and missing_state:
            raise ContractError(
                f"scoreable S-enabled row requires complete shared market state: {missing_state}"
            )

        core_values = tuple(values[name] for name in self.core)
        if gate.f_enabled:
            fundamental_values = tuple(values[name] for name in self.fundamental)
            fundamental_missing = tuple(
                float(missing_flags[name]) for name in self.fundamental
            )
        else:
            fundamental_values = (0.0,) * len(self.fundamental)
            fundamental_missing = (0.0,) * len(self.fundamental)
        state_values = (
            tuple(values[name] for name in self.market_state)
            if gate.s_enabled
            else (0.0,) * len(self.market_state)
        )
        return core_values + fundamental_values + fundamental_missing + state_values


def default_feature_layout() -> FeatureLayout:
    definitions = d0_features()
    return FeatureLayout(
        core=tuple(
            item.name for item in definitions if item.information_class is InformationClass.CORE
        ),
        fundamental=tuple(
            item.name for item in definitions if item.information_class is InformationClass.F
        ),
        market_state=tuple(
            item.name for item in definitions if item.information_class is InformationClass.S
        ),
    )
