"""Constant-width Core/F/F-missing/S packing with mandatory mask sidecars."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Mapping

from a_share_research.adapters.common.identity import CausalAssetMaster
from a_share_research.adapters.common.types import AdapterContractError
from a_share_research.contracts import MaskBundle, canonical_hash, validate_mask_series
from a_share_research.features.schema import InformationClass, d0_features


class InformationGate(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"

    @property
    def includes_f(self) -> bool:
        return self in {InformationGate.A1, InformationGate.A3}

    @property
    def includes_s(self) -> bool:
        return self in {InformationGate.A2, InformationGate.A3}


@dataclass(frozen=True)
class FeaturePackingSchema:
    """One fixed channel order shared by all A0-A3 gates."""

    core: tuple[str, ...]
    factors: tuple[str, ...]
    state: tuple[str, ...]
    target_feature: str = "return_1d"
    missing_prefix: str = "missing::"

    def __post_init__(self) -> None:
        groups = self.core + self.factors + self.state
        if not self.core or len(set(groups)) != len(groups):
            raise AdapterContractError("feature groups must be non-empty and disjoint")
        if self.target_feature not in self.core:
            raise AdapterContractError("target_feature must be a Core channel")
        if not self.missing_prefix:
            raise AdapterContractError("missing channel prefix cannot be empty")

    @classmethod
    def from_d0(cls, *, target_feature: str = "return_1d") -> FeaturePackingSchema:
        definitions = d0_features()
        return cls(
            core=tuple(
                item.name
                for item in definitions
                if item.information_class is InformationClass.CORE
            ),
            factors=tuple(
                item.name
                for item in definitions
                if item.information_class is InformationClass.F
            ),
            state=tuple(
                item.name
                for item in definitions
                if item.information_class is InformationClass.S
            ),
            target_feature=target_feature,
        )

    @property
    def channels(self) -> tuple[str, ...]:
        return (
            self.core
            + self.factors
            + tuple(f"{self.missing_prefix}{name}" for name in self.factors)
            + self.state
        )

    @property
    def stable_hash(self) -> str:
        return canonical_hash(
            {
                "core": self.core,
                "factors": self.factors,
                "state": self.state,
                "target_feature": self.target_feature,
                "missing_prefix": self.missing_prefix,
            }
        )


@dataclass(frozen=True)
class PanelWindow:
    """Normalized/imputed-free causal values before information gating."""

    dates: tuple[date, ...]
    asset_master: CausalAssetMaster
    values: Mapping[str, tuple[tuple[float | None, ...], ...]]
    masks: tuple[MaskBundle, ...]

    def __post_init__(self) -> None:
        if not self.dates or self.dates != tuple(sorted(set(self.dates))):
            raise AdapterContractError("panel dates must be unique and increasing")
        if len(self.masks) != len(self.dates):
            raise AdapterContractError("one mandatory MaskBundle is required per panel date")
        validate_mask_series(self.masks)
        for signal_date, bundle in zip(self.dates, self.masks, strict=True):
            if bundle.signal_date != signal_date:
                raise AdapterContractError("mask date does not match panel date")
            if bundle.asset_ids != self.asset_master.asset_ids:
                raise AdapterContractError("panel mask slots do not match the causal master")
        slot_count = len(self.asset_master.asset_ids)
        for feature_name, grid in self.values.items():
            if len(grid) != len(self.dates):
                raise AdapterContractError(f"date count mismatch for feature {feature_name}")
            if any(len(row) != slot_count for row in grid):
                raise AdapterContractError(f"slot count mismatch for feature {feature_name}")
            for row in grid:
                for value in row:
                    if value is not None and not math.isfinite(value):
                        raise AdapterContractError(f"non-finite input for feature {feature_name}")


@dataclass(frozen=True)
class PackedWindow:
    """Feature-major cube plus unchanged mandatory mask evidence."""

    dates: tuple[date, ...]
    asset_ids: tuple[str, ...]
    channels: tuple[str, ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]
    masks: tuple[MaskBundle, ...]
    gate: InformationGate
    schema_hash: str
    target_channel_index: int

    @property
    def native_variate_count(self) -> int:
        """Legacy flattened width; never pass this width to a deep backbone."""
        return len(self.asset_ids) * len(self.channels)

    @property
    def model_variate_count(self) -> int:
        """One upstream variate/token per stock after shared feature projection."""
        return len(self.asset_ids)

    @property
    def input_channel_count(self) -> int:
        return len(self.channels)

    def projector_values(self) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Return `(lookback, assets, channels)` for the shared per-stock projector."""
        return self.values

    def observed_values(self) -> tuple[tuple[bool, ...], ...]:
        """Return the causal input-observation mask aligned to projector values."""
        return tuple(bundle.observed for bundle in self.masks)

    def flattened_values(self) -> tuple[tuple[float, ...], ...]:
        """Return the reversible legacy layout for audit only, not model input."""
        return tuple(
            tuple(value for asset_values in date_values for value in asset_values)
            for date_values in self.values
        )

    def target_variate_indices(self) -> tuple[int, ...]:
        width = len(self.channels)
        return tuple(
            slot * width + self.target_channel_index for slot in range(len(self.asset_ids))
        )

    def config_shape_hash(self) -> str:
        """Gate-independent model-shape identity used by information ablations."""
        return canonical_hash(
            {
                "asset_ids": self.asset_ids,
                "channels": self.channels,
                "model_variate_count": self.model_variate_count,
                "input_channel_count": self.input_channel_count,
                "projection": "shared_linear_per_asset_v1",
                "target_channel_index": self.target_channel_index,
                "schema_hash": self.schema_hash,
            }
        )


def _required_features(schema: FeaturePackingSchema) -> set[str]:
    return set(schema.core + schema.factors + schema.state)


def pack_feature_window(
    panel: PanelWindow,
    *,
    schema: FeaturePackingSchema,
    gate: InformationGate,
    impute_value: float = 0.0,
) -> PackedWindow:
    """Pack every gate to the same shape; only its information values differ."""
    if not isinstance(gate, InformationGate):
        raise AdapterContractError("gate must use InformationGate")
    if not math.isfinite(impute_value):
        raise AdapterContractError("impute_value must be finite")
    missing_features = _required_features(schema) - set(panel.values)
    if missing_features:
        raise AdapterContractError(f"panel lacks required features: {sorted(missing_features)}")
    for bundle in panel.masks:
        missing_masks = _required_features(schema) - set(bundle.feature_missing)
        if missing_masks:
            raise AdapterContractError(
                f"mandatory per-feature missing masks absent: {sorted(missing_masks)}"
            )
    for date_index, bundle in enumerate(panel.masks):
        for name in _required_features(schema):
            for slot, value in enumerate(panel.values[name][date_index]):
                if (value is None) != bundle.feature_missing[name][slot]:
                    raise AdapterContractError(
                        f"value/missing-mask mismatch for {name} on {panel.dates[date_index]}"
                    )
        for name in schema.state:
            observed_values = {
                value
                for slot, value in enumerate(panel.values[name][date_index])
                if bundle.observed[slot] and value is not None
            }
            if len(observed_values) > 1:
                raise AdapterContractError(
                    f"shared CSI300 state {name} differs across assets on {panel.dates[date_index]}"
                )

    packed_dates: list[tuple[tuple[float, ...], ...]] = []
    for date_index, bundle in enumerate(panel.masks):
        packed_assets: list[tuple[float, ...]] = []
        for slot in range(len(panel.asset_master.asset_ids)):
            observed = bundle.observed[slot]
            channel_values: list[float] = []
            for name in schema.core:
                value = panel.values[name][date_index][slot]
                channel_values.append(value if observed and value is not None else impute_value)
            for name in schema.factors:
                value = panel.values[name][date_index][slot]
                active = gate.includes_f and observed
                channel_values.append(value if active and value is not None else impute_value)
            for name in schema.factors:
                is_missing = bundle.feature_missing[name][slot]
                channel_values.append(float(is_missing) if gate.includes_f and observed else 0.0)
            for name in schema.state:
                value = panel.values[name][date_index][slot]
                channel_values.append(
                    value if gate.includes_s and observed and value is not None else impute_value
                )
            if any(not math.isfinite(value) for value in channel_values):
                raise AdapterContractError("packed feature cube contains a non-finite value")
            packed_assets.append(tuple(channel_values))
        packed_dates.append(tuple(packed_assets))

    return PackedWindow(
        dates=panel.dates,
        asset_ids=panel.asset_master.asset_ids,
        channels=schema.channels,
        values=tuple(packed_dates),
        masks=panel.masks,
        gate=gate,
        schema_hash=schema.stable_hash,
        target_channel_index=schema.channels.index(schema.target_feature),
    )


def assert_constant_parameter_count(parameter_counts: Mapping[InformationGate, int]) -> None:
    """Fail if an A0-A3 run changed architecture under an information ablation."""
    if set(parameter_counts) != set(InformationGate):
        raise AdapterContractError("parameter-count evidence must cover A0, A1, A2 and A3")
    counts = tuple(parameter_counts[gate] for gate in InformationGate)
    if any(type(count) is not int or count <= 0 for count in counts):
        raise AdapterContractError("parameter counts must be positive integers")
    if len(set(counts)) != 1:
        raise AdapterContractError(
            "A0-A3 parameter counts differ; this is not an information ablation"
        )
