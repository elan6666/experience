"""Independent mask semantics and stable asset identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

from a_share_research.contracts.base import CanonicalModel, ContractError
from a_share_research.contracts.data import _validate_ts_code


@dataclass(frozen=True)
class AssetRegistry(CanonicalModel):
    """Permanent ordered identity; appending assets never reorders existing slots."""

    SCHEMA_NAME: ClassVar[str] = "asset_registry"

    asset_ids: tuple[str, ...]
    identity_version: str = "ts_code-v1"

    def validate(self) -> None:
        if not self.asset_ids:
            raise ContractError("asset registry cannot be empty")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ContractError("asset registry contains duplicate identities")
        for ts_code in self.asset_ids:
            _validate_ts_code(ts_code)
        if not self.identity_version:
            raise ContractError("identity_version is required")

    def index_of(self, ts_code: str) -> int:
        try:
            return self.asset_ids.index(ts_code)
        except ValueError as error:
            raise ContractError(f"unknown asset identity: {ts_code}") from error

    def append(self, *new_asset_ids: str) -> AssetRegistry:
        additions = tuple(ts_code for ts_code in new_asset_ids if ts_code not in self.asset_ids)
        return AssetRegistry(self.asset_ids + additions, self.identity_version)


@dataclass(frozen=True)
class MaskBundle(CanonicalModel):
    """Masks remain separate even when downstream code combines them."""

    SCHEMA_NAME: ClassVar[str] = "mask_bundle"

    signal_date: date
    asset_ids: tuple[str, ...]
    asset_registry_hash: str
    member: tuple[bool, ...]
    observed: tuple[bool, ...]
    feature_missing: dict[str, tuple[bool, ...]]
    label_available: tuple[bool, ...]
    buyable: tuple[bool, ...]
    sellable: tuple[bool, ...]
    loss: tuple[bool, ...]
    evaluation: tuple[bool, ...]

    def validate(self) -> None:
        registry = AssetRegistry(self.asset_ids)
        if self.asset_registry_hash != registry.stable_hash():
            raise ContractError("asset_registry_hash does not match permanent identity")
        size = len(self.asset_ids)
        named_masks = {
            "member": self.member,
            "observed": self.observed,
            "label_available": self.label_available,
            "buyable": self.buyable,
            "sellable": self.sellable,
            "loss": self.loss,
            "evaluation": self.evaluation,
        }
        if not self.feature_missing:
            raise ContractError("feature_missing must contain one marker per input feature")
        for feature_name, mask in self.feature_missing.items():
            if not feature_name:
                raise ContractError("feature missing-mask names cannot be empty")
            named_masks[f"feature_missing.{feature_name}"] = mask
        for name, mask in named_masks.items():
            if len(mask) != size:
                raise ContractError(f"{name} length does not match asset identity")
            if any(type(value) is not bool for value in mask):
                raise ContractError(f"{name} must contain booleans only")
        for index in range(size):
            if self.buyable[index] and not (self.member[index] and self.observed[index]):
                raise ContractError("buyable requires member and observed")
            if self.sellable[index] and not self.observed[index]:
                raise ContractError("sellable requires observed")
            if self.loss[index] and not (self.observed[index] and self.label_available[index]):
                raise ContractError("loss mask requires observed label")
            if self.evaluation[index] and not (
                self.member[index] and self.observed[index] and self.label_available[index]
            ):
                raise ContractError("evaluation mask requires member, observation and label")

    def combined_training_mask(self) -> tuple[bool, ...]:
        """Derive a convenience mask without mutating any source truth mask."""
        return tuple(
            member and observed and labelled and include
            for member, observed, labelled, include in zip(
                self.member,
                self.observed,
                self.label_available,
                self.loss,
                strict=True,
            )
        )

    def with_imputed_values(
        self, values: dict[str, tuple[float, ...]]
    ) -> dict[str, tuple[float, ...]]:
        """Validate imputed shapes; observation and missing truth remain unchanged."""
        if set(values) != set(self.feature_missing):
            raise ContractError("imputed feature set must match per-feature missing masks exactly")
        for feature_name, feature_values in values.items():
            if len(feature_values) != len(self.asset_ids):
                raise ContractError(f"imputed feature length mismatch: {feature_name}")
        return {name: tuple(values[name]) for name in sorted(values)}


def validate_mask_series(bundles: tuple[MaskBundle, ...]) -> None:
    """Reject any cross-date slot reordering or registry drift."""
    if not bundles:
        raise ContractError("mask series cannot be empty")
    dates = tuple(bundle.signal_date for bundle in bundles)
    if dates != tuple(sorted(set(dates))):
        raise ContractError("mask series dates must be unique and increasing")
    previous_ids = bundles[0].asset_ids
    bundles[0].validate()
    for bundle in bundles[1:]:
        bundle.validate()
        if len(bundle.asset_ids) < len(previous_ids):
            raise ContractError("asset identity was deleted across mask dates")
        if bundle.asset_ids[: len(previous_ids)] != previous_ids:
            raise ContractError("existing asset identity reordered across mask dates")
        previous_ids = bundle.asset_ids
