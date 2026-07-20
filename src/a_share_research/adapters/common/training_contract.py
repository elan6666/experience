"""Framework-independent contracts for official deep-model runtime fidelity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence

from a_share_research.adapters.common.types import AdapterContractError

DEEP_SEEDS = (20260719, 20260720, 20260721)


@dataclass(frozen=True)
class OfficialSemantics:
    """Frozen author training semantics plus reviewed A-share deviations."""

    loss: str = "MSELoss"
    optimizer: str = "Adam"
    scheduler: str = "upstream_adjust_learning_rate"
    checkpoint_rule: str = "minimum_validation_mse"
    test_visible_during_fit: bool = False
    missing_target_adaptation: str = "observed_and_label_available_slice_then_same_mse"

    def __post_init__(self) -> None:
        expected = (
            self.loss == "MSELoss"
            and self.optimizer == "Adam"
            and self.scheduler == "upstream_adjust_learning_rate"
            and self.checkpoint_rule == "minimum_validation_mse"
            and not self.test_visible_during_fit
            and self.missing_target_adaptation
            == "observed_and_label_available_slice_then_same_mse"
        )
        if not expected:
            raise AdapterContractError("deep runtime drifted from frozen author semantics")


@dataclass(frozen=True)
class RunIsolation:
    model: str
    universe: str
    gate: str
    seed: int
    physical_gpu: int
    output_root: PurePosixPath
    checkpoint_root: PurePosixPath

    def __post_init__(self) -> None:
        expected_gpu = {
            "itransformer": 0,
            "fact": 1,
            "timexer": 0,
            "timepro": 1,
            "s4m": 0,
        }.get(self.model)
        if expected_gpu is None or self.physical_gpu != expected_gpu:
            raise AdapterContractError("deep model is not assigned to its frozen physical GPU")
        if self.seed not in DEEP_SEEDS:
            raise AdapterContractError("deep run seed is outside the frozen three-seed policy")
        suffix = PurePosixPath(self.model, self.universe, self.gate, str(self.seed))
        for name, root in (
            ("output", self.output_root),
            ("checkpoint", self.checkpoint_root),
        ):
            if not root.is_absolute() or root.parts[-4:] != suffix.parts:
                raise AdapterContractError(
                    f"{name} path is not isolated by model/universe/gate/seed"
                )
        if self.output_root == self.checkpoint_root:
            raise AdapterContractError("outputs and checkpoints must use separate roots")


@dataclass(frozen=True)
class DeepRuntimePolicy:
    seeds: tuple[int, ...] = DEEP_SEEDS
    prediction_drop_last: bool = False
    maximum_asset_tokens: int = 653
    projector: str = "shared_linear_per_asset_v1"
    semantics: OfficialSemantics = OfficialSemantics()

    def __post_init__(self) -> None:
        if self.seeds != DEEP_SEEDS:
            raise AdapterContractError("deep runtime must use the frozen three seeds")
        if self.prediction_drop_last:
            raise AdapterContractError("prediction loaders must keep the final partial batch")
        if self.maximum_asset_tokens != 653:
            raise AdapterContractError("maximum stock token capacity must remain frozen at 653")
        if self.projector != "shared_linear_per_asset_v1":
            raise AdapterContractError("unreviewed feature projection")

    def validate_asset_count(self, asset_count: int) -> None:
        if type(asset_count) is not int or not 0 < asset_count <= self.maximum_asset_tokens:
            raise AdapterContractError("asset token count is outside the frozen runtime capacity")


def eligible_target_mask(
    observed: Sequence[Sequence[bool]],
    label_available: Sequence[Sequence[bool]],
) -> tuple[tuple[bool, ...], ...]:
    """Combine only target observation and label availability for MSE selection."""
    if len(observed) != len(label_available) or not observed:
        raise AdapterContractError("target observed/label mask row counts differ or are empty")
    combined: list[tuple[bool, ...]] = []
    for observed_row, label_row in zip(observed, label_available, strict=True):
        if len(observed_row) != len(label_row) or not observed_row:
            raise AdapterContractError("target observed/label mask widths differ or are empty")
        if any(type(value) is not bool for value in (*observed_row, *label_row)):
            raise AdapterContractError("target masks must contain booleans")
        combined.append(
            tuple(
                is_observed and has_label
                for is_observed, has_label in zip(observed_row, label_row, strict=True)
            )
        )
    if not any(any(row) for row in combined):
        raise AdapterContractError("batch has no observed target with an available label")
    return tuple(combined)
