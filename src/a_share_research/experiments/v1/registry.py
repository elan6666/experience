"""Plan010 V1 registry derived from the sealed Plan009 V0 registry.

The module freezes intent only. It has no trainer, evaluator, result reader, or
access to validation/legacy labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from a_share_research.contracts.base import CanonicalModel, ContractError, canonical_hash
from a_share_research.experiments.v0 import (
    BoundV0Cell,
    BoundV0Registry,
    CellState,
    FrozenV0Cell,
    FrozenV0Registry,
)
from a_share_research.protocol.splits import Partition

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODELS = ("ridge", "lightgbm", "itransformer", "fact", "timepro", "timexer", "s4m")
UNIVERSES = ("CSI300", "STAR50", "TECH32", "TECH100")
BLOCKED_LICENSE_MODELS = frozenset({"s4m"})
EXPLORATORY_UNIVERSES = frozenset({"TECH32", "TECH100"})


class InformationSet(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"

    @property
    def groups(self) -> tuple[str, ...]:
        return {
            InformationSet.A0: ("CORE",),
            InformationSet.A1: ("CORE", "F", "F_MISSING"),
            InformationSet.A2: ("CORE", "S"),
            InformationSet.A3: ("CORE", "F", "F_MISSING", "S"),
        }[self]


class CellDisposition(str, Enum):
    RUNNABLE = "RUNNABLE"
    EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
    BLOCKED_DATA = "BLOCKED_DATA"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"


class CellAction(str, Enum):
    REFERENCE_V0 = "REFERENCE_V0"
    TRAIN_INCREMENTAL = "TRAIN_INCREMENTAL"
    RECORD_BLOCK = "RECORD_BLOCK"


@dataclass(frozen=True)
class TrainingSignature(CanonicalModel):
    """Gate-independent fields that must be equal across one A0-A3 family."""

    SCHEMA_NAME: ClassVar[str] = "v1_training_signature"

    model_capacity_hash: str
    model_hyperparameters_hash: str
    train_window: tuple[str, str]
    validation_window: tuple[str, str]
    label: str
    frequency: str
    optimizer: str
    max_epochs: int | None
    early_stopping: str
    seeds: tuple[int, ...]

    def validate(self) -> None:
        for name in ("model_capacity_hash", "model_hyperparameters_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be a lowercase SHA-256")
        if self.train_window != ("2019-01-01", "2024-12-31"):
            raise ContractError("V1 must inherit the frozen 2019-2024 training window")
        if self.validation_window != ("2025-01-01", "2025-12-31"):
            raise ContractError("V1 selection must use paired 2025 validation only")
        if self.label != "future_5d_open_to_open_excess_return":
            raise ContractError("V1 primary label must remain frozen at future 5-day excess return")
        if self.frequency != "WEEKLY":
            raise ContractError("V1 primary frequency must remain WEEKLY")
        if not self.optimizer or not self.early_stopping:
            raise ContractError("optimizer and early_stopping must be explicit")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ContractError("max_epochs must be positive when applicable")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ContractError("seeds must be non-empty and unique")


@dataclass(frozen=True)
class V1Cell(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v1_information_ablation_cell"

    cell_id: str
    model: str
    universe: str
    information_set: InformationSet
    information_groups: tuple[str, ...]
    config_hash: str
    parent_v0_cell_id: str
    parent_v0_config_hash: str
    parent_v0_registry_hash: str
    training_signature: TrainingSignature
    market_state_hash: str
    disposition: CellDisposition
    action: CellAction
    scope: str

    def validate(self) -> None:
        if self.model not in MODELS or self.universe not in UNIVERSES:
            raise ContractError("V1 cell is outside the frozen model/universe matrix")
        expected_id = (
            self.parent_v0_cell_id
            if self.information_set is InformationSet.A0
            else f"v1-{self.information_set.value.lower()}-{self.universe.lower()}-{self.model}"
        )
        if self.cell_id != expected_id:
            raise ContractError("V1 cell_id is not canonical")
        if self.information_groups != self.information_set.groups:
            raise ContractError("information groups do not match the frozen A0-A3 gate")
        for name in (
            "config_hash",
            "parent_v0_config_hash",
            "parent_v0_registry_hash",
            "market_state_hash",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be a lowercase SHA-256")
        if self.information_set is InformationSet.A0:
            if self.action is not CellAction.REFERENCE_V0:
                raise ContractError("A0 is reference-only and must never be retrained in V1")
            if self.config_hash != self.parent_v0_config_hash:
                raise ContractError("A0 config hash must exactly equal the frozen V0 hash")
        elif self.model in BLOCKED_LICENSE_MODELS:
            if self.action is not CellAction.RECORD_BLOCK:
                raise ContractError("license-blocked increments can only record the block")
        elif self.disposition is CellDisposition.BLOCKED_DATA:
            if self.action is not CellAction.RECORD_BLOCK:
                raise ContractError("data-blocked increments can only record the block")
        elif self.action is not CellAction.TRAIN_INCREMENTAL:
            raise ContractError("A1-A3 executable cells must be incremental training jobs")
        if self.model in BLOCKED_LICENSE_MODELS:
            if self.disposition is not CellDisposition.BLOCKED_LICENSE:
                raise ContractError("all gates for blocked models must stay BLOCKED_LICENSE")
        elif self.disposition is CellDisposition.BLOCKED_LICENSE:
            raise ContractError("no model may carry BLOCKED_LICENSE disposition")
        expected_scope = "EXPLORATORY_ONLY" if self.universe in EXPLORATORY_UNIVERSES else "FORMAL"
        if self.scope != expected_scope:
            raise ContractError("universe formal/exploratory scope drifted")


@dataclass(frozen=True)
class ComparisonPair(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v1_comparison_pair"

    candidate: InformationSet
    reference: InformationSet
    split: Partition = Partition.VALIDATION
    paired_support: bool = True

    def validate(self) -> None:
        if self.candidate is self.reference:
            raise ContractError("comparison pair must contain two different information sets")
        if self.split is not Partition.VALIDATION or not self.paired_support:
            raise ContractError("V1 comparisons must be paired on 2025 validation support")


FROZEN_COMPARISONS = (
    ComparisonPair(InformationSet.A1, InformationSet.A0),
    ComparisonPair(InformationSet.A2, InformationSet.A0),
    ComparisonPair(InformationSet.A3, InformationSet.A0),
    ComparisonPair(InformationSet.A3, InformationSet.A1),
    ComparisonPair(InformationSet.A3, InformationSet.A2),
)

_FROZEN_TRAINING_FIELDS = (
    "model_capacity_hash",
    "model_hyperparameters_hash",
    "train_window",
    "validation_window",
    "label",
    "frequency",
    "optimizer",
    "max_epochs",
    "early_stopping",
    "seeds",
)


@dataclass(frozen=True)
class V1Registry(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v1_information_ablation_registry"

    parent_v0_registry_hash: str
    cells: tuple[V1Cell, ...]
    comparisons: tuple[ComparisonPair, ...] = FROZEN_COMPARISONS
    selection_split: Partition = Partition.VALIDATION
    legacy_viewed_selection_allowed: bool = False

    def validate(self) -> None:
        if not _SHA256.fullmatch(self.parent_v0_registry_hash):
            raise ContractError("parent_v0_registry_hash must be a lowercase SHA-256")
        if len(self.cells) != 112:
            raise ContractError("V1 registry must contain exactly 112 cells")
        expected = {
            (model, universe, information_set)
            for model in MODELS
            for universe in UNIVERSES
            for information_set in InformationSet
        }
        actual = {(cell.model, cell.universe, cell.information_set) for cell in self.cells}
        if actual != expected:
            raise ContractError("V1 registry must cover 7 models x 4 universes x 4 gates")
        if len({cell.cell_id for cell in self.cells}) != 112:
            raise ContractError("V1 cell ids must be unique")
        a0 = tuple(cell for cell in self.cells if cell.information_set is InformationSet.A0)
        increments = tuple(
            cell for cell in self.cells if cell.information_set is not InformationSet.A0
        )
        if len(a0) != 28 or any(cell.action is not CellAction.REFERENCE_V0 for cell in a0):
            raise ContractError("V1 must reference exactly 28 V0 A0 cells without rerunning")
        if len(increments) != 84:
            raise ContractError("V1 must account for exactly 84 A1/A2/A3 increments")
        if {cell.parent_v0_registry_hash for cell in self.cells} != {
            self.parent_v0_registry_hash
        }:
            raise ContractError("every V1 cell must reference the same sealed V0 registry")
        if len({cell.market_state_hash for cell in self.cells}) != 1:
            raise ContractError("all 112 cells must share one independent CSI300 market_state_hash")
        for model in MODELS:
            for universe in UNIVERSES:
                family = tuple(
                    cell
                    for cell in self.cells
                    if cell.model == model and cell.universe == universe
                )
                if len({cell.training_signature.stable_hash() for cell in family}) != 1:
                    raise ContractError(
                        "A0-A3 changed capacity/training semantics; only information gate may vary"
                    )
                if len({cell.parent_v0_config_hash for cell in family}) != 1:
                    raise ContractError("A0-A3 must share one frozen V0 parent config")
        blocked = tuple(cell for cell in self.cells if cell.model in BLOCKED_LICENSE_MODELS)
        if {cell.model for cell in blocked} != set(BLOCKED_LICENSE_MODELS) or any(
            cell.disposition is not CellDisposition.BLOCKED_LICENSE for cell in blocked
        ):
            raise ContractError(
                "license-blocked models must be exactly the frozen set and stay BLOCKED_LICENSE"
            )
        exploratory = tuple(cell for cell in self.cells if cell.universe in EXPLORATORY_UNIVERSES)
        if any(cell.scope != "EXPLORATORY_ONLY" for cell in exploratory):
            raise ContractError("tech32/tech100 must remain EXPLORATORY_ONLY")
        if self.comparisons != FROZEN_COMPARISONS:
            raise ContractError("V1 comparison family is frozen and cannot expand after results")
        if self.selection_split is not Partition.VALIDATION:
            raise ContractError("V1 selection is restricted to 2025 validation")
        if self.legacy_viewed_selection_allowed:
            raise ContractError("legacy-viewed 2026 data can never select V1")


def build_v1_registry(
    v0: FrozenV0Registry,
    bound_v0: BoundV0Registry,
    *,
    training_signatures: Mapping[str, TrainingSignature],
) -> V1Registry:
    """Expand 28 evidence-bound A0 references into 112 frozen V1 intents."""

    v0.validate()
    bound_v0.validate()
    if bound_v0.frozen_registry_hash != v0.stable_hash():
        raise ContractError("bound V0 evidence belongs to another frozen registry")
    if set(training_signatures) != set(MODELS):
        raise ContractError("V1 training signatures must account for all seven models")
    for model, signature in training_signatures.items():
        signature.validate()
        expected_seeds = next(cell.seeds for cell in v0.cells if cell.model == model)
        if signature.seeds != expected_seeds:
            raise ContractError("V1 seeds must exactly inherit the V0 model policy")
    bound_by_id = {cell.frozen.cell_id: cell for cell in bound_v0.cells}
    if set(bound_by_id) != set(v0.by_id):
        raise ContractError("bound V0 evidence lost or added a frozen A0 cell")
    market_hashes = {cell.market_state_hash for cell in bound_v0.cells}
    if len(market_hashes) != 1:
        raise ContractError("V1 requires one shared CSI300 market_state_hash")
    parent_hash = v0.stable_hash()
    cells: list[V1Cell] = []
    for parent in sorted(v0.cells, key=_v0_sort_key):
        bound = bound_by_id[parent.cell_id]
        if bound.frozen != parent:
            raise ContractError("bound V0 cell differs from its frozen A0 reference")
        signature = training_signatures[parent.model]
        for information_set in InformationSet:
            disposition = _disposition(parent, bound)
            action = _action(information_set, disposition)
            config_hash = (
                parent.model_config_hash
                if information_set is InformationSet.A0
                else canonical_hash(
                    {
                        "parent_v0_model_config_hash": parent.model_config_hash,
                        "parent_v0_adapter_hash": parent.adapter_hash,
                        "training_signature_hash": signature.stable_hash(),
                        "information_set": information_set.value,
                        "information_groups": information_set.groups,
                        "market_state_hash": bound.market_state_hash,
                    }
                )
            )
            cells.append(
                V1Cell(
                    cell_id=(
                        parent.cell_id
                        if information_set is InformationSet.A0
                        else f"v1-{information_set.value.lower()}-"
                        f"{parent.universe.lower()}-{parent.model}"
                    ),
                    model=parent.model,
                    universe=parent.universe,
                    information_set=information_set,
                    information_groups=information_set.groups,
                    config_hash=config_hash,
                    parent_v0_cell_id=parent.cell_id,
                    parent_v0_config_hash=parent.model_config_hash,
                    parent_v0_registry_hash=parent_hash,
                    training_signature=signature,
                    market_state_hash=bound.market_state_hash,
                    disposition=disposition,
                    action=action,
                    scope=(
                        "EXPLORATORY_ONLY"
                        if parent.universe in EXPLORATORY_UNIVERSES
                        else "FORMAL"
                    ),
                )
            )
    return V1Registry(parent_v0_registry_hash=parent_hash, cells=tuple(cells))


def load_v1_blueprint(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("V1 blueprint must be a JSON object")
    validate_v1_blueprint(payload)
    return MappingProxyType(payload)


def validate_v1_blueprint(blueprint: Mapping[str, Any]) -> None:
    """Fail closed if cardinality, gates, comparisons, or holdout rules drift."""

    parent = blueprint.get("parent")
    if not isinstance(parent, dict) or (
        parent.get("a0_policy"),
        parent.get("a0_expected_cells"),
        parent.get("incremental_expected_cells"),
        parent.get("complete_expected_cells"),
    ) != ("REFERENCE_ONLY", 28, 84, 112):
        raise ContractError("V1 parent counts or A0 reference policy drifted")
    if tuple(blueprint.get("models", ())) != MODELS:
        raise ContractError("V1 blueprint must contain the frozen seven-model order")
    if tuple(blueprint.get("universes", ())) != UNIVERSES:
        raise ContractError("V1 blueprint must contain four separate frozen universes")
    expected_gates = {gate.value: list(gate.groups) for gate in InformationSet}
    if blueprint.get("information_sets") != expected_gates:
        raise ContractError("V1 information gates drifted")
    if tuple(blueprint.get("frozen_training_fields", ())) != _FROZEN_TRAINING_FIELDS:
        raise ContractError("V1 frozen training fields drifted")
    primary = blueprint.get("primary_protocol")
    if not isinstance(primary, dict) or (
        primary.get("train"),
        primary.get("paired_validation"),
        primary.get("label"),
        primary.get("frequency"),
    ) != (
        ["2019-01-01", "2024-12-31"],
        ["2025-01-01", "2025-12-31"],
        "future_5d_open_to_open_excess_return",
        "WEEKLY",
    ):
        raise ContractError("V1 primary split/label/frequency drifted")
    if tuple(blueprint.get("blocked_license_models", ())) != ("s4m",):
        raise ContractError("V1 blocked-license model set drifted")
    if tuple(blueprint.get("exploratory_only_universes", ())) != (
        "TECH32",
        "TECH100",
    ):
        raise ContractError("V1 exploratory universe set drifted")
    if tuple(blueprint.get("comparisons", ())) != (
        "A1-A0",
        "A2-A0",
        "A3-A0",
        "A3-A1",
        "A3-A2",
    ):
        raise ContractError("V1 comparison family drifted")
    selection = blueprint.get("selection")
    if not isinstance(selection, dict) or selection != {
        "allowed_split": "VALIDATION",
        "legacy_viewed_2026_allowed": False,
        "future_unseen_allowed": False,
    }:
        raise ContractError("V1 selection may use only 2025 validation")
    model_training = blueprint.get("model_training")
    if not isinstance(model_training, dict) or set(model_training) != set(MODELS):
        raise ContractError("V1 model_training must account for all seven models")
    for model in MODELS:
        raw = model_training[model]
        if not isinstance(raw, dict) or set(raw) != {
            "optimizer",
            "max_epochs",
            "early_stopping",
        }:
            raise ContractError(f"V1 model_training is incomplete for {model}")
        if not raw["optimizer"] or not raw["early_stopping"]:
            raise ContractError(f"V1 optimizer/early stopping is empty for {model}")
        if raw["max_epochs"] is not None and (
            not isinstance(raw["max_epochs"], int) or raw["max_epochs"] <= 0
        ):
            raise ContractError(f"V1 max_epochs is invalid for {model}")
    if blueprint.get("execution") != "REGISTRY_ONLY_NO_TRAINING":
        raise ContractError("V1 blueprint must not claim to implement training")


def training_signatures_from_blueprint(
    v0: FrozenV0Registry,
    blueprint: Mapping[str, Any],
) -> Mapping[str, TrainingSignature]:
    """Bind explicit optimizer/stopping fields to the actual frozen V0 hashes."""

    v0.validate()
    validate_v1_blueprint(blueprint)
    raw_training = blueprint["model_training"]
    signatures: dict[str, TrainingSignature] = {}
    for model in MODELS:
        parent = next(cell for cell in v0.cells if cell.model == model)
        raw = raw_training[model]
        signatures[model] = TrainingSignature(
            model_capacity_hash=canonical_hash(
                {
                    "model_config_hash": parent.model_config_hash,
                    "adapter_hash": parent.adapter_hash,
                }
            ),
            model_hyperparameters_hash=parent.model_config_hash,
            train_window=(v0.train_start.isoformat(), v0.train_end.isoformat()),
            validation_window=(
                v0.validation_start.isoformat(),
                v0.validation_end.isoformat(),
            ),
            label="future_5d_open_to_open_excess_return",
            frequency="WEEKLY",
            optimizer=str(raw["optimizer"]),
            max_epochs=raw["max_epochs"],
            early_stopping=str(raw["early_stopping"]),
            seeds=parent.seeds,
        )
    return MappingProxyType(signatures)


def _disposition(parent: FrozenV0Cell, bound: BoundV0Cell) -> CellDisposition:
    if parent.model in BLOCKED_LICENSE_MODELS:
        return CellDisposition.BLOCKED_LICENSE
    if bound.state in {CellState.BLOCKED_D0, CellState.INVALID_DATA}:
        return CellDisposition.BLOCKED_DATA
    if bound.state is not CellState.RUNNABLE:
        raise ContractError("V1 accepts only runnable or typed preflight-blocked V0 cells")
    if parent.universe in EXPLORATORY_UNIVERSES:
        return CellDisposition.EXPLORATORY_ONLY
    return CellDisposition.RUNNABLE


def _action(
    information_set: InformationSet,
    disposition: CellDisposition,
) -> CellAction:
    if information_set is InformationSet.A0:
        return CellAction.REFERENCE_V0
    if disposition in {CellDisposition.BLOCKED_LICENSE, CellDisposition.BLOCKED_DATA}:
        return CellAction.RECORD_BLOCK
    return CellAction.TRAIN_INCREMENTAL


def _v0_sort_key(cell: FrozenV0Cell) -> tuple[int, int]:
    return MODELS.index(cell.model), UNIVERSES.index(cell.universe)
