"""Static V1 registry invariants; execute on the approved server only."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from a_share_research.contracts import ContractError
from a_share_research.contracts.base import canonical_hash
from a_share_research.experiments.v0 import (
    BoundV0Cell,
    BoundV0Registry,
    CellState,
    FrozenV0Cell,
    FrozenV0Registry,
    build_frozen_v0_registry,
    load_v0_blueprint,
)
from a_share_research.experiments.v1 import (
    CellAction,
    CellDisposition,
    InformationSet,
    V1Registry,
    build_v1_registry,
    load_v1_blueprint,
    training_signatures_from_blueprint,
)
from a_share_research.protocol import Partition

HASH = "a" * 64
MODELS = ("ridge", "lightgbm", "itransformer", "fact", "timepro", "timexer", "s4m")
UNIVERSES = ("CSI300", "STAR50", "TECH32", "TECH90")
BLOCKED = {"s4m"}
DEEP = {"itransformer", "fact", "timepro", "timexer", "s4m"}


def frozen_v0() -> FrozenV0Registry:
    cells = []
    for universe in UNIVERSES:
        for model in MODELS:
            blocked = model in BLOCKED
            cells.append(
                FrozenV0Cell(
                    cell_id=f"v0-a0-{universe.lower()}-{model}",
                    model=model,
                    universe=universe,
                    information_set="A0",
                    information_groups=("CORE",),
                    lane={
                        "ridge": "CPU",
                        "lightgbm": "CPU",
                        "itransformer": "GPU0",
                        "fact": "GPU1",
                        "timepro": "GPU1",
                        "timexer": "GPU0",
                        "s4m": "NONE",
                    }[model],
                    seeds=(20260719, 20260720, 20260721) if model in DEEP else (20260719,),
                    initial_state=(
                        CellState.BLOCKED_LICENSE
                        if blocked
                        else CellState.RUNNABLE_PENDING_D0
                    ),
                    model_config_hash=canonical_hash({"model": model, "config": "v0"}),
                    adapter_hash=canonical_hash({"model": model, "adapter": "v0"}),
                    upstream_registry_hash="b" * 64,
                    upstream_commit=(
                        f"internal:{model}-v1" if model in {"ridge", "lightgbm"} else "c" * 40
                    ),
                    d0_manifest_path="/data/yilangliu/a_share_research/data/manifests/d0-v1.json",
                    environment_receipt_path=(
                        None
                        if blocked
                        else f"/data/yilangliu/a_share_research/receipts/{model}-env.json"
                    ),
                    integrity_receipt_path=(
                        None
                        if blocked
                        else f"/data/yilangliu/a_share_research/receipts/{model}-integrity.json"
                    ),
                    license_receipt_path=(
                        f"/data/yilangliu/a_share_research/receipts/{model}-license.json"
                        if blocked
                        else None
                    ),
                    output_root="/data/yilangliu/a_share_research/runs/v0",
                    checkpoint_root="/data/yilangliu/a_share_research/checkpoints/v0",
                )
            )
    registry = FrozenV0Registry(
        schema_version="v0_registry_v1",
        protocol_version="v1",
        frequency="weekly",
        horizon=5,
        entry="T+1_OPEN",
        train_start=date(2019, 1, 1),
        train_end=date(2024, 12, 31),
        validation_start=date(2025, 1, 1),
        validation_end=date(2025, 12, 31),
        legacy_viewed_start=date(2026, 1, 1),
        legacy_viewed_end=date(2026, 7, 17),
        legacy_policy="REPORT_ONLY_NO_SELECTION",
        cells=tuple(cells),
    )
    registry.validate()
    return registry


def bound_v0(registry: FrozenV0Registry, *, market_state_hash: str = HASH) -> BoundV0Registry:
    cells = []
    for parent in registry.cells:
        blocked = parent.model in BLOCKED
        cells.append(
            BoundV0Cell(
                frozen=parent,
                state=CellState.BLOCKED_LICENSE if blocked else CellState.RUNNABLE,
                d0_manifest_hash="d" * 64,
                asset_registry_hash="e" * 64,
                execution_calendar_manifest_hash="f" * 64,
                feature_schema_hash="1" * 64,
                market_state_hash=market_state_hash,
                environment_receipt_hash=None if blocked else "2" * 64,
                integrity_receipt_hash=None if blocked else "3" * 64,
                license_receipt_hash="4" * 64 if blocked else None,
            )
        )
    bound = BoundV0Registry(registry.stable_hash(), tuple(cells))
    bound.validate()
    return bound


def blueprint_path() -> Path:
    return Path(__file__).parents[2] / "configs/experiments/v1/information-ablation-v1.json"


def project_root() -> Path:
    return Path(__file__).parents[2]


def registry() -> V1Registry:
    parent = frozen_v0()
    blueprint = load_v1_blueprint(blueprint_path())
    signatures = training_signatures_from_blueprint(parent, blueprint)
    return build_v1_registry(parent, bound_v0(parent), training_signatures=signatures)


def test_v1_is_28_references_plus_84_increments_without_a0_rerun() -> None:
    result = registry()
    assert len(result.cells) == 112
    a0 = [cell for cell in result.cells if cell.information_set is InformationSet.A0]
    increments = [cell for cell in result.cells if cell.information_set is not InformationSet.A0]
    assert len(a0) == 28
    assert len(increments) == 84
    assert {cell.action for cell in a0} == {CellAction.REFERENCE_V0}
    assert all(cell.config_hash == cell.parent_v0_config_hash for cell in a0)


def test_v1_directly_consumes_the_real_v0_blueprint_registry() -> None:
    root = project_root()
    parent = build_frozen_v0_registry(
        project_root=root,
        blueprint=load_v0_blueprint(root / "configs/experiments/v0/registry-v1.json"),
    )
    blueprint = load_v1_blueprint(blueprint_path())
    result = build_v1_registry(
        parent,
        bound_v0(parent),
        training_signatures=training_signatures_from_blueprint(parent, blueprint),
    )
    assert result.parent_v0_registry_hash == parent.stable_hash()
    assert {
        cell.parent_v0_cell_id
        for cell in result.cells
        if cell.information_set is InformationSet.A0
    } == set(parent.by_id)


def test_only_information_gate_changes_within_each_model_universe_family() -> None:
    result = registry()
    for model in MODELS:
        for universe in UNIVERSES:
            family = [
                cell
                for cell in result.cells
                if cell.model == model and cell.universe == universe
            ]
            assert len({cell.training_signature.stable_hash() for cell in family}) == 1
            assert {cell.information_set for cell in family} == set(InformationSet)
            assert {cell.information_groups for cell in family} == {
                gate.groups for gate in InformationSet
            }


def test_shared_market_state_blocked_models_and_exploratory_pools_are_typed() -> None:
    result = registry()
    assert {cell.market_state_hash for cell in result.cells} == {HASH}
    blocked = [cell for cell in result.cells if cell.model in BLOCKED]
    assert len(blocked) == 16
    assert {cell.disposition for cell in blocked} == {CellDisposition.BLOCKED_LICENSE}
    assert {
        cell.action for cell in blocked if cell.information_set is not InformationSet.A0
    } == {CellAction.RECORD_BLOCK}
    exploratory = [cell for cell in result.cells if cell.universe in {"TECH32", "TECH90"}]
    assert {cell.scope for cell in exploratory} == {"EXPLORATORY_ONLY"}
    assert {
        cell.disposition for cell in exploratory if cell.model not in BLOCKED
    } == {CellDisposition.EXPLORATORY_ONLY}


def test_comparison_family_is_paired_2025_and_legacy_cannot_select() -> None:
    result = registry()
    assert [(pair.candidate.value, pair.reference.value) for pair in result.comparisons] == [
        ("A1", "A0"),
        ("A2", "A0"),
        ("A3", "A0"),
        ("A3", "A1"),
        ("A3", "A2"),
    ]
    assert {pair.split for pair in result.comparisons} == {Partition.VALIDATION}
    assert all(pair.paired_support for pair in result.comparisons)
    assert result.selection_split is Partition.VALIDATION
    assert not result.legacy_viewed_selection_allowed


def test_registry_rejects_capacity_drift_and_any_a0_retraining() -> None:
    result = registry()
    cells = list(result.cells)
    target = next(
        index
        for index, cell in enumerate(cells)
        if cell.model == "fact"
        and cell.universe == "CSI300"
        and cell.information_set is InformationSet.A3
    )
    drifted = replace(cells[target].training_signature, model_capacity_hash="9" * 64)
    cells[target] = replace(cells[target], training_signature=drifted)
    with pytest.raises(ContractError, match="capacity/training semantics"):
        V1Registry(parent_v0_registry_hash=result.parent_v0_registry_hash, cells=tuple(cells))

    a0 = next(cell for cell in result.cells if cell.information_set is InformationSet.A0)
    with pytest.raises(ContractError, match="A0 is reference-only"):
        replace(a0, action=CellAction.TRAIN_INCREMENTAL)


def test_builder_rejects_split_market_state_and_signature_seed_drift() -> None:
    parent = frozen_v0()
    blueprint = load_v1_blueprint(blueprint_path())
    signatures = dict(training_signatures_from_blueprint(parent, blueprint))
    bound = bound_v0(parent)
    split = list(bound.cells)
    split[0] = replace(split[0], market_state_hash="8" * 64)
    with pytest.raises(ContractError, match="shared CSI300"):
        build_v1_registry(
            parent,
            BoundV0Registry(parent.stable_hash(), tuple(split)),
            training_signatures=signatures,
        )
    signatures["fact"] = replace(signatures["fact"], seeds=(1, 2, 3))
    with pytest.raises(ContractError, match="seeds must exactly inherit"):
        build_v1_registry(parent, bound, training_signatures=signatures)


def test_blueprint_is_registry_only_and_rejects_comparison_or_holdout_drift() -> None:
    blueprint = dict(load_v1_blueprint(blueprint_path()))
    blueprint["comparisons"] = ["A1-A0"]
    with pytest.raises(ContractError, match="comparison family"):
        training_signatures_from_blueprint(frozen_v0(), blueprint)
    blueprint = dict(load_v1_blueprint(blueprint_path()))
    blueprint["selection"] = {
        "allowed_split": "LEGACY_VIEWED",
        "legacy_viewed_2026_allowed": True,
        "future_unseen_allowed": False,
    }
    with pytest.raises(ContractError, match="2025 validation"):
        training_signatures_from_blueprint(frozen_v0(), blueprint)
