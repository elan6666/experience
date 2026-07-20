"""Synthetic V0 registry checks; execute only on the approved server."""

from dataclasses import replace
from pathlib import Path

import pytest

from a_share_research.contracts import ContractError
from a_share_research.experiments.v0 import (
    CellState,
    D0Binding,
    D0GateState,
    ModelRuntimeBinding,
    V0StatusTable,
    bind_runtime_evidence,
    build_frozen_v0_registry,
    execution_units,
    load_v0_blueprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT = PROJECT_ROOT / "configs/experiments/v0/registry-v1.json"
HASH = "a" * 64
MODELS = (
    "ridge",
    "lightgbm",
    "itransformer",
    "fact",
    "timepro",
    "timexer",
    "s4m",
)
BLOCKED = {"s4m"}


def _frozen():
    return build_frozen_v0_registry(
        project_root=PROJECT_ROOT,
        blueprint=load_v0_blueprint(BLUEPRINT),
    )


def _d0(*, star50: D0GateState = D0GateState.PASS) -> D0Binding:
    return D0Binding(
        manifest_path="/data/yilangliu/a_share_research/data/manifests/d0-v1.json",
        manifest_hash="1" * 64,
        asset_registry_hash="2" * 64,
        execution_calendar_manifest_hash="3" * 64,
        feature_schema_hash="4" * 64,
        market_state_hash="5" * 64,
        universe_gates=(
            ("CSI300", D0GateState.PASS),
            ("STAR50", star50),
            ("TECH32", D0GateState.EXPLORATORY_ONLY),
            ("TECH100", D0GateState.EXPLORATORY_ONLY),
        ),
    )


def _model_bindings() -> dict[str, ModelRuntimeBinding]:
    return {
        model: (
            ModelRuntimeBinding(model=model, license_receipt_hash=HASH)
            if model in BLOCKED
            else ModelRuntimeBinding(
                model=model,
                environment_receipt_hash=HASH,
                integrity_receipt_hash="b" * 64,
            )
        )
        for model in MODELS
    }


def test_frozen_registry_is_exactly_28_core_only_cells() -> None:
    registry = _frozen()
    assert len(registry.cells) == 28
    assert len(registry.by_id) == 28
    assert {(cell.model, cell.universe) for cell in registry.cells} == {
        (model, universe)
        for model in MODELS
        for universe in ("CSI300", "STAR50", "TECH32", "TECH100")
    }
    assert {cell.information_set for cell in registry.cells} == {"A0"}
    assert {cell.information_groups for cell in registry.cells} == {("CORE",)}
    assert len(registry.stable_hash()) == 64


def test_split_frequency_label_entry_and_legacy_selection_are_frozen() -> None:
    registry = _frozen()
    assert (registry.frequency, registry.horizon, registry.entry) == ("weekly", 5, "T+1_OPEN")
    assert registry.train_end.isoformat() == "2024-12-31"
    assert registry.validation_start.isoformat() == "2025-01-01"
    assert registry.validation_end.isoformat() == "2025-12-31"
    assert registry.legacy_viewed_start.isoformat() == "2026-01-01"
    assert registry.legacy_policy == "REPORT_ONLY_NO_SELECTION"
    with pytest.raises(ContractError, match="split or legacy-viewed"):
        replace(registry, legacy_policy="SELECT").validate()


def test_license_gate_is_explicit_and_cannot_create_execution_units() -> None:
    frozen = _frozen()
    assert sum(cell.initial_state is CellState.RUNNABLE_PENDING_D0 for cell in frozen.cells) == 24
    assert sum(cell.initial_state is CellState.BLOCKED_LICENSE for cell in frozen.cells) == 4
    bound = bind_runtime_evidence(frozen, d0=_d0(), model_bindings=_model_bindings())
    assert all(
        cell.state is CellState.BLOCKED_LICENSE
        for cell in bound.cells
        if cell.frozen.model in BLOCKED
    )
    assert not ({unit.model for unit in execution_units(bound)} & BLOCKED)


def test_exact_receipt_and_d0_hashes_are_required_before_runnable() -> None:
    frozen = _frozen()
    bindings = _model_bindings()
    bindings["fact"] = ModelRuntimeBinding(
        model="fact",
        environment_receipt_hash="not-a-hash",
        integrity_receipt_hash=HASH,
    )
    with pytest.raises(ContractError, match="required receipt hashes"):
        bind_runtime_evidence(frozen, d0=_d0(), model_bindings=bindings)
    with pytest.raises(ContractError, match="D0 manifest_hash"):
        bind_runtime_evidence(
            frozen,
            d0=replace(_d0(), manifest_hash="bad"),
            model_bindings=_model_bindings(),
        )


def test_d0_gate_is_inherited_without_dropping_star50_cells() -> None:
    frozen = _frozen()
    bound = bind_runtime_evidence(
        frozen,
        d0=_d0(star50=D0GateState.BLOCKED),
        model_bindings=_model_bindings(),
    )
    star50 = [cell for cell in bound.cells if cell.frozen.universe == "STAR50"]
    assert len(star50) == 7
    assert {
        cell.state for cell in star50
    } == {CellState.BLOCKED_D0, CellState.BLOCKED_LICENSE}
    assert len(execution_units(bound)) == 42


def test_cpu_gpu0_gpu1_plan_has_three_deep_seeds_and_isolated_paths() -> None:
    bound = bind_runtime_evidence(_frozen(), d0=_d0(), model_bindings=_model_bindings())
    units = execution_units(bound)
    assert len(units) == 56
    assert {unit.lane for unit in units} == {"CPU", "GPU0", "GPU1"}
    assert len({unit.run_id for unit in units}) == len(units)
    assert len({unit.output_dir for unit in units}) == len(units)
    assert len({unit.checkpoint_dir for unit in units}) == len(units)
    deep = [unit for unit in units if unit.model in {"itransformer", "fact", "timexer"}]
    assert {unit.seed for unit in deep} == {20260719, 20260720, 20260721}


def test_all_cells_reference_config_adapter_registry_and_runtime_evidence() -> None:
    frozen = _frozen()
    assert all(len(cell.model_config_hash) == 64 for cell in frozen.cells)
    assert all(len(cell.adapter_hash) == 64 for cell in frozen.cells)
    assert len({cell.upstream_registry_hash for cell in frozen.cells}) == 1
    bound = bind_runtime_evidence(frozen, d0=_d0(), model_bindings=_model_bindings())
    assert {cell.market_state_hash for cell in bound.cells} == {"5" * 64}
    assert {cell.d0_manifest_hash for cell in bound.cells} == {"1" * 64}


def test_status_table_cannot_silently_lose_a_cell_or_seed() -> None:
    bound = bind_runtime_evidence(_frozen(), d0=_d0(), model_bindings=_model_bindings())
    table = V0StatusTable.from_bound_registry(bound)
    assert len(table.rows) == 68
    with pytest.raises(ContractError, match="lost or duplicated"):
        replace(table, rows=table.rows[:-1]).validate(bound)
    with pytest.raises(ContractError, match="non-terminal"):
        table.assert_terminal(bound)


def test_failures_are_typed_and_unknown_attempts_are_rejected() -> None:
    bound = bind_runtime_evidence(_frozen(), d0=_d0(), model_bindings=_model_bindings())
    table = V0StatusTable.from_bound_registry(bound)
    runnable = next(row for row in table.rows if row.state is CellState.RUNNABLE)
    failed = table.replace_state(
        bound,
        attempt_id=runnable.attempt_id,
        state=CellState.TRAIN_FAIL,
        reason_code="NONFINITE_LOSS",
    )
    recorded = next(row for row in failed.rows if row.attempt_id == runnable.attempt_id)
    assert recorded.state is CellState.TRAIN_FAIL
    with pytest.raises(ContractError, match="unregistered"):
        table.replace_state(
            bound,
            attempt_id="missing",
            state=CellState.EVAL_FAIL,
            reason_code="MISSING_PREDICTIONS",
        )


def test_blocked_model_binding_cannot_smuggle_environment_evidence() -> None:
    with pytest.raises(ContractError, match="crosses the license gate"):
        ModelRuntimeBinding(
            model="s4m",
            environment_receipt_hash=HASH,
            integrity_receipt_hash=HASH,
            license_receipt_hash=HASH,
        ).validate()
