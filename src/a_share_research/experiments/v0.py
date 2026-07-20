"""V0/A0 pre-registration, evidence binding and isolated execution planning.

This module performs no training and imports no model framework.  It freezes the
logical 4-universe by 7-model matrix first, then requires server-produced D0 and
upstream receipt hashes before a runnable execution unit can exist.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from a_share_research.contracts import ContractError, canonical_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_MODELS = (
    "ridge",
    "lightgbm",
    "itransformer",
    "fact",
    "timepro",
    "timexer",
    "s4m",
)
_EXPECTED_UNIVERSES = ("CSI300", "STAR50", "TECH32", "TECH100")
_BLOCKED_MODELS = frozenset({"s4m"})
_DEEP_MODELS = frozenset({"itransformer", "fact", "timepro", "timexer", "s4m"})
_TERMINAL_STATES = frozenset(
    {
        "BLOCKED_LICENSE",
        "BLOCKED_D0",
        "INVALID_DATA",
        "INVALID_PROTOCOL",
        "UPSTREAM_FAIL",
        "ENV_FAIL",
        "ADAPTER_FAIL",
        "TRAIN_FAIL",
        "EVAL_FAIL",
        "PASS",
        "PASS_WITH_WARNING",
        "EXPLORATORY_ONLY",
        "VALID_NEGATIVE",
    }
)


class CellState(str, Enum):
    """Typed lifecycle; a failed cell remains visible in the frozen matrix."""

    RUNNABLE_PENDING_D0 = "RUNNABLE_PENDING_D0"
    RUNNABLE = "RUNNABLE"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    BLOCKED_D0 = "BLOCKED_D0"
    INVALID_DATA = "INVALID_DATA"
    INVALID_PROTOCOL = "INVALID_PROTOCOL"
    UPSTREAM_FAIL = "UPSTREAM_FAIL"
    ENV_FAIL = "ENV_FAIL"
    ADAPTER_FAIL = "ADAPTER_FAIL"
    TRAIN_FAIL = "TRAIN_FAIL"
    EVAL_FAIL = "EVAL_FAIL"
    PASS = "PASS"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
    VALID_NEGATIVE = "VALID_NEGATIVE"

    @property
    def is_terminal(self) -> bool:
        return self.value in _TERMINAL_STATES


class D0GateState(str, Enum):
    PASS = "PASS"
    EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
    BLOCKED = "BLOCKED"
    INVALID_DATA = "INVALID_DATA"


@dataclass(frozen=True)
class FrozenV0Cell:
    cell_id: str
    model: str
    universe: str
    information_set: str
    information_groups: tuple[str, ...]
    lane: str
    seeds: tuple[int, ...]
    initial_state: CellState
    model_config_hash: str
    adapter_hash: str
    upstream_registry_hash: str
    upstream_commit: str
    d0_manifest_path: str
    environment_receipt_path: str | None
    integrity_receipt_path: str | None
    license_receipt_path: str | None
    output_root: str
    checkpoint_root: str

    def validate(self) -> None:
        if self.cell_id != f"v0-a0-{self.universe.lower()}-{self.model}":
            raise ContractError("V0 cell_id is not canonical")
        if self.model not in _EXPECTED_MODELS or self.universe not in _EXPECTED_UNIVERSES:
            raise ContractError("V0 cell model/universe is outside the frozen matrix")
        if self.information_set != "A0" or self.information_groups != ("CORE",):
            raise ContractError("V0 is Core-only A0")
        for name in ("model_config_hash", "adapter_hash", "upstream_registry_hash"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ContractError(f"{name} must be SHA-256")
        if self.model in {"ridge", "lightgbm"}:
            if re.fullmatch(r"internal:[a-z0-9._-]+", self.upstream_commit) is None:
                raise ContractError("baseline requires a versioned internal provenance pin")
        elif _COMMIT.fullmatch(self.upstream_commit) is None:
            raise ContractError("paper model upstream_commit must be a full commit")
        expected_seeds = (
            (20260719, 20260720, 20260721)
            if self.model in _DEEP_MODELS
            else (20260719,)
        )
        if self.seeds != expected_seeds:
            raise ContractError("V0 seeds differ from the frozen model class policy")
        expected_lane = {
            "ridge": "CPU",
            "lightgbm": "CPU",
            "itransformer": "GPU0",
            "fact": "GPU1",
            "timepro": "GPU1",
            "timexer": "GPU0",
            "s4m": "NONE",
        }[self.model]
        if self.lane != expected_lane:
            raise ContractError("model execution lane differs from the frozen policy")
        blocked = self.model in _BLOCKED_MODELS
        if blocked != (self.initial_state is CellState.BLOCKED_LICENSE):
            raise ContractError("license state and frozen model policy disagree")
        if blocked:
            if self.license_receipt_path is None:
                raise ContractError("blocked model requires a license-gate receipt reference")
            if self.environment_receipt_path is not None or self.integrity_receipt_path is not None:
                raise ContractError(
                    "blocked model cannot reference executable environment evidence"
                )
        else:
            if self.initial_state is not CellState.RUNNABLE_PENDING_D0:
                raise ContractError("licensed model must wait for D0")
            if self.environment_receipt_path is None or self.integrity_receipt_path is None:
                raise ContractError("licensed model requires environment and integrity receipts")
        receipt_paths = tuple(
            path
            for path in (
                self.environment_receipt_path,
                self.integrity_receipt_path,
                self.license_receipt_path,
            )
            if path is not None
        )
        for path in (self.d0_manifest_path, self.output_root, self.checkpoint_root, *receipt_paths):
            if not PurePosixPath(path).is_absolute():
                raise ContractError("server evidence/output paths must be absolute")


@dataclass(frozen=True)
class FrozenV0Registry:
    schema_version: str
    protocol_version: str
    frequency: str
    horizon: int
    entry: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    legacy_viewed_start: date
    legacy_viewed_end: date
    legacy_policy: str
    cells: tuple[FrozenV0Cell, ...]

    def validate(self) -> None:
        if self.schema_version != "v0_registry_v1" or self.protocol_version != "v1":
            raise ContractError("unsupported V0 registry version")
        if (self.frequency, self.horizon, self.entry) != ("weekly", 5, "T+1_OPEN"):
            raise ContractError("V0 primary protocol must be weekly future5d with T+1 open entry")
        expected_dates = (
            date(2019, 1, 1),
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2025, 12, 31),
            date(2026, 1, 1),
            date(2026, 7, 17),
        )
        actual_dates = (
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
            self.legacy_viewed_start,
            self.legacy_viewed_end,
        )
        if actual_dates != expected_dates or self.legacy_policy != "REPORT_ONLY_NO_SELECTION":
            raise ContractError("V0 split or legacy-viewed policy drifted")
        expected_pairs = {
            (model, universe)
            for model in _EXPECTED_MODELS
            for universe in _EXPECTED_UNIVERSES
        }
        actual_pairs = {(cell.model, cell.universe) for cell in self.cells}
        if len(self.cells) != 28 or actual_pairs != expected_pairs:
            raise ContractError("V0 registry must contain exactly the 28 model/universe cells")
        if len({cell.cell_id for cell in self.cells}) != 28:
            raise ContractError("V0 cell IDs must be unique")
        for cell in self.cells:
            cell.validate()

    def stable_hash(self) -> str:
        self.validate()
        return canonical_hash(_registry_payload(self))

    @property
    def by_id(self) -> Mapping[str, FrozenV0Cell]:
        self.validate()
        return MappingProxyType({cell.cell_id: cell for cell in self.cells})


@dataclass(frozen=True)
class D0Binding:
    manifest_path: str
    manifest_hash: str
    asset_registry_hash: str
    execution_calendar_manifest_hash: str
    feature_schema_hash: str
    market_state_hash: str
    universe_gates: tuple[tuple[str, D0GateState], ...]

    def validate(self) -> None:
        if not PurePosixPath(self.manifest_path).is_absolute():
            raise ContractError("D0 manifest path must be absolute")
        for name in (
            "manifest_hash",
            "asset_registry_hash",
            "execution_calendar_manifest_hash",
            "feature_schema_hash",
            "market_state_hash",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ContractError(f"D0 {name} must be SHA-256")
        gates = dict(self.universe_gates)
        if (
            set(gates) != set(_EXPECTED_UNIVERSES)
            or len(gates) != 4
            or len(self.universe_gates) != 4
        ):
            raise ContractError("D0 binding requires exactly four universe gates")
        if not all(isinstance(state, D0GateState) for state in gates.values()):
            raise ContractError("D0 gates must be typed")


@dataclass(frozen=True)
class ModelRuntimeBinding:
    model: str
    environment_receipt_hash: str | None = None
    integrity_receipt_hash: str | None = None
    license_receipt_hash: str | None = None

    def validate(self) -> None:
        if self.model not in _EXPECTED_MODELS:
            raise ContractError("runtime binding names an unknown model")
        blocked = self.model in _BLOCKED_MODELS
        required = (
            (self.license_receipt_hash,)
            if blocked
            else (self.environment_receipt_hash, self.integrity_receipt_hash)
        )
        if any(value is None or _SHA256.fullmatch(value) is None for value in required):
            raise ContractError("runtime model binding lacks the required receipt hashes")
        forbidden = (
            (self.environment_receipt_hash, self.integrity_receipt_hash)
            if blocked
            else (self.license_receipt_hash,)
        )
        if any(value is not None for value in forbidden):
            raise ContractError("runtime model binding crosses the license gate")


@dataclass(frozen=True)
class BoundV0Cell:
    frozen: FrozenV0Cell
    state: CellState
    d0_manifest_hash: str
    asset_registry_hash: str
    execution_calendar_manifest_hash: str
    feature_schema_hash: str
    market_state_hash: str
    environment_receipt_hash: str | None
    integrity_receipt_hash: str | None
    license_receipt_hash: str | None

    def validate(self) -> None:
        self.frozen.validate()
        for name in (
            "d0_manifest_hash",
            "asset_registry_hash",
            "execution_calendar_manifest_hash",
            "feature_schema_hash",
            "market_state_hash",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ContractError(f"bound {name} must be SHA-256")
        ModelRuntimeBinding(
            model=self.frozen.model,
            environment_receipt_hash=self.environment_receipt_hash,
            integrity_receipt_hash=self.integrity_receipt_hash,
            license_receipt_hash=self.license_receipt_hash,
        ).validate()
        if self.frozen.model in _BLOCKED_MODELS:
            if self.state is not CellState.BLOCKED_LICENSE:
                raise ContractError("blocked-license cell cannot become executable")
        elif self.state not in {
            CellState.RUNNABLE,
            CellState.BLOCKED_D0,
            CellState.INVALID_DATA,
        }:
            raise ContractError("D0 binding produced an unsupported preflight state")


@dataclass(frozen=True)
class BoundV0Registry:
    frozen_registry_hash: str
    cells: tuple[BoundV0Cell, ...]

    def validate(self) -> None:
        if _SHA256.fullmatch(self.frozen_registry_hash) is None:
            raise ContractError("frozen registry hash must be SHA-256")
        if len(self.cells) != 28 or len({cell.frozen.cell_id for cell in self.cells}) != 28:
            raise ContractError("bound registry cannot lose or duplicate a frozen cell")
        for cell in self.cells:
            cell.validate()


@dataclass(frozen=True)
class ExecutionUnit:
    attempt_id: str
    run_id: str
    cell_id: str
    model: str
    universe: str
    seed: int
    lane: str
    output_dir: str
    checkpoint_dir: str


@dataclass(frozen=True)
class AttemptStatus:
    attempt_id: str
    cell_id: str
    seed: int
    state: CellState
    reason_code: str | None = None

    def validate(self) -> None:
        if self.state.is_terminal and not self.reason_code:
            raise ContractError("terminal attempt state requires a reason_code")
        if not self.state.is_terminal and self.reason_code is not None:
            raise ContractError("non-terminal attempt state cannot carry a reason_code")


@dataclass(frozen=True)
class V0StatusTable:
    """Immutable complete attempt table; updates replace rows and never drop cells."""

    registry_hash: str
    rows: tuple[AttemptStatus, ...]

    @classmethod
    def from_bound_registry(cls, registry: BoundV0Registry) -> V0StatusTable:
        registry.validate()
        rows = []
        for cell in registry.cells:
            for seed in cell.frozen.seeds:
                reason = "LICENSE_GATE" if cell.state is CellState.BLOCKED_LICENSE else None
                if cell.state is CellState.BLOCKED_D0:
                    reason = "D0_GATE_BLOCKED"
                elif cell.state is CellState.INVALID_DATA:
                    reason = "D0_GATE_INVALID"
                attempt_id = _attempt_id(cell.frozen.cell_id, seed)
                rows.append(
                    AttemptStatus(attempt_id, cell.frozen.cell_id, seed, cell.state, reason)
                )
        table = cls(
            registry.frozen_registry_hash,
            tuple(sorted(rows, key=lambda row: row.attempt_id)),
        )
        table.validate(registry)
        return table

    def validate(self, registry: BoundV0Registry) -> None:
        registry.validate()
        if self.registry_hash != registry.frozen_registry_hash:
            raise ContractError("status table belongs to a different frozen registry")
        expected = {
            _attempt_id(cell.frozen.cell_id, seed): (cell.frozen.cell_id, seed)
            for cell in registry.cells
            for seed in cell.frozen.seeds
        }
        actual = {row.attempt_id for row in self.rows}
        if actual != set(expected) or len(actual) != len(self.rows):
            raise ContractError("status table silently lost or duplicated a V0 attempt")
        for row in self.rows:
            row.validate()
            if (row.cell_id, row.seed) != expected[row.attempt_id]:
                raise ContractError("status row identity differs from its registered attempt")

    def replace_state(
        self,
        registry: BoundV0Registry,
        *,
        attempt_id: str,
        state: CellState,
        reason_code: str | None = None,
    ) -> V0StatusTable:
        self.validate(registry)
        if attempt_id not in {row.attempt_id for row in self.rows}:
            raise ContractError("cannot record an unregistered V0 attempt")
        rows = tuple(
            replace(row, state=state, reason_code=reason_code)
            if row.attempt_id == attempt_id
            else row
            for row in self.rows
        )
        updated = V0StatusTable(self.registry_hash, rows)
        updated.validate(registry)
        return updated

    def assert_terminal(self, registry: BoundV0Registry) -> None:
        self.validate(registry)
        pending = sorted(row.attempt_id for row in self.rows if not row.state.is_terminal)
        if pending:
            raise ContractError(f"V0 has non-terminal attempts: {pending}")


def load_v0_blueprint(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("V0 blueprint must be a JSON object")
    return MappingProxyType(payload)


def build_frozen_v0_registry(
    *,
    project_root: Path,
    blueprint: Mapping[str, Any],
) -> FrozenV0Registry:
    """Expand the frozen matrix and hash every declared source/config input."""

    models = blueprint.get("models")
    universes = blueprint.get("universes")
    protocol = blueprint.get("protocol")
    server = blueprint.get("server")
    if not isinstance(models, list) or not isinstance(universes, list):
        raise ContractError("V0 blueprint models/universes must be lists")
    if not isinstance(protocol, dict) or not isinstance(server, dict):
        raise ContractError("V0 blueprint protocol/server must be objects")
    if (
        protocol.get("information_set") != "A0"
        or protocol.get("information_groups") != ["CORE"]
        or protocol.get("selection_partition") != "VALIDATION_ONLY"
    ):
        raise ContractError("V0 blueprint must be Core-only and select on 2025 validation")
    if tuple(item.get("name") for item in models if isinstance(item, dict)) != _EXPECTED_MODELS:
        raise ContractError("V0 blueprint model order/set drifted")
    if tuple(universes) != _EXPECTED_UNIVERSES:
        raise ContractError("V0 blueprint universe order/set drifted")
    upstream_registry_path = _project_file(project_root, str(blueprint["upstream_registry"]))
    upstream_registry_hash = _sha256(upstream_registry_path)
    cells: list[FrozenV0Cell] = []
    for universe in universes:
        for raw in models:
            if not isinstance(raw, dict):
                raise ContractError("V0 model entry must be an object")
            model = str(raw["name"])
            config_hash = _hash_declared_files(project_root, raw.get("model_config_files"))
            adapter_hash = _hash_declared_files(project_root, raw.get("adapter_files"))
            blocked = model in _BLOCKED_MODELS
            cell = FrozenV0Cell(
                cell_id=f"v0-a0-{str(universe).lower()}-{model}",
                model=model,
                universe=str(universe),
                information_set="A0",
                information_groups=("CORE",),
                lane=str(raw["lane"]),
                seeds=(
                    (20260719, 20260720, 20260721)
                    if model in _DEEP_MODELS
                    else (20260719,)
                ),
                initial_state=(
                    CellState.BLOCKED_LICENSE if blocked else CellState.RUNNABLE_PENDING_D0
                ),
                model_config_hash=config_hash,
                adapter_hash=adapter_hash,
                upstream_registry_hash=upstream_registry_hash,
                upstream_commit=str(raw["upstream_commit"]),
                d0_manifest_path=str(server["d0_manifest_path"]),
                environment_receipt_path=(
                    None if blocked else str(raw["environment_receipt_path"])
                ),
                integrity_receipt_path=(
                    None if blocked else str(raw["integrity_receipt_path"])
                ),
                license_receipt_path=(
                    str(raw["license_receipt_path"]) if blocked else None
                ),
                output_root=str(server["output_root"]),
                checkpoint_root=str(server["checkpoint_root"]),
            )
            cell.validate()
            cells.append(cell)
    registry = FrozenV0Registry(
        schema_version=str(blueprint["schema_version"]),
        protocol_version=str(protocol["protocol_version"]),
        frequency=str(protocol["frequency"]),
        horizon=int(protocol["horizon"]),
        entry=str(protocol["entry"]),
        train_start=date.fromisoformat(str(protocol["train"][0])),
        train_end=date.fromisoformat(str(protocol["train"][1])),
        validation_start=date.fromisoformat(str(protocol["validation"][0])),
        validation_end=date.fromisoformat(str(protocol["validation"][1])),
        legacy_viewed_start=date.fromisoformat(str(protocol["legacy_viewed"][0])),
        legacy_viewed_end=date.fromisoformat(str(protocol["legacy_viewed"][1])),
        legacy_policy=str(protocol["legacy_policy"]),
        cells=tuple(cells),
    )
    registry.validate()
    return registry


def bind_runtime_evidence(
    registry: FrozenV0Registry,
    *,
    d0: D0Binding,
    model_bindings: Mapping[str, ModelRuntimeBinding],
) -> BoundV0Registry:
    """Bind exact server receipts without ever unblocking a no-license model."""

    registry.validate()
    d0.validate()
    if d0.manifest_path != registry.cells[0].d0_manifest_path:
        raise ContractError("D0 binding path differs from the frozen manifest reference")
    if set(model_bindings) != set(_EXPECTED_MODELS):
        raise ContractError("runtime evidence must account for all seven models")
    for name, binding in model_bindings.items():
        if name != binding.model:
            raise ContractError("runtime evidence mapping key/model disagree")
        binding.validate()
    gates = dict(d0.universe_gates)
    cells = []
    for frozen in registry.cells:
        binding = model_bindings[frozen.model]
        gate = gates[frozen.universe]
        if frozen.initial_state is CellState.BLOCKED_LICENSE:
            state = CellState.BLOCKED_LICENSE
        elif gate in {D0GateState.PASS, D0GateState.EXPLORATORY_ONLY}:
            state = CellState.RUNNABLE
        elif gate is D0GateState.BLOCKED:
            state = CellState.BLOCKED_D0
        else:
            state = CellState.INVALID_DATA
        cells.append(
            BoundV0Cell(
                frozen=frozen,
                state=state,
                d0_manifest_hash=d0.manifest_hash,
                asset_registry_hash=d0.asset_registry_hash,
                execution_calendar_manifest_hash=d0.execution_calendar_manifest_hash,
                feature_schema_hash=d0.feature_schema_hash,
                market_state_hash=d0.market_state_hash,
                environment_receipt_hash=binding.environment_receipt_hash,
                integrity_receipt_hash=binding.integrity_receipt_hash,
                license_receipt_hash=binding.license_receipt_hash,
            )
        )
    bound = BoundV0Registry(registry.stable_hash(), tuple(cells))
    bound.validate()
    return bound


def execution_units(registry: BoundV0Registry) -> tuple[ExecutionUnit, ...]:
    """Return only evidence-bound runnable attempts with isolated paths."""

    registry.validate()
    units: list[ExecutionUnit] = []
    for cell in registry.cells:
        if cell.state is not CellState.RUNNABLE:
            continue
        for seed in cell.frozen.seeds:
            attempt_id = _attempt_id(cell.frozen.cell_id, seed)
            output_dir = str(
                PurePosixPath(cell.frozen.output_root) / cell.frozen.cell_id / str(seed)
            )
            checkpoint_dir = str(
                PurePosixPath(cell.frozen.checkpoint_root) / cell.frozen.cell_id / str(seed)
            )
            units.append(
                ExecutionUnit(
                    attempt_id=attempt_id,
                    run_id=attempt_id,
                    cell_id=cell.frozen.cell_id,
                    model=cell.frozen.model,
                    universe=cell.frozen.universe,
                    seed=seed,
                    lane=cell.frozen.lane,
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                )
            )
    if len({unit.run_id for unit in units}) != len(units):
        raise ContractError("V0 run IDs are not isolated")
    if len({unit.output_dir for unit in units}) != len(units):
        raise ContractError("V0 output directories are not isolated")
    if len({unit.checkpoint_dir for unit in units}) != len(units):
        raise ContractError("V0 checkpoint directories are not isolated")
    return tuple(sorted(units, key=lambda unit: unit.attempt_id))


def _project_file(project_root: Path, relative: str) -> Path:
    path = project_root / relative
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ContractError("declared V0 source path escapes the project root") from error
    if not path.is_file():
        raise ContractError(f"declared V0 source file is missing: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_declared_files(project_root: Path, raw_paths: Any) -> str:
    if (
        not isinstance(raw_paths, list)
        or not raw_paths
        or not all(isinstance(path, str) for path in raw_paths)
    ):
        raise ContractError("V0 model/adapter file declaration must be a non-empty string list")
    hashes = tuple((path, _sha256(_project_file(project_root, path))) for path in raw_paths)
    return canonical_hash(hashes)


def _attempt_id(cell_id: str, seed: int) -> str:
    return f"{cell_id}-seed-{seed}"


def _registry_payload(registry: FrozenV0Registry) -> dict[str, Any]:
    return {
        "schema_version": registry.schema_version,
        "protocol_version": registry.protocol_version,
        "frequency": registry.frequency,
        "horizon": registry.horizon,
        "entry": registry.entry,
        "train": [registry.train_start.isoformat(), registry.train_end.isoformat()],
        "validation": [registry.validation_start.isoformat(), registry.validation_end.isoformat()],
        "legacy_viewed": [
            registry.legacy_viewed_start.isoformat(),
            registry.legacy_viewed_end.isoformat(),
        ],
        "legacy_policy": registry.legacy_policy,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "model": cell.model,
                "universe": cell.universe,
                "information_set": cell.information_set,
                "information_groups": list(cell.information_groups),
                "lane": cell.lane,
                "seeds": list(cell.seeds),
                "initial_state": cell.initial_state.value,
                "model_config_hash": cell.model_config_hash,
                "adapter_hash": cell.adapter_hash,
                "upstream_registry_hash": cell.upstream_registry_hash,
                "upstream_commit": cell.upstream_commit,
                "d0_manifest_path": cell.d0_manifest_path,
                "environment_receipt_path": cell.environment_receipt_path,
                "integrity_receipt_path": cell.integrity_receipt_path,
                "license_receipt_path": cell.license_receipt_path,
                "output_root": cell.output_root,
                "checkpoint_root": cell.checkpoint_root,
            }
            for cell in registry.cells
        ],
    }
