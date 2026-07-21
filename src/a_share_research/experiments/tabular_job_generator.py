"""Fail-closed generation of sealed Ridge/LightGBM V0/V1 server jobs.

This module performs no fitting.  It resolves the final D0 evidence, derives
the append-only 2025 asset identity through the canonical loader, and emits
only explicit jobs.  Formal CSI300/STAR50 jobs additionally require an
existing :class:`FormalFeatureManifest`; absence is a hard error rather than
an invitation to synthesize an eligibility hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar, Final

from a_share_research.contracts import (
    AssetRegistry,
    CanonicalModel,
    ContractError,
    FormalFeatureManifest,
    canonical_hash,
)
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.data.manifest import D0Manifest
from a_share_research.experiments.tabular_runner import (
    EvidenceFile,
    TabularJobSpec,
    TabularQueueManifest,
)
from a_share_research.models.tabular import InformationSet, default_feature_layout
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

SEED: Final = 20260719
VALIDATION_START: Final = date(2025, 1, 1)
VALIDATION_END: Final = date(2025, 12, 31)
APPROVED_SERVER_ROOT: Final = Path("/data/yilangliu/a_share_research")
MODELS: Final = ("ridge", "lightgbm")
UNIVERSES: Final = tuple(UniverseClass)
FORMAL_UNIVERSES: Final = frozenset({UniverseClass.CSI300, UniverseClass.STAR50})
UPSTREAM: Final = {
    "ridge": "internal:scikit-learn-1.9.0",
    "lightgbm": "internal:lightgbm-4.7.0",
}

LoaderFactory = Callable[[Path, str], CanonicalDatasetLoader]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _within(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    root = root.expanduser().resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ContractError(f"path leaves approved server root: {resolved}")
    return resolved


def _future_within(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = root.expanduser().resolve(strict=True)
    if resolved == root or root not in resolved.parents:
        raise ContractError(f"output path leaves approved server root: {resolved}")
    return resolved


def _evidence(path: Path, approved_root: Path) -> EvidenceFile:
    resolved = _within(path, approved_root)
    if not resolved.is_file():
        raise ContractError(f"evidence is not a regular file: {resolved}")
    return EvidenceFile(path=resolved.as_posix(), sha256=_sha256(resolved))


def _verify_canonical_tables(
    canonical_root: Path,
    d0: D0Manifest,
    universe: UniverseClass,
) -> None:
    """Prove that registry discovery reads the tables sealed by final D0."""
    relatives = (
        f"{universe.value.lower()}/membership.jsonl",
        f"{universe.value.lower()}/features.jsonl",
        f"{universe.value.lower()}/labels.jsonl",
        f"{universe.value.lower()}/masks.jsonl",
        "shared_market_state.jsonl",
    )
    for relative in relatives:
        expected = d0.canonical_table_hashes.get(relative)
        path = canonical_root / relative
        if expected is None or not path.is_file() or _sha256(path) != expected:
            raise ContractError(
                f"registry discovery rejects unsealed canonical table: {relative}"
            )


def _information_features(information_set: InformationSet) -> tuple[str, ...]:
    layout = default_feature_layout()
    names = list(layout.core)
    if information_set.enables_f:
        names.extend(layout.fundamental)
        names.extend(layout.fundamental_missing)
    if information_set.enables_s:
        names.extend(layout.market_state)
    return tuple(names)


def _formal_receipt(
    *,
    path: Path | None,
    d0: D0Manifest,
    universe: UniverseClass,
    information_set: InformationSet,
    approved_root: Path,
) -> tuple[FormalFeatureManifest, EvidenceFile]:
    if path is None:
        raise ContractError(
            "formal D0 feature receipt is absent; generation fails closed "
            f"for {universe.value}/{information_set.value}"
        )
    evidence = _evidence(path, approved_root)
    receipt = FormalFeatureManifest.from_dict(_json_object(Path(evidence.path)))
    if receipt.d0_manifest_hash != d0.content_hash:
        raise ContractError("formal feature receipt is not anchored to the final D0 content hash")
    expected = set(_information_features(information_set))
    actual = set(receipt.feature_eligibility)
    if actual != expected:
        raise ContractError(
            "formal feature receipt does not exactly cover the active information set; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    receipt.require_formal_eligible()
    return receipt, evidence


def validation_registry_hash(
    loader: CanonicalDatasetLoader,
) -> str:
    """Hash the exact append-only asset registry visible in 2025.

    ``complete_panel=True`` is essential: non-members and temporarily
    unobserved identities remain represented, so the identity cannot drift
    with a model's scoreable subset.
    """
    samples = tuple(
        loader.iter_tabular_samples(
            horizon=5,
            relative_target=True,
            start=VALIDATION_START,
            end=VALIDATION_END,
            complete_panel=True,
        )
    )
    if not samples:
        raise ContractError("canonical 2025 validation panel is empty")
    order: list[str] = []
    known: set[str] = set()
    offset = 0
    previous: date | None = None
    while offset < len(samples):
        signal_date = samples[offset].signal_date
        if signal_date.year != 2025:
            raise ContractError(
                "generator rejects any row outside the sealed 2025 selection window"
            )
        if previous is not None and signal_date <= previous:
            raise ContractError("canonical validation dates are not strictly increasing")
        codes: list[str] = []
        while offset < len(samples) and samples[offset].signal_date == signal_date:
            codes.append(samples[offset].ts_code)
            offset += 1
        if len(codes) != len(set(codes)):
            raise ContractError("canonical validation panel has duplicate date/asset identities")
        additions = [code for code in codes if code not in known]
        order.extend(additions)
        known.update(additions)
        if tuple(codes) != tuple(order):
            raise ContractError("canonical 2025 asset registry is not append-only")
        previous = signal_date
    return AssetRegistry(tuple(order)).stable_hash()


def _cell_payload(
    *,
    phase: str,
    model: str,
    universe: UniverseClass,
    information_set: InformationSet,
    d0: D0Manifest,
    evidence: Mapping[str, EvidenceFile],
    asset_registry_hash: str,
    formal_feature_manifest_hash: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "tabular_cell_config_v1",
        "phase": phase,
        "model": model,
        "universe": universe.value,
        "scope": (
            "EXPLORATORY_ONLY"
            if universe in {UniverseClass.TECH32, UniverseClass.TECH90}
            else "FORMAL"
        ),
        "information_set": information_set.value,
        "protocol": {
            "frequency": "WEEKLY",
            "horizon": 5,
            "label": "future_5d_open_to_open_excess_return",
            "entry": "T+1_OPEN",
            "train": ["2019-01-01", "2024-12-31"],
            "validation": ["2025-01-01", "2025-12-31"],
            "legacy_2026_selection_allowed": False,
        },
        "seed": SEED,
        "upstream_commit": UPSTREAM[model],
        "d0_content_hash": d0.content_hash,
        "asset_registry_hash": asset_registry_hash,
        "formal_feature_manifest_hash": formal_feature_manifest_hash,
        "evidence_files": {
            name: value.to_dict() for name, value in sorted(evidence.items())
        },
    }


@dataclass(frozen=True)
class GeneratedTabularJobs:
    jobs: tuple[TabularJobSpec, ...]
    queues: tuple[TabularQueueManifest, ...]
    formal_receipts: tuple[tuple[str, EvidenceFile, str], ...]
    blocked_cells: tuple[BlockedTabularCell, ...]


@dataclass(frozen=True)
class BlockedTabularCell(CanonicalModel):
    """One planned matrix cell that was explicitly prevented from running."""

    SCHEMA_NAME: ClassVar[str] = "blocked_tabular_cell"

    phase: str
    run_id: str
    model: str
    universe: UniverseClass
    information_set: InformationSet
    reason_code: str
    detail: str
    state: ResultState = ResultState.BLOCKED

    def validate(self) -> None:
        if self.phase not in {"V0", "V1"}:
            raise ContractError("blocked tabular phase must be V0 or V1")
        if self.model not in MODELS:
            raise ContractError("blocked tabular model is outside the CPU matrix")
        if not isinstance(self.universe, UniverseClass) or not isinstance(
            self.information_set, InformationSet
        ):
            raise ContractError("blocked cell requires typed universe and information set")
        expected = (
            f"{self.phase.lower()}-{self.information_set.value.lower()}-"
            f"{self.universe.value.lower()}-{self.model}-seed-{SEED}"
        )
        if self.run_id != expected:
            raise ContractError("blocked cell run_id is not canonical")
        if self.reason_code not in {"D0_GATE_BLOCKED", "FORMAL_RECEIPT_MISSING"}:
            raise ContractError("blocked cell reason code is not registered")
        if self.state is not ResultState.BLOCKED:
            raise ContractError("blocked tabular cell must remain in BLOCKED state")
        if not self.detail:
            raise ContractError("blocked cell requires an auditable detail")


def _run_id(
    phase: str,
    information_set: InformationSet,
    universe: UniverseClass,
    model: str,
) -> str:
    return (
        f"{phase.lower()}-{information_set.value.lower()}-"
        f"{universe.value.lower()}-{model}-seed-{SEED}"
    )


def _blocked_family(
    *,
    phase: str,
    universe: UniverseClass,
    information_sets: tuple[InformationSet, ...],
    reason_code: str,
    detail: str,
) -> tuple[BlockedTabularCell, ...]:
    return tuple(
        BlockedTabularCell(
            phase=phase,
            run_id=_run_id(phase, information_set, universe, model),
            model=model,
            universe=universe,
            information_set=information_set,
            reason_code=reason_code,
            detail=detail,
        )
        for information_set in information_sets
        for model in MODELS
    )


def build_tabular_jobs(
    *,
    phase: str,
    d0_manifest: Path,
    canonical_root: Path,
    environment_receipts: Mapping[str, Path],
    code_receipt: Path,
    model_configs: Mapping[str, Path],
    layout_config: Path,
    formal_feature_receipts: Mapping[tuple[UniverseClass, InformationSet], Path],
    output_root: Path,
    approved_root: Path = APPROVED_SERVER_ROOT,
    loader_factory: LoaderFactory = CanonicalDatasetLoader,
) -> GeneratedTabularJobs:
    """Build the exact 8-cell V0 or 24-cell V1 CPU matrix."""
    if phase not in {"V0", "V1"}:
        raise ContractError("generator phase must be V0 or V1")
    if set(environment_receipts) != set(MODELS) or set(model_configs) != set(MODELS):
        raise ContractError("generator requires exact Ridge and LightGBM evidence mappings")
    approved_root = approved_root.expanduser().resolve(strict=True)
    d0_evidence = _evidence(d0_manifest, approved_root)
    d0 = D0Manifest.from_dict(_json_object(Path(d0_evidence.path)))
    if d0.cutoff_date < VALIDATION_END:
        raise ContractError("final D0 does not cover the full 2025 validation window")
    canonical = _within(canonical_root, approved_root)
    if not canonical.is_dir():
        raise ContractError("canonical root is not a directory")
    code_evidence = _evidence(code_receipt, approved_root)
    layout_evidence = _evidence(layout_config, approved_root)
    environment = {
        model: _evidence(environment_receipts[model], approved_root) for model in MODELS
    }
    configs = {model: _evidence(model_configs[model], approved_root) for model in MODELS}
    run_root = _future_within(output_root / phase.lower(), approved_root)
    information_sets = (
        (InformationSet.A0,)
        if phase == "V0"
        else (InformationSet.A1, InformationSet.A2, InformationSet.A3)
    )
    gates = {gate.universe: gate for gate in d0.universe_gates}
    jobs: list[TabularJobSpec] = []
    blocked: list[BlockedTabularCell] = []
    formal_audit: list[tuple[str, EvidenceFile, str]] = []
    for universe in UNIVERSES:
        gate = gates[universe]
        allowed = (
            {ResultState.PASS, ResultState.PASS_WITH_WARNING}
            if universe in FORMAL_UNIVERSES
            else {ResultState.EXPLORATORY_ONLY}
        )
        if gate.status not in allowed:
            blocked.extend(
                _blocked_family(
                    phase=phase,
                    universe=universe,
                    information_sets=information_sets,
                    reason_code="D0_GATE_BLOCKED",
                    detail=(
                        f"final D0 gate for {universe.value} is {gate.status.value}; "
                        f"warnings={list(gate.warnings)}"
                    ),
                )
            )
            continue
        try:
            _verify_canonical_tables(canonical, d0, universe)
            loader = loader_factory(canonical, universe.value)
            asset_hash = validation_registry_hash(loader)
        except (ContractError, OSError, ValueError) as error:
            blocked.extend(
                _blocked_family(
                    phase=phase,
                    universe=universe,
                    information_sets=information_sets,
                    reason_code="D0_GATE_BLOCKED",
                    detail=f"canonical D0 evidence rejected for {universe.value}: {error}",
                )
            )
            continue
        for information_set in information_sets:
            formal_hash: str | None = None
            formal_evidence: EvidenceFile | None = None
            if universe in FORMAL_UNIVERSES:
                try:
                    receipt, formal_evidence = _formal_receipt(
                        path=formal_feature_receipts.get((universe, information_set)),
                        d0=d0,
                        universe=universe,
                        information_set=information_set,
                        approved_root=approved_root,
                    )
                except (ContractError, OSError, ValueError) as error:
                    blocked.extend(
                        _blocked_family(
                            phase=phase,
                            universe=universe,
                            information_sets=(information_set,),
                            reason_code="FORMAL_RECEIPT_MISSING",
                            detail=(
                                "formal feature eligibility evidence is absent or invalid "
                                f"for {universe.value}/{information_set.value}: {error}"
                            ),
                        )
                    )
                    continue
                formal_hash = receipt.require_formal_eligible()
                formal_audit.append(
                    (f"{universe.value}:{information_set.value}", formal_evidence, formal_hash)
                )
            for model in MODELS:
                run_id = _run_id(phase, information_set, universe, model)
                evidence = {
                    "d0_manifest": d0_evidence,
                    "environment_receipt": environment[model],
                    "code_receipt": code_evidence,
                    "model_config": configs[model],
                    "layout_config": layout_evidence,
                }
                if formal_evidence is not None:
                    evidence["formal_feature_receipt"] = formal_evidence
                cell_hash = canonical_hash(
                    _cell_payload(
                        phase=phase,
                        model=model,
                        universe=universe,
                        information_set=information_set,
                        d0=d0,
                        evidence=evidence,
                        asset_registry_hash=asset_hash,
                        formal_feature_manifest_hash=formal_hash,
                    )
                )
                jobs.append(
                    TabularJobSpec(
                        phase=phase,
                        run_id=run_id,
                        model=model,
                        universe=universe,
                        information_set=information_set,
                        seed=SEED,
                        canonical_root=canonical.as_posix(),
                        output_dir=(run_root / run_id).as_posix(),
                        d0_manifest=d0_evidence,
                        environment_receipt=environment[model],
                        code_receipt=code_evidence,
                        model_config=configs[model],
                        layout_config=layout_evidence,
                        asset_registry_hash=asset_hash,
                        cell_config_hash=cell_hash,
                        upstream_commit=UPSTREAM[model],
                        formal_feature_manifest_hash=formal_hash,
                        formal_feature_receipt=formal_evidence,
                    )
                )
    expected = 8 if phase == "V0" else 24
    if len(jobs) + len(blocked) != expected:
        raise ContractError(
            "tabular matrix accounting drifted; "
            f"runnable={len(jobs)}, blocked={len(blocked)}, expected={expected}"
        )
    identities = {job.run_id for job in jobs} | {cell.run_id for cell in blocked}
    if len(identities) != expected:
        raise ContractError("tabular matrix contains a duplicate or missing planned cell")
    queues = tuple(
        TabularQueueManifest(
            queue_id=f"{phase.lower()}-tabular-cpu-{index // 16 + 1:02d}",
            jobs=tuple(jobs[index : index + 16]),
            max_jobs=16,
        )
        for index in range(0, len(jobs), 16)
    )
    return GeneratedTabularJobs(
        tuple(jobs), queues, tuple(formal_audit), tuple(blocked)
    )


def write_tabular_jobs(
    generated: GeneratedTabularJobs,
    *,
    phase: str,
    job_root: Path,
    queue_root: Path,
    approved_root: Path = APPROVED_SERVER_ROOT,
) -> tuple[Path, ...]:
    """Atomically publish explicit job/queue files; never overwrite evidence."""
    if phase not in {"V0", "V1"}:
        raise ContractError("published tabular phase must be V0 or V1")
    expected = 8 if phase == "V0" else 24
    if (
        len(generated.jobs) + len(generated.blocked_cells) != expected
        or any(job.phase != phase for job in generated.jobs)
        or any(cell.phase != phase for cell in generated.blocked_cells)
    ):
        raise ContractError("published tabular matrix does not match its declared phase")
    job_dir = _future_within(job_root / phase.lower(), approved_root)
    queue_dir = _future_within(queue_root / phase.lower(), approved_root)
    if job_dir.exists() or queue_dir.exists():
        raise ContractError("generated job/queue directory already exists; refusing overwrite")
    job_tmp = job_dir.with_name(job_dir.name + ".tmp")
    queue_tmp = queue_dir.with_name(queue_dir.name + ".tmp")
    if job_tmp.exists() or queue_tmp.exists():
        raise ContractError("stale job-generation temporary directory exists")
    job_tmp.mkdir(parents=True)
    queue_tmp.mkdir(parents=True)
    written: list[Path] = []
    try:
        for job in generated.jobs:
            path = job_tmp / f"{job.run_id}.json"
            path.write_text(
                json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(job_dir / path.name)
        for queue in generated.queues:
            path = queue_tmp / f"{queue.queue_id}.json"
            path.write_text(
                json.dumps(queue.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(queue_dir / path.name)
        audit = {
            "schema_version": "tabular_job_generation_receipt_v1",
            "phase": phase,
            "planned_cell_count": len(generated.jobs) + len(generated.blocked_cells),
            "runnable_job_count": len(generated.jobs),
            "blocked_cell_count": len(generated.blocked_cells),
            "queue_count": len(generated.queues),
            "blocked_cells": [cell.to_dict() for cell in generated.blocked_cells],
            "formal_feature_receipts": {
                key: {
                    "file": evidence.to_dict(),
                    "formal_feature_manifest_hash": stable_hash,
                }
                for key, evidence, stable_hash in generated.formal_receipts
            },
            "legacy_2026_selection_allowed": False,
        }
        receipt_path = job_tmp / "generation_receipt.json"
        receipt_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(job_dir / receipt_path.name)
        job_tmp.replace(job_dir)
        queue_tmp.replace(queue_dir)
    except Exception:
        # Preserve temporaries as audit evidence; operators decide whether a
        # retry is safe rather than silently deleting a partial generation.
        raise
    return tuple(written)
