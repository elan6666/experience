"""Fail-closed iTransformer/FACT V0/V1 job and GPU-queue generation.

This module never imports a tensor runtime or author model.  It turns the
sealed D0, exact receipt files and frozen experiment protocol into explicit
``DeepJobSpec`` files.  Every planned cell is accounted for: cells that may
not run are emitted as typed ``BLOCKED`` audit records, never silently
dropped or replaced by an in-house model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar, Final

from a_share_research.adapters.common import (
    DeepRuntimePolicy,
    InformationGate,
    build_causal_asset_master,
)
from a_share_research.contracts import (
    CanonicalModel,
    ContractError,
    FormalFeatureManifest,
    canonical_hash,
)
from a_share_research.data.manifest import D0Manifest
from a_share_research.experiments.deep_runner import (
    DeepEvidenceFile,
    DeepHyperparameters,
    DeepJobSpec,
    _membership_intervals,
    _technology_master,
)
from a_share_research.models.tabular import default_feature_layout
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

APPROVED_SERVER_ROOT: Final = Path("/data/yilangliu/a_share_research")
VALIDATION_END: Final = date(2025, 12, 31)
MODELS: Final = ("itransformer", "fact", "timexer", "timepro")
UNIVERSES: Final = tuple(UniverseClass)
FORMAL_UNIVERSES: Final = frozenset({UniverseClass.CSI300, UniverseClass.STAR50})
COMMITS: Final = {
    "itransformer": "c2426e68ca13f74aaec08045c5c724d8ad328124",
    "fact": "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4",
    "timexer": "76011909357972bd55a27adba2e1be994d81b327",
    "timepro": "70a20e5a257b30eb026ee4316293cf4feeb92a1f",
}
PHYSICAL_GPUS: Final = {"itransformer": 0, "fact": 1, "timexer": 0, "timepro": 1}


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


def _evidence(path: Path, root: Path) -> DeepEvidenceFile:
    resolved = _within(path, root)
    if not resolved.is_file():
        raise ContractError(f"evidence is not a regular file: {resolved}")
    return DeepEvidenceFile(path=resolved.as_posix(), sha256=_sha256(resolved))


def _verify_model_receipt(
    evidence: DeepEvidenceFile, *, model: str, role: str
) -> None:
    payload = _json_object(Path(evidence.path))
    if payload.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
        raise ContractError(f"{role} receipt is not passing for {model}")
    if payload.get("model") not in {None, model}:
        raise ContractError(f"{role} receipt model differs from {model}")
    if payload.get("commit") not in {None, COMMITS[model]}:
        raise ContractError(f"{role} receipt commit differs from the frozen pin")


def _verify_adapter_config(evidence: DeepEvidenceFile, *, model: str) -> None:
    payload = _json_object(Path(evidence.path))
    if payload.get("model") != model or payload.get("upstream_commit") != COMMITS[model]:
        raise ContractError(f"adapter config differs from the frozen {model} binding")


def _gates_for_phase(phase: str) -> tuple[InformationGate, ...]:
    if phase == "V0":
        return (InformationGate.A0,)
    if phase == "V1":
        return (InformationGate.A1, InformationGate.A2, InformationGate.A3)
    raise ContractError("deep generator phase must be V0 or V1")


def _expected_cells(phase: str) -> int:
    return len(UNIVERSES) * len(MODELS) * len(DeepRuntimePolicy().seeds) * len(
        _gates_for_phase(phase)
    )


def _run_id(
    phase: str,
    gate: InformationGate,
    universe: UniverseClass,
    model: str,
    seed: int,
) -> str:
    return (
        f"{phase.lower()}-{gate.value.lower()}-{universe.value.lower()}-"
        f"{model}-seed-{seed}"
    )


def _required_features(gate: InformationGate) -> tuple[str, ...]:
    """Names sealed by the shared formal-receipt producer.

    The receipt is a data-eligibility artifact shared by tabular and deep
    runners.  The deep packer's internal ``missing::`` channel labels are an
    adapter representation; the receipt intentionally retains the canonical
    ``__missing`` names emitted by ``generate_formal_feature_receipts``.
    """
    layout = default_feature_layout()
    names = list(layout.core)
    if gate.includes_f:
        names.extend(layout.fundamental)
        names.extend(layout.fundamental_missing)
    if gate.includes_s:
        names.extend(layout.market_state)
    return tuple(names)


def _formal_receipt(
    *,
    path: Path | None,
    d0: D0Manifest,
    universe: UniverseClass,
    gate: InformationGate,
    root: Path,
) -> tuple[DeepEvidenceFile, str]:
    if path is None:
        raise ContractError("formal feature receipt is absent")
    evidence = _evidence(path, root)
    receipt = FormalFeatureManifest.from_dict(_json_object(Path(evidence.path)))
    if receipt.d0_manifest_hash != d0.content_hash:
        raise ContractError("formal feature receipt is anchored to a different D0")
    if receipt.dataset_id != f"{d0.dataset_id}:{universe.value}:{gate.value}":
        raise ContractError("formal feature receipt names a different universe/gate")
    expected = set(_required_features(gate))
    actual = set(receipt.feature_eligibility)
    if actual != expected:
        raise ContractError(
            "formal feature receipt does not exactly cover the active gate; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return evidence, receipt.require_formal_eligible()


def _verify_canonical_and_registry(
    *, canonical_root: Path, d0: D0Manifest, universe: UniverseClass
) -> str:
    relatives = (
        f"{universe.value.lower()}/membership.jsonl",
        f"{universe.value.lower()}/features.jsonl",
        f"{universe.value.lower()}/labels.jsonl",
        f"{universe.value.lower()}/masks.jsonl",
        "shared_market_state.jsonl",
    )
    for relative in relatives:
        path = canonical_root / relative
        expected = d0.canonical_table_hashes.get(relative)
        if expected is None or not path.is_file() or _sha256(path) != expected:
            raise ContractError(f"canonical D0 table is absent or unsealed: {relative}")
    memberships = _membership_intervals(canonical_root / relatives[0])
    if universe in {UniverseClass.TECH32, UniverseClass.TECH90}:
        master = _technology_master(memberships, universe=universe)
    else:
        train_master = build_causal_asset_master(
            memberships, known_through=date(2024, 12, 31)
        )
        master = build_causal_asset_master(
            memberships, known_through=VALIDATION_END, previous=train_master
        )
    DeepRuntimePolicy().validate_asset_count(len(master.asset_ids))
    return master.registry.stable_hash()


def _hyperparameters(model: str) -> DeepHyperparameters:
    common: dict[str, object] = {
        "use_norm": 1,
        "freq": "h",
        "d_model": 512,
        "d_ff": 2048,
        "dropout": 0.1,
        "lradj": "type1",
    }
    if model == "itransformer":
        common.update(
            {
                "output_attention": False,
                "embed": "timeF",
                "class_strategy": "projection",
                "factor": 1,
                "n_heads": 8,
                "e_layers": 2,
                "activation": "gelu",
            }
        )
    elif model == "fact":
        common.update(
            {
                "task_name": "long_term_forecast",
                "freq": "n",
                "d_ff": 1024,
                "dilation": [1, 2, 3, 2, 1],
                "num_kernels": 4,
                "core": 0.5,
            }
        )
    elif model == "timexer":
        common.update(
            {
                "task_name": "long_term_forecast",
                "features": "M",
                "patch_len": 16,
                "embed": "timeF",
                "factor": 1,
                "n_heads": 8,
                "e_layers": 2,
               "activation": "gelu",
           }
       )
    elif model == "timepro":
        common.update(
            {
                "patch_len": 12,
                "stride": 6,
                "e_layers": 2,
            }
        )
    elif model == "s4m":
        common.update(
            {
                "e_layers": 4,
                "n_heads": 8,
                "factor": 1,
                "output_attention": False,
                "mask": True,
                "classification": False,
                "plot": 0,
                "num_class": 10,
                "short_len": 50,
                "n": 10,
                "W": 6,
                "en_conv_hidden_size": 256,
                "en_rnn_hidden_sizes": [20, 32],
                "output_keep_prob": 0.9,
                "input_keep_prob": 0.9,
                "K": 10,
                "topK": 10,
                "topM": 100,
                "thres1": 0.6,
                "thres2": 0.3,
                "M": 30,
                "momentum": 0.99,
                "memory_size": 256,
                "per_mem_size": 50,
                "is_training": 1,
            }
        )
    else:
        raise ContractError(f"unsupported deep model: {model}")
    batch_size = 32
    learning_rate = 0.001 if model in {"fact", "s4m"} else 0.0001
    return DeepHyperparameters(
        lookback_weeks=96,
        forecast_steps=1,
        batch_size=batch_size,
        maximum_epochs=10,
        patience=3,
        learning_rate=learning_rate,
        author_arguments=common,
    )


@dataclass(frozen=True)
class BlockedDeepCell(CanonicalModel):
    """One planned deep matrix cell prevented from entering a GPU queue."""

    SCHEMA_NAME: ClassVar[str] = "blocked_deep_cell"

    phase: str
    run_id: str
    model: str
    universe: UniverseClass
    gate: InformationGate
    seed: int
    physical_gpu: int
    reason_code: str
    detail: str
    state: ResultState = ResultState.BLOCKED

    def validate(self) -> None:
        if self.phase not in {"V0", "V1"} or self.gate not in _gates_for_phase(self.phase):
            raise ContractError("blocked deep phase/gate is invalid")
        if self.model not in MODELS or self.seed not in DeepRuntimePolicy().seeds:
            raise ContractError("blocked deep model/seed is outside the frozen matrix")
        if self.physical_gpu != PHYSICAL_GPUS[self.model]:
            raise ContractError("blocked deep cell has the wrong physical GPU")
        if self.run_id != _run_id(
            self.phase, self.gate, self.universe, self.model, self.seed
        ):
            raise ContractError("blocked deep run_id is not canonical")
        if self.reason_code not in {"D0_GATE_BLOCKED", "FORMAL_RECEIPT_MISSING"}:
            raise ContractError("blocked deep reason code is unregistered")
        if self.state is not ResultState.BLOCKED or not self.detail:
            raise ContractError("blocked deep cell requires BLOCKED state and detail")


@dataclass(frozen=True)
class DeepGpuQueueManifest(CanonicalModel):
    """A FIFO queue for one fixed physical GPU; jobs run strictly serially."""

    SCHEMA_NAME: ClassVar[str] = "deep_gpu_queue_manifest"

    queue_id: str
    phase: str
    model: str
    physical_gpu: int
    jobs: tuple[DeepJobSpec, ...]
    max_parallel_jobs: int = 1

    def validate(self) -> None:
        expected_id = f"{self.phase.lower()}-{self.model}-gpu{self.physical_gpu}-serial"
        if self.queue_id != expected_id or self.phase not in {"V0", "V1"}:
            raise ContractError("deep GPU queue identity is invalid")
        if self.model not in MODELS or self.physical_gpu != PHYSICAL_GPUS[self.model]:
            raise ContractError("deep GPU queue model/GPU binding is invalid")
        if self.max_parallel_jobs != 1:
            raise ContractError("jobs on one physical GPU must execute serially")
        if not self.jobs:
            raise ContractError("empty GPU queues must not be emitted")
        if any(
            job.phase != self.phase
            or job.model != self.model
            or job.physical_gpu != self.physical_gpu
            for job in self.jobs
        ):
            raise ContractError("deep GPU queue contains a foreign job")
        identities = [job.run_id for job in self.jobs]
        if len(identities) != len(set(identities)):
            raise ContractError("deep GPU queue contains duplicate jobs")


@dataclass(frozen=True)
class GeneratedDeepJobs:
    jobs: tuple[DeepJobSpec, ...]
    queues: tuple[DeepGpuQueueManifest, ...]
    blocked_cells: tuple[BlockedDeepCell, ...]
    formal_receipts: tuple[tuple[str, DeepEvidenceFile, str], ...]


def _blocked_family(
    *,
    phase: str,
    universe: UniverseClass,
    gates: tuple[InformationGate, ...],
    reason_code: str,
    detail: str,
) -> tuple[BlockedDeepCell, ...]:
    return tuple(
        BlockedDeepCell(
            phase=phase,
            run_id=_run_id(phase, gate, universe, model, seed),
            model=model,
            universe=universe,
            gate=gate,
            seed=seed,
            physical_gpu=PHYSICAL_GPUS[model],
            reason_code=reason_code,
            detail=detail,
        )
        for gate in gates
        for model in MODELS
        for seed in DeepRuntimePolicy().seeds
    )


def build_deep_jobs(
    *,
    phase: str,
    d0_manifest: Path,
    canonical_root: Path,
    upstream_roots: Mapping[str, Path],
    environment_receipts: Mapping[str, Path],
    integrity_receipts: Mapping[str, Path],
    code_receipt: Path,
    adapter_configs: Mapping[str, Path],
    common_config: Path,
    formal_feature_receipts: Mapping[tuple[UniverseClass, InformationGate], Path],
    run_root: Path,
    checkpoint_root: Path,
    approved_root: Path = APPROVED_SERVER_ROOT,
) -> GeneratedDeepJobs:
    """Build the complete V0 (24) or V1 (72) deep-cell accounting."""
    gates = _gates_for_phase(phase)
    for name, mapping in (
        ("upstream", upstream_roots),
        ("environment", environment_receipts),
        ("integrity", integrity_receipts),
        ("adapter", adapter_configs),
    ):
        if set(mapping) != set(MODELS):
            raise ContractError(f"deep generator requires exact {name} mappings")
    root = approved_root.expanduser().resolve(strict=True)
    canonical = _within(canonical_root, root)
    if not canonical.is_dir():
        raise ContractError("canonical root is not a directory")
    d0_evidence = _evidence(d0_manifest, root)
    d0 = D0Manifest.from_dict(_json_object(Path(d0_evidence.path)))
    if d0.cutoff_date < VALIDATION_END:
        raise ContractError("final D0 does not cover all of 2025")
    code_evidence = _evidence(code_receipt, root)
    common_evidence = _evidence(common_config, root)
    environments = {
        model: _evidence(environment_receipts[model], root) for model in MODELS
    }
    integrities = {model: _evidence(integrity_receipts[model], root) for model in MODELS}
    adapters = {model: _evidence(adapter_configs[model], root) for model in MODELS}
    for model in MODELS:
        _verify_model_receipt(environments[model], model=model, role="environment")
        _verify_model_receipt(integrities[model], model=model, role="integrity")
        _verify_adapter_config(adapters[model], model=model)
    if _json_object(Path(common_evidence.path)).get("schema_version") != "deep_adapter_v1":
        raise ContractError("deep common config schema is unsupported")
    upstreams = {model: _within(upstream_roots[model], root) for model in MODELS}
    if any(not path.is_dir() for path in upstreams.values()):
        raise ContractError("one or more pinned author checkout roots are absent")
    output_base = _future_within(run_root / phase.lower(), root)
    checkpoint_base = _future_within(checkpoint_root / phase.lower(), root)
    d0_gates = {item.universe: item for item in d0.universe_gates}
    jobs: list[DeepJobSpec] = []
    blocked: list[BlockedDeepCell] = []
    formal_audit: list[tuple[str, DeepEvidenceFile, str]] = []

    for universe in UNIVERSES:
        universe_gate = d0_gates[universe]
        allowed = (
            {ResultState.PASS, ResultState.PASS_WITH_WARNING}
            if universe in FORMAL_UNIVERSES
            else {ResultState.EXPLORATORY_ONLY}
        )
        if universe_gate.status not in allowed:
            blocked.extend(
                _blocked_family(
                    phase=phase,
                    universe=universe,
                    gates=gates,
                    reason_code="D0_GATE_BLOCKED",
                    detail=(
                        f"final D0 gate is {universe_gate.status.value}; "
                        f"warnings={list(universe_gate.warnings)}"
                    ),
                )
            )
            continue
        try:
            asset_hash = _verify_canonical_and_registry(
                canonical_root=canonical, d0=d0, universe=universe
            )
        except (ContractError, OSError, ValueError) as error:
            blocked.extend(
                _blocked_family(
                    phase=phase,
                    universe=universe,
                    gates=gates,
                    reason_code="D0_GATE_BLOCKED",
                    detail=f"canonical D0 evidence rejected: {error}",
                )
            )
            continue

        for gate in gates:
            formal_evidence: DeepEvidenceFile | None = None
            formal_hash: str | None = None
            if universe in FORMAL_UNIVERSES:
                try:
                    formal_evidence, formal_hash = _formal_receipt(
                        path=formal_feature_receipts.get((universe, gate)),
                        d0=d0,
                        universe=universe,
                        gate=gate,
                        root=root,
                    )
                except (ContractError, OSError, ValueError) as error:
                    blocked.extend(
                        _blocked_family(
                            phase=phase,
                            universe=universe,
                            gates=(gate,),
                            reason_code="FORMAL_RECEIPT_MISSING",
                            detail=f"formal feature evidence rejected: {error}",
                        )
                    )
                    continue
                formal_audit.append(
                    (f"{universe.value}:{gate.value}", formal_evidence, formal_hash)
                )

            for model in MODELS:
                hyperparameters = _hyperparameters(model)
                for seed in DeepRuntimePolicy().seeds:
                    run_id = _run_id(phase, gate, universe, model, seed)
                    output_dir = (
                        output_base
                        / model
                        / universe.value.lower()
                        / gate.value
                        / str(seed)
                    )
                    checkpoint_dir = (
                        checkpoint_base / model / universe.value.lower() / gate.value / str(seed)
                    )
                    evidence_payload = {
                        "d0_manifest": d0_evidence.to_dict(),
                        "environment_receipt": environments[model].to_dict(),
                        "integrity_receipt": integrities[model].to_dict(),
                        "code_receipt": code_evidence.to_dict(),
                        "adapter_config": adapters[model].to_dict(),
                        "common_config": common_evidence.to_dict(),
                        "formal_feature_manifest": (
                            formal_evidence.to_dict() if formal_evidence is not None else None
                        ),
                    }
                    cell_hash = canonical_hash(
                        {
                            "schema_version": "deep_cell_config_v1",
                            "phase": phase,
                            "run_id": run_id,
                            "model": model,
                            "universe": universe.value,
                            "scope": (
                                "EXPLORATORY_ONLY"
                                if universe not in FORMAL_UNIVERSES
                                else "FORMAL"
                            ),
                            "gate": gate.value,
                            "seed": seed,
                            "physical_gpu": PHYSICAL_GPUS[model],
                            "upstream_commit": COMMITS[model],
                            "asset_registry_hash": asset_hash,
                            "formal_feature_manifest_hash": formal_hash,
                            "hyperparameters": hyperparameters.to_dict(),
                            "evidence": evidence_payload,
                            "selection_window": ["2025-01-01", "2025-12-31"],
                            "legacy_2026_selection_allowed": False,
                        }
                    )
                    jobs.append(
                        DeepJobSpec(
                            phase=phase,
                            run_id=run_id,
                            model=model,
                            universe=universe,
                            gate=gate,
                            seed=seed,
                            physical_gpu=PHYSICAL_GPUS[model],
                            canonical_root=canonical.as_posix(),
                            upstream_root=upstreams[model].as_posix(),
                            output_dir=output_dir.as_posix(),
                            checkpoint_dir=checkpoint_dir.as_posix(),
                            upstream_commit=COMMITS[model],
                            asset_registry_hash=asset_hash,
                            cell_config_hash=cell_hash,
                            hyperparameters=hyperparameters,
                            d0_manifest=d0_evidence,
                            environment_receipt=environments[model],
                            integrity_receipt=integrities[model],
                            code_receipt=code_evidence,
                            adapter_config=adapters[model],
                            common_config=common_evidence,
                            formal_feature_manifest=formal_evidence,
                        )
                    )

    expected = _expected_cells(phase)
    if len(jobs) + len(blocked) != expected:
        raise ContractError(
            "deep matrix accounting drifted; "
            f"runnable={len(jobs)}, blocked={len(blocked)}, expected={expected}"
        )
    identities = {job.run_id for job in jobs} | {cell.run_id for cell in blocked}
    if len(identities) != expected:
        raise ContractError("deep matrix contains duplicate or missing planned cells")
    queues = tuple(
        DeepGpuQueueManifest(
            queue_id=f"{phase.lower()}-{model}-gpu{PHYSICAL_GPUS[model]}-serial",
            phase=phase,
            model=model,
            physical_gpu=PHYSICAL_GPUS[model],
            jobs=tuple(job for job in jobs if job.model == model),
        )
        for model in MODELS
        if any(job.model == model for job in jobs)
    )
    return GeneratedDeepJobs(tuple(jobs), queues, tuple(blocked), tuple(formal_audit))


def write_deep_jobs(
    generated: GeneratedDeepJobs,
    *,
    phase: str,
    job_root: Path,
    queue_root: Path,
    approved_root: Path = APPROVED_SERVER_ROOT,
) -> tuple[Path, ...]:
    """Atomically publish immutable job files, GPU queues and audit receipt."""
    expected = _expected_cells(phase)
    if (
        len(generated.jobs) + len(generated.blocked_cells) != expected
        or any(job.phase != phase for job in generated.jobs)
        or any(cell.phase != phase for cell in generated.blocked_cells)
    ):
        raise ContractError("published deep matrix does not match its declared phase")
    queued = tuple(job.run_id for queue in generated.queues for job in queue.jobs)
    runnable = tuple(job.run_id for job in generated.jobs)
    if len(queued) != len(set(queued)) or set(queued) != set(runnable):
        raise ContractError("deep GPU queues do not cover runnable jobs exactly once")
    root = approved_root.expanduser().resolve(strict=True)
    job_dir = _future_within(job_root / phase.lower(), root)
    queue_dir = _future_within(queue_root / phase.lower(), root)
    if job_dir.exists() or queue_dir.exists():
        raise ContractError("generated deep job/queue directory exists; refusing overwrite")
    job_tmp = job_dir.with_name(job_dir.name + ".tmp")
    queue_tmp = queue_dir.with_name(queue_dir.name + ".tmp")
    if job_tmp.exists() or queue_tmp.exists():
        raise ContractError("stale deep job-generation temporary directory exists")
    job_tmp.mkdir(parents=True)
    queue_tmp.mkdir(parents=True)
    written: list[Path] = []
    for job in generated.jobs:
        path = job_tmp / f"{job.run_id}.json"
        path.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n")
        written.append(job_dir / path.name)
    for queue in generated.queues:
        path = queue_tmp / f"{queue.queue_id}.json"
        path.write_text(json.dumps(queue.to_dict(), indent=2, sort_keys=True) + "\n")
        written.append(queue_dir / path.name)
    receipt = {
        "schema_version": "deep_job_generation_receipt_v1",
        "phase": phase,
        "planned_cell_count": expected,
        "runnable_job_count": len(generated.jobs),
        "blocked_cell_count": len(generated.blocked_cells),
        "gpu_queues": {
            str(queue.physical_gpu): {
                "model": queue.model,
                "serial_within_gpu": True,
                "job_count": len(queue.jobs),
            }
            for queue in generated.queues
        },
        "gpu_queues_may_run_in_parallel": True,
        "blocked_cells": [cell.to_dict() for cell in generated.blocked_cells],
        "formal_feature_receipts": {
            key: {"file": evidence.to_dict(), "manifest_hash": stable_hash}
            for key, evidence, stable_hash in generated.formal_receipts
        },
        "selection_window": ["2025-01-01", "2025-12-31"],
        "legacy_2026_selection_allowed": False,
    }
    receipt_path = job_tmp / "generation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    written.append(job_dir / receipt_path.name)
    job_tmp.replace(job_dir)
    queue_tmp.replace(queue_dir)
    return tuple(written)
