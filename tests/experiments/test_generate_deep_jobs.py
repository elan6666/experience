"""Synthetic deep job-generation checks; execute only on the server."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import a_share_research.experiments.deep_runner as deep_runner
from a_share_research.adapters.common import InformationGate
from a_share_research.contracts import ContractError, FormalFeatureManifest
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.experiments.deep_job_generator import (
    COMMITS,
    MODELS,
    PHYSICAL_GPUS,
    build_deep_jobs,
    write_deep_jobs,
)
from a_share_research.models.tabular import default_feature_layout
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

HASH = "a" * 64


def _required_features(gate: InformationGate) -> tuple[str, ...]:
    layout = default_feature_layout()
    names = list(layout.core)
    if gate.includes_f:
        names.extend(layout.fundamental)
        names.extend(layout.fundamental_missing)
    if gate.includes_s:
        names.extend(layout.market_state)
    return tuple(names)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    star50_status: ResultState = ResultState.PASS,
) -> dict[str, object]:
    root = tmp_path / "server"
    root.mkdir()
    monkeypatch.setattr(deep_runner, "_SERVER_ROOT", root.resolve())
    canonical = root / "data/canonical/d0-v1"
    canonical_hashes: dict[str, str] = {}
    for universe in UniverseClass:
        for filename in ("membership.jsonl", "features.jsonl", "labels.jsonl", "masks.jsonl"):
            path = canonical / universe.value.lower() / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{universe.value}:{filename}\n", encoding="utf-8")
            canonical_hashes[path.relative_to(canonical).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    market = canonical / "shared_market_state.jsonl"
    market.write_text("shared-market-state\n", encoding="utf-8")
    canonical_hashes["shared_market_state.jsonl"] = hashlib.sha256(
        market.read_bytes()
    ).hexdigest()
    d0 = D0Manifest(
        dataset_id="d0-deep-generator-test",
        created_at_utc=datetime(2026, 7, 19, tzinfo=timezone.utc),
        cutoff_date=date(2026, 7, 17),
        raw_snapshot_hashes={"raw": "1" * 64},
        canonical_table_hashes=canonical_hashes,
        security_master_hash="2" * 64,
        trading_calendar_hash="3" * 64,
        feature_schema_hash="4" * 64,
        market_state_hash=canonical_hashes["shared_market_state.jsonl"],
        universe_gates=tuple(
            UniverseGate(
                universe=universe,
                status=(
                    star50_status
                    if universe is UniverseClass.STAR50
                    else (
                        ResultState.EXPLORATORY_ONLY
                        if universe in {UniverseClass.TECH32, UniverseClass.TECH100}
                        else ResultState.PASS
                    )
                ),
                membership_coverage=1.0,
                core_coverage=1.0,
                duplicate_keys=0,
                pit_violations=0,
                label_boundary_violations=0,
            )
            for universe in UniverseClass
        ),
        provider_transport_notice="provider uses plain HTTP",
    )
    d0_path = root / "data/manifests/d0-v1.json"
    d0_path.parent.mkdir(parents=True)
    d0_path.write_text(json.dumps(d0.to_dict()), encoding="utf-8")
    receipts = root / "receipts"
    receipts.mkdir()
    code = receipts / "source.json"
    code.write_text('{"source":"test"}\n', encoding="utf-8")
    common = receipts / "deep-common.json"
    common.write_text('{"schema_version":"deep_adapter_v1"}\n', encoding="utf-8")
    environments: dict[str, Path] = {}
    integrities: dict[str, Path] = {}
    adapters: dict[str, Path] = {}
    upstreams: dict[str, Path] = {}
    for model in MODELS:
        upstream = root / "upstreams" / model
        upstream.mkdir(parents=True)
        upstreams[model] = upstream
        environment = receipts / f"{model}-environment.json"
        environment.write_text(
            json.dumps({"status": "PASS", "model": model, "commit": COMMITS[model]}),
            encoding="utf-8",
        )
        environments[model] = environment
        integrity = receipts / f"{model}-integrity.json"
        integrity.write_text(
            json.dumps({"status": "PASS", "model": model, "commit": COMMITS[model]}),
            encoding="utf-8",
        )
        integrities[model] = integrity
        adapter = receipts / f"{model}-adapter.json"
        adapter.write_text(
            json.dumps({"model": model, "upstream_commit": COMMITS[model]}),
            encoding="utf-8",
        )
        adapters[model] = adapter
    formal: dict[tuple[UniverseClass, InformationGate], Path] = {}
    for universe in (UniverseClass.CSI300, UniverseClass.STAR50):
        for gate in InformationGate:
            receipt = FormalFeatureManifest(
                dataset_id=f"{d0.dataset_id}:{universe.value}:{gate.value}",
                d0_manifest_hash=d0.content_hash,
                feature_eligibility={name: True for name in _required_features(gate)},
            )
            path = receipts / f"formal-{universe.value}-{gate.value}.json"
            path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
            formal[(universe, gate)] = path
    return {
        "approved_root": root,
        "d0_manifest": d0_path,
        "canonical_root": canonical,
        "upstream_roots": upstreams,
        "environment_receipts": environments,
        "integrity_receipts": integrities,
        "code_receipt": code,
        "adapter_configs": adapters,
        "common_config": common,
        "formal_feature_receipts": formal,
        "run_root": root / "runs",
        "checkpoint_root": root / "checkpoints",
    }


def _stub_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "a_share_research.experiments.deep_job_generator._verify_canonical_and_registry",
        lambda **_: HASH,
    )


def test_v0_emits_24_cells_and_two_parallel_capable_serial_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    _stub_registry(monkeypatch)
    generated = build_deep_jobs(phase="V0", **args)
    assert len(generated.jobs) == 48
    assert not generated.blocked_cells
    assert {job.gate for job in generated.jobs} == {InformationGate.A0}
    assert {job.seed for job in generated.jobs} == {20260719, 20260720, 20260721}
    assert {queue.physical_gpu for queue in generated.queues} == {0, 1}
    assert all(queue.max_parallel_jobs == 1 and len(queue.jobs) == 12 for queue in generated.queues)
    assert all(job.physical_gpu == PHYSICAL_GPUS[job.model] for job in generated.jobs)
    assert all(job.upstream_commit == COMMITS[job.model] for job in generated.jobs)
    assert all(job.asset_registry_hash == HASH for job in generated.jobs)
    assert all(
        job.formal_feature_manifest is None
        for job in generated.jobs
        if job.universe in {UniverseClass.TECH32, UniverseClass.TECH100}
    )
    assert all(
        job.formal_feature_manifest is not None
        for job in generated.jobs
        if job.universe in {UniverseClass.CSI300, UniverseClass.STAR50}
    )
    environment_hashes = {
        model: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for model, path in args["environment_receipts"].items()
    }
    assert all(
        job.environment_receipt.sha256 == environment_hashes[job.model]
        for job in generated.jobs
    )


def test_v1_emits_72_incremental_cells_without_retraining_a0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    _stub_registry(monkeypatch)
    generated = build_deep_jobs(phase="V1", **args)
    assert len(generated.jobs) == 144
    assert not generated.blocked_cells
    assert {job.gate for job in generated.jobs} == {
        InformationGate.A1,
        InformationGate.A2,
        InformationGate.A3,
    }
    assert all(len(queue.jobs) == 36 for queue in generated.queues)


def test_blocked_d0_gate_accounts_for_every_star50_seed_model_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch, star50_status=ResultState.BLOCKED)
    _stub_registry(monkeypatch)
    generated = build_deep_jobs(phase="V0", **args)
    assert len(generated.jobs) == 36
    assert len(generated.blocked_cells) == 12
    assert {cell.universe for cell in generated.blocked_cells} == {UniverseClass.STAR50}
    assert {cell.reason_code for cell in generated.blocked_cells} == {"D0_GATE_BLOCKED"}
    assert len({cell.run_id for cell in generated.blocked_cells}) == 12


def test_missing_formal_receipt_blocks_only_one_universe_gate_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    _stub_registry(monkeypatch)
    args["formal_feature_receipts"].pop((UniverseClass.CSI300, InformationGate.A2))
    generated = build_deep_jobs(phase="V1", **args)
    assert len(generated.jobs) == 132
    assert len(generated.blocked_cells) == 12
    assert {
        (cell.universe, cell.gate, cell.reason_code) for cell in generated.blocked_cells
    } == {
        (
            UniverseClass.CSI300,
            InformationGate.A2,
            "FORMAL_RECEIPT_MISSING",
        )
    }


def test_write_receipt_proves_2026_is_excluded_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    _stub_registry(monkeypatch)
    generated = build_deep_jobs(phase="V0", **args)
    job_root = args["approved_root"] / "jobs/deep"
    queue_root = args["approved_root"] / "queues/deep"
    write_deep_jobs(
        generated,
        phase="V0",
        job_root=job_root,
        queue_root=queue_root,
        approved_root=args["approved_root"],
    )
    receipt = json.loads((job_root / "v0/generation_receipt.json").read_text())
    assert receipt["planned_cell_count"] == 48
    assert receipt["selection_window"] == ["2025-01-01", "2025-12-31"]
    assert receipt["legacy_2026_selection_allowed"] is False
    assert receipt["gpu_queues_may_run_in_parallel"] is True
    assert all(item["serial_within_gpu"] for item in receipt["gpu_queues"].values())
    with pytest.raises(ContractError, match="refusing overwrite"):
        write_deep_jobs(
            generated,
            phase="V0",
            job_root=job_root,
            queue_root=queue_root,
            approved_root=args["approved_root"],
        )


def test_exact_model_evidence_mappings_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    args["integrity_receipts"].pop("fact")
    with pytest.raises(ContractError, match="exact integrity mappings"):
        build_deep_jobs(phase="V0", **args)
