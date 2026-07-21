"""Synthetic generator checks; execute only on the approved server."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from a_share_research.contracts import ContractError, FormalFeatureManifest
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.experiments.tabular_job_generator import (
    MODELS,
    build_tabular_jobs,
    validation_registry_hash,
    write_tabular_jobs,
)
from a_share_research.models.tabular import (
    InformationSet,
    TabularSample,
    default_feature_layout,
)
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _samples(*, include_2026: bool = False, shrink: bool = False):
    rows = [
        TabularSample(date(2025, 1, 3), "000001.SZ", {}, {}, None, False, False, False),
        TabularSample(date(2025, 1, 10), "000001.SZ", {}, {}, None, False, False, False),
    ]
    if not shrink:
        rows.extend(
            [
                TabularSample(
                    date(2025, 1, 10), "000002.SZ", {}, {}, None, False, False, False
                ),
                TabularSample(
                    date(2025, 1, 17), "000001.SZ", {}, {}, None, False, False, False
                ),
                TabularSample(
                    date(2025, 1, 17), "000002.SZ", {}, {}, None, False, False, False
                ),
            ]
        )
    if include_2026:
        rows.append(
            TabularSample(date(2026, 1, 2), "000001.SZ", {}, {}, None, False, False, False)
        )
    return tuple(rows)


class _Loader:
    rows = _samples()
    calls: list[dict[str, object]] = []

    def __init__(self, root: Path, universe: str) -> None:
        self.root = root
        self.universe = universe

    def iter_tabular_samples(self, **kwargs: object):
        self.calls.append(kwargs)
        yield from self.rows


def _information_names(information_set: InformationSet) -> tuple[str, ...]:
    layout = default_feature_layout()
    names = list(layout.core)
    if information_set.enables_f:
        names.extend(layout.fundamental)
        names.extend(layout.fundamental_missing)
    if information_set.enables_s:
        names.extend(layout.market_state)
    return tuple(names)


def _fixture(
    tmp_path: Path,
    *,
    star50_status: ResultState = ResultState.PASS,
):
    root = tmp_path / "server"
    canonical = root / "data/canonical/d0-v1"
    canonical.mkdir(parents=True)
    canonical_hashes: dict[str, str] = {}
    for universe in UniverseClass:
        for filename in (
            "membership.jsonl",
            "features.jsonl",
            "labels.jsonl",
            "masks.jsonl",
        ):
            path = canonical / universe.value.lower() / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{universe.value}:{filename}\n", encoding="utf-8")
            canonical_hashes[path.relative_to(canonical).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    market_state = canonical / "shared_market_state.jsonl"
    market_state.write_text("shared-market-state\n", encoding="utf-8")
    canonical_hashes["shared_market_state.jsonl"] = hashlib.sha256(
        market_state.read_bytes()
    ).hexdigest()
    d0 = D0Manifest(
        dataset_id="d0-final-test",
        created_at_utc=datetime(2026, 7, 19, tzinfo=timezone.utc),
        cutoff_date=date(2026, 7, 17),
        raw_snapshot_hashes={"raw": "1" * 64},
        canonical_table_hashes=canonical_hashes,
        security_master_hash="3" * 64,
        trading_calendar_hash="4" * 64,
        feature_schema_hash="5" * 64,
        market_state_hash=canonical_hashes["shared_market_state.jsonl"],
        universe_gates=tuple(
            UniverseGate(
                universe=universe,
                status=(
                    star50_status
                    if universe is UniverseClass.STAR50
                    else (
                        ResultState.EXPLORATORY_ONLY
                        if universe in {UniverseClass.TECH32, UniverseClass.TECH90}
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
    evidence_root = root / "receipts"
    evidence_root.mkdir()
    code = evidence_root / "source-manifest.json"
    code.write_text('{"source":"test"}', encoding="utf-8")
    environments: dict[str, Path] = {}
    configs: dict[str, Path] = {}
    for model in MODELS:
        environment = evidence_root / f"{model}-environment.json"
        environment.write_text(json.dumps({"model": model}), encoding="utf-8")
        environments[model] = environment
        source = PROJECT_ROOT / f"configs/models/{model}-v1.json"
        target = evidence_root / f"{model}-config.json"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        configs[model] = target
    layout = evidence_root / "tabular-layout-v1.json"
    layout.write_text(
        (PROJECT_ROOT / "configs/features/tabular-layout-v1.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    formal: dict[tuple[UniverseClass, InformationSet], Path] = {}
    for universe in (UniverseClass.CSI300, UniverseClass.STAR50):
        for information_set in InformationSet:
            receipt = FormalFeatureManifest(
                dataset_id=f"{d0.dataset_id}:{universe.value}:{information_set.value}",
                d0_manifest_hash=d0.content_hash,
                feature_eligibility={
                    name: True for name in _information_names(information_set)
                },
            )
            path = evidence_root / f"formal-{universe.value}-{information_set.value}.json"
            path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
            formal[(universe, information_set)] = path
    return {
        "approved_root": root,
        "d0_manifest": d0_path,
        "canonical_root": canonical,
        "environment_receipts": environments,
        "code_receipt": code,
        "model_configs": configs,
        "layout_config": layout,
        "formal_feature_receipts": formal,
        "output_root": root / "runs",
        "loader_factory": _Loader,
    }


def test_v0_generator_emits_exact_eight_cells_and_one_serial_queue(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    _Loader.rows = _samples()
    _Loader.calls = []
    generated = build_tabular_jobs(phase="V0", **args)
    assert len(generated.jobs) == 8
    assert not generated.blocked_cells
    assert len(generated.queues) == 1
    assert generated.queues[0].max_jobs == 16
    assert generated.queues[0].jobs == generated.jobs
    assert {job.information_set for job in generated.jobs} == {InformationSet.A0}
    assert {job.model for job in generated.jobs} == set(MODELS)
    assert {job.universe for job in generated.jobs} == set(UniverseClass)
    assert all(Path(job.output_dir).name == job.run_id for job in generated.jobs)
    assert all(job.cell_config_hash != "0" * 64 for job in generated.jobs)
    assert all(job.run_id.endswith("seed-20260719") for job in generated.jobs)
    assert all(
        job.formal_feature_manifest_hash is not None
        for job in generated.jobs
        if job.universe in {UniverseClass.CSI300, UniverseClass.STAR50}
    )
    assert all(
        job.formal_feature_manifest_hash is None
        for job in generated.jobs
        if job.universe in {UniverseClass.TECH32, UniverseClass.TECH90}
    )
    assert len(_Loader.calls) == 4
    assert all(
        call
        == {
            "horizon": 5,
            "relative_target": True,
            "start": date(2025, 1, 1),
            "end": date(2025, 12, 31),
            "complete_panel": True,
        }
        for call in _Loader.calls
    )


def test_v1_generator_emits_only_24_incremental_cells_in_two_queues(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    _Loader.rows = _samples()
    generated = build_tabular_jobs(phase="V1", **args)
    assert len(generated.jobs) == 24
    assert not generated.blocked_cells
    assert [len(queue.jobs) for queue in generated.queues] == [16, 8]
    assert all(queue.max_jobs == 16 for queue in generated.queues)
    assert {job.information_set for job in generated.jobs} == {
        InformationSet.A1,
        InformationSet.A2,
        InformationSet.A3,
    }
    assert all(job.information_set is not InformationSet.A0 for job in generated.jobs)


def test_missing_formal_receipt_blocks_only_its_cells(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args["formal_feature_receipts"].pop((UniverseClass.CSI300, InformationSet.A0))
    generated = build_tabular_jobs(phase="V0", **args)
    assert len(generated.jobs) == 6
    assert len(generated.blocked_cells) == 2
    assert {cell.universe for cell in generated.blocked_cells} == {
        UniverseClass.CSI300
    }
    assert {cell.reason_code for cell in generated.blocked_cells} == {
        "FORMAL_RECEIPT_MISSING"
    }

    args = _fixture(tmp_path / "wrong")
    key = (UniverseClass.CSI300, InformationSet.A0)
    wrong = FormalFeatureManifest(
        dataset_id="wrong",
        d0_manifest_hash="f" * 64,
        feature_eligibility={name: True for name in _information_names(InformationSet.A0)},
    )
    args["formal_feature_receipts"][key].write_text(
        json.dumps(wrong.to_dict()), encoding="utf-8"
    )
    generated = build_tabular_jobs(phase="V0", **args)
    assert len(generated.jobs) == 6
    assert len(generated.blocked_cells) == 2
    assert all("final D0 content hash" in cell.detail for cell in generated.blocked_cells)

    args = _fixture(tmp_path / "v1")
    args["formal_feature_receipts"] = {
        key: path
        for key, path in args["formal_feature_receipts"].items()
        if key[0] is not UniverseClass.CSI300
    }
    generated = build_tabular_jobs(phase="V1", **args)
    assert len(generated.jobs) == 18
    assert len(generated.blocked_cells) == 6
    assert all(
        cell.reason_code == "FORMAL_RECEIPT_MISSING"
        for cell in generated.blocked_cells
    )


@pytest.mark.parametrize(
    ("phase", "runnable", "blocked", "queue_sizes"),
    (("V0", 6, 2, [6]), ("V1", 18, 6, [16, 2])),
)
def test_star50_block_is_recorded_while_other_universes_continue(
    tmp_path: Path,
    phase: str,
    runnable: int,
    blocked: int,
    queue_sizes: list[int],
) -> None:
    args = _fixture(tmp_path, star50_status=ResultState.BLOCKED)
    generated = build_tabular_jobs(phase=phase, **args)
    assert len(generated.jobs) == runnable
    assert len(generated.blocked_cells) == blocked
    assert [len(queue.jobs) for queue in generated.queues] == queue_sizes
    assert {cell.universe for cell in generated.blocked_cells} == {
        UniverseClass.STAR50
    }
    assert {cell.reason_code for cell in generated.blocked_cells} == {
        "D0_GATE_BLOCKED"
    }
    write_tabular_jobs(
        generated,
        phase=phase,
        job_root=args["approved_root"] / "runtime/jobs",
        queue_root=args["approved_root"] / "runtime/queues",
        approved_root=args["approved_root"],
    )
    receipt = json.loads(
        (
            args["approved_root"]
            / "runtime/jobs"
            / phase.lower()
            / "generation_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["planned_cell_count"] == (8 if phase == "V0" else 24)
    assert receipt["runnable_job_count"] == runnable
    assert receipt["blocked_cell_count"] == blocked
    assert len(receipt["blocked_cells"]) == blocked


def test_registry_hash_rejects_2026_and_non_append_only_panels(tmp_path: Path) -> None:
    _Loader.rows = _samples(include_2026=True)
    with pytest.raises(ContractError, match="outside the sealed 2025"):
        validation_registry_hash(_Loader(tmp_path, "CSI300"))

    rows = list(_samples())
    rows[-1] = replace(rows[-1], ts_code="000003.SZ")
    _Loader.rows = tuple(rows)
    with pytest.raises(ContractError, match="not append-only"):
        validation_registry_hash(_Loader(tmp_path, "CSI300"))


def test_formal_gate_and_exploratory_gate_cannot_be_interchanged(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    payload = json.loads(args["d0_manifest"].read_text(encoding="utf-8"))
    gates = payload["universe_gates"]
    for gate in gates:
        if gate["universe"] == "TECH32":
            gate["status"] = "PASS"
    args["d0_manifest"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        build_tabular_jobs(phase="V0", **args)
