"""Synthetic orchestration checks; execute only on the approved server."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from a_share_research.contracts import (
    AssetRegistry,
    ContractError,
    CoverageState,
    FormalFeatureManifest,
    PredictionFrame,
    PredictionRecord,
)
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.experiments.source_evidence import source_manifest_payload
from a_share_research.experiments.tabular_runner import (
    EvidenceFile,
    TabularCellRunner,
    TabularJobSpec,
    TabularQueueManifest,
    TabularRunFailure,
    run_cpu_queue,
    tabular_cell_config_hash,
)
from a_share_research.models.tabular import (
    InformationSet,
    RidgeConfig,
    TabularDiagnostics,
    TabularModelResult,
    TabularSample,
    default_feature_layout,
)
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> EvidenceFile:
    return EvidenceFile(path.resolve().as_posix(), _sha(path))


def _feature_payload() -> tuple[dict[str, float], dict[str, bool]]:
    layout = default_feature_layout()
    names = layout.core + layout.fundamental + layout.market_state
    return ({name: 1.0 for name in names}, {name: False for name in names})


def _samples(*, include_legacy: bool = False) -> tuple[TabularSample, ...]:
    values, missing = _feature_payload()
    rows = [
        TabularSample(date(2019, 1, 4), "000001.SZ", values, missing, 0.01),
        TabularSample(date(2024, 12, 27), "000001.SZ", values, missing, 0.02),
        TabularSample(date(2025, 1, 3), "000001.SZ", values, missing, 0.03),
        TabularSample(date(2025, 1, 10), "000001.SZ", values, missing, 0.04),
        TabularSample(
            date(2025, 1, 10),
            "000002.SZ",
            {},
            {},
            None,
            member=False,
            observed=False,
            complete_history=False,
        ),
    ]
    if include_legacy:
        rows.append(TabularSample(date(2026, 1, 2), "000001.SZ", values, missing, 0.05))
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

    def iter_labels(self):
        exits = {
            date(2019, 1, 4): date(2019, 1, 14),
            date(2024, 12, 27): date(2025, 1, 7),
            date(2025, 1, 3): date(2025, 1, 13),
            date(2025, 1, 10): date(2025, 1, 20),
        }
        for row in self.rows:
            if row.target is not None and row.signal_date in exits:
                yield SimpleNamespace(
                    horizon=5,
                    signal_date=row.signal_date,
                    ts_code=row.ts_code,
                    exit_date=exits[row.signal_date],
                )


class _FakeAdapter:
    def __init__(self, job, layout, gate, payload, *, corrupt: bool = False) -> None:
        self.job = job
        self.layout = layout
        self.gate = gate
        self.payload = payload
        self.corrupt = corrupt
        self.preprocessor = SimpleNamespace(state={"fit_end": "2024-12-31"})

    def fit_predict(self, **kwargs: object) -> TabularModelResult:
        training = kwargs["training"]
        prediction = kwargs["prediction"]
        assert all(row.signal_date <= date(2024, 12, 31) for row in training)
        assert all(row.signal_date != date(2024, 12, 27) for row in training)
        assert all(row.signal_date.year == 2025 for row in prediction)
        records = tuple(
            PredictionRecord(
                row.signal_date,
                row.ts_code,
                0.1 if row.coverage_state is CoverageState.SCORED else None,
                row.coverage_state,
            )
            for row in prediction
        )
        if self.corrupt:
            records = records[:-1]
        frame = PredictionFrame(run_id=self.job.run_id, records=records)
        config_hash = RidgeConfig.from_mapping(self.payload).stable_hash()
        diagnostics = TabularDiagnostics(
            model="Ridge",
            information_set=self.gate.information_set.value,
            config_hash=config_hash,
            layout_hash=self.layout.stable_hash(),
            gate_hash=self.gate.stable_hash(),
            preprocessing_hash=HASH,
            fit_data_hash=kwargs["fit_data_hash"],
            fold_id=kwargs["fold_id"],
            training_start=min(row.signal_date for row in training),
            training_end=kwargs["fit_end"],
            validation_start=None,
            validation_end=None,
            n_train=len(training),
            n_validation=0,
            n_prediction=len(records),
            n_scored=sum(row.coverage_state is CoverageState.SCORED for row in records),
            seed=20260719,
            feature_importance=tuple((name, 0.0) for name in self.layout.columns),
        )
        return TabularModelResult(frame, diagnostics)


def _write_fixture(tmp_path: Path) -> tuple[TabularJobSpec, Path]:
    canonical = tmp_path / "canonical"
    universe_root = canonical / "csi300"
    universe_root.mkdir(parents=True)
    tables: dict[str, str] = {}
    for relative in (
        "csi300/membership.jsonl",
        "csi300/features.jsonl",
        "csi300/labels.jsonl",
        "csi300/masks.jsonl",
        "shared_market_state.jsonl",
    ):
        path = canonical / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
        tables[relative] = _sha(path)
    d0 = D0Manifest(
        dataset_id="d0-test",
        created_at_utc=datetime(2026, 7, 19, tzinfo=timezone.utc),
        cutoff_date=date(2026, 7, 17),
        raw_snapshot_hashes={"raw": "1" * 64},
        canonical_table_hashes=tables,
        security_master_hash="2" * 64,
        trading_calendar_hash="3" * 64,
        feature_schema_hash="4" * 64,
        market_state_hash=tables["shared_market_state.jsonl"],
        universe_gates=tuple(
            UniverseGate(
                universe=universe,
                status=(
                    ResultState.EXPLORATORY_ONLY
                    if universe in {UniverseClass.TECH32, UniverseClass.TECH100}
                    else ResultState.PASS
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
    d0_path = tmp_path / "d0.json"
    d0_path.write_text(json.dumps(d0.to_dict()), encoding="utf-8")
    formal = FormalFeatureManifest(
        dataset_id="d0-test:CSI300:A0",
        d0_manifest_hash=d0.content_hash,
        feature_eligibility={
            name: True for name in default_feature_layout().core
        },
    )
    formal_path = tmp_path / "formal.json"
    formal_path.write_text(json.dumps(formal.to_dict()), encoding="utf-8")
    env = tmp_path / "environment.json"
    code = tmp_path / "code.json"
    env.write_text(json.dumps({"status": "PASS", "model": "ridge"}), encoding="utf-8")
    code.write_text(
        json.dumps(source_manifest_payload(PROJECT_ROOT)), encoding="utf-8"
    )
    model = tmp_path / "ridge.json"
    layout = tmp_path / "layout.json"
    model.write_text(
        (PROJECT_ROOT / "configs/models/ridge-v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    layout.write_text(
        (PROJECT_ROOT / "configs/features/tabular-layout-v1.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    run_id = "v0-a0-csi300-ridge-seed-20260719"
    job = TabularJobSpec(
        phase="V0",
        run_id=run_id,
        model="ridge",
        universe=UniverseClass.CSI300,
        information_set=InformationSet.A0,
        seed=20260719,
        canonical_root=canonical.resolve().as_posix(),
        output_dir=(tmp_path / "runs" / run_id).resolve().as_posix(),
        d0_manifest=_evidence(d0_path),
        environment_receipt=_evidence(env),
        code_receipt=_evidence(code),
        model_config=_evidence(model),
        layout_config=_evidence(layout),
        asset_registry_hash=AssetRegistry(("000001.SZ", "000002.SZ")).stable_hash(),
        cell_config_hash="0" * 64,
        upstream_commit="internal:scikit-learn-1.9.0",
        formal_feature_manifest_hash=formal.stable_hash(),
        formal_feature_receipt=_evidence(formal_path),
    )
    job = replace(job, cell_config_hash=tabular_cell_config_hash(job, d0))
    return job, canonical


def test_single_cell_is_sealed_complete_and_atomically_published(tmp_path: Path) -> None:
    job, _ = _write_fixture(tmp_path)
    _Loader.rows = _samples()
    _Loader.calls = []
    runner = TabularCellRunner(
        loader_factory=_Loader,
        adapter_factory=lambda *args: _FakeAdapter(*args),
        clock=lambda: datetime(2026, 7, 19, tzinfo=timezone.utc),
        approved_root=tmp_path,
    )
    output = runner.run(job)
    assert output.is_dir() and not output.with_name(output.name + ".tmp").exists()
    assert {path.name for path in output.iterdir()} == {
        "predictions.json",
        "diagnostics.json",
        "preprocessing_state.json",
        "run_manifest.json",
        "run_receipt.json",
    }
    receipt = json.loads((output / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["protocol"]["train"] == ["2019-01-01", "2024-12-31"]
    assert receipt["protocol"]["validation"] == ["2025-01-01", "2025-12-31"]
    assert receipt["protocol"]["legacy_2026_selection_allowed"] is False
    assert receipt["counts"]["prediction_complete_panel"] == 3
    assert _Loader.calls == [
        {
            "horizon": 5,
            "relative_target": True,
            "start": date(2019, 1, 1),
            "end": date(2025, 12, 31),
            "complete_panel": True,
        }
    ]


def test_legacy_row_and_wrong_registry_fail_as_invalid_data(tmp_path: Path) -> None:
    job, _ = _write_fixture(tmp_path)
    _Loader.rows = _samples(include_legacy=True)
    runner = TabularCellRunner(loader_factory=_Loader, approved_root=tmp_path)
    with pytest.raises(TabularRunFailure) as caught:
        runner.run(job)
    assert caught.value.state is ResultState.INVALID_DATA
    assert not Path(job.output_dir).exists()
    assert Path(job.output_dir + ".failure.json").is_file()

    clean_job, _ = _write_fixture(tmp_path / "second")
    _Loader.rows = _samples()
    wrong = replace(clean_job, asset_registry_hash="9" * 64)
    with pytest.raises(TabularRunFailure) as caught:
        runner.run(wrong)
    assert caught.value.state is ResultState.INVALID_DATA


def test_adapter_cannot_drop_complete_panel_rows(tmp_path: Path) -> None:
    job, _ = _write_fixture(tmp_path)
    _Loader.rows = _samples()
    runner = TabularCellRunner(
        loader_factory=_Loader,
        adapter_factory=lambda *args: _FakeAdapter(*args, corrupt=True),
        clock=lambda: datetime(2026, 7, 19, tzinfo=timezone.utc),
        approved_root=tmp_path,
    )
    with pytest.raises(TabularRunFailure) as caught:
        runner.run(job)
    assert caught.value.state is ResultState.EVAL_FAIL
    assert caught.value.reason_code == "INVALID_PREDICTION_FRAME"
    assert not Path(job.output_dir).exists()


def test_cpu_queue_is_bounded_serial_and_rejects_duplicate_runs(tmp_path: Path) -> None:
    job, _ = _write_fixture(tmp_path)
    second = replace(
        job,
        run_id="v1-a1-csi300-ridge-seed-20260719",
        phase="V1",
        information_set=InformationSet.A1,
        output_dir=(tmp_path / "runs/v1-a1-csi300-ridge-seed-20260719").resolve().as_posix(),
    )
    queue = TabularQueueManifest("cpu-test", (job, second), max_jobs=2)
    calls: list[str] = []

    class _Recorder:
        def run(self, item: TabularJobSpec) -> Path:
            calls.append(item.run_id)
            return Path(item.output_dir)

    assert run_cpu_queue(queue, runner=_Recorder()) == tuple(
        Path(item.output_dir) for item in queue.jobs
    )
    assert calls == [job.run_id, second.run_id]
    with pytest.raises(ContractError, match="duplicate run_ids"):
        replace(queue, jobs=(job, job)).validate()
