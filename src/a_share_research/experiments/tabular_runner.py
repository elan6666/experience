"""Receipt-bound V0/V1 orchestration for the two CPU tabular models.

The runner deliberately owns no estimator implementation.  It joins the
canonical D0 loader to the already reviewed Ridge/LightGBM adapters, preserves
complete dynamic-membership coverage, and writes common PredictionFrame and
RunManifest artifacts.  Portfolio construction and metric selection remain
downstream concerns.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import ClassVar, NoReturn, Protocol

from a_share_research.contracts import (
    AssetRegistry,
    CanonicalModel,
    ContractError,
    FormalFeatureManifest,
    RunManifest,
    canonical_hash,
)
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.data.manifest import D0Manifest
from a_share_research.experiments.source_evidence import verify_source_manifest
from a_share_research.models.tabular import (
    FeatureGate,
    InformationSet,
    LightGBMAdapter,
    LightGBMConfig,
    RidgeAdapter,
    RidgeConfig,
    TabularModelResult,
    TabularSample,
    complete_run_manifest,
    default_feature_layout,
)
from a_share_research.protocol import Partition, ProtocolSpec, Purpose, UniverseClass
from a_share_research.quality.states import ResultState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAIN_START = date(2019, 1, 1)
_TRAIN_END = date(2024, 12, 31)
_VALIDATION_START = date(2025, 1, 1)
_VALIDATION_END = date(2025, 12, 31)
_TABULAR_MODELS = {"ridge", "lightgbm"}
_UPSTREAM = {
    "ridge": "internal:scikit-learn-1.9.0",
    "lightgbm": "internal:lightgbm-4.7.0",
}
_SERVER_ROOT = Path("/data/yilangliu/a_share_research")


class TabularRunFailure(RuntimeError):
    """Typed, non-rankable failure emitted at an orchestration boundary."""

    def __init__(
        self,
        *,
        run_id: str,
        state: ResultState,
        stage: str,
        reason_code: str,
        detail: str,
    ) -> None:
        if state not in {
            ResultState.INVALID_DATA,
            ResultState.INVALID_PROTOCOL,
            ResultState.ADAPTER_FAIL,
            ResultState.TRAIN_FAIL,
            ResultState.EVAL_FAIL,
        }:
            raise ContractError("tabular run failure must use a typed failure state")
        if not run_id or not stage or not re.fullmatch(r"[A-Z][A-Z0-9_]*", reason_code):
            raise ContractError("tabular run failure identity is invalid")
        self.run_id = run_id
        self.state = state
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{run_id}: {state.value}/{reason_code} at {stage}: {detail}")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": "tabular_run_failure_v1",
            "run_id": self.run_id,
            "state": self.state.value,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class EvidenceFile(CanonicalModel):
    """An exact server-side file admitted into one run."""

    SCHEMA_NAME: ClassVar[str] = "tabular_evidence_file"

    path: str
    sha256: str

    def validate(self) -> None:
        if not self.path or not Path(self.path).is_absolute():
            raise ContractError("evidence file path must be absolute")
        if not _SHA256.fullmatch(self.sha256):
            raise ContractError("evidence file hash must be SHA-256")

    def verify(self) -> Path:
        path = Path(self.path)
        if not path.is_file():
            raise ContractError(f"evidence file is absent: {path}")
        if _sha256(path) != self.sha256:
            raise ContractError(f"evidence file hash mismatch: {path}")
        return path


@dataclass(frozen=True)
class TabularJobSpec(CanonicalModel):
    """One isolated V0/A0 or V1/A1-A3 CPU attempt."""

    SCHEMA_NAME: ClassVar[str] = "tabular_job_spec"

    phase: str
    run_id: str
    model: str
    universe: UniverseClass
    information_set: InformationSet
    seed: int
    canonical_root: str
    output_dir: str
    d0_manifest: EvidenceFile
    environment_receipt: EvidenceFile
    code_receipt: EvidenceFile
    model_config: EvidenceFile
    layout_config: EvidenceFile
    asset_registry_hash: str
    cell_config_hash: str
    upstream_commit: str
    formal_feature_manifest_hash: str | None = None
    formal_feature_receipt: EvidenceFile | None = None

    def validate(self) -> None:
        if self.phase not in {"V0", "V1"}:
            raise ContractError("tabular phase must be V0 or V1")
        if self.model not in _TABULAR_MODELS:
            raise ContractError("tabular runner accepts only Ridge or LightGBM")
        if not isinstance(self.universe, UniverseClass):
            raise ContractError("tabular universe must use UniverseClass")
        if not isinstance(self.information_set, InformationSet):
            raise ContractError("information_set must use InformationSet")
        if self.phase == "V0" and self.information_set is not InformationSet.A0:
            raise ContractError("V0 tabular jobs are A0 only")
        if self.phase == "V1" and self.information_set is InformationSet.A0:
            raise ContractError("V1 A0 is reference-only and cannot be retrained")
        if type(self.seed) is not int or self.seed != 20260719:
            raise ContractError("tabular seed must inherit the frozen V0 seed")
        expected_run_id = (
            f"{self.phase.lower()}-{self.information_set.value.lower()}-"
            f"{self.universe.value.lower()}-{self.model}-seed-{self.seed}"
        )
        if self.run_id != expected_run_id:
            raise ContractError("tabular run_id is not canonical and isolated")
        if self.upstream_commit != _UPSTREAM[self.model]:
            raise ContractError("tabular upstream/package pin differs from the frozen registry")
        if not Path(self.canonical_root).is_absolute():
            raise ContractError("canonical_root must be absolute")
        output = Path(self.output_dir)
        if not output.is_absolute() or output.name != self.run_id:
            raise ContractError("output_dir must be absolute and end in the exact run_id")
        for name in ("asset_registry_hash", "cell_config_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        formal = self.universe in {UniverseClass.CSI300, UniverseClass.STAR50}
        if formal and not (
            self.formal_feature_manifest_hash
            and _SHA256.fullmatch(self.formal_feature_manifest_hash)
        ):
            raise ContractError("formal universe job requires its D0 eligibility receipt hash")
        if formal and self.formal_feature_receipt is None:
            raise ContractError("formal universe job requires its exact eligibility receipt")
        if not formal and self.formal_feature_receipt is not None:
            raise ContractError("exploratory universe cannot claim a formal receipt")
        if self.formal_feature_manifest_hash is not None and not _SHA256.fullmatch(
            self.formal_feature_manifest_hash
        ):
            raise ContractError("formal_feature_manifest_hash must be SHA-256")


def tabular_cell_config_hash(job: TabularJobSpec, d0: D0Manifest) -> str:
    """Rebuild the generator-owned cell identity at execution time."""
    evidence: dict[str, EvidenceFile] = {
        "d0_manifest": job.d0_manifest,
        "environment_receipt": job.environment_receipt,
        "code_receipt": job.code_receipt,
        "model_config": job.model_config,
        "layout_config": job.layout_config,
    }
    if job.formal_feature_receipt is not None:
        evidence["formal_feature_receipt"] = job.formal_feature_receipt
    return canonical_hash(
        {
            "schema_version": "tabular_cell_config_v1",
            "phase": job.phase,
            "model": job.model,
            "universe": job.universe.value,
            "scope": (
                "EXPLORATORY_ONLY"
                if job.universe in {UniverseClass.TECH32, UniverseClass.TECH90}
                else "FORMAL"
            ),
            "information_set": job.information_set.value,
            "protocol": {
                "frequency": "WEEKLY",
                "horizon": 5,
                "label": "future_5d_open_to_open_excess_return",
                "entry": "T+1_OPEN",
                "train": ["2019-01-01", "2024-12-31"],
                "validation": ["2025-01-01", "2025-12-31"],
                "legacy_2026_selection_allowed": False,
            },
            "seed": job.seed,
            "upstream_commit": job.upstream_commit,
            "d0_content_hash": d0.content_hash,
            "asset_registry_hash": job.asset_registry_hash,
            "formal_feature_manifest_hash": job.formal_feature_manifest_hash,
            "evidence_files": {
                name: value.to_dict() for name, value in sorted(evidence.items())
            },
        }
    )


@dataclass(frozen=True)
class TabularQueueManifest(CanonicalModel):
    """A bounded, serial CPU queue; it cannot discover or synthesize jobs."""

    SCHEMA_NAME: ClassVar[str] = "tabular_cpu_queue"

    queue_id: str
    jobs: tuple[TabularJobSpec, ...]
    max_jobs: int

    def validate(self) -> None:
        if not self.queue_id:
            raise ContractError("queue_id is required")
        if type(self.max_jobs) is not int or not 1 <= self.max_jobs <= 16:
            raise ContractError("CPU queue max_jobs must be in [1, 16]")
        if not self.jobs or len(self.jobs) > self.max_jobs:
            raise ContractError("CPU queue is empty or exceeds its declared bound")
        if len({job.run_id for job in self.jobs}) != len(self.jobs):
            raise ContractError("CPU queue contains duplicate run_ids")


class _Adapter(Protocol):
    def fit_predict(self, **kwargs: object) -> TabularModelResult: ...


AdapterFactory = Callable[[TabularJobSpec, object, FeatureGate, Mapping[str, object]], _Adapter]
LoaderFactory = Callable[[Path, str], CanonicalDatasetLoader]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class PreparedTabularJob:
    job: TabularJobSpec
    d0: D0Manifest
    universe_table_hashes: tuple[tuple[str, str], ...]
    universe_bundle_hash: str
    model_payload: Mapping[str, object]
    training: tuple[TabularSample, ...]
    validation: tuple[TabularSample, ...]
    prediction: tuple[TabularSample, ...]
    fold_id: str


def _default_adapter_factory(
    job: TabularJobSpec,
    layout: object,
    gate: FeatureGate,
    payload: Mapping[str, object],
) -> _Adapter:
    if job.model == "ridge":
        return RidgeAdapter(  # type: ignore[arg-type]
            layout,
            gate,
            model_config=RidgeConfig.from_mapping(payload),
        )
    return LightGBMAdapter(  # type: ignore[arg-type]
        layout,
        gate,
        model_config=LightGBMConfig.from_mapping(payload),
    )


def _load_layout(path: Path):
    payload = _read_json(path)
    layout = default_feature_layout()
    groups = payload.get("ordered_groups")
    if not isinstance(groups, dict):
        raise ContractError("tabular layout config lacks ordered_groups")
    expected = {
        "Core": list(layout.core),
        "F": list(layout.fundamental),
        "F_missing": list(layout.fundamental_missing),
        "S": list(layout.market_state),
    }
    if groups != expected:
        raise ContractError("checked-in tabular layout differs from the code-level frozen layout")
    expected_gates = {
        "A0": {"Core": 1, "F": 0, "F_missing": 0, "S": 0},
        "A1": {"Core": 1, "F": 1, "F_missing": 1, "S": 0},
        "A2": {"Core": 1, "F": 0, "F_missing": 0, "S": 1},
        "A3": {"Core": 1, "F": 1, "F_missing": 1, "S": 1},
    }
    if payload.get("gates") != expected_gates:
        raise ContractError("checked-in A0-A3 information gates have drifted")
    return layout


def _information_coverage(
    samples: Sequence[TabularSample],
    *,
    gate: FeatureGate,
    state_names: tuple[str, ...],
) -> tuple[TabularSample, ...]:
    """Apply only gate-required coverage; F values remain imputable with masks."""
    if not gate.s_enabled:
        return tuple(samples)
    adjusted = []
    for sample in samples:
        missing_state = sample.member and (
            not sample.values
            or any(sample.missing_flags.get(name, True) for name in state_names)
        )
        adjusted.append(
            replace(sample, complete_history=False) if missing_state else sample
        )
    return tuple(adjusted)


def _validation_registry_hash(samples: Sequence[TabularSample]) -> str:
    """Verify the complete panel is an append-only causal registry and hash it."""
    if not samples:
        raise ContractError("validation panel cannot be empty")
    registry_order: list[str] = []
    offset = 0
    previous_date: date | None = None
    while offset < len(samples):
        signal_date = samples[offset].signal_date
        if previous_date is not None and signal_date <= previous_date:
            raise ContractError("validation panel dates must be strictly increasing")
        date_codes: list[str] = []
        while offset < len(samples) and samples[offset].signal_date == signal_date:
            date_codes.append(samples[offset].ts_code)
            offset += 1
        if len(date_codes) != len(set(date_codes)):
            raise ContractError("validation panel contains a duplicate date/asset identity")
        known = set(registry_order)
        registry_order.extend(code for code in date_codes if code not in known)
        if tuple(date_codes) != tuple(registry_order):
            raise ContractError(
                "validation panel asset registry is not append-only and causally ordered"
            )
        previous_date = signal_date
    return AssetRegistry(tuple(registry_order)).stable_hash()


def _assert_prediction_coverage(
    expected: Sequence[TabularSample],
    result: TabularModelResult,
) -> None:
    """Forbid an adapter from dropping, replacing or reclassifying panel rows."""
    expected_rows = tuple(
        (row.signal_date, row.ts_code, row.coverage_state) for row in expected
    )
    actual_rows = tuple(
        (row.signal_date, row.ts_code, row.coverage_state)
        for row in result.predictions.records
    )
    if actual_rows != expected_rows:
        raise ContractError(
            "PredictionFrame does not preserve the complete canonical panel and coverage masks"
        )


class TabularCellRunner:
    """Execute one sealed 2019-2024 -> 2025 model-selection cell."""

    def __init__(
        self,
        *,
        loader_factory: LoaderFactory = CanonicalDatasetLoader,
        adapter_factory: AdapterFactory = _default_adapter_factory,
        clock: Clock = lambda: datetime.now(timezone.utc),
        approved_root: Path = _SERVER_ROOT,
    ) -> None:
        self.loader_factory = loader_factory
        self.adapter_factory = adapter_factory
        self.clock = clock
        self.approved_root = approved_root.resolve()

    def _within_root(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.approved_root and self.approved_root not in resolved.parents:
            raise ContractError(f"tabular runtime path leaves approved root: {resolved}")
        return resolved

    def prepare(self, job: TabularJobSpec) -> PreparedTabularJob:
        job.validate()
        d0_path = job.d0_manifest.verify()
        environment_path = job.environment_receipt.verify()
        code_path = job.code_receipt.verify()
        verify_source_manifest(code_path, Path(__file__).resolve().parents[3])
        model_config_path = job.model_config.verify()
        layout_config_path = job.layout_config.verify()
        formal_path = (
            job.formal_feature_receipt.verify()
            if job.formal_feature_receipt is not None
            else None
        )
        for path in (
            d0_path,
            environment_path,
            code_path,
            model_config_path,
            layout_config_path,
            Path(job.canonical_root),
            Path(job.output_dir),
            *( (formal_path,) if formal_path is not None else () ),
        ):
            self._within_root(path)
        if len({environment_path, code_path, model_config_path, layout_config_path}) != 4:
            raise ContractError("tabular evidence roles must reference distinct files")

        d0 = D0Manifest.from_dict(dict(_read_json(d0_path)))
        if tabular_cell_config_hash(job, d0) != job.cell_config_hash:
            raise ContractError("tabular cell_config_hash does not match the job payload")
        environment = _read_json(environment_path)
        if environment.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
            raise ContractError("tabular environment receipt is not passing")
        if str(environment.get("model", "")).lower() != job.model:
            raise ContractError("tabular environment receipt names a different model")
        if d0.cutoff_date < _VALIDATION_END:
            raise ContractError("D0 does not cover the complete 2025 validation window")
        gate_by_universe = {item.universe: item for item in d0.universe_gates}
        universe_gate = gate_by_universe[job.universe]
        if universe_gate.status not in {
            ResultState.PASS,
            ResultState.PASS_WITH_WARNING,
            ResultState.EXPLORATORY_ONLY,
        }:
            raise ContractError(
                f"D0 universe gate is not runnable: {universe_gate.status.value}"
            )

        if formal_path is not None:
            formal = FormalFeatureManifest.from_dict(dict(_read_json(formal_path)))
            expected_id = (
                f"{d0.dataset_id}:{job.universe.value}:{job.information_set.value}"
            )
            if formal.dataset_id != expected_id or formal.d0_manifest_hash != d0.content_hash:
                raise ContractError("formal feature receipt names a different D0 cell")
            layout_for_receipt = default_feature_layout()
            expected_names = list(layout_for_receipt.core)
            if job.information_set.enables_f:
                expected_names.extend(layout_for_receipt.fundamental)
                expected_names.extend(layout_for_receipt.fundamental_missing)
            if job.information_set.enables_s:
                expected_names.extend(layout_for_receipt.market_state)
            if set(formal.feature_eligibility) != set(expected_names):
                raise ContractError("formal feature receipt has the wrong input set")
            if formal.require_formal_eligible() != job.formal_feature_manifest_hash:
                raise ContractError("formal feature receipt hash differs from the job")

        canonical_root = Path(job.canonical_root)
        relative_paths = (
            f"{job.universe.value.lower()}/membership.jsonl",
            f"{job.universe.value.lower()}/features.jsonl",
            f"{job.universe.value.lower()}/labels.jsonl",
            f"{job.universe.value.lower()}/masks.jsonl",
            "shared_market_state.jsonl",
        )
        table_hashes: list[tuple[str, str]] = []
        for relative in relative_paths:
            expected = d0.canonical_table_hashes.get(relative)
            path = canonical_root / relative
            if expected is None or not path.is_file() or _sha256(path) != expected:
                raise ContractError(f"canonical D0 table is absent or hash-mismatched: {relative}")
            table_hashes.append((relative, expected))

        layout = _load_layout(layout_config_path)
        model_payload = _read_json(model_config_path)
        model_config = (
            RidgeConfig.from_mapping(model_payload)
            if job.model == "ridge"
            else LightGBMConfig.from_mapping(model_payload)
        )
        if model_config.seed != job.seed:
            raise ContractError("model config seed and isolated job seed disagree")
        gate = FeatureGate(job.information_set)
        loader = self.loader_factory(canonical_root, job.universe.value)
        admissible_training_labels: set[tuple[date, str]] = set()
        admissible_validation_labels: set[tuple[date, str]] = set()
        for label in loader.iter_labels():
            if label.horizon != 5:
                continue
            key = (label.signal_date, label.ts_code)
            if _TRAIN_START <= label.signal_date <= _TRAIN_END:
                if label.exit_date < _VALIDATION_START:
                    admissible_training_labels.add(key)
            elif _VALIDATION_START <= label.signal_date <= _VALIDATION_END:
                if label.exit_date <= _VALIDATION_END:
                    admissible_validation_labels.add(key)
        samples = tuple(
            loader.iter_tabular_samples(
                horizon=5,
                relative_target=True,
                start=_TRAIN_START,
                end=_VALIDATION_END,
                complete_panel=True,
            )
        )
        samples = _information_coverage(
            samples,
            gate=gate,
            state_names=layout.market_state,
        )
        if not samples or any(sample.signal_date > _VALIDATION_END for sample in samples):
            raise ContractError("tabular runner admitted no rows or rows after 2025 validation")

        protocol = ProtocolSpec.research_v1()
        training = tuple(
            sample
            for sample in samples
            if _TRAIN_START <= sample.signal_date <= _TRAIN_END
            and (sample.signal_date, sample.ts_code) in admissible_training_labels
            and sample.coverage_state.value == "SCORED"
            and sample.target is not None
        )
        validation = tuple(
            sample
            for sample in samples
            if _VALIDATION_START <= sample.signal_date <= _VALIDATION_END
            and (sample.signal_date, sample.ts_code) in admissible_validation_labels
            and sample.coverage_state.value == "SCORED"
            and sample.target is not None
        )
        prediction = tuple(
            sample
            for sample in samples
            if _VALIDATION_START <= sample.signal_date <= _VALIDATION_END
        )
        if not training or not validation or not prediction:
            raise ContractError("sealed tabular train/validation fold is empty")
        for sample in training:
            protocol.assert_access(sample.signal_date, Purpose.FIT)
        for sample in prediction:
            protocol.assert_access(sample.signal_date, Purpose.SELECT)
        if len({(row.signal_date, row.ts_code) for row in prediction}) != len(prediction):
            raise ContractError("complete validation coverage contains duplicate identities")
        actual_asset_registry_hash = _validation_registry_hash(prediction)
        if actual_asset_registry_hash != job.asset_registry_hash:
            raise ContractError(
                "job asset_registry_hash does not match the causal 2025 validation panel"
            )

        universe_table_hashes = tuple(table_hashes)
        return PreparedTabularJob(
            job=job,
            d0=d0,
            universe_table_hashes=universe_table_hashes,
            universe_bundle_hash=canonical_hash(universe_table_hashes),
            model_payload=model_payload,
            training=training,
            validation=validation,
            prediction=prediction,
            fold_id=f"weekly-future5d-{job.universe.value.lower()}-train2019-2024-val2025",
        )

    def _raise_failure(
        self,
        job: TabularJobSpec,
        *,
        state: ResultState,
        stage: str,
        reason_code: str,
        error: Exception,
        record: bool = True,
    ) -> NoReturn:
        failure = TabularRunFailure(
            run_id=job.run_id or "INVALID_RUN_ID",
            state=state,
            stage=stage,
            reason_code=reason_code,
            detail=str(error),
        )
        output_dir = Path(job.output_dir)
        failure_path = output_dir.parent / f"{output_dir.name}.failure.json"
        if (
            record
            and output_dir.is_absolute()
            and not output_dir.exists()
            and not failure_path.exists()
        ):
            payload = {
                **failure.to_dict(),
                "evidence": {
                    "d0_manifest_file_hash": job.d0_manifest.sha256,
                    "environment_receipt_hash": job.environment_receipt.sha256,
                    "code_receipt_hash": job.code_receipt.sha256,
                    "model_config_file_hash": job.model_config.sha256,
                    "layout_config_file_hash": job.layout_config.sha256,
                    "cell_config_hash": job.cell_config_hash,
                },
            }
            _atomic_json(failure_path, payload)
        raise failure from error

    def run(self, job: TabularJobSpec) -> Path:
        try:
            job.validate()
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.INVALID_PROTOCOL,
                stage="JOB_SPEC",
                reason_code="INVALID_JOB_SPEC",
                error=error,
                record=False,
            )
        try:
            prepared = self.prepare(job)
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.INVALID_DATA,
                stage="PREPARE_D0",
                reason_code="D0_OR_EVIDENCE_REJECTED",
                error=error,
            )
        output_dir = Path(job.output_dir)
        if output_dir.exists():
            self._raise_failure(
                job,
                state=ResultState.INVALID_PROTOCOL,
                stage="OUTPUT_RESERVATION",
                reason_code="OUTPUT_ALREADY_EXISTS",
                error=ContractError(
                    "isolated output directory already exists; never overwrite a run"
                ),
                record=False,
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.with_name(output_dir.name + ".tmp")
        if temporary.exists():
            self._raise_failure(
                job,
                state=ResultState.INVALID_PROTOCOL,
                stage="OUTPUT_RESERVATION",
                reason_code="STALE_TEMPORARY_OUTPUT",
                error=ContractError("stale temporary output exists; inspect before retrying"),
            )
        temporary.mkdir()
        started_at = self.clock()
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            self._raise_failure(
                job,
                state=ResultState.INVALID_PROTOCOL,
                stage="RUN_CLOCK",
                reason_code="NAIVE_RUN_CLOCK",
                error=ContractError("runner clock must be timezone-aware"),
            )
        layout = _load_layout(job.layout_config.verify())
        gate = FeatureGate(job.information_set)
        try:
            adapter = self.adapter_factory(job, layout, gate, prepared.model_payload)
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.ADAPTER_FAIL,
                stage="ADAPTER_CONSTRUCTION",
                reason_code="ADAPTER_CONSTRUCTION_FAILED",
                error=error,
            )
        common = {
            "run_id": job.run_id,
            "training": prepared.training,
            "prediction": prepared.prediction,
            "fit_end": _TRAIN_END,
            "fit_data_hash": prepared.d0.content_hash,
            "fold_id": prepared.fold_id,
        }
        try:
            if job.model == "ridge":
                result = adapter.fit_predict(**common)
            else:
                result = adapter.fit_predict(
                    **common,
                    validation=prepared.validation,
                    validation_end=_VALIDATION_END,
                )
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.TRAIN_FAIL,
                stage="FIT_PREDICT",
                reason_code="FIT_OR_PREDICT_FAILED",
                error=error,
            )
        try:
            result.predictions.validate()
            result.diagnostics.validate()
            _assert_prediction_coverage(prepared.prediction, result)
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.EVAL_FAIL,
                stage="PREDICTION_CONTRACT",
                reason_code="INVALID_PREDICTION_FRAME",
                error=error,
            )

        universe_gate = next(
            item for item in prepared.d0.universe_gates if item.universe is job.universe
        )
        status = universe_gate.status
        if job.universe in {UniverseClass.TECH32, UniverseClass.TECH90}:
            status = ResultState.EXPLORATORY_ONLY
        draft = RunManifest(
            run_id=job.run_id,
            model=result.diagnostics.model,
            universe=job.universe,
            information_set=job.information_set.value,
            split=Partition.VALIDATION,
            purpose=Purpose.SELECT,
            data_hash=prepared.d0.content_hash,
            asset_registry_hash=job.asset_registry_hash,
            execution_calendar_manifest_hash=prepared.d0.trading_calendar_hash,
            feature_schema_hash=layout.stable_hash(),
            market_state_hash=prepared.d0.market_state_hash,
            config_hash=result.diagnostics.config_hash,
            code_hash=job.code_receipt.sha256,
            upstream_commit=job.upstream_commit,
            seed=job.seed,
            status=status,
            started_at=started_at,
            completed_at=None,
            formal_feature_manifest_hash=job.formal_feature_manifest_hash,
            deviations=(
                "Tabular adapter around package-native estimator; no estimator/loss rewrite.",
            ),
        )
        completed_at = self.clock()
        try:
            manifest = complete_run_manifest(
                draft,
                result,
                status=status,
                completed_at=completed_at,
            )
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.EVAL_FAIL,
                stage="RUN_MANIFEST",
                reason_code="RUN_MANIFEST_REJECTED",
                error=error,
            )
        preprocessing_state = getattr(adapter, "preprocessor", None)
        preprocessing_state = getattr(preprocessing_state, "state", None)
        if preprocessing_state is None:
            self._raise_failure(
                job,
                state=ResultState.EVAL_FAIL,
                stage="PREPROCESSING_RECEIPT",
                reason_code="MISSING_PREPROCESSING_RECEIPT",
                error=ContractError(
                    "tabular adapter did not expose its fitted preprocessing receipt"
                ),
            )

        predictions_path = temporary / "predictions.json"
        diagnostics_path = temporary / "diagnostics.json"
        preprocessing_path = temporary / "preprocessing_state.json"
        manifest_path = temporary / "run_manifest.json"
        try:
            _atomic_json(predictions_path, result.predictions)
            _atomic_json(diagnostics_path, result.diagnostics)
            _atomic_json(preprocessing_path, preprocessing_state)
            _atomic_json(manifest_path, manifest)
            output_hashes = {
                path.name: _sha256(path)
                for path in (
                    predictions_path,
                    diagnostics_path,
                    preprocessing_path,
                    manifest_path,
                )
            }
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.EVAL_FAIL,
                stage="ATOMIC_OUTPUT",
                reason_code="OUTPUT_WRITE_FAILED",
                error=error,
            )
        coverage = Counter(
            row.coverage_state.value for row in result.predictions.records
        )
        receipt = {
            "schema_version": "tabular_run_receipt_v1",
            "run_id": job.run_id,
            "phase": job.phase,
            "model": job.model,
            "universe": job.universe.value,
            "information_set": job.information_set.value,
            "protocol": {
                "frequency": "WEEKLY",
                "horizon": 5,
                "label": "future_5d_open_to_open_excess_return",
                "entry": "T+1_OPEN",
                "train": [_TRAIN_START.isoformat(), _TRAIN_END.isoformat()],
                "validation": [
                    _VALIDATION_START.isoformat(),
                    _VALIDATION_END.isoformat(),
                ],
                "max_admitted_date": max(
                    row.signal_date for row in prepared.prediction
                ).isoformat(),
                "legacy_2026_selection_allowed": False,
            },
            "evidence": {
                "d0_manifest_file_hash": job.d0_manifest.sha256,
                "d0_content_hash": prepared.d0.content_hash,
                "d0_feature_schema_hash": prepared.d0.feature_schema_hash,
                "universe_table_hashes": dict(prepared.universe_table_hashes),
                "universe_bundle_hash": prepared.universe_bundle_hash,
                "asset_registry_hash": job.asset_registry_hash,
                "market_state_hash": prepared.d0.market_state_hash,
                "environment_receipt_hash": job.environment_receipt.sha256,
                "code_receipt_hash": job.code_receipt.sha256,
                "model_config_file_hash": job.model_config.sha256,
                "model_config_hash": result.diagnostics.config_hash,
                "cell_config_hash": job.cell_config_hash,
                "layout_config_file_hash": job.layout_config.sha256,
                "layout_hash": layout.stable_hash(),
                "gate_hash": gate.stable_hash(),
                "upstream_commit": job.upstream_commit,
                "fold_id": prepared.fold_id,
                "preprocessing_hash": result.diagnostics.preprocessing_hash,
            },
            "counts": {
                "training_scored": len(prepared.training),
                "validation_for_selection": len(prepared.validation),
                "prediction_complete_panel": len(prepared.prediction),
                "prediction_coverage": dict(sorted(coverage.items())),
            },
            "output_hashes": output_hashes,
        }
        try:
            _atomic_json(temporary / "run_receipt.json", receipt)
            temporary.replace(output_dir)
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.EVAL_FAIL,
                stage="ATOMIC_PUBLISH",
                reason_code="OUTPUT_PUBLISH_FAILED",
                error=error,
            )
        return output_dir


def run_cpu_queue(
    queue: TabularQueueManifest,
    *,
    runner: TabularCellRunner | None = None,
) -> tuple[Path, ...]:
    """Run the exact declared jobs serially; no hidden retry or job discovery."""
    queue.validate()
    cell_runner = runner or TabularCellRunner()
    return tuple(cell_runner.run(job) for job in queue.jobs)
