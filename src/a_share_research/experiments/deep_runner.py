"""Receipt-bound iTransformer/FACT V0/V1 server runner.

The module deliberately imports neither PyTorch nor author code at import time.
Those imports occur only after D0, provenance, GPU-isolation and output-path
gates pass on the approved server.  Author repositories remain detached,
clean, and bytecode-free; all A-share adaptation lives in this project.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, NoReturn

from a_share_research.adapters.common import (
    AdapterContractError,
    CausalAssetMaster,
    DeepRuntimePolicy,
    FeaturePackingSchema,
    InformationGate,
    PredictionBatch,
    RunIsolation,
    UpstreamBinding,
    build_causal_asset_master,
    export_prediction_batches,
    pack_feature_window,
)
from a_share_research.adapters.fact import FactAdapter
from a_share_research.adapters.itransformer import ITransformerAdapter
from a_share_research.adapters.s4m import S4MAdapter
from a_share_research.adapters.timepro import TimeProAdapter
from a_share_research.adapters.timexer import TimeXerAdapter
from a_share_research.contracts import (
    AssetRegistry,
    CanonicalModel,
    ContractError,
    FormalFeatureManifest,
    MaskBundle,
    PredictionFrame,
    RunManifest,
    UniverseMembership,
    canonical_hash,
)
from a_share_research.data.formal_receipts import information_inputs
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.data.manifest import D0Manifest
from a_share_research.experiments.source_evidence import verify_source_manifest
from a_share_research.models.tabular.layout import InformationSet
from a_share_research.protocol import Partition, ProtocolSpec, Purpose, UniverseClass
from a_share_research.quality.states import ResultState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SERVER_ROOT = Path("/data/yilangliu/a_share_research")
_TRAIN_START = date(2019, 1, 1)
_TRAIN_END = date(2024, 12, 31)
_VALIDATION_START = date(2025, 1, 1)
_VALIDATION_END = date(2025, 12, 31)
_EXPECTED_COMMITS = {
    "itransformer": "c2426e68ca13f74aaec08045c5c724d8ad328124",
    "fact": "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4",
    "timexer": "76011909357972bd55a27adba2e1be994d81b327",
    "timepro": "70a20e5a257b30eb026ee4316293cf4feeb92a1f",
    "s4m": "a718823addd3606e763dfc261174e0135b2535f4",
}
_PHYSICAL_GPUS = {"itransformer": 0, "fact": 1, "timexer": 0, "timepro": 1, "s4m": 0}
_TECH_UNIVERSES = {UniverseClass.TECH32, UniverseClass.TECH100}
_FORBIDDEN_AUTHOR_ARGUMENTS = {
    "seq_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
    "learning_rate",
    "train_epochs",
    "patience",
    "batch_size",
}


class DeepRunFailure(RuntimeError):
    """Typed, non-rankable failure at one deep-cell boundary."""

    def __init__(
        self,
        *,
        run_id: str,
        state: ResultState,
        stage: str,
        reason_code: str,
        detail: str,
    ) -> None:
        allowed = {
            ResultState.INVALID_DATA,
            ResultState.INVALID_PROTOCOL,
            ResultState.ADAPTER_FAIL,
            ResultState.TRAIN_FAIL,
            ResultState.EVAL_FAIL,
        }
        if state not in allowed:
            raise ContractError("deep run failure must use a typed failure state")
        if not run_id or not stage or re.fullmatch(r"[A-Z][A-Z0-9_]*", reason_code) is None:
            raise ContractError("deep run failure identity is invalid")
        self.run_id = run_id
        self.state = state
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{run_id}: {state.value}/{reason_code} at {stage}: {detail}")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": "deep_run_failure_v1",
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _under_server_root(path: Path) -> bool:
    try:
        path.relative_to(_SERVER_ROOT)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class DeepEvidenceFile(CanonicalModel):
    """Exact server file admitted into a deep run."""

    SCHEMA_NAME: ClassVar[str] = "deep_evidence_file"

    path: str
    sha256: str

    def validate(self) -> None:
        path = Path(self.path)
        if not path.is_absolute() or not _under_server_root(path):
            raise ContractError("deep evidence must be an absolute server-root path")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ContractError("deep evidence hash must be SHA-256")

    def verify(self) -> Path:
        path = Path(self.path)
        if not path.is_file():
            raise ContractError(f"deep evidence file is absent: {path}")
        if _sha256(path) != self.sha256:
            raise ContractError(f"deep evidence hash mismatch: {path}")
        return path


@dataclass(frozen=True)
class DeepHyperparameters(CanonicalModel):
    """Hashable runtime parameters; A0-A3 may change information, not capacity."""

    SCHEMA_NAME: ClassVar[str] = "deep_hyperparameters"

    lookback_weeks: int
    forecast_steps: int
    batch_size: int
    maximum_epochs: int
    patience: int
    learning_rate: float
    author_arguments: dict[str, object]

    def validate(self) -> None:
        if self.lookback_weeks != 96 or self.forecast_steps != 1:
            raise ContractError(
                "deep protocol is frozen to 96 weekly inputs and one future5d score"
            )
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 32:
            raise ContractError("deep batch_size must be in [1, 32]")
        if type(self.maximum_epochs) is not int or not 1 <= self.maximum_epochs <= 200:
            raise ContractError("deep maximum_epochs must be in [1, 200]")
        if type(self.patience) is not int or not 1 <= self.patience <= self.maximum_epochs:
            raise ContractError("deep patience must be positive and no larger than epochs")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ContractError("deep learning_rate must be finite and positive")
        if not self.author_arguments:
            raise ContractError("explicit author architecture arguments are required")
        forbidden = set(self.author_arguments) & _FORBIDDEN_AUTHOR_ARGUMENTS
        if forbidden:
            raise ContractError(
                "runner-owned author arguments cannot be overridden: "
                f"{sorted(forbidden)}"
            )
        canonical_hash(self.author_arguments)

    @property
    def capacity_hash(self) -> str:
        """Information-gate-independent architecture/training identity."""
        return canonical_hash(
            {
                "lookback_weeks": self.lookback_weeks,
                "forecast_steps": self.forecast_steps,
                "batch_size": self.batch_size,
                "maximum_epochs": self.maximum_epochs,
                "patience": self.patience,
                "learning_rate": self.learning_rate,
                "author_arguments": self.author_arguments,
                "projector": "shared_linear_per_asset_v1",
            }
        )


@dataclass(frozen=True)
class DeepJobSpec(CanonicalModel):
    """One isolated model/universe/gate/seed attempt."""

    SCHEMA_NAME: ClassVar[str] = "deep_job_spec"

    phase: str
    run_id: str
    model: str
    universe: UniverseClass
    gate: InformationGate
    seed: int
    physical_gpu: int
    canonical_root: str
    upstream_root: str
    output_dir: str
    checkpoint_dir: str
    upstream_commit: str
    asset_registry_hash: str
    cell_config_hash: str
    hyperparameters: DeepHyperparameters
    d0_manifest: DeepEvidenceFile
    environment_receipt: DeepEvidenceFile
    integrity_receipt: DeepEvidenceFile
    code_receipt: DeepEvidenceFile
    adapter_config: DeepEvidenceFile
    common_config: DeepEvidenceFile
    formal_feature_manifest: DeepEvidenceFile | None = None

    def validate(self) -> None:
        if self.phase not in {"V0", "V1"}:
            raise ContractError("deep phase must be V0 or V1")
        if self.model not in _EXPECTED_COMMITS:
            raise ContractError("deep runner accepts only iTransformer, FACT or TimeXer")
        if not isinstance(self.universe, UniverseClass) or not isinstance(
            self.gate, InformationGate
        ):
            raise ContractError("deep job requires typed universe and information gate")
        if self.phase == "V0" and self.gate is not InformationGate.A0:
            raise ContractError("V0 deep jobs are A0 only")
        if self.phase == "V1" and self.gate is InformationGate.A0:
            raise ContractError("V1 A0 is reference-only and cannot be retrained")
        if self.upstream_commit != _EXPECTED_COMMITS[self.model] or _COMMIT.fullmatch(
            self.upstream_commit
        ) is None:
            raise ContractError("deep upstream commit differs from the frozen pin")
        DeepRuntimePolicy().validate_asset_count(1)
        if self.seed not in DeepRuntimePolicy().seeds:
            raise ContractError("deep seed is outside the frozen three-seed policy")
        if self.physical_gpu != _PHYSICAL_GPUS[self.model]:
            raise ContractError("deep model is not assigned to its frozen physical GPU")
        expected = (
            f"{self.phase.lower()}-{self.gate.value.lower()}-"
            f"{self.universe.value.lower()}-{self.model}-seed-{self.seed}"
        )
        if self.run_id != expected:
            raise ContractError("deep run_id is not canonical and isolated")
        for name in ("canonical_root", "upstream_root", "output_dir", "checkpoint_dir"):
            path = Path(getattr(self, name))
            if not path.is_absolute() or not _under_server_root(path):
                raise ContractError(f"{name} must remain below the approved server root")
        isolation = RunIsolation(
            model=self.model,
            universe=self.universe.value.lower(),
            gate=self.gate.value,
            seed=self.seed,
            physical_gpu=self.physical_gpu,
            output_root=Path(self.output_dir),
            checkpoint_root=Path(self.checkpoint_dir),
        )
        if isolation.output_root == isolation.checkpoint_root:
            raise ContractError("deep output and checkpoint paths overlap")
        for name in ("asset_registry_hash", "cell_config_hash"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ContractError(f"{name} must be SHA-256")
        if self.universe in _TECH_UNIVERSES:
            if self.formal_feature_manifest is not None:
                raise ContractError("exploratory technology pools cannot claim a formal receipt")
        elif self.formal_feature_manifest is None:
            raise ContractError("formal deep job requires its feature-eligibility receipt")
        common_arguments = {
            "d_model",
            "d_ff",
            "dropout",
            "freq",
            "lradj",
            "use_norm",
        }
        if self.model == "itransformer":
            model_arguments = {
                "activation",
                "class_strategy",
                "e_layers",
                "embed",
                "factor",
                "n_heads",
                "output_attention",
            }
        elif self.model == "fact":
            model_arguments = {"core", "dilation", "num_kernels", "task_name"}
        elif self.model == "timexer":
            model_arguments = {
                "activation",
                "e_layers",
                "embed",
                "factor",
                "features",
                "n_heads",
                "patch_len",
                "task_name",
            }
        elif self.model == "timepro":
            model_arguments = {"patch_len", "stride", "e_layers"}
        elif self.model == "s4m":
            model_arguments = {
                "e_layers",
                "n_heads",
                "factor",
                "output_attention",
                "mask",
                "classification",
                "plot",
                "num_class",
                "short_len",
                "n",
                "W",
                "en_conv_hidden_size",
                "en_rnn_hidden_sizes",
                "output_keep_prob",
                "input_keep_prob",
                "K",
                "topK",
                "topM",
                "momentum",
                "is_training",
                "memory_size",
                "M",
                "per_mem_size",
                "thres1",
                "thres2",
            }
        else:
            raise ContractError(f"unsupported deep model: {self.model}")

        missing_arguments = (
            common_arguments | model_arguments
        ) - set(self.hyperparameters.author_arguments)
        if missing_arguments:
            raise ContractError(
                f"deep author architecture arguments are incomplete: {sorted(missing_arguments)}"
            )
        if self.model == "fact" and self.hyperparameters.author_arguments.get("core") != 0.5:
            raise ContractError("formal FACT runs require unmodified upstream core=0.5")


def deep_cell_config_hash(
    job: DeepJobSpec,
    d0: D0Manifest,
    formal_feature_manifest_hash: str | None,
) -> str:
    """Rebuild the complete generated cell payload before execution."""
    return canonical_hash(
        {
            "schema_version": "deep_cell_config_v1",
            "phase": job.phase,
            "run_id": job.run_id,
            "model": job.model,
            "universe": job.universe.value,
            "scope": (
                "EXPLORATORY_ONLY"
                if job.universe in _TECH_UNIVERSES
                else "FORMAL"
            ),
            "gate": job.gate.value,
            "seed": job.seed,
            "physical_gpu": job.physical_gpu,
            "upstream_commit": job.upstream_commit,
            "asset_registry_hash": job.asset_registry_hash,
            "formal_feature_manifest_hash": formal_feature_manifest_hash,
            "hyperparameters": job.hyperparameters.to_dict(),
            "evidence": {
                "d0_manifest": job.d0_manifest.to_dict(),
                "environment_receipt": job.environment_receipt.to_dict(),
                "integrity_receipt": job.integrity_receipt.to_dict(),
                "code_receipt": job.code_receipt.to_dict(),
                "adapter_config": job.adapter_config.to_dict(),
                "common_config": job.common_config.to_dict(),
                "formal_feature_manifest": (
                    job.formal_feature_manifest.to_dict()
                    if job.formal_feature_manifest is not None
                    else None
                ),
            },
            "selection_window": ["2025-01-01", "2025-12-31"],
            "legacy_2026_selection_allowed": False,
        }
    )


@dataclass(frozen=True)
class DeepWindowPlan:
    """Pure split evidence used before any tensor/model construction."""

    input_dates: tuple[date, ...]
    train_anchor_indices: tuple[int, ...]
    validation_anchor_indices: tuple[int, ...]
    lookback: int

    @classmethod
    def build(cls, dates: tuple[date, ...], *, lookback: int) -> DeepWindowPlan:
        if dates != tuple(sorted(set(dates))) or not dates:
            raise ContractError("deep signal dates must be unique and increasing")
        if any(day > _VALIDATION_END for day in dates):
            raise ContractError("deep model-selection runner must never admit 2026 rows")
        if type(lookback) is not int or lookback <= 0:
            raise ContractError("deep lookback must be a positive integer")
        train = tuple(
            index
            for index, day in enumerate(dates)
            if index >= lookback and _TRAIN_START <= day <= _TRAIN_END
        )
        validation = tuple(
            index
            for index, day in enumerate(dates)
            if index >= lookback and _VALIDATION_START <= day <= _VALIDATION_END
        )
        expected_validation = tuple(
            index for index, day in enumerate(dates) if _VALIDATION_START <= day <= _VALIDATION_END
        )
        if not train or not validation:
            raise ContractError("sealed deep train/validation fold is empty")
        if validation != expected_validation:
            raise ContractError("D0 lacks enough pre-2025 history for complete validation coverage")
        protocol = ProtocolSpec.research_v1()
        for index in train:
            protocol.assert_access(dates[index], Purpose.FIT)
        for index in validation:
            protocol.assert_access(dates[index], Purpose.SELECT)
        return cls(dates, train, validation, lookback)


@dataclass(frozen=True)
class PreparedDeepJob:
    job: DeepJobSpec
    d0: D0Manifest
    loader: CanonicalDatasetLoader
    model_master: CausalAssetMaster
    evaluation_master: CausalAssetMaster
    model_panel: object
    evaluation_panel: object
    packed: object
    labels: Mapping[tuple[date, str], float]
    plan: DeepWindowPlan
    universe_table_hashes: tuple[tuple[str, str], ...]
    formal_feature_hash: str | None
    technology_selection_deviation: bool


def _membership_intervals(path: Path) -> tuple[UniverseMembership, ...]:
    """Collapse daily membership expansion to its earliest typed interval rows."""
    rows: dict[
        tuple[str, date, date | None, str], UniverseMembership
    ] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = UniverseMembership.from_dict(json.loads(line))
                key = (
                    row.ts_code,
                    row.effective_from,
                    row.effective_to,
                    row.source,
                )
                previous = rows.get(key)
                if previous is None or row.asof_date < previous.asof_date:
                    rows[key] = row
    return tuple(
        sorted(
            rows.values(),
            key=lambda row: (
                row.effective_from,
                row.asof_date,
                row.ts_code,
                row.source,
            ),
        )
    )


def _technology_master(
    memberships: tuple[UniverseMembership, ...], *, universe: UniverseClass
) -> CausalAssetMaster:
    """Preserve the disclosed 2026-selected static list as exploratory evidence."""
    ordered = tuple(
        sorted(
            {row.ts_code for row in memberships},
            key=lambda code: (
                min(row.effective_from for row in memberships if row.ts_code == code),
                code,
            ),
        )
    )
    if not ordered:
        raise ContractError("exploratory technology membership is empty")
    return CausalAssetMaster(
        registry=AssetRegistry(ordered),
        universe=universe.value,
        known_through=_VALIDATION_END,
        source_membership_hash=canonical_hash(tuple(row.to_dict() for row in memberships)),
    )


def _restrict_panel(panel: Any, master: CausalAssetMaster) -> Any:
    """Restrict the evaluation registry to the frozen 2024 model slots."""
    from a_share_research.adapters.common import PanelWindow

    source_slots = {code: index for index, code in enumerate(panel.asset_master.asset_ids)}
    if any(code not in source_slots for code in master.asset_ids):
        raise ContractError("model master contains an identity absent from the D0 panel")
    slots = tuple(source_slots[code] for code in master.asset_ids)
    registry_hash = master.registry.stable_hash()
    masks = tuple(
        MaskBundle(
            signal_date=bundle.signal_date,
            asset_ids=master.asset_ids,
            asset_registry_hash=registry_hash,
            member=tuple(bundle.member[index] for index in slots),
            observed=tuple(bundle.observed[index] for index in slots),
            feature_missing={
                name: tuple(values[index] for index in slots)
                for name, values in bundle.feature_missing.items()
            },
            label_available=tuple(bundle.label_available[index] for index in slots),
            buyable=tuple(bundle.buyable[index] for index in slots),
            sellable=tuple(bundle.sellable[index] for index in slots),
            loss=tuple(bundle.loss[index] for index in slots),
            evaluation=tuple(bundle.evaluation[index] for index in slots),
        )
        for bundle in panel.masks
    )
    values = {
        name: tuple(tuple(row[index] for index in slots) for row in grid)
        for name, grid in panel.values.items()
    }
    return PanelWindow(panel.dates, master, values, masks)


def _history_ready(
    masks: Sequence[MaskBundle], *, anchor_index: int, lookback: int
) -> tuple[bool, ...]:
    if anchor_index < lookback - 1:
        return tuple(False for _ in masks[anchor_index].asset_ids)
    history = masks[anchor_index - lookback + 1 : anchor_index + 1]
    return tuple(
        all(bundle.member[slot] and bundle.observed[slot] for bundle in history)
        for slot in range(len(masks[anchor_index].asset_ids))
    )


def _admissible_label_scores(rows: Sequence[Any]) -> dict[tuple[date, str], float]:
    """Keep only labels whose realized exit stays inside its assigned fold."""
    labels: dict[tuple[date, str], float] = {}
    for row in rows:
        if row.horizon != 5:
            continue
        is_purged_train = (
            _TRAIN_START <= row.signal_date <= _TRAIN_END
            and row.exit_date < _VALIDATION_START
        )
        is_closed_validation = (
            _VALIDATION_START <= row.signal_date <= _VALIDATION_END
            and row.exit_date <= _VALIDATION_END
        )
        if is_purged_train or is_closed_validation:
            labels[(row.signal_date, row.ts_code)] = (
                row.open_to_open_return - row.benchmark_return
            )
    return labels


class DeepCellRunner:
    """Execute one sealed 2019-2024 -> 2025 deep-model cell."""

    def __init__(
        self,
        *,
        loader_factory: Callable[[Path, str], CanonicalDatasetLoader] = CanonicalDatasetLoader,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.loader_factory = loader_factory
        self.clock = clock

    @staticmethod
    def _verify_receipt(
        payload: Mapping[str, object], job: DeepJobSpec, *, name: str
    ) -> None:
        status = payload.get("status")
        if status not in {"PASS", "PASS_WITH_WARNING"}:
            raise ContractError(f"{name} receipt is not passing")
        model = payload.get("model")
        if model is not None and str(model).lower() != job.model:
            raise ContractError(f"{name} receipt model differs from the job")
        commit = payload.get("commit")
        if commit is not None and commit != job.upstream_commit:
            raise ContractError(f"{name} receipt commit differs from the frozen checkout")
        physical_gpu = payload.get("physical_gpu")
        if physical_gpu is not None and physical_gpu != job.physical_gpu:
            raise ContractError(f"{name} receipt physical GPU differs from the job")

    @staticmethod
    def _git(upstream_root: Path, *arguments: str) -> str:
        return subprocess.run(
            ("git", "-C", str(upstream_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    @staticmethod
    def _require_checkout_read_only(upstream_root: Path) -> None:
        """Reject a checkout whose worktree could be modified by this run."""
        if not upstream_root.is_dir():
            raise ContractError("author checkout root is absent")
        candidates = (upstream_root, *upstream_root.rglob("*"))
        for path in candidates:
            try:
                relative = path.relative_to(upstream_root)
            except ValueError as error:
                raise ContractError("author checkout traversal escaped its root") from error
            if relative.parts and relative.parts[0] == ".git":
                continue
            if path.is_symlink():
                raise ContractError(
                    f"author checkout contains an unreviewed symlink: {relative.as_posix()}"
                )
            if path.stat().st_mode & 0o222:
                raise ContractError(
                    f"author checkout is not read-only: {relative.as_posix() or '.'}"
                )

    def prepare(self, job: DeepJobSpec) -> PreparedDeepJob:
        job.validate()
        evidence_paths = {
            "d0": job.d0_manifest.verify(),
            "environment": job.environment_receipt.verify(),
            "integrity": job.integrity_receipt.verify(),
            "code": job.code_receipt.verify(),
            "adapter": job.adapter_config.verify(),
            "common": job.common_config.verify(),
        }
        if len(set(evidence_paths.values())) != len(evidence_paths):
            raise ContractError("deep evidence roles must reference distinct files")
        verify_source_manifest(
            evidence_paths["code"], Path(__file__).resolve().parents[3]
        )
        self._verify_receipt(_read_json(evidence_paths["environment"]), job, name="environment")
        self._verify_receipt(_read_json(evidence_paths["integrity"]), job, name="integrity")
        adapter_payload = _read_json(evidence_paths["adapter"])
        if adapter_payload.get("model") != job.model or adapter_payload.get(
            "upstream_commit"
        ) != job.upstream_commit:
            raise ContractError("adapter config does not bind the requested author commit")
        common_payload = _read_json(evidence_paths["common"])
        if common_payload.get("schema_version") != "deep_adapter_v1":
            raise ContractError("deep common adapter schema is unsupported")

        d0 = D0Manifest.from_dict(dict(_read_json(evidence_paths["d0"])))
        if d0.cutoff_date < _VALIDATION_END:
            raise ContractError("D0 does not cover the complete 2025 validation window")
        universe_gate = next(item for item in d0.universe_gates if item.universe is job.universe)
        if universe_gate.status not in {
            ResultState.PASS,
            ResultState.PASS_WITH_WARNING,
            ResultState.EXPLORATORY_ONLY,
        }:
            raise ContractError(f"D0 universe gate is not runnable: {universe_gate.status.value}")

        canonical_root = Path(job.canonical_root)
        relative_paths = (
            f"{job.universe.value.lower()}/membership.jsonl",
            f"{job.universe.value.lower()}/features.jsonl",
            f"{job.universe.value.lower()}/labels.jsonl",
            f"{job.universe.value.lower()}/masks.jsonl",
            "shared_market_state.jsonl",
        )
        table_hashes = []
        for relative in relative_paths:
            path = canonical_root / relative
            expected = d0.canonical_table_hashes.get(relative)
            if expected is None or not path.is_file() or _sha256(path) != expected:
                raise ContractError(f"canonical D0 table is absent or hash-mismatched: {relative}")
            table_hashes.append((relative, expected))

        formal_hash = None
        if job.formal_feature_manifest is not None:
            formal = FormalFeatureManifest.from_dict(
                dict(_read_json(job.formal_feature_manifest.verify()))
            )
            if formal.d0_manifest_hash != d0.content_hash:
                raise ContractError("formal feature receipt refers to a different D0")
            expected_dataset_id = (
                f"{d0.dataset_id}:{job.universe.value}:{job.gate.value}"
            )
            if formal.dataset_id != expected_dataset_id:
                raise ContractError("formal feature receipt names a different universe/gate")
            expected_inputs = set(information_inputs(InformationSet(job.gate.value)))
            if set(formal.feature_eligibility) != expected_inputs:
                raise ContractError("formal feature receipt has the wrong input set")
            formal_hash = formal.require_formal_eligible()
        if deep_cell_config_hash(job, d0, formal_hash) != job.cell_config_hash:
            raise ContractError("deep cell_config_hash does not match the job payload")

        upstream_root = Path(job.upstream_root)
        self._require_checkout_read_only(upstream_root)
        if self._git(upstream_root, "rev-parse", "HEAD") != job.upstream_commit:
            raise ContractError("author checkout HEAD differs from the frozen commit")
        if self._git(upstream_root, "status", "--porcelain", "--untracked-files=all"):
            raise ContractError("author checkout is not clean before the run")

        membership_path = canonical_root / relative_paths[0]
        memberships = _membership_intervals(membership_path)
        technology_deviation = job.universe in _TECH_UNIVERSES
        if technology_deviation:
            model_master = _technology_master(memberships, universe=job.universe)
            evaluation_master = model_master
        else:
            model_master = build_causal_asset_master(
                memberships, known_through=_TRAIN_END
            )
            evaluation_master = build_causal_asset_master(
                memberships, known_through=_VALIDATION_END, previous=model_master
            )
        DeepRuntimePolicy().validate_asset_count(len(model_master.asset_ids))
        if evaluation_master.registry.stable_hash() != job.asset_registry_hash:
            raise ContractError("job asset registry differs from the causal validation registry")

        loader = self.loader_factory(canonical_root, job.universe.value)
        dates = tuple(
            row.signal_date for row in loader.iter_masks() if row.signal_date <= _VALIDATION_END
        )
        plan = DeepWindowPlan.build(dates, lookback=job.hyperparameters.lookback_weeks)
        evaluation_panel = loader.load_panel_window(
            dates=plan.input_dates, asset_master=evaluation_master
        )
        model_panel = _restrict_panel(evaluation_panel, model_master)
        schema = FeaturePackingSchema.from_d0(target_feature="return_1d")
        packed = pack_feature_window(model_panel, schema=schema, gate=job.gate)
        labels = _admissible_label_scores(tuple(loader.iter_labels()))
        if not labels:
            raise ContractError("D0 contains no weekly future5d excess labels")
        return PreparedDeepJob(
            job=job,
            d0=d0,
            loader=loader,
            model_master=model_master,
            evaluation_master=evaluation_master,
            model_panel=model_panel,
            evaluation_panel=evaluation_panel,
            packed=packed,
            labels=labels,
            plan=plan,
            universe_table_hashes=tuple(table_hashes),
            formal_feature_hash=formal_hash,
            technology_selection_deviation=technology_deviation,
        )

    def _raise_failure(
        self,
        job: DeepJobSpec,
        *,
        state: ResultState,
        stage: str,
        reason_code: str,
        error: Exception,
        record: bool = True,
    ) -> NoReturn:
        failure = DeepRunFailure(
            run_id=job.run_id or "INVALID_RUN_ID",
            state=state,
            stage=stage,
            reason_code=reason_code,
            detail=str(error),
        )
        output = Path(job.output_dir)
        failure_path = output.parent / f"{output.name}.failure.json"
        if record and output.is_absolute() and not output.exists() and not failure_path.exists():
            _atomic_json(
                failure_path,
                {
                    **failure.to_dict(),
                    "evidence": {
                        "d0_manifest_file_hash": job.d0_manifest.sha256,
                        "environment_receipt_hash": job.environment_receipt.sha256,
                        "integrity_receipt_hash": job.integrity_receipt.sha256,
                        "code_receipt_hash": job.code_receipt.sha256,
                        "adapter_config_hash": job.adapter_config.sha256,
                        "common_config_hash": job.common_config.sha256,
                        "cell_config_hash": job.cell_config_hash,
                    },
                },
            )
        raise failure from error

    @staticmethod
    def _author_model(prepared: PreparedDeepJob, assets: int) -> tuple[Any, Any, Any]:
        job = prepared.job
        arguments = dict(job.hyperparameters.author_arguments)
        arguments.update(
            {
                "seq_len": job.hyperparameters.lookback_weeks,
                "pred_len": job.hyperparameters.forecast_steps,
                "enc_in": assets,
                "dec_in": assets,
                "c_out": assets,
                "learning_rate": job.hyperparameters.learning_rate,
                "train_epochs": job.hyperparameters.maximum_epochs,
                "patience": job.hyperparameters.patience,
                "batch_size": job.hyperparameters.batch_size,
            }
        )
        if job.model == "itransformer":
            module = importlib.import_module("model.iTransformer")
        elif job.model == "timexer":
            module = importlib.import_module("models.TimeXer")
        elif job.model == "timepro":
            module = importlib.import_module("model.TimePro")
        elif job.model == "s4m":
            sys.path.insert(0, os.path.join(str(job.upstream_root), "model"))
            module = importlib.import_module("model.S4M")
            arguments["d_var"] = assets
        else:
            module = importlib.import_module("models.FACT")
        config = SimpleNamespace(**arguments)
        backbone = module.Model(config)
        tools = importlib.import_module("utils.tools")

        def adjust_learning_rate(optimizer: Any, epoch: int) -> None:
            tools.adjust_learning_rate(optimizer, epoch, config)

        return backbone, config, adjust_learning_rate

    @staticmethod
    def _tensor_batches(
        prepared: PreparedDeepJob, torch: Any, device: Any
    ) -> tuple[Any, Any, Any, Any]:
        from a_share_research.adapters.common.torch_runtime import DeepForecastBatch

        packed = prepared.packed
        panel = prepared.model_panel
        job = prepared.job
        values = torch.tensor(packed.values, dtype=torch.float32)
        input_valid = torch.tensor(
            [
                [
                    member and observed
                    for member, observed in zip(
                        mask.member, mask.observed, strict=True
                    )
                ]
                for mask in panel.masks
            ],
            dtype=torch.bool,
        )

        schema = FeaturePackingSchema.from_d0(target_feature="return_1d")
        active_continuous = set(schema.core)
        if job.gate.includes_f:
            active_continuous.update(schema.factors)
        if job.gate.includes_s:
            active_continuous.update(schema.state)
        train_date_end = max(prepared.plan.train_anchor_indices)
        means = torch.zeros(values.shape[-1], dtype=torch.float32)
        scales = torch.ones(values.shape[-1], dtype=torch.float32)
        normalized = values.clone()
        for channel_index, name in enumerate(packed.channels):
            if name not in active_continuous:
                continue
            feature_observed = torch.tensor(
                [
                    [not missing for missing in mask.feature_missing[name]]
                    for mask in panel.masks
                ],
                dtype=torch.bool,
            )
            channel_valid = input_valid & feature_observed
            selected = values[: train_date_end + 1, :, channel_index][
                channel_valid[: train_date_end + 1]
            ]
            if selected.numel() == 0:
                raise AdapterContractError(f"training fold has no observed values for {name}")
            means[channel_index] = selected.mean()
            standard_deviation = selected.std(unbiased=False)
            if bool(torch.isfinite(standard_deviation)) and float(standard_deviation) > 1e-8:
                scales[channel_index] = standard_deviation
            normalized[:, :, channel_index] = (
                (values[:, :, channel_index] - means[channel_index])
                / scales[channel_index]
            ).masked_fill(~channel_valid, 0.0)
        values = normalized
        values = values.masked_fill(~input_valid.unsqueeze(-1), 0.0)

        def materialize(
            indices: tuple[int, ...], *, require_targets: bool
        ) -> tuple[Any, ...]:
            result = []
            batch_size = job.hyperparameters.batch_size
            for offset in range(0, len(indices), batch_size):
                anchors = indices[offset : offset + batch_size]
                x_rows = []
                observed_rows = []
                targets = []
                target_observed = []
                label_available = []
                retained = []
                for anchor in anchors:
                    readiness = _history_ready(
                        panel.masks,
                        anchor_index=anchor,
                        lookback=job.hyperparameters.lookback_weeks,
                    )
                    current = panel.masks[anchor]
                    target_values = []
                    available = []
                    observed_target = []
                    for slot, code in enumerate(prepared.model_master.asset_ids):
                        label = prepared.labels.get((panel.dates[anchor], code))
                        target_values.append(0.0 if label is None else label)
                        available.append(label is not None and current.label_available[slot])
                        observed_target.append(
                            readiness[slot] and current.member[slot] and current.observed[slot]
                        )
                    has_target = any(
                        observed and available_value
                        for observed, available_value in zip(
                            observed_target, available, strict=True
                        )
                    )
                    if require_targets and not has_target:
                        continue
                    retained.append(anchor)
                    window_start = anchor - job.hyperparameters.lookback_weeks + 1
                    x_rows.append(values[window_start : anchor + 1])
                    observed_rows.append(
                        input_valid[window_start : anchor + 1]
                    )
                    targets.append(target_values)
                    target_observed.append(observed_target)
                    label_available.append(available)
                if not retained:
                    continue
                result.append(
                    (
                        tuple(panel.dates[index] for index in retained),
                        DeepForecastBatch(
                            x_enc=torch.stack(x_rows).to(device),
                            x_mark_enc=None,
                            x_dec=None,
                            x_mark_dec=None,
                            observed_mask=torch.stack(observed_rows).to(device),
                            target=torch.tensor(
                                targets, dtype=torch.float32, device=device
                            ).unsqueeze(1),
                            target_observed=torch.tensor(
                                target_observed, dtype=torch.bool, device=device
                            ).unsqueeze(1),
                            label_available=torch.tensor(
                                label_available, dtype=torch.bool, device=device
                            ).unsqueeze(1),
                        ),
                    )
                )
            if not result:
                raise AdapterContractError("deep split produced no eligible tensor batches")
            return tuple(result)

        train = materialize(
            prepared.plan.train_anchor_indices, require_targets=True
        )
        validation = materialize(
            prepared.plan.validation_anchor_indices, require_targets=True
        )
        prediction = materialize(
            prepared.plan.validation_anchor_indices, require_targets=False
        )
        normalizer = {
            "schema_version": "deep_train_normalizer_v1",
            "fit_end": _TRAIN_END.isoformat(),
            "channels": list(packed.channels),
            "active_continuous": sorted(active_continuous),
            "fit_selection": "member_and_observed_and_feature_not_missing",
            "missing_continuous_imputation_after_transform": 0.0,
            "means": [float(value) for value in means.tolist()],
            "scales": [float(value) for value in scales.tolist()],
        }
        return train, validation, prediction, normalizer

    def run(self, job: DeepJobSpec) -> Path:
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

        output = Path(job.output_dir)
        checkpoint = Path(job.checkpoint_dir)
        temporary_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        temporary_checkpoint = checkpoint.with_name(
            f".{checkpoint.name}.tmp-{os.getpid()}"
        )
        reserved_paths = (output, checkpoint, temporary_output, temporary_checkpoint)
        if any(path.exists() for path in reserved_paths):
            self._raise_failure(
                job,
                state=ResultState.INVALID_PROTOCOL,
                stage="OUTPUT_RESERVATION",
                reason_code="ISOLATED_OUTPUT_EXISTS",
                error=ContractError(
                    "deep output/checkpoint path already exists; never overwrite"
                ),
                record=False,
            )
        temporary_output.mkdir(parents=True)
        temporary_checkpoint.mkdir(parents=True)
        started_at = self.clock()
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            self._raise_failure(
                job,
                state=ResultState.INVALID_PROTOCOL,
                stage="RUN_CLOCK",
                reason_code="NAIVE_RUN_CLOCK",
                error=ContractError("runner clock must be timezone-aware"),
            )

        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible != str(job.physical_gpu):
                raise AdapterContractError(
                    "CUDA_VISIBLE_DEVICES must expose only the frozen physical GPU"
                )
            sys.dont_write_bytecode = True
            sys.path.insert(0, job.upstream_root)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise AdapterContractError("deep cell requires exactly one visible CUDA device")
            torch.cuda.set_device(0)
            torch.manual_seed(job.seed)
            torch.cuda.manual_seed_all(job.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0")
            from a_share_research.adapters.common.torch_runtime import (
                ProjectedForecastModule,
                S4MForecastModule,
                SharedPerAssetProjector,
                fit_protocol_safe,
            )

            backbone, _, adjust_learning_rate = self._author_model(
                prepared, len(prepared.model_master.asset_ids)
            )
            if job.model == "s4m":
                module = S4MForecastModule(
                    SharedPerAssetProjector(prepared.packed.input_channel_count),
                    backbone,
                    pred_len=job.hyperparameters.forecast_steps,
                ).to(device)
            else:
                module = ProjectedForecastModule(
                    SharedPerAssetProjector(prepared.packed.input_channel_count),
                    backbone,
                ).to(device)
            binding = UpstreamBinding(
                job.model,
                job.upstream_commit,
                job.integrity_receipt.sha256,
                job.environment_receipt.sha256,
            )
            if job.model == "itransformer":
                adapter = ITransformerAdapter(binding=binding, model=module)
            elif job.model == "timexer":
                adapter = TimeXerAdapter(binding=binding, model=module)
            elif job.model == "timepro":
                adapter = TimeProAdapter(binding=binding, model=module)
            elif job.model == "s4m":
                adapter = S4MAdapter(binding=binding, model=module)
            else:
                adapter = FactAdapter(binding=binding, model=module)
            if isinstance(adapter, FactAdapter):
                adapter.require_supported_core_mix(0.5)
            parameter_count = adapter.parameter_count()
            architecture_hash = adapter.architecture_hash(prepared.packed)
            (
                train_batches,
                validation_batches,
                prediction_tensor_batches,
                normalizer,
            ) = self._tensor_batches(prepared, torch, device)
            if job.model == "s4m":

                def _s4m_sgd(parameters, lr):
                    return torch.optim.SGD(
                        parameters, lr=lr, momentum=0.9, weight_decay=1e-5
                    )

                optimizer_factory = _s4m_sgd
            else:
                optimizer_factory = None
            summary = fit_protocol_safe(
                module,
                tuple(item[1] for item in train_batches),
                tuple(item[1] for item in validation_batches),
                learning_rate=job.hyperparameters.learning_rate,
                maximum_epochs=job.hyperparameters.maximum_epochs,
                patience=job.hyperparameters.patience,
                adjust_learning_rate=adjust_learning_rate,
                checkpoint_path=temporary_checkpoint / "best.pt",
                optimizer_factory=optimizer_factory,
            )
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.TRAIN_FAIL,
                stage="OFFICIAL_FIT",
                reason_code="GPU_FIT_FAILED",
                error=error,
            )

        try:
            module.eval()
            prediction_batches = []
            with torch.no_grad():
                for signal_dates, batch in prediction_tensor_batches:
                    output_tensor = module(
                        batch.x_enc,
                        batch.x_mark_enc,
                        batch.x_dec,
                        batch.x_mark_dec,
                        batch.observed_mask,
                    )
                    if isinstance(output_tensor, tuple):
                        output_tensor = output_tensor[0]
                    if output_tensor.ndim != 3 or output_tensor.shape[1] != 1:
                        raise AdapterContractError(
                            "author validation output must be [B,1,asset]"
                        )
                    score_rows = tuple(
                        tuple(float(value) for value in row)
                        for row in output_tensor[:, 0, :].detach().cpu().tolist()
                    )
                    prediction_batches.append(PredictionBatch(signal_dates, score_rows))
            torch.cuda.synchronize(device)

            expected_dates = tuple(
                prepared.plan.input_dates[index]
                for index in prepared.plan.validation_anchor_indices
            )
            date_slot = {
                day: index for index, day in enumerate(prepared.evaluation_panel.dates)
            }
            evaluation_masks = tuple(
                prepared.evaluation_panel.masks[date_slot[day]] for day in expected_dates
            )
            history_ready = tuple(
                _history_ready(
                    prepared.evaluation_panel.masks,
                    anchor_index=date_slot[day],
                    lookback=job.hyperparameters.lookback_weeks,
                )
                for day in expected_dates
            )
            predictions: PredictionFrame = export_prediction_batches(
                run_id=job.run_id,
                evaluation_registry=prepared.evaluation_master.registry,
                model_master=prepared.model_master,
                expected_dates=expected_dates,
                masks=evaluation_masks,
                history_ready=history_ready,
                batches=prediction_batches,
            )
            predictions.validate()
            universe_gate = next(
                item for item in prepared.d0.universe_gates if item.universe is job.universe
            )
            status = (
                ResultState.EXPLORATORY_ONLY
                if job.universe in _TECH_UNIVERSES
                else universe_gate.status
            )
            completed_at = self.clock()
            manifest = RunManifest(
                run_id=job.run_id,
                model=job.model,
                universe=job.universe,
                information_set=job.gate.value,
                split=Partition.VALIDATION,
                purpose=Purpose.SELECT,
                data_hash=prepared.d0.content_hash,
                asset_registry_hash=job.asset_registry_hash,
                execution_calendar_manifest_hash=prepared.d0.trading_calendar_hash,
                feature_schema_hash=prepared.packed.schema_hash,
                market_state_hash=prepared.d0.market_state_hash,
                config_hash=canonical_hash(
                    {
                        "cell_config_hash": job.cell_config_hash,
                        "capacity_hash": job.hyperparameters.capacity_hash,
                        "gate": job.gate.value,
                    }
                ),
                code_hash=job.code_receipt.sha256,
                upstream_commit=job.upstream_commit,
                seed=job.seed,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                prediction_hash=predictions.stable_hash(),
                formal_feature_manifest_hash=prepared.formal_feature_hash,
                deviations=(
                    "Official author backbone/loss/Adam/scheduler/checkpoint inference preserved.",
                    "External shared per-asset C-to-1 projector adapts PIT A-share channels.",
                    "Weekly 96-step inputs predict one T+1-open future5d excess-return score.",
                    "Loss is restricted to member+observed+history-ready+label-available rows.",
                    "CUDA deterministic algorithms are fail-closed for three-seed replay.",
                    *(
                        ("2026-selected technology list is retrospective and exploratory only.",)
                        if prepared.technology_selection_deviation
                        else ()
                    ),
                ),
            )
            if self._git(
                Path(job.upstream_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ):
                raise AdapterContractError("author checkout changed during the run")
            provenance = {
                "schema_version": "deep_run_provenance_v1",
                "run_id": job.run_id,
                "model": job.model,
                "upstream_commit": job.upstream_commit,
                "upstream_clean_before_and_after": True,
                "physical_gpu": job.physical_gpu,
                "logical_gpu": 0,
                "torch_version": torch.__version__,
                "d0_content_hash": prepared.d0.content_hash,
                "d0_manifest_file_hash": job.d0_manifest.sha256,
                "canonical_table_hashes": dict(prepared.universe_table_hashes),
                "environment_receipt_hash": job.environment_receipt.sha256,
                "integrity_receipt_hash": job.integrity_receipt.sha256,
                "code_receipt_hash": job.code_receipt.sha256,
                "adapter_config_hash": job.adapter_config.sha256,
                "common_config_hash": job.common_config.sha256,
                "cell_config_hash": job.cell_config_hash,
                "job_spec_hash": job.stable_hash(),
                "capacity_hash": job.hyperparameters.capacity_hash,
                "architecture_hash": architecture_hash,
                "parameter_count": parameter_count,
                "input_channels": prepared.packed.input_channel_count,
                "asset_tokens": prepared.packed.model_variate_count,
                "target": "future5d_stock_open_to_open_log_return_minus_csi300",
                "entry": "T+1_OPEN",
                "train_window": [_TRAIN_START.isoformat(), _TRAIN_END.isoformat()],
                "actual_train_anchor_window": [
                    prepared.plan.input_dates[
                        prepared.plan.train_anchor_indices[0]
                    ].isoformat(),
                    prepared.plan.input_dates[
                        prepared.plan.train_anchor_indices[-1]
                    ].isoformat(),
                ],
                "train_anchor_count": len(prepared.plan.train_anchor_indices),
                "fit_train_anchor_count": sum(
                    len(signal_dates) for signal_dates, _ in train_batches
                ),
                "validation_window": [
                    _VALIDATION_START.isoformat(),
                    _VALIDATION_END.isoformat(),
                ],
                "validation_anchor_count": len(
                    prepared.plan.validation_anchor_indices
                ),
                "fit_validation_anchor_count": sum(
                    len(signal_dates) for signal_dates, _ in validation_batches
                ),
                "best_epoch": summary.best_epoch,
                "epochs_completed": summary.epochs_completed,
                "best_validation_mse": summary.best_validation_mse,
                "selected_target_count": summary.selected_target_count,
                "prediction_rows": len(predictions.records),
                "prediction_coverage": predictions.coverage,
                "fact_core": 0.5 if job.model == "fact" else None,
                "deterministic_algorithms": True,
                "test_loader_constructed_during_fit": False,
                "legacy_2026_market_feature_or_label_read": False,
                "technology_2026_selection_metadata_used": (
                    prepared.technology_selection_deviation
                ),
            }
            _atomic_json(temporary_output / "predictions.json", predictions)
            _atomic_json(temporary_output / "run_manifest.json", manifest)
            _atomic_json(temporary_output / "normalizer.json", normalizer)
            _atomic_json(temporary_output / "provenance.json", provenance)
            os.replace(temporary_checkpoint, checkpoint)
            os.replace(temporary_output, output)
            return output
        except Exception as error:
            self._raise_failure(
                job,
                state=ResultState.EVAL_FAIL,
                stage="PREDICTION_EXPORT",
                reason_code="PREDICTION_OR_MANIFEST_FAILED",
                error=error,
            )
