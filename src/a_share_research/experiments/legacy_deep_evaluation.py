"""Contracts for the closed 2026 deep-model reporting fold.

This module deliberately does *not* change :mod:`deep_runner`.  That runner is
the sealed 2019--2024 fit / 2025 selection boundary and must reject every 2026
row.  A legacy evaluation consumes that sealed boundary and adds a separate
prediction-only window which is permanently marked ``LEGACY_VIEWED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar

from a_share_research.contracts import CanonicalModel, ContractError, canonical_hash
from a_share_research.experiments.deep_runner import DeepJobSpec
from a_share_research.protocol import Partition, ProtocolSpec, Purpose

_EVALUATION_START = date(2026, 1, 1)
_EVALUATION_END = date(2026, 7, 17)
_SERVER_ROOT = Path("/data/yilangliu/a_share_research")


def _under_server_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(_SERVER_ROOT)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class LegacyDeepEvaluationPlan:
    """Indices for fit, selection and prediction-only legacy reporting."""

    input_dates: tuple[date, ...]
    train_anchor_indices: tuple[int, ...]
    validation_anchor_indices: tuple[int, ...]
    legacy_anchor_indices: tuple[int, ...]
    lookback: int

    @classmethod
    def build(cls, dates: tuple[date, ...], *, lookback: int) -> "LegacyDeepEvaluationPlan":
        if dates != tuple(sorted(set(dates))) or not dates:
            raise ContractError("legacy deep signal dates must be unique and increasing")
        if dates[-1] > _EVALUATION_END:
            raise ContractError("legacy deep evaluation cannot admit future-unseen rows")
        if type(lookback) is not int or lookback <= 0:
            raise ContractError("deep lookback must be a positive integer")

        protocol = ProtocolSpec.research_v1()
        train = tuple(
            index
            for index, day in enumerate(dates)
            if index >= lookback and protocol.partition_for(day) is Partition.TRAIN
        )
        validation = tuple(
            index
            for index, day in enumerate(dates)
            if index >= lookback and protocol.partition_for(day) is Partition.VALIDATION
        )
        legacy = tuple(
            index
            for index, day in enumerate(dates)
            if index >= lookback and protocol.partition_for(day) is Partition.LEGACY_VIEWED
        )
        expected_validation = tuple(
            index
            for index, day in enumerate(dates)
            if protocol.partition_for(day) is Partition.VALIDATION
        )
        expected_legacy = tuple(
            index
            for index, day in enumerate(dates)
            if protocol.partition_for(day) is Partition.LEGACY_VIEWED
        )
        if not train or not validation or not legacy:
            raise ContractError("legacy deep fit, selection or report fold is empty")
        if validation != expected_validation:
            raise ContractError("D0 lacks enough history for complete 2025 selection coverage")
        if legacy != expected_legacy:
            raise ContractError("D0 lacks enough history for complete 2026 report coverage")
        for index in train:
            protocol.assert_access(dates[index], Purpose.FIT)
        for index in validation:
            protocol.assert_access(dates[index], Purpose.SELECT)
        for index in legacy:
            protocol.assert_access(dates[index], Purpose.LEGACY_REPORT)
        return cls(dates, train, validation, legacy, lookback)


@dataclass(frozen=True)
class LegacyDeepEvaluationSpec(CanonicalModel):
    """Immutable, isolated replay request for one sealed deep-model cell.

    ``source_job`` preserves the exact author model, hyperparameters, evidence,
    seed and frozen GPU contract.  It is an input specification, not a
    checkpoint: a legacy evaluation must retrain from the 2019--2024 fold.
    """

    SCHEMA_NAME: ClassVar[str] = "legacy_deep_evaluation_spec"

    source_job: DeepJobSpec
    run_id: str
    output_dir: str
    checkpoint_dir: str
    evaluation_asset_registry_hash: str

    def validate(self) -> None:
        self.source_job.validate()
        expected_run_id = (
            f"eval-2026-a{self.source_job.gate.value.lower()[-1]}-"
            f"{self.source_job.universe.value.lower()}-{self.source_job.model}-"
            f"seed-{self.source_job.seed:08d}"
        )
        if self.run_id != expected_run_id:
            raise ContractError("legacy deep evaluation run_id is not canonical")
        for name in ("output_dir", "checkpoint_dir"):
            path = Path(getattr(self, name))
            if not path.is_absolute() or not _under_server_root(path):
                raise ContractError(f"{name} must remain below the approved server root")
        if Path(self.output_dir) == Path(self.checkpoint_dir):
            raise ContractError("legacy deep output and checkpoint paths overlap")
        if Path(self.output_dir) == Path(self.source_job.output_dir):
            raise ContractError("legacy deep output must not overwrite its sealed source cell")
        if Path(self.checkpoint_dir) == Path(self.source_job.checkpoint_dir):
            raise ContractError("legacy deep checkpoint must not overwrite its sealed source cell")
        if len(self.evaluation_asset_registry_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.evaluation_asset_registry_hash
        ):
            raise ContractError("evaluation_asset_registry_hash must be SHA-256")

    @property
    def config_hash(self) -> str:
        """Full replay identity; retained independently of the source V0/V1 cell."""
        return canonical_hash(
            {
                "schema_version": "legacy_deep_evaluation_config_v1",
                "source_job_hash": self.source_job.stable_hash(),
                "run_id": self.run_id,
                "evaluation_asset_registry_hash": self.evaluation_asset_registry_hash,
                "fit_window": ["2019-01-01", "2024-12-31"],
                "selection_window": ["2025-01-01", "2025-12-31"],
                "report_window": [_EVALUATION_START.isoformat(), _EVALUATION_END.isoformat()],
                "report_policy": "LEGACY_REPORT_ONLY_NO_SELECTION",
            }
        )
