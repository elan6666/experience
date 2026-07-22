"""Contracts for the separate closed 2026 deep evaluation boundary."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from a_share_research.adapters.common import InformationGate
from a_share_research.contracts import ContractError
from a_share_research.experiments.legacy_deep_evaluation import (
    LegacyDeepEvaluationPlan,
    LegacyDeepEvaluationSpec,
)
from a_share_research.protocol import UniverseClass

from .test_deep_runner import job


def _dates() -> tuple[date, ...]:
    start = date(2017, 1, 6)
    return tuple(start + timedelta(days=7 * index) for index in range(500))


def _spec() -> LegacyDeepEvaluationSpec:
    source = job(phase="V1", gate=InformationGate.A1)
    run_id = "eval-2026-a1-csi300-itransformer-seed-20260719"
    return LegacyDeepEvaluationSpec(
        source_job=source,
        run_id=run_id,
        output_dir=f"/data/yilangliu/a_share_research/runs/eval-2026/{run_id}",
        checkpoint_dir=f"/data/yilangliu/a_share_research/checkpoints/eval-2026/{run_id}",
        evaluation_asset_registry_hash="b" * 64,
    )


def test_legacy_plan_keeps_2026_out_of_fit_and_selection() -> None:
    plan = LegacyDeepEvaluationPlan.build(_dates(), lookback=96)
    assert all(plan.input_dates[index] < date(2025, 1, 1) for index in plan.train_anchor_indices)
    assert all(
        date(2025, 1, 1) <= plan.input_dates[index] <= date(2025, 12, 31)
        for index in plan.validation_anchor_indices
    )
    assert all(
        date(2026, 1, 1) <= plan.input_dates[index] <= date(2026, 7, 17)
        for index in plan.legacy_anchor_indices
    )


def test_legacy_plan_rejects_future_unseen_dates() -> None:
    with pytest.raises(ContractError, match="future-unseen"):
        LegacyDeepEvaluationPlan.build(_dates() + (date(2026, 7, 24),), lookback=96)


def test_legacy_spec_isolated_from_source_outputs_and_has_stable_identity() -> None:
    spec = _spec()
    spec.validate()
    assert len(spec.config_hash) == 64
    with pytest.raises(ContractError, match="must not overwrite"):
        LegacyDeepEvaluationSpec(
            **{**spec.__dict__, "output_dir": spec.source_job.output_dir}
        )
