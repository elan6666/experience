from __future__ import annotations

import runpy
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from a_share_research.models.tabular.common import TabularDiagnostics
from a_share_research.models.tabular.layout import InformationSet
from a_share_research.protocol import Partition, Purpose, UniverseClass
from a_share_research.quality.states import ResultState


def test_legacy_tabular_evaluation_manifest_is_complete_and_non_rankable() -> None:
    script = runpy.run_path(
        Path(__file__).parents[2] / "scripts" / "run_tabular_evaluation_2026.py"
    )
    build = script["_legacy_evaluation_manifest"]
    digest = "a" * 64
    job = SimpleNamespace(
        model="ridge",
        universe=UniverseClass.CSI300,
        information_set=InformationSet.A1,
        asset_registry_hash=digest,
        code_receipt=SimpleNamespace(sha256=digest),
        upstream_commit="internal:scikit-learn",
        seed=20260719,
        formal_feature_manifest_hash=digest,
    )
    prepared = SimpleNamespace(
        d0=SimpleNamespace(
            content_hash=digest,
            trading_calendar_hash=digest,
            market_state_hash=digest,
        )
    )
    layout = SimpleNamespace(stable_hash=lambda: digest)
    diagnostics = TabularDiagnostics(
        model="ridge",
        information_set="A1",
        config_hash=digest,
        layout_hash=digest,
        gate_hash=digest,
        preprocessing_hash=digest,
        fit_data_hash=digest,
        fold_id="weekly-future5d-csi300-train2019-2024-eval2026",
        training_start=date(2019, 1, 1),
        training_end=date(2024, 12, 31),
        validation_start=None,
        validation_end=None,
        n_train=1,
        n_validation=0,
        n_prediction=1,
        n_scored=1,
        seed=20260719,
        feature_importance=(),
    )
    stamp = datetime(2026, 7, 22, tzinfo=timezone.utc)

    manifest = build(
        job=job,
        prepared=prepared,
        layout=layout,
        diagnostics=diagnostics,
        prediction_hash=digest,
        run_id="eval-2026-a1-csi300-ridge-seed-20260719",
        started_at=stamp,
        completed_at=stamp,
    )

    manifest.validate()
    assert manifest.seed == 20260719
    assert manifest.split is Partition.LEGACY_VIEWED
    assert manifest.purpose is Purpose.LEGACY_REPORT
    assert manifest.status is ResultState.EXPLORATORY_ONLY
