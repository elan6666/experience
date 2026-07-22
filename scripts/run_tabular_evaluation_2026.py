#!/usr/bin/env python3
"""Server-only 2026 LEGACY_VIEWED evaluation runner for tabular models.

Trains on 2019-2024 (identical to V0/V1), uses 2025 for lightgbm early-stop,
then predicts on the 2026-01-01..2026-07-17 legacy-viewed fold.  Produces
``eval-2026-a{gate}-...`` prediction frames consumable by ``score_2026.py``.

CPU-only; no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from a_share_research.contracts import ContractError, RunManifest
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.experiments.tabular_runner import (
    _TRAIN_END,
    _VALIDATION_END,
    TabularCellRunner,
    TabularJobSpec,
    _default_adapter_factory,
    _information_coverage,
    _load_layout,
)
from a_share_research.models.tabular import FeatureGate
from a_share_research.models.tabular.common import TabularDiagnostics
from a_share_research.protocol import Partition, Purpose
from a_share_research.quality.states import ResultState

_EVAL_START = date(2026, 1, 1)
_EVAL_END = date(2026, 7, 17)


def _legacy_evaluation_manifest(
    *,
    job: TabularJobSpec,
    prepared: object,
    layout: object,
    diagnostics: TabularDiagnostics,
    prediction_hash: str,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
) -> RunManifest:
    """Create the non-rankable manifest for the closed 2026 legacy-viewed fold."""
    d0 = getattr(prepared, "d0")
    return RunManifest(
        run_id=run_id,
        model=job.model,
        universe=job.universe,
        information_set=job.information_set.value,
        split=Partition.LEGACY_VIEWED,
        purpose=Purpose.LEGACY_REPORT,
        data_hash=d0.content_hash,
        asset_registry_hash=job.asset_registry_hash,
        execution_calendar_manifest_hash=d0.trading_calendar_hash,
        feature_schema_hash=layout.stable_hash(),
        market_state_hash=d0.market_state_hash,
        config_hash=diagnostics.config_hash,
        code_hash=job.code_receipt.sha256,
        upstream_commit=job.upstream_commit,
        seed=job.seed,
        status=ResultState.EXPLORATORY_ONLY,
        started_at=started_at,
        completed_at=completed_at,
        prediction_hash=prediction_hash,
        formal_feature_manifest_hash=job.formal_feature_manifest_hash,
        deviations=(
            "2026-01-01..2026-07-17 is LEGACY_VIEWED and excluded from selection.",
            "Tabular adapter around package-native estimator; no estimator/loss rewrite.",
        ),
    )


def _load_eval_prediction_samples(
    loader: CanonicalDatasetLoader,
    gate: FeatureGate,
    layout: object,
) -> tuple:
    """Load 2026 LEGACY_VIEWED complete-panel samples for prediction."""
    samples = tuple(
        loader.iter_tabular_samples(
            horizon=5,
            relative_target=True,
            start=_EVAL_START,
            end=_EVAL_END,
            complete_panel=True,
        )
    )
    if not samples:
        raise ContractError("no 2026 legacy-viewed samples found in D0")
    samples = _information_coverage(
        samples,
        gate=gate,
        state_names=layout.market_state,
    )
    return samples


def run_eval_2026(job_spec_path: Path, output_dir: Path | None = None) -> Path:
    """Run one tabular cell on the 2026 legacy-viewed fold."""
    job_payload = json.loads(job_spec_path.read_text(encoding="utf-8"))
    job = TabularJobSpec.from_dict(job_payload)

    runner = TabularCellRunner()
    prepared = runner.prepare(job)

    layout = _load_layout(job.layout_config.verify())
    gate = FeatureGate(job.information_set)
    loader = CanonicalDatasetLoader(Path(job.canonical_root), job.universe.value)

    eval_prediction = _load_eval_prediction_samples(loader, gate, layout)

    eval_run_id = (
        f"eval-2026-a{job.information_set.value.lower()[-1]}-"
        f"{job.universe.value.lower()}-{job.model}-seed-{job.seed:08d}"
    )

    if output_dir is None:
        output_dir = Path(job.output_dir).parent / eval_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)

    adapter = _default_adapter_factory(job, layout, gate, prepared.model_payload)

    common = {
        "run_id": eval_run_id,
        "training": prepared.training,
        "prediction": eval_prediction,
        "fit_end": _TRAIN_END,
        "fit_data_hash": prepared.d0.content_hash,
        "fold_id": f"weekly-future5d-{job.universe.value.lower()}-train2019-2024-eval2026",
    }

    if job.model == "ridge":
        result = adapter.fit_predict(**common)
    else:
        result = adapter.fit_predict(
            **common,
            validation=prepared.validation,
            validation_end=_VALIDATION_END,
        )

    result.predictions.validate()

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(result.predictions.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    manifest = _legacy_evaluation_manifest(
        job=job,
        prepared=prepared,
        layout=layout,
        diagnostics=result.diagnostics,
        prediction_hash=result.predictions.stable_hash(),
        run_id=eval_run_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "run_id": eval_run_id,
                "state": "PASS",
                "predictions": str(predictions_path),
                "paired_dates": len({s.signal_date for s in eval_prediction}),
                "paired_rows": len(eval_prediction),
            },
            sort_keys=True,
        )
    )
    return predictions_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run tabular 2026 legacy-viewed evaluation (CPU only)."
    )
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        run_eval_2026(args.job_spec, args.output_dir)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"state": "FAILED", "reason": str(error), "job_spec": str(args.job_spec)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
