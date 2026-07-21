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
from datetime import date
from pathlib import Path
from collections.abc import Sequence

from a_share_research.contracts import ContractError, PredictionFrame
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.experiments.tabular_runner import (
    TabularCellRunner,
    TabularJobSpec,
    _TRAIN_END,
    _VALIDATION_END,
    _VALIDATION_START,
    _information_coverage,
    _load_layout,
    _default_adapter_factory,
)
from a_share_research.models.tabular import FeatureGate

_EVAL_START = date(2026, 1, 1)
_EVAL_END = date(2026, 7, 17)


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

    eval_run_id = f"eval-2026-a{job.information_set.value.lower()[-1]}-{job.universe.value.lower()}-{job.model}-seed-{job.seed:08d}"

    if output_dir is None:
        output_dir = Path(job.output_dir).parent / eval_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

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

    manifest = {
        "run_id": eval_run_id,
        "model": job.model,
        "universe": job.universe.value,
        "information_set": job.information_set.value,
        "partition": "LEGACY_VIEWED",
        "purpose": "LEGACY_REPORT",
        "prediction_hash": result.predictions.stable_hash(),
        "source_run_id": job.run_id,
        "status": "PASS",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
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
