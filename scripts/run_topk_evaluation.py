#!/usr/bin/env python3
"""Server-only Top-K realistic portfolio evaluation (CPU, no GPU).

Loads model predictions and canonical labels, builds weekly Top-K long-only
portfolios with A-share transaction costs, and reports net-of-cost performance
metrics (net return, Sharpe, max drawdown, win rate, turnover) for multiple
K values and strategies.

Works with both 2025 validation and 2026 evaluation prediction sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from a_share_research.contracts import ContractError, PredictionFrame
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.data.labels import CompactLabel
from a_share_research.evaluation.topk import (
    TopKConfig,
    TransactionCostModel,
    evaluate_multiple_k,
)


def _load_prediction_frame(path: Path) -> PredictionFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PredictionFrame.from_dict(payload)


def _discover_predictions(
    runs_root: Path,
) -> dict[str, dict[str, dict[str, PredictionFrame]]]:
    """Group prediction frames by universe -> model -> run_id."""
    grouped: dict[str, dict[str, dict[str, PredictionFrame]]] = {}
    for path in sorted(runs_root.rglob("predictions.json")):
        try:
            frame = _load_prediction_frame(path)
        except Exception:
            continue
        run_id = frame.run_id
        parts = run_id.split("-")
        universe = None
        model = None
        for p in parts:
            if p in ("csi300", "star50", "tech32", "tech90"):
                universe = p
            elif p in ("ridge", "lightgbm", "itransformer", "fact", "timepro", "timexer"):
                model = p
        if universe is None or model is None:
            continue
        grouped.setdefault(universe, {}).setdefault(model, {})[run_id] = frame
    return grouped


def _load_labels(
    canonical_root: Path,
    universe: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[tuple[date, str], CompactLabel]:
    """Load canonical labels for a universe, optionally filtered by date range."""
    loader = CanonicalDatasetLoader(
        canonical_root=canonical_root,
        universe=universe,
    )
    labels: dict[tuple[date, str], CompactLabel] = {}
    for compact in loader.iter_labels():
        if compact.horizon != 5:
            continue
        if start_date is not None and compact.signal_date < start_date:
            continue
        if end_date is not None and compact.signal_date > end_date:
            continue
        labels[(compact.signal_date, compact.ts_code)] = compact
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Top-K realistic portfolio evaluation with A-share costs."
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--staged-calendar", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--universes",
        nargs="+",
        default=["csi300", "star50", "tech32", "tech90"],
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 8])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["equal_weight", "turnover_control", "kelly"],
    )
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.fromisoformat(args.end_date) if args.end_date else None

    all_predictions = _discover_predictions(args.runs_root)
    all_summaries: list[dict[str, object]] = []

    for universe in args.universes:
        models = all_predictions.get(universe, {})
        if not models:
            print(f"[{universe}] No predictions found", file=sys.stderr)
            continue

        print(f"[{universe}] Loading labels...", file=sys.stderr)
        labels = _load_labels(
            args.canonical_root,
            universe,
            start,
            end,
        )
        print(f"[{universe}] {len(labels)} labels loaded", file=sys.stderr)

        for model, frames in sorted(models.items()):
            for run_id, frame in sorted(frames.items()):
                print(f"[{universe}/{model}] Evaluating {run_id}...", file=sys.stderr)
                try:
                    results = evaluate_multiple_k(
                        frame=frame,
                        labels=labels,
                        k_values=args.k_values,
                        strategies=args.strategies,
                        capital=args.capital,
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    continue

                for result in results:
                    summary = result.to_summary()
                    summary["universe"] = universe
                    summary["model"] = model
                    summary["run_id"] = run_id
                    all_summaries.append(summary)

                    print(
                        f"  {summary['strategy']} K={summary['k']}: "
                        f"net={summary['total_net_return_pct']}% "
                        f"sharpe={summary['sharpe_ratio']} "
                        f"maxdd={summary['max_drawdown_pct']}% "
                        f"turnover={summary['annual_turnover_pct']}%",
                        file=sys.stderr,
                    )

    output_path = args.output_dir / "topk_summary.json"
    output_path.write_text(
        json.dumps(all_summaries, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")
    print(f"Total evaluations: {len(all_summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
