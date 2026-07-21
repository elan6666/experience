#!/usr/bin/env python3
"""Server-only sealed entry point for V0 Step 4 diagnostic scoring.

Scores every discovered V0 model on the frozen 2025 validation fold for each
requested universe, writing one ``V0UniverseScore`` JSON per universe plus a
compact COMMON/BENCHMARK_RELATIVE rank-IC summary. CPU-only; no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from a_share_research.contracts import ContractError
from a_share_research.evaluation.schema import OutcomeMode, SupportMode
from a_share_research.evaluation.v0_scoring import (
    discover_all_v0_predictions,
    score_v0_universe,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score V0 predictions on the 2025 validation fold (CPU only)."
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--staged-calendar", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--universes",
        nargs="+",
        default=["csi300", "tech32", "tech90"],
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = discover_all_v0_predictions(args.runs_root)
    summary: list[dict[str, object]] = []
    exit_code = 0
    for universe in args.universes:
        frames_by_model = all_frames.get(universe, {})
        if not frames_by_model:
            print(
                json.dumps(
                    {"universe": universe, "state": "NO_PREDICTIONS"},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            exit_code = 1
            continue
        try:
            result = score_v0_universe(
                canonical_root=args.canonical_root,
                universe=universe,
                staged_calendar=args.staged_calendar,
                frames_by_model=frames_by_model,
            )
        except ContractError as error:
            print(
                json.dumps(
                    {"universe": universe, "state": "FAILED", "reason": str(error)},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            exit_code = 1
            continue
        out = args.output_dir / f"v0-score-{universe}.json"
        out.write_text(
            json.dumps(result.to_dict(), sort_keys=True), encoding="utf-8"
        )
        for agg in result.aggregates:
            if (
                agg.support is SupportMode.COMMON
                and agg.outcome is OutcomeMode.BENCHMARK_RELATIVE
            ):
                summary.append(
                    {
                        "universe": universe,
                        "model": agg.model,
                        "seeds": agg.seed_count,
                        "rank_ic_mean": agg.rank_ic_mean,
                        "rank_ic_std": agg.rank_ic_std,
                    }
                )
        print(
            json.dumps(
                {
                    "universe": universe,
                    "state": "SCORED",
                    "models": result.model_count,
                    "eligible_keys": result.eligible_keys_count,
                    "common_support": result.common_support_count,
                    "scorecards": len(result.scorecards),
                    "failures": len(result.failures),
                    "output": str(out),
                },
                sort_keys=True,
            )
        )

    summary_path = args.output_dir / "v0-score-summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
