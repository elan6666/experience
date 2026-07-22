#!/usr/bin/env python3
"""Audit the exact lineage of server-only 2026 evaluation predictions.

The report is deliberately descriptive: it records what is present, validates
the sibling prediction/run-manifest hash edge, and reports a requested coverage
matrix.  It never promotes an incomplete or legacy-viewed result to a formal
selection result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from a_share_research.contracts import PredictionFrame
from a_share_research.evaluation.evaluation_2026 import parse_eval_2026_run_id


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON document must be an object")
    return payload


def audit_eval_2026_lineage(
    runs_root: Path,
    *,
    expected_models: Iterable[str],
    expected_universes: Iterable[str],
    expected_gates: Iterable[str],
    expected_seeds: int,
    single_seed_models: Iterable[str] = (),
) -> dict[str, object]:
    """Return a complete inventory and fail-closed sibling-hash findings."""
    cells: list[dict[str, object]] = []
    findings: list[dict[str, str]] = []
    observed: Counter[tuple[str, str, str]] = Counter()
    code_hashes: set[str] = set()
    data_hashes: set[str] = set()
    schema_hashes: set[str] = set()

    for prediction_path in sorted(runs_root.rglob("predictions.json")):
        manifest_path = prediction_path.with_name("run_manifest.json")
        relative_path = str(prediction_path.relative_to(runs_root))
        try:
            frame = PredictionFrame.from_dict(_load_json(prediction_path))
        except Exception as error:  # report every malformed output, not just the first
            findings.append(
                {"path": relative_path, "reason": f"invalid_prediction:{error}"}
            )
            continue
        key = parse_eval_2026_run_id(frame.run_id)
        if key is None:
            continue
        if not manifest_path.is_file():
            findings.append({"path": relative_path, "reason": "missing_run_manifest"})
            continue
        try:
            manifest = _load_json(manifest_path)
        except Exception as error:
            findings.append(
                {"path": relative_path, "reason": f"invalid_run_manifest:{error}"}
            )
            continue
        reason: str | None = None
        if manifest.get("run_id") != frame.run_id:
            reason = "run_id_mismatch"
        elif manifest.get("prediction_hash") != frame.stable_hash():
            reason = "prediction_hash_mismatch"
        elif str(manifest.get("information_set", "")).lower() != f"a{key.gate}":
            reason = "information_set_mismatch"
        elif str(manifest.get("universe", "")).lower() != key.universe:
            reason = "universe_mismatch"
        elif str(manifest.get("model", "")).lower() != key.model:
            reason = "model_mismatch"
        elif manifest.get("seed") != key.seed:
            reason = "seed_mismatch"
        if reason is not None:
            findings.append({"path": relative_path, "reason": reason})
            continue

        observed[(key.universe, f"A{key.gate}", key.model)] += 1
        for field, values in (
            ("code_hash", code_hashes),
            ("data_hash", data_hashes),
            ("feature_schema_hash", schema_hashes),
        ):
            value = manifest.get(field)
            if isinstance(value, str) and value:
                values.add(value)
        cells.append(
            {
                "run_id": frame.run_id,
                "path": relative_path,
                "status": manifest.get("status"),
                "purpose": manifest.get("purpose"),
                "split": manifest.get("split"),
                "formal_feature_manifest_hash": manifest.get("formal_feature_manifest_hash"),
            }
        )

    single_seed = set(single_seed_models)
    coverage: list[dict[str, object]] = []
    for universe in sorted(set(expected_universes)):
        for gate in sorted(set(expected_gates)):
            for model in sorted(set(expected_models)):
                expected_count = 1 if model in single_seed else expected_seeds
                coverage.append(
                    {
                        "universe": universe,
                        "gate": gate,
                        "model": model,
                        "observed_seed_count": observed[(universe, gate, model)],
                        "expected_seed_count": expected_count,
                        "state": (
                            "COMPLETE"
                            if observed[(universe, gate, model)] == expected_count
                            else "INCOMPLETE"
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "runs_root": str(runs_root.resolve()),
        "valid_cell_count": len(cells),
        "invalid_cell_count": len(findings),
        "lineage_hashes": {
            "code_hashes": sorted(code_hashes),
            "data_hashes": sorted(data_hashes),
            "feature_schema_hashes": sorted(schema_hashes),
        },
        "cells": cells,
        "coverage": coverage,
        "findings": findings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-models",
        nargs="+",
        default=["ridge", "lightgbm", "itransformer", "fact", "timexer"],
    )
    parser.add_argument(
        "--expected-universes", nargs="+", default=["csi300", "tech32", "tech90"]
    )
    parser.add_argument("--expected-gates", nargs="+", default=["A0", "A1", "A2", "A3"])
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument(
        "--single-seed-models", nargs="+", default=["ridge", "lightgbm"]
    )
    args = parser.parse_args(argv)
    if args.expected_seeds < 1:
        parser.error("--expected-seeds must be positive")
    report = audit_eval_2026_lineage(
        args.runs_root,
        expected_models=args.expected_models,
        expected_universes=args.expected_universes,
        expected_gates=args.expected_gates,
        expected_seeds=args.expected_seeds,
        single_seed_models=args.single_seed_models,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid_cells": report["valid_cell_count"],
                "invalid_cells": report["invalid_cell_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 1 if report["invalid_cell_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
