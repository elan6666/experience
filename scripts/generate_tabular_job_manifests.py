"""Generate sealed Ridge/LightGBM V0/V1 manifests on the approved server."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from a_share_research.contracts import ContractError
from a_share_research.experiments.tabular_job_generator import (
    APPROVED_SERVER_ROOT,
    build_tabular_jobs,
    write_tabular_jobs,
)
from a_share_research.models.tabular import InformationSet
from a_share_research.protocol import UniverseClass


def _model_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or key in result:
            raise ContractError(f"expected unique MODEL=/absolute/path mapping: {value}")
        result[key] = Path(raw_path)
    return result


def _formal_paths(values: list[str]) -> dict[tuple[UniverseClass, InformationSet], Path]:
    result: dict[tuple[UniverseClass, InformationSet], Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        universe_name, colon, information_name = key.partition(":")
        if not separator or not colon:
            raise ContractError(
                "formal receipt mapping must be UNIVERSE:INFORMATION_SET=/absolute/path"
            )
        identity = (UniverseClass(universe_name.upper()), InformationSet(information_name.upper()))
        if identity in result:
            raise ContractError(f"duplicate formal receipt mapping: {key}")
        result[identity] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate evidence-bound server-only tabular V0/V1 jobs."
    )
    parser.add_argument("--phase", choices=("V0", "V1"), required=True)
    parser.add_argument("--d0-manifest", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--environment-receipt", action="append", default=[], required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--model-config", action="append", default=[], required=True)
    parser.add_argument("--layout-config", type=Path, required=True)
    parser.add_argument(
        "--formal-feature-receipt",
        action="append",
        default=[],
        help="Existing D0-anchored UNIVERSE:A0=/absolute/path receipt; repeat as needed.",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--approved-root", type=Path, default=APPROVED_SERVER_ROOT)
    args = parser.parse_args(argv)
    try:
        generated = build_tabular_jobs(
            phase=args.phase,
            d0_manifest=args.d0_manifest,
            canonical_root=args.canonical_root,
            environment_receipts=_model_paths(args.environment_receipt),
            code_receipt=args.code_receipt,
            model_configs=_model_paths(args.model_config),
            layout_config=args.layout_config,
            formal_feature_receipts=_formal_paths(args.formal_feature_receipt),
            output_root=args.run_root,
            approved_root=args.approved_root,
        )
        written = write_tabular_jobs(
            generated,
            phase=args.phase,
            job_root=args.job_root,
            queue_root=args.queue_root,
            approved_root=args.approved_root,
        )
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "tabular_job_generation_failure_v1",
                    "state": "INVALID_PROTOCOL",
                    "reason_code": "JOB_GENERATION_REJECTED",
                    "detail": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": "tabular_job_generation_success_v1",
                "phase": args.phase,
                "planned_cell_count": len(generated.jobs) + len(generated.blocked_cells),
                "runnable_job_count": len(generated.jobs),
                "blocked_cell_count": len(generated.blocked_cells),
                "queue_count": len(generated.queues),
                "written": [path.as_posix() for path in written],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
