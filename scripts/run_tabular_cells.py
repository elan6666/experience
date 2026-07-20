"""Server-only sealed entry point for one tabular cell or a bounded CPU queue."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from a_share_research.contracts import ContractError
from a_share_research.experiments.tabular_runner import (
    TabularCellRunner,
    TabularJobSpec,
    TabularQueueManifest,
    TabularRunFailure,
    run_cpu_queue,
)


def _payload(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run sealed Ridge/LightGBM V0/V1 jobs on the approved server."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-spec", type=Path)
    source.add_argument("--queue-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.job_spec is not None:
            job = TabularJobSpec.from_dict(_payload(args.job_spec))
            outputs = (TabularCellRunner().run(job),)
        else:
            queue = TabularQueueManifest.from_dict(_payload(args.queue_manifest))
            outputs = run_cpu_queue(queue)
    except TabularRunFailure as error:
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        return 2
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "tabular_entry_failure_v1",
                    "state": "INVALID_PROTOCOL",
                    "reason_code": "ENTRY_MANIFEST_REJECTED",
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
                "schema_version": "tabular_entry_success_v1",
                "outputs": [path.as_posix() for path in outputs],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
