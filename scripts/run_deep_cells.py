#!/usr/bin/env python3
"""Server-only sealed entry point for one iTransformer/FACT cell."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from a_share_research.contracts import ContractError
from a_share_research.experiments.deep_runner import (
    DeepCellRunner,
    DeepJobSpec,
    DeepRunFailure,
)


def _payload(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one sealed iTransformer/FACT V0/V1 cell on its frozen GPU."
    )
    parser.add_argument("--job-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        job = DeepJobSpec.from_dict(_payload(args.job_spec))
        output = DeepCellRunner().run(job)
    except DeepRunFailure as error:
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        return 2
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "deep_entry_failure_v1",
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
                "schema_version": "deep_entry_success_v1",
                "output": output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

