"""Generate D0-anchored A0--A3 formal feature sidecars on the server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from a_share_research.contracts import ContractError
from a_share_research.data.formal_receipts import generate_formal_feature_receipts

APPROVED_SERVER_ROOT = Path("/data/yilangliu/a_share_research")


def _require_server_path(path: Path, root: Path, *, must_exist: bool) -> Path:
    root = root.expanduser().resolve(strict=True)
    resolved = path.expanduser().resolve(strict=must_exist)
    if resolved != root and root not in resolved.parents:
        raise ContractError(f"path leaves approved server root: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-manifest", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path)
    parser.add_argument("--approved-root", type=Path, default=APPROVED_SERVER_ROOT)
    args = parser.parse_args()
    try:
        approved = args.approved_root.expanduser().resolve(strict=True)
        audit = generate_formal_feature_receipts(
            d0_manifest_path=_require_server_path(
                args.d0_manifest, approved, must_exist=True
            ),
            canonical_root=_require_server_path(
                args.canonical_root, approved, must_exist=True
            ),
            feature_schema_path=_require_server_path(
                args.feature_schema, approved, must_exist=True
            ),
            out_dir=_require_server_path(args.out_dir, approved, must_exist=False),
            audit_out=(
                _require_server_path(args.audit_out, approved, must_exist=False)
                if args.audit_out is not None
                else None
            ),
        )
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "formal_feature_generation_failure_v1",
                    "state": "INVALID_DATA",
                    "reason_code": "FORMAL_FEATURE_RECEIPT_REJECTED",
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
                "schema_version": "formal_feature_generation_success_v1",
                "d0_content_hash": audit["d0_content_hash"],
                "universe_decisions": {
                    name: value["decision"]
                    for name, value in audit["universes"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
