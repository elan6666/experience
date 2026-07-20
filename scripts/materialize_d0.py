"""Audit/plan/fetch D0 raw inputs on the approved server only.

This script does not accept credential arguments and cannot construct a default
Tushare client. It delegates only to the parent server research root's approved
proxy client.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_research.data.inventory import audit_existing_server_data
from a_share_research.data.pipeline import RateLimitedFetcher, plan_incremental_requests
from a_share_research.data.providers import (
    ApprovedProxyProvider,
    ImmutableRawStore,
    QueryRequest,
    assert_private_credential_metadata,
)


def _load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _requests(config: dict[str, object]) -> tuple[QueryRequest, ...]:
    requests = config.get("bounded_requests")
    if not isinstance(requests, list):
        raise ValueError("source config bounded_requests must be a list")
    return tuple(
        QueryRequest(
            endpoint=str(item["endpoint"]),
            params=dict(item["params"]),
            fields=tuple(item["fields"]),
            partition_key=str(item["partition_key"]),
            min_row_count=int(item.get("min_row_count", 0)),
            reject_at_row_count=int(item.get("reject_at_row_count", 5000)),
        )
        for item in requests
        if isinstance(item, dict)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "plan", "fetch"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--request-manifest", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config or args.project_root / "configs" / "data" / "source.yaml"
    config = _load_config(config_path)
    candidates = config.get("existing_candidates")
    inventory = audit_existing_server_data(
        args.research_root,
        dict(candidates) if isinstance(candidates, dict) else None,
    )
    request_config = config
    if args.request_manifest is not None:
        request_config = _load_config(args.request_manifest)
    plan = plan_incremental_requests(inventory, _requests(request_config))
    summary = {
        "schema_version": "d0_acquisition_plan_v1",
        "plan_hash": plan.plan_hash,
        "inventory": [
            {
                "logical_name": item.logical_name,
                "path": item.path.as_posix(),
                "exists": item.exists,
                "manifest_sha256": item.manifest_sha256,
                "verified": item.verified,
                "verification_reasons": list(item.verification_reasons),
            }
            for item in inventory
        ],
        "incremental_request_hashes": [request.request_hash for request in plan.requests],
    }
    if args.mode in {"audit", "plan"}:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not plan.requests:
        raise SystemExit("no verified incremental requests; run audit/plan and generate manifest")
    assert_private_credential_metadata(Path.home() / ".config" / "tushare" / "token")
    provider = ApprovedProxyProvider.from_server_research_root(args.research_root)
    fetcher = RateLimitedFetcher(
        provider=provider,
        store=ImmutableRawStore(args.research_root / "data" / "raw" / "d0_v1"),
        checkpoint_path=args.research_root / "data" / "checkpoints" / "d0_v1.json",
        minimum_interval_seconds=float(config.get("minimum_interval_seconds", 0.5)),
        max_attempts=int(config.get("max_attempts", 3)),
        retry_base_seconds=float(config.get("retry_base_seconds", 2.0)),
    )
    stored = fetcher.run(plan.requests)
    summary["stored_manifest_hashes"] = list(stored)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
