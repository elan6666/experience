"""Validate an already materialized D0 manifest without exposing raw rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_research.data.manifest import D0Manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = D0Manifest.from_dict(payload)
    compact = {
        "dataset_id": manifest.dataset_id,
        "cutoff_date": manifest.cutoff_date.isoformat(),
        "manifest_hash": manifest.content_hash,
        "market_state_hash": manifest.market_state_hash,
        "universe_gates": {
            gate.universe.value: {
                "status": gate.status.value,
                "membership_coverage": gate.membership_coverage,
                "core_coverage": gate.core_coverage,
                "warnings": list(gate.warnings),
            }
            for gate in manifest.universe_gates
        },
        "provider_transport_notice": manifest.provider_transport_notice,
    }
    print(json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
