#!/usr/bin/env python3
"""Install the previously approved proxy client at the server-root contract path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

APPROVED_ROOT = Path("/data/yilangliu/a_share_research")
SOURCE_RELATIVE = Path("star50_tech32_strategy/code/scripts/tushare_proxy_client.py")
DESTINATION_RELATIVE = Path("scripts/tushare_proxy_client.py")
EXPECTED_SOURCE_SHA256 = "04742eabccd7fdecf5ebd3da3ef6db94c1eb0bf22dfe3c16e6bfb98fe66e0ee4"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.research_root.resolve()
    if root != APPROVED_ROOT:
        raise ValueError(f"research root must be {APPROVED_ROOT}")
    receipt = args.receipt.resolve()
    if APPROVED_ROOT not in receipt.parents:
        raise ValueError("receipt must remain under the approved server root")

    source = root / SOURCE_RELATIVE
    destination = root / DESTINATION_RELATIVE
    payload = source.read_bytes()
    source_hash = _sha256(payload)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("approved proxy client source hash changed; manual review required")
    if destination.exists() and _sha256(destination.read_bytes()) != source_hash:
        raise RuntimeError("destination contains a different proxy client")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.chmod(temporary, 0o644)
        temporary.replace(destination)

    record = {
        "schema_version": "approved_proxy_client_install_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source.as_posix(),
        "source_sha256": source_hash,
        "destination": destination.as_posix(),
        "destination_sha256": _sha256(destination.read_bytes()),
        "credential_read": False,
        "status": "PASS",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt.with_suffix(".tmp")
    temporary_receipt.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_receipt.replace(receipt)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
