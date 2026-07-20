"""Immutable, resumable raw partition storage for server-side provider calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from a_share_research.contracts import ContractError, canonical_hash
from a_share_research.data.providers.approved_proxy import QueryResult


@dataclass(frozen=True)
class StoredPartition:
    partition_dir: Path
    manifest_hash: str
    row_count: int


class ImmutableRawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def store(self, result: QueryResult) -> StoredPartition:
        request = result.request
        target = self.root / request.endpoint / request.partition_key / request.request_hash
        data_path = target / "rows.jsonl"
        manifest_path = target / "manifest.json"
        manifest = {
            "schema_version": "d0_raw_partition_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": request.endpoint,
            "partition_key": request.partition_key,
            "request_hash": request.request_hash,
            "params": dict(request.params),
            "fields": list(request.fields),
            "min_row_count": request.min_row_count,
            "content_hash": result.content_hash,
            "row_count": len(result.rows),
        }
        stable_manifest = {key: value for key, value in manifest.items() if key != "created_at_utc"}
        manifest_hash = canonical_hash(stable_manifest)
        if target.exists():
            if not data_path.is_file() or not manifest_path.is_file():
                raise ContractError("partial immutable raw partition exists")
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_stable = {
                key: value for key, value in existing.items() if key != "created_at_utc"
            }
            if canonical_hash(existing_stable) != manifest_hash:
                raise ContractError("immutable raw partition conflicts with existing evidence")
            existing_rows = tuple(
                json.loads(line)
                for line in data_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if canonical_hash(existing_rows) != existing.get("content_hash"):
                raise ContractError("immutable raw partition content hash mismatch")
            return StoredPartition(target, manifest_hash, int(existing["row_count"]))
        target.mkdir(parents=True, exist_ok=False)
        temporary_data = data_path.with_suffix(".tmp")
        with temporary_data.open("w", encoding="utf-8") as handle:
            for row in result.rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(temporary_data, data_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return StoredPartition(target, manifest_hash, len(result.rows))
