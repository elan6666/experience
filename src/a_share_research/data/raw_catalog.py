"""Fail-closed reader for the exact raw partitions named by a request manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from a_share_research.contracts import ContractError, canonical_hash
from a_share_research.data.providers import QueryRequest


def _request(item: dict[str, Any]) -> QueryRequest:
    return QueryRequest(
        endpoint=str(item["endpoint"]),
        params=dict(item["params"]),
        fields=tuple(item["fields"]),
        partition_key=str(item["partition_key"]),
        min_row_count=int(item.get("min_row_count", 0)),
        reject_at_row_count=int(item.get("reject_at_row_count", 5000)),
    )


@dataclass(frozen=True)
class RawPartitionEvidence:
    request_hash: str
    manifest_hash: str
    content_hash: str
    row_count: int


class ExactRawCatalog:
    """Expose only content-addressed partitions explicitly listed by one manifest.

    Directory scans are deliberately forbidden.  Consequently, unrelated legacy
    partitions and any ``quarantine`` subtree cannot enter a D0 build.
    """

    def __init__(self, *, raw_root: Path, request_manifest: Path) -> None:
        if "quarantine" in {part.lower() for part in raw_root.parts}:
            raise ContractError("quarantine cannot be used as a canonical raw root")
        document = json.loads(request_manifest.read_text(encoding="utf-8"))
        items = document.get("bounded_requests")
        if not isinstance(items, list) or not items:
            raise ContractError("request manifest must contain bounded_requests")
        requests: list[QueryRequest] = []
        for item in items:
            if not isinstance(item, dict):
                raise ContractError("request manifest contains a non-object request")
            request = _request(item)
            if item.get("request_hash") != request.request_hash:
                raise ContractError("embedded request hash does not match request content")
            requests.append(request)
        hashes = tuple(request.request_hash for request in requests)
        if len(set(hashes)) != len(hashes):
            raise ContractError("request manifest contains duplicate request hashes")
        self.raw_root = raw_root.resolve()
        self.request_manifest = request_manifest.resolve()
        self.document = document
        self.requests = tuple(requests)
        self._request_hashes = frozenset(hashes)
        self._evidence: dict[str, RawPartitionEvidence] = {}

    def matching(
        self,
        endpoint: str,
        *,
        param_name: str | None = None,
        param_value: str | None = None,
    ) -> tuple[QueryRequest, ...]:
        matches = tuple(
            request
            for request in self.requests
            if request.endpoint == endpoint
            and (
                param_name is None
                or str(request.params.get(param_name, "")) == str(param_value)
            )
        )
        return tuple(sorted(matches, key=lambda request: request.request_hash))

    def _paths(self, request: QueryRequest) -> tuple[Path, Path]:
        if request.request_hash not in self._request_hashes:
            raise ContractError("request was not declared by the exact manifest")
        target = (
            self.raw_root
            / request.endpoint
            / request.partition_key
            / request.request_hash
        ).resolve()
        if self.raw_root not in target.parents:
            raise ContractError("raw partition escaped the canonical root")
        if "quarantine" in {part.lower() for part in target.parts}:
            raise ContractError("quarantined raw content is forbidden")
        return target / "manifest.json", target / "rows.jsonl"

    def iter_rows(self, request: QueryRequest) -> Iterator[dict[str, object]]:
        """Validate one bounded partition, then stream its already-bounded rows."""
        manifest_path, rows_path = self._paths(request)
        if not manifest_path.is_file() or not rows_path.is_file():
            raise ContractError(f"missing exact raw partition: {request.request_hash}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("request_hash") != request.request_hash:
            raise ContractError("raw manifest request hash mismatch")
        if manifest.get("endpoint") != request.endpoint:
            raise ContractError("raw manifest endpoint mismatch")
        rows: list[dict[str, object]] = []
        with rows_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ContractError("raw partition contains a non-object row")
                rows.append(row)
        row_count = len(rows)
        if row_count != int(manifest.get("row_count", -1)):
            raise ContractError("raw partition row count mismatch")
        if not request.min_row_count <= row_count < request.reject_at_row_count:
            raise ContractError("raw partition violates request row-count bounds")
        content_hash = canonical_hash(tuple(rows))
        if content_hash != manifest.get("content_hash"):
            raise ContractError("raw partition content hash mismatch")
        stable_manifest = {
            key: value for key, value in manifest.items() if key != "created_at_utc"
        }
        evidence = RawPartitionEvidence(
            request.request_hash,
            canonical_hash(stable_manifest),
            content_hash,
            row_count,
        )
        self._evidence[request.request_hash] = evidence
        yield from rows

    def rows(self, request: QueryRequest) -> tuple[dict[str, object], ...]:
        return tuple(self.iter_rows(request))

    def missing_partition_requests(self) -> tuple[QueryRequest, ...]:
        """Return every declared request whose two immutable files are absent.

        This preflight intentionally checks only file presence.  Content and
        row-count validation remains fail-closed in :meth:`iter_rows`, but a
        D0 operator can now obtain the complete fetch queue instead of fixing
        one missing partition per rebuild attempt.
        """
        missing: list[QueryRequest] = []
        for request in self.requests:
            manifest_path, rows_path = self._paths(request)
            if not manifest_path.is_file() or not rows_path.is_file():
                missing.append(request)
        return tuple(sorted(missing, key=lambda request: request.request_hash))

    def require_all_partitions(self) -> tuple[RawPartitionEvidence, ...]:
        for request in self.requests:
            if request.request_hash not in self._evidence:
                tuple(self.iter_rows(request))
        return tuple(self._evidence[key] for key in sorted(self._evidence))

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(
            tuple(
                {
                    "request_hash": item.request_hash,
                    "manifest_hash": item.manifest_hash,
                    "content_hash": item.content_hash,
                    "row_count": item.row_count,
                }
                for item in sorted(self._evidence.values(), key=lambda value: value.request_hash)
            )
        )
