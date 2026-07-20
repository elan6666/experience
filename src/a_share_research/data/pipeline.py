"""Server-only acquisition planning: audit first, fetch only verified gaps."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from a_share_research.contracts import ContractError, canonical_hash
from a_share_research.data.inventory import ExistingDataset, missing_datasets
from a_share_research.data.providers import (
    ApprovedProxyProvider,
    ImmutableRawStore,
    QueryRequest,
    QueryResult,
)


@dataclass(frozen=True)
class AcquisitionPlan:
    inventory: tuple[ExistingDataset, ...]
    requests: tuple[QueryRequest, ...]

    @property
    def plan_hash(self) -> str:
        return canonical_hash(
            {
                "inventory": tuple(
                    {
                        "logical_name": item.logical_name,
                        "path": item.path.as_posix(),
                        "exists": item.exists,
                        "manifest_sha256": item.manifest_sha256,
                        "verified": item.verified,
                        "verification_reasons": item.verification_reasons,
                    }
                    for item in self.inventory
                ),
                "request_hashes": tuple(request.request_hash for request in self.requests),
            }
        )


def plan_incremental_requests(
    inventory: tuple[ExistingDataset, ...],
    configured_requests: tuple[QueryRequest, ...],
) -> AcquisitionPlan:
    gaps = set(missing_datasets(inventory))
    selected = tuple(
        request
        for request in configured_requests
        if (
            any(request.partition_key.startswith(f"{logical_name}:") for logical_name in gaps)
            or (
                bool(gaps)
                and request.partition_key.split(":", maxsplit=1)[0]
                in {"market_all", "financial_all", "industry_pit"}
            )
        )
    )
    return AcquisitionPlan(inventory, selected)


class RateLimitedFetcher:
    def __init__(
        self,
        *,
        provider: ApprovedProxyProvider,
        store: ImmutableRawStore,
        checkpoint_path: Path,
        minimum_interval_seconds: float,
        max_attempts: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        if minimum_interval_seconds < 0:
            raise ContractError("rate-limit interval cannot be negative")
        if max_attempts < 1 or retry_base_seconds < 0:
            raise ContractError("provider retry policy is invalid")
        self.provider = provider
        self.store = store
        self.checkpoint_path = checkpoint_path
        self.minimum_interval_seconds = minimum_interval_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds

    def _completed(self) -> set[str]:
        if not self.checkpoint_path.exists():
            return set()
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        completed = payload.get("completed_request_hashes", [])
        if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
            raise ContractError("acquisition checkpoint is malformed")
        return set(completed)

    def _write_checkpoint(self, completed: set[str]) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "d0_acquisition_checkpoint_v1",
                    "completed_request_hashes": sorted(completed),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.checkpoint_path)

    def run(self, requests: tuple[QueryRequest, ...]) -> tuple[str, ...]:
        completed = self._completed()
        stored_hashes: list[str] = []
        for request in requests:
            if request.request_hash in completed:
                continue
            result: QueryResult
            for attempt in range(self.max_attempts):
                try:
                    result = self.provider.execute(request)
                    break
                except Exception:
                    if attempt + 1 == self.max_attempts:
                        raise
                    delay = self.retry_base_seconds * (2**attempt)
                    if delay:
                        time.sleep(delay)
            partition = self.store.store(result)
            stored_hashes.append(partition.manifest_hash)
            completed.add(request.request_hash)
            self._write_checkpoint(completed)
            if self.minimum_interval_seconds:
                time.sleep(self.minimum_interval_seconds)
        return tuple(stored_hashes)
