from __future__ import annotations

from pathlib import Path

from a_share_research.data.pipeline import RateLimitedFetcher
from a_share_research.data.providers import ImmutableRawStore, QueryRequest, QueryResult


class FlakyProvider:
    def __init__(self) -> None:
        self.attempts = 0

    def execute(self, request: QueryRequest) -> QueryResult:
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("transient provider failure")
        return QueryResult(request, ({"cal_date": "20250102", "is_open": 1},))


def test_fetcher_retries_bounded_transient_failures(tmp_path: Path) -> None:
    provider = FlakyProvider()
    fetcher = RateLimitedFetcher(
        provider=provider,  # type: ignore[arg-type]
        store=ImmutableRawStore(tmp_path / "raw"),
        checkpoint_path=tmp_path / "checkpoint.json",
        minimum_interval_seconds=0,
        max_attempts=3,
        retry_base_seconds=0,
    )
    request = QueryRequest(
        endpoint="trade_cal",
        params={"exchange": "SSE"},
        fields=("cal_date", "is_open"),
        partition_key="test:calendar",
        min_row_count=1,
    )
    hashes = fetcher.run((request,))
    assert len(hashes) == 1
    assert provider.attempts == 3
