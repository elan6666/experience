from __future__ import annotations

from pathlib import Path

import pytest

from a_share_research.contracts import ContractError
from a_share_research.data.inventory import ExistingDataset, missing_datasets
from a_share_research.data.pipeline import plan_incremental_requests
from a_share_research.data.providers import (
    ApprovedProxyProvider,
    QueryRequest,
    assert_private_credential_metadata,
)


class FakeFrame:
    def to_dict(self, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return [{"cal_date": "20250102", "is_open": 1}]


class NonFiniteFrame:
    def to_dict(self, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return [{"ts_code": "000001.SZ", "industry": float("nan")}]


class FakeApprovedClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def query(self, api_name: str, fields: str = "", **kwargs: object) -> FakeFrame:
        self.calls.append((api_name, fields, kwargs))
        return FakeFrame()


class NonFiniteClient(FakeApprovedClient):
    def query(self, api_name: str, fields: str = "", **kwargs: object) -> NonFiniteFrame:
        self.calls.append((api_name, fields, kwargs))
        return NonFiniteFrame()


def test_provider_is_injected_and_query_manifest_has_no_credential_surface() -> None:
    client = FakeApprovedClient()
    provider = ApprovedProxyProvider(client)
    request = QueryRequest(
        endpoint="trade_cal",
        params={"exchange": "SSE", "start_date": "20250101", "end_date": "20250131"},
        fields=("cal_date", "is_open"),
        partition_key="historical_csi300:calendar_202501",
    )
    result = provider.execute(request)
    assert result.rows == ({"cal_date": "20250102", "is_open": 1},)
    assert client.calls[0][0] == "trade_cal"
    assert "token" not in repr(request).lower()


@pytest.mark.parametrize("name", ["token", "api_token", "password", "secret_value"])
def test_provider_request_rejects_credential_like_params(name: str) -> None:
    with pytest.raises(ContractError, match="credential-like"):
        QueryRequest("daily", {name: "forbidden"}, ("trade_date",), "gap:one")


def test_existing_data_audit_drives_incremental_gaps_only() -> None:
    inventory = (
        ExistingDataset(
            "historical_csi300", Path("/server/csi300"), True, "a" * 64, True
        ),
        ExistingDataset("historical_star50", Path("/server/star50"), True, None),
        ExistingDataset("tech32", Path("/server/tech32"), False, None),
    )
    assert missing_datasets(inventory) == ("historical_star50", "tech32")


def test_incremental_plan_admits_bounded_industry_pit_requests() -> None:
    inventory = (
        ExistingDataset("tech100", Path("/server/tech100"), False, None),
    )
    request = QueryRequest(
        endpoint="index_member_all",
        params={"ts_code": "000001.SZ", "is_new": "Y"},
        fields=("l1_code", "ts_code", "in_date", "out_date", "is_new"),
        partition_key="industry_pit:index_member_all_000001.SZ_Y",
        min_row_count=1,
        reject_at_row_count=1000,
    )
    plan = plan_incremental_requests(inventory, (request,))
    assert plan.requests == (request,)


def test_provider_fails_closed_at_truncation_boundary() -> None:
    provider = ApprovedProxyProvider(FakeApprovedClient())
    request = QueryRequest(
        endpoint="trade_cal",
        params={"exchange": "SSE"},
        fields=("cal_date", "is_open"),
        partition_key="gap:truncation",
        reject_at_row_count=1,
    )
    with pytest.raises(ContractError, match="truncation"):
        provider.execute(request)


def test_provider_fails_closed_below_minimum_row_count() -> None:
    provider = ApprovedProxyProvider(FakeApprovedClient())
    request = QueryRequest(
        endpoint="trade_cal",
        params={"exchange": "SSE"},
        fields=("cal_date", "is_open"),
        partition_key="gap:minimum",
        min_row_count=2,
    )
    with pytest.raises(ContractError, match="minimum"):
        provider.execute(request)


def test_provider_normalizes_non_finite_table_values_to_missing() -> None:
    provider = ApprovedProxyProvider(NonFiniteClient())
    request = QueryRequest(
        endpoint="stock_basic",
        params={"exchange": "SZSE", "list_status": "L"},
        fields=("ts_code", "industry"),
        partition_key="market_all:stock_basic_SZSE_L",
    )
    result = provider.execute(request)
    assert result.rows == ({"industry": None, "ts_code": "000001.SZ"},)


def test_credential_metadata_check_never_reads_secret(tmp_path: Path) -> None:
    credential = tmp_path / "credential"
    credential.write_text("not-read-by-check", encoding="utf-8")
    credential.chmod(0o600)
    assert_private_credential_metadata(credential)
    credential.chmod(0o644)
    with pytest.raises(ContractError, match="0600"):
        assert_private_credential_metadata(credential)
