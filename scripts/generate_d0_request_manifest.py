"""Generate bounded server request partitions from audited calendar/security evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from a_share_research.data.normalization import parse_provider_date
from a_share_research.data.providers import QueryRequest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _request_dict(request: QueryRequest) -> dict[str, object]:
    return {
        "endpoint": request.endpoint,
        "params": dict(request.params),
        "fields": list(request.fields),
        "partition_key": request.partition_key,
        "min_row_count": request.min_row_count,
        "reject_at_row_count": request.reject_at_row_count,
        "request_hash": request.request_hash,
    }


def _market_requests(code: str, start: str, end: str) -> tuple[QueryRequest, ...]:
    definitions = (
        (
            "daily",
            ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"),
        ),
        (
            "daily_basic",
            (
                "ts_code", "trade_date", "turnover_rate", "pe_ttm", "pb",
                "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
            ),
        ),
        ("adj_factor", ("ts_code", "trade_date", "adj_factor")),
        ("stk_limit", ("ts_code", "trade_date", "up_limit", "down_limit")),
    )
    return tuple(
        QueryRequest(
            endpoint=endpoint,
            params={"ts_code": code, "start_date": start, "end_date": end},
            fields=fields,
            partition_key=f"market_all:{endpoint}_{code}",
        )
        for endpoint, fields in definitions
    )


def _industry_requests(code: str) -> tuple[QueryRequest, ...]:
    """Bound every industry query by permanent stock identity and member state."""
    fields = (
        "l1_code",
        "l1_name",
        "ts_code",
        "in_date",
        "out_date",
        "is_new",
    )
    return tuple(
        QueryRequest(
            endpoint="index_member_all",
            params={"ts_code": code, "is_new": status},
            fields=fields,
            partition_key=f"industry_pit:index_member_all_{code}_{status}",
            min_row_count=1 if status == "Y" else 0,
            reject_at_row_count=1000,
        )
        for status in ("Y", "N")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-calendar-jsonl", type=Path, required=True)
    parser.add_argument("--security-master-jsonl", type=Path, required=True)
    parser.add_argument("--universe-codes-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="20190101")
    parser.add_argument("--end", default="20260717")
    args = parser.parse_args()
    start = parse_provider_date(args.start)
    end = parse_provider_date(args.end)
    calendar_rows = _jsonl(args.trade_calendar_jsonl)
    security_rows = _jsonl(args.security_master_jsonl)
    universe_codes_payload = json.loads(args.universe_codes_json.read_text(encoding="utf-8"))
    if not isinstance(universe_codes_payload, list):
        raise SystemExit("universe codes must be a JSON list")
    universe_codes = {str(code) for code in universe_codes_payload}
    trading_days = tuple(
        parse_provider_date(row["cal_date"])
        for row in calendar_rows
        if int(row.get("is_open", 0)) == 1
        and start <= parse_provider_date(row["cal_date"]) <= end
    )
    security_codes = {str(row["ts_code"]) for row in security_rows}
    missing_identities = universe_codes - security_codes
    if missing_identities:
        raise SystemExit("universe code list contains identities absent from security master")
    codes = tuple(sorted(universe_codes))
    if not trading_days or not codes:
        raise SystemExit("audited calendar and security master must be non-empty")
    requests: list[QueryRequest] = []
    for code in codes:
        requests.extend(_market_requests(code, args.start, args.end))
        requests.extend(_industry_requests(code))
        requests.extend(
            (
                QueryRequest(
                    "fina_indicator",
                    {"ts_code": code, "start_date": args.start, "end_date": args.end},
                    (
                        "ts_code", "ann_date", "end_date", "roe", "roa",
                        "grossprofit_margin", "debt_to_assets", "current_ratio", "or_yoy",
                        "netprofit_yoy", "ocf_to_or", "assets_turn",
                    ),
                    f"financial_all:fina_indicator_{code}",
                ),
                QueryRequest(
                    "namechange",
                    {"ts_code": code, "start_date": args.start, "end_date": args.end},
                    ("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"),
                    f"market_all:namechange_{code}",
                ),
            )
        )
    for year in range(start.year, end.year + 1):
        year_start = max(start, date(year, 1, 1)).strftime("%Y%m%d")
        year_end = min(end, date(year, 12, 31)).strftime("%Y%m%d")
        for logical_name, index_code in (
            ("historical_csi300", "000300.SH"),
            ("historical_star50", "000688.SH"),
        ):
            requests.extend(
                (
                    QueryRequest(
                        "index_weight",
                        {
                            "index_code": index_code,
                            "start_date": year_start,
                            "end_date": year_end,
                        },
                        ("index_code", "con_code", "trade_date", "weight"),
                        f"{logical_name}:index_weight_{year}",
                    ),
                    QueryRequest(
                        "index_daily",
                        {
                            "ts_code": index_code,
                            "start_date": year_start,
                            "end_date": year_end,
                        },
                        (
                            "ts_code", "trade_date", "open", "high", "low",
                            "close", "vol", "amount",
                        ),
                        f"{logical_name}:index_daily_{year}",
                    ),
                )
            )
        for month in range(1, 13):
            month_start = max(start, date(year, month, 1))
            next_year = year + 1 if month == 12 else year
            next_month_number = 1 if month == 12 else month + 1
            next_month = date(next_year, next_month_number, 1)
            month_end = min(end, date.fromordinal(next_month.toordinal() - 1))
            if month_start > month_end:
                continue
            requests.append(
                QueryRequest(
                    "suspend_d",
                    {
                        "start_date": month_start.strftime("%Y%m%d"),
                        "end_date": month_end.strftime("%Y%m%d"),
                    },
                    ("ts_code", "trade_date", "suspend_timing", "suspend_type"),
                    f"market_all:suspend_d_{year}_{month:02d}",
                )
            )
    manifest = {
        "schema_version": "d0_bounded_request_manifest_v2_pit_industry",
        "calendar_sha256": _sha256(args.trade_calendar_jsonl),
        "security_master_sha256": _sha256(args.security_master_jsonl),
        "universe_codes_sha256": _sha256(args.universe_codes_json),
        "start": args.start,
        "end": args.end,
        "bounded_requests": [_request_dict(request) for request in requests],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"request_count": len(requests), "manifest": args.out.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
