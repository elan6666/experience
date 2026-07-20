"""Generate the bounded CSI300/STAR50 history acquisition manifest."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from a_share_research.data.providers import QueryRequest


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="20190101")
    parser.add_argument("--end", default="20260717")
    args = parser.parse_args()
    start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
    end = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:]))
    requests: list[QueryRequest] = []
    definitions = (
        ("historical_csi300", "000300.SH", 2019),
        ("historical_star50", "000688.SH", 2020),
    )
    for logical_name, index_code, first_year in definitions:
        for year in range(max(start.year, first_year), end.year + 1):
            year_start = max(start, date(year, 1, 1)).strftime("%Y%m%d")
            year_end = min(end, date(year, 12, 31)).strftime("%Y%m%d")
            requests.extend(
                (
                    QueryRequest(
                        endpoint="index_weight",
                        params={
                            "index_code": index_code,
                            "start_date": year_start,
                            "end_date": year_end,
                        },
                        fields=("index_code", "con_code", "trade_date", "weight"),
                        partition_key=f"{logical_name}:index_weight_{year}",
                        min_row_count=50,
                    ),
                    QueryRequest(
                        endpoint="index_daily",
                        params={
                            "ts_code": index_code,
                            "start_date": year_start,
                            "end_date": year_end,
                        },
                        fields=(
                            "ts_code",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "vol",
                            "amount",
                        ),
                        partition_key=f"{logical_name}:index_daily_{year}",
                        min_row_count=50,
                    ),
                )
            )
    manifest = {
        "schema_version": "d0_index_history_requests_v1",
        "start": args.start,
        "end": args.end,
        "bounded_requests": [_request_dict(request) for request in requests],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"request_count": len(requests), "out": args.out.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
