"""Server-only entry point for restartable canonical D0 materialization."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from a_share_research.data.canonical import CanonicalD0Materializer, CanonicalInputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-07-17")
    args = parser.parse_args()
    staged = args.staged_root
    materializer = CanonicalD0Materializer(
        raw_root=args.research_root / "data/raw/d0_v1",
        output_root=args.out,
        inputs=CanonicalInputs(
            request_manifest=args.request_manifest,
            staged_calendar=staged / "trade_calendar.jsonl",
            staged_security_master=staged / "security_master.jsonl",
            union_codes=staged / "four_universe_union_codes.json",
            csi300_codes=staged / "csi300_union_codes.json",
            star50_codes=staged / "star50_union_codes.json",
            tech32_codes=staged / "tech32_codes.json",
            tech90_codes=staged / "tech90_codes.json",
        ),
        cutoff_date=date.fromisoformat(args.cutoff),
    )
    receipt = materializer.run()
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
