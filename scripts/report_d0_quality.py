"""Create a compact D0 coverage/missing/mask report on the server."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from a_share_research.data.manifest import D0Manifest
from a_share_research.protocol import UniverseClass


def _rows(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object in {path}")
                yield value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = D0Manifest.from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
    report: dict[str, object] = {
        "schema_version": "d0_quality_report_v1",
        "d0_manifest_hash": manifest.content_hash,
        "cutoff_date": manifest.cutoff_date.isoformat(),
        "universes": {},
    }
    universe_reports: dict[str, object] = {}
    gates = {gate.universe: gate for gate in manifest.universe_gates}
    for universe in UniverseClass:
        root = args.canonical_root / universe.value.lower()
        membership_rows = 0
        membership_by_date: Counter[object] = Counter()
        for row in _rows(root / "membership.jsonl"):
            membership_rows += 1
            membership_by_date[row.get("asof_date")] += 1

        feature_total: Counter[str] = Counter()
        feature_missing: Counter[str] = Counter()
        monthly_missing: dict[str, Counter[str]] = defaultdict(Counter)
        for row in _rows(root / "features.jsonl"):
            feature_name = str(row.get("feature_name"))
            feature_total[feature_name] += 1
            if bool(row.get("missing_flag")):
                month = str(row.get("asof_date", ""))[:7]
                feature_missing[feature_name] += 1
                monthly_missing[month][feature_name] += 1
        mask_counts: Counter[str] = Counter()
        for row in _rows(root / "masks.jsonl"):
            for name in (
                "member", "observed", "label_available", "buyable",
                "sellable", "loss", "evaluation",
            ):
                values = row.get(name, [])
                if isinstance(values, list):
                    mask_counts[name] += sum(bool(value) for value in values)
        universe_reports[universe.value] = {
            "gate": gates[universe].to_dict(),
            "membership_rows": membership_rows,
            "membership_dates": len(membership_by_date),
            "max_members_per_date": max(membership_by_date.values(), default=0),
            "feature_total": dict(sorted(feature_total.items())),
            "feature_missing": dict(sorted(feature_missing.items())),
            "monthly_missing": {
                month: dict(sorted(counts.items()))
                for month, counts in sorted(monthly_missing.items())
            },
            "mask_true_counts": dict(sorted(mask_counts.items())),
        }
    report["universes"] = universe_reports
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"d0_manifest_hash": manifest.content_hash, "out": args.out.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
