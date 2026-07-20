"""Compile the fail-closed D0 gate manifest from canonical server JSONL tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from a_share_research.contracts import PITFeature, UniverseMembership
from a_share_research.data.labels import CompactLabel
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.data.normalization import parse_provider_date
from a_share_research.features.schema import d0_features
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calendar(path: Path, cutoff: date) -> tuple[date, ...]:
    return tuple(
        sorted(
            {
                parse_provider_date(row["cal_date"])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
                for row in (json.loads(line),)
                if int(row.get("is_open", 0)) == 1
                and date(2019, 1, 1) <= parse_provider_date(row["cal_date"]) <= cutoff
            }
        )
    )


def _stream_gate(
    *,
    universe: UniverseClass,
    root: Path,
    calendar: tuple[date, ...],
    expected_member_dates: int,
    expected_core_values: int,
    star50_history_complete: bool,
) -> UniverseGate:
    duplicates = 0
    pit_violations = 0
    label_violations = 0
    feature_schema_violations = 0
    membership_dates: set[date] = set()
    current_day: date | None = None
    member_codes: set[str] = set()
    with (root / "membership.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = UniverseMembership.from_dict(json.loads(line))
            if current_day != row.asof_date:
                current_day = row.asof_date
                member_codes = set()
            if row.ts_code in member_codes:
                duplicates += 1
            member_codes.add(row.ts_code)
            membership_dates.add(row.asof_date)

    expected_names = {item.name for item in d0_features()}
    core_present = 0
    current_feature_key: tuple[date, str] | None = None
    feature_names: set[str] = set()

    def finish_feature_key() -> None:
        nonlocal feature_schema_violations
        if current_feature_key is not None:
            feature_schema_violations += len(expected_names.symmetric_difference(feature_names))

    with (root / "features.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = PITFeature.from_dict(json.loads(line))
            key = (row.asof_date, row.ts_code)
            if key != current_feature_key:
                finish_feature_key()
                current_feature_key = key
                feature_names = set()
            if row.feature_name in feature_names:
                duplicates += 1
            feature_names.add(row.feature_name)
            if row.feature_group.value == "CORE" and not row.missing_flag:
                core_present += 1
            if row.availability_time > row.signal_cutoff_time:
                pit_violations += 1
            if row.source_date > row.signal_cutoff_time.date():
                pit_violations += 1
            if row.announce_time is not None and row.announce_time > row.availability_time:
                pit_violations += 1
            if row.missing_flag != (row.value is None):
                pit_violations += 1
    finish_feature_key()

    current_label_key: tuple[date, str] | None = None
    horizons: set[int] = set()
    with (root / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = CompactLabel.from_dict(json.loads(line))
            key = (row.signal_date, row.ts_code)
            if key != current_label_key:
                current_label_key = key
                horizons = set()
            if row.horizon in horizons:
                duplicates += 1
            horizons.add(row.horizon)
            try:
                row.verify_calendar(calendar)
            except ValueError:
                label_violations += 1

    membership_coverage = min(1.0, len(membership_dates) / max(expected_member_dates, 1))
    core_coverage = min(1.0, core_present / max(expected_core_values, 1))
    if membership_dates:
        first_membership = min(membership_dates)
        expected_after_first = {day for day in calendar if day >= first_membership}
        leading_edge_only = membership_dates == expected_after_first
    else:
        first_membership = None
        leading_edge_only = False
    warnings: list[str] = []
    if duplicates or pit_violations or label_violations or feature_schema_violations:
        status = ResultState.INVALID_DATA
    elif universe is UniverseClass.STAR50 and not star50_history_complete:
        status = ResultState.BLOCKED
        warnings.append("official STAR50 historical membership is incomplete")
    elif universe in {UniverseClass.TECH32, UniverseClass.TECH100}:
        status = ResultState.EXPLORATORY_ONLY
        warnings.append("2026-selected universe; conditional-selection bias cannot be removed")
        if core_coverage < 0.995:
            warnings.append(
                "pre-listing and unavailable observations reduce masked Core coverage"
            )
    elif core_coverage < 0.995:
        status = ResultState.BLOCKED
        warnings.append("membership/core coverage is below the formal D0 threshold")
    elif membership_coverage < 1.0:
        if universe is UniverseClass.CSI300 and leading_edge_only:
            status = ResultState.PASS_WITH_WARNING
            warnings.append(
                "formal history is left-truncated before the first official CSI300 "
                f"snapshot on {first_membership.isoformat()}"
            )
        else:
            status = ResultState.BLOCKED
            warnings.append("membership/core coverage is below the formal D0 threshold")
    else:
        status = ResultState.PASS
    return UniverseGate(
        universe=universe,
        status=status,
        membership_coverage=membership_coverage,
        core_coverage=core_coverage,
        duplicate_keys=duplicates,
        pit_violations=pit_violations,
        label_boundary_violations=label_violations,
        feature_schema_violations=feature_schema_violations,
        warnings=tuple(warnings),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--raw-manifest-root", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--security-master", type=Path, required=True)
    parser.add_argument("--trading-calendar", type=Path, required=True)
    parser.add_argument("--market-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-07-17")
    parser.add_argument("--star50-history-complete", action="store_true")
    args = parser.parse_args()
    # Never scan the raw root: only exact request hashes admitted by the
    # canonical materialization receipt can enter the D0 manifest.
    materialization = json.loads(args.materialization_receipt.read_text(encoding="utf-8"))
    raw_partitions = materialization.get("raw_partitions")
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise SystemExit("materialization receipt lacks exact raw partition evidence")
    raw_hashes = {
        str(item["request_hash"]): str(item["content_hash"])
        for item in raw_partitions
        if isinstance(item, dict)
    }
    if len(raw_hashes) != len(raw_partitions):
        raise SystemExit("duplicate or malformed raw partition evidence")
    canonical_hashes: dict[str, str] = {}
    canonical_hashes["materialization_receipt.json"] = _sha256(
        args.materialization_receipt
    )
    canonical_hashes["shared_market_state.jsonl"] = _sha256(args.market_state)
    for path in sorted((args.canonical_root / "common" / "daily_market").glob("*.jsonl")):
        canonical_hashes[
            path.relative_to(args.canonical_root).as_posix()
        ] = _sha256(path)
    gates = []
    trading_calendar = _calendar(args.trading_calendar, date.fromisoformat(args.cutoff))
    for universe in UniverseClass:
        root = args.canonical_root / universe.value.lower()
        for filename in (
            "membership.jsonl", "features.jsonl", "labels.jsonl",
            "masks.jsonl", "coverage.json",
        ):
            path = root / filename
            if not path.is_file():
                raise SystemExit(f"missing canonical D0 table: {path}")
            canonical_hashes[f"{universe.value.lower()}/{filename}"] = _sha256(path)
        metadata = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
        gates.append(
            _stream_gate(
                universe=universe,
                root=root,
                calendar=trading_calendar,
                expected_member_dates=int(metadata["expected_member_dates"]),
                expected_core_values=int(metadata["expected_core_values"]),
                star50_history_complete=(
                    args.star50_history_complete if universe is UniverseClass.STAR50 else True
                ),
            )
        )
    manifest = D0Manifest(
        dataset_id=f"d0-v1-{args.cutoff}",
        created_at_utc=datetime.now(timezone.utc),
        cutoff_date=date.fromisoformat(args.cutoff),
        raw_snapshot_hashes=raw_hashes,
        canonical_table_hashes=canonical_hashes,
        security_master_hash=_sha256(args.security_master),
        trading_calendar_hash=_sha256(args.trading_calendar),
        feature_schema_hash=_sha256(args.feature_schema),
        market_state_hash=str(materialization["shared_market_state_hash"]),
        universe_gates=tuple(gates),
        provider_transport_notice=(
            "Tutorial-compatible provider proxy uses plain HTTP; credential remains server-only."
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest_hash": manifest.content_hash, "out": args.out.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
