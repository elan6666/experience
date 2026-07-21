"""Streaming, restartable construction of the canonical D0 research tables."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

from a_share_research.contracts import (
    AssetRegistry,
    ContractError,
    DailyMarket,
    MarketState,
    PITFeature,
    canonical_hash,
)
from a_share_research.data.eligibility import ExecutionStatus, build_mask_bundle
from a_share_research.data.industry import (
    IndustryInterval,
    build_industry_intervals,
    industry_at,
    industry_by_date,
    numeric_industry_id,
)
from a_share_research.data.labels import build_compact_open_labels
from a_share_research.data.market_state import SharedMarketState, build_shared_market_state
from a_share_research.data.normalization import normalize_daily_market, parse_provider_date
from a_share_research.data.raw_catalog import ExactRawCatalog
from a_share_research.features.availability import signal_cutoff
from a_share_research.features.builders import build_feature_row, per_feature_missing
from a_share_research.features.core import build_core_features
from a_share_research.features.fundamental import build_fundamental_features
from a_share_research.features.schema import InformationClass, d0_features, feature_schema_hash
from a_share_research.protocol import UniverseClass
from a_share_research.universes.builders import (
    MembershipInterval,
    build_dynamic_intervals,
    static_selected_intervals,
)

_UNIVERSE_YEAR_SHARD_SCHEMA = "d0_universe_year_shard_v2_observation_gated_execution"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_line(handle: TextIO, row: object) -> None:
    payload = row.to_dict() if hasattr(row, "to_dict") else row
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[object]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            _write_line(handle, row)
            count += 1
    temporary.replace(path)
    return count


def weekly_signal_dates(trading_dates: tuple[date, ...]) -> tuple[date, ...]:
    """Use the final trading day in each ISO week, without calendar look-ahead in inputs."""
    by_week: dict[tuple[int, int], date] = {}
    for day in trading_dates:
        iso = day.isocalendar()
        by_week[(iso.year, iso.week)] = day
    return tuple(sorted(by_week.values()))


def _unique_rows(
    rows: Iterable[dict[str, object]], key_names: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, object]]:
    result: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in key_names)
        existing = result.get(key)
        if existing is not None and existing != row:
            raise ContractError(f"conflicting raw rows for key {key}")
        result[key] = row
    return result


def _active(interval: MembershipInterval, day: date) -> bool:
    return day >= interval.effective_from and (
        interval.effective_to is None or day <= interval.effective_to
    )


@dataclass(frozen=True)
class CanonicalInputs:
    request_manifest: Path
    staged_calendar: Path
    staged_security_master: Path
    union_codes: Path
    csi300_codes: Path
    star50_codes: Path
    tech32_codes: Path
    tech90_codes: Path


class CanonicalD0Materializer:
    """Build D0 from exact raw request hashes; no directory discovery is performed."""

    def __init__(
        self,
        *,
        raw_root: Path,
        output_root: Path,
        inputs: CanonicalInputs,
        cutoff_date: date,
    ) -> None:
        self.raw_root = raw_root
        self.output_root = output_root
        self.inputs = inputs
        self.cutoff_date = cutoff_date
        self.catalog = ExactRawCatalog(
            raw_root=raw_root,
            request_manifest=inputs.request_manifest,
        )
        for field, path in (
            ("calendar_sha256", inputs.staged_calendar),
            ("security_master_sha256", inputs.staged_security_master),
            ("universe_codes_sha256", inputs.union_codes),
        ):
            if self.catalog.document.get(field) != _sha256(path):
                raise ContractError(f"request manifest input receipt mismatch: {field}")
        self.trading_dates = self._load_calendar()
        self.trading_calendar_hash = canonical_hash(self.trading_dates)
        if len(self.trading_dates) <= 20:
            raise ContractError("D0 calendar does not cover the mandatory 20-day lookback")
        self.signal_dates = weekly_signal_dates(self.trading_dates[20:])
        self.security_master = self._load_jsonl(inputs.staged_security_master)
        self.codes = {
            UniverseClass.CSI300: self._load_codes(inputs.csi300_codes),
            UniverseClass.STAR50: self._load_codes(inputs.star50_codes),
            UniverseClass.TECH32: self._load_codes(inputs.tech32_codes),
            UniverseClass.TECH90: self._load_codes(inputs.tech90_codes),
        }
        self.all_codes = tuple(sorted(set().union(*map(set, self.codes.values()))))
        if self.all_codes != tuple(sorted(self._load_codes(inputs.union_codes))):
            raise ContractError("four-universe union code receipt does not match its components")
        identities = {str(row.get("ts_code", "")) for row in self.security_master}
        missing_identities = set(self.all_codes) - identities
        if missing_identities:
            raise ContractError(
                f"canonical universes contain identities absent from security master: "
                f"{sorted(missing_identities)}"
            )
        self._suspended: set[tuple[str, date]] = set()
        self._industry_intervals: dict[str, tuple[IndustryInterval, ...]] = {}
        self._raw_catalog_hash: str | None = None

    @staticmethod
    def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
        return tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    @staticmethod
    def _load_codes(path: Path) -> tuple[str, ...]:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values:
            raise ContractError(f"universe code list is invalid: {path}")
        codes = tuple(str(value) for value in values)
        if len(codes) != len(set(codes)):
            raise ContractError(f"universe code list contains duplicates: {path}")
        return codes

    def _load_calendar(self) -> tuple[date, ...]:
        rows = self._load_jsonl(self.inputs.staged_calendar)
        dates = tuple(
            sorted(
                {
                    parse_provider_date(row["cal_date"])
                    for row in rows
                    if int(row.get("is_open", 0)) == 1
                    and parse_provider_date(row["cal_date"]) >= date(2019, 1, 1)
                    and parse_provider_date(row["cal_date"]) <= self.cutoff_date
                }
            )
        )
        if not dates:
            raise ContractError("canonical trading calendar is empty")
        return dates

    def _raw_rows(
        self,
        endpoint: str,
        *,
        param_name: str | None = None,
        param_value: str | None = None,
        required: bool = True,
    ) -> tuple[dict[str, object], ...]:
        requests = self.catalog.matching(
            endpoint,
            param_name=param_name,
            param_value=param_value,
        )
        if required and not requests:
            raise ContractError(f"request manifest lacks required {endpoint} partition")
        return tuple(row for request in requests for row in self.catalog.iter_rows(request))

    def _membership_intervals(
        self, universe: UniverseClass
    ) -> tuple[MembershipInterval, ...]:
        if universe in {UniverseClass.TECH32, UniverseClass.TECH90}:
            source_path = (
                self.inputs.tech32_codes
                if universe is UniverseClass.TECH32
                else self.inputs.tech90_codes
            )
            return static_selected_intervals(
                self.codes[universe],
                selection_date=date(2026, 7, 17),
                source=f"{source_path.name};sha256={_sha256(source_path)}",
                research_start=self.trading_dates[0],
            )
        index_code = "000300.SH" if universe is UniverseClass.CSI300 else "000688.SH"
        rows = self._raw_rows("index_weight", param_name="index_code", param_value=index_code)
        snapshots: dict[date, set[str]] = defaultdict(set)
        for row in rows:
            snapshot_date = parse_provider_date(row["trade_date"])
            if snapshot_date <= self.cutoff_date:
                snapshots[snapshot_date].add(str(row["con_code"]))
        if not snapshots:
            raise ContractError(f"{universe.value} has no historical membership snapshots")
        prior = tuple(day for day in snapshots if day <= self.trading_dates[0])
        retained_dates = (
            ((max(prior),) if prior else ())
            + tuple(day for day in sorted(snapshots) if day > self.trading_dates[0])
        )
        immutable = {
            day: tuple(sorted(snapshots[day]))
            for day in retained_dates
        }
        # Never project the first known composition backwards.  Pre-snapshot
        # dates remain uncovered and will therefore block a formal gate.
        covered_calendar = tuple(day for day in self.trading_dates if day >= min(immutable))
        return build_dynamic_intervals(
            immutable,
            covered_calendar,
            source=(
                f"provider:index_weight:{index_code};"
                f"manifest={_sha256(self.inputs.request_manifest)}"
            ),
        )

    def _load_suspensions(self) -> None:
        for row in self._raw_rows("suspend_d"):
            day = parse_provider_date(row["trade_date"])
            if day <= self.cutoff_date:
                self._suspended.add((str(row["ts_code"]), day))

    def _load_industries(self) -> None:
        """Load exact per-stock Y/N partitions; never infer current classifications."""
        for code in self.all_codes:
            requests = self.catalog.matching(
                "index_member_all", param_name="ts_code", param_value=code
            )
            states = {str(request.params.get("is_new", "")) for request in requests}
            if states != {"Y", "N"}:
                raise ContractError(
                    f"{code} requires bounded index_member_all Y and N partitions"
                )
            rows = tuple(row for request in requests for row in self.catalog.iter_rows(request))
            self._industry_intervals[code] = build_industry_intervals(
                rows=rows,
                trading_dates=self.trading_dates,
                expected_code=code,
            )

    def _st_dates(self, code: str) -> set[date]:
        result: set[date] = set()
        for row in self._raw_rows(
            "namechange", param_name="ts_code", param_value=code, required=False
        ):
            name = str(row.get("name", "")).upper()
            if "ST" not in name:
                continue
            start = parse_provider_date(row["start_date"])
            end_value = row.get("end_date")
            end = (
                parse_provider_date(end_value)
                if end_value not in (None, "")
                else self.cutoff_date
            )
            ann_value = row.get("ann_date")
            if ann_value not in (None, ""):
                start = max(start, parse_provider_date(ann_value))
            result.update(day for day in self.trading_dates if start <= day <= end)
        return result

    def _build_market_code(self, code: str) -> tuple[DailyMarket, ...]:
        requests = tuple(
            request.request_hash
            for endpoint in ("daily", "daily_basic", "adj_factor", "stk_limit", "namechange")
            for request in self.catalog.matching(endpoint, param_name="ts_code", param_value=code)
        )
        input_hash = canonical_hash(
            {
                "requests": requests,
                "suspension_requests": tuple(
                    item.request_hash for item in self.catalog.matching("suspend_d")
                ),
                "cutoff": self.cutoff_date,
                "exact_raw_catalog_hash": self._raw_catalog_hash,
            }
        )
        target = self.output_root / "common" / "daily_market" / f"{code}.jsonl"
        receipt_path = target.with_suffix(".receipt.json")
        if target.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("input_hash") == input_hash
                and receipt.get("output_hash") == _sha256(target)
            ):
                return tuple(
                    DailyMarket.from_dict(json.loads(line))
                    for line in target.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        daily = _unique_rows(
            self._raw_rows("daily", param_name="ts_code", param_value=code),
            ("trade_date",),
        )
        basics = _unique_rows(
            self._raw_rows(
                "daily_basic", param_name="ts_code", param_value=code, required=False
            ),
            ("trade_date",),
        )
        adjustments = _unique_rows(
            self._raw_rows("adj_factor", param_name="ts_code", param_value=code),
            ("trade_date",),
        )
        limits = _unique_rows(
            self._raw_rows("stk_limit", param_name="ts_code", param_value=code, required=False),
            ("trade_date",),
        )
        st_dates = self._st_dates(code)
        rows: list[DailyMarket] = []
        for key, source in sorted(daily.items()):
            day = parse_provider_date(source["trade_date"])
            if day > self.cutoff_date:
                continue
            adjustment = adjustments.get(key)
            if adjustment is None:
                raise ContractError(f"missing adjustment evidence for {code} on {day}")
            rows.append(
                normalize_daily_market(
                    source,
                    daily_basic=basics.get(key),
                    adjustment=adjustment,
                    limits=limits.get(key),
                    suspended=(code, day) in self._suspended,
                    st_state=day in st_dates,
                )
            )
        _atomic_jsonl(target, rows)
        _atomic_json(
            receipt_path,
            {
                "schema_version": "d0_market_code_receipt_v1",
                "input_hash": input_hash,
                "output_hash": _sha256(target),
                "row_count": len(rows),
            },
        )
        return tuple(rows)

    def _market_path(self, code: str) -> Path:
        return self.output_root / "common" / "daily_market" / f"{code}.jsonl"

    def _read_market(self, code: str) -> tuple[DailyMarket, ...]:
        path = self._market_path(code)
        return tuple(
            DailyMarket.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def _index_prices(self, code: str) -> tuple[dict[date, float], dict[date, float]]:
        rows = self._raw_rows("index_daily", param_name="ts_code", param_value=code)
        opens: dict[date, float] = {}
        closes: dict[date, float] = {}
        for row in rows:
            day = parse_provider_date(row["trade_date"])
            if day > self.cutoff_date:
                continue
            if day in opens and opens[day] != float(row["open"]):
                raise ContractError("conflicting index daily rows")
            opens[day] = float(row["open"])
            closes[day] = float(row["close"])
        return opens, closes

    def _build_shared_state(
        self, csi_intervals: tuple[MembershipInterval, ...]
    ) -> SharedMarketState:
        active_by_day = {
            day: frozenset(
                interval.ts_code for interval in csi_intervals if _active(interval, day)
            )
            for day in self.trading_dates
        }
        member_returns: dict[date, dict[str, float]] = defaultdict(dict)
        member_amount: dict[date, dict[str, float]] = defaultdict(dict)
        member_turnover: dict[date, dict[str, float]] = defaultdict(dict)
        for code in self.codes[UniverseClass.CSI300]:
            rows = self._read_market(code)
            for index, row in enumerate(rows):
                if code not in active_by_day.get(row.trade_date, frozenset()):
                    continue
                member_amount[row.trade_date][code] = row.amount
                if index > 0 and rows[index - 1].close > 0 and row.close > 0:
                    member_returns[row.trade_date][code] = math.log(
                        row.close / rows[index - 1].close
                    )
                # Daily-basic has no exact publication time: use only the prior
                # observed trading day's value for the current state.
                if index > 0 and rows[index - 1].turnover is not None:
                    member_turnover[row.trade_date][code] = float(rows[index - 1].turnover)
        _, index_close = self._index_prices("000300.SH")
        industries = industry_by_date(
            intervals=(
                interval
                for code in self.codes[UniverseClass.CSI300]
                for interval in self._industry_intervals.get(code, ())
            ),
            trading_dates=self.trading_dates,
            eligible_codes_by_date=active_by_day,
        )
        state = build_shared_market_state(
            trading_dates=self.trading_dates,
            index_close=index_close,
            member_returns=member_returns,
            member_amount=member_amount,
            member_turnover=member_turnover,
            member_industry_by_date=industries,
            eligible_member_codes_by_date=active_by_day,
            source_membership_hash=canonical_hash(csi_intervals),
        )
        _atomic_jsonl(self.output_root / "shared_market_state.jsonl", state.rows)
        _atomic_json(
            self.output_root / "shared_market_state.receipt.json",
            {
                "schema_version": "shared_market_state_receipt_v1",
                "state_hash": state.stable_hash,
                "source_membership_hash": state.source_membership_hash,
                "trading_calendar_hash": state.trading_calendar_hash,
                "industry_dispersion_status": (
                    "AVAILABLE_WHEN_PIT_COVERAGE_SUFFICIENT"
                    if any(item.sufficient for item in state.industry_coverage)
                    else "MISSING_INSUFFICIENT_PIT_COVERAGE"
                ),
                "industry_coverage_threshold": (
                    state.industry_coverage[0].threshold
                    if state.industry_coverage
                    else 0.8
                ),
                "industry_coverage": [
                    {
                        "asof_date": item.asof_date.isoformat(),
                        "active_count": item.active_count,
                        "known_count": item.known_count,
                        "coverage": item.coverage,
                        "sufficient": item.sufficient,
                    }
                    for item in state.industry_coverage
                ],
            },
        )
        return state

    @staticmethod
    def _missing_core(code: str, day: date) -> tuple[PITFeature, ...]:
        cutoff = signal_cutoff(day)
        return tuple(
            build_feature_row(
                definition,
                asof_date=day,
                ts_code=code,
                value=None,
                source_date=day,
                announce_time=None,
                availability_time=cutoff,
                signal_cutoff_time=cutoff,
                source="canonical_raw_market_v1",
                formal_eligible=False,
            )
            for definition in d0_features()
            if definition.information_class is InformationClass.CORE
        )

    @staticmethod
    def _state_features(
        code: str,
        day: date,
        state_by_date: Mapping[date, Mapping[str, MarketState]],
    ) -> tuple[PITFeature, ...]:
        cutoff = signal_cutoff(day)
        source = state_by_date.get(day, {})
        rows: list[PITFeature] = []
        for definition in d0_features():
            if definition.information_class is not InformationClass.S:
                continue
            state_name = definition.source_field
            state_row = source.get(state_name)
            rows.append(
                build_feature_row(
                    definition,
                    asof_date=day,
                    ts_code=code,
                    value=state_row.value if state_row is not None else None,
                    source_date=day,
                    announce_time=None,
                    availability_time=cutoff,
                    signal_cutoff_time=cutoff,
                    source="shared_csi300_market_state_v1",
                    formal_eligible=state_row is not None,
                )
            )
        return tuple(rows)

    def _fundamental_sources(
        self, code: str
    ) -> tuple[
        dict[date, dict[str, object]],
        tuple[dict[str, object], ...],
        tuple[IndustryInterval, ...],
    ]:
        basics = {
            parse_provider_date(row["trade_date"]): row
            for row in self._raw_rows(
                "daily_basic", param_name="ts_code", param_value=code, required=False
            )
        }
        financial = tuple(
            sorted(
                (
                    row
                    for row in self._raw_rows(
                        "fina_indicator",
                        param_name="ts_code",
                        param_value=code,
                        required=False,
                    )
                    if row.get("ann_date") not in (None, "")
                ),
                key=lambda row: (
                    parse_provider_date(row["ann_date"]),
                    str(row.get("end_date", "")),
                ),
            )
        )
        return basics, financial, self._industry_intervals.get(code, ())

    def _write_membership(
        self,
        root: Path,
        universe: UniverseClass,
        intervals: tuple[MembershipInterval, ...],
    ) -> int:
        def rows() -> Iterable[dict[str, object]]:
            for day in self.trading_dates:
                for interval in intervals:
                    if _active(interval, day):
                        yield {
                            "_schema": "universe_membership",
                            "_version": "1.0",
                            "asof_date": day.isoformat(),
                            "ts_code": interval.ts_code,
                            "universe": universe.value,
                            "effective_from": interval.effective_from.isoformat(),
                            "effective_to": (
                                interval.effective_to.isoformat()
                                if interval.effective_to is not None
                                else None
                            ),
                            "source": interval.source,
                        }

        return _atomic_jsonl(root / "membership.jsonl", rows())

    def _materialize_universe_year(
        self,
        *,
        universe: UniverseClass,
        intervals: tuple[MembershipInterval, ...],
        year: int,
        state: SharedMarketState,
        benchmark_opens: Mapping[date, float],
        market_by_date: Mapping[str, Mapping[date, DailyMarket]],
        fundamentals: Mapping[
            str,
            tuple[
                dict[date, dict[str, object]],
                tuple[dict[str, object], ...],
                tuple[IndustryInterval, ...],
            ],
        ],
        opens_by_code: Mapping[str, Mapping[date, float]],
    ) -> Path:
        shard = self.output_root / universe.value.lower() / "_shards" / str(year)
        shard_input_hash = canonical_hash(
            {
                "shard_schema": _UNIVERSE_YEAR_SHARD_SCHEMA,
                "universe": universe.value,
                "year": year,
                "raw_manifest": _sha256(self.inputs.request_manifest),
                "exact_raw_catalog_hash": self._raw_catalog_hash,
                "state_hash": state.stable_hash,
                "feature_schema": feature_schema_hash(),
                "cutoff": self.cutoff_date,
            }
        )
        receipt_path = shard / "receipt.json"
        outputs = ("features.jsonl", "labels.jsonl", "masks.jsonl")
        if receipt_path.is_file() and all((shard / name).is_file() for name in outputs):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("input_hash") == shard_input_hash and all(
                receipt.get("outputs", {}).get(name) == _sha256(shard / name)
                for name in outputs
            ):
                return shard
        shard.mkdir(parents=True, exist_ok=True)
        temporary_paths = {name: shard / f"{name}.tmp" for name in outputs}
        handles = {name: path.open("w", encoding="utf-8") for name, path in temporary_paths.items()}
        counts = defaultdict(int)
        try:
            state_by_date: dict[date, dict[str, MarketState]] = defaultdict(dict)
            for row in state.rows:
                state_by_date[row.asof_date][row.feature_name] = row
            year_signals = tuple(day for day in self.signal_dates if day.year == year)
            first_seen = {
                code: min(
                    interval.effective_from
                    for interval in intervals
                    if interval.ts_code == code
                )
                for code in {item.ts_code for item in intervals}
            }
            registry_order = tuple(sorted(first_seen, key=lambda code: (first_seen[code], code)))
            calendar_index = {day: index for index, day in enumerate(self.trading_dates)}
            expected_names = tuple(item.name for item in d0_features())
            for day in year_signals:
                active = {
                    interval.ts_code for interval in intervals if _active(interval, day)
                }
                known_codes = tuple(code for code in registry_order if first_seen[code] <= day)
                if not known_codes:
                    continue
                registry = AssetRegistry(known_codes)
                signal_index = calendar_index[day]
                next_day = (
                    self.trading_dates[signal_index + 1]
                    if signal_index + 1 < len(self.trading_dates)
                    else None
                )
                missing_by_feature = {name: {} for name in expected_names}
                label_available: dict[str, bool] = {}
                for code in sorted(active):
                    market_map = market_by_date.get(code, {})
                    history = tuple(
                        market_map[candidate]
                        for candidate in self.trading_dates[
                            max(0, signal_index - 20) : signal_index + 1
                        ]
                        if candidate in market_map
                    )
                    if history and history[-1].trade_date == day:
                        core = build_core_features(history, signal_date=day)
                    else:
                        core = self._missing_core(code, day)
                    basics, financial_rows, industry_rows = fundamentals.get(
                        code, ({}, (), ())
                    )
                    basic_day = next(
                        (
                            candidate
                            for candidate in reversed(self.trading_dates[:signal_index])
                            if candidate in basics
                        ),
                        None,
                    )
                    eligible_financial = tuple(
                        row
                        for row in financial_rows
                        if parse_provider_date(row["ann_date"]) < day
                    )
                    finance = eligible_financial[-1] if eligible_financial else None
                    finance_date = (
                        parse_provider_date(finance["ann_date"]) if finance is not None else None
                    )
                    industry = industry_at(industry_rows, day)
                    factors = build_fundamental_features(
                        asof_date=day,
                        ts_code=code,
                        trading_dates=self.trading_dates,
                        daily_basic=basics.get(basic_day) if basic_day is not None else None,
                        daily_basic_source_date=basic_day,
                        financial=finance,
                        financial_announcement_date=finance_date,
                        financial_announcement_time=None,
                        daily_basic_publish_time=None,
                        industry_id=(
                            numeric_industry_id(industry.industry_id)
                            if industry is not None
                            else None
                        ),
                        industry_effective_date=(
                            industry.availability_date if industry is not None else None
                        ),
                        industry_source_date=(
                            industry.in_date if industry is not None else None
                        ),
                    )
                    feature_rows = core + factors + self._state_features(code, day, state_by_date)
                    feature_missing = per_feature_missing(
                        feature_rows,
                        expected_features=expected_names,
                    )
                    for row in feature_rows:
                        _write_line(handles["features.jsonl"], row)
                        counts["features"] += 1
                    for name, value in feature_missing.items():
                        missing_by_feature[name][code] = value
                    labels = build_compact_open_labels(
                        ts_code=code,
                        signal_index=signal_index,
                        trading_calendar=self.trading_dates,
                        trading_calendar_hash=self.trading_calendar_hash,
                        opens=opens_by_code.get(code, {}),
                        benchmark_opens=benchmark_opens,
                    )
                    for label in labels:
                        _write_line(handles["labels.jsonl"], label)
                        counts["labels"] += 1
                    label_available[code] = any(label.horizon == 5 for label in labels)
                observation_statuses = {
                    code: self._execution_status(market_by_date.get(code, {}).get(day))
                    for code in known_codes
                }
                execution_statuses = {
                    code: self._execution_status(
                        market_by_date.get(code, {}).get(next_day) if next_day is not None else None
                    )
                    for code in known_codes
                }
                masks = build_mask_bundle(
                    signal_date=day,
                    asset_registry=registry,
                    member={code: code in active for code in known_codes},
                    statuses=observation_statuses,
                    execution_statuses=execution_statuses,
                    feature_missing=missing_by_feature,
                    label_available=label_available,
                )
                _write_line(handles["masks.jsonl"], masks)
                counts["masks"] += 1
        finally:
            for handle in handles.values():
                handle.close()
        for name, temporary in temporary_paths.items():
            temporary.replace(shard / name)
        _atomic_json(
            receipt_path,
            {
                "schema_version": _UNIVERSE_YEAR_SHARD_SCHEMA,
                "input_hash": shard_input_hash,
                "counts": dict(counts),
                "outputs": {name: _sha256(shard / name) for name in outputs},
            },
        )
        return shard

    @staticmethod
    def _execution_status(row: DailyMarket | None) -> ExecutionStatus:
        if row is None:
            return ExecutionStatus(False, False, None, None, None)
        return ExecutionStatus(
            observed=True,
            suspended_at_open=row.suspended,
            open_price=row.open,
            up_limit=row.up_limit,
            down_limit=row.down_limit,
        )

    @staticmethod
    def _join_shards(root: Path, shards: tuple[Path, ...], filename: str) -> None:
        target = root / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as output:
            for shard in shards:
                with (shard / filename).open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
        temporary.replace(target)

    @staticmethod
    def _missing_counts(path: Path) -> tuple[dict[str, int], dict[str, int]]:
        totals: dict[str, int] = defaultdict(int)
        missing: dict[str, int] = defaultdict(int)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = str(row["feature_name"])
                totals[name] += 1
                if bool(row["missing_flag"]):
                    missing[name] += 1
        return dict(sorted(totals.items())), dict(sorted(missing.items()))

    def run(self) -> dict[str, object]:
        raw_evidence = self.catalog.require_all_partitions()
        self._raw_catalog_hash = self.catalog.manifest_hash
        self._load_suspensions()
        self._load_industries()
        intervals = {universe: self._membership_intervals(universe) for universe in UniverseClass}
        for code in self.all_codes:
            self._build_market_code(code)
        state = self._build_shared_state(intervals[UniverseClass.CSI300])
        benchmark_opens, _ = self._index_prices("000300.SH")
        universe_receipts: dict[str, object] = {}
        years = tuple(sorted({day.year for day in self.signal_dates}))
        core_names = tuple(
            item.name for item in d0_features() if item.information_class is InformationClass.CORE
        )
        for universe in UniverseClass:
            root = self.output_root / universe.value.lower()
            membership_count = self._write_membership(root, universe, intervals[universe])
            universe_codes = tuple(sorted({item.ts_code for item in intervals[universe]}))
            market_by_date = {
                code: {row.trade_date: row for row in self._read_market(code)}
                for code in universe_codes
            }
            fundamentals = {
                code: self._fundamental_sources(code) for code in universe_codes
            }
            opens_by_code = {
                code: {
                    day: row.open
                    for day, row in rows.items()
                    if row.open > 0
                }
                for code, rows in market_by_date.items()
            }
            shards = tuple(
                self._materialize_universe_year(
                    universe=universe,
                    intervals=intervals[universe],
                    year=year,
                    state=state,
                    benchmark_opens=benchmark_opens,
                    market_by_date=market_by_date,
                    fundamentals=fundamentals,
                    opens_by_code=opens_by_code,
                )
                for year in years
            )
            for filename in ("features.jsonl", "labels.jsonl", "masks.jsonl"):
                self._join_shards(root, shards, filename)
            feature_totals, feature_missing = self._missing_counts(root / "features.jsonl")
            member_signal_count = sum(
                1
                for day in self.signal_dates
                for interval in intervals[universe]
                if _active(interval, day)
            )
            coverage = {
                "schema_version": "d0_coverage_input_v1",
                "universe": universe.value,
                "formal_status": (
                    "EXPLORATORY_ONLY"
                    if universe in {UniverseClass.TECH32, UniverseClass.TECH90}
                    else "PENDING_GATE"
                ),
                "expected_member_dates": len(self.trading_dates),
                "expected_core_values": member_signal_count * len(core_names),
                "membership_rows": membership_count,
                "signal_dates": len(self.signal_dates),
                "feature_schema_hash": feature_schema_hash(),
                "shared_market_state_hash": state.stable_hash,
                "request_manifest_sha256": _sha256(self.inputs.request_manifest),
                "industry_evidence": "PIT_SW_L1_INDEX_MEMBER_ALL_Y_N",
                "per_factor_missing_required": True,
                "feature_row_counts": feature_totals,
                "feature_missing_counts": feature_missing,
            }
            _atomic_json(root / "coverage.json", coverage)
            universe_receipts[universe.value] = {
                "coverage_hash": _sha256(root / "coverage.json"),
                "membership_hash": _sha256(root / "membership.jsonl"),
                "features_hash": _sha256(root / "features.jsonl"),
                "labels_hash": _sha256(root / "labels.jsonl"),
                "masks_hash": _sha256(root / "masks.jsonl"),
            }
        receipt = {
            "schema_version": "canonical_d0_materialization_v1",
            "cutoff_date": self.cutoff_date.isoformat(),
            "request_manifest_sha256": _sha256(self.inputs.request_manifest),
            "exact_raw_catalog_hash": self._raw_catalog_hash,
            "raw_partition_count": len(raw_evidence),
            "raw_partitions": [
                {
                    "request_hash": item.request_hash,
                    "manifest_hash": item.manifest_hash,
                    "content_hash": item.content_hash,
                    "row_count": item.row_count,
                }
                for item in raw_evidence
            ],
            "staged_input_hashes": {
                name: _sha256(path)
                for name, path in {
                    "calendar": self.inputs.staged_calendar,
                    "security_master": self.inputs.staged_security_master,
                    "union_codes": self.inputs.union_codes,
                    "csi300_codes": self.inputs.csi300_codes,
                    "star50_codes": self.inputs.star50_codes,
                    "tech32_codes": self.inputs.tech32_codes,
                    "tech90_codes": self.inputs.tech90_codes,
                }.items()
            },
            "shared_market_state_hash": state.stable_hash,
            "universes": universe_receipts,
        }
        _atomic_json(self.output_root / "materialization_receipt.json", receipt)
        return receipt
