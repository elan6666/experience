from datetime import date, datetime, timezone

from a_share_research.contracts import FeatureGroup, PITFeature, UniverseMembership
from a_share_research.data import (
    D0Manifest,
    assert_shared_market_state_hashes,
    build_index_state_rows,
    build_open_to_open_labels,
    build_shared_market_state,
)
from a_share_research.data.manifest import UniverseGate
from a_share_research.protocol import UniverseClass
from a_share_research.quality import ResultState
from a_share_research.quality.d0 import assess_universe_gate


def _dates() -> tuple[date, ...]:
    return tuple(date(2025, 1, day) for day in (2, 3, 6, 7, 8, 9, 10, 13))


def test_labels_use_t_plus_one_and_exact_trade_horizon() -> None:
    dates = _dates()
    opens = {day: 10.0 + index for index, day in enumerate(dates)}
    benchmark = {day: 100.0 + index for index, day in enumerate(dates)}
    labels = build_open_to_open_labels(
        ts_code="000001.SZ",
        signal_dates=(dates[0],),
        trading_calendar=dates,
        opens=opens,
        benchmark_opens=benchmark,
        horizons=(1, 5),
    )
    one = labels[0]
    assert one.entry_date == dates[1]
    assert one.exit_date == dates[2]
    assert labels[1].exit_date == dates[6]


def test_market_state_hash_is_one_csi300_receipt_for_all_consumers() -> None:
    dates = tuple(date(2025, 1, day) for day in range(1, 8))
    index_close = {day: 100.0 + index for index, day in enumerate(dates)}
    returns = {day: {"000001.SZ": 0.01, "600000.SH": -0.005} for day in dates}
    amount = {day: {"000001.SZ": 100.0, "600000.SH": 200.0} for day in dates}
    turnover = {day: {"000001.SZ": 1.0, "600000.SH": 2.0} for day in dates}
    state = build_shared_market_state(
        trading_dates=dates,
        index_close=index_close,
        member_returns=returns,
        member_amount=amount,
        member_turnover=turnover,
        member_industry_by_date={
            day: {"000001.SZ": "bank", "600000.SH": "energy"} for day in dates
        },
        eligible_member_codes_by_date={
            day: frozenset(("000001.SZ", "600000.SH")) for day in dates
        },
        source_membership_hash="a" * 64,
        lookback=2,
    )
    assert state.stable_hash == state.stable_hash
    assert {row.source_universe for row in state.rows} == {"CSI300"}
    assert assert_shared_market_state_hashes((state.stable_hash,) * 4) == state.stable_hash


def test_tech_gate_remains_exploratory_even_with_clean_data() -> None:
    days = _dates()
    memberships = tuple(
        UniverseMembership(day, "000001.SZ", "TECH32", days[0], None, "frozen")
        for day in days
    )
    cutoff = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    features = (
        PITFeature(
            asof_date=days[0],
            ts_code="000001.SZ",
            feature_name="close",
            feature_group=FeatureGroup.CORE,
            value=10.0,
            source_date=days[0],
            announce_time=None,
            availability_time=cutoff,
            signal_cutoff_time=cutoff,
            missing_flag=False,
            source="synthetic",
        ),
    )
    gate = assess_universe_gate(
        universe=UniverseClass.TECH32,
        memberships=memberships,
        features=features,
        labels=(),
        expected_member_dates=len(days),
        expected_core_values=1,
    )
    assert gate.status is ResultState.EXPLORATORY_ONLY


def test_incomplete_star50_history_blocks_formal_ranking() -> None:
    days = _dates()
    memberships = tuple(
        UniverseMembership(day, "688001.SH", "STAR50", days[0], None, "official")
        for day in days
    )
    gate = assess_universe_gate(
        universe=UniverseClass.STAR50,
        memberships=memberships,
        features=(),
        labels=(),
        expected_member_dates=len(days),
        expected_core_values=1,
        star50_history_complete=False,
    )
    assert gate.status is ResultState.BLOCKED


def test_d0_manifest_round_trip_requires_all_four_universes_and_http_notice() -> None:
    gates = tuple(
        UniverseGate(
            universe=universe,
            status=(
                ResultState.EXPLORATORY_ONLY
                if universe in {UniverseClass.TECH32, UniverseClass.TECH90}
                else ResultState.PASS
            ),
            membership_coverage=1.0,
            core_coverage=1.0,
            duplicate_keys=0,
            pit_violations=0,
            label_boundary_violations=0,
        )
        for universe in UniverseClass
    )
    manifest = D0Manifest(
        dataset_id="synthetic-d0",
        created_at_utc=datetime(2026, 7, 19, tzinfo=timezone.utc),
        cutoff_date=date(2026, 7, 17),
        raw_snapshot_hashes={"calendar": "a" * 64},
        canonical_table_hashes={"csi300/features": "f" * 64},
        security_master_hash="b" * 64,
        trading_calendar_hash="c" * 64,
        feature_schema_hash="d" * 64,
        market_state_hash="e" * 64,
        universe_gates=gates,
        provider_transport_notice="approved proxy uses plain HTTP",
    )
    restored = D0Manifest.from_dict(manifest.to_dict())
    assert restored.content_hash == manifest.content_hash

def test_index_state_rows_produce_per_index_features() -> None:
    """Index factors produce idx_<code>_<feature> rows with correct source_universe."""
    dates = tuple(date(2025, 1, day) for day in range(1, 12))
    closes_000001 = {day: 100.0 * (1.01 ** idx) for idx, day in enumerate(dates)}
    rows = build_index_state_rows(
        trading_dates=dates,
        index_closes={"000001.SH": closes_000001},
        lookback=5,
    )
    assert len(rows) > 0
    names = {r.feature_name for r in rows}
    assert "idx_000001_trend_20d" in names
    assert "idx_000001_volatility_20d" in names
    assert "idx_000001_return_5d" in names
    assert all(r.source_universe == "000001.SH" for r in rows)


def test_shared_market_state_includes_index_rows_when_closes_provided() -> None:
    """build_shared_market_state adds index rows when index_closes is provided."""
    dates = tuple(date(2025, 1, day) for day in range(1, 12))
    index_close = {day: 100.0 + idx for idx, day in enumerate(dates)}
    closes_sh = {day: 200.0 * (1.005 ** idx) for idx, day in enumerate(dates)}
    returns = {day: {"000001.SZ": 0.01} for day in dates}
    amount = {day: {"000001.SZ": 100.0} for day in dates}
    turnover = {day: {"000001.SZ": 1.0} for day in dates}
    state = build_shared_market_state(
        trading_dates=dates,
        index_close=index_close,
        member_returns=returns,
        member_amount=amount,
        member_turnover=turnover,
        member_industry_by_date={day: {"000001.SZ": "bank"} for day in dates},
        eligible_member_codes_by_date={day: frozenset(("000001.SZ",)) for day in dates},
        source_membership_hash="a" * 64,
        lookback=5,
        index_closes={"000001.SH": closes_sh},
    )
    sources = {r.source_universe for r in state.rows}
    assert "CSI300" in sources
    assert "000001.SH" in sources
    idx_names = {r.feature_name for r in state.rows if r.source_universe == "000001.SH"}
    assert "idx_000001_trend_20d" in idx_names


def test_shared_market_state_without_index_closes_still_csi300_only() -> None:
    """When index_closes is None, state rows remain CSI300-only (backward compat)."""
    dates = tuple(date(2025, 1, day) for day in range(1, 8))
    index_close = {day: 100.0 + idx for idx, day in enumerate(dates)}
    returns = {day: {"000001.SZ": 0.01} for day in dates}
    amount = {day: {"000001.SZ": 100.0} for day in dates}
    turnover = {day: {"000001.SZ": 1.0} for day in dates}
    state = build_shared_market_state(
        trading_dates=dates,
        index_close=index_close,
        member_returns=returns,
        member_amount=amount,
        member_turnover=turnover,
        member_industry_by_date={day: {"000001.SZ": "bank"} for day in dates},
        eligible_member_codes_by_date={day: frozenset(("000001.SZ",)) for day in dates},
        source_membership_hash="a" * 64,
        lookback=2,
    )
    assert {r.source_universe for r in state.rows} == {"CSI300"}
