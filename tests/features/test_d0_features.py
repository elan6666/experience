from datetime import date, datetime

from a_share_research.contracts import AssetRegistry, FeatureGroup
from a_share_research.data import ExecutionStatus, build_execution_receipts, build_mask_bundle
from a_share_research.features.availability import SHANGHAI, date_only_availability
from a_share_research.features.builders import build_feature_row, per_feature_missing
from a_share_research.features.fundamental import build_fundamental_features
from a_share_research.features.schema import FeatureDefinition, InformationClass, d0_features


def test_date_only_financial_is_available_next_trading_day() -> None:
    dates = (date(2025, 4, 30), date(2025, 5, 6), date(2025, 5, 7))
    availability = date_only_availability(date(2025, 4, 30), dates)
    assert availability.date() == date(2025, 5, 6)


def test_each_feature_has_an_independent_missing_marker() -> None:
    definition = FeatureDefinition(
        "roe",
        InformationClass.F,
        FeatureGroup.FINANCIAL,
        "fina_indicator",
        "roe",
        "announcement_next_trade",
    )
    announce = datetime(2025, 5, 6, 9, tzinfo=SHANGHAI)
    row = build_feature_row(
        definition,
        asof_date=date(2025, 5, 6),
        ts_code="000001.SZ",
        value=None,
        source_date=date(2025, 4, 30),
        announce_time=announce,
        availability_time=announce,
        signal_cutoff_time=datetime(2025, 5, 6, 16, tzinfo=SHANGHAI),
        source="synthetic",
    )
    missing = per_feature_missing((row,), expected_features=("roe", "pb"))
    assert missing == {"roe": True, "pb": True}


def test_mask_builder_keeps_member_observed_missing_and_execution_separate() -> None:
    assets = AssetRegistry(("000001.SZ", "600000.SH"))
    masks = build_mask_bundle(
        signal_date=date(2025, 1, 2),
        asset_registry=assets,
        member={"000001.SZ": True, "600000.SH": False},
        statuses={
            "000001.SZ": ExecutionStatus(True, False, 10.0, 11.0, 9.0),
            "600000.SH": ExecutionStatus(True, True, 8.0, 9.0, 7.0),
        },
        feature_missing={
            "pe_ttm": {"000001.SZ": False, "600000.SH": True},
            "roe": {"000001.SZ": True, "600000.SH": False},
        },
        label_available={"000001.SZ": True, "600000.SH": True},
    )
    assert masks.member != masks.observed
    assert masks.feature_missing["pe_ttm"] != masks.feature_missing["roe"]
    assert masks.buyable == (True, False)
    assert masks.sellable == (True, False)


def test_catalog_contains_core_f_and_shared_s_without_global_transforms() -> None:
    groups = {feature.information_class for feature in d0_features()}
    assert groups == {InformationClass.CORE, InformationClass.F, InformationClass.S}


def test_date_only_financial_row_is_not_used_on_announcement_day() -> None:
    dates = (date(2025, 4, 30), date(2025, 5, 6), date(2025, 5, 7))
    announcement_day_rows = build_fundamental_features(
        asof_date=dates[0],
        ts_code="000001.SZ",
        trading_dates=dates,
        daily_basic=None,
        daily_basic_source_date=None,
        financial={"roe": 12.0},
        financial_announcement_date=dates[0],
        financial_announcement_time=None,
        daily_basic_publish_time=None,
    )
    assert next(row for row in announcement_day_rows if row.feature_name == "roe").value is None
    next_trade_rows = build_fundamental_features(
        asof_date=dates[1],
        ts_code="000001.SZ",
        trading_dates=dates,
        daily_basic=None,
        daily_basic_source_date=None,
        financial={"roe": 12.0},
        financial_announcement_date=dates[0],
        financial_announcement_time=None,
        daily_basic_publish_time=None,
    )
    assert next(row for row in next_trade_rows if row.feature_name == "roe").value == 12.0


def test_daily_basic_valuation_is_conservatively_lagged_one_trade_day() -> None:
    dates = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    rows = build_fundamental_features(
        asof_date=dates[1],
        ts_code="000001.SZ",
        trading_dates=dates,
        daily_basic={"pe_ttm": 10.0},
        daily_basic_source_date=dates[0],
        financial=None,
        financial_announcement_date=None,
        financial_announcement_time=None,
        daily_basic_publish_time=None,
    )
    pe = next(row for row in rows if row.feature_name == "pe_ttm")
    assert pe.value == 10.0
    assert pe.availability_time.date() == dates[1]


def test_current_industry_is_missing_without_historical_effective_date() -> None:
    rows = build_fundamental_features(
        asof_date=date(2025, 1, 3),
        ts_code="000001.SZ",
        trading_dates=(date(2025, 1, 2), date(2025, 1, 3)),
        daily_basic=None,
        daily_basic_source_date=None,
        financial=None,
        financial_announcement_date=None,
        financial_announcement_time=None,
        daily_basic_publish_time=None,
        industry_id=7.0,
        industry_effective_date=None,
    )
    industry = next(row for row in rows if row.feature_name == "industry_id")
    assert industry.value is None


def test_historical_industry_keeps_source_and_next_trade_availability() -> None:
    rows = build_fundamental_features(
        asof_date=date(2025, 1, 6),
        ts_code="000001.SZ",
        trading_dates=(date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)),
        daily_basic=None,
        daily_basic_source_date=None,
        financial=None,
        financial_announcement_date=None,
        financial_announcement_time=None,
        daily_basic_publish_time=None,
        industry_id=801010.0,
        industry_effective_date=date(2025, 1, 6),
        industry_source_date=date(2025, 1, 3),
    )
    industry = next(row for row in rows if row.feature_name == "industry_id")
    assert industry.value == 801010.0
    assert industry.source_date == date(2025, 1, 3)
    assert industry.availability_time.date() == date(2025, 1, 6)
    assert not industry.missing_flag
    assert industry.formal_eligible


def test_execution_receipts_bind_t_plus_one_masks_to_d0() -> None:
    assets = AssetRegistry(("000001.SZ",))
    calendar = (date(2025, 1, 2), date(2025, 1, 3))
    receipt, evidence = build_execution_receipts(
        signal_date=calendar[0],
        next_trade_date=calendar[1],
        asset_registry=assets,
        statuses={"000001.SZ": ExecutionStatus(True, False, 10.0, 11.0, 9.0)},
        member={"000001.SZ": True},
        d0_manifest_hash="a" * 64,
        trading_calendar=calendar,
    )
    assert receipt.next_trade_date == calendar[1]
    assert evidence[0].buyable and evidence[0].sellable
