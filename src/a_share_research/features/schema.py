"""Frozen Core/F/S feature catalog; D0 stores values without global transforms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from a_share_research.contracts import ContractError, FeatureGroup, canonical_hash


class InformationClass(str, Enum):
    CORE = "CORE"
    F = "F"
    S = "S"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    information_class: InformationClass
    contract_group: FeatureGroup
    endpoint: str
    source_field: str
    availability_rule: str
    lookback: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.endpoint or not self.source_field:
            raise ContractError("feature definition fields cannot be empty")
        if self.lookback < 1:
            raise ContractError("feature lookback must be positive")
        if self.information_class is InformationClass.S:
            if self.contract_group is not FeatureGroup.MARKET_STATE:
                raise ContractError("S features must use shared market-state contract")


def _f(
    name: str,
    info: InformationClass,
    group: FeatureGroup,
    endpoint: str,
    field: str,
    availability: str,
    lookback: int = 1,
) -> FeatureDefinition:
    return FeatureDefinition(name, info, group, endpoint, field, availability, lookback)


_CORE = InformationClass.CORE
_FUNDAMENTAL = InformationClass.F
_STATE = InformationClass.S
_CORE_GROUP = FeatureGroup.CORE
_FINANCIAL_GROUP = FeatureGroup.FINANCIAL
_VALUATION_GROUP = FeatureGroup.VALUATION
_STATE_GROUP = FeatureGroup.MARKET_STATE
_ANNOUNCEMENT = "announcement_next_trade"
_SHARED_STATE = "shared_market_state"


_D0_FEATURES = (
    _f("open", _CORE, _CORE_GROUP, "daily", "open", "close_t"),
    _f("high", _CORE, _CORE_GROUP, "daily", "high", "close_t"),
    _f("low", _CORE, _CORE_GROUP, "daily", "low", "close_t"),
    _f("close", _CORE, _CORE_GROUP, "daily", "close", "close_t"),
    _f("volume", _CORE, _CORE_GROUP, "daily", "vol", "close_t"),
    _f("amount", _CORE, _CORE_GROUP, "daily", "amount", "close_t"),
    _f(
        "turnover_rate",
        _CORE,
        _CORE_GROUP,
        "daily_basic",
        "turnover_rate",
        "close_t",
    ),
    _f("return_1d", _CORE, _CORE_GROUP, "derived_market", "close", "close_t", 2),
    _f("return_5d", _CORE, _CORE_GROUP, "derived_market", "close", "close_t", 6),
    _f("return_20d", _CORE, _CORE_GROUP, "derived_market", "close", "close_t", 21),
    _f("volatility_20d", _CORE, _CORE_GROUP, "derived_market", "close", "close_t", 21),
    _f("amount_mean_20d", _CORE, _CORE_GROUP, "derived_market", "amount", "close_t", 20),
    _f("pe_ttm", _FUNDAMENTAL, _VALUATION_GROUP, "daily_basic", "pe_ttm", "provider_publish_t"),
    _f("pb", _FUNDAMENTAL, _VALUATION_GROUP, "daily_basic", "pb", "provider_publish_t"),
    _f("ps_ttm", _FUNDAMENTAL, _VALUATION_GROUP, "daily_basic", "ps_ttm", "provider_publish_t"),
    _f("dv_ttm", _FUNDAMENTAL, _VALUATION_GROUP, "daily_basic", "dv_ttm", "provider_publish_t"),
    _f("total_mv", _FUNDAMENTAL, _VALUATION_GROUP, "daily_basic", "total_mv", "provider_publish_t"),
    _f("circ_mv", _FUNDAMENTAL, _VALUATION_GROUP, "daily_basic", "circ_mv", "provider_publish_t"),
    _f("roe", _FUNDAMENTAL, _FINANCIAL_GROUP, "fina_indicator", "roe", _ANNOUNCEMENT),
    _f("roa", _FUNDAMENTAL, _FINANCIAL_GROUP, "fina_indicator", "roa", _ANNOUNCEMENT),
    _f(
        "grossprofit_margin", _FUNDAMENTAL, _FINANCIAL_GROUP,
        "fina_indicator", "grossprofit_margin", _ANNOUNCEMENT,
    ),
    _f(
        "debt_to_assets", _FUNDAMENTAL, _FINANCIAL_GROUP,
        "fina_indicator", "debt_to_assets", _ANNOUNCEMENT,
    ),
    _f(
        "current_ratio", _FUNDAMENTAL, _FINANCIAL_GROUP,
        "fina_indicator", "current_ratio", _ANNOUNCEMENT,
    ),
    _f("or_yoy", _FUNDAMENTAL, _FINANCIAL_GROUP, "fina_indicator", "or_yoy", _ANNOUNCEMENT),
    _f(
        "netprofit_yoy", _FUNDAMENTAL, _FINANCIAL_GROUP,
        "fina_indicator", "netprofit_yoy", _ANNOUNCEMENT,
    ),
    _f("ocf_to_or", _FUNDAMENTAL, _FINANCIAL_GROUP, "fina_indicator", "ocf_to_or", _ANNOUNCEMENT),
    _f(
        "assets_turn", _FUNDAMENTAL, _FINANCIAL_GROUP,
        "fina_indicator", "assets_turn", _ANNOUNCEMENT,
    ),
    _f(
        "industry_id", _FUNDAMENTAL, _FINANCIAL_GROUP,
        "security_master", "industry_id", "identity_effective",
    ),
    _f("csi300_trend_20d", _STATE, _STATE_GROUP, _SHARED_STATE, "trend_20d", "close_t", 20),
    _f(
        "csi300_volatility_20d", _STATE, _STATE_GROUP,
        _SHARED_STATE, "volatility_20d", "close_t", 20,
    ),
    _f("csi300_turnover_20d", _STATE, _STATE_GROUP, _SHARED_STATE, "turnover_20d", "close_t", 20),
    _f("csi300_breadth", _STATE, _STATE_GROUP, _SHARED_STATE, "breadth", "close_t"),
    _f(
        "csi300_industry_dispersion", _STATE, _STATE_GROUP,
        _SHARED_STATE, "industry_dispersion", "close_t",
    ),
    _f("csi300_liquidity", _STATE, _STATE_GROUP, _SHARED_STATE, "liquidity", "close_t", 20),
)


def d0_features() -> tuple[FeatureDefinition, ...]:
    if len({item.name for item in _D0_FEATURES}) != len(_D0_FEATURES):
        raise ContractError("D0 feature names must be unique")
    return _D0_FEATURES


def feature_schema_hash() -> str:
    return canonical_hash(
        tuple(
            {
                "name": item.name,
                "information_class": item.information_class.value,
                "contract_group": item.contract_group.value,
                "endpoint": item.endpoint,
                "source_field": item.source_field,
                "availability_rule": item.availability_rule,
                "lookback": item.lookback,
            }
            for item in d0_features()
        )
    )
