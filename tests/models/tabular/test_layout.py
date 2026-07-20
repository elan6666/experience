import json
from pathlib import Path

import pytest

from a_share_research.contracts import ContractError
from a_share_research.models.tabular import (
    FeatureGate,
    InformationSet,
    default_feature_layout,
)


def _payload():
    layout = default_feature_layout()
    values = {name: float(index + 1) for index, name in enumerate(
        layout.core + layout.fundamental + layout.market_state
    )}
    missing = {name: False for name in values}
    return layout, values, missing


def test_a0_a3_share_one_union_schema_hash_and_width() -> None:
    layout, values, missing = _payload()
    hashes = set()
    widths = set()
    for information_set in InformationSet:
        gate = FeatureGate(information_set)
        hashes.add(layout.stable_hash())
        widths.add(len(layout.vectorize(values, missing, gate)))
    assert len(hashes) == 1
    assert widths == {len(layout.columns)}


def test_disabled_groups_are_constant_and_cannot_leak_values_or_missingness() -> None:
    layout, first, first_missing = _payload()
    second = dict(first)
    second_missing = dict(first_missing)
    for name in layout.fundamental:
        second[name] = None
        second_missing[name] = True
    for name in layout.market_state:
        second[name] = -9999.0

    a0 = FeatureGate(InformationSet.A0)
    assert layout.vectorize(first, first_missing, a0) == layout.vectorize(
        second, second_missing, a0
    )


def test_each_f_factor_has_its_own_missing_column() -> None:
    layout, values, missing = _payload()
    values[layout.fundamental[0]] = None
    missing[layout.fundamental[0]] = True
    vector = layout.vectorize(values, missing, FeatureGate(InformationSet.A1))
    offset = len(layout.core) + len(layout.fundamental)
    indicators = vector[offset : offset + len(layout.fundamental)]
    assert indicators[0] == 1.0
    assert all(value == 0.0 for value in indicators[1:])


def test_core_and_enabled_shared_state_fail_closed_instead_of_silent_imputation() -> None:
    layout, values, missing = _payload()
    values[layout.core[0]] = None
    missing[layout.core[0]] = True
    with pytest.raises(ContractError, match="mandatory Core"):
        layout.vectorize(values, missing, FeatureGate(InformationSet.A0))

    layout, values, missing = _payload()
    values[layout.market_state[0]] = None
    missing[layout.market_state[0]] = True
    with pytest.raises(ContractError, match="complete shared market state"):
        layout.vectorize(values, missing, FeatureGate(InformationSet.A2))


def test_checked_in_layout_config_matches_code_catalog() -> None:
    layout = default_feature_layout()
    path = Path(__file__).parents[3] / "configs/features/tabular-layout-v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    groups = config["ordered_groups"]
    assert tuple(groups["Core"]) == layout.core
    assert tuple(groups["F"]) == layout.fundamental
    assert tuple(groups["F_missing"]) == layout.fundamental_missing
    assert tuple(groups["S"]) == layout.market_state
