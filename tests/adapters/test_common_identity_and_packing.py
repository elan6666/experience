"""Synthetic adapter invariants; execute only on the approved server."""

from datetime import date

import pytest

from a_share_research.adapters.common import (
    AdapterContractError,
    FeaturePackingSchema,
    InformationGate,
    PanelWindow,
    assert_constant_parameter_count,
    build_causal_asset_master,
    pack_feature_window,
    require_stable_slot_series,
)
from a_share_research.contracts import AssetRegistry, MaskBundle, UniverseMembership


def membership(ts_code: str, effective_from: date) -> UniverseMembership:
    return UniverseMembership(
        asof_date=effective_from,
        ts_code=ts_code,
        universe="CSI300",
        effective_from=effective_from,
        effective_to=None,
        source="synthetic_membership",
    )


def test_causal_master_excludes_future_member_then_appends_without_reordering() -> None:
    rows = (
        membership("000001.SZ", date(2019, 1, 2)),
        membership("600000.SH", date(2021, 6, 1)),
    )
    first = build_causal_asset_master(rows, known_through=date(2020, 12, 31))
    assert first.asset_ids == ("000001.SZ",)
    assert not first.supports("600000.SH")

    retrained = build_causal_asset_master(
        rows,
        known_through=date(2022, 12, 31),
        previous=first,
    )
    assert retrained.asset_ids == ("000001.SZ", "600000.SH")
    assert retrained.slot("000001.SZ") == first.slot("000001.SZ")
    assert retrained.parent_registry_hash == first.registry.stable_hash()


def test_daily_stock_slots_cannot_be_reordered() -> None:
    master = build_causal_asset_master(
        (
            membership("000001.SZ", date(2019, 1, 2)),
            membership("600000.SH", date(2019, 1, 2)),
        ),
        known_through=date(2020, 12, 31),
    )
    require_stable_slot_series(
        master,
        (
            (date(2020, 1, 2), master.asset_ids),
            (date(2020, 1, 3), master.asset_ids),
        ),
    )
    with pytest.raises(AdapterContractError, match="daily slot reordering"):
        require_stable_slot_series(
            master,
            ((date(2020, 1, 2), tuple(reversed(master.asset_ids))),),
        )


def make_mask(signal_date: date, asset_ids: tuple[str, ...]) -> MaskBundle:
    return MaskBundle(
        signal_date=signal_date,
        asset_ids=asset_ids,
        asset_registry_hash=AssetRegistry(asset_ids).stable_hash(),
        member=(True, True),
        observed=(True, False),
        feature_missing={
            "return_1d": (False, True),
            "pe_ttm": (False, True),
            "pb": (signal_date == date(2020, 1, 2), True),
            "csi300_trend_20d": (False, True),
        },
        label_available=(True, False),
        buyable=(True, False),
        sellable=(True, False),
        loss=(True, False),
        evaluation=(True, False),
    )


def make_panel() -> PanelWindow:
    dates = (date(2020, 1, 2), date(2020, 1, 3))
    master = build_causal_asset_master(
        (
            membership("000001.SZ", date(2019, 1, 2)),
            membership("600000.SH", date(2019, 1, 2)),
        ),
        known_through=date(2019, 12, 31),
    )
    return PanelWindow(
        dates=dates,
        asset_master=master,
        values={
            "return_1d": ((0.01, None), (0.02, None)),
            "pe_ttm": ((10.0, None), (11.0, None)),
            "pb": ((None, None), (1.2, None)),
            "csi300_trend_20d": ((0.1, None), (0.2, None)),
        },
        masks=tuple(make_mask(signal_date, master.asset_ids) for signal_date in dates),
    )


def test_a0_a3_have_identical_shape_but_isolated_information() -> None:
    schema = FeaturePackingSchema(
        core=("return_1d",),
        factors=("pe_ttm", "pb"),
        state=("csi300_trend_20d",),
    )
    packed = {
        gate: pack_feature_window(make_panel(), schema=schema, gate=gate)
        for gate in InformationGate
    }
    assert len({item.channels for item in packed.values()}) == 1
    assert len({item.native_variate_count for item in packed.values()}) == 1
    assert len({item.config_shape_hash() for item in packed.values()}) == 1
    assert all(item.masks == make_panel().masks for item in packed.values())

    a0_first = packed[InformationGate.A0].values[0][0]
    a1_first = packed[InformationGate.A1].values[0][0]
    a2_first = packed[InformationGate.A2].values[0][0]
    a3_first = packed[InformationGate.A3].values[0][0]
    # channel order: return, pe, pb, missing-pe, missing-pb, state
    assert a0_first == (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert a1_first == (0.01, 10.0, 0.0, 0.0, 1.0, 0.0)
    assert a2_first == (0.01, 0.0, 0.0, 0.0, 0.0, 0.1)
    assert a3_first == (0.01, 10.0, 0.0, 0.0, 1.0, 0.1)
    assert packed[InformationGate.A3].values[0][1] == (0.0,) * 6


def test_packing_rejects_missing_per_factor_mask_and_nonconstant_parameters() -> None:
    schema = FeaturePackingSchema(
        core=("return_1d",),
        factors=("pe_ttm", "pb"),
        state=("csi300_trend_20d",),
    )
    panel = make_panel()
    bad_masks = tuple(
        MaskBundle(
            signal_date=bundle.signal_date,
            asset_ids=bundle.asset_ids,
            asset_registry_hash=bundle.asset_registry_hash,
            member=bundle.member,
            observed=bundle.observed,
            feature_missing={
                name: mask for name, mask in bundle.feature_missing.items() if name != "pb"
            },
            label_available=bundle.label_available,
            buyable=bundle.buyable,
            sellable=bundle.sellable,
            loss=bundle.loss,
            evaluation=bundle.evaluation,
        )
        for bundle in panel.masks
    )
    bad_panel = PanelWindow(panel.dates, panel.asset_master, panel.values, bad_masks)
    with pytest.raises(AdapterContractError, match="per-feature missing masks"):
        pack_feature_window(bad_panel, schema=schema, gate=InformationGate.A3)

    assert_constant_parameter_count({gate: 100 for gate in InformationGate})
    with pytest.raises(AdapterContractError, match="parameter counts differ"):
        assert_constant_parameter_count(
            {gate: 100 + int(gate is InformationGate.A3) for gate in InformationGate}
        )
