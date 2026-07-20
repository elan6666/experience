"""Complete prediction export and typed coverage tests; server execution only."""

from datetime import date

import pytest

from a_share_research.adapters.common import (
    AdapterContractError,
    PredictionBatch,
    build_causal_asset_master,
    export_prediction_batches,
)
from a_share_research.contracts import (
    AssetRegistry,
    CoverageState,
    MaskBundle,
    UniverseMembership,
)


def model_master():
    rows = tuple(
        UniverseMembership(
            asof_date=date(2019, 1, 2),
            ts_code=ts_code,
            universe="CSI300",
            effective_from=date(2019, 1, 2),
            effective_to=None,
            source="synthetic",
        )
        for ts_code in ("000001.SZ", "600000.SH")
    )
    return build_causal_asset_master(rows, known_through=date(2019, 12, 31))


def mask(signal_date: date, registry: AssetRegistry, *, second_member: bool) -> MaskBundle:
    return MaskBundle(
        signal_date=signal_date,
        asset_ids=registry.asset_ids,
        asset_registry_hash=registry.stable_hash(),
        member=(True, second_member, True),
        observed=(True, True, True),
        feature_missing={"return_1d": (False, False, False)},
        label_available=(True, True, True),
        buyable=(True, second_member, True),
        sellable=(True, True, True),
        loss=(True, second_member, True),
        evaluation=(True, second_member, True),
    )


def test_export_includes_final_partial_batch_and_unsupported_new_member() -> None:
    master = model_master()
    registry = AssetRegistry(master.asset_ids + ("688001.SH",))
    dates = (date(2025, 1, 2), date(2025, 1, 3))
    frame = export_prediction_batches(
        run_id="itransformer-csi300-a0",
        evaluation_registry=registry,
        model_master=master,
        expected_dates=dates,
        masks=(
            mask(dates[0], registry, second_member=False),
            mask(dates[1], registry, second_member=True),
        ),
        history_ready=((True, True, True), (True, False, True)),
        batches=(
            PredictionBatch((dates[0],), ((0.2, 0.1),)),
            PredictionBatch((dates[1],), ((0.3, 0.4),)),
        ),
    )
    assert len(frame.records) == 6
    states = {(row.signal_date, row.ts_code): row.coverage_state for row in frame.records}
    assert states[(dates[0], "000001.SZ")] is CoverageState.SCORED
    assert states[(dates[0], "600000.SH")] is CoverageState.NOT_MEMBER
    assert states[(dates[0], "688001.SH")] is CoverageState.MODEL_UNSUPPORTED
    assert states[(dates[1], "600000.SH")] is CoverageState.INSUFFICIENT_HISTORY


def test_export_rejects_loader_that_dropped_the_final_batch() -> None:
    master = model_master()
    registry = AssetRegistry(master.asset_ids + ("688001.SH",))
    dates = (date(2025, 1, 2), date(2025, 1, 3))
    masks = tuple(mask(signal_date, registry, second_member=True) for signal_date in dates)
    with pytest.raises(AdapterContractError, match="incomplete prediction batches"):
        export_prediction_batches(
            run_id="fact-csi300-a0",
            evaluation_registry=registry,
            model_master=master,
            expected_dates=dates,
            masks=masks,
            history_ready=((True, True, True), (True, True, True)),
            batches=(PredictionBatch((dates[0],), ((0.2, 0.1),)),),
        )
