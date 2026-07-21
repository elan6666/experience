"""Canonical contract invariants; execute only on the approved server."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from a_share_research.contracts import (
    AssetRegistry,
    ContractError,
    CoverageState,
    FeatureGroup,
    FormalFeatureManifest,
    Label,
    MaskBundle,
    PITFeature,
    PredictionFrame,
    PredictionRecord,
    RunManifest,
    SecurityMaster,
    canonical_hash,
    validate_mask_series,
)
from a_share_research.protocol import Partition, ProtocolSpec, Purpose, UniverseClass
from a_share_research.quality import ResultState, assert_formal_rankable, is_candidate_state

HASH = "a" * 64


def test_versioned_contract_round_trip_and_hash_are_stable() -> None:
    original = SecurityMaster(
        ts_code="000001.SZ",
        list_date=date(1991, 4, 3),
        delist_date=None,
        board="main",
        industry="bank",
    )
    restored = SecurityMaster.from_dict(original.to_dict())
    assert restored == original
    assert restored.stable_hash() == original.stable_hash()
    assert original.to_dict()["_version"] == "1.0"


def test_prediction_frame_round_trip_preserves_coverage_state() -> None:
    frame = PredictionFrame(
        run_id="ridge-csi300-a0",
        records=(
            PredictionRecord(date(2025, 1, 2), "000001.SZ", 0.25, CoverageState.SCORED),
            PredictionRecord(
                date(2025, 1, 2),
                "600000.SH",
                None,
                CoverageState.INSUFFICIENT_HISTORY,
            ),
        ),
    )
    restored = PredictionFrame.from_dict(frame.to_dict())
    assert restored == frame
    assert restored.coverage == 0.5


def test_run_manifest_round_trip_records_all_reproducibility_hashes() -> None:
    manifest = RunManifest(
        run_id="run-001",
        model="Ridge",
        universe=UniverseClass.CSI300,
        information_set="A0",
        split=Partition.VALIDATION,
        purpose=Purpose.SELECT,
        data_hash=HASH,
        asset_registry_hash=HASH,
        execution_calendar_manifest_hash=HASH,
        feature_schema_hash=HASH,
        market_state_hash=HASH,
        config_hash=HASH,
        code_hash=HASH,
        upstream_commit="internal-ridge-v1",
        seed=7,
        status=ResultState.PASS,
        started_at=datetime(2026, 7, 19, 8, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 19, 9, tzinfo=timezone.utc),
        prediction_hash=HASH,
        formal_feature_manifest_hash=HASH,
    )
    assert RunManifest.from_dict(manifest.to_dict()) == manifest


def test_future_feature_is_rejected_and_missing_marker_is_feature_specific() -> None:
    announce = datetime(2025, 1, 2, 8, tzinfo=timezone.utc)
    cutoff = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    with pytest.raises(ContractError, match="future source_date"):
        PITFeature(
            asof_date=date(2025, 1, 2),
            ts_code="000001.SZ",
            feature_name="pe_ttm",
            feature_group=FeatureGroup.VALUATION,
            value=12.0,
            source_date=date(2025, 1, 3),
            announce_time=announce,
            availability_time=announce,
            signal_cutoff_time=cutoff,
            missing_flag=False,
            source="synthetic",
        ).validate()
    with pytest.raises(ContractError, match="missing_flag"):
        PITFeature(
            asof_date=date(2025, 1, 2),
            ts_code="000001.SZ",
            feature_name="pb",
            feature_group=FeatureGroup.VALUATION,
            value=None,
            source_date=date(2025, 1, 2),
            announce_time=announce,
            availability_time=announce,
            signal_cutoff_time=cutoff,
            missing_flag=False,
            source="synthetic",
        ).validate()


def test_asset_identity_appends_without_reordering_existing_slots() -> None:
    registry = AssetRegistry(("000001.SZ", "600000.SH"))
    expanded = registry.append("688001.SH", "000001.SZ")
    assert expanded.asset_ids == ("000001.SZ", "600000.SH", "688001.SH")
    assert expanded.index_of("600000.SH") == registry.index_of("600000.SH")


def test_masks_remain_independent_through_imputation() -> None:
    asset_ids = ("000001.SZ", "600000.SH", "688001.SH", "300001.SZ")
    masks = MaskBundle(
        signal_date=date(2025, 1, 8),
        asset_ids=asset_ids,
        asset_registry_hash=AssetRegistry(asset_ids).stable_hash(),
        member=(True, True, True, False),
        observed=(True, False, True, True),
        feature_missing={
            "pe_ttm": (False, True, True, False),
            "pb": (False, True, False, False),
            "roe": (True, True, False, False),
        },
        label_available=(True, False, True, True),
        buyable=(True, False, False, False),
        sellable=(True, False, True, True),
        loss=(True, False, True, False),
        evaluation=(True, False, True, False),
    )
    before = masks.to_dict()
    output = masks.with_imputed_values(
        {
            "pe_ttm": (10.0, 0.0, 0.0, 12.0),
            "pb": (1.0, 0.0, 2.0, 3.0),
            "roe": (0.0, 0.0, 0.1, 0.2),
        }
    )
    assert output["pe_ttm"][1] == 0.0
    assert masks.to_dict() == before
    assert masks.feature_missing["pe_ttm"] != masks.feature_missing["pb"]
    assert masks.observed != masks.member


def test_pre_ipo_or_non_member_cannot_be_buyable() -> None:
    with pytest.raises(ContractError, match="buyable requires"):
        MaskBundle(
            signal_date=date(2024, 12, 20),
            asset_ids=("688001.SH",),
            asset_registry_hash=AssetRegistry(("688001.SH",)).stable_hash(),
            member=(False,),
            observed=(False,),
            feature_missing={"pe_ttm": (True,)},
            label_available=(False,),
            buyable=(True,),
            sellable=(False,),
            loss=(False,),
            evaluation=(False,),
        )


def test_execution_masks_cover_suspension_limits_exit_and_delisting() -> None:
    asset_ids = ("000001.SZ", "600000.SH", "688001.SH", "300001.SZ")
    masks = MaskBundle(
        signal_date=date(2025, 1, 10),
        asset_ids=asset_ids,
        asset_registry_hash=AssetRegistry(asset_ids).stable_hash(),
        member=(True, True, False, False),
        observed=(True, True, True, False),
        feature_missing={"pe_ttm": (False, False, True, True)},
        label_available=(True, True, True, False),
        buyable=(False, False, False, False),
        sellable=(True, False, True, False),
        loss=(True, True, False, False),
        evaluation=(True, True, False, False),
    )
    masks.validate()
    assert not masks.buyable[0]  # limit-up
    assert not masks.buyable[1] and not masks.sellable[1]  # suspension
    assert not masks.member[2] and masks.sellable[2]  # exited member can be liquidated
    assert not masks.observed[3] and not masks.sellable[3]  # delisted/unobserved


def test_mask_series_rejects_cross_date_asset_reordering() -> None:
    first_ids = ("000001.SZ", "600000.SH")
    second_ids = tuple(reversed(first_ids))

    def make_bundle(signal_date: date, asset_ids: tuple[str, ...]) -> MaskBundle:
        return MaskBundle(
            signal_date=signal_date,
            asset_ids=asset_ids,
            asset_registry_hash=AssetRegistry(asset_ids).stable_hash(),
            member=(True, True),
            observed=(True, True),
            feature_missing={"pe_ttm": (False, False)},
            label_available=(True, True),
            buyable=(True, True),
            sellable=(True, True),
            loss=(True, True),
            evaluation=(True, True),
        )

    with pytest.raises(ContractError, match="identity reordered"):
        validate_mask_series(
            (
                make_bundle(date(2025, 1, 2), first_ids),
                make_bundle(date(2025, 1, 3), second_ids),
            )
        )


def test_mask_series_allows_only_tail_append_for_newly_known_assets() -> None:
    first_ids = ("000001.SZ", "600000.SH")
    second_ids = first_ids + ("688001.SH",)

    def bundle(signal_date: date, asset_ids: tuple[str, ...]) -> MaskBundle:
        size = len(asset_ids)
        return MaskBundle(
            signal_date=signal_date,
            asset_ids=asset_ids,
            asset_registry_hash=AssetRegistry(asset_ids).stable_hash(),
            member=(True,) * size,
            observed=(True,) * size,
            feature_missing={"pe_ttm": (False,) * size},
            label_available=(True,) * size,
            buyable=(True,) * size,
            sellable=(True,) * size,
            loss=(True,) * size,
            evaluation=(True,) * size,
        )

    validate_mask_series(
        (bundle(date(2025, 1, 2), first_ids), bundle(date(2025, 1, 3), second_ids))
    )
    with pytest.raises(ContractError, match="deleted"):
        validate_mask_series(
            (bundle(date(2025, 1, 2), second_ids), bundle(date(2025, 1, 3), first_ids))
        )


def test_financial_feature_without_announcement_is_not_formally_eligible() -> None:
    cutoff = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    with pytest.raises(ContractError, match="requires announce_time"):
        PITFeature(
            asof_date=date(2025, 1, 2),
            ts_code="000001.SZ",
            feature_name="pe_ttm",
            feature_group=FeatureGroup.VALUATION,
            value=10.0,
            source_date=date(2025, 1, 2),
            announce_time=None,
            availability_time=cutoff,
            signal_cutoff_time=cutoff,
            missing_flag=False,
            source="synthetic",
        )


def test_label_requires_next_trade_entry_and_horizon_trade_exit() -> None:
    calendar = tuple(date(2025, 1, day) for day in (2, 3, 6, 7, 8, 9, 10))
    calendar_hash = canonical_hash(calendar)
    valid = Label(
        signal_date=calendar[0],
        ts_code="000001.SZ",
        horizon=5,
        entry_date=calendar[1],
        exit_date=calendar[6],
        open_to_open_return=0.1,
        benchmark_return=0.02,
        trading_calendar=calendar,
        trading_calendar_hash=calendar_hash,
    )
    assert valid.relative_return == pytest.approx(0.08)
    with pytest.raises(ContractError, match="next trading day"):
        Label(
            signal_date=calendar[0],
            ts_code="000001.SZ",
            horizon=1,
            entry_date=calendar[2],
            exit_date=calendar[3],
            open_to_open_return=0.0,
            benchmark_return=0.0,
            trading_calendar=calendar,
            trading_calendar_hash=calendar_hash,
        )


def test_run_context_prevents_legacy_or_selected_universe_ranking() -> None:
    common = {
        "run_id": "context-test",
        "model": "Ridge",
        "information_set": "A0",
        "data_hash": HASH,
        "asset_registry_hash": HASH,
        "execution_calendar_manifest_hash": HASH,
        "feature_schema_hash": HASH,
        "market_state_hash": HASH,
        "config_hash": HASH,
        "code_hash": HASH,
        "upstream_commit": "internal-ridge-v1",
        "seed": 7,
        "started_at": datetime(2026, 7, 19, 8, tzinfo=timezone.utc),
        "completed_at": datetime(2026, 7, 19, 9, tzinfo=timezone.utc),
        "formal_feature_manifest_hash": HASH,
    }
    with pytest.raises(ContractError, match="LEGACY_VIEWED"):
        RunManifest(
            **common,
            universe=UniverseClass.CSI300,
            split=Partition.LEGACY_VIEWED,
            purpose=Purpose.LEGACY_REPORT,
            status=ResultState.PASS,
        )
    with pytest.raises(ContractError, match="EXPLORATORY_ONLY"):
        RunManifest(
            **common,
            universe=UniverseClass.TECH32,
            split=Partition.VALIDATION,
            purpose=Purpose.SELECT,
            status=ResultState.PASS,
        )
    assert not is_candidate_state(
        ResultState.PASS,
        partition=Partition.LEGACY_VIEWED,
        universe=UniverseClass.CSI300,
    )
    assert not is_candidate_state(
        ResultState.PASS,
        partition=Partition.VALIDATION,
        universe=UniverseClass.TECH90,
    )


def test_run_manifest_rejects_untyped_partition_purpose_and_universe() -> None:
    with pytest.raises(ContractError, match="UniverseClass"):
        RunManifest(
            run_id="untyped",
            model="Ridge",
            universe="CSI300",  # type: ignore[arg-type]
            information_set="A0",
            split=Partition.VALIDATION,
            purpose=Purpose.SELECT,
            data_hash=HASH,
            asset_registry_hash=HASH,
            execution_calendar_manifest_hash=HASH,
            feature_schema_hash=HASH,
            market_state_hash=HASH,
            config_hash=HASH,
            code_hash=HASH,
            upstream_commit="internal-ridge-v1",
            seed=7,
            status=ResultState.PASS,
            started_at=datetime(2026, 7, 19, 8, tzinfo=timezone.utc),
            completed_at=None,
            formal_feature_manifest_hash=HASH,
        )


def test_formal_feature_manifest_fails_when_any_input_is_ineligible() -> None:
    receipt = FormalFeatureManifest(
        dataset_id="csi300-a3",
        d0_manifest_hash=HASH,
        feature_eligibility={"pe_ttm": True, "roe": False},
    )
    with pytest.raises(ContractError, match="ineligible inputs"):
        receipt.require_formal_eligible()


def test_formal_pass_requires_feature_receipt_hash() -> None:
    with pytest.raises(ContractError, match="D0 feature eligibility"):
        RunManifest(
            run_id="missing-formal-receipt",
            model="Ridge",
            universe=UniverseClass.CSI300,
            information_set="A0",
            split=Partition.VALIDATION,
            purpose=Purpose.SELECT,
            data_hash=HASH,
            asset_registry_hash=HASH,
            execution_calendar_manifest_hash=HASH,
            feature_schema_hash=HASH,
            market_state_hash=HASH,
            config_hash=HASH,
            code_hash=HASH,
            upstream_commit="internal-ridge-v1",
            seed=7,
            status=ResultState.PASS,
            started_at=datetime(2026, 7, 19, 8, tzinfo=timezone.utc),
            completed_at=None,
        )


def test_future_formal_ranking_is_bound_to_open_protocol_and_feature_receipt() -> None:
    feature_receipt = FormalFeatureManifest(
        dataset_id="csi300-a3",
        d0_manifest_hash=HASH,
        feature_eligibility={"pe_ttm": True, "roe": True},
    )
    feature_hash = feature_receipt.require_formal_eligible()
    opening_hash = "f" * 64
    manifest = RunManifest(
        run_id="future-formal",
        model="Ridge",
        universe=UniverseClass.CSI300,
        information_set="A3",
        split=Partition.FUTURE_UNSEEN,
        purpose=Purpose.FINAL_EVALUATION,
        data_hash=HASH,
        asset_registry_hash=HASH,
        execution_calendar_manifest_hash=HASH,
        feature_schema_hash=HASH,
        market_state_hash=HASH,
        config_hash=HASH,
        code_hash=HASH,
        upstream_commit="internal-ridge-v1",
        seed=7,
        status=ResultState.PASS,
        started_at=datetime(2026, 7, 20, 8, tzinfo=timezone.utc),
        completed_at=None,
        formal_feature_manifest_hash=feature_hash,
        protocol_open_receipt_hash=opening_hash,
    )
    opened = ProtocolSpec.research_v1().open_future(opening_hash)
    assert_formal_rankable(
        manifest=manifest,
        protocol=opened,
        feature_manifest=feature_receipt,
    )
    wrong_protocol = ProtocolSpec.research_v1().open_future("e" * 64)
    with pytest.raises(ContractError, match="opening receipt"):
        assert_formal_rankable(
            manifest=manifest,
            protocol=wrong_protocol,
            feature_manifest=feature_receipt,
        )
    wrong_data_receipt = FormalFeatureManifest(
        dataset_id="csi300-a3",
        d0_manifest_hash="e" * 64,
        feature_eligibility={"pe_ttm": True, "roe": True},
    )
    with pytest.raises(ValueError, match="D0 hash"):
        assert_formal_rankable(
            manifest=manifest,
            protocol=opened,
            feature_manifest=wrong_data_receipt,
        )


def test_future_pass_without_protocol_opening_receipt_fails() -> None:
    with pytest.raises(ContractError, match="protocol opening receipt"):
        RunManifest(
            run_id="future-missing-opening",
            model="Ridge",
            universe=UniverseClass.CSI300,
            information_set="A0",
            split=Partition.FUTURE_UNSEEN,
            purpose=Purpose.FINAL_EVALUATION,
            data_hash=HASH,
            asset_registry_hash=HASH,
            execution_calendar_manifest_hash=HASH,
            feature_schema_hash=HASH,
            market_state_hash=HASH,
            config_hash=HASH,
            code_hash=HASH,
            upstream_commit="internal-ridge-v1",
            seed=7,
            status=ResultState.PASS,
            started_at=datetime(2026, 7, 20, 8, tzinfo=timezone.utc),
            completed_at=None,
            formal_feature_manifest_hash=HASH,
        )


def test_synthetic_fixture_pre_registers_every_required_edge_case() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_events.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    events = {item["event"] for item in fixture["events"]}
    assert events == {
        "pre_ipo",
        "constituent_entry",
        "suspension",
        "limit_up",
        "limit_down",
        "financial_missing",
        "constituent_exit",
        "delisting",
    }


def test_json_schema_constrains_real_contract_fields() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "configs" / "schema" / "contracts-v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    pit = schema["$defs"]["pit_feature"]
    manifest = schema["$defs"]["run_manifest"]
    fill = schema["$defs"]["portfolio_fill"]
    evidence = schema["$defs"]["eligibility_evidence"]
    ledger = schema["$defs"]["portfolio_ledger"]
    assert pit["additionalProperties"] is False
    assert {"availability_time", "signal_cutoff_time", "feature_group"} <= set(pit["required"])
    assert {"universe", "split", "purpose", "status", "asset_registry_hash"} <= set(
        manifest["required"]
    )
    assert {"eligibility_evidence_id"} <= set(fill["required"])
    assert {"asset_registry_hash", "evidence_source_hash", "buyable", "sellable"} <= set(
        evidence["required"]
    )
    assert {
        "run_data_hash",
        "eligibility_source_hash",
        "execution_calendar_receipts",
        "eligibility_evidence",
    } <= set(ledger["required"])
