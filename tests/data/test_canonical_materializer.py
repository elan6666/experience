import json
from datetime import date, datetime

import pytest

from a_share_research.adapters.common.identity import CausalAssetMaster
from a_share_research.contracts import AssetRegistry, ContractError, MaskBundle, canonical_hash
from a_share_research.data.canonical import weekly_signal_dates
from a_share_research.data.eligibility import ExecutionStatus, build_mask_bundle
from a_share_research.data.labels import CompactLabel, build_open_to_open_labels, compact_label
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.data.providers import ImmutableRawStore, QueryRequest, QueryResult
from a_share_research.data.raw_catalog import ExactRawCatalog
from a_share_research.features.availability import SHANGHAI
from a_share_research.features.builders import build_feature_row
from a_share_research.features.schema import InformationClass, d0_features


def _request(code: str) -> QueryRequest:
    return QueryRequest(
        endpoint="daily",
        params={"ts_code": code, "start_date": "20250101", "end_date": "20250131"},
        fields=("ts_code", "trade_date", "open"),
        partition_key=f"market_all:daily_{code}",
    )


def _manifest_item(request: QueryRequest) -> dict[str, object]:
    return {
        "endpoint": request.endpoint,
        "params": dict(request.params),
        "fields": list(request.fields),
        "partition_key": request.partition_key,
        "min_row_count": request.min_row_count,
        "reject_at_row_count": request.reject_at_row_count,
        "request_hash": request.request_hash,
    }


def test_exact_raw_catalog_reads_only_declared_request_hash(tmp_path) -> None:
    declared = _request("000001.SZ")
    rogue = _request("600000.SH")
    store = ImmutableRawStore(tmp_path / "raw" / "d0_v1")
    declared_rows = ({"ts_code": "000001.SZ", "trade_date": "20250102", "open": 10.0},)
    rogue_rows = ({"ts_code": "600000.SH", "trade_date": "20250102", "open": 20.0},)
    store.store(QueryResult(declared, declared_rows))
    store.store(QueryResult(rogue, rogue_rows))
    manifest = tmp_path / "requests.json"
    manifest.write_text(
        json.dumps({"bounded_requests": [_manifest_item(declared)]}),
        encoding="utf-8",
    )
    catalog = ExactRawCatalog(raw_root=tmp_path / "raw" / "d0_v1", request_manifest=manifest)
    assert catalog.rows(declared) == declared_rows
    with pytest.raises(ContractError, match="not declared"):
        catalog.rows(rogue)
    assert catalog.require_all_partitions()[0].request_hash == declared.request_hash


def test_exact_raw_catalog_rejects_tampered_embedded_hash(tmp_path) -> None:
    request = _request("000001.SZ")
    manifest = tmp_path / "requests.json"
    item = _manifest_item(request)
    item["request_hash"] = "0" * 64
    manifest.write_text(json.dumps({"bounded_requests": [item]}), encoding="utf-8")
    with pytest.raises(ContractError, match="embedded request hash"):
        ExactRawCatalog(raw_root=tmp_path / "raw" / "d0_v1", request_manifest=manifest)


def test_weekly_signal_dates_are_last_observed_trade_of_each_week() -> None:
    dates = (
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 10),
    )
    assert weekly_signal_dates(dates) == (date(2025, 1, 3), date(2025, 1, 10))


def test_compact_labels_keep_external_calendar_evidence_without_row_duplication() -> None:
    calendar = tuple(date(2025, 1, day) for day in (2, 3, 6, 7, 8, 9, 10))
    opens = {day: 10.0 + index for index, day in enumerate(calendar)}
    benchmark = {day: 100.0 + index for index, day in enumerate(calendar)}
    source = build_open_to_open_labels(
        ts_code="000001.SZ",
        signal_dates=(calendar[0],),
        trading_calendar=calendar,
        opens=opens,
        benchmark_opens=benchmark,
        horizons=(1, 5),
    )
    compact = tuple(compact_label(row) for row in source)
    assert compact[0].entry_index == compact[0].signal_index + 1
    assert "trading_calendar" not in compact[0].to_dict()
    compact[1].verify_calendar(calendar)


def test_signal_observation_and_t_plus_one_execution_status_are_separate() -> None:
    assets = AssetRegistry(("000001.SZ",))
    masks = build_mask_bundle(
        signal_date=date(2025, 1, 2),
        asset_registry=assets,
        member={"000001.SZ": True},
        statuses={"000001.SZ": ExecutionStatus(True, False, 10.0, 11.0, 9.0)},
        execution_statuses={
            "000001.SZ": ExecutionStatus(True, True, 10.0, 11.0, 9.0)
        },
        feature_missing={"close": {"000001.SZ": False}},
        label_available={"000001.SZ": True},
    )
    assert masks.observed == (True,)
    assert masks.buyable == (False,)
    assert masks.sellable == (False,)


def test_next_day_trade_status_cannot_make_an_unobserved_signal_row_tradable() -> None:
    code = "000001.SZ"
    masks = build_mask_bundle(
        signal_date=date(2025, 1, 2),
        asset_registry=AssetRegistry((code,)),
        member={code: True},
        statuses={code: ExecutionStatus(False, False, None, None, None)},
        execution_statuses={code: ExecutionStatus(True, False, 10.0, 11.0, 9.0)},
        feature_missing={"close": {code: True}},
        label_available={code: False},
    )
    assert masks.observed == (False,)
    assert masks.buyable == (False,)
    assert masks.sellable == (False,)


def test_canonical_loader_emits_tabular_sample_and_fixed_slot_panel(tmp_path) -> None:
    root = tmp_path / "canonical"
    universe = root / "csi300"
    market = root / "common" / "daily_market"
    universe.mkdir(parents=True)
    market.mkdir(parents=True)
    signal = date(2025, 1, 3)
    cutoff = datetime(2025, 1, 3, 16, tzinfo=SHANGHAI)
    feature_rows = []
    for definition in d0_features():
        value = 1.0 if definition.information_class is InformationClass.CORE else None
        feature_rows.append(
            build_feature_row(
                definition,
                asof_date=signal,
                ts_code="000001.SZ",
                value=value,
                source_date=signal,
                announce_time=None,
                availability_time=cutoff,
                signal_cutoff_time=cutoff,
                source="synthetic",
                formal_eligible=definition.information_class is InformationClass.CORE,
            )
        )
    feature_payload = "".join(
        json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in feature_rows
    )
    (universe / "features.jsonl").write_text(feature_payload, encoding="utf-8")
    registry = AssetRegistry(("000001.SZ",))
    missing = {
        row.feature_name: (row.missing_flag,) for row in feature_rows
    }
    mask = MaskBundle(
        signal_date=signal,
        asset_ids=registry.asset_ids,
        asset_registry_hash=registry.stable_hash(),
        member=(True,),
        observed=(True,),
        feature_missing=missing,
        label_available=(True,),
        buyable=(True,),
        sellable=(True,),
        loss=(True,),
        evaluation=(True,),
    )
    (universe / "masks.jsonl").write_text(
        json.dumps(mask.to_dict()) + "\n", encoding="utf-8"
    )
    calendar = (
        signal,
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 8),
        date(2025, 1, 9),
        date(2025, 1, 10),
        date(2025, 1, 13),
    )
    label = CompactLabel(
        signal_date=signal,
        ts_code="000001.SZ",
        horizon=5,
        entry_date=calendar[1],
        exit_date=calendar[6],
        open_to_open_return=0.1,
        benchmark_return=0.02,
        trading_calendar_hash=canonical_hash(calendar),
        signal_index=0,
        entry_index=1,
        exit_index=6,
    )
    (universe / "labels.jsonl").write_text(
        json.dumps(label.to_dict()) + "\n", encoding="utf-8"
    )
    (universe / "membership.jsonl").write_text("{}\n", encoding="utf-8")
    loader = CanonicalDatasetLoader(root, "CSI300")
    sample = next(loader.iter_tabular_samples())
    assert sample.target == pytest.approx(0.08)
    master = CausalAssetMaster(
        registry=registry,
        universe="CSI300",
        known_through=signal,
        source_membership_hash="a" * 64,
    )
    panel = loader.load_panel_window(dates=(signal,), asset_master=master)
    assert panel.asset_master.asset_ids == ("000001.SZ",)
    assert panel.values["close"] == ((1.0,),)
