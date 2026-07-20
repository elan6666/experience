from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from a_share_research.contracts import (
    AssetRegistry,
    ContractError,
    FormalFeatureManifest,
    MarketState,
    MaskBundle,
    PITFeature,
)
from a_share_research.data.formal_receipts import (
    generate_formal_feature_receipts,
    information_inputs,
)
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.features.schema import InformationClass, d0_features, feature_schema_hash
from a_share_research.models.tabular.layout import InformationSet
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    canonical = tmp_path / "data/canonical/d0-v1"
    schema = tmp_path / "configs/features/d0-v1.json"
    schema.parent.mkdir(parents=True)
    schema.write_bytes((PROJECT_ROOT / "configs/features/d0-v1.json").read_bytes())
    day = date(2025, 1, 3)
    cutoff = datetime(2025, 1, 3, 16, tzinfo=timezone.utc)
    announced = datetime(2025, 1, 2, 8, tzinfo=timezone.utc)
    code = "000001.SZ"
    registry = AssetRegistry((code,))

    shared_rows = tuple(
        MarketState(
            asof_date=day,
            feature_name=definition.source_field,
            value=float(index + 1),
            source_universe="CSI300",
            source_hash="a" * 64,
        ).to_dict()
        for index, definition in enumerate(d0_features())
        if definition.information_class is InformationClass.S
    )
    shared = canonical / "shared_market_state.jsonl"
    _write_jsonl(shared, shared_rows)
    state_values = {
        row["feature_name"]: row["value"] for row in shared_rows
    }

    features: list[dict[str, object]] = []
    missing: dict[str, bool] = {}
    for index, definition in enumerate(d0_features()):
        is_causal_missing = definition.name == "roe"
        value = None if is_causal_missing else float(index + 1)
        if definition.information_class is InformationClass.S:
            value = state_values[definition.source_field]
        feature = PITFeature(
            asof_date=day,
            ts_code=code,
            feature_name=definition.name,
            feature_group=definition.contract_group,
            value=value,
            source_date=date(2025, 1, 2),
            announce_time=(
                announced
                if definition.information_class is InformationClass.F and value is not None
                else None
            ),
            availability_time=cutoff,
            signal_cutoff_time=cutoff,
            missing_flag=value is None,
            source="fixture",
            formal_eligible=value is not None,
        )
        features.append(feature.to_dict())
        missing[definition.name] = value is None
    bundle = MaskBundle(
        signal_date=day,
        asset_ids=(code,),
        asset_registry_hash=registry.stable_hash(),
        member=(True,),
        observed=(True,),
        feature_missing={name: (value,) for name, value in missing.items()},
        label_available=(True,),
        buyable=(True,),
        sellable=(True,),
        loss=(True,),
        evaluation=(True,),
    )

    csi = canonical / "csi300"
    _write_jsonl(csi / "features.jsonl", tuple(features))
    _write_jsonl(csi / "masks.jsonl", (bundle.to_dict(),))
    _write_jsonl(csi / "membership.jsonl", ({"fixture": "sealed"},))
    _write_jsonl(csi / "labels.jsonl", ({"fixture": "sealed"},))
    row_counts = {name: 1 for name in missing}
    missing_counts = {name: int(value) for name, value in missing.items()}
    _write_json(
        csi / "coverage.json",
        {
            "schema_version": "d0_coverage_input_v1",
            "universe": "CSI300",
            "formal_status": "PENDING_GATE",
            "feature_schema_hash": feature_schema_hash(),
            "shared_market_state_hash": "e" * 64,
            "per_factor_missing_required": True,
            "feature_row_counts": row_counts,
            "feature_missing_counts": missing_counts,
        },
    )

    hashes = {"shared_market_state.jsonl": _sha(shared)}
    for filename in (
        "membership.jsonl",
        "features.jsonl",
        "labels.jsonl",
        "masks.jsonl",
        "coverage.json",
    ):
        hashes[f"csi300/{filename}"] = _sha(csi / filename)
    gates = tuple(
        UniverseGate(
            universe=universe,
            status=(
                ResultState.PASS
                if universe is UniverseClass.CSI300
                else ResultState.BLOCKED
                if universe is UniverseClass.STAR50
                else ResultState.EXPLORATORY_ONLY
            ),
            membership_coverage=1.0 if universe is UniverseClass.CSI300 else 0.0,
            core_coverage=1.0 if universe is UniverseClass.CSI300 else 0.0,
            duplicate_keys=0,
            pit_violations=0,
            label_boundary_violations=0,
            warnings=(
                ("official STAR50 historical membership is incomplete",)
                if universe is UniverseClass.STAR50
                else ()
            ),
        )
        for universe in UniverseClass
    )
    manifest = D0Manifest(
        dataset_id="fixture-d0",
        created_at_utc=datetime(2026, 7, 19, tzinfo=timezone.utc),
        cutoff_date=date(2026, 7, 17),
        raw_snapshot_hashes={"raw": "b" * 64},
        canonical_table_hashes=hashes,
        security_master_hash="c" * 64,
        trading_calendar_hash="d" * 64,
        feature_schema_hash=_sha(schema),
        market_state_hash="e" * 64,
        universe_gates=gates,
        provider_transport_notice="fixture proxy uses plain HTTP",
    )
    d0 = tmp_path / "data/manifests/d0-v1.json"
    _write_json(d0, manifest.to_dict())
    return {
        "d0": d0,
        "canonical": canonical,
        "schema": schema,
        "out": tmp_path / "receipts/d0",
    }


def _reseal_csi(paths: dict[str, Path]) -> None:
    payload = json.loads(paths["d0"].read_text(encoding="utf-8"))
    root = paths["canonical"] / "csi300"
    for filename in ("features.jsonl", "masks.jsonl", "coverage.json"):
        payload["canonical_table_hashes"][f"csi300/{filename}"] = _sha(root / filename)
    _write_json(paths["d0"], payload)


def test_generates_exact_receipts_and_records_blocked_star50(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    audit = generate_formal_feature_receipts(
        d0_manifest_path=paths["d0"],
        canonical_root=paths["canonical"],
        feature_schema_path=paths["schema"],
        out_dir=paths["out"],
    )
    assert audit["universes"]["CSI300"]["decision"] == "GENERATED"
    assert audit["universes"]["STAR50"] == {
        "d0_gate_status": "BLOCKED",
        "hard_violations": {
            "duplicate_keys": 0,
            "pit_violations": 0,
            "label_boundary_violations": 0,
            "feature_schema_violations": 0,
        },
        "warnings": ["official STAR50 historical membership is incomplete"],
        "decision": "NOT_GENERATED",
        "reason": "D0_GATE_BLOCKED",
        "formal_receipts": {},
    }
    assert not tuple(paths["out"].glob("formal-star50-*.json"))
    for information_set in InformationSet:
        path = paths["out"] / f"formal-csi300-{information_set.value.lower()}.json"
        receipt = FormalFeatureManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert tuple(receipt.feature_eligibility) == tuple(
            sorted(information_inputs(information_set))
        )
        assert all(receipt.feature_eligibility.values())
    a1 = json.loads((paths["out"] / "formal-csi300-a1.json").read_text(encoding="utf-8"))
    assert "roe" in a1["feature_eligibility"]
    assert "roe__missing" in a1["feature_eligibility"]
    assert audit["universes"]["CSI300"]["missing_counts"]["roe"] == 1


def test_rejects_absent_independent_feature_mask(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    mask_path = paths["canonical"] / "csi300/masks.jsonl"
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    del payload["feature_missing"]["roe"]
    _write_json(mask_path, payload)
    _reseal_csi(paths)
    with pytest.raises(ContractError, match="independent mask per feature"):
        generate_formal_feature_receipts(
            d0_manifest_path=paths["d0"],
            canonical_root=paths["canonical"],
            feature_schema_path=paths["schema"],
            out_dir=paths["out"],
        )


def test_rejects_nonmissing_value_marked_formally_ineligible(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    feature_path = paths["canonical"] / "csi300/features.jsonl"
    rows = [json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["formal_eligible"] = False
    _write_jsonl(feature_path, tuple(rows))
    _reseal_csi(paths)
    with pytest.raises(ContractError, match="not formally eligible"):
        generate_formal_feature_receipts(
            d0_manifest_path=paths["d0"],
            canonical_root=paths["canonical"],
            feature_schema_path=paths["schema"],
            out_dir=paths["out"],
        )


def test_refuses_to_overwrite_any_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    generate_formal_feature_receipts(
        d0_manifest_path=paths["d0"],
        canonical_root=paths["canonical"],
        feature_schema_path=paths["schema"],
        out_dir=paths["out"],
    )
    with pytest.raises(ContractError, match="already exists"):
        generate_formal_feature_receipts(
            d0_manifest_path=paths["d0"],
            canonical_root=paths["canonical"],
            feature_schema_path=paths["schema"],
            out_dir=paths["out"],
        )


def test_accepts_zero_missing_counts_for_present_features(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    audit = generate_formal_feature_receipts(
        d0_manifest_path=paths["d0"],
        canonical_root=paths["canonical"],
        feature_schema_path=paths["schema"],
        out_dir=paths["out"],
    )
    missing_counts = audit["universes"]["CSI300"]["missing_counts"]
    assert missing_counts["roe"] == 1
    assert missing_counts["return_1d"] == 0
