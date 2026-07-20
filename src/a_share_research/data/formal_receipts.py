"""Generate fail-closed formal feature receipts from a sealed final D0.

The small :class:`FormalFeatureManifest` is the artifact consumed by model
runners.  This module creates it only after independently proving the D0 gate,
sealed table bytes, feature schema, PIT rows and one-per-feature missing masks.
Detailed evidence remains in a separate audit receipt so the model contract
does not need to grow data-pipeline concerns.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from a_share_research.contracts import (
    ContractError,
    FormalFeatureManifest,
    MarketState,
    MaskBundle,
    PITFeature,
)
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.features.schema import d0_features, feature_schema_hash
from a_share_research.models.tabular.layout import InformationSet, default_feature_layout
from a_share_research.protocol import UniverseClass
from a_share_research.quality.states import ResultState

FORMAL_UNIVERSES: Final = (UniverseClass.CSI300, UniverseClass.STAR50)
PASS_STATES: Final = frozenset({ResultState.PASS, ResultState.PASS_WITH_WARNING})
_TABLE_NAMES: Final = (
    "membership.jsonl",
    "features.jsonl",
    "labels.jsonl",
    "masks.jsonl",
    "coverage.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"expected JSON object at {path}:{number}")
            yield value


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ContractError(f"formal feature evidence is absent: {label}")
    actual = _sha256(path)
    if actual != expected:
        raise ContractError(f"formal feature evidence hash mismatch: {label}")
    return actual


def information_inputs(information_set: InformationSet) -> tuple[str, ...]:
    """Return the exact model columns admitted by one A0--A3 gate."""
    layout = default_feature_layout()
    names = list(layout.core)
    if information_set.enables_f:
        names.extend(layout.fundamental)
        names.extend(layout.fundamental_missing)
    if information_set.enables_s:
        names.extend(layout.market_state)
    return tuple(names)


@dataclass(frozen=True)
class _UniverseEvidence:
    feature_rows: int
    mask_rows: int
    present_counts: dict[str, int]
    missing_counts: dict[str, int]
    table_hashes: dict[str, str]


def _gate_by_universe(d0: D0Manifest) -> dict[UniverseClass, UniverseGate]:
    gates = {gate.universe: gate for gate in d0.universe_gates}
    missing = set(FORMAL_UNIVERSES) - set(gates)
    if missing:
        names = ", ".join(sorted(universe.value for universe in missing))
        raise ContractError(f"D0 manifest lacks required universe gates: {names}")
    return gates


def _verify_gate(gate: UniverseGate) -> None:
    if gate.status not in PASS_STATES:
        raise ContractError(
            f"formal feature receipt requires a passing D0 gate: "
            f"{gate.universe.value}={gate.status.value}"
        )
    if any(
        (
            gate.duplicate_keys,
            gate.pit_violations,
            gate.label_boundary_violations,
            gate.feature_schema_violations,
        )
    ):
        raise ContractError("passing D0 gate contains a hard data violation")


def _verify_shared_state(
    *, canonical_root: Path, d0: D0Manifest
) -> dict[tuple[date, str], float]:
    path = canonical_root / "shared_market_state.jsonl"
    expected = d0.canonical_table_hashes.get("shared_market_state.jsonl")
    if expected is None:
        raise ContractError("D0 manifest does not seal shared_market_state.jsonl")
    _verify_hash(path, expected, "shared_market_state.jsonl")
    state: dict[tuple[date, str], float] = {}
    for value in _jsonl(path):
        row = MarketState.from_dict(value)
        key = (row.asof_date, row.feature_name)
        if key in state:
            raise ContractError("shared market-state table contains duplicate keys")
        state[key] = row.value
    if not state:
        raise ContractError("shared market-state table is empty")
    return state


def _verify_feature_config(path: Path, d0: D0Manifest) -> str:
    actual_hash = _verify_hash(path, d0.feature_schema_hash, "feature schema config")
    payload = _json_object(path)
    expected_sets = {
        "A0": ["CORE"],
        "A1": ["CORE", "F"],
        "A2": ["CORE", "S"],
        "A3": ["CORE", "F", "S"],
    }
    if payload.get("schema_version") != "d0_feature_catalog_v1":
        raise ContractError("unexpected D0 feature schema version")
    if payload.get("information_sets") != expected_sets:
        raise ContractError("D0 feature schema has unexpected A0-A3 information sets")
    missing_policy = str(payload.get("missing_policy", ""))
    if "independent missing flag per feature" not in missing_policy:
        raise ContractError("D0 feature schema does not require per-feature missing masks")
    return actual_hash


def _verify_universe(
    *,
    universe: UniverseClass,
    canonical_root: Path,
    d0: D0Manifest,
    shared_state: Mapping[tuple[date, str], float],
) -> _UniverseEvidence:
    root = canonical_root / universe.value.lower()
    table_hashes: dict[str, str] = {}
    for filename in _TABLE_NAMES:
        relative = f"{universe.value.lower()}/{filename}"
        expected = d0.canonical_table_hashes.get(relative)
        if expected is None:
            raise ContractError(f"D0 manifest does not seal {relative}")
        table_hashes[relative] = _verify_hash(root / filename, expected, relative)

    coverage = _json_object(root / "coverage.json")
    expected_names = tuple(item.name for item in d0_features())
    expected_set = set(expected_names)
    if coverage.get("schema_version") != "d0_coverage_input_v1":
        raise ContractError("unexpected canonical coverage schema version")
    if coverage.get("universe") != universe.value:
        raise ContractError("coverage receipt names a different universe")
    if coverage.get("formal_status") != "PENDING_GATE":
        raise ContractError("formal canonical coverage is not awaiting the final D0 gate")
    if coverage.get("feature_schema_hash") != feature_schema_hash():
        raise ContractError("coverage receipt does not bind the runtime D0 feature schema")
    if coverage.get("shared_market_state_hash") != d0.market_state_hash:
        raise ContractError("coverage receipt does not bind final shared market state")
    if coverage.get("per_factor_missing_required") is not True:
        raise ContractError("coverage receipt does not require one mask per feature")

    masks: dict[date, tuple[MaskBundle, dict[str, int]]] = {}
    mask_rows = 0
    for value in _jsonl(root / "masks.jsonl"):
        bundle = MaskBundle.from_dict(value)
        if bundle.signal_date in masks:
            raise ContractError("canonical mask table contains duplicate signal dates")
        if set(bundle.feature_missing) != expected_set:
            raise ContractError("canonical mask table lacks an independent mask per feature")
        masks[bundle.signal_date] = (
            bundle,
            {code: index for index, code in enumerate(bundle.asset_ids)},
        )
        mask_rows += 1
    if not masks:
        raise ContractError("canonical mask table is empty")

    definitions = {item.name: item for item in d0_features()}
    state_source_by_feature = {
        item.name: item.source_field
        for item in d0_features()
        if item.information_class.value == "S"
    }
    feature_rows = 0
    row_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    current_group: tuple[date, str] | None = None
    group_names: set[str] = set()

    def finish_group() -> None:
        if current_group is not None and group_names != expected_set:
            raise ContractError("canonical feature group does not contain the full D0 schema")

    for value in _jsonl(root / "features.jsonl"):
        row = PITFeature.from_dict(value)
        definition = definitions.get(row.feature_name)
        if definition is None or row.feature_group is not definition.contract_group:
            raise ContractError("canonical feature row violates the frozen feature schema")
        group = (row.asof_date, row.ts_code)
        if group != current_group:
            finish_group()
            if current_group is not None and group <= current_group:
                raise ContractError("canonical feature groups are duplicated or out of order")
            current_group = group
            group_names = set()
        if row.feature_name in group_names:
            raise ContractError("canonical feature table contains duplicate keys")
        group_names.add(row.feature_name)
        mask_evidence = masks.get(row.asof_date)
        if mask_evidence is None or row.ts_code not in mask_evidence[1]:
            raise ContractError("canonical feature row has no matching mask identity")
        bundle, slots = mask_evidence
        slot = slots[row.ts_code]
        if not bundle.member[slot]:
            raise ContractError("canonical feature row exists for a non-member identity")
        if bundle.feature_missing[row.feature_name][slot] != row.missing_flag:
            raise ContractError("feature value and independent missing mask disagree")
        # A causal absence is admissible evidence.  A value that exists but was
        # declared PIT-ineligible is not.
        if not row.missing_flag and not row.formal_eligible:
            raise ContractError("non-missing canonical feature is not formally eligible")
        if row.feature_name in state_source_by_feature and not row.missing_flag:
            state_key = (row.asof_date, state_source_by_feature[row.feature_name])
            state_value = shared_state.get(state_key)
            if state_value is None or not math.isclose(
                row.value, state_value, rel_tol=0.0, abs_tol=0.0
            ):
                raise ContractError("S feature does not match sealed shared market state")
        feature_rows += 1
        row_counts[row.feature_name] += 1
        missing_counts[row.feature_name] += int(row.missing_flag)
    finish_group()
    if not feature_rows or set(row_counts) != expected_set:
        raise ContractError("canonical feature table does not contain the full D0 schema")

    declared_rows = coverage.get("feature_row_counts")
    declared_missing = coverage.get("feature_missing_counts")
    if not isinstance(declared_rows, dict) or not isinstance(declared_missing, dict):
        raise ContractError("coverage receipt lacks feature/missing count mappings")
    if set(declared_rows) != expected_set or set(declared_missing) - expected_set:
        raise ContractError("coverage count names do not match the D0 feature schema")
    normalized_missing = {
        name: int(declared_missing.get(name, 0)) for name in expected_names
    }
    normalized_actual_rows = {name: row_counts[name] for name in expected_names}
    normalized_actual_missing = {name: missing_counts[name] for name in expected_names}
    if (
        declared_rows != normalized_actual_rows
        or normalized_missing != normalized_actual_missing
    ):
        raise ContractError("coverage counts do not match actual feature/missing rows")
    return _UniverseEvidence(
        feature_rows=feature_rows,
        mask_rows=mask_rows,
        present_counts={name: row_counts[name] - missing_counts[name] for name in expected_names},
        missing_counts=normalized_actual_missing,
        table_hashes=table_hashes,
    )


def _atomic_write_all(payloads: Mapping[Path, object]) -> None:
    for path in payloads:
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise ContractError(f"formal receipt output already exists: {path}")
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_suffix(path.suffix + ".tmp")
            staged.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.append(staged)
        for path in payloads:
            path.with_suffix(path.suffix + ".tmp").replace(path)
            published.append(path)
    except Exception:
        for path in temporary:
            path.unlink(missing_ok=True)
        for path in published:
            path.unlink(missing_ok=True)
        raise


def generate_formal_feature_receipts(
    *,
    d0_manifest_path: Path,
    canonical_root: Path,
    feature_schema_path: Path,
    out_dir: Path,
    audit_out: Path | None = None,
) -> dict[str, Any]:
    """Create A0--A3 receipts only for formal universes whose final D0 passed."""
    d0 = D0Manifest.from_dict(_json_object(d0_manifest_path))
    d0_file_hash = _sha256(d0_manifest_path)
    schema_file_hash = _verify_feature_config(feature_schema_path, d0)
    shared_state = _verify_shared_state(canonical_root=canonical_root, d0=d0)
    gates = _gate_by_universe(d0)
    audit_path = audit_out or out_dir / "formal-feature-generation-audit.json"
    planned_paths = {audit_path}
    for universe in FORMAL_UNIVERSES:
        if gates[universe].status in PASS_STATES:
            planned_paths.update(
                out_dir
                / f"formal-{universe.value.lower()}-{information_set.value.lower()}.json"
                for information_set in InformationSet
            )
    for path in planned_paths:
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise ContractError(f"formal receipt output already exists: {path}")
    audit: dict[str, Any] = {
        "schema_version": "formal_feature_generation_audit_v1",
        "dataset_id": d0.dataset_id,
        "d0_content_hash": d0.content_hash,
        "d0_manifest_file_sha256": d0_file_hash,
        "feature_schema_file_sha256": schema_file_hash,
        "market_state_content_hash": d0.market_state_hash,
        "universes": {},
    }
    outputs: dict[Path, object] = {}
    for universe in FORMAL_UNIVERSES:
        gate = gates[universe]
        universe_audit: dict[str, Any] = {
            "d0_gate_status": gate.status.value,
            "hard_violations": {
                "duplicate_keys": gate.duplicate_keys,
                "pit_violations": gate.pit_violations,
                "label_boundary_violations": gate.label_boundary_violations,
                "feature_schema_violations": gate.feature_schema_violations,
            },
            "warnings": list(gate.warnings),
        }
        if gate.status not in PASS_STATES:
            universe_audit.update(
                {
                    "decision": "NOT_GENERATED",
                    "reason": f"D0_GATE_{gate.status.value}",
                    "formal_receipts": {},
                }
            )
            audit["universes"][universe.value] = universe_audit
            continue
        _verify_gate(gate)
        evidence = _verify_universe(
            universe=universe,
            canonical_root=canonical_root,
            d0=d0,
            shared_state=shared_state,
        )
        receipts: dict[str, Any] = {}
        for information_set in InformationSet:
            names = information_inputs(information_set)
            manifest = FormalFeatureManifest(
                dataset_id=f"{d0.dataset_id}:{universe.value}:{information_set.value}",
                d0_manifest_hash=d0.content_hash,
                feature_eligibility={name: True for name in names},
            )
            manifest.require_formal_eligible()
            path = out_dir / (
                f"formal-{universe.value.lower()}-{information_set.value.lower()}.json"
            )
            outputs[path] = manifest.to_dict()
            receipts[information_set.value] = {
                "path": path.as_posix(),
                "manifest_stable_hash": manifest.stable_hash(),
                "input_names": list(names),
            }
        universe_audit.update(
            {
                "decision": "GENERATED",
                "feature_rows": evidence.feature_rows,
                "mask_rows": evidence.mask_rows,
                "present_counts": evidence.present_counts,
                "missing_counts": evidence.missing_counts,
                "canonical_table_hashes": evidence.table_hashes,
                "formal_receipts": receipts,
            }
        )
        audit["universes"][universe.value] = universe_audit

    # Receipt file SHA values are computed from the exact canonical JSON bytes
    # before publishing, then recorded in the audit generated in the same batch.
    for universe_payload in audit["universes"].values():
        for receipt in universe_payload["formal_receipts"].values():
            path = Path(receipt["path"])
            encoded = (
                json.dumps(outputs[path], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            receipt["file_sha256"] = hashlib.sha256(encoded).hexdigest()
    if audit_path in outputs:
        raise ContractError("audit output collides with a formal feature receipt")
    outputs[audit_path] = audit
    _atomic_write_all(outputs)
    return audit
