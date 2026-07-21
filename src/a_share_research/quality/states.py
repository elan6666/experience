"""States distinguish a valid negative result from pipeline failures."""

from __future__ import annotations

from enum import Enum


class ResultState(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    EXPLORATORY_ONLY = "EXPLORATORY_ONLY"
    BLOCKED = "BLOCKED"
    INVALID_DATA = "INVALID_DATA"
    INVALID_PROTOCOL = "INVALID_PROTOCOL"
    ADAPTER_FAIL = "ADAPTER_FAIL"
    TRAIN_FAIL = "TRAIN_FAIL"
    EVAL_FAIL = "EVAL_FAIL"
    VALID_NEGATIVE = "VALID_NEGATIVE"


def is_candidate_state(
    state: ResultState,
    *,
    partition: object,
    universe: object,
) -> bool:
    """Only protocol-valid formal results enter model ranking."""
    from a_share_research.protocol.splits import Partition, UniverseClass

    if not isinstance(partition, Partition) or not isinstance(universe, UniverseClass):
        raise TypeError("is_candidate_state requires typed Partition and UniverseClass context")
    if partition is Partition.LEGACY_VIEWED:
        return False
    if universe in {UniverseClass.TECH32, UniverseClass.TECH90}:
        return False
    return state in {
        ResultState.PASS,
        ResultState.PASS_WITH_WARNING,
        ResultState.VALID_NEGATIVE,
    }


def assert_formal_rankable(
    *,
    manifest: object,
    protocol: object,
    feature_manifest: object,
) -> None:
    """Single formal ranking gate for context, opening and D0 feature evidence."""
    from a_share_research.contracts.data import FormalFeatureManifest
    from a_share_research.contracts.run import RunManifest
    from a_share_research.protocol.splits import ProtocolSpec

    if not isinstance(manifest, RunManifest):
        raise TypeError("formal ranking requires RunManifest")
    if not isinstance(protocol, ProtocolSpec):
        raise TypeError("formal ranking requires ProtocolSpec")
    if not isinstance(feature_manifest, FormalFeatureManifest):
        raise TypeError("formal ranking requires FormalFeatureManifest")
    state = manifest.status
    partition = manifest.split
    universe = manifest.universe
    if not is_candidate_state(state, partition=partition, universe=universe):
        raise ValueError("manifest context is not rankable")
    protocol.assert_manifest_opening(manifest)
    if feature_manifest.d0_manifest_hash != manifest.data_hash:
        raise ValueError("formal feature D0 hash does not match RunManifest data_hash")
    receipt_hash = feature_manifest.require_formal_eligible()
    if getattr(manifest, "formal_feature_manifest_hash", None) != receipt_hash:
        raise ValueError("RunManifest does not reference the supplied formal feature receipt")
