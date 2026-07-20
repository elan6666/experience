"""B3 admission-only tests; no fusion model is implemented in Plan011 static scope."""

import pytest

from a_share_research.contracts import ContractError
from a_share_research.portfolio import (
    FusionGatePolicy,
    FusionGateStatus,
    FusionPairEvidence,
    evaluate_fusion_gate,
)


def _pair(
    left: str,
    right: str,
    *,
    left_pass: bool = True,
    right_pass: bool = True,
    error_correlation: float = 0.4,
    holding_overlap: float = 0.5,
) -> FusionPairEvidence:
    return FusionPairEvidence(
        left_model=left,
        right_model=right,
        left_v1_pass=left_pass,
        right_v1_pass=right_pass,
        error_correlation=error_correlation,
        holding_overlap=holding_overlap,
        validation_evidence_hash="a" * 64,
    )


def _policy() -> FusionGatePolicy:
    return FusionGatePolicy(
        version="fusion-gate-v1",
        minimum_v1_pass_models=2,
        maximum_error_correlation=0.8,
        maximum_holding_overlap=0.75,
    )


def test_b3_is_not_run_when_fewer_than_two_v1_models_pass() -> None:
    decision = evaluate_fusion_gate(
        pair_evidence=(_pair("fact", "ridge", right_pass=False),),
        policy=_policy(),
    )
    assert decision.status is FusionGateStatus.NOT_RUN
    assert decision.admitted_models == ()


def test_b3_is_not_run_when_errors_or_holdings_are_redundant() -> None:
    decision = evaluate_fusion_gate(
        pair_evidence=(
            _pair("fact", "ridge", error_correlation=0.95, holding_overlap=0.9),
        ),
        policy=_policy(),
    )
    assert decision.status is FusionGateStatus.NOT_RUN
    assert "redundant" in decision.reason


def test_b3_gate_only_marks_complementary_validation_pair_eligible() -> None:
    decision = evaluate_fusion_gate(
        pair_evidence=(
            _pair("fact", "ridge"),
            _pair("itransformer", "ridge", error_correlation=0.9),
        ),
        policy=_policy(),
    )
    assert decision.status is FusionGateStatus.ELIGIBLE
    assert decision.admitted_models == ("fact", "ridge")
    assert decision.qualifying_pairs == (("fact", "ridge"),)


def test_b3_rejects_inconsistent_model_pass_evidence() -> None:
    with pytest.raises(ContractError, match="inconsistent V1 pass"):
        evaluate_fusion_gate(
            pair_evidence=(
                _pair("fact", "ridge"),
                _pair("itransformer", "ridge", right_pass=False),
            ),
            policy=_policy(),
        )


def test_b3_evidence_hash_is_independent_of_pair_input_order() -> None:
    first = _pair("fact", "ridge")
    second = _pair("itransformer", "ridge")
    forward = evaluate_fusion_gate(pair_evidence=(first, second), policy=_policy())
    reverse = evaluate_fusion_gate(pair_evidence=(second, first), policy=_policy())
    assert forward.evidence_hash == reverse.evidence_hash
    assert forward.stable_hash() == reverse.stable_hash()
