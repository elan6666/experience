"""V2 B3 admission contract; fusion itself remains intentionally unimplemented."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from a_share_research.contracts import ContractError
from a_share_research.contracts.base import CanonicalModel, canonical_hash, require_finite

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FusionGateStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class FusionPairEvidence(CanonicalModel):
    """Validation-only pair evidence frozen before any B3 weight fitting."""

    SCHEMA_NAME: ClassVar[str] = "v2_fusion_pair_evidence"

    left_model: str
    right_model: str
    left_v1_pass: bool
    right_v1_pass: bool
    error_correlation: float
    holding_overlap: float
    validation_evidence_hash: str

    def validate(self) -> None:
        if not self.left_model or not self.right_model or self.left_model >= self.right_model:
            raise ContractError("fusion pair models must be unique and canonically ordered")
        if type(self.left_v1_pass) is not bool or type(self.right_v1_pass) is not bool:
            raise ContractError("fusion V1 pass evidence must be boolean")
        require_finite(self.error_correlation, "fusion error correlation")
        require_finite(self.holding_overlap, "fusion holding overlap")
        if not -1 <= self.error_correlation <= 1:
            raise ContractError("fusion error correlation must be in [-1, 1]")
        if not 0 <= self.holding_overlap <= 1:
            raise ContractError("fusion holding overlap must be in [0, 1]")
        if not _SHA256.fullmatch(self.validation_evidence_hash):
            raise ContractError("fusion validation evidence hash must be SHA-256")


@dataclass(frozen=True)
class FusionGatePolicy(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_fusion_gate_policy"

    version: str
    minimum_v1_pass_models: int
    maximum_error_correlation: float
    maximum_holding_overlap: float
    selection_partition: str = "VALIDATION_2025"

    def validate(self) -> None:
        if not self.version or self.minimum_v1_pass_models < 2:
            raise ContractError("fusion gate requires a version and at least two V1 passes")
        for name in ("maximum_error_correlation", "maximum_holding_overlap"):
            value = getattr(self, name)
            require_finite(value, name)
            if not 0 <= value <= 1:
                raise ContractError(f"{name} must be in [0, 1]")
        if self.selection_partition != "VALIDATION_2025":
            raise ContractError("fusion gate can use only frozen 2025 validation evidence")


@dataclass(frozen=True)
class FusionGateDecision(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_fusion_gate_decision"

    status: FusionGateStatus
    admitted_models: tuple[str, ...]
    qualifying_pairs: tuple[tuple[str, str], ...]
    reason: str
    policy_hash: str
    evidence_hash: str

    def validate(self) -> None:
        if not isinstance(self.status, FusionGateStatus) or not self.reason:
            raise ContractError("fusion gate decision requires typed status and reason")
        if tuple(sorted(set(self.admitted_models))) != self.admitted_models:
            raise ContractError("fusion admitted models must be unique and sorted")
        if tuple(sorted(set(self.qualifying_pairs))) != self.qualifying_pairs:
            raise ContractError("fusion qualifying pairs must be unique and sorted")
        for name in ("policy_hash", "evidence_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if self.status is FusionGateStatus.ELIGIBLE:
            if len(self.admitted_models) < 2 or not self.qualifying_pairs:
                raise ContractError("eligible fusion needs two complementary V1 models")
        elif self.admitted_models or self.qualifying_pairs:
            raise ContractError("NOT_RUN fusion cannot admit models or pairs")


def evaluate_fusion_gate(
    *,
    pair_evidence: tuple[FusionPairEvidence, ...],
    policy: FusionGatePolicy,
) -> FusionGateDecision:
    """Return ELIGIBLE or explicit NOT_RUN; never fit or average model scores."""
    policy.validate()
    seen_pairs: set[tuple[str, str]] = set()
    pass_by_model: dict[str, bool] = {}
    qualifying: list[tuple[str, str]] = []
    passing_models: set[str] = set()
    for row in pair_evidence:
        row.validate()
        pair = (row.left_model, row.right_model)
        if pair in seen_pairs:
            raise ContractError("duplicate fusion pair evidence")
        seen_pairs.add(pair)
        for model, passed in (
            (row.left_model, row.left_v1_pass),
            (row.right_model, row.right_v1_pass),
        ):
            prior = pass_by_model.setdefault(model, passed)
            if prior is not passed:
                raise ContractError("inconsistent V1 pass evidence for fusion model")
        if row.left_v1_pass:
            passing_models.add(row.left_model)
        if row.right_v1_pass:
            passing_models.add(row.right_model)
        if (
            row.left_v1_pass
            and row.right_v1_pass
            and abs(row.error_correlation) <= policy.maximum_error_correlation
            and row.holding_overlap <= policy.maximum_holding_overlap
        ):
            qualifying.append(pair)
    canonical_evidence = tuple(
        sorted(pair_evidence, key=lambda row: (row.left_model, row.right_model))
    )
    evidence_hash = canonical_hash(canonical_evidence)
    if len(passing_models) < policy.minimum_v1_pass_models:
        return FusionGateDecision(
            status=FusionGateStatus.NOT_RUN,
            admitted_models=(),
            qualifying_pairs=(),
            reason="fewer than two V1 models passed the frozen validation gate",
            policy_hash=policy.stable_hash(),
            evidence_hash=evidence_hash,
        )
    if not qualifying:
        return FusionGateDecision(
            status=FusionGateStatus.NOT_RUN,
            admitted_models=(),
            qualifying_pairs=(),
            reason="V1 errors or holdings are redundant under frozen thresholds",
            policy_hash=policy.stable_hash(),
            evidence_hash=evidence_hash,
        )
    admitted = tuple(sorted({model for pair in qualifying for model in pair}))
    return FusionGateDecision(
        status=FusionGateStatus.ELIGIBLE,
        admitted_models=admitted,
        qualifying_pairs=tuple(sorted(qualifying)),
        reason="at least one validation-only V1 pair is complementary",
        policy_hash=policy.stable_hash(),
        evidence_hash=evidence_hash,
    )
