"""Model-independent CSI300 market-state risk budgets for V2 B1."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

from a_share_research.contracts import ContractError, MarketState
from a_share_research.contracts.base import CanonicalModel, require_finite
from a_share_research.data.market_state import SharedMarketState

from .intents import TargetFrame, TargetWeight

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_BUDGETS = (1.0, 0.6, 0.3, 0.0)


@dataclass(frozen=True)
class RiskBudgetPolicy(CanonicalModel):
    """Thresholds calibrated before V2 from one shared CSI300 state table."""

    SCHEMA_NAME: ClassVar[str] = "v2_risk_budget_policy"

    version: str
    market_state_hash: str
    calibration_data_hash: str
    calibrated_through: date
    feature_weights: dict[str, float]
    full_if_score_at_most: float
    sixty_if_score_at_most: float
    thirty_if_score_at_most: float
    source_universe: str = "CSI300"
    calibration_partition: str = "TRAIN_2019_2024_PLUS_VALIDATION_2025"

    def validate(self) -> None:
        if not self.version or self.source_universe != "CSI300":
            raise ContractError("risk policy must be versioned and CSI300-sourced")
        for name in ("market_state_hash", "calibration_data_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if self.calibrated_through > date(2025, 12, 31):
            raise ContractError("risk thresholds cannot be calibrated on viewed 2026 data")
        if self.calibration_partition != "TRAIN_2019_2024_PLUS_VALIDATION_2025":
            raise ContractError("risk threshold calibration partition is not frozen")
        if not self.feature_weights or any(not name for name in self.feature_weights):
            raise ContractError("risk policy requires named market-state features")
        for weight in self.feature_weights.values():
            require_finite(weight, "risk feature weight")
        if not any(weight != 0 for weight in self.feature_weights.values()):
            raise ContractError("risk policy requires at least one non-zero feature weight")
        thresholds = (
            self.full_if_score_at_most,
            self.sixty_if_score_at_most,
            self.thirty_if_score_at_most,
        )
        for threshold in thresholds:
            require_finite(threshold, "risk threshold")
        if not thresholds[0] < thresholds[1] < thresholds[2]:
            raise ContractError("risk thresholds must be strictly increasing")

    def budget(self, score: float) -> float:
        require_finite(score, "risk score")
        if score <= self.full_if_score_at_most:
            return 1.0
        if score <= self.sixty_if_score_at_most:
            return 0.6
        if score <= self.thirty_if_score_at_most:
            return 0.3
        return 0.0


@dataclass(frozen=True)
class RiskBudgetPoint(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "v2_risk_budget_point"

    signal_date: date
    risk_score: float
    equity_budget: float
    market_state_hash: str
    policy_hash: str

    def validate(self) -> None:
        require_finite(self.risk_score, "risk score")
        if self.equity_budget not in ALLOWED_BUDGETS:
            raise ContractError("equity budget must be exactly 100/60/30/0 percent")
        for name in ("market_state_hash", "policy_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")


@dataclass(frozen=True)
class RiskBudgetSchedule(CanonicalModel):
    """A content-addressed schedule reused unchanged by all models and pools."""

    SCHEMA_NAME: ClassVar[str] = "v2_risk_budget_schedule"

    market_state_hash: str
    policy_hash: str
    points: tuple[RiskBudgetPoint, ...]

    def validate(self) -> None:
        for name in ("market_state_hash", "policy_hash"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ContractError(f"{name} must be SHA-256")
        if not self.points:
            raise ContractError("risk budget schedule cannot be empty")
        dates = tuple(point.signal_date for point in self.points)
        if tuple(sorted(set(dates))) != dates:
            raise ContractError("risk budget dates must be unique and increasing")
        for point in self.points:
            point.validate()
            if (
                point.market_state_hash != self.market_state_hash
                or point.policy_hash != self.policy_hash
            ):
                raise ContractError("risk budget point is not anchored to the schedule")

    @property
    def by_date(self) -> dict[date, float]:
        return {point.signal_date: point.equity_budget for point in self.points}


def _state_rows_by_date(rows: tuple[MarketState, ...]) -> dict[date, dict[str, MarketState]]:
    grouped: dict[date, dict[str, MarketState]] = defaultdict(dict)
    for row in rows:
        row.validate()
        if row.feature_name in grouped[row.asof_date]:
            raise ContractError("duplicate market-state feature on one date")
        grouped[row.asof_date][row.feature_name] = row
    return grouped


def build_shared_risk_budget_schedule(
    *,
    shared_state: SharedMarketState,
    policy: RiskBudgetPolicy,
    signal_dates: tuple[date, ...],
) -> RiskBudgetSchedule:
    """Build B1 once; consumers may not provide model or universe coverage."""
    policy.validate()
    # ``policy.market_state_hash`` anchors the <=2025 calibration snapshot.
    # The scoring table may later append genuinely prospective rows, so its
    # content hash must not be required to equal the frozen calibration hash.
    # Feature names and the CSI300-only SharedMarketState contract are the
    # stable interface between calibration and scoring.
    if tuple(sorted(set(signal_dates))) != signal_dates or not signal_dates:
        raise ContractError("risk budget signal dates must be unique and increasing")
    grouped = _state_rows_by_date(shared_state.rows)
    policy_hash = policy.stable_hash()
    points: list[RiskBudgetPoint] = []
    for signal_date in signal_dates:
        state = grouped.get(signal_date)
        if state is None:
            raise ContractError("risk-budget date is absent from shared CSI300 state")
        missing = set(policy.feature_weights) - set(state)
        if missing:
            raise ContractError(f"risk-budget date lacks frozen features: {sorted(missing)}")
        score = math.fsum(
            state[name].value * weight for name, weight in policy.feature_weights.items()
        )
        points.append(
            RiskBudgetPoint(
                signal_date=signal_date,
                risk_score=score,
                equity_budget=policy.budget(score),
                market_state_hash=shared_state.stable_hash,
                policy_hash=policy_hash,
            )
        )
    return RiskBudgetSchedule(shared_state.stable_hash, policy_hash, tuple(points))


def apply_shared_risk_budget(
    *,
    always_full_targets: TargetFrame,
    schedule: RiskBudgetSchedule,
    output_run_id: str,
) -> TargetFrame:
    """Scale frozen B0 intents without reading scores, model identity or pool coverage."""
    always_full_targets.validate()
    schedule.validate()
    if not output_run_id:
        raise ContractError("B1 output_run_id is required")
    expected_dates = tuple(
        sorted(
            date.fromisoformat(value)
            for value in always_full_targets.cash_weight_by_date
        )
    )
    if expected_dates != tuple(point.signal_date for point in schedule.points):
        raise ContractError("B0 targets and shared risk schedule dates differ")
    targets_by_date: dict[date, list[TargetWeight]] = defaultdict(list)
    for target in always_full_targets.targets:
        targets_by_date[target.signal_date].append(target)
    scaled: list[TargetWeight] = []
    cash: dict[str, float] = {}
    for point in schedule.points:
        date_text = point.signal_date.isoformat()
        source_targets = targets_by_date.get(point.signal_date, [])
        source_total = math.fsum(target.weight for target in source_targets)
        if always_full_targets.cash_weight_by_date[date_text] != 0 or not math.isclose(
            source_total, 1.0, abs_tol=1e-10
        ):
            raise ContractError("B1 input must be the frozen always-full B0 intent")
        if point.equity_budget > 0 and not source_targets:
            raise ContractError("non-zero risk budget cannot silently become all cash")
        scaled.extend(
            TargetWeight(
                signal_date=target.signal_date,
                ts_code=target.ts_code,
                weight=target.weight * point.equity_budget,
            )
            for target in source_targets
            if point.equity_budget > 0
        )
        cash[date_text] = 1.0 - point.equity_budget
    frame = TargetFrame(output_run_id, tuple(scaled), cash)
    frame.validate()
    return frame
