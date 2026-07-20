"""Small, dependency-free statistical controls for frozen evaluation series."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import ClassVar

from a_share_research.contracts.base import (
    CanonicalModel,
    ContractError,
    require_finite,
)


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ContractError("statistic requires a non-empty series")
    if any(not math.isfinite(value) for value in values):
        raise ContractError("statistic series must be finite")
    return sum(values) / len(values)


def _sample_variance(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


@dataclass(frozen=True)
class MeanInference(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "mean_inference"

    estimate: float
    standard_error: float
    lower: float
    upper: float
    method: str
    observations: int
    dependence_lag: int

    def validate(self) -> None:
        for name in ("estimate", "standard_error", "lower", "upper"):
            require_finite(getattr(self, name), name)
        if self.standard_error < 0 or self.lower > self.upper:
            raise ContractError("invalid inference interval")
        if not self.method or type(self.observations) is not int or self.observations < 2:
            raise ContractError("inference method and at least two observations are required")
        if type(self.dependence_lag) is not int or not 0 <= self.dependence_lag < self.observations:
            raise ContractError("invalid dependence lag")


def newey_west_mean(
    values: tuple[float, ...],
    *,
    lag: int,
    confidence: float = 0.95,
) -> MeanInference:
    """HAC inference for a mean using Bartlett weights."""
    if len(values) < 2:
        raise ContractError("HAC requires at least two observations")
    if type(lag) is not int or not 0 <= lag < len(values):
        raise ContractError("HAC lag must be in [0, n)")
    if not 0 < confidence < 1:
        raise ContractError("confidence must be in (0, 1)")
    center = _mean(values)
    residuals = tuple(value - center for value in values)
    n = len(values)
    long_run_variance = sum(value * value for value in residuals) / n
    for distance in range(1, lag + 1):
        autocovariance = sum(
            residuals[index] * residuals[index - distance]
            for index in range(distance, n)
        ) / n
        weight = 1 - distance / (lag + 1)
        long_run_variance += 2 * weight * autocovariance
    standard_error = math.sqrt(max(0.0, long_run_variance) / n)
    critical = NormalDist().inv_cdf(0.5 + confidence / 2)
    return MeanInference(
        estimate=center,
        standard_error=standard_error,
        lower=center - critical * standard_error,
        upper=center + critical * standard_error,
        method="NEWEY_WEST",
        observations=n,
        dependence_lag=lag,
    )


def moving_block_bootstrap_mean(
    values: tuple[float, ...],
    *,
    block_length: int,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> MeanInference:
    """Circular moving-block percentile interval with a recorded seed."""
    n = len(values)
    if n < 2:
        raise ContractError("block bootstrap requires at least two observations")
    if type(block_length) is not int or not 1 <= block_length <= n:
        raise ContractError("block_length must be in [1, n]")
    if type(draws) is not int or draws < 200:
        raise ContractError("at least 200 bootstrap draws are required")
    if type(seed) is not int:
        raise ContractError("bootstrap seed must be an integer")
    if not 0 < confidence < 1:
        raise ContractError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    estimates: list[float] = []
    block_count = math.ceil(n / block_length)
    for _ in range(draws):
        sample: list[float] = []
        for _ in range(block_count):
            start = rng.randrange(n)
            sample.extend(values[(start + offset) % n] for offset in range(block_length))
        estimates.append(sum(sample[:n]) / n)
    estimates.sort()
    tail = (1 - confidence) / 2
    lower_index = max(0, min(draws - 1, math.floor(tail * draws)))
    upper_index = max(0, min(draws - 1, math.ceil((1 - tail) * draws) - 1))
    return MeanInference(
        estimate=_mean(values),
        standard_error=math.sqrt(_sample_variance(tuple(estimates))),
        lower=estimates[lower_index],
        upper=estimates[upper_index],
        method=f"MOVING_BLOCK_BOOTSTRAP_SEED_{seed}",
        observations=n,
        dependence_lag=block_length - 1,
    )


@dataclass(frozen=True)
class AttemptFamily(CanonicalModel):
    """Prevents p-value correction from silently omitting failed attempts."""

    SCHEMA_NAME: ClassVar[str] = "attempt_family"

    family_id: str
    attempt_ids: tuple[str, ...]
    p_values: dict[str, float | None]

    def validate(self) -> None:
        if not self.family_id or not self.attempt_ids:
            raise ContractError("attempt family id and attempts are required")
        if len(self.attempt_ids) != len(set(self.attempt_ids)) or any(
            not attempt_id for attempt_id in self.attempt_ids
        ):
            raise ContractError("attempt ids must be unique and non-empty")
        if set(self.p_values) != set(self.attempt_ids):
            raise ContractError("p-value ledger must include every registered attempt")
        for value in self.p_values.values():
            if value is not None and not 0 <= value <= 1:
                raise ContractError("p-values must be in [0, 1]")


def holm_adjust(family: AttemptFamily) -> dict[str, float | None]:
    family.validate()
    observed = sorted(
        ((attempt, value) for attempt, value in family.p_values.items() if value is not None),
        key=lambda pair: (pair[1], pair[0]),
    )
    adjusted: dict[str, float | None] = {attempt: None for attempt in family.attempt_ids}
    running = 0.0
    count = len(family.attempt_ids)
    for rank, (attempt, value) in enumerate(observed):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[attempt] = running
    return adjusted


def benjamini_hochberg_adjust(family: AttemptFamily) -> dict[str, float | None]:
    family.validate()
    observed = sorted(
        ((attempt, value) for attempt, value in family.p_values.items() if value is not None),
        key=lambda pair: (pair[1], pair[0]),
    )
    adjusted: dict[str, float | None] = {attempt: None for attempt in family.attempt_ids}
    running = 1.0
    count = len(family.attempt_ids)
    for rank in range(len(observed), 0, -1):
        attempt, value = observed[rank - 1]
        running = min(running, value * count / rank)
        adjusted[attempt] = min(1.0, running)
    return adjusted


@dataclass(frozen=True)
class DeflatedSharpeRecord(CanonicalModel):
    SCHEMA_NAME: ClassVar[str] = "deflated_sharpe_record"

    attempt_id: str
    observations: int
    annualization: int
    attempted_trials: int
    observed_sharpe: float
    expected_max_sharpe: float
    probability: float | None
    status: str
    annualization_basis: str
    overlapping_labels: bool
    overlap_disclosure: str | None

    def validate(self) -> None:
        if not self.attempt_id or self.status not in {"COMPUTED", "NOT_IDENTIFIED"}:
            raise ContractError("invalid Deflated Sharpe attempt record")
        if not self.annualization_basis:
            raise ContractError("annualization basis is required")
        if type(self.overlapping_labels) is not bool:
            raise ContractError("overlapping_labels must be bool")
        if self.overlapping_labels and not self.overlap_disclosure:
            raise ContractError("overlapping-label annualization requires disclosure")
        if not self.overlapping_labels and self.overlap_disclosure is not None:
            raise ContractError("non-overlapping series cannot carry overlap disclosure")
        for name in ("observations", "annualization", "attempted_trials"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ContractError(f"{name} must be a positive integer")
        for name in ("observed_sharpe", "expected_max_sharpe"):
            require_finite(getattr(self, name), name)
        if self.probability is not None and not 0 <= self.probability <= 1:
            raise ContractError("Deflated Sharpe probability must be in [0, 1]")
        if (self.status == "COMPUTED") != (self.probability is not None):
            raise ContractError("Deflated Sharpe status and probability disagree")


def deflated_sharpe(
    *,
    attempt_id: str,
    returns: tuple[float, ...],
    annualization: int,
    attempted_trials: int,
    annualization_basis: str,
    overlapping_labels: bool = False,
    overlap_disclosure: str | None = None,
) -> DeflatedSharpeRecord:
    """Compute a transparent normal-approximation DSR, or record non-identification."""
    n = len(returns)
    if n < 3 or annualization <= 0 or attempted_trials <= 0:
        raise ContractError("Deflated Sharpe inputs are invalid")
    if not annualization_basis:
        raise ContractError("annualization basis is required")
    if overlapping_labels and not overlap_disclosure:
        raise ContractError("overlapping-label annualization requires disclosure")
    center = _mean(returns)
    variance = _sample_variance(returns)
    if variance == 0:
        return DeflatedSharpeRecord(
            attempt_id=attempt_id,
            observations=n,
            annualization=annualization,
            attempted_trials=attempted_trials,
            observed_sharpe=0.0,
            expected_max_sharpe=0.0,
            probability=None,
            status="NOT_IDENTIFIED",
            annualization_basis=annualization_basis,
            overlapping_labels=overlapping_labels,
            overlap_disclosure=overlap_disclosure,
        )
    standard_deviation = math.sqrt(variance)
    observed = center / standard_deviation * math.sqrt(annualization)
    if attempted_trials == 1:
        expected_max = 0.0
    else:
        # Expected maximum of N standard normals, scaled to annual Sharpe units.
        quantile = min(1 - 1e-12, max(1e-12, 1 - 1 / attempted_trials))
        expected_max = NormalDist().inv_cdf(quantile) * math.sqrt(annualization / n)
    standardized = (observed - expected_max) * math.sqrt(max(1.0, n - 1) / annualization)
    probability = NormalDist().cdf(standardized)
    return DeflatedSharpeRecord(
        attempt_id=attempt_id,
        observations=n,
        annualization=annualization,
        attempted_trials=attempted_trials,
        observed_sharpe=observed,
        expected_max_sharpe=expected_max,
        probability=probability,
        status="COMPUTED",
        annualization_basis=annualization_basis,
        overlapping_labels=overlapping_labels,
        overlap_disclosure=overlap_disclosure,
    )


def aggregate_seed_estimates(seed_values: dict[int, float]) -> tuple[float, float]:
    if len(seed_values) < 2 or any(type(seed) is not int for seed in seed_values):
        raise ContractError("seed aggregation requires at least two integer-labelled seeds")
    values = tuple(seed_values[seed] for seed in sorted(seed_values))
    return _mean(values), math.sqrt(_sample_variance(values))
