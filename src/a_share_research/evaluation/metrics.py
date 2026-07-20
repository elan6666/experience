"""Model-independent prediction metrics on strictly paired observations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date

from a_share_research.contracts import (
    ContractError,
    CoverageState,
    FormalFeatureManifest,
    Label,
    PredictionFrame,
    RunManifest,
)
from a_share_research.protocol import ProtocolSpec
from a_share_research.quality import assert_formal_rankable

from .schema import (
    EvaluationFrequency,
    OutcomeMode,
    PredictionScorecard,
    SupportMode,
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ContractError("metric requires at least one value")
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def average_ranks(values: list[float]) -> list[float]:
    """Stable 1-based ranks with average ranks for exact ties."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = ((cursor + 1) + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    return covariance / math.sqrt(left_ss * right_ss)


def spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(average_ranks(left), average_ranks(right))


def _group_means(rows: list[tuple[float, float]], group_count: int) -> tuple[float, ...]:
    if group_count < 2:
        raise ContractError("group_count must be at least two")
    # Python's stable sort preserves the caller's ts_code order for score ties;
    # realized outcomes must never break ties because that would leak labels.
    ordered = sorted(rows, key=lambda row: row[0])
    actual_groups = min(group_count, len(ordered))
    groups: list[list[float]] = [[] for _ in range(actual_groups)]
    for position, (_, outcome) in enumerate(ordered):
        group = min(actual_groups - 1, position * actual_groups // len(ordered))
        groups[group].append(outcome)
    return tuple(_mean(group) for group in groups)


def _cross_sectional_group_means(
    paired: Mapping[date, list[tuple[float, float]]], group_count: int
) -> tuple[float, ...]:
    """Form portfolios within date, then average like-numbered groups over time."""
    effective_groups = min(group_count, min(len(rows) for rows in paired.values()))
    if effective_groups < 2:
        return ()
    by_group: list[list[float]] = [[] for _ in range(effective_groups)]
    for signal_date in sorted(paired):
        date_groups = _group_means(paired[signal_date], effective_groups)
        for index, value in enumerate(date_groups):
            by_group[index].append(value)
    return tuple(_mean(values) for values in by_group)


def _monotone_fraction(group_returns: tuple[float, ...]) -> float | None:
    if len(group_returns) < 2:
        return None
    comparisons = [
        right >= left for left, right in zip(group_returns, group_returns[1:])
    ]
    return sum(comparisons) / len(comparisons)


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def evaluate_predictions(
    *,
    predictions: PredictionFrame,
    labels: Iterable[Label],
    frequency: EvaluationFrequency,
    support: SupportMode,
    outcome: OutcomeMode,
    eligible_keys: set[tuple[date, str]],
    common_support_keys: set[tuple[date, str]] | None = None,
    group_count: int = 5,
) -> PredictionScorecard:
    """Evaluate only explicitly eligible, horizon-matched, paired rows.

    COMMON support requires an externally frozen intersection. NATIVE support uses
    the model's scored rows. This function never constructs common support from a
    single model, preventing a model-specific comparison set.
    """
    predictions.validate()
    if not isinstance(frequency, EvaluationFrequency):
        raise TypeError("frequency must use EvaluationFrequency")
    if not isinstance(support, SupportMode):
        raise TypeError("support must use SupportMode")
    if not isinstance(outcome, OutcomeMode):
        raise TypeError("outcome must use OutcomeMode")
    if support is SupportMode.COMMON and common_support_keys is None:
        raise ContractError("COMMON evaluation requires a frozen common support set")
    allowed = eligible_keys
    if support is SupportMode.COMMON:
        allowed = eligible_keys & (common_support_keys or set())
    label_by_key: dict[tuple[date, str], Label] = {}
    for label in labels:
        label.validate()
        if label.horizon != frequency.horizon:
            continue
        key = (label.signal_date, label.ts_code)
        if key in label_by_key:
            raise ContractError(f"duplicate label key for horizon: {key}")
        label_by_key[key] = label
    scored = {
        (record.signal_date, record.ts_code): record.score
        for record in predictions.records
        if record.coverage_state is CoverageState.SCORED
    }
    denominator = len(allowed & set(label_by_key))
    if denominator == 0:
        raise ContractError("evaluation support contains no eligible labelled rows")
    paired: dict[date, list[tuple[float, float]]] = defaultdict(list)
    for key in sorted(allowed & set(label_by_key) & set(scored)):
        score = scored[key]
        if score is None:
            raise ContractError("SCORED row unexpectedly lacks a score")
        label = label_by_key[key]
        target = (
            label.open_to_open_return
            if outcome is OutcomeMode.ABSOLUTE
            else label.relative_return
        )
        paired[key[0]].append((score, target))
    if not paired:
        raise ContractError("no scored rows overlap the frozen eligible support")
    all_rows = [row for signal_date in sorted(paired) for row in paired[signal_date]]
    errors = [score - target for score, target in all_rows]
    date_ics: list[float] = []
    excluded_constant_dates = 0
    for signal_date in sorted(paired):
        rows = paired[signal_date]
        rank_ic = spearman(
            [score for score, _ in rows],
            [target for _, target in rows],
        )
        if rank_ic is None:
            excluded_constant_dates += 1
        else:
            date_ics.append(rank_ic)
    mean_ic = _mean(date_ics) if date_ics else None
    ic_std = _sample_std(date_ics)
    icir = None
    if mean_ic is not None and len(date_ics) >= 2 and ic_std > 0:
        icir = mean_ic / ic_std
    group_returns = _cross_sectional_group_means(paired, group_count)
    return PredictionScorecard(
        run_id=predictions.run_id,
        frequency=frequency,
        support=support,
        outcome=outcome,
        horizon=frequency.horizon,
        paired_dates=len(paired),
        paired_rows=len(all_rows),
        coverage=len(all_rows) / denominator,
        rank_ic=mean_ic,
        icir=icir,
        mae=_mean([abs(error) for error in errors]),
        rmse=math.sqrt(_mean([error**2 for error in errors])),
        sign_accuracy=_mean(
            [float(_sign(score) == _sign(target)) for score, target in all_rows]
        ),
        group_returns=group_returns,
        monotone_fraction=_monotone_fraction(group_returns),
        excluded_constant_dates=excluded_constant_dates,
    )


def freeze_common_support(
    frames: Mapping[str, PredictionFrame],
    eligible_keys: set[tuple[date, str]],
) -> set[tuple[date, str]]:
    """Return the exact scored intersection for a predeclared model family."""
    if not frames:
        raise ContractError("common support requires a non-empty model family")
    scored_sets: list[set[tuple[date, str]]] = []
    for name, frame in sorted(frames.items()):
        if not name:
            raise ContractError("model family names cannot be empty")
        frame.validate()
        scored_sets.append(
            {
                (record.signal_date, record.ts_code)
                for record in frame.records
                if record.coverage_state is CoverageState.SCORED
            }
        )
    return eligible_keys.intersection(*scored_sets)


def evaluate_formal_predictions(
    *,
    manifest: RunManifest,
    protocol: ProtocolSpec,
    feature_manifest: FormalFeatureManifest,
    predictions: PredictionFrame,
    labels: Iterable[Label],
    frequency: EvaluationFrequency,
    support: SupportMode,
    outcome: OutcomeMode,
    eligible_keys: set[tuple[date, str]],
    common_support_keys: set[tuple[date, str]] | None = None,
    group_count: int = 5,
) -> PredictionScorecard:
    """Formal entry point: verify registration and protocol before metrics."""
    assert_formal_rankable(
        manifest=manifest,
        protocol=protocol,
        feature_manifest=feature_manifest,
    )
    predictions.validate()
    if predictions.run_id != manifest.run_id:
        raise ContractError("prediction frame and RunManifest run_id differ")
    if manifest.prediction_hash != predictions.stable_hash():
        raise ContractError("prediction frame hash does not match RunManifest")
    return evaluate_predictions(
        predictions=predictions,
        labels=labels,
        frequency=frequency,
        support=support,
        outcome=outcome,
        eligible_keys=eligible_keys,
        common_support_keys=common_support_keys,
        group_count=group_count,
    )
