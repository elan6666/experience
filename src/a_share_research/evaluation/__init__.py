"""Common model-independent evaluation."""

from .metrics import (
    average_ranks,
    evaluate_formal_predictions,
    evaluate_predictions,
    freeze_common_support,
    spearman,
)
from .schema import EvaluationFrequency, OutcomeMode, PredictionScorecard, SupportMode
from .statistics import (
    AttemptFamily,
    DeflatedSharpeRecord,
    MeanInference,
    aggregate_seed_estimates,
    benjamini_hochberg_adjust,
    deflated_sharpe,
    holm_adjust,
    moving_block_bootstrap_mean,
    newey_west_mean,
)

__all__ = [
    "EvaluationFrequency",
    "OutcomeMode",
    "PredictionScorecard",
    "SupportMode",
    "AttemptFamily",
    "DeflatedSharpeRecord",
    "MeanInference",
    "aggregate_seed_estimates",
    "average_ranks",
    "benjamini_hochberg_adjust",
    "deflated_sharpe",
    "evaluate_formal_predictions",
    "evaluate_predictions",
    "freeze_common_support",
    "holm_adjust",
    "moving_block_bootstrap_mean",
    "newey_west_mean",
    "spearman",
]
