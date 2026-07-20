"""External no-skill and simple-signal baselines."""

from .builders import (
    cash_baseline,
    eligible_equal_weight,
    momentum_prediction_frame,
    top_fraction_targets,
)
from .contracts import (
    BaselineKind,
    IndexReference,
    IndexReferencePoint,
    IndexReturnKind,
    MomentumObservation,
)

__all__ = [
    "BaselineKind",
    "IndexReference",
    "IndexReferencePoint",
    "IndexReturnKind",
    "MomentumObservation",
    "cash_baseline",
    "eligible_equal_weight",
    "momentum_prediction_frame",
    "top_fraction_targets",
]
