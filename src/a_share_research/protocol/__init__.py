"""Time, split and experiment-access guards."""

from a_share_research.protocol.registry import ExperimentRegistry, RegisteredExperiment
from a_share_research.protocol.splits import (
    Partition,
    ProtocolSpec,
    Purpose,
    SplitWindow,
    UniverseClass,
)
from a_share_research.protocol.time_guards import (
    assert_no_future_availability,
    embargoed_dates,
    hash_rows_asof,
    purged_training_dates,
)

__all__ = [
    "ExperimentRegistry",
    "Partition",
    "ProtocolSpec",
    "Purpose",
    "RegisteredExperiment",
    "SplitWindow",
    "UniverseClass",
    "assert_no_future_availability",
    "embargoed_dates",
    "hash_rows_asof",
    "purged_training_dates",
]
