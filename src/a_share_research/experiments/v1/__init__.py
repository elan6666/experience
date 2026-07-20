"""Frozen V1 information-ablation registry contracts."""

from a_share_research.experiments.v1.registry import (
    CellAction,
    CellDisposition,
    ComparisonPair,
    InformationSet,
    TrainingSignature,
    V1Cell,
    V1Registry,
    build_v1_registry,
    load_v1_blueprint,
    training_signatures_from_blueprint,
    validate_v1_blueprint,
)

__all__ = [
    "CellAction",
    "CellDisposition",
    "ComparisonPair",
    "InformationSet",
    "TrainingSignature",
    "V1Cell",
    "V1Registry",
    "build_v1_registry",
    "load_v1_blueprint",
    "training_signatures_from_blueprint",
    "validate_v1_blueprint",
]
