"""Shared identity, packing and prediction-export boundaries."""

from a_share_research.adapters.common.identity import (
    CausalAssetMaster,
    build_causal_asset_master,
    require_stable_slot_series,
)
from a_share_research.adapters.common.packing import (
    FeaturePackingSchema,
    InformationGate,
    PackedWindow,
    PanelWindow,
    assert_constant_parameter_count,
    pack_feature_window,
)
from a_share_research.adapters.common.predictions import (
    PredictionBatch,
    export_prediction_batches,
)
from a_share_research.adapters.common.runtime import (
    ExternalForecastAdapter,
    ProjectedForecastBoundary,
    UpstreamBinding,
    extract_target_scores,
)
from a_share_research.adapters.common.training_contract import (
    DeepRuntimePolicy,
    OfficialSemantics,
    RunIsolation,
    eligible_target_mask,
)
from a_share_research.adapters.common.types import AdapterContractError

__all__ = [
    "AdapterContractError",
    "CausalAssetMaster",
    "FeaturePackingSchema",
    "ExternalForecastAdapter",
    "DeepRuntimePolicy",
    "InformationGate",
    "PackedWindow",
    "PanelWindow",
    "PredictionBatch",
    "ProjectedForecastBoundary",
    "UpstreamBinding",
    "OfficialSemantics",
    "RunIsolation",
    "assert_constant_parameter_count",
    "build_causal_asset_master",
    "eligible_target_mask",
    "export_prediction_batches",
    "extract_target_scores",
    "pack_feature_window",
    "require_stable_slot_series",
]
