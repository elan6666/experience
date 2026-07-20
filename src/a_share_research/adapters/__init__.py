"""External, model-fidelity preserving A-share adapters."""

from a_share_research.adapters.common import (
    AdapterContractError,
    CausalAssetMaster,
    FeaturePackingSchema,
    InformationGate,
    PackedWindow,
    PanelWindow,
    PredictionBatch,
    build_causal_asset_master,
    export_prediction_batches,
    pack_feature_window,
)

__all__ = [
    "AdapterContractError",
    "CausalAssetMaster",
    "FeaturePackingSchema",
    "InformationGate",
    "PackedWindow",
    "PanelWindow",
    "PredictionBatch",
    "build_causal_asset_master",
    "export_prediction_batches",
    "pack_feature_window",
]

