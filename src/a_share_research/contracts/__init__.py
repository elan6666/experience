"""Versioned, model-independent research contracts."""

from a_share_research.contracts.base import CanonicalModel, ContractError, canonical_hash
from a_share_research.contracts.data import (
    DailyMarket,
    Eligibility,
    FeatureGroup,
    FormalFeatureManifest,
    Label,
    MarketState,
    PITFeature,
    SecurityMaster,
    UniverseMembership,
)
from a_share_research.contracts.masks import AssetRegistry, MaskBundle, validate_mask_series
from a_share_research.contracts.portfolio import (
    CashSnapshot,
    EligibilityEvidence,
    ExecutionCalendarReceipt,
    FillSide,
    HoldingSnapshot,
    PortfolioFill,
    PortfolioLedger,
    assert_ledger_matches_run,
    eligibility_evidence_hash,
    execution_calendar_manifest_hash,
    execution_calendar_receipt_id,
)
from a_share_research.contracts.prediction import CoverageState, PredictionFrame, PredictionRecord
from a_share_research.contracts.run import RunManifest

__all__ = [
    "AssetRegistry",
    "CanonicalModel",
    "CashSnapshot",
    "ContractError",
    "CoverageState",
    "DailyMarket",
    "Eligibility",
    "FeatureGroup",
    "FormalFeatureManifest",
    "FillSide",
    "HoldingSnapshot",
    "EligibilityEvidence",
    "ExecutionCalendarReceipt",
    "Label",
    "MarketState",
    "MaskBundle",
    "PITFeature",
    "PortfolioFill",
    "PortfolioLedger",
    "PredictionFrame",
    "PredictionRecord",
    "RunManifest",
    "SecurityMaster",
    "UniverseMembership",
    "canonical_hash",
    "assert_ledger_matches_run",
    "eligibility_evidence_hash",
    "execution_calendar_receipt_id",
    "execution_calendar_manifest_hash",
    "validate_mask_series",
]
