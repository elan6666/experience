"""Server-only D0 acquisition, normalization and manifest helpers."""

from a_share_research.data.eligibility import (
    ExecutionStatus,
    build_execution_receipts,
    build_mask_bundle,
)
from a_share_research.data.industry import (
    IndustryInterval,
    build_industry_intervals,
    industry_at,
    industry_by_date,
)
from a_share_research.data.labels import (
    CompactLabel,
    build_compact_open_labels,
    build_open_to_open_labels,
    compact_label,
)
from a_share_research.data.manifest import D0Manifest, UniverseGate
from a_share_research.data.market_state import (
    IndustryCoverage,
    SharedMarketState,
    assert_shared_market_state_hashes,
    build_shared_market_state,
)

__all__ = [
    "D0Manifest",
    "CompactLabel",
    "ExecutionStatus",
    "IndustryCoverage",
    "IndustryInterval",
    "SharedMarketState",
    "UniverseGate",
    "build_execution_receipts",
    "build_industry_intervals",
    "build_compact_open_labels",
    "build_mask_bundle",
    "build_open_to_open_labels",
    "compact_label",
    "build_shared_market_state",
    "industry_at",
    "industry_by_date",
    "assert_shared_market_state_hashes",
]
