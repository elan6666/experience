"""Approved-provider boundary; no credential handling is implemented here."""

from a_share_research.data.providers.approved_proxy import (
    ApprovedProxyProvider,
    QueryRequest,
    QueryResult,
    assert_private_credential_metadata,
)
from a_share_research.data.providers.raw_store import ImmutableRawStore

__all__ = [
    "ApprovedProxyProvider",
    "ImmutableRawStore",
    "QueryRequest",
    "QueryResult",
    "assert_private_credential_metadata",
]
