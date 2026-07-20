"""Adapter-specific fail-closed exceptions."""

from a_share_research.contracts import ContractError


class AdapterContractError(ContractError):
    """Raised when an adapter would change identity or model semantics silently."""


class AdapterBlockedError(AdapterContractError):
    """Raised before any operation on a blocked upstream implementation."""

