"""Pinned FACT forward wrapper; stable variable order is part of the input contract."""

from a_share_research.adapters.common.runtime import ExternalForecastAdapter
from a_share_research.adapters.common.types import AdapterContractError


class FactAdapter(ExternalForecastAdapter):
    MODEL_NAME = "fact"
    EXPECTED_COMMIT = "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4"

    def require_supported_core_mix(self, core: float) -> None:
        """Pinned upstream has a known frequency-only branch defect; do not repair it here."""
        if core != 0.5:
            raise AdapterContractError("formal FACT adapter requires unmodified upstream core=0.5")

