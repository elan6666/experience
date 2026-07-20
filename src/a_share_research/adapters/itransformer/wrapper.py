"""Pinned iTransformer forward wrapper; author source remains read-only."""

from a_share_research.adapters.common.runtime import ExternalForecastAdapter


class ITransformerAdapter(ExternalForecastAdapter):
    MODEL_NAME = "itransformer"
    EXPECTED_COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"

