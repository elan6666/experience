"""Pinned S4M forward wrapper; author source remains read-only."""

from a_share_research.adapters.common.runtime import ExternalForecastAdapter


class S4MAdapter(ExternalForecastAdapter):
    MODEL_NAME = "s4m"
    EXPECTED_COMMIT = "a718823addd3606e763dfc261174e0135b2535f4"
    NATIVE_OPTIMIZER = "SGD"
