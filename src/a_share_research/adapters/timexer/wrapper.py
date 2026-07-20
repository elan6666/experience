"""Pinned TimeXer forward wrapper; author source remains read-only."""

from a_share_research.adapters.common.runtime import ExternalForecastAdapter


class TimeXerAdapter(ExternalForecastAdapter):
    MODEL_NAME = "timexer"
    EXPECTED_COMMIT = "76011909357972bd55a27adba2e1be994d81b327"
