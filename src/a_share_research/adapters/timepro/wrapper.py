"""Pinned TimePro forward wrapper; author source remains read-only."""

from a_share_research.adapters.common.runtime import ExternalForecastAdapter


class TimeProAdapter(ExternalForecastAdapter):
    MODEL_NAME = "timepro"
    EXPECTED_COMMIT = "70a20e5a257b30eb026ee4316293cf4feeb92a1f"
