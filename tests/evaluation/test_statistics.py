import pytest

from a_share_research.contracts import ContractError
from a_share_research.evaluation import (
    AttemptFamily,
    aggregate_seed_estimates,
    benjamini_hochberg_adjust,
    deflated_sharpe,
    holm_adjust,
    moving_block_bootstrap_mean,
    newey_west_mean,
)


def test_hac_and_block_bootstrap_are_seeded_and_finite() -> None:
    values = (0.01, 0.02, -0.01, 0.03, 0.00, 0.01)
    hac = newey_west_mean(values, lag=1)
    first = moving_block_bootstrap_mean(
        values, block_length=2, draws=300, seed=7
    )
    second = moving_block_bootstrap_mean(
        values, block_length=2, draws=300, seed=7
    )
    assert hac.estimate == pytest.approx(sum(values) / len(values))
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_multiple_testing_ledger_cannot_silently_drop_attempts() -> None:
    with pytest.raises(ContractError, match="every registered attempt"):
        AttemptFamily("family", ("a", "b"), {"a": 0.01})
    family = AttemptFamily("family", ("a", "b", "failed"), {"a": 0.01, "b": 0.04, "failed": None})
    holm = holm_adjust(family)
    bh = benjamini_hochberg_adjust(family)
    assert holm == {"a": pytest.approx(0.03), "b": pytest.approx(0.08), "failed": None}
    assert bh == {"a": pytest.approx(0.03), "b": pytest.approx(0.06), "failed": None}


def test_deflated_sharpe_records_non_identification_and_seed_dispersion() -> None:
    record = deflated_sharpe(
        attempt_id="flat",
        returns=(0.0, 0.0, 0.0),
        annualization=52,
        attempted_trials=10,
        annualization_basis="weekly non-overlapping portfolio returns",
    )
    assert record.status == "NOT_IDENTIFIED"
    assert record.probability is None
    center, dispersion = aggregate_seed_estimates({3: 0.1, 1: 0.3, 2: 0.2})
    assert center == pytest.approx(0.2)
    assert dispersion == pytest.approx(0.1)
    with pytest.raises(ContractError, match="requires disclosure"):
        deflated_sharpe(
            attempt_id="overlap",
            returns=(0.01, 0.02, -0.01),
            annualization=252,
            attempted_trials=10,
            annualization_basis="daily observations of 5-day labels",
            overlapping_labels=True,
        )
