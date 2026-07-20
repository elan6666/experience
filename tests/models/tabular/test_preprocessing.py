from datetime import date, timedelta

import pytest

from a_share_research.contracts import ContractError
from a_share_research.models.tabular import (
    FeatureGate,
    InformationSet,
    PreprocessingState,
    TabularSample,
    TrainOnlyPreprocessor,
    default_feature_layout,
)


def _sample(day: date, index: int, *, missing_f: str | None = None) -> TabularSample:
    layout = default_feature_layout()
    names = layout.core + layout.fundamental + layout.market_state
    values = {name: float(index + position + 1) for position, name in enumerate(names)}
    values["industry_id"] = float(index % 3)
    values["total_mv"] = float(1000 + index * 10)
    missing = {name: False for name in names}
    if missing_f is not None:
        values[missing_f] = None
        missing[missing_f] = True
    return TabularSample(day, "000001.SZ", values, missing, target=float(index))


def test_preprocessing_state_is_training_only_and_round_trips() -> None:
    start = date(2024, 1, 2)
    training = tuple(_sample(start + timedelta(days=index), index) for index in range(10))
    gate = FeatureGate(InformationSet.A3)
    preprocessor = TrainOnlyPreprocessor(default_feature_layout(), gate)
    state = preprocessor.fit(
        training,
        fit_end=training[-1].signal_date,
        fit_data_hash="a" * 64,
        fold_id="synthetic-fold-0",
    )
    restored = PreprocessingState.from_dict(state.to_dict())
    assert restored.stable_hash() == state.stable_hash()

    past_before = preprocessor.transform((training[0],))
    future = _sample(date(2025, 1, 2), 999)
    preprocessor.transform((future,))
    assert preprocessor.transform((training[0],)) == past_before
    assert preprocessor.state is not None
    assert preprocessor.state.stable_hash() == state.stable_hash()


def test_fit_rejects_any_row_after_declared_cutoff() -> None:
    samples = (_sample(date(2024, 12, 31), 1), _sample(date(2025, 1, 2), 2))
    preprocessor = TrainOnlyPreprocessor(
        default_feature_layout(), FeatureGate(InformationSet.A0)
    )
    with pytest.raises(ContractError, match="after declared training cutoff"):
        preprocessor.fit(
            samples,
            fit_end=date(2024, 12, 31),
            fit_data_hash="a" * 64,
            fold_id="synthetic-fold-0",
        )


def test_factor_missing_indicators_remain_distinct_after_transform() -> None:
    layout = default_feature_layout()
    samples = (
        _sample(date(2024, 1, 2), 1, missing_f="pe_ttm"),
        _sample(date(2024, 1, 3), 2, missing_f="pb"),
    )
    preprocessor = TrainOnlyPreprocessor(layout, FeatureGate(InformationSet.A1))
    preprocessor.fit(
        samples,
        fit_end=date(2024, 1, 3),
        fit_data_hash="b" * 64,
        fold_id="synthetic-fold-1",
    )
    transformed = preprocessor.transform(samples)
    start = len(layout.core) + len(layout.fundamental)
    pe_index = start + layout.fundamental.index("pe_ttm")
    pb_index = start + layout.fundamental.index("pb")
    assert (transformed[0][pe_index], transformed[0][pb_index]) == (1.0, 0.0)
    assert (transformed[1][pe_index], transformed[1][pb_index]) == (0.0, 1.0)
