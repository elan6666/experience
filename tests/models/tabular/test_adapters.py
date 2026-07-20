import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from a_share_research.contracts import ContractError, CoverageState, RunManifest
from a_share_research.models.tabular import (
    FeatureGate,
    InformationSet,
    LightGBMAdapter,
    LightGBMConfig,
    RidgeAdapter,
    RidgeConfig,
    TabularSample,
    complete_run_manifest,
    default_feature_layout,
)
from a_share_research.protocol.splits import Partition, Purpose, UniverseClass
from a_share_research.quality.states import ResultState


def _sample(day: date, index: int, *, target: bool = True) -> TabularSample:
    layout = default_feature_layout()
    names = layout.core + layout.fundamental + layout.market_state
    values = {
        name: float((index + 1) * (position % 5 + 1))
        for position, name in enumerate(names)
    }
    values["industry_id"] = float(index % 3)
    values["total_mv"] = float(1000 + index * 25)
    missing = {name: False for name in names}
    return TabularSample(
        day,
        f"{index % 999999:06d}.SZ",
        values,
        missing,
        target=float(index % 7) if target else None,
    )


def test_ridge_is_deterministic_and_exports_uncovered_rows() -> None:
    pytest.importorskip("sklearn")
    start = date(2024, 1, 2)
    training = tuple(_sample(start + timedelta(days=index), index) for index in range(24))
    prediction = tuple(
        _sample(date(2025, 1, 2) + timedelta(days=index), index + 100, target=False)
        for index in range(3)
    ) + (
        TabularSample(
            date(2025, 1, 2),
            "600000.SH",
            {},
            {},
            member=True,
            observed=False,
            complete_history=False,
        ),
    )
    kwargs = {
        "layout": default_feature_layout(),
        "gate": FeatureGate(InformationSet.A0),
        "model_config": RidgeConfig(alpha=0.1),
    }
    first = RidgeAdapter(**kwargs).fit_predict(
        run_id="ridge-synthetic",
        training=training,
        prediction=prediction,
        fit_end=training[-1].signal_date,
        fit_data_hash="a" * 64,
        fold_id="ridge-synthetic-fold",
    )
    second = RidgeAdapter(**kwargs).fit_predict(
        run_id="ridge-synthetic",
        training=training,
        prediction=prediction,
        fit_end=training[-1].signal_date,
        fit_data_hash="a" * 64,
        fold_id="ridge-synthetic-fold",
    )
    assert first.predictions.stable_hash() == second.predictions.stable_hash()
    assert first.predictions.records[-1].coverage_state is CoverageState.NOT_OBSERVED
    assert first.predictions.records[-1].score is None


def test_lightgbm_early_stopping_is_never_implicit() -> None:
    adapter = LightGBMAdapter(
        default_feature_layout(),
        FeatureGate(InformationSet.A0),
        model_config=LightGBMConfig(n_estimators=10, early_stopping_rounds=3),
    )
    training = (_sample(date(2024, 1, 2), 1), _sample(date(2024, 1, 3), 2))
    prediction = (_sample(date(2025, 1, 2), 3, target=False),)
    with pytest.raises(ContractError, match="requires validation"):
        adapter.fit_predict(
            run_id="lgb-no-validation",
            training=training,
            validation=(),
            prediction=prediction,
            fit_end=date(2024, 1, 3),
            validation_end=None,
            fit_data_hash="a" * 64,
            fold_id="lgb-no-validation-fold",
        )


def test_checked_in_model_configs_load_without_implicit_field_drift() -> None:
    root = Path(__file__).parents[3]
    ridge_payload = json.loads(
        (root / "configs/models/ridge-v1.json").read_text(encoding="utf-8")
    )
    lightgbm_payload = json.loads(
        (root / "configs/models/lightgbm-v1.json").read_text(encoding="utf-8")
    )
    assert RidgeConfig.from_mapping(ridge_payload).version == "ridge-v1"
    assert LightGBMConfig.from_mapping(lightgbm_payload).early_stopping_rounds == 100


def test_lightgbm_uses_only_declared_validation_for_stopping() -> None:
    pytest.importorskip("lightgbm")
    start = date(2024, 1, 2)
    training = tuple(_sample(start + timedelta(days=index), index) for index in range(40))
    validation = tuple(
        _sample(date(2025, 1, 2) + timedelta(days=index), index + 100)
        for index in range(8)
    )
    prediction = tuple(
        _sample(date(2025, 2, 2) + timedelta(days=index), index + 200, target=False)
        for index in range(3)
    )
    result = LightGBMAdapter(
        default_feature_layout(),
        FeatureGate(InformationSet.A3),
        model_config=LightGBMConfig(
            n_estimators=30,
            min_child_samples=2,
            early_stopping_rounds=5,
        ),
    ).fit_predict(
        run_id="lgb-synthetic",
        training=training,
        validation=validation,
        prediction=prediction,
        fit_end=training[-1].signal_date,
        validation_end=validation[-1].signal_date,
        fit_data_hash="b" * 64,
        fold_id="lgb-synthetic-fold",
    )
    assert result.predictions.coverage == 1.0
    assert result.diagnostics.n_validation == len(validation)
    assert result.diagnostics.best_iteration is not None


def test_prediction_hash_can_finalize_only_a_matching_manifest() -> None:
    pytest.importorskip("sklearn")
    training = tuple(_sample(date(2024, 1, 2) + timedelta(days=i), i) for i in range(10))
    prediction = (_sample(date(2025, 1, 2), 50, target=False),)
    result = RidgeAdapter(
        default_feature_layout(), FeatureGate(InformationSet.A0)
    ).fit_predict(
        run_id="ridge-manifest",
        training=training,
        prediction=prediction,
        fit_end=training[-1].signal_date,
        fit_data_hash="c" * 64,
        fold_id="ridge-manifest-fold",
    )
    started = datetime(2026, 7, 19, 8, tzinfo=timezone.utc)
    draft = RunManifest(
        run_id="ridge-manifest",
        model="Ridge",
        universe=UniverseClass.TECH32,
        information_set="A0",
        split=Partition.VALIDATION,
        purpose=Purpose.SELECT,
        data_hash="c" * 64,
        asset_registry_hash="b" * 64,
        execution_calendar_manifest_hash="c" * 64,
        feature_schema_hash=default_feature_layout().stable_hash(),
        market_state_hash="d" * 64,
        config_hash=RidgeConfig().stable_hash(),
        code_hash="e" * 64,
        upstream_commit="internal:scikit-learn-1.9.0",
        seed=20260719,
        status=ResultState.EXPLORATORY_ONLY,
        started_at=started,
        completed_at=None,
    )
    completed = complete_run_manifest(
        draft,
        result,
        status=ResultState.EXPLORATORY_ONLY,
        completed_at=started + timedelta(minutes=1),
    )
    assert completed.prediction_hash == result.predictions.stable_hash()
