"""Synthetic deep-runner contracts; execute only on the approved server."""

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from a_share_research.adapters.common import InformationGate
from a_share_research.contracts import AssetRegistry, ContractError, MaskBundle
from a_share_research.experiments.deep_runner import (
    DeepEvidenceFile,
    DeepHyperparameters,
    DeepJobSpec,
    DeepWindowPlan,
    _admissible_label_scores,
    _history_ready,
)
from a_share_research.protocol import UniverseClass

HASH = "a" * 64
IT_COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"
FACT_COMMIT = "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4"
ROOT = Path("/data/yilangliu/a_share_research")


def evidence(name: str) -> DeepEvidenceFile:
    return DeepEvidenceFile(str(ROOT / "receipts" / name), HASH)


def hyperparameters(model: str) -> DeepHyperparameters:
    arguments: dict[str, object] = {
        "use_norm": 1,
        "freq": "h",
        "d_model": 16,
        "d_ff": 32,
        "dropout": 0.0,
        "lradj": "type1",
    }
    if model == "itransformer":
        arguments.update(
            {
                "output_attention": False,
                "embed": "timeF",
                "class_strategy": "projection",
                "factor": 1,
                "n_heads": 4,
                "e_layers": 1,
                "activation": "gelu",
            }
        )
    else:
        arguments.update(
            {
                "task_name": "long_term_forecast",
                "dilation": [1],
                "num_kernels": 2,
                "core": 0.5,
            }
        )
    return DeepHyperparameters(96, 1, 4, 10, 3, 0.001, arguments)


def job(
    *,
    model: str = "itransformer",
    universe: UniverseClass = UniverseClass.CSI300,
    phase: str = "V0",
    gate: InformationGate = InformationGate.A0,
    seed: int = 20260719,
) -> DeepJobSpec:
    gpu = 0 if model == "itransformer" else 1
    commit = IT_COMMIT if model == "itransformer" else FACT_COMMIT
    suffix = Path(model) / universe.value.lower() / gate.value / str(seed)
    return DeepJobSpec(
        phase=phase,
        run_id=(
            f"{phase.lower()}-{gate.value.lower()}-{universe.value.lower()}-"
            f"{model}-seed-{seed}"
        ),
        model=model,
        universe=universe,
        gate=gate,
        seed=seed,
        physical_gpu=gpu,
        canonical_root=str(ROOT / "data" / "canonical" / "d0-v1"),
        upstream_root=str(ROOT / "upstreams" / model),
        output_dir=str(ROOT / "runs" / phase.lower() / suffix),
        checkpoint_dir=str(ROOT / "checkpoints" / phase.lower() / suffix),
        upstream_commit=commit,
        asset_registry_hash=HASH,
        cell_config_hash=HASH,
        hyperparameters=hyperparameters(model),
        d0_manifest=evidence("d0.json"),
        environment_receipt=evidence("environment.json"),
        integrity_receipt=evidence("integrity.json"),
        code_receipt=evidence("code.json"),
        adapter_config=evidence("adapter.json"),
        common_config=evidence("common.json"),
        formal_feature_manifest=(
            None if universe in {UniverseClass.TECH32, UniverseClass.TECH90}
            else evidence("formal.json")
        ),
    )


@pytest.mark.parametrize("model", ("itransformer", "fact"))
@pytest.mark.parametrize("seed", (20260719, 20260720, 20260721))
def test_three_seed_gpu_isolation_is_frozen(model: str, seed: int) -> None:
    job(model=model, seed=seed)


def test_wrong_gpu_and_fact_core_fail_closed() -> None:
    valid = job(model="itransformer")
    with pytest.raises(ContractError, match="frozen physical GPU"):
        DeepJobSpec(**{**valid.__dict__, "physical_gpu": 1})
    fact = job(model="fact")
    bad_hyperparameters = DeepHyperparameters(
        96,
        1,
        4,
        10,
        3,
        0.001,
        {**fact.hyperparameters.author_arguments, "core": 0.0},
    )
    with pytest.raises(ContractError, match="core=0.5"):
        DeepJobSpec(**{**fact.__dict__, "hyperparameters": bad_hyperparameters})


def test_v0_v1_gate_contract_and_technology_receipts() -> None:
    with pytest.raises(ContractError, match="V0 deep jobs are A0"):
        job(phase="V0", gate=InformationGate.A1)
    job(phase="V1", gate=InformationGate.A3)
    job(universe=UniverseClass.TECH32)
    tech = job(universe=UniverseClass.TECH90)
    with pytest.raises(ContractError, match="cannot claim a formal receipt"):
        DeepJobSpec(**{**tech.__dict__, "formal_feature_manifest": evidence("formal.json")})


def test_capacity_hash_is_gate_independent_and_runner_fields_are_owned() -> None:
    parameters = hyperparameters("itransformer")
    assert len(parameters.capacity_hash) == 64
    with pytest.raises(ContractError, match="cannot be overridden"):
        DeepHyperparameters(
            96,
            1,
            4,
            10,
            3,
            0.001,
            {**parameters.author_arguments, "seq_len": 24},
        )


def test_window_plan_uses_only_2019_2024_fit_and_2025_validation() -> None:
    start = date(2017, 1, 6)
    dates = tuple(start + timedelta(days=7 * index) for index in range(470))
    dates = tuple(day for day in dates if day <= date(2025, 12, 26))
    plan = DeepWindowPlan.build(dates, lookback=96)
    assert all(
        date(2019, 1, 1) <= plan.input_dates[index] <= date(2024, 12, 31)
        for index in plan.train_anchor_indices
    )
    assert all(
        date(2025, 1, 1) <= plan.input_dates[index] <= date(2025, 12, 31)
        for index in plan.validation_anchor_indices
    )
    assert plan.validation_anchor_indices


def test_window_plan_rejects_any_2026_model_selection_row() -> None:
    dates = tuple(date(2024, 1, 5) + timedelta(days=7 * index) for index in range(105))
    with pytest.raises(ContractError, match="never admit 2026"):
        DeepWindowPlan.build(dates, lookback=10)


def test_history_window_includes_the_signal_date() -> None:
    registry = AssetRegistry(("000001.SZ",))
    masks = tuple(
        MaskBundle(
            signal_date=date(2025, 1, 3) + timedelta(days=7 * index),
            asset_ids=registry.asset_ids,
            asset_registry_hash=registry.stable_hash(),
            member=(True,),
            observed=(index != 2,),
            feature_missing={"ret_1d": (False,)},
            label_available=(True,),
            buyable=(index != 2,),
            sellable=(index != 2,),
            loss=(index != 2,),
            evaluation=(index != 2,),
        )
        for index in range(4)
    )
    assert _history_ready(masks, anchor_index=2, lookback=2) == (False,)
    assert _history_ready(masks, anchor_index=3, lookback=2) == (False,)


def test_label_scores_purge_both_fold_boundaries() -> None:
    def row(signal: date, exit_date: date) -> SimpleNamespace:
        return SimpleNamespace(
            horizon=5,
            signal_date=signal,
            exit_date=exit_date,
            ts_code="000001.SZ",
            open_to_open_return=0.02,
            benchmark_return=0.01,
        )

    scores = _admissible_label_scores(
        (
            row(date(2024, 12, 20), date(2024, 12, 30)),
            row(date(2024, 12, 27), date(2025, 1, 7)),
            row(date(2025, 12, 19), date(2025, 12, 29)),
            row(date(2025, 12, 26), date(2026, 1, 7)),
        )
    )
    assert set(scores) == {
        (date(2024, 12, 20), "000001.SZ"),
        (date(2025, 12, 19), "000001.SZ"),
    }
