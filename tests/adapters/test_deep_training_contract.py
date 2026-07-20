"""Deep training contracts; execute only on the approved server."""

from pathlib import PurePosixPath

import pytest

from a_share_research.adapters.common import (
    AdapterContractError,
    DeepRuntimePolicy,
    OfficialSemantics,
    RunIsolation,
    eligible_target_mask,
)


def test_official_semantics_are_fail_closed() -> None:
    OfficialSemantics()
    with pytest.raises(AdapterContractError, match="drifted"):
        OfficialSemantics(optimizer="AdamW")
    with pytest.raises(AdapterContractError, match="drifted"):
        OfficialSemantics(test_visible_during_fit=True)


def test_observed_and_label_available_are_both_required_for_loss() -> None:
    assert eligible_target_mask(
        ((True, True, False), (True, False, True)),
        ((True, False, True), (False, True, True)),
    ) == ((True, False, False), (False, False, True))
    with pytest.raises(AdapterContractError, match="no observed target"):
        eligible_target_mask(((True,),), ((False,),))


def test_runtime_capacity_and_gpu_outputs_are_frozen() -> None:
    policy = DeepRuntimePolicy()
    policy.validate_asset_count(653)
    with pytest.raises(AdapterContractError, match="runtime capacity"):
        policy.validate_asset_count(654)

    RunIsolation(
        model="itransformer",
        universe="CSI300",
        gate="A3",
        seed=20260719,
        physical_gpu=0,
        output_root=PurePosixPath("/runs/itransformer/CSI300/A3/20260719"),
        checkpoint_root=PurePosixPath("/checkpoints/itransformer/CSI300/A3/20260719"),
    )
    with pytest.raises(AdapterContractError, match="frozen physical GPU"):
        RunIsolation(
            model="itransformer",
            universe="CSI300",
            gate="A3",
            seed=20260719,
            physical_gpu=1,
            output_root=PurePosixPath("/runs/itransformer/CSI300/A3/20260719"),
            checkpoint_root=PurePosixPath(
                "/checkpoints/itransformer/CSI300/A3/20260719"
            ),
        )
