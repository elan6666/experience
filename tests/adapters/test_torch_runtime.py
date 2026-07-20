"""Optional Torch runtime tests; execute only in a server deep-model environment."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from a_share_research.adapters.common.torch_runtime import (  # noqa: E402
    DeepForecastBatch,
    OutOfPlaceNativeNormalization,
    ProjectedForecastModule,
    SharedPerAssetProjector,
    fit_protocol_safe,
    masked_mse,
)
from a_share_research.adapters.common.types import AdapterContractError  # noqa: E402


class FourArgumentBackbone(torch.nn.Module):
    def __init__(self, assets: int, horizon: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(assets, assets)
        self.horizon = horizon
        self.last_input_shape = None
        self.use_norm = 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.last_input_shape = tuple(x_enc.shape)
        final = self.head(x_enc[:, -1, :])
        return final.unsqueeze(1).expand(-1, self.horizon, -1)


class InPlaceNativeNormalizationBackbone(torch.nn.Module):
    """Minimal copy of the upstream normalization equations for regression testing."""

    def __init__(self, assets: int, horizon: int, *, output_tuple: bool = False) -> None:
        super().__init__()
        self.head = torch.nn.Linear(assets, assets)
        self.horizon = horizon
        self.output_tuple = output_tuple
        self.use_norm = 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        output = self.head(x_enc[:, -1, :]).unsqueeze(1).expand(
            -1, self.horizon, -1
        )
        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.horizon, 1
            )
            output = output + means[:, 0, :].unsqueeze(1).repeat(
                1, self.horizon, 1
            )
        return (output, "attention") if self.output_tuple else output


def test_shared_projector_has_one_stock_token_and_asset_independent_parameters() -> None:
    projector = SharedPerAssetProjector(5)
    values = torch.ones(2, 8, 9, 5)
    observed = torch.ones(2, 8, 9, dtype=torch.bool)
    observed[:, :, -1] = False
    projected = projector(values, observed)
    assert projected.shape == (2, 8, 9)
    assert torch.count_nonzero(projected[:, :, -1]).item() == 0
    assert sum(parameter.numel() for parameter in projector.parameters()) == 6

    larger_asset_panel = projector(
        torch.ones(2, 8, 17, 5),
        torch.ones(2, 8, 17, dtype=torch.bool),
    )
    assert larger_asset_panel.shape == (2, 8, 17)
    assert sum(parameter.numel() for parameter in projector.parameters()) == 6


def test_projected_module_calls_only_native_four_argument_backbone() -> None:
    backbone = FourArgumentBackbone(assets=9, horizon=2)
    model = ProjectedForecastModule(SharedPerAssetProjector(5), backbone)
    output = model(
        torch.ones(2, 8, 9, 5),
        None,
        None,
        None,
        torch.ones(2, 8, 9, dtype=torch.bool),
    )
    assert output.shape == (2, 2, 9)
    assert backbone.last_input_shape == (2, 8, 9)


@pytest.mark.parametrize("output_tuple", (False, True))
def test_out_of_place_wrapper_is_mathematically_equivalent_and_restores_flag(
    output_tuple,
) -> None:
    backbone = InPlaceNativeNormalizationBackbone(assets=3, horizon=2, output_tuple=output_tuple)
    values = torch.randn(2, 8, 3)
    expected = backbone(values.clone(), None, None, None)
    wrapper = OutOfPlaceNativeNormalization(backbone)
    actual = wrapper(values.clone(), None, None, None)
    expected_prediction = expected[0] if isinstance(expected, tuple) else expected
    actual_prediction = actual[0] if isinstance(actual, tuple) else actual
    assert torch.allclose(actual_prediction, expected_prediction, atol=1e-6, rtol=1e-6)
    assert isinstance(actual, tuple) is output_tuple
    if output_tuple:
        assert actual[1:] == expected[1:]
    assert backbone.use_norm == 1


def test_projector_gradient_survives_native_normalization_regression() -> None:
    projector = SharedPerAssetProjector(2)
    backbone = InPlaceNativeNormalizationBackbone(assets=3, horizon=2)
    model = ProjectedForecastModule(projector, backbone)
    output = model(
        torch.randn(2, 8, 3, 2),
        None,
        None,
        None,
        torch.ones(2, 8, 3, dtype=torch.bool),
    )
    output.square().mean().backward()
    assert projector.linear.weight.grad is not None
    assert bool(torch.isfinite(projector.linear.weight.grad).all().item())
    assert backbone.use_norm == 1


def test_masked_mse_rejects_missing_labels_and_uses_only_valid_slice() -> None:
    prediction = torch.tensor([[[1.0, 9.0]]])
    target = torch.tensor([[[3.0, -9.0]]])
    loss, selected = masked_mse(
        prediction,
        target,
        target_observed=torch.tensor([[[True, True]]]),
        label_available=torch.tensor([[[True, False]]]),
        criterion=torch.nn.MSELoss(),
    )
    assert selected == 1
    assert loss.item() == pytest.approx(4.0)
    with pytest.raises(AdapterContractError, match="no observed target"):
        masked_mse(
            prediction,
            target,
            target_observed=torch.tensor([[[True, True]]]),
            label_available=torch.tensor([[[False, False]]]),
            criterion=torch.nn.MSELoss(),
        )


def test_protocol_safe_fit_uses_validation_checkpoint_without_test_loader(tmp_path) -> None:
    backbone = FourArgumentBackbone(assets=3, horizon=2)
    model = ProjectedForecastModule(SharedPerAssetProjector(2), backbone)
    batch = DeepForecastBatch(
        x_enc=torch.randn(2, 4, 3, 2),
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        observed_mask=torch.ones(2, 4, 3, dtype=torch.bool),
        target=torch.zeros(2, 2, 3),
        target_observed=torch.ones(2, 2, 3, dtype=torch.bool),
        label_available=torch.ones(2, 2, 3, dtype=torch.bool),
    )
    scheduler_epochs = []
    summary = fit_protocol_safe(
        model,
        (batch,),
        (batch,),
        learning_rate=1e-3,
        maximum_epochs=2,
        patience=2,
        adjust_learning_rate=lambda _optimizer, epoch: scheduler_epochs.append(epoch),
        checkpoint_path=Path(tmp_path, "checkpoint.pth"),
    )
    assert summary.best_epoch in {1, 2}
    assert summary.selected_target_count == 12
    assert scheduler_epochs == [1, 2]
    assert Path(tmp_path, "checkpoint.pth").is_file()
