"""Optional PyTorch runtime for the external deep-model adapter boundary.

This module is imported only inside a model-specific server environment.  It
does not vendor or modify author layers.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from a_share_research.adapters.common.types import AdapterContractError


class SharedPerAssetProjector(nn.Module):
    """Apply one shared `C -> 1` linear map to every date and stock."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if type(input_channels) is not int or input_channels <= 0:
            raise AdapterContractError("projector input_channels must be a positive integer")
        self.input_channels = input_channels
        self.linear = nn.Linear(input_channels, 1)

    def forward(self, values: Tensor, observed_mask: Tensor) -> Tensor:
        if values.ndim != 4:
            raise AdapterContractError("projector values must have shape [B,L,A,C]")
        if values.shape[-1] != self.input_channels:
            raise AdapterContractError("projector channel width differs from its frozen schema")
        if observed_mask.ndim != 3 or observed_mask.shape != values.shape[:-1]:
            raise AdapterContractError("observed mask must have shape [B,L,A]")
        if observed_mask.dtype is not torch.bool:
            raise AdapterContractError("observed mask must be boolean")
        masked_values = values.masked_fill(~observed_mask.unsqueeze(-1), 0.0)
        projected = self.linear(masked_values).squeeze(-1)
        return projected.masked_fill(~observed_mask, 0.0)


class OutOfPlaceNativeNormalization(nn.Module):
    """Reproduce author normalization without its autograd-breaking in-place divide."""

    def __init__(self, native_backbone: nn.Module) -> None:
        super().__init__()
        if not hasattr(native_backbone, "use_norm"):
            raise AdapterContractError("author backbone does not expose use_norm")
        self.native_backbone = native_backbone

    @staticmethod
    def _denormalize(output: Any, *, means: Tensor, stdev: Tensor) -> Any:
        metadata: tuple[Any, ...] = ()
        prediction = output
        if isinstance(output, tuple):
            if not output:
                raise AdapterContractError("author model returned an empty tuple")
            prediction, metadata = output[0], output[1:]
        if not isinstance(prediction, Tensor) or prediction.ndim != 3:
            raise AdapterContractError("normalized author output must have shape [B,H,A]")
        if prediction.shape[0] != means.shape[0] or prediction.shape[2] != means.shape[2]:
            raise AdapterContractError("author output does not match normalization statistics")
        horizon = prediction.shape[1]
        scale = stdev[:, 0, :].unsqueeze(1).repeat(1, horizon, 1)
        location = means[:, 0, :].unsqueeze(1).repeat(1, horizon, 1)
        denormalized = prediction * scale + location
        return (denormalized, *metadata) if isinstance(output, tuple) else denormalized

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None,
        x_dec: Tensor | None,
        x_mark_dec: Tensor | None,
    ) -> Any:
        original_use_norm = getattr(self.native_backbone, "use_norm")
        if not bool(original_use_norm):
            return self.native_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / stdev
        setattr(self.native_backbone, "use_norm", False)
        try:
            output = self.native_backbone(normalized, x_mark_enc, x_dec, x_mark_dec)
        finally:
            setattr(self.native_backbone, "use_norm", original_use_norm)
        return self._denormalize(output, means=means, stdev=stdev)


class ProjectedForecastModule(nn.Module):
    """Compose the trainable projector with an unmodified author backbone."""

    def __init__(self, projector: SharedPerAssetProjector, backbone: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.backbone = OutOfPlaceNativeNormalization(backbone)

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None,
        x_dec: Tensor | None,
        x_mark_dec: Tensor | None,
        observed_mask: Tensor,
    ) -> Any:
        projected = self.projector(x_enc, observed_mask)
        return self.backbone(projected, x_mark_enc, x_dec, x_mark_dec)



class S4MForecastModule(nn.Module):
    """S4M decay-imputation wrapper; bypasses RevIN normalization.

    S4M's author forward signature is ``forward(seq_x, seq_x_mask, max_idx,
    min_idx, max_value, min_value)`` and requires a one-shot ``warmup`` to
    initialise the memory bank via k-means clustering.  This module adapts the
    shared per-asset projector output to that interface, computes the
    positional auxiliaries exactly as the upstream data loader does, and
    returns only the last ``pred_len`` timesteps as the forecast horizon.

    Deviations from the upstream training loop (labelled for provenance):

    * The first training batch is **not** skipped — warmup runs inline on the
      first forward call, then a normal forward + backward follows.  In the
      upstream loop the first batch is consumed by warmup only.
    * ``OutOfPlaceNativeNormalization`` is intentionally absent: S4M uses
      decay-based imputation rather than non-stationary RevIN normalization.
    """

    def __init__(
        self,
        projector: SharedPerAssetProjector,
        backbone: nn.Module,
        *,
        pred_len: int,
    ) -> None:
        super().__init__()
        if type(pred_len) is not int or pred_len <= 0:
            raise AdapterContractError("S4M pred_len must be a positive integer")
        self.projector = projector
        self.backbone = backbone
        self.pred_len = pred_len
        self._warmed_up = False

    @staticmethod
    def _decay_auxiliaries(seq_x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Replicate the upstream max_idx/min_idx/max_value/min_value logic."""
        batch_size, length, _ = seq_x.shape
        argmax_pos = seq_x.argmax(dim=1)
        argmin_pos = seq_x.argmin(dim=1)
        positions = torch.arange(length, device=seq_x.device).view(1, length, 1)
        max_idx = (argmax_pos.unsqueeze(1) - positions).abs().float()
        min_idx = (argmin_pos.unsqueeze(1) - positions).abs().float()
        max_value = seq_x.max(dim=1, keepdim=True).values.expand_as(seq_x)
        min_value = seq_x.min(dim=1, keepdim=True).values.expand_as(seq_x)
        return max_idx, min_idx, max_value, min_value

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None,
        x_dec: Tensor | None,
        x_mark_dec: Tensor | None,
        observed_mask: Tensor,
    ) -> Tensor:
        seq_x = self.projector(x_enc, observed_mask)
        seq_x_mask = observed_mask.float()
        max_idx, min_idx, max_value, min_value = self._decay_auxiliaries(seq_x)
        if not self._warmed_up:
            with torch.no_grad():
                self.backbone.warmup(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )
            self._warmed_up = True
        output = self.backbone(
            seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
        )
        return output[:, -self.pred_len:, :]


@dataclass(frozen=True)
class DeepForecastBatch:
    """One protocol-safe batch; no test-loader object is accepted by fit()."""

    x_enc: Tensor
    x_mark_enc: Tensor | None
    x_dec: Tensor | None
    x_mark_dec: Tensor | None
    observed_mask: Tensor
    target: Tensor
    target_observed: Tensor
    label_available: Tensor


@dataclass(frozen=True)
class FitSummary:
    best_epoch: int
    best_validation_mse: float
    epochs_completed: int
    selected_target_count: int


LearningRateAdjuster = Callable[[torch.optim.Optimizer, int], None]
OptimizerFactory = Callable[[Any, float], torch.optim.Optimizer]


def _forecast_tensor(output: Any) -> Tensor:
    if isinstance(output, tuple):
        if not output:
            raise AdapterContractError("author model returned an empty tuple")
        output = output[0]
    if not isinstance(output, Tensor):
        raise AdapterContractError("author model output must be a torch Tensor")
    return output


def masked_mse(
    prediction: Tensor,
    target: Tensor,
    *,
    target_observed: Tensor,
    label_available: Tensor,
    criterion: nn.Module,
) -> tuple[Tensor, int]:
    """Apply the author's MSE to the valid slice without changing its formula."""
    if prediction.shape != target.shape:
        raise AdapterContractError("prediction and target shapes differ")
    if target_observed.shape != target.shape or label_available.shape != target.shape:
        raise AdapterContractError("target masks must match prediction shape")
    if target_observed.dtype is not torch.bool or label_available.dtype is not torch.bool:
        raise AdapterContractError("target masks must be boolean")
    valid = target_observed & label_available
    selected = int(valid.sum().item())
    if selected == 0:
        raise AdapterContractError("batch has no observed target with an available label")
    loss = criterion(prediction[valid], target[valid])
    if not bool(torch.isfinite(loss).item()):
        raise AdapterContractError("masked MSE is non-finite")
    return loss, selected


def _batch_loss(
    model: nn.Module,
    batch: DeepForecastBatch,
    criterion: nn.Module,
) -> tuple[Tensor, int]:
    output = model(
        batch.x_enc,
        batch.x_mark_enc,
        batch.x_dec,
        batch.x_mark_dec,
        batch.observed_mask,
    )
    return masked_mse(
        _forecast_tensor(output),
        batch.target,
        target_observed=batch.target_observed,
        label_available=batch.label_available,
        criterion=criterion,
    )


def _validation_mse(
    model: nn.Module,
    batches: Iterable[DeepForecastBatch],
    criterion: nn.Module,
) -> tuple[float, int]:
    model.eval()
    weighted_loss = 0.0
    selected_total = 0
    with torch.no_grad():
        for batch in batches:
            loss, selected = _batch_loss(model, batch, criterion)
            weighted_loss += float(loss.item()) * selected
            selected_total += selected
    if selected_total == 0:
        raise AdapterContractError("validation loader selected no targets")
    result = weighted_loss / selected_total
    if not math.isfinite(result):
        raise AdapterContractError("validation MSE is non-finite")
    return result, selected_total


def _atomic_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    validation_mse: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_mse": validation_mse,
        },
        temporary,
    )
    os.replace(temporary, path)


def fit_protocol_safe(
    model: nn.Module,
    train_batches: Iterable[DeepForecastBatch],
    validation_batches: Iterable[DeepForecastBatch],
    *,
    learning_rate: float,
    maximum_epochs: int,
    patience: int,
    adjust_learning_rate: LearningRateAdjuster,
    checkpoint_path: Path,
    optimizer_factory: OptimizerFactory | None = None,
) -> FitSummary:
    """Train with author MSE/Adam/schedule while never constructing a test loader."""
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise AdapterContractError("learning_rate must be finite and positive")
    if type(maximum_epochs) is not int or maximum_epochs <= 0:
        raise AdapterContractError("maximum_epochs must be a positive integer")
    if type(patience) is not int or patience <= 0:
        raise AdapterContractError("patience must be a positive integer")
    if not checkpoint_path.is_absolute():
        raise AdapterContractError("checkpoint_path must be absolute")
    train_materialized = tuple(train_batches)
    validation_materialized = tuple(validation_batches)
    if not train_materialized or not validation_materialized:
        raise AdapterContractError("train and validation loaders must be non-empty")

    criterion = nn.MSELoss()
    if optimizer_factory is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optimizer_factory(model.parameters(), learning_rate)
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_validation = math.inf
    stale_epochs = 0
    selected_at_best = 0
    epochs_completed = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for batch in train_materialized:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch, criterion)
            loss.backward()
            optimizer.step()
        validation_mse, selected = _validation_mse(
            model, validation_materialized, criterion
        )
        epochs_completed = epoch
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            selected_at_best = selected
            best_state = copy.deepcopy(model.state_dict())
            _atomic_checkpoint(
                checkpoint_path,
                model=model,
                epoch=epoch,
                validation_mse=validation_mse,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        adjust_learning_rate(optimizer, epoch)
        if stale_epochs >= patience:
            break

    if best_state is None or best_epoch < 1:
        raise AdapterContractError("training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    return FitSummary(
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
        epochs_completed=epochs_completed,
        selected_target_count=selected_at_best,
    )
