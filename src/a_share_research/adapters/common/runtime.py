"""Duck-typed external runtime bridge; imports no upstream or tensor framework."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from a_share_research.adapters.common.packing import PackedWindow
from a_share_research.adapters.common.types import AdapterContractError
from a_share_research.contracts import canonical_hash


class AuthorForecastModel(Protocol):
    """Pinned author forecasting interface; author source stays unchanged."""

    def __call__(
        self,
        x_enc: Any,
        x_mark_enc: Any,
        x_dec: Any,
        x_mark_dec: Any,
    ) -> Any: ...

    def parameters(self) -> Any: ...


class PerAssetProjector(Protocol):
    """Shared projection from feature channels to one value per stock token."""

    def __call__(self, x_enc: Any, observed_mask: Any) -> Any: ...

    def parameters(self) -> Any: ...


class ProjectedForecastModel(Protocol):
    """External projector plus pinned four-argument author-model boundary."""

    def __call__(
        self,
        x_enc: Any,
        x_mark_enc: Any,
        x_dec: Any,
        x_mark_dec: Any,
        observed_mask: Any,
    ) -> Any: ...

    def parameters(self) -> Any: ...


TensorFactory = Callable[[Any], Any]


@dataclass
class ProjectedForecastBoundary:
    """Framework-neutral composition that never edits the author backbone."""

    projector: PerAssetProjector
    backbone: AuthorForecastModel

    def __call__(
        self,
        x_enc: Any,
        x_mark_enc: Any,
        x_dec: Any,
        x_mark_dec: Any,
        observed_mask: Any,
    ) -> Any:
        projected = self.projector(x_enc, observed_mask)
        return self.backbone(projected, x_mark_enc, x_dec, x_mark_dec)

    def parameters(self) -> Any:
        """Expose adaptation and author parameters to one optimizer."""
        return itertools.chain(self.projector.parameters(), self.backbone.parameters())


@dataclass(frozen=True)
class UpstreamBinding:
    """Evidence binding supplied only after the server provenance gate passes."""

    model_name: str
    commit: str
    integrity_receipt_hash: str
    environment_lock_hash: str

    def __post_init__(self) -> None:
        if not self.model_name or re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            raise AdapterContractError("upstream binding requires model name and full commit")
        for name in ("integrity_receipt_hash", "environment_lock_hash"):
            value = getattr(self, name)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise AdapterContractError(f"{name} must be SHA-256")


class ExternalForecastAdapter:
    """Calls an injected projected boundary while preserving author forward semantics."""

    MODEL_NAME = ""
    EXPECTED_COMMIT = ""
    NATIVE_LOSS = "MSELoss"
    NATIVE_OPTIMIZER = "Adam"
    FIDELITY_LABEL = "official backbone + A-share input adaptation"

    def __init__(self, *, binding: UpstreamBinding, model: ProjectedForecastModel) -> None:
        if binding.model_name != self.MODEL_NAME or binding.commit != self.EXPECTED_COMMIT:
            raise AdapterContractError(f"{self.MODEL_NAME} binding does not match the frozen pin")
        self.binding = binding
        self.model = model

    def make_x_enc(self, packed: PackedWindow, *, tensor_factory: TensorFactory) -> Any:
        """Keep channels behind a shared projector; stocks remain upstream tokens."""
        batch = (packed.projector_values(),)
        return tensor_factory(batch)

    def make_observed_mask(self, packed: PackedWindow, *, tensor_factory: TensorFactory) -> Any:
        return tensor_factory((packed.observed_values(),))

    def forward(
        self,
        packed: PackedWindow,
        *,
        tensor_factory: TensorFactory,
        x_mark_enc: Any,
        x_dec: Any,
        x_mark_dec: Any,
    ) -> Any:
        x_enc = self.make_x_enc(packed, tensor_factory=tensor_factory)
        observed_mask = self.make_observed_mask(packed, tensor_factory=tensor_factory)
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, observed_mask)

    def parameter_count(self) -> int:
        """Count injected author parameters without importing torch."""
        parameters = getattr(self.model, "parameters", None)
        if parameters is None:
            raise AdapterContractError("author model does not expose parameters()")
        total = 0
        for parameter in parameters():
            if getattr(parameter, "requires_grad", True):
                numel = getattr(parameter, "numel", None)
                if numel is None:
                    raise AdapterContractError("author parameter does not expose numel()")
                total += int(numel())
        if total <= 0:
            raise AdapterContractError("author model has no trainable parameters")
        return total

    def architecture_hash(self, packed: PackedWindow) -> str:
        """Gate-independent evidence; caller compares this across A0-A3."""
        return canonical_hash(
            {
                "model": self.MODEL_NAME,
                "commit": self.binding.commit,
                "shape_hash": packed.config_shape_hash(),
                "upstream_variate_count": packed.model_variate_count,
                "input_channel_count": packed.input_channel_count,
                "parameter_count": self.parameter_count(),
                "native_loss": self.NATIVE_LOSS,
                "native_optimizer": self.NATIVE_OPTIMIZER,
                "fidelity_label": self.FIDELITY_LABEL,
            }
        )


def output_to_nested(output: Any) -> Any:
    """Detach an author output without making a framework a package dependency."""
    if isinstance(output, tuple):
        if not output:
            raise AdapterContractError("author model returned an empty tuple")
        output = output[0]
    detach = getattr(output, "detach", None)
    if detach is not None:
        output = detach()
    cpu = getattr(output, "cpu", None)
    if cpu is not None:
        output = cpu()
    tolist = getattr(output, "tolist", None)
    return tolist() if tolist is not None else output


def extract_target_scores(
    output: Any,
    *,
    packed: PackedWindow,
    horizon_index: int | None = None,
) -> tuple[float, ...]:
    """Invert targets; by default sum forecast daily log returns over the horizon."""
    nested = output_to_nested(output)
    try:
        batch = nested[0]
    except (IndexError, TypeError) as error:
        raise AdapterContractError("author output is not batch/horizon/variate shaped") from error
    if not batch:
        raise AdapterContractError("author output contains no forecast horizon")
    try:
        horizons = batch if horizon_index is None else (batch[horizon_index],)
    except (IndexError, TypeError) as error:
        raise AdapterContractError("requested forecast horizon is absent") from error
    if any(len(horizon) != packed.model_variate_count for horizon in horizons):
        raise AdapterContractError("author output variate width does not match causal stock master")
    scores = tuple(
        sum(float(horizon[slot]) for horizon in horizons)
        for slot in range(packed.model_variate_count)
    )
    if len(scores) != len(packed.asset_ids):
        raise AdapterContractError("target inverse mapping is incomplete")
    return scores
