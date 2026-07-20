"""Framework-free wrapper smoke; real GPU smoke runs only on the approved server."""

from datetime import date

import pytest

from a_share_research.adapters.common import (
    AdapterContractError,
    FeaturePackingSchema,
    InformationGate,
    PanelWindow,
    ProjectedForecastBoundary,
    UpstreamBinding,
    build_causal_asset_master,
    extract_target_scores,
    pack_feature_window,
)
from a_share_research.adapters.fact import FactAdapter
from a_share_research.adapters.itransformer import ITransformerAdapter
from a_share_research.adapters.s4m import S4MAdapter
from a_share_research.adapters.timepro import TimeProAdapter
from a_share_research.contracts import AssetRegistry, MaskBundle, UniverseMembership

HASH = "a" * 64


class FakeParameter:
    requires_grad = True

    def __init__(self, size: int) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


class FakeAuthorModel:
    def __init__(self) -> None:
        self.call = None

    def parameters(self):
        return (FakeParameter(7), FakeParameter(5))

    def __call__(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.call = (x_enc, x_mark_enc, x_dec, x_mark_dec)
        width = len(x_enc[0][0])
        return [[[float(index) for index in range(width)]]]


class FakeSharedProjector:
    def __init__(self) -> None:
        self.call = None

    def parameters(self):
        return (FakeParameter(3),)

    def __call__(self, x_enc, observed_mask):
        self.call = (x_enc, observed_mask)
        return tuple(
            tuple(
                tuple(
                    asset_values[0] if observed else 0.0
                    for asset_values, observed in zip(
                        date_values, date_mask, strict=True
                    )
                )
                for date_values, date_mask in zip(batch_values, batch_mask, strict=True)
            )
            for batch_values, batch_mask in zip(x_enc, observed_mask, strict=True)
        )


def make_model():
    projector = FakeSharedProjector()
    backbone = FakeAuthorModel()
    return ProjectedForecastBoundary(projector, backbone), projector, backbone


def make_packed(gate: InformationGate):
    row = UniverseMembership(
        asof_date=date(2019, 1, 2),
        ts_code="000001.SZ",
        universe="CSI300",
        effective_from=date(2019, 1, 2),
        effective_to=None,
        source="synthetic",
    )
    master = build_causal_asset_master((row,), known_through=date(2019, 12, 31))
    mask = MaskBundle(
        signal_date=date(2019, 12, 30),
        asset_ids=master.asset_ids,
        asset_registry_hash=AssetRegistry(master.asset_ids).stable_hash(),
        member=(True,),
        observed=(True,),
        feature_missing={"return_1d": (False,), "pe_ttm": (False,), "state": (False,)},
        label_available=(True,),
        buyable=(True,),
        sellable=(True,),
        loss=(True,),
        evaluation=(True,),
    )
    panel = PanelWindow(
        dates=(mask.signal_date,),
        asset_master=master,
        values={"return_1d": ((0.1,),), "pe_ttm": ((10.0,),), "state": ((0.2,),)},
        masks=(mask,),
    )
    schema = FeaturePackingSchema(
        core=("return_1d",), factors=("pe_ttm",), state=("state",)
    )
    return pack_feature_window(panel, schema=schema, gate=gate)


@pytest.mark.parametrize(
    ("adapter_type", "model_name", "commit"),
    (
        (
            ITransformerAdapter,
            "itransformer",
            "c2426e68ca13f74aaec08045c5c724d8ad328124",
        ),
        (FactAdapter, "fact", "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4"),
        (
            S4MAdapter,
            "s4m",
            "a718823addd3606e763dfc261174e0135b2535f4",
        ),
    ),
)
def test_wrapper_calls_injected_author_forward_and_reverses_target(
    adapter_type, model_name: str, commit: str
) -> None:
    model, projector, backbone = make_model()
    adapter = adapter_type(
        binding=UpstreamBinding(model_name, commit, HASH, HASH),
        model=model,
    )
    packed = make_packed(InformationGate.A3)
    output = adapter.forward(
        packed,
        tensor_factory=lambda values: values,
        x_mark_enc="enc-mark",
        x_dec="decoder",
        x_mark_dec="dec-mark",
    )
    assert projector.call is not None
    assert len(projector.call[0][0][0][0]) == packed.input_channel_count
    assert len(backbone.call[0][0][0]) == packed.model_variate_count
    assert backbone.call[1:] == ("enc-mark", "decoder", "dec-mark")
    assert adapter.parameter_count() == 15
    assert len(adapter.architecture_hash(packed)) == 64
    assert extract_target_scores(output, packed=packed) == (0.0,)


def test_architecture_evidence_is_gate_independent() -> None:
    model, _, _ = make_model()
    adapter = ITransformerAdapter(
        binding=UpstreamBinding(
            "itransformer",
            "c2426e68ca13f74aaec08045c5c724d8ad328124",
            HASH,
            HASH,
        ),
        model=model,
    )
    hashes = {adapter.architecture_hash(make_packed(gate)) for gate in InformationGate}
    assert len(hashes) == 1


def test_fact_keeps_known_upstream_core_mix_and_timepro_ready() -> None:
    model, _, _ = make_model()
    adapter = FactAdapter(
        binding=UpstreamBinding(
            "fact",
            "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4",
            HASH,
            HASH,
        ),
        model=model,
    )
    adapter.require_supported_core_mix(0.5)
    with pytest.raises(AdapterContractError, match="core=0.5"):
        adapter.require_supported_core_mix(0.0)
    timepro_adapter = TimeProAdapter(
        binding=UpstreamBinding(
            "timepro",
            "70a20e5a257b30eb026ee4316293cf4feeb92a1f",
            HASH,
            HASH,
        ),
        model=model,
    )
    assert timepro_adapter.MODEL_NAME == "timepro"
    s4m_adapter = S4MAdapter(
        binding=UpstreamBinding(
            "s4m",
            "a718823addd3606e763dfc261174e0135b2535f4",
            HASH,
            HASH,
        ),
        model=model,
    )
    assert s4m_adapter.MODEL_NAME == "s4m"
    assert s4m_adapter.NATIVE_OPTIMIZER == "SGD"
