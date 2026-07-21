"""V1 information-ablation scoring on the 2025 validation fold.

Reuses V0 scoring primitives (labels, eligible keys, evaluation) but discovers
V1 prediction frames (``v1-a{1,2,3}-...``) and groups scorecards by ablation
gate in addition to model and seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar

from a_share_research.contracts import (
    CanonicalModel,
    ContractError,
    PredictionFrame,
    canonical_hash,
)
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.evaluation.metrics import (
    evaluate_predictions,
    freeze_common_support,
)
from a_share_research.evaluation.schema import (
    EvaluationFrequency,
    OutcomeMode,
    PredictionScorecard,
    SupportMode,
)
from a_share_research.evaluation.statistics import aggregate_seed_estimates
from a_share_research.evaluation.v0_scoring import (
    V0CellFailure,
    build_eligible_keys,
    build_validation_labels,
    load_trading_calendar,
)

_V1_RUN_ID = re.compile(
    r"^v1-a([123])-(csi300|star50|tech32|tech90)-([a-z0-9]+)-seed-(\d{8})$"
)
_FREQUENCY = EvaluationFrequency.WEEKLY
_SUPPORTS = (SupportMode.COMMON, SupportMode.NATIVE)
_OUTCOMES = (OutcomeMode.ABSOLUTE, OutcomeMode.BENCHMARK_RELATIVE)


@dataclass(frozen=True)
class V1RunKey:
    """Parsed canonical V1 run identity."""

    gate: str
    universe: str
    model: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"v1-a{self.gate}-{self.universe}-{self.model}-seed-{self.seed:08d}"


def parse_v1_run_id(run_id: str) -> V1RunKey | None:
    """Return the parsed V1 run identity, or ``None`` for non-V1 run ids."""
    match = _V1_RUN_ID.match(run_id)
    if match is None:
        return None
    return V1RunKey(
        gate=match.group(1),
        universe=match.group(2),
        model=match.group(3),
        seed=int(match.group(4)),
    )


@dataclass(frozen=True)
class V1ModelAggregate(CanonicalModel):
    """Seed distribution of the primary ranking metric for one gate+config."""

    SCHEMA_NAME: ClassVar[str] = "v1_model_aggregate"

    gate: str
    model: str
    support: SupportMode
    outcome: OutcomeMode
    seed_count: int
    rank_ic_mean: float | None
    rank_ic_std: float | None

    def validate(self) -> None:
        if self.seed_count < 1:
            raise ContractError("seed_count must be positive")


@dataclass(frozen=True)
class V1UniverseScore(CanonicalModel):
    """All scorecards and seed aggregates for one universe on the 2025 fold."""

    SCHEMA_NAME: ClassVar[str] = "v1_universe_score"

    universe: str
    trading_calendar_hash: str
    eligible_keys_count: int
    common_support_count: int
    gate_count: int
    model_count: int
    scorecards: tuple[PredictionScorecard, ...]
    failures: tuple[V0CellFailure, ...]
    aggregates: tuple[V1ModelAggregate, ...]

    def validate(self) -> None:
        if not self.universe or len(self.trading_calendar_hash) != 64:
            raise ContractError("v1 universe score requires universe and calendar hash")
        if self.gate_count < 1:
            raise ContractError("v1 universe score requires at least one gate")


def discover_all_v1_predictions(
    runs_root: Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, dict[str, dict[str, dict[int, PredictionFrame]]]]:
    """Load V1 prediction frames grouped by universe -> gate -> model -> seed."""
    from a_share_research.evaluation.v0_scoring import _load_prediction_frame

    grouped: dict[str, dict[str, dict[str, dict[int, PredictionFrame]]]] = {}
    for path in sorted(runs_root.rglob("predictions.json")):
        frame = _load_prediction_frame(path, verify_hashes=verify_hashes)
        key = parse_v1_run_id(frame.run_id)
        if key is None:
            continue
        (
            grouped.setdefault(key.universe, {})
            .setdefault(key.gate, {})
            .setdefault(key.model, {})
        )[key.seed] = frame
    return grouped


def _score_cell(
    *,
    frame: PredictionFrame,
    labels: tuple,
    eligible_keys: set[tuple[date, str]],
    common_support: set[tuple[date, str]],
    support: SupportMode,
    outcome: OutcomeMode,
) -> PredictionScorecard | V0CellFailure:
    run_key = parse_v1_run_id(frame.run_id)
    if run_key is None:
        raise ContractError(f"non-V1 prediction frame reached scoring: {frame.run_id}")
    try:
        return evaluate_predictions(
            predictions=frame,
            labels=labels,
            frequency=_FREQUENCY,
            support=support,
            outcome=outcome,
            eligible_keys=eligible_keys,
            common_support_keys=common_support if support is SupportMode.COMMON else None,
        )
    except ContractError as error:
        return V0CellFailure(
            run_id=frame.run_id,
            model=run_key.model,
            seed=run_key.seed,
            support=support,
            outcome=outcome,
            reason_code="EMPTY_EVALUATION_SUPPORT",
            detail=str(error),
        )


def score_v1_universe(
    *,
    canonical_root: Path,
    universe: str,
    staged_calendar: Path,
    frames_by_gate: dict[str, dict[str, dict[int, PredictionFrame]]] | None = None,
    runs_root: Path | None = None,
    verify_hashes: bool = False,
) -> V1UniverseScore:
    """Score all V1 (gate, model, seed) cells for one universe on 2025 fold.

    Pass ``frames_by_gate`` to reuse a once-loaded set (see
    ``discover_all_v1_predictions``); otherwise ``runs_root`` is scanned.
    """
    if frames_by_gate is None:
        if runs_root is None:
            raise ContractError("score_v1_universe requires runs_root or frames_by_gate")
        all_frames = discover_all_v1_predictions(runs_root, verify_hashes=verify_hashes)
        frames_by_gate = all_frames.get(universe, {})

    if not frames_by_gate:
        raise ContractError(f"no V1 predictions found for universe {universe!r}")

    loader = CanonicalDatasetLoader(canonical_root, universe)
    trading_calendar = load_trading_calendar(staged_calendar)
    calendar_hash = canonical_hash(trading_calendar)
    labels = build_validation_labels(loader, trading_calendar, calendar_hash=calendar_hash)
    eligible_keys = build_eligible_keys(loader)

    family: dict[str, PredictionFrame] = {}
    for gate, model_frames in frames_by_gate.items():
        for model, seed_frames in model_frames.items():
            for seed, frame in seed_frames.items():
                family[f"{gate}-{model}@{seed}"] = frame
    common_support = freeze_common_support(family, eligible_keys)

    scorecards: list[PredictionScorecard] = []
    failures: list[V0CellFailure] = []
    rank_ic_by_config: dict[
        tuple[str, str, SupportMode, OutcomeMode], dict[int, float]
    ] = {}
    for gate in sorted(frames_by_gate):
        for model in sorted(frames_by_gate[gate]):
            for seed in sorted(frames_by_gate[gate][model]):
                frame = frames_by_gate[gate][model][seed]
                for support in _SUPPORTS:
                    for outcome in _OUTCOMES:
                        result = _score_cell(
                            frame=frame,
                            labels=labels,
                            eligible_keys=eligible_keys,
                            common_support=common_support,
                            support=support,
                            outcome=outcome,
                        )
                        if isinstance(result, PredictionScorecard):
                            scorecards.append(result)
                            if result.rank_ic is not None:
                                rank_ic_by_config.setdefault(
                                    (gate, model, support, outcome), {}
                                )[seed] = result.rank_ic
                        else:
                            failures.append(result)

    aggregates: list[V1ModelAggregate] = []
    for gate in sorted(frames_by_gate):
        for model in sorted(frames_by_gate[gate]):
            for support in _SUPPORTS:
                for outcome in _OUTCOMES:
                    seed_values = rank_ic_by_config.get(
                        (gate, model, support, outcome), {}
                    )
                    if not seed_values:
                        continue
                    if len(seed_values) >= 2:
                        mean, std = aggregate_seed_estimates(seed_values)
                    else:
                        mean = next(iter(seed_values.values()))
                        std = None
                    aggregates.append(
                        V1ModelAggregate(
                            gate=gate,
                            model=model,
                            support=support,
                            outcome=outcome,
                            seed_count=len(seed_values),
                            rank_ic_mean=mean,
                            rank_ic_std=std,
                        )
                    )

    model_count = len({m for gf in frames_by_gate.values() for m in gf})
    return V1UniverseScore(
        universe=universe,
        trading_calendar_hash=calendar_hash,
        eligible_keys_count=len(eligible_keys),
        common_support_count=len(common_support),
        gate_count=len(frames_by_gate),
        model_count=model_count,
        scorecards=tuple(scorecards),
        failures=tuple(failures),
        aggregates=tuple(aggregates),
    )


__all__ = [
    "V1ModelAggregate",
    "V1RunKey",
    "V1UniverseScore",
    "discover_all_v1_predictions",
    "parse_v1_run_id",
    "score_v1_universe",
]
