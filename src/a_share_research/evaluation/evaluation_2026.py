"""2026 LEGACY_VIEWED evaluation scoring.

Scores predictions on the 2026-01-01..2026-07-17 legacy-viewed fold.  This is
the final evaluation partition (not the 2025 validation/early-stop fold).  The
FUTURE_UNSEEN partition (2026-07-18+) requires ``ProtocolSpec.open_future()``
and is out of scope here.

Reuses V0 scoring primitives (labels, eligible keys, evaluation) but filters to
the LEGACY_VIEWED window and accepts ``eval-2026-a{0,1,2,3}-...`` run ids.
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
    load_trading_calendar,
)

_EVAL_START = date(2026, 1, 1)
_EVAL_END = date(2026, 7, 17)
_HORIZON = 5
_FREQUENCY = EvaluationFrequency.WEEKLY
_SUPPORTS = (SupportMode.COMMON, SupportMode.NATIVE)
_OUTCOMES = (OutcomeMode.ABSOLUTE, OutcomeMode.BENCHMARK_RELATIVE)

_RUN_ID_PATTERN = re.compile(
    r"^eval-2026-a([0123])-(csi300|star50|tech32|tech90)-([a-z0-9]+)-seed-(\d{8})$"
)


@dataclass(frozen=True)
class Eval2026RunKey:
    """Parsed canonical 2026-evaluation run identity."""

    gate: str
    universe: str
    model: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"eval-2026-a{self.gate}-{self.universe}-{self.model}-seed-{self.seed:08d}"


def parse_eval_2026_run_id(run_id: str) -> Eval2026RunKey | None:
    """Return the parsed eval-2026 run identity, or ``None`` for non-matching ids."""
    match = _RUN_ID_PATTERN.match(run_id)
    if match is None:
        return None
    return Eval2026RunKey(
        gate=match.group(1),
        universe=match.group(2),
        model=match.group(3),
        seed=int(match.group(4)),
    )


@dataclass(frozen=True)
class Eval2026ModelAggregate(CanonicalModel):
    """Seed distribution of the primary ranking metric for one config."""

    SCHEMA_NAME: ClassVar[str] = "eval_2026_model_aggregate"

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
class Eval2026UniverseScore(CanonicalModel):
    """All scorecards and seed aggregates for one universe on the 2026 fold."""

    SCHEMA_NAME: ClassVar[str] = "eval_2026_universe_score"

    universe: str
    partition: str
    trading_calendar_hash: str
    eligible_keys_count: int
    common_support_count: int
    gate_count: int
    model_count: int
    scorecards: tuple[PredictionScorecard, ...]
    failures: tuple[V0CellFailure, ...]
    aggregates: tuple[Eval2026ModelAggregate, ...]

    def validate(self) -> None:
        if not self.universe or len(self.trading_calendar_hash) != 64:
            raise ContractError("eval 2026 universe score requires universe and calendar hash")
        if self.model_count < 1:
            raise ContractError("eval 2026 universe score requires at least one model")


def build_evaluation_labels(
    loader: CanonicalDatasetLoader,
    trading_calendar: tuple[date, ...],
    *,
    calendar_hash: str,
) -> tuple:
    """Closed weekly labels whose signal and exit both stay inside 2026 LEGACY_VIEWED."""
    from a_share_research.evaluation.v0_scoring import _ValidationLabel

    labels: list = []
    hash_checked = False
    for compact in loader.iter_labels():
        if not hash_checked:
            if compact.trading_calendar_hash != calendar_hash:
                raise ContractError("D0 label calendar hash disagrees with staged calendar")
            hash_checked = True
        if compact.horizon != _HORIZON:
            continue
        if not (_EVAL_START <= compact.signal_date <= _EVAL_END):
            continue
        if compact.exit_date > _EVAL_END:
            continue
        labels.append(
            _ValidationLabel(
                signal_date=compact.signal_date,
                ts_code=compact.ts_code,
                horizon=compact.horizon,
                entry_date=compact.entry_date,
                exit_date=compact.exit_date,
                open_to_open_return=compact.open_to_open_return,
                benchmark_return=compact.benchmark_return,
            )
        )
    if not labels:
        raise ContractError("D0 contains no closed weekly 2026 legacy-viewed labels")
    return tuple(labels)


def build_evaluation_eligible_keys(loader: CanonicalDatasetLoader) -> set[tuple[date, str]]:
    """Legacy-viewed-fold rows that are member, observed and label-available."""
    eligible: set[tuple[date, str]] = set()
    for bundle in loader.iter_masks():
        if not (_EVAL_START <= bundle.signal_date <= _EVAL_END):
            continue
        for code, flag in zip(bundle.asset_ids, bundle.evaluation, strict=True):
            if flag:
                eligible.add((bundle.signal_date, code))
    if not eligible:
        raise ContractError("D0 masks contain no 2026 legacy-viewed evaluation rows")
    return eligible


def discover_eval_2026_predictions(
    runs_root: Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, dict[str, dict[str, dict[int, PredictionFrame]]]]:
    """Load eval-2026 prediction frames grouped by universe -> gate -> model -> seed."""
    from a_share_research.evaluation.v0_scoring import _load_prediction_frame

    grouped: dict[str, dict[str, dict[str, dict[int, PredictionFrame]]]] = {}
    for path in sorted(runs_root.rglob("predictions.json")):
        frame = _load_prediction_frame(path, verify_hashes=verify_hashes)
        key = parse_eval_2026_run_id(frame.run_id)
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
    run_key = parse_eval_2026_run_id(frame.run_id)
    if run_key is None:
        raise ContractError(f"non-eval-2026 prediction frame reached scoring: {frame.run_id}")
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


def score_eval_2026_universe(
    *,
    canonical_root: Path,
    universe: str,
    staged_calendar: Path,
    frames_by_gate: dict[str, dict[str, dict[int, PredictionFrame]]] | None = None,
    runs_root: Path | None = None,
    verify_hashes: bool = False,
) -> Eval2026UniverseScore:
    """Score all eval-2026 (gate, model, seed) cells for one universe on 2026 fold."""
    if frames_by_gate is None:
        if runs_root is None:
            raise ContractError("score_eval_2026_universe requires runs_root or frames_by_gate")
        all_frames = discover_eval_2026_predictions(runs_root, verify_hashes=verify_hashes)
        frames_by_gate = all_frames.get(universe, {})

    if not frames_by_gate:
        raise ContractError(f"no eval-2026 predictions found for universe {universe!r}")

    loader = CanonicalDatasetLoader(canonical_root, universe)
    trading_calendar = load_trading_calendar(staged_calendar)
    calendar_hash = canonical_hash(trading_calendar)
    labels = build_evaluation_labels(loader, trading_calendar, calendar_hash=calendar_hash)
    eligible_keys = build_evaluation_eligible_keys(loader)

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

    aggregates: list[Eval2026ModelAggregate] = []
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
                        Eval2026ModelAggregate(
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
    return Eval2026UniverseScore(
        universe=universe,
        partition="LEGACY_VIEWED_2026_01_01_2026_07_17",
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
    "Eval2026ModelAggregate",
    "Eval2026RunKey",
    "Eval2026UniverseScore",
    "build_evaluation_eligible_keys",
    "build_evaluation_labels",
    "discover_eval_2026_predictions",
    "parse_eval_2026_run_id",
    "score_eval_2026_universe",
]
