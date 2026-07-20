"""V0 Step 4 diagnostic scoring on the frozen 2025 validation fold.

Loads the PIT D0 labels and evaluation masks, reconstructs full ``Label`` objects
from the compact canonical labels, freezes common support across a model family,
and scores every model's ``PredictionFrame`` on COMMON and NATIVE support under
ABSOLUTE and BENCHMARK_RELATIVE weekly outcomes.

This is the diagnostic prediction-metrics slice of V0 Step 4. External baselines
(equal-weight / momentum / cash / official index), the B0 always-full economic
ledger and the formal A0 freeze (Step 5) are separate, labelled slices.

Fold and horizon constants must stay identical to
``a_share_research.experiments.deep_runner`` so scoring uses the exact cells the
models were trained and exported on.
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
    Label,
    PredictionFrame,
    canonical_hash,
)
from a_share_research.data.labels import CompactLabel
from a_share_research.data.loaders import CanonicalDatasetLoader
from a_share_research.data.normalization import parse_provider_date
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

_TRAIN_START = date(2019, 1, 1)
_TRAIN_END = date(2024, 12, 31)
_VALIDATION_START = date(2025, 1, 1)
_VALIDATION_END = date(2025, 12, 31)
_HORIZON = 5
_FREQUENCY = EvaluationFrequency.WEEKLY
_SUPPORTS = (SupportMode.COMMON, SupportMode.NATIVE)
_OUTCOMES = (OutcomeMode.ABSOLUTE, OutcomeMode.BENCHMARK_RELATIVE)

_RUN_ID_PATTERN = re.compile(
    r"^v0-a0-(csi300|star50|tech32|tech100)-([a-z0-9]+)-seed-(\d{8})$"
)


@dataclass(frozen=True)
class V0RunKey:
    """Parsed canonical V0 run identity."""

    universe: str
    model: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"v0-a0-{self.universe}-{self.model}-seed-{self.seed:08d}"


@dataclass(frozen=True)
class V0CellFailure(CanonicalModel):
    """A scorecard that could not be produced, with a typed reason."""

    SCHEMA_NAME: ClassVar[str] = "v0_cell_failure"

    run_id: str
    model: str
    seed: int
    support: SupportMode
    outcome: OutcomeMode
    reason_code: str
    detail: str

    def validate(self) -> None:
        if not self.run_id or not self.reason_code:
            raise ContractError("v0 cell failure requires run_id and reason_code")


@dataclass(frozen=True)
class V0ModelAggregate(CanonicalModel):
    """Seed distribution of the primary ranking metric for one configuration."""

    SCHEMA_NAME: ClassVar[str] = "v0_model_aggregate"

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
class V0UniverseScore(CanonicalModel):
    """All scorecards and seed aggregates for one universe on the 2025 fold."""

    SCHEMA_NAME: ClassVar[str] = "v0_universe_score"

    universe: str
    trading_calendar_hash: str
    eligible_keys_count: int
    common_support_count: int
    model_count: int
    seed_counts: tuple[tuple[str, int], ...]
    scorecards: tuple[PredictionScorecard, ...]
    failures: tuple[V0CellFailure, ...]
    aggregates: tuple[V0ModelAggregate, ...]

    def validate(self) -> None:
        if not self.universe or len(self.trading_calendar_hash) != 64:
            raise ContractError("v0 universe score requires universe and calendar hash")
        if self.model_count < 1:
            raise ContractError("v0 universe score requires at least one model")


def parse_run_id(run_id: str) -> V0RunKey | None:
    """Return the parsed V0 run identity, or ``None`` for non-V0 run ids."""
    match = _RUN_ID_PATTERN.match(run_id)
    if match is None:
        return None
    return V0RunKey(universe=match.group(1), model=match.group(2), seed=int(match.group(3)))


def load_trading_calendar(staged_calendar: Path) -> tuple[date, ...]:
    """Load the immutable open-day calendar exactly as the canonical builder does.

    The canonical labels carry a hash over the full D0 trading calendar (open days
    from 2019-01-01 through the D0 cutoff). The staged calendar is already D0-scoped,
    so no upper bound is applied here; only the 2019 research-start lower bound.
    """
    if not staged_calendar.is_file():
        raise ContractError(f"staged trading calendar is absent: {staged_calendar}")
    rows: list[dict[str, object]] = []
    for line in staged_calendar.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(_json_object(line))
    dates = tuple(
        sorted(
            {
                parse_provider_date(row["cal_date"])
                for row in rows
                if int(row.get("is_open", 0)) == 1
                and parse_provider_date(row["cal_date"]) >= date(2019, 1, 1)
            }
        )
    )
    if not dates:
        raise ContractError("canonical trading calendar is empty for the scored window")
    return dates


def _json_object(line: str) -> dict[str, object]:
    import json

    value = json.loads(line)
    if not isinstance(value, dict):
        raise ContractError("staged calendar line is not a JSON object")
    return value


def expand_compact_label(compact: CompactLabel, trading_calendar: tuple[date, ...]) -> Label:
    """Rebuild a full ``Label`` from a compact canonical row and the calendar."""
    compact.verify_calendar(trading_calendar)
    return Label(
        signal_date=compact.signal_date,
        ts_code=compact.ts_code,
        horizon=compact.horizon,
        entry_date=compact.entry_date,
        exit_date=compact.exit_date,
        open_to_open_return=compact.open_to_open_return,
        benchmark_return=compact.benchmark_return,
        trading_calendar=trading_calendar,
        trading_calendar_hash=compact.trading_calendar_hash,
    )


@dataclass(frozen=True)
class _ValidationLabel:
    """Lightweight scoring label.

    Full ``Label`` construction re-hashes the 1828-day calendar per row, which is
    prohibitive for hundreds of thousands of labels. Calendar integrity is verified
    once against the staged calendar (see ``build_validation_labels``); the compact
    row's own structural invariants were already checked at parse time.
    """

    signal_date: date
    ts_code: str
    horizon: int
    entry_date: date
    exit_date: date
    open_to_open_return: float
    benchmark_return: float

    @property
    def relative_return(self) -> float:
        return self.open_to_open_return - self.benchmark_return

    def validate(self) -> None:
        if not self.signal_date < self.entry_date < self.exit_date:
            raise ContractError("validation label dates must be strictly increasing")


def build_validation_labels(
    loader: CanonicalDatasetLoader,
    trading_calendar: tuple[date, ...],
    *,
    calendar_hash: str,
) -> tuple[_ValidationLabel, ...]:
    """Closed weekly labels whose signal and exit both stay inside 2025.

    Single pass: verifies the D0 calendar hash on the first row, then keeps
    closed weekly validation rows as lightweight labels.
    """
    labels: list[_ValidationLabel] = []
    hash_checked = False
    for compact in loader.iter_labels():
        if not hash_checked:
            if compact.trading_calendar_hash != calendar_hash:
                raise ContractError("D0 label calendar hash disagrees with staged calendar")
            hash_checked = True
        if compact.horizon != _HORIZON:
            continue
        if not (_VALIDATION_START <= compact.signal_date <= _VALIDATION_END):
            continue
        if compact.exit_date > _VALIDATION_END:
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
        raise ContractError("D0 contains no closed weekly validation labels")
    return tuple(labels)


def build_eligible_keys(loader: CanonicalDatasetLoader) -> set[tuple[date, str]]:
    """Validation-fold rows that are member, observed and label-available."""
    eligible: set[tuple[date, str]] = set()
    for bundle in loader.iter_masks():
        if not (_VALIDATION_START <= bundle.signal_date <= _VALIDATION_END):
            continue
        for code, flag in zip(bundle.asset_ids, bundle.evaluation, strict=True):
            if flag:
                eligible.add((bundle.signal_date, code))
    if not eligible:
        raise ContractError("D0 masks contain no validation-fold evaluation rows")
    return eligible


def _load_prediction_frame(path: Path, *, verify_hashes: bool) -> PredictionFrame:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"predictions payload is not an object: {path}")
    frame = PredictionFrame.from_dict(payload)
    if verify_hashes:
        manifest_path = path.parent / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = manifest.get("prediction_hash")
            if expected_hash and expected_hash != frame.stable_hash():
                raise ContractError(
                    f"prediction hash mismatch for {frame.run_id}: manifest={expected_hash}"
                )
    return frame


def discover_v0_predictions(
    runs_root: Path,
    universe: str,
    *,
    verify_hashes: bool = False,
) -> dict[str, dict[int, PredictionFrame]]:
    """Locate V0 prediction frames for one universe.

    Hash verification against the sibling ``run_manifest.json`` ``prediction_hash``
    is opt-in (``verify_hashes=True``); the training pipeline already binds the
    manifest atomically, so scoring re-derivation is skipped by default for speed.
    """
    discovered: dict[str, dict[int, PredictionFrame]] = {}
    for path in sorted(runs_root.rglob("predictions.json")):
        frame = _load_prediction_frame(path, verify_hashes=verify_hashes)
        key = parse_run_id(frame.run_id)
        if key is None or key.universe != universe:
            continue
        discovered.setdefault(key.model, {})[key.seed] = frame
    return discovered


def discover_all_v0_predictions(
    runs_root: Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, dict[str, dict[int, PredictionFrame]]]:
    """Load every V0 prediction frame once, grouped by universe -> model -> seed."""
    grouped: dict[str, dict[str, dict[int, PredictionFrame]]] = {}
    for path in sorted(runs_root.rglob("predictions.json")):
        frame = _load_prediction_frame(path, verify_hashes=verify_hashes)
        key = parse_run_id(frame.run_id)
        if key is None:
            continue
        grouped.setdefault(key.universe, {}).setdefault(key.model, {})[key.seed] = frame
    return grouped


def _score_cell(
    *,
    frame: PredictionFrame,
    labels: tuple[Label, ...],
    eligible_keys: set[tuple[date, str]],
    common_support: set[tuple[date, str]],
    support: SupportMode,
    outcome: OutcomeMode,
) -> PredictionScorecard | V0CellFailure:
    run_key = parse_run_id(frame.run_id)
    if run_key is None:
        raise ContractError(f"non-V0 prediction frame reached scoring: {frame.run_id}")
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


def score_v0_universe(
    *,
    canonical_root: Path,
    universe: str,
    staged_calendar: Path,
    runs_root: Path | None = None,
    frames_by_model: dict[str, dict[int, PredictionFrame]] | None = None,
    verify_hashes: bool = False,
) -> V0UniverseScore:
    """Score every discovered V0 model for one universe on the 2025 fold.

    Pass ``frames_by_model`` to reuse a once-loaded family (see
    ``discover_all_v0_predictions``); otherwise ``runs_root`` is scanned.
    """
    loader = CanonicalDatasetLoader(canonical_root, universe)
    trading_calendar = load_trading_calendar(staged_calendar)
    calendar_hash = canonical_hash(trading_calendar)
    labels = build_validation_labels(loader, trading_calendar, calendar_hash=calendar_hash)
    eligible_keys = build_eligible_keys(loader)
    if frames_by_model is None:
        if runs_root is None:
            raise ContractError("score_v0_universe requires runs_root or frames_by_model")
        frames_by_model = discover_v0_predictions(runs_root, universe, verify_hashes=verify_hashes)
    if not frames_by_model:
        raise ContractError(f"no V0 predictions discovered for universe {universe!r}")

    family: dict[str, PredictionFrame] = {}
    for model, seeds in frames_by_model.items():
        for seed, frame in seeds.items():
            family[f"{model}@{seed}"] = frame
    common_support = freeze_common_support(family, eligible_keys)

    scorecards: list[PredictionScorecard] = []
    failures: list[V0CellFailure] = []
    rank_ic_by_config: dict[tuple[str, SupportMode, OutcomeMode], dict[int, float]] = {}
    for model in sorted(frames_by_model):
        for seed in sorted(frames_by_model[model]):
            frame = frames_by_model[model][seed]
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
                                (model, support, outcome), {}
                            )[seed] = result.rank_ic
                    else:
                        failures.append(result)

    aggregates: list[V0ModelAggregate] = []
    for model in sorted(frames_by_model):
        for support in _SUPPORTS:
            for outcome in _OUTCOMES:
                seed_values = rank_ic_by_config.get((model, support, outcome), {})
                if not seed_values:
                    continue
                if len(seed_values) >= 2:
                    mean, std = aggregate_seed_estimates(seed_values)
                else:
                    mean = next(iter(seed_values.values()))
                    std = None
                aggregates.append(
                    V0ModelAggregate(
                        model=model,
                        support=support,
                        outcome=outcome,
                        seed_count=len(seed_values),
                        rank_ic_mean=mean,
                        rank_ic_std=std,
                    )
                )

    seed_counts = tuple(
        (model, len(frames_by_model[model])) for model in sorted(frames_by_model)
    )
    return V0UniverseScore(
        universe=universe,
        trading_calendar_hash=calendar_hash,
        eligible_keys_count=len(eligible_keys),
        common_support_count=len(common_support),
        model_count=len(frames_by_model),
        seed_counts=seed_counts,
        scorecards=tuple(scorecards),
        failures=tuple(failures),
        aggregates=tuple(aggregates),
    )


__all__ = [
    "V0CellFailure",
    "V0ModelAggregate",
    "V0RunKey",
    "V0UniverseScore",
    "build_eligible_keys",
    "build_validation_labels",
    "discover_all_v0_predictions",
    "discover_v0_predictions",
    "expand_compact_label",
    "load_trading_calendar",
    "parse_run_id",
    "score_v0_universe",
]
