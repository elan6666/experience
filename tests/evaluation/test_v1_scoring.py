"""End-to-end tests for the V1 information-ablation scoring slice."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from a_share_research.contracts import (
    AssetRegistry,
    CoverageState,
    PredictionFrame,
    PredictionRecord,
    canonical_hash,
)
from a_share_research.data.eligibility import ExecutionStatus, build_mask_bundle
from a_share_research.data.labels import build_compact_open_labels
from a_share_research.evaluation.schema import OutcomeMode, SupportMode
from a_share_research.evaluation.v1_scoring import (
    parse_v1_run_id,
    score_v1_universe,
)

_STOCKS = ("000001.SZ", "000002.SZ")
_SIGNAL_INDICES = (0, 5, 10)


def _weekdays(start: date, count: int) -> tuple[date, ...]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _build_canonical(tmp_path: Path) -> tuple[Path, Path, tuple[date, ...]]:
    calendar = _weekdays(date(2025, 1, 2), 30)
    cal_hash = canonical_hash(calendar)
    benchmark_opens = {day: 100.0 + index for index, day in enumerate(calendar)}
    stock_opens = {
        "000001.SZ": {day: 10.0 + index for index, day in enumerate(calendar)},
        "000002.SZ": {day: 8.0 + index * 1.35 for index, day in enumerate(calendar)},
    }
    label_rows: list[dict[str, object]] = []
    for ts_code in _STOCKS:
        for signal_index in _SIGNAL_INDICES:
            for compact in build_compact_open_labels(
                ts_code=ts_code,
                signal_index=signal_index,
                trading_calendar=calendar,
                trading_calendar_hash=cal_hash,
                opens=stock_opens[ts_code],
                benchmark_opens=benchmark_opens,
                horizons=(5,),
            ):
                label_rows.append(compact.to_dict())
    registry = AssetRegistry(_STOCKS)
    statuses = {
        code: ExecutionStatus(
            observed=True,
            suspended_at_open=False,
            open_price=10.0,
            up_limit=11.0,
            down_limit=9.0,
        )
        for code in _STOCKS
    }
    mask_rows: list[dict[str, object]] = []
    for signal_index in _SIGNAL_INDICES:
        bundle = build_mask_bundle(
            signal_date=calendar[signal_index],
            asset_registry=registry,
            member={code: True for code in _STOCKS},
            statuses=statuses,
            feature_missing={"return_1d": {code: False for code in _STOCKS}},
            label_available={code: True for code in _STOCKS},
        )
        mask_rows.append(bundle.to_dict())
    canonical = tmp_path / "canonical" / "d0-v1"
    universe_root = canonical / "csi300"
    _write_jsonl(universe_root / "labels.jsonl", label_rows)
    _write_jsonl(universe_root / "masks.jsonl", mask_rows)
    (universe_root / "features.jsonl").write_text("", encoding="utf-8")
    (universe_root / "membership.jsonl").write_text("", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    calendar_rows: list[dict[str, object]] = [
        {"cal_date": day.strftime("%Y%m%d"), "exchange": "SSE", "is_open": "1"}
        for day in calendar
    ]
    calendar_rows.append({"cal_date": "20181231", "exchange": "SSE", "is_open": "1"})
    _write_jsonl(staged / "trade_calendar.jsonl", calendar_rows)
    return canonical, staged / "trade_calendar.jsonl", calendar


def _write_v1_predictions(
    runs_root: Path,
    gate: str,
    model: str,
    seed: int,
    scores: list[float],
    calendar: tuple[date, ...],
) -> None:
    signal_dates = tuple(calendar[index] for index in _SIGNAL_INDICES)
    records = tuple(
        PredictionRecord(
            signal_date=signal_date,
            ts_code=code,
            score=scores.pop(0),
            coverage_state=CoverageState.SCORED,
        )
        for signal_date in signal_dates
        for code in _STOCKS
    )
    frame = PredictionFrame(
        run_id=f"v1-a{gate}-csi300-{model}-seed-{seed:08d}",
        records=records,
    )
    cell_root = runs_root / f"v1-a{gate}-csi300-{model}-seed-{seed:08d}"
    cell_root.mkdir(parents=True, exist_ok=True)
    (cell_root / "predictions.json").write_text(
        json.dumps(frame.to_dict(), sort_keys=True), encoding="utf-8"
    )
    (cell_root / "run_manifest.json").write_text(
        json.dumps(
            {"run_id": frame.run_id, "prediction_hash": frame.stable_hash(), "status": "PASS"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_parse_v1_run_id_accepts_v1_and_rejects_other_phases() -> None:
    key = parse_v1_run_id("v1-a2-csi300-ridge-seed-20260719")
    assert key is not None
    assert key.gate == "2"
    assert key.universe == "csi300"
    assert key.model == "ridge"
    assert key.seed == 20260719
    assert key.run_id == "v1-a2-csi300-ridge-seed-20260719"
    assert parse_v1_run_id("v0-a0-csi300-ridge-seed-20260719") is None
    assert parse_v1_run_id("not-a-run-id") is None


def test_score_v1_universe_produces_scorecards_and_aggregates(tmp_path: Path) -> None:
    canonical, staged, calendar = _build_canonical(tmp_path)
    runs_root = tmp_path / "runs" / "v1" / "v1"
    _write_v1_predictions(
        runs_root, "1", "ridge", 20260719,
        [0.10, -0.05, 0.20, -0.10, 0.30, -0.15], calendar,
    )
    _write_v1_predictions(
        runs_root, "2", "ridge", 20260719,
        [-0.10, 0.05, -0.20, 0.10, -0.30, 0.15], calendar,
    )
    _write_v1_predictions(
        runs_root, "1", "lightgbm", 20260719,
        [0.15, -0.02, 0.18, -0.08, 0.25, -0.12], calendar,
    )

    result = score_v1_universe(
        canonical_root=canonical,
        universe="csi300",
        staged_calendar=staged,
        runs_root=runs_root,
    )

    assert result.universe == "csi300"
    assert result.gate_count == 2
    assert result.model_count == 2
    assert result.eligible_keys_count == 6
    assert result.common_support_count == 6
    assert len(result.scorecards) == 12
    assert not result.failures

    common_relative = [
        card
        for card in result.scorecards
        if card.support is SupportMode.COMMON
        and card.outcome is OutcomeMode.BENCHMARK_RELATIVE
    ]
    assert len(common_relative) == 3
    for card in common_relative:
        assert card.paired_rows == 6
        assert card.paired_dates == 3
        assert card.coverage == 1.0
        assert card.rank_ic is not None

    gate1_ridge = [
        agg
        for agg in result.aggregates
        if agg.gate == "1"
        and agg.model == "ridge"
        and agg.support is SupportMode.COMMON
        and agg.outcome is OutcomeMode.BENCHMARK_RELATIVE
    ]
    assert len(gate1_ridge) == 1
    assert gate1_ridge[0].seed_count == 1
    assert gate1_ridge[0].rank_ic_mean is not None
    assert gate1_ridge[0].rank_ic_std is None
