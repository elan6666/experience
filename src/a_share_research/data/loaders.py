"""Stable streaming adapters from canonical D0 rows to model input contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import TypeVar

from a_share_research.adapters.common.identity import CausalAssetMaster
from a_share_research.adapters.common.packing import PanelWindow
from a_share_research.contracts import (
    AssetRegistry,
    CanonicalModel,
    ContractError,
    DailyMarket,
    MaskBundle,
    PITFeature,
)
from a_share_research.data.labels import CompactLabel
from a_share_research.features.schema import InformationClass, d0_features
from a_share_research.models.tabular.samples import TabularSample

TCanonical = TypeVar("TCanonical", bound=CanonicalModel)


class CanonicalDatasetLoader:
    """Read canonical rows without pandas or model-specific preprocessing."""

    def __init__(self, canonical_root: Path, universe: str) -> None:
        self.canonical_root = canonical_root
        self.universe = universe.lower()
        self.universe_root = canonical_root / self.universe
        required = ("features.jsonl", "labels.jsonl", "masks.jsonl", "membership.jsonl")
        missing = tuple(name for name in required if not (self.universe_root / name).is_file())
        if missing:
            raise ContractError(f"canonical universe is incomplete: {missing}")

    @staticmethod
    def _rows(path: Path, row_type: type[TCanonical]) -> Iterator[TCanonical]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield row_type.from_dict(json.loads(line))

    def iter_features(self) -> Iterator[PITFeature]:
        for row in self._rows(self.universe_root / "features.jsonl", PITFeature):
            yield row

    def iter_labels(self) -> Iterator[CompactLabel]:
        for row in self._rows(self.universe_root / "labels.jsonl", CompactLabel):
            yield row

    def iter_masks(self) -> Iterator[MaskBundle]:
        for row in self._rows(self.universe_root / "masks.jsonl", MaskBundle):
            yield row

    def iter_daily_market(self, ts_code: str) -> Iterator[DailyMarket]:
        path = self.canonical_root / "common" / "daily_market" / f"{ts_code}.jsonl"
        if not path.is_file():
            raise ContractError(f"canonical daily market is absent for {ts_code}")
        for row in self._rows(path, DailyMarket):
            yield row

    def _group_features(self) -> Iterator[tuple[date, str, tuple[PITFeature, ...]]]:
        current: tuple[date, str] | None = None
        rows: list[PITFeature] = []
        for row in self.iter_features():
            key = (row.asof_date, row.ts_code)
            if current is not None and key != current:
                yield current[0], current[1], tuple(rows)
                rows = []
            current = key
            rows.append(row)
        if current is not None:
            yield current[0], current[1], tuple(rows)

    def iter_tabular_samples(
        self,
        *,
        horizon: int = 5,
        relative_target: bool = True,
        start: date | None = None,
        end: date | None = None,
        complete_panel: bool = False,
    ) -> Iterator[TabularSample]:
        """Yield adapter-ready rows inside one explicitly bounded protocol window.

        ``complete_panel`` expands every signal-date mask to its causal asset
        registry.  This preserves non-member and unobserved identities in the
        common PredictionFrame instead of silently dropping them.  Model
        training still consumes only scoreable rows with available targets.
        """
        if start is not None and end is not None and end < start:
            raise ContractError("tabular sample window is invalid")

        def in_window(day: date) -> bool:
            return (start is None or day >= start) and (end is None or day <= end)

        masks = {
            row.signal_date: row for row in self.iter_masks() if in_window(row.signal_date)
        }
        labels = {
            (row.signal_date, row.ts_code): row
            for row in self.iter_labels()
            if row.horizon == horizon and in_window(row.signal_date)
        }
        expected = tuple(item.name for item in d0_features())
        core = {
            item.name
            for item in d0_features()
            if item.information_class is InformationClass.CORE
        }
        features_by_date: dict[date, dict[str, tuple[PITFeature, ...]]] = {}
        for signal_date, code, feature_rows in self._group_features():
            if not in_window(signal_date):
                continue
            features_by_date.setdefault(signal_date, {})[code] = feature_rows

        def build_sample(
            signal_date: date,
            code: str,
            feature_rows: tuple[PITFeature, ...] | None,
        ) -> TabularSample:
            bundle = masks.get(signal_date)
            if bundle is None or code not in bundle.asset_ids:
                raise ContractError("canonical feature row lacks its mask identity")
            slot = bundle.asset_ids.index(code)
            member = bundle.member[slot]
            observed = bundle.observed[slot]
            if feature_rows is None:
                if member:
                    raise ContractError("active canonical member lacks its D0 feature group")
                return TabularSample(
                    signal_date=signal_date,
                    ts_code=code,
                    values={},
                    missing_flags={},
                    target=None,
                    member=False,
                    observed=observed,
                    complete_history=False,
                )
            by_name = {row.feature_name: row for row in feature_rows}
            if set(by_name) != set(expected):
                raise ContractError("canonical feature group does not match D0 schema")
            for name in expected:
                if bundle.feature_missing[name][slot] != by_name[name].missing_flag:
                    raise ContractError("feature row and independent mask bundle disagree")
            label = labels.get((signal_date, code))
            target = None
            if label is not None:
                target = (
                    label.open_to_open_return - label.benchmark_return
                    if relative_target
                    else label.open_to_open_return
                )
            missing = {name: by_name[name].missing_flag for name in expected}
            return TabularSample(
                signal_date=signal_date,
                ts_code=code,
                values={name: by_name[name].value for name in expected},
                missing_flags=missing,
                target=target,
                member=member,
                observed=observed,
                complete_history=member
                and observed
                and not any(missing[name] for name in core),
            )

        if complete_panel:
            for signal_date in sorted(masks):
                bundle = masks[signal_date]
                date_features = features_by_date.get(signal_date, {})
                unknown = set(date_features) - set(bundle.asset_ids)
                if unknown:
                    raise ContractError(
                        f"canonical features contain identities outside the mask: {sorted(unknown)}"
                    )
                for code in bundle.asset_ids:
                    yield build_sample(signal_date, code, date_features.get(code))
            return

        for signal_date in sorted(features_by_date):
            for code in sorted(features_by_date[signal_date]):
                yield build_sample(
                    signal_date,
                    code,
                    features_by_date[signal_date][code],
                )

    @staticmethod
    def _expand_mask(bundle: MaskBundle, asset_ids: tuple[str, ...]) -> MaskBundle:
        unknown = set(bundle.asset_ids) - set(asset_ids)
        if unknown:
            raise ContractError(
                f"mask contains identities unknown to the causal fold master: {sorted(unknown)}"
            )
        source_slot = {code: index for index, code in enumerate(bundle.asset_ids)}

        def expand(values: Sequence[bool], default: bool) -> tuple[bool, ...]:
            return tuple(
                values[source_slot[code]] if code in source_slot else default
                for code in asset_ids
            )

        registry = AssetRegistry(asset_ids)
        return MaskBundle(
            signal_date=bundle.signal_date,
            asset_ids=asset_ids,
            asset_registry_hash=registry.stable_hash(),
            member=expand(bundle.member, False),
            observed=expand(bundle.observed, False),
            feature_missing={
                name: expand(values, True)
                for name, values in bundle.feature_missing.items()
            },
            label_available=expand(bundle.label_available, False),
            buyable=expand(bundle.buyable, False),
            sellable=expand(bundle.sellable, False),
            loss=expand(bundle.loss, False),
            evaluation=expand(bundle.evaluation, False),
        )

    def load_panel_window(
        self,
        *,
        dates: tuple[date, ...],
        asset_master: CausalAssetMaster,
    ) -> PanelWindow:
        """Create the exact fixed-slot PanelWindow consumed by deep adapters."""
        if not dates or dates != tuple(sorted(set(dates))):
            raise ContractError("panel dates must be unique and increasing")
        if dates[-1] > asset_master.known_through:
            raise ContractError("panel extends beyond the causal asset-master cutoff")
        requested = set(dates)
        source_masks = {
            row.signal_date: row for row in self.iter_masks() if row.signal_date in requested
        }
        if set(source_masks) != requested:
            raise ContractError("canonical masks do not cover every requested panel date")
        masks = tuple(
            self._expand_mask(source_masks[day], asset_master.asset_ids) for day in dates
        )
        feature_names = tuple(item.name for item in d0_features())
        date_slot = {day: index for index, day in enumerate(dates)}
        asset_slot = {code: index for index, code in enumerate(asset_master.asset_ids)}
        grids: dict[str, list[list[float | None]]] = {
            name: [[None] * len(asset_slot) for _ in dates]
            for name in feature_names
        }
        for row in self.iter_features():
            if row.asof_date not in requested or row.ts_code not in asset_slot:
                continue
            grids[row.feature_name][date_slot[row.asof_date]][asset_slot[row.ts_code]] = row.value
        values = {
            name: tuple(tuple(row) for row in grids[name])
            for name in feature_names
        }
        return PanelWindow(
            dates=dates,
            asset_master=asset_master,
            values=values,
            masks=masks,
        )
