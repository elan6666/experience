"""Causal fold masters with append-only, permanent stock slots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from a_share_research.adapters.common.types import AdapterContractError
from a_share_research.contracts import AssetRegistry, UniverseMembership, canonical_hash


@dataclass(frozen=True)
class CausalAssetMaster:
    """Identity known at a retrain cutoff; later members are unsupported until retrain."""

    registry: AssetRegistry
    universe: str
    known_through: date
    source_membership_hash: str
    parent_registry_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.universe:
            raise AdapterContractError("asset master universe is required")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_membership_hash) is None:
            raise AdapterContractError("source_membership_hash must be SHA-256")
        if self.parent_registry_hash is not None and re.fullmatch(
            r"[0-9a-f]{64}", self.parent_registry_hash
        ) is None:
            raise AdapterContractError("parent_registry_hash must be SHA-256")

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return self.registry.asset_ids

    @property
    def stable_hash(self) -> str:
        return canonical_hash(
            {
                "registry_hash": self.registry.stable_hash(),
                "universe": self.universe,
                "known_through": self.known_through,
                "source_membership_hash": self.source_membership_hash,
                "parent_registry_hash": self.parent_registry_hash,
            }
        )

    def slot(self, ts_code: str) -> int:
        return self.registry.index_of(ts_code)

    def supports(self, ts_code: str) -> bool:
        return ts_code in self.registry.asset_ids


def _membership_evidence(rows: tuple[UniverseMembership, ...]) -> str:
    return canonical_hash(tuple(row.to_dict() for row in rows))


def build_causal_asset_master(
    memberships: Iterable[UniverseMembership],
    *,
    known_through: date,
    previous: CausalAssetMaster | None = None,
) -> CausalAssetMaster:
    """Append only identities whose membership evidence existed by the cutoff."""
    rows = tuple(
        sorted(
            (
                row
                for row in memberships
                if row.asof_date <= known_through and row.effective_from <= known_through
            ),
            key=lambda row: (row.effective_from, row.asof_date, row.ts_code, row.universe),
        )
    )
    if not rows and previous is None:
        raise AdapterContractError("no membership identity was known by the retrain cutoff")
    if previous is not None and known_through <= previous.known_through:
        raise AdapterContractError("a retrain cutoff must advance beyond its parent master")
    universes = {row.universe for row in rows}
    if previous is not None:
        universes.add(previous.universe)
    if len(universes) != 1:
        raise AdapterContractError("one causal asset master cannot mix universes")
    universe = next(iter(universes))
    base = previous.registry if previous is not None else None
    existing = set(base.asset_ids) if base is not None else set()
    additions: list[str] = []
    for row in rows:
        if row.ts_code not in existing:
            additions.append(row.ts_code)
            existing.add(row.ts_code)
    registry = base.append(*additions) if base is not None else AssetRegistry(tuple(additions))
    return CausalAssetMaster(
        registry=registry,
        universe=universe,
        known_through=known_through,
        source_membership_hash=_membership_evidence(rows),
        parent_registry_hash=base.stable_hash() if base is not None else None,
    )


def require_stable_slot_series(
    master: CausalAssetMaster,
    dated_asset_ids: Iterable[tuple[date, tuple[str, ...]]],
) -> None:
    """Reject daily sorting/repacking; every encoded day must use the frozen master."""
    rows = tuple(dated_asset_ids)
    dates = tuple(item[0] for item in rows)
    if dates != tuple(sorted(set(dates))):
        raise AdapterContractError("slot-series dates must be unique and increasing")
    for signal_date, asset_ids in rows:
        if asset_ids != master.asset_ids:
            raise AdapterContractError(
                f"asset slots changed on {signal_date}; daily slot reordering is forbidden"
            )
