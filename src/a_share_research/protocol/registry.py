"""Pre-registered experiment table that cannot drift after sealing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from a_share_research.contracts.base import ContractError, canonical_hash


@dataclass(frozen=True)
class RegisteredExperiment:
    experiment_id: str
    model: str
    universe: str
    information_set: str
    config_hash: str

    def validate(self) -> None:
        if not all(
            (self.experiment_id, self.model, self.universe, self.information_set, self.config_hash)
        ):
            raise ContractError("registered experiment fields cannot be empty")
        if len(self.config_hash) != 64:
            raise ContractError("config_hash must be SHA-256")


class ExperimentRegistry:
    """Mutable only before seal; the seal hash is the protocol receipt."""

    def __init__(self) -> None:
        self._experiments: dict[str, RegisteredExperiment] = {}
        self._sealed_hash: str | None = None

    @property
    def experiments(self) -> Mapping[str, RegisteredExperiment]:
        return MappingProxyType(self._experiments)

    @property
    def sealed_hash(self) -> str | None:
        if self._sealed_hash is not None and self._current_hash() != self._sealed_hash:
            raise ContractError("sealed experiment registry was mutated")
        return self._sealed_hash

    def register(self, experiment: RegisteredExperiment) -> None:
        if self._sealed_hash is not None:
            raise ContractError("experiment registry is sealed")
        experiment.validate()
        if experiment.experiment_id in self._experiments:
            raise ContractError(f"duplicate experiment_id: {experiment.experiment_id}")
        self._experiments[experiment.experiment_id] = experiment

    def _current_hash(self) -> str:
        payload = {
            key: {
                "model": item.model,
                "universe": item.universe,
                "information_set": item.information_set,
                "config_hash": item.config_hash,
            }
            for key, item in sorted(self._experiments.items())
        }
        return canonical_hash(payload)

    def seal(self) -> str:
        if not self._experiments:
            raise ContractError("cannot seal an empty experiment registry")
        current = self._current_hash()
        if self._sealed_hash is not None and current != self._sealed_hash:
            raise ContractError("sealed experiment registry was mutated")
        self._sealed_hash = current
        return current

    def assert_registered(self, experiment_id: str, config_hash: str) -> None:
        if self._sealed_hash is None:
            raise ContractError("experiment registry must be sealed before execution")
        if self._current_hash() != self._sealed_hash:
            raise ContractError("sealed experiment registry was mutated")
        experiment = self._experiments.get(experiment_id)
        if experiment is None or experiment.config_hash != config_hash:
            raise ContractError("experiment or config is not in the sealed registry")
