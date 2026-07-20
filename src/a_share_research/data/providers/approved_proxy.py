"""Thin adapter around the server-owned approved proxy client.

The project deliberately does not know where or how the credential is read.
The only permitted client factory is ``scripts/tushare_proxy_client.py:get_pro``
from the parent server research root.
"""

from __future__ import annotations

import importlib.util
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

from a_share_research.contracts import ContractError, canonical_hash

_SAFE_ENDPOINT = re.compile(r"^[a-z][a-z0-9_]*$")
_FORBIDDEN_PARAM_PARTS = ("token", "secret", "password", "credential", "api_key")


class ProviderClient(Protocol):
    def query(self, api_name: str, fields: str = "", **kwargs: object) -> object: ...


def assert_private_credential_metadata(path: Path) -> None:
    """Check existence/type/mode without opening or reading credential bytes."""
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError("server provider credential must be a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ContractError("server provider credential mode must be 0600")


@dataclass(frozen=True)
class QueryRequest:
    endpoint: str
    params: Mapping[str, str | int | float | bool]
    fields: tuple[str, ...]
    partition_key: str
    min_row_count: int = 0
    reject_at_row_count: int = 5000

    def __post_init__(self) -> None:
        if not _SAFE_ENDPOINT.fullmatch(self.endpoint):
            raise ContractError("unsafe provider endpoint")
        if not self.partition_key or not self.fields:
            raise ContractError("partition_key and explicit fields are required")
        if self.min_row_count < 0:
            raise ContractError("provider minimum row count cannot be negative")
        if self.reject_at_row_count < 1 or self.min_row_count >= self.reject_at_row_count:
            raise ContractError("provider row-count bounds are invalid")
        lowered = tuple(str(key).lower() for key in self.params)
        if any(part in key for key in lowered for part in _FORBIDDEN_PARAM_PARTS):
            raise ContractError("credential-like provider parameters are forbidden")

    @property
    def request_hash(self) -> str:
        return canonical_hash(
            {
                "endpoint": self.endpoint,
                "params": dict(self.params),
                "fields": self.fields,
                "partition_key": self.partition_key,
                "min_row_count": self.min_row_count,
                "reject_at_row_count": self.reject_at_row_count,
            }
        )


@dataclass(frozen=True)
class QueryResult:
    request: QueryRequest
    rows: tuple[dict[str, object], ...]

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.rows)


def _load_module(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("approved_tushare_proxy_client", path)
    if specification is None or specification.loader is None:
        raise ContractError("approved proxy client cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _records(value: object) -> tuple[dict[str, object], ...]:
    if hasattr(value, "to_dict"):
        records = value.to_dict(orient="records")  # type: ignore[call-arg]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records = value
    else:
        raise ContractError("provider response must be a table or row sequence")
    normalized: list[dict[str, object]] = []
    for row in records:
        if not isinstance(row, Mapping):
            raise ContractError("provider response contains a non-row value")
        normalized.append({str(key): _json_scalar(row[key]) for key in sorted(row)})
    return tuple(normalized)


def _json_scalar(value: object) -> object:
    """Normalize provider dataframe scalars without importing pandas or NumPy."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    scalar_type = type(value)
    if scalar_type.__module__.startswith("pandas") and scalar_type.__name__ in {
        "NAType",
        "NaTType",
    }:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _json_scalar(converted)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    raise ContractError(f"provider response contains unsupported scalar {scalar_type.__name__}")


class ApprovedProxyProvider:
    """Injected or approved-script-created provider with no credential surface."""

    def __init__(self, client: ProviderClient) -> None:
        self._client = client

    @classmethod
    def from_server_research_root(cls, server_research_root: Path) -> ApprovedProxyProvider:
        script = (server_research_root / "scripts" / "tushare_proxy_client.py").resolve()
        if script.name != "tushare_proxy_client.py" or not script.is_file():
            raise ContractError("approved server proxy client is missing")
        module = _load_module(script)
        factory = getattr(module, "get_pro", None)
        if not callable(factory):
            raise ContractError("approved proxy client must expose get_pro")
        return cls(factory())

    def execute(self, request: QueryRequest) -> QueryResult:
        response = self._client.query(
            request.endpoint,
            fields=",".join(request.fields),
            **dict(request.params),
        )
        rows = _records(response)
        if len(rows) < request.min_row_count:
            raise ContractError("provider response fell below the configured minimum row count")
        if len(rows) >= request.reject_at_row_count:
            raise ContractError("provider response reached the configured truncation boundary")
        return QueryResult(request=request, rows=rows)
