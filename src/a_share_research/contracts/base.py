"""Deterministic serialization primitives shared by all contracts."""

from __future__ import annotations

import hashlib
import json
import math
import types
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar, TypeVar, Union, get_args, get_origin, get_type_hints


class ContractError(ValueError):
    """Raised when a research boundary would otherwise fail open."""


def _encode(value: Any) -> Any:
    if isinstance(value, CanonicalModel):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(value[key]) for key in sorted(value)}
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("non-finite numbers cannot be serialized")
    return value


def _decode(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        candidates = [candidate for candidate in arguments if candidate is not type(None)]
        if len(candidates) == 1:
            return _decode(candidates[0], value)
    if origin is tuple:
        item_type = arguments[0] if arguments else Any
        return tuple(_decode(item_type, item) for item in value)
    if origin is list:
        item_type = arguments[0] if arguments else Any
        return [_decode(item_type, item) for item in value]
    if origin is dict:
        key_type, item_type = arguments if arguments else (Any, Any)
        return {
            _decode(key_type, key): _decode(item_type, item)
            for key, item in value.items()
        }
    if annotation is datetime:
        return datetime.fromisoformat(value)
    if annotation is date:
        return date.fromisoformat(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, CanonicalModel):
        return annotation.from_dict(value)
    return value


TCanonical = TypeVar("TCanonical", bound="CanonicalModel")


class CanonicalModel:
    """Mixin for strict, versioned and deterministically hashed dataclasses."""

    SCHEMA_NAME: ClassVar[str]
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        """Make invalid contract objects impossible to construct silently."""
        self.validate()

    def validate(self) -> None:
        """Validate semantic invariants; subclasses must implement this."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            field.name: _encode(getattr(self, field.name))
            for field in fields(self)
            if field.init
        }
        return {
            "_schema": self.SCHEMA_NAME,
            "_version": self.SCHEMA_VERSION,
            **payload,
        }

    @classmethod
    def from_dict(cls: type[TCanonical], payload: dict[str, Any]) -> TCanonical:
        if payload.get("_schema") != cls.SCHEMA_NAME:
            raise ContractError(f"expected schema {cls.SCHEMA_NAME!r}")
        if payload.get("_version") != cls.SCHEMA_VERSION:
            raise ContractError(f"unsupported version for {cls.SCHEMA_NAME}")
        declared = {field.name: field for field in fields(cls) if field.init}
        supplied = set(payload) - {"_schema", "_version"}
        unknown = supplied - set(declared)
        missing = {
            name
            for name, field in declared.items()
            if name not in supplied
            and field.default is MISSING
            and field.default_factory is MISSING
        }
        if unknown or missing:
            raise ContractError(
                f"invalid {cls.SCHEMA_NAME} keys; "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        type_hints = get_type_hints(cls)
        kwargs = {
            name: _decode(type_hints.get(name, Any), payload[name])
            for name in supplied
        }
        instance = cls(**kwargs)
        instance.validate()
        return instance

    def stable_hash(self) -> str:
        return canonical_hash(self.to_dict())


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for content addressing."""
    return json.dumps(
        _encode(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    """Hash a canonical JSON payload with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite")


def require_nonnegative(value: float, name: str) -> None:
    require_finite(value, name)
    if value < 0:
        raise ContractError(f"{name} must be non-negative")
