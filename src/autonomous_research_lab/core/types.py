"""Shared primitive types and immutability helpers for the domain core."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar

#: Values permitted in experiment configuration. Deliberately flat and scalar:
#: a configuration that cannot be written down as scalars is usually hiding a
#: dependency that belongs in the experiment code, not in the config.
ConfigValue = str | int | float | bool | None

_K = TypeVar("_K")
_V = TypeVar("_V")


def freeze_mapping(mapping: Mapping[_K, _V]) -> Mapping[_K, _V]:
    """Return a read-only view over a copy of ``mapping``."""
    return MappingProxyType(dict(mapping))


def canonical(value: object) -> str:
    """Render ``value`` deterministically for content-addressed identifiers."""
    if isinstance(value, Mapping):
        items = sorted((str(k), canonical(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    return str(value)
