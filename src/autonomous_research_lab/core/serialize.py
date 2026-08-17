"""Deterministic JSON-compatible rendering of core domain objects.

Used by trajectory logging. Deliberately one-way: parsing serialized objects
back into the domain is a boundary concern (it needs validation), and boundary
concerns stay out of ``core``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum

Jsonable = None | bool | int | float | str | list["Jsonable"] | dict[str, "Jsonable"]


def to_jsonable(value: object) -> Jsonable:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")
