"""Role abstractions.

Only the contracts live here for now. Concrete roles need a model provider
to do anything, and adding stub roles that cannot act would make the package
look further along than it is.
"""

from .base import (
    ResearchRole,
    RoleContext,
    RoleInvocation,
    RoleName,
    RoleSuitability,
    SuitabilityFunction,
    WeightedSuitability,
)

__all__ = [
    "ResearchRole",
    "RoleContext",
    "RoleInvocation",
    "RoleName",
    "RoleSuitability",
    "SuitabilityFunction",
    "WeightedSuitability",
]
