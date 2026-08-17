"""Role abstractions.

Only the base contract lives here for now. Concrete roles need a model provider
to do anything, and adding stub roles that cannot act would make the package
look further along than it is.
"""

from .base import (
    ResearchRole,
    RoleName,
    UtilityFunction,
    UtilityScore,
    WeightedUtility,
)

__all__ = [
    "ResearchRole",
    "RoleName",
    "UtilityFunction",
    "UtilityScore",
    "WeightedUtility",
]
