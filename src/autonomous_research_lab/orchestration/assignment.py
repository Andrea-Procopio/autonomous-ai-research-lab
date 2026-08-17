"""Deciding who performs a chosen action.

Kept separate from the director because "what should we do next" and "who
should do it" are different decisions with different failure modes, and folding
them together is how a system ends up with a single manager object that owns
everything.

Assignment consults each capable role's own utility, so the choice reflects
which role values the work most highly by its own objective, rather than a
central ranking imposed on all of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from ..core.actions import ResearchAction
from ..core.state import ResearchState
from ..roles.base import ResearchRole


class RoleAssigner(Protocol):
    def assign(
        self, state: ResearchState, action: ResearchAction
    ) -> ResearchRole | None: ...


class RegistryAssigner:
    def __init__(self, roles: Iterable[ResearchRole]) -> None:
        self._roles: Sequence[ResearchRole] = tuple(roles)

    @property
    def roles(self) -> Sequence[ResearchRole]:
        return self._roles

    def assign(
        self, state: ResearchState, action: ResearchAction
    ) -> ResearchRole | None:
        capable = [r for r in self._roles if r.can_perform(action.action_type)]
        if not capable:
            return None
        return max(
            capable,
            key=lambda role: (role.utility(state, action).value, role.name),
        )
