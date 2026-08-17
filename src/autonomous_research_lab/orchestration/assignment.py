"""Deciding who performs a chosen action.

Kept separate from the director because "what should we do next" and "who
should do it" are different decisions with different failure modes, and
folding them together is how a system ends up with a single manager object
that owns everything. The split, stated once:

    ActionUtility     U(a | state)                     — is this worth doing?
    RoleSuitability   ≈ P(role succeeds | action, state) — who should do it?

Assignment happens strictly *after* selection: suitability never feeds back
into which action is chosen, so "what is scientifically valuable" cannot
quietly become "what our current roles are good at".
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
            key=lambda role: (role.suitability(state, action).value, role.name),
        )
