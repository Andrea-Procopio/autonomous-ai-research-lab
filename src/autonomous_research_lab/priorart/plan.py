"""The trusted query-rendering boundary: structured search plans in,
bounded Boolean OpenAlex queries out.

Task 5D's live evidence isolated a query-semantics defect: the model
proposed ten-plus-term search strings, OpenAlex's
``title_and_abstract.search`` treats bare whitespace-separated terms
conjunctively, and sixteen of eighteen searches returned nothing —
synonyms the model intended as alternatives executed as simultaneous
requirements. The repair is structural: the model no longer supplies a
final query string at all. It supplies *concept groups* — each group a
set of alternative terms or phrases for one concept — and trusted code
renders the one Boolean expression those groups mean:

    groups are conjoined with AND, alternatives within a group are
    joined with OR, and every term is quoted as an exact phrase:

        ("attention head reweighting" OR "head gating") AND
        ("in-context learning")

The Boolean behavior was verified live against OpenAlex on 2026-08-19
through the title/abstract-restricted path (union arithmetic closed
exactly: 9942 + 188 - 86 = 10044 for A, B, A OR B with A AND B = 86;
the ten-bare-term failure shape reproduced 0; lowercase operators read
as plain terms). Operators are uppercase and structural only — a model
term containing quotes, parentheses, wildcards, or operator words is a
gate rejection, never silently escaped.

Bounds are mechanical and justified by the observed failure: the 5D
queries conflated roughly ten concepts, so a rendered query may conjoin
at most :data:`MAX_CONCEPT_GROUPS` groups — enough for a concept, a
qualifier, and a context — and offer at most
:data:`MAX_ALTERNATIVES_PER_GROUP` alternatives per group. Excess is
rejected, never silently truncated. Rendering canonicalizes order
(groups and alternatives sort case-insensitively): the vocabulary
declares ordering non-semantic, so two plans meaning the same search
render — and fingerprint — identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

#: Identifies the deterministic rendering scheme on every execution
#: record, so a replayed run knows exactly how its queries were built.
RENDERER_VERSION: Final = "boolean-v1"

#: A search conjoins at most this many concept groups. Task 5D's failed
#: queries conflated ~10 concepts; three suffices for concept +
#: qualifier + context, and anything wider re-creates the defect.
MAX_CONCEPT_GROUPS: Final = 3

#: Alternatives are synonyms for one concept, not a topic list.
MAX_ALTERNATIVES_PER_GROUP: Final = 4

MIN_TERM_CHARS: Final = 2
MAX_TERM_CHARS: Final = 60

#: The rendered Boolean expression's ceiling: quotes and operators are
#: overhead on top of the term bounds, and a longer expression than
#: this is a plan trying to be an essay.
MAX_RENDERED_CHARS: Final = 400

#: Characters and words that would smuggle Boolean structure, wildcards,
#: or fuzzy syntax through a plain term. Structure comes from the plan
#: shape; a term carrying its own is rejected, never escaped.
FORBIDDEN_TERM_CHARACTERS: Final = frozenset('"()|&*?~[]{}<>\\')
FORBIDDEN_TERM_WORDS: Final = frozenset({"and", "or", "not"})

#: The seven search intents of the Task 5D specification, mapped to the
#: six implemented query families. COMPETING_APPROACHES deliberately
#: serves both the closest-cited-work and the competing-approaches
#: intents: the candidate's cited sources additionally enter the pool
#: directly through the trusted cited-source bridge, so the closest
#: cited work is both searched for and injected. The mapping is data so
#: a test can pin that every intent is served.
REQUIRED_INTENTS: Final = {
    "exact_mechanism": "mechanism",
    "problem_plus_mechanism": "problem_mechanism",
    "task_dataset_metric": "evaluation_setup",
    "synonyms_and_older_terminology": "synonyms_legacy",
    "closest_cited_work": "competing_approaches",
    "competing_approaches": "competing_approaches",
    "latest_work_to_cutoff": "recent",
}


def normalized_term(term: str) -> str:
    """One term's canonical form: casefolded, whitespace-collapsed."""
    return " ".join(term.casefold().split())


def canonical_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """The canonical plan: alternatives normalized and sorted within
    each group, groups sorted by their rendered content. Ordering is
    non-semantic (AND and OR are commutative), so canonicalization
    makes rendering, identity, and duplicate detection order-invariant
    by construction."""
    return tuple(
        sorted(
            (
                tuple(
                    sorted(
                        {normalized_term(term) for term in group}
                    )
                )
                for group in groups
            ),
        )
    )


def render_query(groups: Sequence[Sequence[str]]) -> str:
    """The one Boolean expression a canonical plan means. Every
    alternative is quoted as an exact phrase — single words included —
    so no term can be re-read as an operator; OR joins alternatives
    inside parentheses; AND conjoins the groups. The caller validates
    the plan first: rendering never repairs."""
    rendered_groups = []
    for group in canonical_groups(groups):
        alternatives = " OR ".join(f'"{term}"' for term in group)
        rendered_groups.append(f"({alternatives})")
    return " AND ".join(rendered_groups)
