"""Search over groupings.

Phase 1: greedy.  Phase 2 adds "beam" and "exhaustive" to rank_groups.

Key trick: a board of n words has C(n,4) candidate groups (1,820 for 16). Score
each ONCE into a table; every search method then just reads the table.
"""
from __future__ import annotations

import itertools

Group4 = tuple[str, str, str, str]


def score_all_subsets(words, scorer, context=None) -> dict[Group4, float]:
    """Score every 4-word subset of `words`.

    `context` = the word list handed to scorer.score (defaults to `words`).
    Pass the full 16 here if you want contextual scorers to keep the original
    puzzle as context after some groups are solved.
    """
    words = tuple(words)
    ctx = tuple(context) if context is not None else words
    return {g: scorer.score(ctx, g) for g in itertools.combinations(words, 4)}


def _top_k(table: dict[Group4, float], k: int):
    return sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def greedy_partition(words, scorer, context=None) -> list[tuple[Group4, float]]:
    """Pick the best group, remove its words, repeat -> 4 groups (best first)."""
    remaining = list(words)
    table = score_all_subsets(remaining, scorer, context)
    picks = []
    while len(remaining) >= 4:
        live = {g: s for g, s in table.items() if set(g) <= set(remaining)}
        g, s = _top_k(live, 1)[0]  # ties broken alphabetically -> deterministic
        picks.append((g, s))
        remaining = [w for w in remaining if w not in g]
    return picks


def rank_groups(words, scorer, method: str = "greedy", k: int = 20, context=None):
    """Ranked candidate groups (best first) among `words`.

    greedy: the top-k individually-highest-scoring groups. The greedy *solver*
            is: guess rank_groups(...)[0], remove its words on "correct", call again.
            (Policy decides what to do on "wrong"/"one_away" -- e.g. take [1].)
    beam / exhaustive: Phase 2.
    """
    if method == "greedy":
        return _top_k(score_all_subsets(words, scorer, context), k)
    if method in ("beam", "exhaustive"):
        raise NotImplementedError(f"{method} search is Phase 2")
    raise ValueError(f"unknown method {method!r}")
