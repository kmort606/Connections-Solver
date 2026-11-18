"""Step 1 check. Run from repo root:  python -m pytest -q"""
import dataclasses

import pytest

from src.puzzle import LEVELS, Group, Puzzle


def make_puzzle():
    groups = (
        Group(("BASS", "PIKE", "SOLE", "TROUT"), "FISH", 0),
        Group(("DRUM", "HARP", "OBOE", "TUBA"), "INSTRUMENTS", 1),
        Group(("NA", "NAN", "NONE", "NULL"), "NOTHING-ISH", 2),
        Group(("100", "11", "42", "7"), "NUMBERS", 3),
    )
    words = tuple(w for g in groups for w in g.words)
    return Puzzle(id=1, date="2024-01-02", words=words, groups=groups)


def test_shape():
    p = make_puzzle()
    assert len(p.words) == 16 and len(p.groups) == 4
    assert [LEVELS[g.level] for g in p.groups] == ["yellow", "green", "blue", "purple"]


def test_frozen_and_hashable():
    p = make_puzzle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.id = 2                      # frozen: can't be mutated by accident
    assert {p.words: "ok"}[p.words]   # hashable: usable as a cache key
