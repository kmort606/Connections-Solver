"""Step 4 checks with a perfect toy scorer. Run:  python -m pytest -q"""
import numpy as np

from scripts.run_baseline import _play
from src.baseline import EmbeddingScorer
from src.puzzle import Group, Puzzle
from src.search import greedy_partition, rank_groups, score_all_subsets

GROUPS = [["BASS", "PIKE", "SOLE", "TROUT"], ["DRUM", "HARP", "OBOE", "TUBA"],
          ["NA", "NAN", "NONE", "NULL"], ["100", "11", "42", "7"]]
P = Puzzle(1, "2024-01-02", tuple(w for g in GROUPS for w in g),
           tuple(Group(tuple(g), f"G{i}", i) for i, g in enumerate(GROUPS)))


class OneHot(EmbeddingScorer):
    """Each word's vector points at its true group -> true groups score 1, others less."""
    def _embed_puzzle(self, words):
        gi = {w: i for i, g in enumerate(GROUPS) for w in g}
        return np.stack([np.eye(4)[gi[w]] + 0.01 for w in words])


def test_all_1820_subsets_scored():
    assert len(score_all_subsets(P.words, OneHot())) == 1820


def test_rank_groups_top4_are_true_groups():
    ranked = rank_groups(P.words, OneHot(), k=10)
    assert len(ranked) == 10 and all(ranked[i][1] >= ranked[i + 1][1] for i in range(9))
    assert {frozenset(g) for g, _ in ranked[:4]} == {frozenset(g) for g in GROUPS}


def test_rank_groups_on_remaining_words():
    remaining = tuple(w for g in GROUPS[1:] for w in g)          # 12 words left
    ranked = rank_groups(remaining, OneHot(), k=5)
    assert all(set(g) <= set(remaining) for g, _ in ranked)


def test_greedy_partition_solves_toy():
    assert {frozenset(g) for g, _ in greedy_partition(P.words, OneHot())} == {frozenset(g) for g in GROUPS}


def test_play_perfect_scorer_wins_with_no_mistakes():
    r = _play(P, OneHot())
    assert r == {"won": True, "groups_correct": [1, 1, 1, 1], "mistakes": 0}
