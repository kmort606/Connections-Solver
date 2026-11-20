"""Run embedding baselines + greedy search on val/test -> results/baseline_*.json

    python -m scripts.run_baseline --scorer st_single --split val
    python -m scripts.run_baseline --scorer all --split val test

Two kinds of numbers:
  * partition:  greedy partition vs. truth, no game rules (pure scorer+search quality)
  * play:       real rules, 4 mistakes. Guess best remaining candidate; on a miss,
                try the next-best untried one. Ignores "one away" (that's Emme's policy).

NOTE: `_play` is a stand-in so we have numbers for the TA meeting. Once Emme's
game.py / eval.py land, switch to those -- they are the source of truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from src.baseline import make_scorer
from src.data import load_split
from src.puzzle import LEVELS, Puzzle
from src.search import greedy_partition, score_all_subsets, _top_k

RESULTS = Path(__file__).resolve().parent.parent / "results"
KINDS = ["st_single", "contextual", "st_template", "glove"]


def _play(p: Puzzle, scorer, max_mistakes: int = 4) -> dict:
    truth = {frozenset(g.words): g.level for g in p.groups}
    table = score_all_subsets(p.words, scorer)  # context = full board throughout
    remaining, tried, mistakes, correct = set(p.words), set(), 0, [0, 0, 0, 0]
    while remaining and mistakes < max_mistakes:
        live = {g: s for g, s in table.items() if set(g) <= remaining and frozenset(g) not in tried}
        g = frozenset(_top_k(live, 1)[0][0])
        tried.add(g)
        if g in truth:
            correct[truth[g]] = 1
            remaining -= g
        else:
            mistakes += 1
    return {"won": not remaining, "groups_correct": correct, "mistakes": mistakes}


def evaluate(puzzles, scorer) -> tuple[dict, list]:
    per = []
    for p in tqdm(puzzles, desc=scorer.name, leave=False):
        truth = {frozenset(g.words): g.level for g in p.groups}
        part = [0, 0, 0, 0]
        for g, _ in greedy_partition(p.words, scorer):
            if frozenset(g) in truth:
                part[truth[frozenset(g)]] = 1
        per.append({"id": p.id, **_play(p, scorer), "partition_correct": part})

    n = len(per)
    summary = {
        "solve_rate": sum(r["won"] for r in per) / n,
        "group_acc_by_level": [sum(r["groups_correct"][i] for r in per) / n for i in range(4)],
        "avg_mistakes": sum(r["mistakes"] for r in per) / n,
        "partition_solve_rate": sum(all(r["partition_correct"]) for r in per) / n,
        "partition_acc_by_level": [sum(r["partition_correct"][i] for r in per) / n for i in range(4)],
    }
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", nargs="+", default=["st_single", "contextual"], help=f"{KINDS} or 'all'")
    ap.add_argument("--split", nargs="+", default=["val"])
    ap.add_argument("--glove_path", default=None)
    ap.add_argument("--n_orders", type=int, default=1)
    args = ap.parse_args()
    kinds = KINDS if args.scorer == ["all"] else args.scorer
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for kind in kinds:
        scorer = make_scorer(kind, glove_path=args.glove_path, n_orders=args.n_orders)
        for split in args.split:
            t0 = time.time()
            summary, per = evaluate(load_split(split), scorer)
            config = {"scorer": kind, "search": "greedy", "policy": "next_best_no_one_away",
                      "n_orders": args.n_orders if kind == "contextual" else None}
            if hasattr(scorer, "oov_rate"):
                config["oov_rate"] = round(scorer.oov_rate, 4)
                config["oov_words"] = scorer.oov_words
            run = f"baseline_{kind}_greedy"
            out = {"run": run, "config": config, "split": split, "summary": summary, "per_puzzle": per}
            path = RESULTS / f"{run}_{split}.json"
            path.write_text(json.dumps(out, indent=1))
            rows.append((kind, split, summary))
            print(f"[{kind}/{split}] {time.time() - t0:.0f}s -> {path.name}")

    # Per-color table for the TA
    hdr = f"{'scorer':<12}{'split':<6}{'solve':>7}{'mist':>6} " + "".join(f"{l:>8}" for l in LEVELS) + "   | partition solve"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for kind, split, s in rows:
        lv = "".join(f"{a:>8.2f}" for a in s["group_acc_by_level"])
        print(f"{kind:<12}{split:<6}{s['solve_rate']:>7.2f}{s['avg_mistakes']:>6.2f} {lv}   | {s['partition_solve_rate']:.2f}")


if __name__ == "__main__":
    main()
