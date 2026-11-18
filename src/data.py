"""Load, clean, and split the NYT Connections dataset.

Usage (from repo root):
    python -m src.data            # download, clean, split, save to data/processed/
    python -m src.data --show 5   # also pretty-print 5 random puzzles to eyeball

Other modules should NOT re-download; use:
    from src.data import load_split
    train = load_split("train")
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.puzzle import LEVELS, Group, Puzzle

log = logging.getLogger(__name__)

HF_REPO = "eric27n/NYT-Connections"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

N_TEST = 100
N_VAL = 50

# Column names on the HF dataset card. Normalized to snake_case on load.
COLS = {
    "Game ID": "game_id",
    "Puzzle Date": "date",
    "Word": "word",
    "Group Name": "group_name",
    "Group Level": "level",
    "Starting Row": "row",
    "Starting Column": "col",
}


# --------------------------------------------------------------------------- #
# 1. Raw download
# --------------------------------------------------------------------------- #
def _download_raw_csv() -> Path:
    """Fetch the dataset's CSV from the HF Hub into data/raw/ (cached)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    csvs = [f for f in list_repo_files(HF_REPO, repo_type="dataset") if f.endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"No CSV found in {HF_REPO}")
    path = hf_hub_download(HF_REPO, csvs[0], repo_type="dataset", local_dir=RAW_DIR)
    return Path(path)


def read_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV with EVERYTHING as strings and NaN-parsing turned off.

    Why: pandas' defaults turn the words "NULL", "NAN", "NA", "NONE" into NaN and
    "7" into 7.0 -- both happen in real Connections puzzles. Reading as str with
    keep_default_na=False keeps the word exactly as written.
    """
    if csv_path is None:
        cached = sorted(RAW_DIR.glob("*.csv"))
        csv_path = cached[0] if cached else _download_raw_csv()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    missing = set(COLS) - set(df.columns)
    if missing:
        raise KeyError(f"Expected columns {missing} not in CSV; found {list(df.columns)}")
    return df.rename(columns=COLS)[list(COLS.values())]


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
_FLOAT_INT = re.compile(r"^(-?\d+)\.0+$")  # "7.0" -> "7" (in case a float sneaks in)


def clean_word(w) -> str:
    """Uppercase, strip, collapse inner whitespace, undo float-ification."""
    if w is None or (isinstance(w, float) and pd.isna(w)):
        return ""
    w = str(w).strip().upper()
    w = re.sub(r"\s+", " ", w)
    w = _FLOAT_INT.sub(r"\1", w)
    # A literal NaN that made it through some other path; can't recover it.
    return w


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word"] = df["word"].map(clean_word)
    df["group_name"] = df["group_name"].astype(str).str.strip().str.upper()
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["col"] = pd.to_numeric(df["col"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


# --------------------------------------------------------------------------- #
# 3. Group rows -> Puzzle, with validation
# --------------------------------------------------------------------------- #
def _build_puzzle(gid: int, g: pd.DataFrame) -> tuple[Puzzle | None, str]:
    """Return (puzzle, "") if valid, else (None, reason)."""
    if g["date"].isna().any() or g["date"].nunique() != 1:
        return None, "bad_date"
    if len(g) != 16:
        return None, f"n_words={len(g)}"
    if (g["word"] == "").any():
        return None, "empty_word"
    if g["word"].nunique() != 16:
        return None, "duplicate_word"
    if g["level"].isna().any() or sorted(g["level"].unique()) != [0, 1, 2, 3]:
        return None, "bad_levels"

    groups = []
    for lvl in range(4):
        gg = g[g["level"] == lvl]
        if len(gg) != 4:
            return None, "group_size"
        if gg["group_name"].nunique() != 1:
            return None, "group_name_mismatch"
        groups.append(Group(words=tuple(sorted(gg["word"])), name=gg["group_name"].iloc[0], level=lvl))

    # Keep the original NYT board order (left->right, top->bottom) for `words`.
    board = g.sort_values(["row", "col"], na_position="last")["word"]
    return Puzzle(id=int(gid), date=g["date"].iloc[0], words=tuple(board), groups=tuple(groups)), ""


def build_puzzles(df: pd.DataFrame) -> list[Puzzle]:
    df = clean_df(df)
    df = df[df["game_id"].notna()]
    puzzles, dropped = [], {}
    for gid, g in df.groupby("game_id", sort=True):
        p, reason = _build_puzzle(gid, g)
        if p is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            log.debug("drop game %s: %s", gid, reason)
        else:
            puzzles.append(p)
    n_drop = sum(dropped.values())
    log.info("Kept %d puzzles, dropped %d %s", len(puzzles), n_drop, dropped or "")
    print(f"[data] kept {len(puzzles)} puzzles, dropped {n_drop} {dropped or ''}")
    return puzzles


def load_puzzles(csv_path: str | Path | None = None) -> list[Puzzle]:
    """Download (if needed), clean, and return all valid puzzles."""
    return build_puzzles(read_raw(csv_path))


# --------------------------------------------------------------------------- #
# 4. Split + save / load
# --------------------------------------------------------------------------- #
def split(puzzles: list[Puzzle], n_test: int = N_TEST, n_val: int = N_VAL):
    """Chronological split: test = newest n_test, val = the n_val before, train = rest."""
    ps = sorted(puzzles, key=lambda p: (p.date, p.id))
    if len(ps) <= n_test + n_val:
        raise ValueError(f"Only {len(ps)} puzzles; need > {n_test + n_val}")
    test = ps[-n_test:]
    val = ps[-(n_test + n_val):-n_test]
    train = ps[:-(n_test + n_val)]
    return train, val, test


def puzzle_to_dict(p: Puzzle) -> dict:
    return asdict(p)


def puzzle_from_dict(d: dict) -> Puzzle:
    groups = tuple(Group(words=tuple(g["words"]), name=g["name"], level=int(g["level"])) for g in d["groups"])
    return Puzzle(id=int(d["id"]), date=d["date"], words=tuple(d["words"]), groups=groups)


def save_splits(train, val, test, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ps in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump([puzzle_to_dict(p) for p in ps], f, indent=1)


def load_split(name: str, data_dir: Path = PROCESSED_DIR) -> list[Puzzle]:
    """Load a processed split ("train" | "val" | "test") without re-downloading."""
    with open(Path(data_dir) / f"{name}.json") as f:
        return [puzzle_from_dict(d) for d in json.load(f)]


# --------------------------------------------------------------------------- #
# 5. Pretty print for eyeballing against the NYT archive
# --------------------------------------------------------------------------- #
def pretty(p: Puzzle) -> str:
    lines = [f"Puzzle #{p.id}  ({p.date})", "Board:"]
    for r in range(4):
        lines.append("   " + " | ".join(f"{w:<14}" for w in p.words[4 * r: 4 * r + 4]))
    for g in p.groups:
        lines.append(f"  {LEVELS[g.level]:<6} {g.name}: {', '.join(g.words)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local CSV path (skips download)")
    ap.add_argument("--show", type=int, default=1, help="pretty-print N random puzzles")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    puzzles = load_puzzles(args.csv)
    train, val, test = split(puzzles)
    save_splits(train, val, test)
    for name, ps in [("train", train), ("val", val), ("test", test)]:
        print(f"[data] {name:<5} {len(ps):>4} puzzles  {ps[0].date} -> {ps[-1].date}")
    rng = random.Random(0)
    for p in rng.sample(puzzles, min(args.show, len(puzzles))):
        print("\n" + pretty(p))
    print(f"\n[data] saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
