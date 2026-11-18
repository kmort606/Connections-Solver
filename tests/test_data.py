"""Step 2 checks on a fake CSV (no download). Run from repo root:  python -m pytest -q"""
import io

from src.data import build_puzzles, clean_word, puzzle_from_dict, puzzle_to_dict, read_raw, split

HEADER = "Game ID,Puzzle Date,Word,Group Name,Group Level,Starting Row,Starting Column"
GROUPS = [
    ("FISH", 0, ["BASS", "PIKE", "TROUT", "SOLE"]),
    ("INSTRUMENTS", 1, ["DRUM", "HARP", "OBOE", "TUBA"]),
    ("NOTHING-ISH", 2, ["NULL", "NAN", "NONE", "NA"]),  # words pandas turns into NaN by default
    ("NUMBERS", 3, ["7", "11", "42", "100"]),
]


def game_rows(game_id, date):
    rows, i = [], 0
    for name, lvl, ws in GROUPS:
        for w in ws:
            rows.append(f"{game_id},{date},{w},{name},{lvl},{i // 4 + 1},{i % 4 + 1}")
            i += 1
    return rows


def load(rows):
    return build_puzzles(read_raw(io.StringIO("\n".join([HEADER] + rows))))


def test_null_words_and_numbers_survive():
    (p,) = load(game_rows(1, "2024-01-02"))
    assert {"NULL", "NAN", "NONE", "NA", "7", "100"} <= set(p.words)
    assert len(p.words) == 16 and [g.level for g in p.groups] == [0, 1, 2, 3]


def test_clean_word():
    assert clean_word("  ice  cream ") == "ICE CREAM"
    assert clean_word("7.0") == "7"
    assert clean_word(float("nan")) == ""


def test_invalid_puzzles_dropped():
    good = game_rows(1, "2024-01-02")
    short = game_rows(2, "2024-01-03")[:-1]                  # only 15 words
    dup = game_rows(3, "2024-01-04")
    dup[1] = dup[1].replace("PIKE", "BASS")                  # duplicate word
    assert [p.id for p in load(good + short + dup)] == [1]


def test_split_is_chronological():
    rows = []
    for i in range(40):
        rows += game_rows(i, f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}")
    tr, va, te = split(load(rows), n_test=5, n_val=3)
    assert (len(tr), len(va), len(te)) == (32, 3, 5)
    assert max(p.date for p in tr) < min(p.date for p in va)
    assert max(p.date for p in va) < min(p.date for p in te)


def test_json_roundtrip():
    (p,) = load(game_rows(1, "2024-01-02"))
    assert puzzle_from_dict(puzzle_to_dict(p)) == p
