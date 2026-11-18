# src/puzzle.py
from dataclasses import dataclass

LEVELS = ["yellow", "green", "blue", "purple"]  # index 0..3

@dataclass(frozen=True)
class Group:
    words: tuple[str, str, str, str]   # uppercase, stripped
    name: str                          # category name, e.g. "TYPES OF FISH"
    level: int                         # 0=yellow ... 3=purple

@dataclass(frozen=True)
class Puzzle:
    id: int
    date: str                          # "YYYY-MM-DD"
    words: tuple[str, ...]             # all 16
    groups: tuple[Group, Group, Group, Group]
