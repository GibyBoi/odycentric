"""Every finished photo gets a score. The history of scores drives the suggestions.

Score (0 to 100) blends quality and speed, weighted by what you optimize for:
  quality  your star rating if you gave one (1 star = 20 ... 5 stars = 100),
           otherwise the automatic edge clarity (see engine.edge_clarity)
  speed    100 for FAST_SECONDS or less per photo, halving each time the time doubles

Scores are worked out from the stored measurements each time, so changing what
you optimize for re-ranks the whole history.
"""

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from odycentric.engine import ROOT

DB_PATH = ROOT / "data" / "history.sqlite"
GOALS = {"quality": ("Quality", 0.8), "balanced": ("Balanced", 0.6), "speed": ("Speed", 0.3)}
FAST_SECONDS = 10
# The settings that change how a cutout looks or how long it takes. Background
# colour and output folder change neither, so they don't split the history.
KEY = ("model", "resolution", "threads", "priority", "edges", "cleanup")
# The ones that change the cutout itself. Threads and priority only change speed.
OUTPUT_KEY = ("model", "resolution", "edges", "cleanup")
MIN_SAME_SITUATION = 3
# Score points a setting's average is discounted by when it has one photo behind
# it; the discount shrinks with the square root of the number of photos.
UNCERTAINTY = 8

_COLUMNS = (
    ("ts", "REAL"), ("photo", "TEXT"), ("output", "TEXT"), ("megapixels", "REAL"),
    ("model", "TEXT"), ("resolution", "INTEGER"), ("threads", "INTEGER"), ("priority", "TEXT"),
    ("edges", "TEXT"), ("cleanup", "INTEGER"), ("max_side", "INTEGER"), ("power", "TEXT"),
    ("seconds", "REAL"), ("ai_seconds", "REAL"), ("passes", "INTEGER"), ("clarity", "INTEGER"),
    ("rating", "INTEGER"),
)


def quality(run):
    return run["rating"] * 20 if run["rating"] else run["clarity"]


def speed(run):
    return 100 * min(1.0, FAST_SECONDS / max(run["seconds"], 0.01))


def score(run, goal):
    weight = GOALS[goal][1]
    return round(weight * quality(run) + (1 - weight) * speed(run))


class History:
    def __init__(self, path=DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        columns = ", ".join(f"{name} {kind}" for name, kind in _COLUMNS)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, {columns})")
        self.db.commit()

    def add(self, run):
        names = [name for name, _ in _COLUMNS]
        cursor = self.db.execute(
            f"INSERT INTO runs ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
            [run.get(name) for name in names])
        self.db.commit()
        return cursor.lastrowid

    def rate(self, run_id, stars):
        self.db.execute("UPDATE runs SET rating = ? WHERE id = ?", (stars or None, run_id))
        self.db.commit()

    def get(self, run_id):
        row = self.db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def all(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM runs ORDER BY ts DESC")]

    def clear(self):
        self.db.execute("DELETE FROM runs")
        self.db.commit()


@dataclass
class Suggestion:
    settings: dict
    score: float  # expected score for one photo
    runs: int  # photos done with exactly these settings (speed comes from these)
    seconds: float
    quality: float
    quality_runs: int  # photos with the same model, resolution and edges (quality comes from these)


def suggest(runs, goal, power, limit=3):
    """The settings with the best expected score, best first.

    Threads and priority cannot change a cutout, only how long it takes. So
    quality is averaged over every photo made with the same OUTPUT_KEY settings,
    whatever the threads and priority were, and speed over the photos made with
    exactly these settings. That keeps one easy photo from making a thread count
    look better than it is.

    Uses only photos done in the same power situation (plugged in or on battery)
    once there are enough of them, since the same settings run at very
    different speeds in each. Ranks by expected score minus UNCERTAINTY points
    divided by the square root of the photos behind each half, so a setting
    tried once needs a clear lead to beat a steady record.
    Returns (suggestions, situation used).
    """
    same = [run for run in runs if run["power"] == power]
    pool, situation = (same, power) if len(same) >= MIN_SAME_SITUATION else (runs, "all")
    weight = GOALS[goal][1]
    qualities, groups = defaultdict(list), defaultdict(list)
    for run in pool:
        qualities[tuple(run[name] for name in OUTPUT_KEY)].append(quality(run))
        groups[tuple(run[name] for name in KEY)].append(run)
    ranked = []
    for key, group in groups.items():
        settings = dict(zip(KEY, key))
        pooled = qualities[tuple(settings[name] for name in OUTPUT_KEY)]
        expected = weight * mean(pooled) + (1 - weight) * mean(speed(run) for run in group)
        doubt = UNCERTAINTY * (weight / len(pooled) ** 0.5 + (1 - weight) / len(group) ** 0.5)
        ranked.append((expected - doubt, Suggestion(
            settings, expected, len(group), mean(run["seconds"] for run in group), mean(pooled), len(pooled))))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [suggestion for _, suggestion in ranked[:limit]], situation
