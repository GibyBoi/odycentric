import tempfile
import unittest
from pathlib import Path

from odycentric import history


def run(seconds, clarity, rating=None, power="plugged in", model="people", resolution=1024, threads=0):
    return {"ts": 0, "photo": "p.jpg", "output": "o.png", "megapixels": 8.5, "model": model,
            "resolution": resolution, "threads": threads, "priority": "high", "edges": "soft", "cleanup": 1,
            "max_side": None, "power": power, "seconds": seconds, "ai_seconds": seconds, "passes": 1,
            "clarity": clarity, "rating": rating}


class ScoreTest(unittest.TestCase):
    def test_matches_the_documented_formula(self):
        # balanced: 60% quality, 40% speed; speed = 100 * 10 / 14.1
        self.assertEqual(history.score(run(14.1, 89), "balanced"), 82)
        self.assertEqual(history.score(run(94.9, 100), "balanced"), 64)
        self.assertEqual(history.score(run(5, 50), "speed"), round(0.3 * 50 + 0.7 * 100))

    def test_rating_replaces_clarity(self):
        self.assertEqual(history.quality(run(10, 30, rating=5)), 100)
        self.assertEqual(history.quality(run(10, 30)), 30)


class SuggestTest(unittest.TestCase):
    def test_best_average_wins_and_uses_matching_power_once_there_are_enough(self):
        runs = [run(14, 90) for _ in range(3)] + [run(95, 100, resolution=2048) for _ in range(3)]
        runs += [run(5, 95, power="battery", model="fast")]
        best, situation = history.suggest(runs, "balanced", "plugged in")
        self.assertEqual(situation, "plugged in")
        self.assertEqual(best[0].settings["resolution"], 1024)
        self.assertTrue(all(s.settings["model"] != "fast" for s in best))

        best, situation = history.suggest(runs, "quality", "battery")  # only 1 battery run: use all
        self.assertEqual(situation, "all")

    def test_one_lucky_photo_does_not_beat_a_steady_record(self):
        runs = [run(10, 85) for _ in range(6)] + [run(10, 90, resolution=2048)]
        best, _ = history.suggest(runs, "quality", "plugged in")
        self.assertEqual(best[0].settings["resolution"], 1024)
        self.assertEqual(round(best[1].score), 92)  # shown score is the expected score, without the doubt

    def test_one_clearly_better_photo_does_win(self):
        runs = [run(10, 70) for _ in range(6)] + [run(10, 95, resolution=2048)]
        best, _ = history.suggest(runs, "quality", "plugged in")
        self.assertEqual(best[0].settings["resolution"], 2048)

    def test_threads_are_judged_on_speed_alone(self):
        # 4 threads happened to get the easy photo but was slower: it must not win
        runs = [run(14, 63), run(14, 89), run(14, 70), run(19, 95, threads=4)]
        best, _ = history.suggest(runs, "balanced", "plugged in")
        self.assertEqual(best[0].settings["threads"], 0)
        self.assertEqual(best[0].quality_runs, 4)


class StorageTest(unittest.TestCase):
    def test_add_rate_and_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            store = history.History(Path(folder) / "h.sqlite")
            run_id = store.add(run(12, 80))
            store.rate(run_id, 4)
            self.assertEqual(store.get(run_id)["rating"], 4)
            store.rate(run_id, 0)
            self.assertIsNone(store.get(run_id)["rating"])
            store.clear()
            self.assertEqual(store.all(), [])
            store.db.close()


if __name__ == "__main__":
    unittest.main()
