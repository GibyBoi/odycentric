"""Fast checks of the cutout maths. No AI model needed: run with  python -m unittest"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from odycentric import engine


def photo_and_alpha(w=320, h=240, seed=1):
    rng = np.random.default_rng(seed)
    photo = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    yy, xx = np.mgrid[:h, :w]
    distance = np.hypot(xx - w / 2, yy - h / 2)
    alpha = np.clip((80 - distance) * 32 + 128, 0, 255).astype(np.uint8)  # solid disc, soft rim, clear outside
    return photo, alpha


class ComposeTest(unittest.TestCase):
    def test_solid_pixels_are_the_original_pixels(self):
        photo, alpha = photo_and_alpha()
        out = np.asarray(engine.compose(photo, alpha))
        solid = alpha == 255
        self.assertTrue(solid.any())
        np.testing.assert_array_equal(out[solid, :3], np.asarray(photo)[solid])

    def test_background_is_erased_not_hidden(self):
        photo, alpha = photo_and_alpha()
        out = np.asarray(engine.compose(photo, alpha))
        clear = alpha == 0
        self.assertTrue(clear.any())
        self.assertEqual(int(out[clear, :3].max()), 0)
        np.testing.assert_array_equal(out[..., 3], alpha)

    def test_cleanup_off_leaves_soft_edges_original(self):
        photo, alpha = photo_and_alpha()
        out = np.asarray(engine.compose(photo, alpha, cleanup=False))
        soft = (alpha > 0) & (alpha < 255)
        self.assertTrue(soft.any())
        np.testing.assert_array_equal(out[soft, :3], np.asarray(photo)[soft])

    def test_solid_pixels_survive_a_background_colour(self):
        photo, alpha = photo_and_alpha()
        out = np.asarray(engine.compose(photo, alpha, background=(255, 255, 255)))
        solid = alpha == 255
        np.testing.assert_array_equal(out[solid], np.asarray(photo)[solid])
        self.assertTrue((out[alpha == 0] == 255).all())


class AlphaTest(unittest.TestCase):
    def test_sharp_edges_are_fully_solid_or_fully_clear(self):
        prob = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        self.assertEqual(set(np.unique(engine.to_alpha(prob, (100, 80), sharp=True))), {0, 255})

    def test_near_certain_values_snap(self):
        prob = np.array([[0.005, 0.5, 0.995]], np.float32)
        self.assertEqual(engine.to_alpha(prob, (3, 1)).tolist(), [[0, 128, 255]])


class ClarityTest(unittest.TestCase):
    def disc(self, softness):
        yy, xx = np.mgrid[:1024, :1024]
        distance = np.hypot(xx - 512, yy - 512)
        return (1 / (1 + np.exp(np.clip((distance - 300) / softness, -50, 50)))).astype(np.float32)

    def test_crisp_edge_scores_higher_than_blurry_edge(self):
        crisp = engine.edge_clarity(self.disc(0.3), (2000, 2000))
        blurry = engine.edge_clarity(self.disc(8), (2000, 2000))
        self.assertEqual(crisp, 100)
        self.assertLess(blurry, 50)

    def test_nothing_found_scores_zero(self):
        self.assertEqual(engine.edge_clarity(np.zeros((64, 64), np.float32), (640, 640)), 0)


class FakeRemover(engine.Remover):
    """Treats brightness as subject probability, so tiling can be tested without a model."""

    def __init__(self, native=64):
        self.native = native
        self.calls = 0

    def _predict(self, image):
        self.calls += 1
        return np.asarray(image, np.float32)[..., 0] / 255


class TilingTest(unittest.TestCase):
    def photo(self):
        yy, xx = np.mgrid[:300, :400]
        disc = (np.hypot(xx - 200, yy - 150) < 90).astype(np.uint8) * 255
        return Image.fromarray(np.dstack([disc] * 3))

    def test_native_resolution_is_one_pass(self):
        remover = FakeRemover()
        prob, passes = remover.probability(self.photo(), 64)
        self.assertEqual((passes, remover.calls, prob.shape), (1, 1, (64, 64)))

    def test_higher_resolution_adds_passes_only_along_the_outline(self):
        remover = FakeRemover()
        prob, passes = remover.probability(self.photo(), 256)
        self.assertEqual(prob.shape, (192, 256))
        self.assertGreater(passes, 1)
        self.assertEqual(passes, remover.calls)
        # far from the disc the first pass was sure, so it is left alone
        self.assertEqual(float(prob[2, 2]), 0.0)
        self.assertGreater(float(prob[96, 128]), 0.99)

    def test_never_goes_past_the_photo_size(self):
        remover = FakeRemover()
        small = self.photo().resize((60, 45))
        _, passes = remover.probability(small, 4096)
        self.assertEqual(passes, 1)


class OutputPathTest(unittest.TestCase):
    def test_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            first = engine.output_path("C:/photos/me.jpg", folder)
            first.write_bytes(b"x")
            second = engine.output_path("C:/photos/me.jpg", folder)
            self.assertEqual((first.name, second.name), ("me-nobg.png", "me-nobg (2).png"))


if __name__ == "__main__":
    unittest.main()
