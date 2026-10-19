# Ported from Building_load_ANKYRA/tests/test_analog.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before).
"""Within-day anchoring (2.0; one trust value since 2.1): analog-day shapes, the error-weighted trust and the invariance of the other blocks."""
import unittest
from datetime import date

import numpy as np
import torch

from ankyra_lite import anchor
from ankyra_lite.anchor import blocks
from ankyra_lite.anchor.analog import AnalogShapes, anchored_within_day, within_trust, dst_state, K0_WITHIN, W_CAP
from ankyra_lite.anchor.synthetic import synthetic_history, seasonal_naive

torch.set_num_threads(1)


class DstTests(unittest.TestCase):
    def test_rules(self):
        self.assertTrue(dst_state(date(2021, 7, 1), "EU")); self.assertFalse(dst_state(date(2021, 1, 15), "EU"))
        self.assertTrue(dst_state(date(2021, 3, 14), "US")); self.assertFalse(dst_state(date(2021, 3, 13), "US"))
        self.assertFalse(dst_state(date(2021, 7, 1), "none"))
        with self.assertRaises(ValueError):
            dst_state(date(2021, 7, 1), "Mars")


class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = synthetic_history(16800)                                    # 700 days, origin at hour 16800
        cls.o = len(cls.h.load_kw)
        cls.A = AnalogShapes(cls.h.load_kw, cls.h.temperature_c, cls.h.day_types, cls.h.start_timestamp, "EU", origin=cls.o)

    def test_shape_has_zero_daily_means_and_uses_the_fallback_where_needed(self):
        fb = blocks.within_day(seasonal_naive(self.h.load_kw[-1344:][None])[0])
        s = self.A.shape(self.o, fb)
        np.testing.assert_allclose(blocks.daily_means(s), 0.0, atol=1e-9)
        self.assertEqual(s.shape, (744,)); self.assertTrue(np.isfinite(s).all())
        self.assertGreater(self.A.days_built, 0)

    def test_shape_depends_only_on_data_before_its_origin(self):
        fb = np.zeros(744); s1 = self.A.shape(self.o - 744, fb)
        load2 = self.h.load_kw.copy(); load2[self.o - 744:] *= 3.0                    # change everything at or after that origin
        A2 = AnalogShapes(load2, self.h.temperature_c, self.h.day_types, self.h.start_timestamp, "EU", origin=self.o)
        np.testing.assert_allclose(A2.shape(self.o - 744, fb), s1, rtol=0, atol=1e-12)

    def test_whole_window_guard_returns_the_model_shape(self):
        fw = np.zeros(744)
        s, kept = self.A.shape_with_sanity(self.o, fw)                                # an ordinary record keeps its analog shape
        self.assertTrue(kept); self.assertGreater(np.abs(s).max(), 0.0)
        load = self.h.load_kw.copy(); load[-1344:] = 1e-4 * (1 + 0.1 * np.sin(np.arange(1344)))   # a context far below the scale floor
        tiny = AnalogShapes(load, self.h.temperature_c, self.h.day_types, self.h.start_timestamp, "EU", origin=self.o)
        s, kept = tiny.shape_with_sanity(self.o, fw)                                  # analog shape above three times the largest context value
        self.assertFalse(kept); np.testing.assert_array_equal(s, fw)


class TrustTests(unittest.TestCase):
    def test_bounds_and_shrinkage(self):
        rng = np.random.default_rng(0); f = rng.normal(size=744); s = f + rng.normal(size=744)
        self.assertEqual(within_trust([])[0].tolist(), [0.0] * 4)
        w, n = within_trust([(s, f, s)] * 3)                                            # the analog shape is exactly right
        self.assertEqual(n, 3); np.testing.assert_allclose(w, [min(1.0 * 3 / (3 + K0_WITHIN), W_CAP)] * 4)
        w, _ = within_trust([(s, f, f)] * 3)                                            # the model is exactly right
        np.testing.assert_allclose(w, [0.0] * 4)
        w, _ = within_trust([(s, f, f - (s - f))] * 3)                                  # the analog points the wrong way
        np.testing.assert_allclose(w, [0.0] * 4)
        self.assertTrue(all(0.0 <= x <= W_CAP for x in w))

    def test_single_trust_pools_the_four_lead_blocks(self):
        rng = np.random.default_rng(2); triples = []
        for _ in range(3):
            f = rng.normal(size=744); s = f + rng.normal(size=744); y = f + 0.3 * (s - f) * np.repeat(np.r_[np.ones(14), -np.ones(17)], 24) + rng.normal(size=744)
            triples.append((s, f, y))
        w1, n = within_trust(triples); w4, _ = within_trust(triples, single=False)
        self.assertEqual(n, 3); self.assertEqual(len(set(w1.tolist())), 1)              # 2.1: one value repeated
        num = sum(float((s - f) @ (y - f)) for s, f, y in triples); den = sum(float((s - f) @ (s - f)) for s, f, y in triples)
        self.assertAlmostEqual(w1[0], min(float(np.clip(num / den, 0, 1)) * 3 / (3 + K0_WITHIN), W_CAP), places=12)
        self.assertGreater(w4[0], 0.0); self.assertEqual(w4[3], 0.0)                     # 2.0.1: blocks 1-2 trusted, block 4 points the wrong way

    def test_anchoring_keeps_daily_means(self):
        rng = np.random.default_rng(1); f = blocks.within_day(rng.normal(size=744)); s = blocks.within_day(rng.normal(size=744))
        W = anchored_within_day(f, s, [0.5, 0.25, 0.0, 0.5])
        np.testing.assert_allclose(blocks.daily_means(W), 0.0, atol=1e-12)
        np.testing.assert_allclose(W[:7 * 24], 0.5 * f[:7 * 24] + 0.5 * s[:7 * 24], atol=1e-12); np.testing.assert_array_equal(W[14 * 24:21 * 24], f[14 * 24:21 * 24])


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = synthetic_history(16800)
        cls.f6 = anchor.forecast(cls.h, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive, dst_region="EU")
        cls.f1 = anchor.forecast(cls.h, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive, within_anchor=False)

    def test_level_daily_path_and_readouts_are_unchanged_by_the_anchoring(self):
        self.assertAlmostEqual(self.f6.level_kw, self.f1.level_kw, places=12)
        np.testing.assert_allclose(self.f6.daily_means_kw, self.f1.daily_means_kw, atol=1e-12)
        self.assertAlmostEqual(self.f6.energy_kwh, self.f1.energy_kwh, places=9)
        np.testing.assert_allclose(blocks.daily_means(self.f6.within_day_kw), 0.0, atol=1e-9)

    def test_fields(self):
        self.assertEqual(len(self.f6.within_trust), 4); self.assertTrue(all(0.0 <= w <= 0.5 for w in self.f6.within_trust))
        self.assertEqual(self.f6.within_pseudo_pairs, 3); self.assertEqual(self.f6.analog_shape_kw.shape, (744,))
        np.testing.assert_array_equal(self.f6.foundation_within_day_kw, self.f1.within_day_kw)

    def test_single_trust_false_reproduces_2_0_1(self):
        f201 = anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive, dst_region="EU", single_trust=False)
        self.assertEqual(len(set(self.f6.within_trust)), 1)
        np.testing.assert_allclose(f201.daily_means_kw, self.f6.daily_means_kw, atol=1e-12)          # the trust touches the within-day block only
        np.testing.assert_array_equal(f201.analog_shape_kw, self.f6.analog_shape_kw)
        W = anchored_within_day(f201.foundation_within_day_kw, f201.analog_shape_kw, f201.within_trust)
        np.testing.assert_allclose(f201.within_day_kw, W, atol=1e-12)

    def test_history_only_configuration(self):
        f0 = anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation=None)
        self.assertEqual(f0.trajectory_kw.shape, (744,)); self.assertTrue((f0.trajectory_kw >= 0).all())
        np.testing.assert_allclose(f0.daily_means_kw, f0.fixed_division_daily_means_kw)

    def test_off_state_is_untouched(self):
        h = synthetic_history(16800); load = h.load_kw.copy(); load[-168:] = 0.0
        hz = anchor.History(load_kw=load, temperature_c=h.temperature_c, day_types=h.day_types, start_timestamp=h.start_timestamp)
        f = anchor.forecast(hz, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive, dst_region="EU")
        self.assertTrue(f.off_state); self.assertEqual(f.within_trust, (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
