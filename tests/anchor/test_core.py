# Ported from Building_load_ANKYRA/tests/test_core.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before); TimesfmAdapterTests and the imports only it used (sys, types, mock, timesfm_adapter)
# are left out because ankyra_lite.anchor does not ship timesfm_adapter.py.
"""Behaviour of the handover, the off-state rule and the complete forecast on an artificial building."""
import unittest

import numpy as np
import torch

from ankyra_lite import anchor
from ankyra_lite.anchor import blocks
from ankyra_lite.anchor.core import lead_week_weights, lead_week_transition, pseudo_origin_contexts
from ankyra_lite.anchor.history import estimate_from_history
from ankyra_lite.anchor.synthetic import synthetic_history, seasonal_naive

torch.set_num_threads(1)


class HandoverTests(unittest.TestCase):
    def test_lead_week_weights_are_shrunk_towards_one_half(self):
        y = np.linspace(10, 12, 31)
        np.testing.assert_allclose(lead_week_weights([(y, y + 1, y + 1)] * 3), [0.5] * 4)        # no disagreement, no evidence
        np.testing.assert_allclose(lead_week_weights([(y, y + 1, y)] * 6), [(6 + 1) / 8] * 4)    # model right six times
        np.testing.assert_allclose(lead_week_weights([(y, y, y + 1)] * 6), [1 / 8] * 4)          # history right six times

    def test_lead_week_transition_endpoints(self):
        h, f = np.arange(31.0), np.arange(31.0) * 2
        np.testing.assert_array_equal(lead_week_transition(h, f, [0, 0, 0, 0]), h)
        np.testing.assert_array_equal(lead_week_transition(h, f, [1, 1, 1, 1]), f)

    def test_pseudo_origin_contexts(self):
        h = synthetic_history(16000)
        ctx = pseudo_origin_contexts(h.load_kw)
        self.assertEqual(sorted(ctx), list(range(7)))
        np.testing.assert_array_equal(ctx[0], h.load_kw[-1344:])


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = synthetic_history(16000)
        cls.f = anchor.forecast(cls.h, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive)

    def test_contract(self):
        f = self.f
        self.assertEqual(f.trajectory_kw.shape, (744,))
        self.assertTrue(np.isfinite(f.trajectory_kw).all() and (f.trajectory_kw >= 0).all())
        self.assertFalse(f.off_state)
        self.assertEqual(f.pseudo_pairs, 6)
        self.assertAlmostEqual(sum(f.level_weights.values()), 1.0, places=12)
        self.assertIn("fm", f.level_weights)
        self.assertTrue(all(0 <= a <= 1 for a in f.lead_week_weights))
        self.assertAlmostEqual(f.level_kw, f.daily_means_kw.mean(), places=12)
        self.assertAlmostEqual(f.energy_kwh, 744 * f.level_kw, places=9)

    def test_within_day_block_is_the_anchored_foundation_shape(self):
        T0 = seasonal_naive(self.h.load_kw[-1344:][None])[0]
        np.testing.assert_array_equal(self.f.foundation_within_day_kw, blocks.within_day(T0))
        self.assertTrue((np.repeat(self.f.daily_means_kw, 24) + self.f.within_day_kw >= 0).all())   # no projection needed here
        np.testing.assert_allclose(blocks.within_day(self.f.trajectory_kw), self.f.within_day_kw, atol=1e-10)
        f1 = anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive, within_anchor=False)
        np.testing.assert_allclose(blocks.within_day(f1.trajectory_kw), blocks.within_day(T0), atol=1e-10)   # 1.x mode

    def test_callable_and_mapping_foundation_agree(self):
        T = {k: seasonal_naive(v[None])[0] for k, v in pseudo_origin_contexts(self.h.load_kw).items()}
        g = anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation=T)
        np.testing.assert_array_equal(g.trajectory_kw, self.f.trajectory_kw)

    def test_fixed_division_is_the_reference_estimator(self):
        e = estimate_from_history(self.h, group="Office", temp_sigma_std=0.25)
        np.testing.assert_allclose(self.f.fixed_division_daily_means_kw, e.level_kw + e.daily_path_kw, atol=1e-12)

    def test_off_state_returns_the_foundation_forecast(self):
        h = synthetic_history(16000, off_last_hours=200)
        T0 = np.zeros(744)
        f = anchor.forecast(h, group="Office", temp_sigma_std=0.25, foundation={0: T0})
        self.assertTrue(f.off_state)
        np.testing.assert_array_equal(f.trajectory_kw, T0)

    def test_origin_forecast_is_required(self):
        with self.assertRaises(ValueError):
            anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation={1: np.zeros(744)})

    def test_input_mistakes_give_errors_that_name_the_cause(self):
        kw = dict(temp_sigma_std=0.25, foundation=seasonal_naive)
        with self.assertRaisesRegex(anchor.InputError, "group must be one of"):          # a category outside the five
            anchor.forecast(self.h, group="office", **kw)
        with self.assertRaisesRegex(anchor.InputError, "1344"):                          # a record shorter than the context
            anchor.forecast(synthetic_history(1000), group="Office", **kw)
        gap = synthetic_history(16000); gap.load_kw[-10] = np.nan
        with self.assertRaisesRegex(anchor.InputError, "completely observed"):           # a gap in the context, callable model
            anchor.forecast(gap, group="Office", **kw)
        with self.assertRaisesRegex(ValueError, "dst_region"):
            anchor.forecast(self.h, group="Office", dst_region="Mars", **kw)
        with self.assertRaises(TypeError):                                               # neither a mapping, a callable nor None
            anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation="timesfm")
        T = {k: seasonal_naive(v[None])[0] for k, v in pseudo_origin_contexts(self.h.load_kw).items()}
        with self.assertRaisesRegex(ValueError, "k=2 must have shape"):                  # a pseudo-origin forecast one hour short
            anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation={**T, 2: T[2][:743]})
        with self.assertRaisesRegex(ValueError, "finite"):                               # a model forecast with missing values
            anchor.forecast(self.h, group="Office", temp_sigma_std=0.25, foundation={**T, 0: np.full(744, np.nan)})


if __name__ == "__main__":
    unittest.main()
