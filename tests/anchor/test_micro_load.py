# Ported from Building_load_ANKYRA/tests/test_micro_load.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before).
"""The micro-load rule (2.0.1): a context that stays within 1e-3 kW of zero is handed to the foundation model."""
import unittest

import numpy as np
import torch

from ankyra_lite import anchor
from ankyra_lite.anchor.core import MICRO_KW, ZERO_KW, CONTEXT
from ankyra_lite.anchor.history import History, InputError, _eo
from ankyra_lite.anchor.synthetic import synthetic_history, seasonal_naive

torch.set_num_threads(1)


def with_context(values, off_last_hours=0):
    """The artificial building with its last 1,344 hours replaced by ``values`` (a scalar or an array)."""
    h = synthetic_history(16000)
    load = np.array(h.load_kw, dtype=float)
    load[-CONTEXT:] = values
    if off_last_hours:
        load[-off_last_hours:] = 0.0
    return History(load, h.temperature_c, h.day_types, h.start_timestamp)


def run(history, **kw):
    kw.setdefault("foundation", {0: np.full(744, 2e-4)})
    kw.setdefault("group", "Office")
    kw.setdefault("temp_sigma_std", 0.25)
    return anchor.forecast(history, **kw)


class MicroLoadRuleTests(unittest.TestCase):
    def test_threshold_is_the_estimators_scale_floor_and_above_the_off_state_threshold(self):
        floor = float(_eo.origin_scale(torch.zeros(1, CONTEXT, dtype=torch.float64))[1][0])   # the scale of an all-zero context
        self.assertEqual(MICRO_KW, floor)
        self.assertGreater(MICRO_KW, ZERO_KW)

    def test_fires_on_a_context_at_or_below_the_floor_and_returns_the_foundation_forecast(self):
        T0 = np.full(744, 2e-4)
        for level in (2e-4, MICRO_KW, -2e-4):                           # below the floor, exactly at it, and slightly negative
            f = run(with_context(level), foundation={0: T0})
            self.assertTrue(f.micro_load)
            self.assertEqual(f.off_state, level < 0)                    # the off-state test is a signed comparison, as in 2.0.0
            np.testing.assert_array_equal(f.trajectory_kw, T0)
            self.assertEqual(f.level_kw, float(T0.mean()))
            self.assertEqual(f.within_trust, (0.0, 0.0, 0.0, 0.0))

    def test_one_hour_above_the_floor_is_enough_not_to_fire(self):
        values = np.full(CONTEXT, 2e-4)
        values[300] = 2 * MICRO_KW
        f = run(with_context(values), foundation=seasonal_naive)
        self.assertFalse(f.micro_load)
        self.assertFalse(f.off_state)

    def test_the_threshold_is_on_the_magnitude(self):
        values = np.full(CONTEXT, -50.0)                                # an exporting record: its maximum is below the threshold, its magnitude is not
        values[-168:] = 5e-4
        self.assertFalse(run(with_context(values), foundation=seasonal_naive).micro_load)

    def test_only_the_context_is_looked_at(self):
        a = with_context(2e-4)                                          # ordinary load before the context
        b = History(np.concatenate([np.full(len(a.load_kw) - CONTEXT, 500.0), a.load_kw[-CONTEXT:]]), a.temperature_c, a.day_types, a.start_timestamp)
        fa, fb = run(a), run(b)
        self.assertTrue(fa.micro_load and fb.micro_load)
        np.testing.assert_array_equal(fa.trajectory_kw, fb.trajectory_kw)

    def test_a_callable_foundation_and_the_1x_switch(self):
        h = with_context(2e-4)
        f = run(h, foundation=seasonal_naive)
        np.testing.assert_array_equal(f.trajectory_kw, seasonal_naive(h.load_kw[None, -CONTEXT:])[0])   # the model's forecast at the origin
        g = run(h, foundation=seasonal_naive, within_anchor=False)     # the rule does not depend on the anchoring switch
        self.assertTrue(g.micro_load)
        np.testing.assert_array_equal(g.trajectory_kw, f.trajectory_kw)

    def test_switching_the_rule_off_runs_the_2_0_0_path(self):
        h = with_context(2e-4)
        f_on = run(h, foundation=seasonal_naive)
        f_off = run(h, foundation=seasonal_naive, micro_load_rule=False)
        self.assertTrue(f_on.micro_load)
        self.assertFalse(f_off.micro_load)
        self.assertEqual(f_off.pseudo_pairs, 6)                         # the estimator ran, as in 2.0.0
        ordinary = synthetic_history(16000)                             # an ordinary unit is untouched by the rule
        g_on = run(ordinary, foundation=seasonal_naive)
        g_off = run(ordinary, foundation=seasonal_naive, micro_load_rule=False)
        self.assertFalse(g_on.micro_load)
        np.testing.assert_array_equal(g_on.trajectory_kw, g_off.trajectory_kw)

    def test_off_state_and_micro_load_flags(self):
        T0 = np.zeros(744)
        both = run(with_context(2e-4, off_last_hours=200), foundation={0: T0})
        self.assertTrue(both.off_state and both.micro_load)
        off_only = run(synthetic_history(16000, off_last_hours=200), foundation={0: T0})
        self.assertTrue(off_only.off_state)
        self.assertFalse(off_only.micro_load)                           # the context holds ordinary load before the last week

    def test_incomplete_context_is_not_handed_over(self):
        values = np.full(CONTEXT, 2e-4)
        values[10] = np.nan
        with self.assertRaises(InputError):                             # the estimator's input check rejects the gap, as in 2.0.0
            run(with_context(values))
        h = with_context(2e-4)                                          # finite values, but one context hour flagged as not observed
        observed = np.isfinite(h.load_kw); observed[-20] = False
        with self.assertRaises(InputError):
            run(History(h.load_kw, h.temperature_c, h.day_types, h.start_timestamp, observed=observed))

    def test_invalid_inputs_are_rejected_on_a_micro_load_context(self):
        h = with_context(2e-4)
        with self.assertRaises(InputError):
            run(h, group="Nonsense")
        with self.assertRaises(InputError):
            run(h, temp_sigma_std=-1.0)
        with self.assertRaises(InputError):                             # a mask of the wrong length
            run(History(h.load_kw, h.temperature_c, h.day_types, h.start_timestamp, observed=np.ones(100, bool)))
        with self.assertRaises(InputError):                             # day types that stop short of the horizon
            run(History(h.load_kw, h.temperature_c, h.day_types[:-10], h.start_timestamp))


if __name__ == "__main__":
    unittest.main()
