# Ported from Building_load_ANKYRA/tests/test_no_future_information.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before).
"""No information from after the origin reaches ANKYRA.

The interface takes load and temperature only up to the origin, and the calendar through the horizon, so the forecast
cannot see future load or weather by construction.  These tests check the parts where information could still leak:

* the contexts handed to the foundation model;
* the pseudo-origin forecasts inside the record, which must use only data before their own pseudo-origin;
* the interval, whose function receives the full arrays and an origin index.
"""
import unittest

import numpy as np
import torch

from ankyra_lite import anchor
from ankyra_lite.anchor import readouts
from ankyra_lite.anchor.core import _level_parts, pseudo_origin_contexts
from ankyra_lite.anchor.history import History
from ankyra_lite.anchor.synthetic import synthetic_history, seasonal_naive

torch.set_num_threads(1)
O, STEP, CONTEXT = 12000, 744, 1344


def record():
    """An artificial record that continues 744 hours past the origin O (calendar a further 744 hours)."""
    full = synthetic_history(O + STEP)
    return (np.array(full.load_kw, dtype=np.float64), np.array(full.temperature_c, dtype=np.float64),
            np.array(full.day_types), full.start_timestamp)


def history_at(load, temp, types, start, origin=O):
    return History(load_kw=load[:origin], temperature_c=temp[:origin], day_types=types[:origin + STEP], start_timestamp=start)


class NoFutureInformationTests(unittest.TestCase):
    def test_foundation_model_sees_only_contexts_that_end_at_or_before_their_origin(self):
        load, temp, types, start = record()
        seen = []

        def spy(contexts):
            seen.append(np.array(contexts))
            return seasonal_naive(contexts)

        anchor.forecast(history_at(load, temp, types, start), group="Office", temp_sigma_std=0.25, foundation=spy)
        ctx = seen[0]
        self.assertEqual(len(ctx), 7)                                                  # origin and six pseudo-origins
        for k, c in enumerate(ctx):
            ok = O - k * STEP
            np.testing.assert_array_equal(c, load[ok - CONTEXT:ok])                    # ends at its own origin

    def test_forecast_is_unchanged_when_everything_after_the_origin_changes(self):
        load, temp, types, start = record()
        rng = np.random.default_rng(0)
        load2, temp2 = load.copy(), temp.copy()
        load2[O:] += rng.normal(0, 50, len(load) - O); temp2[O:] += rng.normal(0, 20, len(temp) - O)
        f1 = anchor.forecast(history_at(load, temp, types, start), group="Office", temp_sigma_std=0.25, foundation=seasonal_naive)
        f2 = anchor.forecast(history_at(load2, temp2, types, start), group="Office", temp_sigma_std=0.25, foundation=seasonal_naive)
        np.testing.assert_array_equal(f1.trajectory_kw, f2.trajectory_kw)

    def test_pseudo_origin_level_forecasts_use_only_data_before_their_own_pseudo_origin(self):
        load, temp, types, start = record()
        base = _level_parts(history_at(load, temp, types, start), "Office", 0.25)["err"]
        rng = np.random.default_rng(1)
        for k in range(1, 12):
            load2, temp2 = load.copy(), temp.copy()
            cut = O - k * STEP                                                         # change everything after o_k
            load2[cut:O] *= rng.uniform(0.5, 1.5, O - cut); temp2[cut:O] += rng.normal(0, 5, O - cut)
            err = _level_parts(history_at(load2, temp2, types, start), "Office", 0.25)["err"]
            for name, e in base.items():
                # pseudo-origins j > k lie before the change and their targets end by o_k: identical errors
                np.testing.assert_array_equal(err[name][0, k:], e[0, k:], err_msg=f"{name}, change after o_{k}")

    def test_interval_reads_nothing_at_or_after_the_origin(self):
        load, temp, types, start = record()
        Q1, n1 = readouts.residual_quantiles(load, temp, types, start, "Office", 0.25, O)
        rng = np.random.default_rng(2)
        load2, temp2 = load.copy(), temp.copy()
        load2[O:] = rng.normal(0, 100, len(load) - O); temp2[O:] = rng.normal(0, 30, len(temp) - O)
        Q2, n2 = readouts.residual_quantiles(load2, temp2, types, start, "Office", 0.25, O)
        self.assertEqual(n1, n2)
        np.testing.assert_array_equal(Q1, Q2)

    def test_values_at_unobserved_hours_do_not_reach_the_forecast(self):
        """observed=False means the hour is not known: its placeholder value must not matter to any branch (the
        foundation-model contexts, the off-state and micro-load rules, the analog days, the estimator)."""
        load, temp, types, start = record()
        obs = np.ones(O, bool)
        obs[O - 3 * STEP - 200:O - 3 * STEP - 100] = False        # inside the pseudo-origin contexts k = 3, 4
        obs[O - 365 * 24 + 5 * 24:O - 365 * 24 + 7 * 24] = False  # two analog-candidate days a year back
        obs[3000:3400] = False                                     # an early gap
        ref = {}
        for fill in (np.nan, 0.0, 500.0, -40.0):
            lf = load[:O].copy(); lf[~obs] = fill
            h = History(load_kw=lf, temperature_c=temp[:O], day_types=types[:O + STEP], start_timestamp=start, observed=obs)
            mapping = {k: seasonal_naive(c[None])[0] for k, c in pseudo_origin_contexts(np.where(obs, load[:O], np.nan)).items()}
            for name, fdn in (("callable", seasonal_naive), ("mapping", mapping)):
                f = anchor.forecast(h, group="Office", temp_sigma_std=0.25, foundation=fdn, dst_region="EU").trajectory_kw
                if name in ref:
                    np.testing.assert_array_equal(f, ref[name], err_msg=f"{name}, placeholder {fill}")
                else:
                    ref[name] = f
        np.testing.assert_array_equal(ref["callable"], ref["mapping"])

    def test_pseudo_origin_contexts_end_inside_the_record(self):
        load, _, _, _ = record()
        for k, c in pseudo_origin_contexts(load[:O]).items():
            np.testing.assert_array_equal(c, load[O - k * STEP - CONTEXT:O - k * STEP])


if __name__ == "__main__":
    unittest.main()
