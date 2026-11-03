"""Tests of the API around the anchoring layer: the readouts of ``ankyra_lite.readouts`` (interval, blocks, raw
trajectory, energy added by the projection, peak), the history-only configuration through ``ankyra_lite.forecast``,
the fields read through to the anchoring layer, the absence of information from after the origin end to end, the
top-level names, and the two examples.  The checks hold with the continuation layer off and on (release default
``ankyra_lite._release.CONTINUATION_DEFAULT``): the readouts that are read before the layer are checked against
``trajectory_before_continuation_kw``, which is ``trajectory_kw`` when the layer is off.

Synthetic data and an untrained carrier (``build_carrier()``, seed 0); no checkpoint, study data or ANKYRA checkout is
needed.  Run from the repository root:  python -m unittest tests.test_readouts_lite -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import contextlib
import dataclasses
import importlib.util
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch.set_num_threads(1)

import ankyra_lite                                                                    # noqa: E402
from ankyra_lite import anchor as A                                                   # noqa: E402
from ankyra_lite import build_carrier, carrier_forecasts, readouts                    # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                            # noqa: E402
from ankyra_lite.forecast import ForecastResult, forecast, peak                       # noqa: E402

O, STEP, CONTEXT, HORIZON = 12000, 744, 1344, 744
GROUP, SIGMA, DST = "Office", 0.25, "EU"
RECORD_KEYS = {"carrier_level_weight", "history_vs_carrier_log_ratio", "pseudo_origins"}
LEVEL_KEYS = {"s_u", "l_u", "a_u", "s_w", "l_w", "a_w", "fm"}


def carrier():
    torch.manual_seed(0)
    return build_carrier().eval()


def record(hours=O + STEP, seed=3):
    """An artificial record that continues 744 hours past the origin O (calendar a further 744 hours), with noise."""
    full = synthetic_history(hours)
    load = np.asarray(full.load_kw, dtype=np.float64) + np.random.default_rng(seed).normal(0.0, 0.3, hours)
    return load, np.asarray(full.temperature_c, dtype=np.float64), np.asarray(full.day_types), full.start_timestamp


def history_at(load, temp, types, start, origin=O, observed=None):
    return A.History(load_kw=load[:origin], temperature_c=temp[:origin], day_types=types[:origin + STEP], start_timestamp=start,
                     observed=None if observed is None else observed[:origin])


def state_history(kind):
    """A record whose last 200 hours are off (``off``) or whose whole record stays within 1e-3 kW of zero (``micro``)."""
    h = synthetic_history(O)
    load = np.asarray(h.load_kw, dtype=np.float64)
    if kind == "off":
        load[-200:] = 1e-7 * (np.arange(200) % 3)
    else:
        load = 5e-4 * load / load.max()
    return A.History(load, h.temperature_c, h.day_types, h.start_timestamp)


def run_example(name, argv):
    spec = importlib.util.spec_from_file_location(f"example_{name}", REPO / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mod.main(argv)
    return json.loads(out.getvalue())


class LiteReadoutBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = carrier()
        cls.load, cls.temp, cls.types, cls.start = record()
        cls.h = history_at(cls.load, cls.temp, cls.types, cls.start)
        cls.r = forecast(cls.h, cls.m, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)


class TestInterval(LiteReadoutBase):
    def test_interval_equals_the_three_calls_of_the_anchoring_layer(self):
        bands, n = readouts.interval(self.h, self.r, group=GROUP, temp_sigma_std=SIGMA)
        load, types = np.asarray(self.h.load_kw, dtype=np.float64), np.asarray(self.h.day_types)
        Q, n_ref = A.readouts.residual_quantiles(load, self.h.temperature_c, types, self.h.start_timestamp, GROUP, SIGMA, O)
        args = (A.readouts.origin_scale(load[O - CONTEXT:O]), Q, (O + np.arange(HORIZON)) % 24, (types[O:O + HORIZON] < 5).astype(int))
        ref = A.readouts.interval_bands(self.r.trajectory_kw, *args)
        self.assertEqual(bands.shape, (5, HORIZON))
        self.assertTrue(np.array_equal(bands, ref))                                   # bit for bit
        self.assertEqual((n, n_ref), (12, 12))
        self.assertTrue(np.isfinite(bands).all() and (bands >= 0).all())
        self.assertTrue((np.diff(bands, axis=0) >= 0).all())                          # monotone across the levels
        self.assertEqual(readouts.LEVELS, A.readouts.LEVELS)
        same, _ = readouts.interval(self.h, self.r.trajectory_kw, group=GROUP, temp_sigma_std=SIGMA)   # a point array works too
        self.assertTrue(np.array_equal(same, bands))
        before = A.readouts.interval_bands(self.r.trajectory_before_continuation_kw, *args)   # = bands with the layer off
        self.assertTrue(np.array_equal(ankyra_lite.interval(self.h, self.r.anchored, group=GROUP, temp_sigma_std=SIGMA)[0], before))

    def test_interval_ignores_values_at_unobserved_hours(self):
        obs = np.ones(O, bool)
        obs[O - 4 * STEP - 300:O - 4 * STEP - 250] = False
        ref = None
        for fill in (np.nan, 0.0, 500.0):
            lf = self.load.copy()
            lf[:O][~obs] = fill
            h = history_at(lf, self.temp, self.types, self.start, observed=np.concatenate([obs, np.ones(STEP, bool)]))
            got = readouts.interval(h, self.r, group=GROUP, temp_sigma_std=SIGMA)
            if ref is None:
                ref = got
                self.assertLess(got[1], 12)                                          # the pseudo-window with the gap drops out
            self.assertEqual(got[1], ref[1], fill)
            self.assertTrue(np.array_equal(got[0], ref[0]), fill)


class TestNoFutureInformation(unittest.TestCase):
    def test_nothing_at_or_after_the_origin_reaches_the_forecast_or_its_readouts(self):
        """Change every load and temperature value at or after the origin: the carrier forecasts (computed from the full
        arrays and from the history), the trajectory, level, peak, interval and anchoring record stay bit-identical."""
        m = carrier()
        load, temp, types, start = record()
        rng = np.random.default_rng(7)
        load2, temp2 = load.copy(), temp.copy()
        load2[O:] = rng.normal(0.0, 100.0, len(load) - O)
        temp2[O:] += rng.normal(0.0, 20.0, len(temp) - O)
        out = []
        for ld, tp in ((load, temp), (load2, temp2)):
            F = carrier_forecasts(m, ld, tp, types, GROUP, start, 0, O)              # reads the arrays that run past the origin
            h = history_at(ld, tp, types, start)
            r = forecast(h, m, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)
            rF = forecast(h, F, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)
            self.assertEqual(sorted(F), sorted(r.carrier))
            for k in F:
                self.assertTrue(np.array_equal(F[k], r.carrier[k]), k)
            self.assertTrue(np.array_equal(rF.trajectory_kw, r.trajectory_kw))
            bands, n = readouts.interval(h, r, group=GROUP, temp_sigma_std=SIGMA)
            out.append((F, r, bands, n))
        (F1, r1, b1, n1), (F2, r2, b2, n2) = out
        self.assertEqual(sorted(F1), list(range(7)))
        for k in F1:
            self.assertTrue(np.array_equal(F1[k], F2[k]), k)
        self.assertTrue(np.array_equal(r1.trajectory_kw, r2.trajectory_kw))
        self.assertEqual(r1.level_kw, r2.level_kw)
        self.assertEqual(r1.peak_kw, r2.peak_kw)
        self.assertEqual(n1, n2)
        self.assertTrue(np.array_equal(b1, b2))
        self.assertEqual(set(r1.anchoring_record), RECORD_KEYS)
        np.testing.assert_equal(r1.anchoring_record, r2.anchoring_record)
        self.assertFalse(np.array_equal(load[O:], load2[O:]))


class TestReadThroughFields(LiteReadoutBase):
    def test_fields_on_a_window_that_is_not_a_state_window(self):
        r = self.r
        self.assertFalse(r.state_window)
        self.assertTrue(np.array_equal(r.foundation_within_day_kw, A.blocks.within_day(r.carrier[0])))   # the carrier's own within-day block
        for name, shape in (("daily_means_kw", (31,)), ("fixed_division_daily_means_kw", (31,)), ("within_day_kw", (HORIZON,)),
                            ("analog_shape_kw", (HORIZON,)), ("foundation_within_day_kw", (HORIZON,))):
            v = getattr(r, name)
            self.assertIs(v, getattr(r.anchored, name), name)                       # read through, not copied
            self.assertEqual(np.shape(v), shape, name)
            self.assertTrue(np.isfinite(v).all(), name)
        self.assertEqual(set(r.anchoring_record), RECORD_KEYS)
        self.assertEqual(r.anchoring_record["carrier_level_weight"], r.level_weights["fm"])
        self.assertEqual(r.anchoring_record["pseudo_origins"], 6)
        self.assertTrue(np.isfinite(r.anchoring_record["history_vs_carrier_log_ratio"]))
        self.assertEqual(set(r.level_weights), LEVEL_KEYS)
        self.assertAlmostEqual(sum(r.level_weights.values()), 1.0, places=12)
        self.assertEqual(len(r.lead_week_weights), 4)
        self.assertEqual(r.pseudo_pairs, 6)
        self.assertEqual(r.within_pseudo_pairs, 3)
        self.assertEqual(len(set(r.within_trust)), 1)                              # one trust value per window
        self.assertTrue(np.allclose(A.blocks.within_day(r.within_day_kw), r.within_day_kw, atol=1e-12))   # zero daily means

    def test_fields_on_state_windows_keep_the_anchoring_layers_defaults(self):
        for kind in ("off", "micro"):
            r = forecast(state_history(kind), self.m, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)
            self.assertTrue(r.state_window, kind)
            self.assertEqual((r.off_state, r.micro_load), (kind == "off", kind == "micro"))
            self.assertEqual(r.state_rule, "seasonal_naive")
            T0 = r.carrier[0]
            self.assertTrue(np.array_equal(r.anchored.trajectory_kw, T0), kind)        # the anchoring layer's return
            self.assertTrue(np.array_equal(r.daily_means_kw, T0.reshape(31, 24).mean(1)), kind)
            self.assertTrue(np.array_equal(r.within_day_kw, A.blocks.within_day(T0)), kind)
            self.assertTrue(np.array_equal(r.foundation_within_day_kw, r.within_day_kw), kind)
            self.assertIsNone(r.fixed_division_daily_means_kw, kind)
            self.assertIsNone(r.analog_shape_kw, kind)
            self.assertEqual(r.level_weights, {}, kind)
            self.assertEqual(r.lead_week_weights, (), kind)
            self.assertEqual(r.anchoring_record, {}, kind)
            self.assertEqual((r.pseudo_pairs, r.within_pseudo_pairs, r.analog_kept), (0, 0, False), kind)
            self.assertEqual(r.within_trust, (0.0, 0.0, 0.0, 0.0), kind)
            self.assertTrue(np.array_equal(readouts.raw_trajectory(r), r.trajectory_kw), kind)   # no projection on a state window
            self.assertEqual(readouts.projection_energy_increase(r), 0.0, kind)


class TestBlocksAndEnergy(LiteReadoutBase):
    def _check(self, r):
        raw = readouts.raw_trajectory(r)
        before = r.trajectory_before_continuation_kw                               # the delivered trajectory when the layer is off
        self.assertTrue(np.array_equal(np.maximum(raw, 0.0), before))              # bit for bit
        b = readouts.blocks(raw)
        scale = float(np.abs(raw).max())
        self.assertTrue(np.allclose(A.blocks.compose(b.level_kw, b.daily_path_kw, b.within_day_kw), raw, rtol=0, atol=1e-12 * scale))
        lv, path, wd = b                                                           # a named tuple
        self.assertAlmostEqual(float(lv), r.level_kw, delta=1e-12 * scale)
        self.assertTrue(np.allclose(path, r.daily_means_kw - r.daily_means_kw.mean(), rtol=0, atol=1e-12 * scale))
        self.assertTrue(np.allclose(wd, r.within_day_kw, rtol=0, atol=1e-12 * scale))
        self.assertAlmostEqual(float(path.sum()), 0.0, delta=1e-12 * scale)
        self.assertEqual(r.energy_kwh, float(HORIZON * np.mean(r.daily_means_kw)))   # energy readout: 744 x level, before the projection
        inc = readouts.projection_energy_increase(r)
        self.assertEqual(inc, A.readouts.projection_energy_increase_kwh(raw))
        self.assertAlmostEqual(float(before.sum()) - r.energy_kwh, inc, delta=1e-9 * HORIZON * scale)
        self.assertAlmostEqual(float(r.trajectory_kw.sum()) - r.energy_kwh, inc + readouts.continuation_energy_change(r),
                               delta=1e-9 * HORIZON * scale)

    def test_blocks_of_the_raw_forecast_sum_back_to_it(self):
        self._check(self.r)
        self._check(forecast(self.h, None, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST))
        batch = readouts.blocks(np.stack([self.r.trajectory_kw, readouts.raw_trajectory(self.r)]))
        self.assertEqual((batch.level_kw.shape, batch.daily_path_kw.shape, batch.within_day_kw.shape), ((2,), (2, 31), (2, HORIZON)))

    def test_projection_energy_increase_where_the_projection_binds(self):
        rng = np.random.default_rng(0)
        daily = rng.normal(0.5, 1.0, 31)
        W = A.blocks.within_day(rng.normal(0.0, 2.0, HORIZON))
        raw = np.repeat(daily, 24) + W
        r = SimpleNamespace(state_window=False, daily_means_kw=daily, within_day_kw=W, trajectory_kw=np.maximum(raw, 0.0))
        inc = readouts.projection_energy_increase(r)
        self.assertGreater(inc, 0.0)
        self.assertEqual(inc, float(np.maximum(-raw, 0.0).sum()))
        self.assertAlmostEqual(inc, float(r.trajectory_kw.sum() - raw.sum()), places=9)

    def test_peak_is_the_readout_of_the_trajectory_before_the_continuation_layer(self):
        self.assertIs(readouts.peak, peak)
        before = self.r.trajectory_before_continuation_kw                          # the delivered trajectory when the layer is off
        self.assertEqual(self.r.peak_kw, readouts.peak(before, self.h.load_kw, self.h.day_types))
        ctx = np.asarray(self.h.load_kw)[O - CONTEXT:O]
        ty = np.asarray(self.h.day_types)
        ref = A.readouts.peak_readout(before, ctx[None], ty[O - CONTEXT + 12 + 24 * np.arange(56)][None],
                                      ty[O + 12 + 24 * np.arange(31)][None])[0]
        self.assertEqual(self.r.peak_kw, float(ref))
        self.assertGreaterEqual(self.r.peak_kw, float(before.reshape(31, 24).mean(1).max()))


class TestHistoryOnly(LiteReadoutBase):
    def test_history_only_equals_the_anchoring_layer_field_by_field(self):
        r = forecast(self.h, None, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)
        ref = A.forecast(self.h, group=GROUP, temp_sigma_std=SIGMA, foundation=None, dst_region=DST)
        self.assertIsInstance(r, ForecastResult)
        for f in dataclasses.fields(A.AnkyraForecast):
            np.testing.assert_equal(getattr(r.anchored, f.name), getattr(ref, f.name), err_msg=f.name)
            np.testing.assert_equal(getattr(r, f.name), getattr(ref, f.name), err_msg=f.name)
        self.assertTrue(np.array_equal(r.trajectory_kw, ref.trajectory_kw))
        self.assertEqual(r.level_kw, ref.level_kw)
        self.assertEqual(r.energy_kwh, ref.energy_kwh)
        self.assertEqual(r.peak_kw, peak(ref.trajectory_kw, self.h.load_kw, self.h.day_types))
        self.assertEqual((r.state_window, r.state_rule, dict(r.carrier)), (False, None, {}))
        self.assertTrue(np.array_equal(r.daily_means_kw, r.fixed_division_daily_means_kw))   # no handover
        self.assertEqual((r.level_weights, r.lead_week_weights, r.anchoring_record), ({}, (), {}))
        self.assertFalse(np.array_equal(r.trajectory_kw, self.r.trajectory_kw))
        same = forecast(self.h, None, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST, state_rule=None, cut=O + 1)
        self.assertTrue(np.array_equal(same.trajectory_kw, r.trajectory_kw))      # no state rule; cut is not used
        with self.assertRaises(ValueError):
            forecast(self.h, None, group=GROUP, temp_sigma_std=SIGMA, state_rule="other")

    def test_no_state_rule_for_the_history_only_configuration(self):
        h = state_history("off")
        r = forecast(h, None, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)
        ref = A.forecast(h, group=GROUP, temp_sigma_std=SIGMA, foundation=None, dst_region=DST)
        self.assertFalse(r.state_window or r.off_state or r.micro_load)
        self.assertIsNone(r.state_rule)
        self.assertTrue(np.array_equal(r.trajectory_kw, ref.trajectory_kw))


class TestTopLevelNames(unittest.TestCase):
    def test_names_mirror_the_anchoring_layer(self):
        for name in ("History", "InputError", "SignatureDegeneracy", "fill_short_gaps", "pseudo_origin_contexts"):
            self.assertIs(getattr(ankyra_lite, name), getattr(A, name), name)
            self.assertIn(name, ankyra_lite.__all__)
        self.assertIs(ankyra_lite.interval, readouts.interval)
        self.assertIs(ankyra_lite.readouts, readouts)
        self.assertIn("interval", ankyra_lite.__all__)
        self.assertIn("readouts", ankyra_lite.__all__)
        self.assertIs(ankyra_lite.forecast, forecast)                              # the function (see the package docstring)
        for name in ankyra_lite.__all__:
            self.assertTrue(hasattr(ankyra_lite, name), name)
        self.assertTrue(issubclass(ankyra_lite.InputError, ValueError))


class TestExamples(unittest.TestCase):
    def test_quickstart_prints_the_expected_lines(self):
        out = run_example("quickstart", ["--untrained"])
        layer = ["continuation_x_fast_x_day", "continuation_energy_change_kwh"] if ankyra_lite._release.CONTINUATION_DEFAULT else []
        self.assertEqual(list(out), ["carrier", "carrier_parameters", "carrier_issue_times_k", "state_window", "state_rule", "energy_kwh",
                                     "level_kw", "peak_kw"] + layer + ["interval_80pct_first_hour_kw", "interval_pseudo_windows",
                                     "lead_week_weights_on_carrier", "level_weights", "pseudo_origin_pairs",
                                     "within_day_trust_on_analog_shape", "within_day_pseudo_origin_triples", "analog_shape_kept",
                                     "trajectory_first_day_kw", "seconds"])
        if layer:
            self.assertEqual(len(out["continuation_x_fast_x_day"]), 2)
            self.assertTrue(all(-1.0 <= x <= 1.0 for x in out["continuation_x_fast_x_day"]))
        self.assertTrue(out["carrier"].startswith("untrained stand-in"))
        self.assertEqual(out["carrier_parameters"], 61613)
        self.assertEqual(out["carrier_issue_times_k"], list(range(7)))
        self.assertEqual(out["interval_pseudo_windows"], 12)
        lo, hi = out["interval_80pct_first_hour_kw"]
        self.assertLessEqual(lo, hi)
        self.assertEqual(set(out["level_weights"]), LEVEL_KEYS)
        self.assertEqual(len(out["trajectory_first_day_kw"]), 24)

    def test_own_unit_example(self):
        out = run_example("own_unit", ["--untrained"])
        rec = out["record"]
        self.assertEqual(rec["start_timestamp"], "2019-01-01T00:00:00+00:00")
        self.assertEqual(rec["padded_hours"], (31 + 28 + 17) * 24 + 9)              # 1 January 00:00 to 18 March 09:00
        self.assertEqual(rec["origin_hour_index"], (365 + 274) * 24)                  # 1 October 2020 00:00
        self.assertEqual((rec["missing_hours_after_padding"], rec["missing_hours_after_fill_short_gaps"]), (37, 30))
        self.assertEqual(rec["complete_hours_before_origin"], rec["origin_hour_index"] - rec["padded_hours"] - 30)
        self.assertEqual(rec["holiday_days_in_forecast_month"], 1)
        self.assertIn("1344", out["refused_without_filling"])
        self.assertEqual(out["carrier_issue_times_k"], list(range(7)))
        self.assertFalse(out["state_window"])
        self.assertEqual(set(out["anchoring_record"]), RECORD_KEYS)
        self.assertEqual(out["history_only"]["carrier"], [])
        self.assertFalse(out["history_only"]["state_window"])
        self.assertGreaterEqual(out["projection_energy_increase_kwh"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
