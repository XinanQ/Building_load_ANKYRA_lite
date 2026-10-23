"""Exact properties O1-O9 of the continuation layer (theory/PROOFS.md, "Exact properties of the continuation layer"),
checked numerically, together with the counterexamples that bound them.

The layer is ``ankyra_lite.continuation``.  The window-level checks call it on synthetic windows (load rows with daily
and weekly cycles, a calendar with holidays, and trajectories that miss the load by a random level and shape error);
the end-to-end checks call ``ankyra_lite.forecast`` on an artificial building with a stand-in carrier (the seasonal
naive of each issue time's context, precomputed) and with an untrained carrier (``build_carrier`` under a torch seed).
CPU only; no checkpoint and no study data are needed.  Run from the repository root:

    python -m unittest discover -s theory -t .
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch.set_num_threads(1)

from ankyra_lite import anchor as A                                                   # noqa: E402
from ankyra_lite import continuation as CT                                            # noqa: E402
from ankyra_lite import readouts                                                      # noqa: E402
from ankyra_lite.anchor.synthetic import seasonal_naive as context_naive              # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                            # noqa: E402
from ankyra_lite.carrier import build_carrier                                         # noqa: E402
from ankyra_lite.forecast import forecast, peak                                       # noqa: E402

H, N_PRE, STEP, CONTEXT = 744, 672, 744, 1344
ORIGIN = 2000                                     # origin of the synthetic windows (an origin at 08:00)
O = 12000                                         # origin of the artificial building (end-to-end checks)
GROUP, SIGMA, DST = "Office", 0.25, "EU"
FROZEN_DIGEST = "d514e3ec93a745b14950671297bf2f922fd512424a804e292377e1b0c4cc1dce"
FROZEN_CONSTANTS = {"kappa1": 0.479, "rho1": 12.0, "kappa2": 0.330, "rho2": 168.0}
FROZEN_RULE = {"fast_hours": 3, "day_hours": 24, "scale_hours": 672, "scale_floor": 1e-3, "eps_fraction": 0.01, "clip": 1.0,
               "workday_type_below": 5, "type_hour": 12, "reference_days": 7, "no_match_day": 6, "horizon": 744,
               "bypass": "state windows (off_state or micro_load) and missing last hours (x = 0)"}


# ----------------------------------------------------------------------------- helpers
def calendar(hours: int) -> np.ndarray:
    """Day types of ``hours`` hours starting on a Tuesday: Monday = 0 ... Sunday = 6, every 37th day a holiday (7)."""
    day = np.arange(hours) // 24
    types = ((day + 1) % 7).astype(np.int64)
    types[day % 37 == 5] = 7
    return types


def windows(n: int = 40, seed: int = 0, error: float = 0.4, origin: int = ORIGIN):
    """``n`` synthetic windows on one calendar: (load rows, calendar, trajectories F).

    The load is nonnegative, with daily and weekly cycles, a row-specific size and 1% unobserved hours (NaN), none of
    them in the 24 hours before the origin.  F >= 0 is the load after the origin times a lognormal level error (sigma
    ``error``) and an hourly error (sigma ``error`` / 4), with 3% of the hours set to exactly zero."""
    rng = np.random.default_rng(seed)
    T = origin + H + 48
    t = np.arange(T)
    load = (5 + 3 * np.sin(2 * np.pi * t / 24) + np.cos(2 * np.pi * t / 168))[None] + rng.gamma(2.0, 1.0, (n, T))
    load *= rng.uniform(0.2, 5.0, (n, 1))
    gaps = rng.random((n, T)) < 0.01
    gaps[:, origin - 24:origin] = False
    load[gaps] = np.nan
    F = np.nan_to_num(load[:, origin:origin + H], nan=1.0) * rng.lognormal(0.0, error, (n, 1)) * rng.lognormal(0.0, error / 4, (n, H))
    F[rng.random((n, H)) < 0.03] = 0.0
    return load, calendar(T), F


def batch_inputs(load, types, origin: int = ORIGIN):
    """(pre, last_type, day_types) of every row for ``continuation.apply_batch``."""
    pre, last, days = zip(*(CT.window_inputs(row, types, origin) for row in load))
    return np.stack(pre), np.array(last), np.array(days)


def envelope(constants=CT.CONSTANTS) -> np.ndarray:
    """E(h) evaluated with the layer's own operations, (kappa1 * 1) g1 + (kappa2 * 1) g2: the value of Delta(h) when
    both anomalies are clipped at +1."""
    g1, g2 = CT.decay_curves(constants)
    return (constants.kappa1 * 1.0) * g1 + (constants.kappa2 * 1.0) * g2


def building(seed: int = 3):
    """An artificial office building ending at the origin O (calendar through the horizon), with noise."""
    full = synthetic_history(O + STEP)
    load = np.asarray(full.load_kw, dtype=np.float64) + np.random.default_rng(seed).normal(0.0, 0.3, O + STEP)
    types = np.asarray(full.day_types)
    h = A.History(load_kw=load[:O], temperature_c=np.asarray(full.temperature_c)[:O], day_types=types[:O + STEP],
                  start_timestamp=full.start_timestamp)
    return h, load[:O], types


def stand_in_carrier(load, o: int = O, bias: float = 1.12) -> dict:
    """{k: forecast issued at o - 744 k}: the seasonal naive of each issue time's 1,344-hour context; the forecast at the
    origin is scaled by ``bias`` (a carrier that misses the level, so that the continuation layer acts)."""
    q = [o - k * STEP for k in range(7)]
    F = context_naive(np.stack([load[x - CONTEXT:x] for x in q]))
    return {k: F[k] * (bias if k == 0 else 1.0) for k in range(7)}


def lite(h, carriers, **kw):
    return forecast(h, carriers, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST, **kw)


_END_TO_END: dict = {}


class EndToEndBase(unittest.TestCase):
    """One artificial building, forecast with the layer on and off, with a stand-in carrier and an untrained carrier
    (computed once for the classes that share it)."""

    @classmethod
    def setUpClass(cls):
        if not _END_TO_END:
            h, load, types = building()
            F = stand_in_carrier(load)
            torch.manual_seed(0)
            net = [build_carrier().eval()]
            _END_TO_END.update(h=h, load=load, types=types, F=F, net=net,
                               on=lite(h, F, continuation=True), off=lite(h, F, continuation=False),
                               on_net=lite(h, net, continuation=True), off_net=lite(h, net, continuation=False))
        for k, v in _END_TO_END.items():
            setattr(cls, k, v)


# ============================================================================= O1 nonnegativity, sign and zeros
class NonnegativityTests(unittest.TestCase):
    def test_nonnegative_trajectories_stay_nonnegative_and_zero_hours_stay_zero(self):
        for seed in (0, 1, 2):
            load, types, F = windows(seed=seed)
            pre, last, days = batch_inputs(load, types)
            self.assertGreater(int((F == 0).sum()), 0)
            for dtype in (np.float32, np.float64):
                Fd = F.astype(dtype)
                Fp, _ = CT.apply_batch(Fd, pre, last, days, np.zeros(len(F), bool))
                self.assertEqual(Fp.dtype, np.float64)
                self.assertTrue(np.isfinite(Fp).all() and (Fp >= 0).all(), (seed, dtype))
                self.assertTrue(np.array_equal(Fp == 0, Fd == 0), (seed, dtype))

    def test_the_sign_of_every_hour_is_kept(self):
        load, types, F = windows(seed=3)
        signed = F * np.where(np.random.default_rng(3).random(F.shape) < 0.3, -1.0, 1.0)
        pre, last, days = batch_inputs(load, types)
        Fp, _ = CT.apply_batch(signed, pre, last, days, np.zeros(len(F), bool))
        self.assertTrue(np.array_equal(np.sign(Fp), np.sign(signed)))

    def test_the_smallest_subnormal_hour_can_underflow_to_zero(self):
        # counterexample to "F'(h) = 0 only where F(h) = 0": 2^-1074 kW times exp(-0.809) rounds to zero
        load, types, _ = windows(n=1, seed=4)
        F = np.full(H, 1e3)                                                          # far above the load: x_f = x_d = -1
        F[0], F[1] = 5e-324, np.finfo(np.float64).tiny
        Fp, rec = CT.apply(F, load[0], types, ORIGIN)
        self.assertEqual((rec.x_fast, rec.x_day), (-1.0, -1.0))
        self.assertGreater(F[0], 0.0)
        self.assertEqual(Fp[0], 0.0)
        self.assertGreater(Fp[1], 0.0)                                               # the smallest normal number stays positive

    def test_a_non_finite_trajectory_is_rejected_on_an_ordinary_window(self):
        load, types, F = windows(n=2, seed=5)
        F[1, 100] = np.nan
        pre, last, days = batch_inputs(load, types)
        with self.assertRaises(ValueError):
            CT.apply_batch(F, pre, last, days, np.zeros(2, bool))
        Fp, _ = CT.apply_batch(F, pre, last, days, np.array([False, True]))         # a state window is passed through
        self.assertTrue(np.array_equal(Fp[1], F[1], equal_nan=True))

    def test_a_nonpositive_last_hours_mean_sets_its_anomaly_to_zero(self):
        # where the claim stops: a last-hours mean at or below -eps has no logarithm; that anomaly is set to 0 and
        # flagged (as for an unobserved span), and F' keeps the sign of F; just above -eps the anomaly is used
        load, types, F = windows(n=1, seed=6, error=0.1)
        row = load[0].copy()
        eps = 0.01 * max(float(np.nanmean(row[ORIGIN - N_PRE:ORIGIN])), 1e-3)
        row[ORIGIN - 3:ORIGIN] = -2 * eps
        Fp, rec = CT.apply(F[0], row, types, ORIGIN)
        self.assertEqual((rec.x_fast, rec.nonpositive_fast), (0.0, True))
        self.assertTrue(np.isfinite(Fp).all() and (Fp >= 0).all())
        self.assertTrue(np.array_equal(Fp == 0, F[0] == 0))
        row[ORIGIN - 24:ORIGIN] = -2 * eps                                            # both spans: F' = F bit for bit
        Fp, rec = CT.apply(F[0], row, types, ORIGIN)
        self.assertEqual(Fp.tobytes(), np.asarray(F[0], np.float64).tobytes())
        self.assertEqual(rec.bypass, "nonpositive load")
        row = load[0].copy()
        row[ORIGIN - 3:ORIGIN] = -0.5 * eps
        Fp, rec = CT.apply(F[0], row, types, ORIGIN)
        self.assertEqual(rec.x_fast, -1.0)
        self.assertTrue(np.isfinite(Fp).all() and (Fp >= 0).all())


# ============================================================================= O2 bounded correction
class BoundTests(unittest.TestCase):
    def test_the_envelope_of_the_frozen_constants(self):
        E = envelope()
        self.assertTrue(np.array_equal(E, CT.envelope(CT.CONSTANTS)))
        self.assertTrue((np.diff(E) < 0).all())                                      # strictly decreasing in h
        self.assertLessEqual(E[0], 0.809)
        self.assertAlmostEqual(E[0], 0.809, delta=1e-15)
        self.assertGreater(E[167], 0.122)                                            # last hour of day 7: above 0.12
        self.assertLessEqual(E[168], 0.1215)                                         # from day 8 on
        self.assertLessEqual(E[743], 0.004)                                          # last hour of day 31
        self.assertGreater(E[168], 0.1214)                                           # 0.121401: the bound is 0.1215
        for h, lo, hi, step in ((0, 0.4453, 2.2457, 1e-4), (23, 0.6989, 1.4309, 1e-4), (167, 0.8850, 1.1299, 1e-4),
                                (168, 0.8856, 1.1291, 1e-4), (743, 0.99604, 1.00397, 1e-5)):   # PROOFS.md O2, rounded outward
            self.assertTrue(lo <= math.exp(-E[h]) < lo + step and hi - step < math.exp(E[h]) <= hi, h)
            self.assertTrue(np.exp(-E[h:]).min() >= lo and np.exp(E[h:]).max() <= hi, h)      # and for every later hour

    def test_zero_constants_have_zero_envelope(self):
        self.assertTrue(np.array_equal(CT.envelope(CT.ZERO), np.zeros(H)))

    def test_every_correction_lies_within_the_envelope(self):
        E = envelope()
        clipped = 0
        for seed in range(4):
            load, types, F = windows(seed=seed, error=0.2 + 0.3 * seed)
            for i in range(0, len(F), 3):
                Fp, rec = CT.apply(F[i], load[i], types, ORIGIN)
                self.assertTrue(-1.0 <= rec.x_fast <= 1.0 and -1.0 <= rec.x_day <= 1.0)
                self.assertTrue((np.abs(rec.delta) <= E).all(), (seed, i))           # bitwise, no tolerance
                self.assertTrue(np.array_equal(Fp, F[i] * np.exp(rec.delta)))
                self.assertEqual(rec.max_abs_delta, float(np.abs(rec.delta).max()))
                pos = F[i] > 0
                ratio = Fp[pos] / F[i][pos]
                self.assertTrue((ratio >= np.exp(-E[pos]) * (1 - 1e-15)).all() and (ratio <= np.exp(E[pos]) * (1 + 1e-15)).all())
                clipped += int(rec.clipped_fast or rec.clipped_day)
        self.assertGreater(clipped, 0)

    def test_the_bound_is_attained_when_both_anomalies_are_clipped_to_the_same_sign(self):
        load, types, F = windows(n=1, seed=7)
        E = envelope()
        for factor, sign in ((1e-3, 1.0), (1e3, -1.0)):                              # forecast far below / above the load
            Fp, rec = CT.apply(F[0] * factor, load[0], types, ORIGIN)
            self.assertTrue(rec.clipped_fast and rec.clipped_day)
            self.assertEqual((rec.x_fast, rec.x_day), (sign, sign))
            self.assertTrue(np.array_equal(rec.delta, sign * E))
            self.assertEqual(rec.max_abs_delta, E[0])


# ============================================================================= O3 causality
class CausalityTests(unittest.TestCase):
    @staticmethod
    def garbage(a, sl):
        """A copy of ``a`` whose hours ``sl`` hold NaN, +inf, a negative value and a huge value."""
        b = np.array(a, dtype=np.float64, copy=True)
        b[sl] = np.resize(np.array([np.nan, np.inf, -5.0, 1e30]), len(b[sl]))
        return b

    def test_only_the_last_672_hours_and_eight_calendar_hours_are_read(self):
        load, types, F = windows(seed=8)
        read = [ORIGIN - 12] + [ORIGIN + 12 + 24 * d for d in range(7)]
        other = np.setdiff1d(np.arange(len(types)), read)
        for i in range(len(F)):
            a, _ = CT.apply(F[i], load[i], types, ORIGIN)
            row = self.garbage(self.garbage(load[i], slice(ORIGIN, None)), slice(0, ORIGIN - N_PRE))
            ty = types.copy()
            ty[other] = (ty[other] + 3) % 8
            b, _ = CT.apply(F[i], row, ty, ORIGIN)
            self.assertTrue(np.array_equal(a, b), i)

    def test_the_last_observed_hour_is_read(self):
        # sanity check of the claim: the hour before the origin enters (windows with observed, unclipped anomalies)
        load, types, F = windows(seed=9, error=0.1)
        changed = 0
        for i in range(len(F)):
            a, ra = CT.apply(F[i], load[i], types, ORIGIN)
            row = load[i].copy()
            row[ORIGIN - 1] *= 1.5
            b, rb = CT.apply(F[i], row, types, ORIGIN)
            if not any((ra.clipped_fast, ra.clipped_day, rb.clipped_fast, rb.clipped_day)):
                self.assertFalse(np.array_equal(a, b), i)
                changed += 1
        self.assertGreaterEqual(changed, 10)

    def test_the_calendar_of_the_first_forecast_week_is_read(self):
        # where the claim stops: the day types of the first forecast week are an input, through the reference day
        load, types, F = windows(n=1, seed=10, error=0.1)
        a, ra = CT.apply(F[0], load[0], types, ORIGIN)
        ty = types.copy()
        ty[ORIGIN - 12] = 6 if ty[ORIGIN - 12] < 5 else 2                            # the last observed day changes class
        b, rb = CT.apply(F[0], load[0], ty, ORIGIN)
        self.assertNotEqual(ra.d_star, rb.d_star)
        self.assertFalse(np.array_equal(a, b))


# ============================================================================= O4 bypass
class BypassTests(unittest.TestCase):
    def test_state_windows_are_returned_bit_for_bit(self):
        load, types, F = windows(seed=11)
        pre, last, days = batch_inputs(load, types)
        state = np.zeros(len(F), bool)
        state[::3] = True
        for dtype in (np.float32, np.float64):
            Fd = (F * 50.0).astype(dtype)                                            # large anomalies: the layer would act
            Fp, _ = CT.apply_batch(Fd, pre, last, days, state)
            self.assertTrue(np.array_equal(Fp[state], Fd[state].astype(np.float64)))
            self.assertFalse(np.array_equal(Fp[~state], Fd[~state].astype(np.float64)))
        Fp, rec = CT.apply(F[0] * 50.0, load[0], types, ORIGIN, state=True)
        self.assertTrue(np.array_equal(Fp, F[0] * 50.0))
        self.assertEqual((rec.bypass, rec.x_fast, rec.x_day, rec.max_abs_delta), ("state window", 0.0, 0.0, 0.0))

    def test_an_unobserved_hour_among_the_last_three_leaves_the_trajectory_unchanged(self):
        load, types, F = windows(n=3, seed=12, error=0.6)
        for i, k in enumerate((1, 2, 3)):
            row = load[i].copy()
            row[ORIGIN - k] = np.nan
            Fp, rec = CT.apply(F[i], row, types, ORIGIN)
            self.assertTrue(np.array_equal(Fp, F[i]), k)
            self.assertTrue(rec.missing_fast and rec.missing_day)
            self.assertEqual((rec.bypass, rec.x_fast, rec.x_day), ("missing hours", 0.0, 0.0))

    def test_an_unobserved_hour_of_the_last_day_drops_only_the_day_anomaly(self):
        load, types, F = windows(n=1, seed=13, error=0.1)
        row = load[0].copy()
        row[ORIGIN - 10] = np.nan
        Fp, rec = CT.apply(F[0], row, types, ORIGIN)
        c = CT.CONSTANTS
        g1, g2 = CT.decay_curves(c)
        self.assertTrue(rec.missing_day and not rec.missing_fast)
        self.assertIsNone(rec.bypass)
        self.assertEqual(rec.x_day, 0.0)
        self.assertNotEqual(rec.x_fast, 0.0)
        self.assertTrue(np.array_equal(Fp, F[0] * np.exp((c.kappa1 * rec.x_fast) * g1 + (c.kappa2 * 0.0) * g2)))

    def test_an_older_unobserved_hour_changes_only_the_scale(self):
        load, types, F = windows(n=1, seed=14, error=0.1)
        row = load[0].copy()
        row[ORIGIN - 100] = np.nan
        _, rec = CT.apply(F[0], row, types, ORIGIN)
        self.assertFalse(rec.missing_fast or rec.missing_day)
        self.assertEqual(rec.scale_kw, max(float(np.nanmean(row[ORIGIN - N_PRE:ORIGIN])), 1e-3))
        self.assertEqual(rec.eps_kw, 0.01 * rec.scale_kw)


class ForecastBypassTests(unittest.TestCase):                                        # O4 through ankyra_lite.forecast
    def test_state_windows_bypass_the_layer_with_either_state_rule(self):
        h = synthetic_history(O)
        load = np.asarray(h.load_kw, dtype=np.float64).copy()
        load[-200:] = 1e-7 * (np.arange(200) % 3)                                    # switched off: an off-state window
        off_state = A.History(load, h.temperature_c, h.day_types, h.start_timestamp)
        F = {k: np.full(H, 5.0) for k in range(7)}                                   # far above the switched-off load
        for rule in ("seasonal_naive", None):
            on = lite(off_state, F, state_rule=rule, continuation=True)
            off = lite(off_state, F, state_rule=rule, continuation=False)
            self.assertTrue(on.state_window)
            self.assertTrue(np.array_equal(on.trajectory_kw, off.trajectory_kw), rule)
            self.assertEqual(on.continuation.bypass, "state window")
        self.assertTrue(np.array_equal(on.trajectory_kw, np.full(H, 5.0)))          # state_rule=None: the carrier's forecast

    def test_the_history_only_configuration_has_no_layer(self):
        h, _, _ = building()
        on, off = lite(h, None, continuation=True), lite(h, None, continuation=False)
        self.assertIsNone(on.continuation)
        self.assertTrue(np.array_equal(on.trajectory_kw, off.trajectory_kw))
        self.assertTrue(np.array_equal(on.trajectory_before_continuation_kw, on.trajectory_kw))


# ============================================================================= O5 degeneracy
class DegeneracyTests(EndToEndBase):
    def test_zero_constants_return_the_trajectory_bit_for_bit(self):
        self.assertEqual((CT.ZERO.kappa1, CT.ZERO.kappa2), (0.0, 0.0))
        for seed in (15, 16):
            load, types, F = windows(seed=seed, error=0.8)
            pre, last, days = batch_inputs(load, types)
            for dtype in (np.float32, np.float64):
                Fd = F.astype(dtype)
                Fp, _ = CT.apply_batch(Fd, pre, last, days, np.zeros(len(F), bool), constants=CT.ZERO)
                self.assertTrue(np.array_equal(Fp, Fd.astype(np.float64)), (seed, dtype))
                self.assertEqual(Fp.astype(dtype).tobytes(), Fd.tobytes())

    def test_with_the_layer_off_the_forecast_is_the_composition_of_the_released_steps(self):
        r = A.forecast(self.h, foundation=self.F, group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)
        self.assertFalse(r.off_state or r.micro_load)
        tr = np.asarray(r.trajectory_kw, dtype=np.float64)
        for off in (self.off, lite(self.h, self.F, continuation=None)):
            self.assertTrue(np.array_equal(off.trajectory_kw, tr))
            self.assertEqual(off.level_kw, float(r.level_kw))
            self.assertEqual(off.peak_kw, peak(tr, self.load, self.types))
            self.assertEqual((off.state_window, off.state_rule), (False, None))
            self.assertTrue(np.array_equal(off.trajectory_before_continuation_kw, off.trajectory_kw))
            self.assertIsNone(off.continuation)

    def test_zero_constants_through_forecast_give_the_layer_off_bit_for_bit(self):
        z = lite(self.h, self.F, continuation=CT.ZERO)
        self.assertTrue(np.array_equal(z.trajectory_kw, self.off.trajectory_kw))
        self.assertEqual((z.level_kw, z.peak_kw), (self.off.level_kw, self.off.peak_kw))
        self.assertEqual(z.continuation.max_abs_delta, 0.0)
        self.assertNotEqual(z.continuation.digest, CT.DIGEST)                        # a record of other constants


# ============================================================================= O6 readouts before the layer
class ReadoutTests(EndToEndBase):
    def test_energy_level_peak_and_blocks_are_read_before_the_layer(self):
        for on, off in ((self.on, self.off), (self.on_net, self.off_net)):
            self.assertFalse(np.array_equal(on.trajectory_kw, off.trajectory_kw))   # the layer acts on this window
            self.assertTrue(np.array_equal(on.trajectory_before_continuation_kw, off.trajectory_kw))
            self.assertEqual((on.level_kw, on.energy_kwh, on.peak_kw), (off.level_kw, off.energy_kwh, off.peak_kw))
            self.assertEqual(on.peak_kw, peak(on.trajectory_before_continuation_kw, self.load, self.types))
            self.assertTrue(np.array_equal(readouts.raw_trajectory(on), readouts.raw_trajectory(off)))
            self.assertEqual(readouts.projection_energy_increase(on), readouts.projection_energy_increase(off))
            for a, b in zip(readouts.blocks(readouts.raw_trajectory(on)), readouts.blocks(readouts.raw_trajectory(off))):
                self.assertTrue(np.array_equal(a, b))
            self.assertTrue(np.array_equal(on.daily_means_kw, off.daily_means_kw))  # the anchoring layer's own fields

    def test_the_delivered_energy_differs_from_the_readout_by_two_terms_known_at_the_origin(self):
        E = envelope()
        for on in (self.on, self.on_net):
            F, Fp = on.trajectory_before_continuation_kw, on.trajectory_kw
            change = readouts.continuation_energy_change(on)
            tol = 1e-9 * float(F.sum())
            self.assertAlmostEqual(change, float(Fp.sum() - F.sum()), delta=tol)
            self.assertAlmostEqual(float(Fp.sum()) - on.energy_kwh, change + readouts.projection_energy_increase(on), delta=tol)
            self.assertLessEqual(change, float((F * np.expm1(E)).sum()) + tol)
            self.assertGreaterEqual(change, float((F * np.expm1(-E)).sum()) - tol)
        self.assertEqual(readouts.continuation_energy_change(self.off), 0.0)

    def test_flat_trajectories_reach_the_energy_bounds(self):
        load, types, _ = windows(n=1, seed=17)
        E = envelope()
        up, down = float(np.expm1(E).mean()), float(np.expm1(-E).mean())
        self.assertTrue(0.0930 < up <= 0.0931 and -0.0734 <= down < -0.0733, (up, down))   # +9.31% / -7.34%, rounded outward
        for level, bound in ((1e-3, up), (1e3, down)):                               # far below / above the load: x = +1 / -1
            Fp, _ = CT.apply(np.full(H, level), load[0], types, ORIGIN)
            self.assertAlmostEqual(Fp.mean() / level - 1.0, bound, delta=1e-12)
        # where the flat bounds stop: a trajectory with its energy in the first hour moves by up to e^(+-E(0)) - 1
        self.assertTrue(1.2456 < math.expm1(E[0]) <= 1.246 and -0.555 <= math.expm1(-E[0]) < -0.5546)   # +124.6% / -55.5%
        _, last, days = CT.window_inputs(load[0], types, ORIGIN)
        d, _ = CT.reference_day(last, days)
        c = 1e3 * float(np.nanmax(load[0][ORIGIN - N_PRE:ORIGIN]))                   # far above the load
        up_F = np.zeros(H)
        up_F[0] = 0.744                                                              # all energy in hour 0, far below the load
        down_F = np.zeros(H)
        down_F[24 * d + 21:24 * d + 24] = c                                          # the reference hours far above the load
        down_F[0] += 1e6 * c                                                         # and almost all energy in hour 0
        for F, sign, bound, tol in ((up_F, 1.0, math.expm1(E[0]), 1e-12), (down_F, -1.0, math.expm1(-E[0]), 1e-5)):
            Fp, rec = CT.apply(F, load[0], types, ORIGIN)
            self.assertEqual((rec.x_fast, rec.x_day), (sign, sign))
            change = float(Fp.sum() / F.sum() - 1.0)
            self.assertAlmostEqual(change, bound, delta=tol)
            self.assertTrue(math.expm1(-E[0]) <= change <= math.expm1(E[0]))
            self.assertGreater(abs(change), max(up, -down))                           # beyond the flat-trajectory bounds

    def test_the_interval_is_placed_around_the_delivered_trajectory(self):
        R = A.readouts
        bands, n = readouts.interval(self.h, self.on, group=GROUP, temp_sigma_std=SIGMA)
        Q, n2 = R.residual_quantiles(self.load, self.h.temperature_c, np.asarray(self.h.day_types), self.h.start_timestamp,
                                     GROUP, SIGMA, O)
        hour, workday = (O + np.arange(H)) % 24, (self.types[O:O + H] < 5).astype(int)
        scale = R.origin_scale(self.load[O - CONTEXT:O])
        self.assertEqual(n, n2)
        self.assertGreater(n, 0)
        self.assertTrue(np.array_equal(bands, R.interval_bands(self.on.trajectory_kw, scale, Q, hour, workday), equal_nan=True))
        before = R.interval_bands(self.on.trajectory_before_continuation_kw, scale, Q, hour, workday)
        self.assertFalse(np.array_equal(bands, before, equal_nan=True))


# ============================================================================= O7 consistency and monotonicity
class ConsistencyTests(unittest.TestCase):
    def test_a_forecast_that_continues_the_last_observed_hours_is_unchanged(self):
        load, types, F = windows(n=6, seed=18)
        for i in range(len(F)):
            _, last, days = CT.window_inputs(load[i], types, ORIGIN)
            d, _ = CT.reference_day(last, days)
            row = load[i].copy()
            row[ORIGIN - 24:ORIGIN] = F[i][24 * d:24 * d + 24]                       # the last day equals the reference day
            Fp, rec = CT.apply(F[i], row, types, ORIGIN)
            self.assertEqual((rec.raw_fast, rec.raw_day, rec.x_fast, rec.x_day), (0.0, 0.0, 0.0, 0.0))
            self.assertTrue(np.array_equal(Fp, F[i]), i)

    def test_the_correction_is_nondecreasing_in_each_anomaly(self):
        grid = np.linspace(-1.0, 1.0, 41)
        for other in (-1.0, -0.3, 0.0, 0.6, 1.0):
            fast = np.stack([CT.delta(x, other, CT.CONSTANTS) for x in grid])
            day = np.stack([CT.delta(other, x, CT.CONSTANTS) for x in grid])
            self.assertTrue((np.diff(fast, axis=0) >= 0).all() and (np.diff(day, axis=0) >= 0).all(), other)

    def test_the_anomaly_is_nondecreasing_in_the_last_observed_load(self):
        load, types, F = windows(n=1, seed=19, error=0.1)
        xs = []
        for f in np.linspace(0.05, 20.0, 60):
            row = load[0].copy()
            row[ORIGIN - 3:ORIGIN] *= f
            xs.append(CT.apply(F[0], row, types, ORIGIN)[1].x_fast)
        self.assertTrue((np.diff(xs) >= 0).all())
        self.assertEqual((xs[0], xs[-1]), (-1.0, 1.0))                               # flat beyond the clip

    def test_same_sign_anomalies_move_every_hour_the_same_way(self):
        load, types, F = windows(seed=20, error=0.5)
        seen = {"up": 0, "down": 0}
        for i in range(len(F)):
            Fp, rec = CT.apply(F[i], load[i], types, ORIGIN)
            if rec.x_fast >= 0 and rec.x_day >= 0:
                self.assertTrue((Fp >= F[i]).all(), i)
                seen["up"] += 1
            elif rec.x_fast <= 0 and rec.x_day <= 0:
                self.assertTrue((Fp <= F[i]).all(), i)
                seen["down"] += 1
        self.assertTrue(seen["up"] > 0 and seen["down"] > 0, seen)

    def test_fixed_opposite_anomaly_example_changes_sign_once(self):
        # This selected example crosses once; opposite anomaly signs alone do not guarantee a crossing.
        D = CT.delta(1.0, -1.0, CT.CONSTANTS)
        self.assertGreater(D[0], 0.0)
        self.assertLess(D[-1], 0.0)
        self.assertEqual(int((np.diff(np.sign(D)) != 0).sum()), 1)

    def test_opposite_anomaly_example_can_remain_negative(self):
        D = CT.delta(0.1, -1.0, CT.CONSTANTS)
        self.assertTrue((D < 0.0).all())


# ============================================================================= O8 scale equivariance
class ScaleTests(unittest.TestCase):
    def test_power_of_two_scaling_is_bitwise_equivariant(self):
        load, types, F = windows(n=8, seed=21)
        for c in (2.0, 0.5, 8.0, 2.0 ** -6, 1024.0):
            for i in range(len(F)):
                a, ra = CT.apply(F[i], load[i], types, ORIGIN)
                b, rb = CT.apply(c * F[i], c * load[i], types, ORIGIN)
                self.assertGreater(min(ra.scale_kw, rb.scale_kw), 1e-3)
                self.assertEqual((ra.x_fast, ra.x_day), (rb.x_fast, rb.x_day))
                self.assertTrue(np.array_equal(b, c * a), (c, i))

    def test_other_factors_are_equivariant_to_rounding(self):
        load, types, F = windows(n=8, seed=22)
        for c in (3.7, 0.3):
            for i in range(len(F)):
                a, _ = CT.apply(F[i], load[i], types, ORIGIN)
                b, _ = CT.apply(c * F[i], c * load[i], types, ORIGIN)
                np.testing.assert_allclose(b, c * a, rtol=1e-12, atol=0)

    def test_the_scale_floor_breaks_equivariance(self):
        # counterexample: a load of about 1e-4 kW keeps its scale at the floor 1e-3 kW for both factors, so eps does not scale
        T = ORIGIN + H + 48
        row = 1e-4 * (1.0 + 0.2 * np.sin(2 * np.pi * np.arange(T) / 24))
        types = calendar(T)
        F = np.full(H, 1.5e-4)
        a, ra = CT.apply(F, row, types, ORIGIN)
        b, rb = CT.apply(8 * F, 8 * row, types, ORIGIN)
        self.assertEqual((ra.scale_kw, rb.scale_kw), (1e-3, 1e-3))
        self.assertFalse(ra.clipped_fast or ra.clipped_day or rb.clipped_fast or rb.clipped_day)
        self.assertFalse(np.allclose(b, 8 * a, rtol=1e-6, atol=0))


# ============================================================================= O9 reference day and identity
class ReferenceDayTests(unittest.TestCase):
    CASES = [((0, [1, 2, 3, 4, 5, 6, 0]), (0, True)), ((4, [5, 6, 0, 1, 2, 3, 4]), (2, True)),
             ((5, [6, 0, 1, 2, 3, 4, 5]), (0, True)), ((6, [0, 1, 2, 3, 4, 5, 6]), (5, True)),
             ((7, [0, 7, 1, 2, 3, 4, 5]), (1, True)), ((2, [7, 7, 7, 7, 7, 7, 7]), (6, False)),
             ((3, [7, 7, 7, 5, 6, 7, 7]), (6, False)), ((6, [0, 1, 2, 3, 4, 0, 1]), (6, False))]

    def test_workday_class_holidays_and_the_fallback(self):
        for (last, days), want in self.CASES:
            d, matched = CT.reference_day(last, days)
            self.assertEqual((int(d), bool(matched)), want, (last, days))

    def test_the_calendar_is_read_at_hour_12_of_each_origin_aligned_day(self):
        rng = np.random.default_rng(23)
        for o in (2000, 2003, 2011, 2016):
            types = rng.integers(0, 8, o + H + 48)
            load = rng.random(o + H + 48)
            pre, last, days = CT.window_inputs(load, types, o)
            self.assertEqual(int(last), int(types[o - 12]))
            self.assertEqual([int(x) for x in days], [int(types[o + 12 + 24 * d]) for d in range(7)])
            self.assertTrue(pre.dtype == np.float64 and np.array_equal(pre, load[o - N_PRE:o]))

    def test_the_anomalies_compare_the_same_clock_hours_and_one_reference_day(self):
        # the definition written out: the last 3 hours against hours 24 d* + 21 .. 23 (the same clock hours for any
        # origin hour), the last 24 hours against forecast day d*
        for o in range(ORIGIN, ORIGIN + 24, 5):
            load, types, F = windows(n=6, seed=24 + o, origin=o)
            for i in range(len(F)):
                pre, last, days = CT.window_inputs(load[i], types, o)
                d, _ = CT.reference_day(last, days)
                self.assertEqual([(o + 24 * d + 21 + j) % 24 for j in range(3)], [(o - 3 + j) % 24 for j in range(3)])
                s = max(float(np.nanmean(pre)), 1e-3)
                e = 0.01 * s
                xf = float(np.clip(math.log((pre[-3:].mean() + e) / (max(F[i][24 * d + 21:24 * d + 24].mean(), 0.0) + e)), -1, 1))
                xd = float(np.clip(math.log((pre[-24:].mean() + e) / (max(F[i][24 * d:24 * d + 24].mean(), 0.0) + e)), -1, 1))
                Fp, rec = CT.apply(F[i], load[i], types, o)
                self.assertEqual((rec.d_star, rec.x_fast, rec.x_day), (d, xf, xd))
                hh = np.arange(H, dtype=float)
                by_hand = F[i] * np.exp(0.479 * np.exp(-hh / 12) * xf + 0.330 * np.exp(-hh / 168) * xd)
                np.testing.assert_allclose(Fp, by_hand, rtol=1e-13, atol=0)

    def test_the_digest_identifies_the_frozen_constants_and_rule(self):
        text = json.dumps({"constants": FROZEN_CONSTANTS, "rule": FROZEN_RULE}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), FROZEN_DIGEST)
        self.assertEqual(CT.DIGEST, FROZEN_DIGEST)
        self.assertEqual(CT.digest(CT.CONSTANTS, CT.RULE), FROZEN_DIGEST)
        self.assertEqual(dict(CT.RULE), FROZEN_RULE)
        self.assertEqual(CT.CONSTANTS.as_dict(), FROZEN_CONSTANTS)
        for name in FROZEN_CONSTANTS:                                                # any change of a constant changes the identity
            changed = dict(FROZEN_CONSTANTS)
            changed[name] = changed[name] * (1 + 1e-12)
            self.assertNotEqual(CT.digest(CT.Constants(**changed), CT.RULE), FROZEN_DIGEST, name)
        rule = dict(FROZEN_RULE)
        rule["no_match_day"] = 0                                                     # the development code's fallback
        self.assertNotEqual(CT.digest(CT.CONSTANTS, rule), FROZEN_DIGEST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
