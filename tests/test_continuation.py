"""Tests of the origin-continuity layer (``ankyra_lite.continuation``) on synthetic arrays: the frozen constants and
their digest, the layer switched off by zero constants, sign and exact zeros, the bounds of Delta, causality, the
state and missing-hour bypasses, the rule for negative loads (x = 0, flagged), the formula and its order of
operations, consistency and monotonicity, scale equivariance, the reference day, the one-window and batch entry
points, and bitwise equality with the study's reference implementation on 200 random windows.

Only numpy is needed.  The equality check imports the study's reference implementation
``tools_lite_v2_continuation_v1.py`` read-only by path (``ANKYRA_LITE_RESEARCH_ROOT``, or a sibling folder ``HKAD_V3``
of the repository) and is skipped when it is absent.  Run from the repository root:
python -m unittest tests.test_continuation -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ankyra_lite import continuation as C                                            # noqa: E402

H = 744
K1, R1, K2, R2 = 0.479, 12.0, 0.330, 168.0
RESEARCH_TOOL = "tools_lite_v2_continuation_v1.py"
RESEARCH_TOOL_SHA256 = "427ac973794f37887304c4744cfc5feeb8d86d221b8ea2ef75539560e197d668"   # as frozen with the constants
FROZEN_RULE = {"fast_hours": 3, "day_hours": 24, "scale_hours": 672, "scale_floor": 0.001, "eps_fraction": 0.01, "clip": 1.0,
               "workday_type_below": 5, "type_hour": 12, "reference_days": 7, "no_match_day": 6, "horizon": 744,
               "bypass": "state windows (off_state or micro_load) and missing last hours (x = 0)"}
INFO_KEYS = ("xf", "xd", "raw_f", "raw_d", "s", "missing_f", "missing_d", "clipped_f", "clipped_d", "matched", "bypass_state", "d_star")


def rows(rng, n=40, o=2000, zeros=True):
    """n load rows (n, o + 744 + 48) >= 0 with a daily shape and 1% unobserved hours (NaN), a shared calendar
    (weekday 0..6, every 37th day a holiday 7) and anchored trajectories F >= 0 with exact zeros (row 0 all zero)."""
    T = o + H + 48
    t = np.arange(T)
    load = (5 + 3 * np.sin(2 * np.pi * t / 24)[None] + rng.gamma(2.0, 1.0, (n, T))) * rng.uniform(0.2, 5, (n, 1))
    load[rng.random((n, T)) < 0.01] = np.nan
    types = ((t // 24) % 7).astype(np.int64)
    types[(t // 24) % 37 == 5] = 7
    F = np.nan_to_num(np.maximum(load[:, o:o + H] * rng.lognormal(0, 0.4, (n, H)), 0.0), nan=1.0)
    if zeros:
        F[rng.random((n, H)) < 0.05] = 0.0
        F[0] = 0.0
    return load, types, F


def inputs(load, types, o):
    """pre (n, 672), last_type (n,), day_types (n, 7) of every row at the origin o."""
    w = [C.window_inputs(row, types, o) for row in load]
    return np.stack([a for a, _, _ in w]), np.array([b for _, b, _ in w]), np.array([c for _, _, c in w])


def clean_window(rng, o=2000, scale=10.0):
    """One completely observed row, its calendar and a positive trajectory whose anomalies are not clipped."""
    T = o + H + 48
    t = np.arange(T)
    load = scale * (1 + 0.3 * np.sin(2 * np.pi * t / 24) + 0.1 * rng.random(T))
    types = ((t // 24) % 7).astype(np.int64)
    F = load[o:o + H] * rng.uniform(0.8, 1.25, H)
    return load, types, F


class TestConstantsAndDigest(unittest.TestCase):
    def test_constants_rule_and_digest_are_the_frozen_ones(self):
        self.assertEqual(C.CONSTANTS.as_dict(), {"kappa1": K1, "rho1": R1, "kappa2": K2, "rho2": R2})
        self.assertEqual(C.ZERO.as_dict(), {"kappa1": 0.0, "rho1": R1, "kappa2": 0.0, "rho2": R2})
        self.assertEqual(C.FULL_PRECISION.as_dict(), {"kappa1": 0.4789714953611109, "rho1": R1, "kappa2": 0.3303470261128675, "rho2": R2})
        self.assertEqual(C.RULE, FROZEN_RULE)
        self.assertEqual(C.DIGEST, "d514e3ec93a745b14950671297bf2f922fd512424a804e292377e1b0c4cc1dce")
        self.assertEqual(C.digest(), C.DIGEST)
        text = json.dumps({"constants": {"kappa1": K1, "rho1": R1, "kappa2": K2, "rho2": R2}, "rule": FROZEN_RULE}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), C.DIGEST)        # the canonical JSON, written out
        self.assertEqual(C.digest(C.Constants(0.479, 12, 0.33, 168)), C.DIGEST)      # integers are stored as floats
        self.assertEqual(C.digest(dict(kappa1=K1, rho1=R1, kappa2=K2, rho2=R2)), C.DIGEST)
        self.assertNotEqual(C.digest(C.Constants(0.4790000001, R1, K2, R2)), C.DIGEST)
        self.assertNotEqual(C.digest(C.ZERO), C.DIGEST)
        rule = dict(C.RULE)
        rule["no_match_day"] = 0
        self.assertNotEqual(C.digest(C.CONSTANTS, rule), C.DIGEST)

    def test_constants_are_validated(self):
        for bad in ((-0.1, R1, K2, R2), (K1, 0.0, K2, R2), (K1, R1, K2, -1.0), (math.nan, R1, K2, R2), (K1, R1, math.inf, R2)):
            with self.assertRaises(ValueError, msg=str(bad)):
                C.Constants(*bad)
        with self.assertRaises(TypeError):
            C.apply_batch(np.ones((1, H)), np.ones((1, 672)), [0], [[0] * 7], [False], constants=(K1, R1, K2, R2))


class TestZeroConstants(unittest.TestCase):
    def test_zero_constants_leave_the_trajectory_unchanged_bit_for_bit(self):
        rng = np.random.default_rng(1)
        load, types, F = rows(rng)
        pre, lt, dt = inputs(load, types, 2000)
        st = np.zeros(len(F), bool)
        st[[3, 7]] = True
        for dtype in (np.float32, np.float64):
            Fd = F.astype(dtype)
            Fp, info = C.apply_batch(Fd, pre, lt, dt, st, C.ZERO, return_delta=True)
            self.assertEqual(Fp.dtype, np.float64)
            self.assertEqual(Fp.astype(dtype).tobytes(), Fd.tobytes(), dtype)
            self.assertEqual(Fp.tobytes(), Fd.astype(np.float64).tobytes(), dtype)
            self.assertTrue((info["delta"] == 0).all())                                # +0 or -0 at every hour
            self.assertEqual(info["max_abs_delta"], 0.0)
        Fq, rec = C.apply(F[5], load[5], types, 2000, constants=C.ZERO)
        self.assertEqual(Fq.tobytes(), F[5].tobytes())
        self.assertEqual(rec.constants, C.ZERO)
        self.assertEqual(rec.digest, C.digest(C.ZERO))


class TestSignAndZeros(unittest.TestCase):
    def test_nonnegative_finite_and_exact_zeros_kept(self):
        rng = np.random.default_rng(2)
        load, types, F = rows(rng)
        pre, lt, dt = inputs(load, types, 2000)
        st = np.zeros(len(F), bool)
        Fp, _ = C.apply_batch(F, pre, lt, dt, st)
        self.assertTrue(np.isfinite(Fp).all() and (Fp >= 0).all())
        self.assertTrue(np.array_equal(F == 0, Fp == 0))                                # zeros in = zeros out, at the same hours
        self.assertGreater(int((F == 0).sum()), 0)
        self.assertTrue(np.array_equal(np.sign(Fp), np.sign(F)))

    def test_normal_floats_stay_positive_subnormal_floats_may_underflow(self):
        tiny = np.finfo(np.float64).tiny                                               # the smallest positive normal float
        self.assertGreater(tiny * math.exp(-0.809), 0.0)
        self.assertEqual(5e-324 * math.exp(-0.809), 0.0)                               # the documented exception
        load, types, _ = clean_window(np.random.default_rng(3))
        pre, lt, dt = C.window_inputs(load, types, 2000)
        d, _ = C.reference_day(lt, dt)
        F = np.full(H, tiny)
        F[24 * d:24 * d + 24] = 1e6                                                    # reference day far above the load: both anomalies -1
        Fq, rec = C.apply(F, load, types, 2000)
        self.assertEqual((rec.x_fast, rec.x_day), (-1.0, -1.0))
        self.assertTrue((Fq > 0).all())


class TestBounds(unittest.TestCase):
    def test_envelope_values(self):
        E = C.envelope()
        g1, g2 = C.decay_curves()
        self.assertEqual(E.tobytes(), (K1 * g1 + K2 * g2).tobytes())
        self.assertEqual(float(E[0]), K1 + K2)
        self.assertLessEqual(float(E[0]), 0.809)
        self.assertTrue((np.diff(E) < 0).all())                                        # decreasing in h
        self.assertGreater(float(E[167]), 0.12)                                        # the last hour of day 7: 0.1221
        self.assertTrue(0.1214 < float(E[168]) < 0.1215)                               # 0.121400614
        self.assertLess(float(E[743]), 0.004)
        self.assertAlmostEqual(math.exp(-E[0]), 0.4453031467923588, places=12)
        self.assertAlmostEqual(math.exp(E[0]), 2.2456612022692304, places=12)
        self.assertTrue(0.8856 < math.exp(-E[168]) and math.exp(E[168]) < 1.1291)
        self.assertTrue(0.99604 < math.exp(-E[743]) and math.exp(E[743]) < 1.00397)
        self.assertAlmostEqual(float(np.mean(np.exp(E) - 1)), 0.0930353, places=6)    # flat F: energy change in [-7.34%, +9.31%]
        self.assertAlmostEqual(float(np.mean(np.exp(-E) - 1)), -0.0733486, places=6)

    def test_delta_within_the_envelope_and_reached_when_both_anomalies_clip(self):
        rng = np.random.default_rng(4)
        load, types, F = rows(rng)
        pre, lt, dt = inputs(load, types, 2000)
        F = F + 0.5                                                                    # positive everywhere: the ratio F'/F is defined
        st = np.zeros(len(F), bool)
        Fp, info = C.apply_batch(F, pre, lt, dt, st, return_delta=True)
        E = C.envelope()
        D = info["delta"]
        self.assertTrue((np.abs(D) <= E[None]).all())                                  # exact in floating point (monotone rounding)
        ratio = Fp / F
        self.assertTrue((ratio >= np.exp(-E)[None] * (1 - 1e-15)).all() and (ratio <= np.exp(E)[None] * (1 + 1e-15)).all())
        self.assertLessEqual(float(np.abs(D[:, 168:]).max()), 0.1215)
        hi = np.stack([np.full(672, 1e6), np.full(672, 1e-9)])                         # both clipped to +1, both to -1
        Fp2, inf2 = C.apply_batch(np.ones((2, H)), hi, lt[:2], dt[:2], [False, False], return_delta=True)
        self.assertTrue(inf2["clipped_f"].all() and inf2["clipped_d"].all())
        self.assertEqual(list(inf2["xf"]), [1.0, -1.0])
        self.assertEqual(list(inf2["xd"]), [1.0, -1.0])
        self.assertEqual(inf2["delta"][0].tobytes(), E.tobytes())
        self.assertEqual(inf2["delta"][1].tobytes(), (-E).tobytes())
        self.assertEqual(inf2["max_abs_delta"], float(E[0]))


class TestCausality(unittest.TestCase):
    def test_only_the_672_pre_origin_hours_and_the_reference_calendar_are_read(self):
        rng = np.random.default_rng(5)
        o = 2000
        load, types, F = rows(rng, n=30)
        read = np.zeros(len(types), bool)
        read[o - 12] = True
        read[o + 12 + 24 * np.arange(7)] = True
        changed_when_used, n_used = True, 0
        for i in range(1, len(F)):
            a, ra = C.apply(F[i], load[i], types, o)
            L2 = load[i].copy()
            L2[o:] = rng.normal(0, 1e6, len(L2) - o)
            L2[o + 5] = np.nan
            L2[:o - 672] = -7.0
            T2 = types.copy()
            T2[~read] = rng.integers(0, 8, int((~read).sum()))
            b, _ = C.apply(F[i], L2, T2, o)
            self.assertEqual(a.tobytes(), b.tobytes(), i)
            L3 = load[i].copy()
            L3[o - 1] = (0.0 if not np.isfinite(L3[o - 1]) else L3[o - 1]) * 3 + 50.0
            c, rc = C.apply(F[i], L3, types, o)
            if not any(getattr(r, k) for r in (ra, rc) for k in ("missing_fast", "missing_day", "clipped_fast", "clipped_day")):
                changed_when_used &= not np.array_equal(a, c)                          # sanity: the last observed hour is read
                n_used += 1
        self.assertTrue(changed_when_used)
        self.assertGreaterEqual(n_used, 10)
        T4 = types.copy()
        T4[o + 12] = 7 if types[o + 12] < 5 else 0                                     # a read calendar entry does matter
        self.assertNotEqual(C.window_inputs(load[1], types, o)[2], C.window_inputs(load[1], T4, o)[2])


class TestBypass(unittest.TestCase):
    def test_state_windows_pass_through_bit_for_bit(self):
        rng = np.random.default_rng(6)
        load, types, F = rows(rng)
        pre, lt, dt = inputs(load, types, 2000)
        pre[:] = 1e6                                                                   # large anomalies everywhere
        st = np.zeros(len(F), bool)
        st[[2, 9]] = True
        F[9, :5] = np.nan                                                              # a state window may carry a non-finite trajectory
        Fp, info = C.apply_batch(F, pre, lt, dt, st)
        self.assertEqual(Fp[st].tobytes(), F[st].tobytes())
        self.assertTrue(info["bypass_state"][st].all() and not info["bypass_state"][~st].any())
        self.assertTrue((info["xf"][st] == 0).all() and (info["xd"][st] == 0).all() and np.isnan(info["raw_f"][st]).all())
        self.assertFalse(np.array_equal(Fp[~st], F[~st]))
        Fq, rec = C.apply(F[2], load[2], types, 2000, state=True)
        self.assertEqual(Fq.tobytes(), F[2].tobytes())
        self.assertEqual(rec.bypass, "state window")
        self.assertEqual((rec.d_star, rec.matched, rec.scale_kw, rec.eps_kw, rec.max_abs_delta), (None, None, None, None, 0.0))
        self.assertTrue((rec.delta == 0).all())
        F[4, 100] = np.inf
        with self.assertRaises(ValueError):                                            # but an ordinary window may not
            C.apply_batch(F, pre, lt, dt, st)

    def test_unobserved_last_hours(self):
        rng = np.random.default_rng(7)
        load, types, F = clean_window(rng)
        o = 2000
        g1, g2 = C.decay_curves()
        L = load.copy()
        L[o - 2] = np.nan                                                              # in the last 3 hours: x_f = x_d = 0
        Fq, rec = C.apply(F, L, types, o)
        self.assertEqual(Fq.tobytes(), F.tobytes())
        self.assertEqual((rec.bypass, rec.x_fast, rec.x_day, rec.missing_fast, rec.missing_day), ("missing hours", 0.0, 0.0, True, True))
        L = load.copy()
        L[o - 10] = np.nan                                                             # in the last 24 hours only: x_d = 0
        Fq, rec = C.apply(F, L, types, o)
        self.assertIsNone(rec.bypass)
        self.assertTrue(rec.missing_day and not rec.missing_fast and rec.x_day == 0.0 and rec.x_fast != 0.0)
        self.assertEqual(Fq.tobytes(), (F * np.exp((K1 * rec.x_fast) * g1 + (K2 * 0.0) * g2)).tobytes())
        L = load.copy()
        L[o - 100] = np.nan                                                            # an older hour: both anomalies, scale on observed hours
        Fq, rec = C.apply(F, L, types, o)
        self.assertFalse(rec.missing_fast or rec.missing_day)
        self.assertEqual(rec.scale_kw, float(np.nanmean(L[o - 672:o])))
        L = load.copy()
        L[o - 672:o] = np.nan                                                          # nothing observed: scale at its floor, x = 0
        Fq, rec = C.apply(F, L, types, o)
        self.assertEqual((rec.scale_kw, rec.eps_kw, rec.bypass), (1e-3, 0.01 * 1e-3, "missing hours"))
        self.assertEqual(Fq.tobytes(), F.tobytes())


class TestFormula(unittest.TestCase):
    def test_formula_written_out(self):
        rng = np.random.default_rng(8)
        load, types, F = rows(rng)
        o = 2000
        hh = np.arange(H, dtype=float)
        checked = 0
        for i in range(1, len(F)):
            pre, lt, dt = C.window_inputs(load[i], types, o)
            if not np.isfinite(pre[-24:]).all():
                continue
            d, _ = C.reference_day(lt, dt)
            s = max(float(np.nanmean(pre)), 1e-3)
            e = 0.01 * s
            xf = float(np.clip(math.log((np.mean(pre[-3:]) + e) / (max(np.mean(F[i][24 * d + 21:24 * d + 24]), 0) + e)), -1, 1))
            xd = float(np.clip(math.log((np.mean(pre[-24:]) + e) / (max(np.mean(F[i][24 * d:24 * d + 24]), 0) + e)), -1, 1))
            Fq, rec = C.apply(F[i], load[i], types, o)
            self.assertEqual((rec.x_fast, rec.x_day, rec.d_star, rec.scale_kw), (xf, xd, d, s))
            man = F[i] * np.exp(0.479 * np.exp(-hh / 12) * xf + 0.330 * np.exp(-hh / 168) * xd)
            self.assertTrue(np.allclose(Fq, man, rtol=1e-13, atol=0))
            ordered = F[i] * np.exp((0.479 * xf) * np.exp(-hh / 12.0) + (0.330 * xd) * np.exp(-hh / 168.0))
            self.assertEqual(Fq.tobytes(), ordered.tobytes())                         # the documented order of operations
            self.assertEqual(rec.delta.tobytes(), ((0.479 * xf) * np.exp(-hh / 12.0) + (0.330 * xd) * np.exp(-hh / 168.0)).tobytes())
            self.assertEqual(rec.max_abs_delta, float(np.abs(rec.delta).max()))
            checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_no_correction_when_the_reference_day_matches_the_last_observed_hours(self):
        rng = np.random.default_rng(9)
        load, types, F = clean_window(rng)
        o = 2000
        pre, lt, dt = C.window_inputs(load, types, o)
        d, _ = C.reference_day(lt, dt)
        F = F.copy()
        F[24 * d:24 * d + 24] = pre[-24:]                                              # the same values in the same order
        Fq, rec = C.apply(F, load, types, o)
        self.assertEqual((rec.raw_fast, rec.raw_day, rec.x_fast, rec.x_day), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(Fq.tobytes(), F.tobytes())

    def test_monotone_in_each_anomaly(self):
        x = np.linspace(-1.0, 1.0, 41)
        Df = C.delta(x, np.zeros_like(x))
        Dd = C.delta(np.zeros_like(x), x)
        self.assertTrue((np.diff(Df, axis=0) >= 0).all() and (np.diff(Dd, axis=0) >= 0).all())
        for xf, xd in ((0.3, 0.7), (1.0, 0.0), (0.0, 0.2)):
            self.assertTrue((C.delta(xf, xd) >= 0).all() and (C.delta(-xf, -xd) <= 0).all())
        self.assertEqual(C.delta(0.3, -0.2).tobytes(), C.delta(np.array([0.3]), np.array([-0.2]))[0].tobytes())   # scalar = batch path


class TestScale(unittest.TestCase):
    def test_scale_equivariance(self):
        rng = np.random.default_rng(10)
        load, types, F = rows(rng, n=12, zeros=False)
        o = 2000
        for i in range(len(F)):
            a, ra = C.apply(F[i], load[i], types, o)
            for c in (2.0 ** -5, 2.0 ** 7, 2.0 ** 20):                                 # powers of two: bit for bit
                b, rb = C.apply(c * F[i], c * load[i], types, o)
                self.assertGreaterEqual(rb.scale_kw, 1e-3)
                self.assertEqual(b.tobytes(), (c * a).tobytes(), (i, c))
                self.assertEqual((rb.x_fast, rb.x_day), (ra.x_fast, ra.x_day))
            b, _ = C.apply(3.7 * F[i], 3.7 * load[i], types, o)
            self.assertTrue(np.allclose(b, 3.7 * a, rtol=1e-12, atol=0))

    def test_the_scale_floor_breaks_equivariance(self):
        o = 2000
        types = ((np.arange(o + H + 48) // 24) % 7).astype(np.int64)
        load = np.full(o + H + 48, 1e-5)                                               # s at its 1e-3 floor
        F = np.full(H, 2e-5)
        a, ra = C.apply(F, load, types, o)
        b, rb = C.apply(1024 * F, 1024 * load, types, o)
        self.assertEqual(ra.scale_kw, 1e-3)
        self.assertNotEqual(ra.x_day, rb.x_day)
        self.assertFalse(np.allclose(b, 1024 * a, rtol=1e-6))


class TestReferenceDay(unittest.TestCase):
    def test_class_rule_holidays_and_fallback(self):
        cases = [((0, [1, 2, 3, 4, 5, 6, 0]), (0, True)), ((4, [5, 6, 0, 1, 2, 3, 4]), (2, True)), ((5, [6, 0, 1, 2, 3, 4, 5]), (0, True)),
                 ((6, [0, 1, 2, 3, 4, 5, 6]), (5, True)), ((7, [0, 7, 1, 2, 3, 4, 5]), (1, True)), ((2, [7, 7, 7, 7, 7, 7, 7]), (6, False)),
                 ((3, [7, 7, 7, 5, 6, 7, 7]), (6, False)), ((6, [0, 1, 2, 3, 4, 0, 1]), (6, False))]
        for (last, days), want in cases:
            self.assertEqual(C.reference_day(last, days), want, (last, days))

    def test_window_inputs_positions_and_clock_hours(self):
        o = 3000
        load = np.arange(o + 900, dtype=np.float32)
        idx = np.arange(o + 900)                                                       # a "calendar" holding its own index
        pre, last, days = C.window_inputs(load, idx, o)
        self.assertEqual(pre.dtype, np.float64)
        self.assertTrue(np.array_equal(pre, np.arange(o - 672, o, dtype=np.float64)))
        self.assertEqual(last, o - 12)
        self.assertEqual(days, tuple(o + 12 + 24 * d for d in range(7)))
        for oh in range(24):                                                           # any origin hour
            for d in range(7):
                self.assertEqual([(oh + 24 * d + 21 + j) % 24 for j in range(3)], [(oh - 3 + j) % 24 for j in range(3)])
        with self.assertRaises(ValueError):
            C.window_inputs(load, idx, 671)

    def test_reference_day_depends_on_the_calendar_only(self):
        rng = np.random.default_rng(11)
        load, types, F = rows(rng, n=8)
        ds = {C.apply(F[i], load[i], types, 2000)[1].d_star for i in range(1, 8)}
        self.assertEqual(len(ds), 1)


class TestEntryPoints(unittest.TestCase):
    def test_batch_equals_the_one_window_calls(self):
        rng = np.random.default_rng(12)
        load, types, F = rows(rng)
        o = 2000
        pre, lt, dt = inputs(load, types, o)
        st = np.zeros(len(F), bool)
        st[[3, 7]] = True
        Fp, info = C.apply_batch(F, pre, lt, dt, st)
        for i in range(len(F)):
            Fq, rec = C.apply(F[i], load[i], types, o, state=bool(st[i]))
            self.assertEqual(Fq.tobytes(), Fp[i].tobytes(), i)
            self.assertEqual(rec.x_fast, info["xf"][i])
            self.assertEqual(rec.x_day, info["xd"][i])
            self.assertEqual(rec.bypass, "state window" if st[i] else ("missing hours" if info["missing_f"][i] else None))
            self.assertEqual(rec.digest, C.DIGEST)
            if not st[i]:
                self.assertEqual((rec.d_star, rec.matched, rec.scale_kw), (int(info["d_star"][i]), bool(info["matched"][i]), float(info["s"][i])))
        Fq, _ = C.apply(F[5], load[5][:o], types, None)                                # the origin defaults to len(load_kw)
        self.assertEqual(Fq.tobytes(), Fp[5].tobytes())
        self.assertEqual(Fp.shape, (len(F), H))
        Fe, ie = C.apply_batch(np.zeros((0, H)), np.zeros((0, 672)), [], np.zeros((0, 7)), np.zeros(0, bool))
        self.assertEqual((Fe.shape, ie["max_abs_delta"]), ((0, H), 0.0))

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            C.apply_batch(np.ones((2, 743)), np.ones((2, 672)), [0, 0], [[0] * 7] * 2, [False, False])
        with self.assertRaises(ValueError):
            C.apply_batch(np.ones((2, H)), np.ones((2, 672)), [0, 0], [[0] * 7] * 2, [False])
        with self.assertRaises(ValueError):
            C.anomalies(np.ones(671), np.ones(H), 0, [0] * 7)

    def test_negative_loads(self):
        # mean + e <= 0 (negative loads, e.g. net metering): no logarithm; the anomaly is set to 0 and flagged, as for
        # an unobserved span (a release rule; the study's reference raised an error there)
        rng = np.random.default_rng(13)
        load, types, F = clean_window(rng)
        o = 2000
        F64 = F.astype(np.float64)
        L = load.copy()
        L[o - 3:o] = -5.0                                                              # fast span only: the day mean stays positive
        Fq, rec = C.apply(F, L, types, o)
        self.assertEqual((rec.x_fast, rec.nonpositive_fast, rec.nonpositive_day, rec.missing_fast), (0.0, True, False, False))
        self.assertTrue(math.isnan(rec.raw_fast) and rec.x_day != 0.0 and not rec.clipped_fast)
        self.assertIsNone(rec.bypass)
        a = C.anomalies(L[o - 672:o], F, *C.window_inputs(L, types, o)[1:])
        D = C.delta(0.0, a["xd"])
        self.assertEqual(Fq.tobytes(), (F64 * np.exp(D)).tobytes())                    # x_f = 0 in the same formula
        L[o - 24:o] = -5.0                                                             # both spans: F' = F bit for bit
        Fq, rec = C.apply(F, L, types, o)
        self.assertEqual(Fq.tobytes(), F64.tobytes())
        self.assertEqual((rec.x_fast, rec.x_day, rec.nonpositive_fast, rec.nonpositive_day, rec.bypass),
                         (0.0, 0.0, True, True, "nonpositive load"))
        self.assertEqual(rec.max_abs_delta, 0.0)
        L2 = load.copy()                                                               # the day span alone, with a positive fast span
        L2[o - 24:o - 3] = -50.0
        Fq, rec = C.apply(F, L2, types, o)
        self.assertEqual((rec.nonpositive_fast, rec.nonpositive_day, rec.x_day), (False, True, 0.0))
        self.assertNotEqual(rec.x_fast, 0.0)
        self.assertIsNone(rec.bypass)
        L3 = load.copy()                                                               # unobserved day span, nonpositive fast span
        L3[o - 10] = np.nan
        L3[o - 3:o] = -5.0
        Fq, rec = C.apply(F, L3, types, o)
        self.assertEqual((rec.missing_day, rec.nonpositive_fast, rec.bypass), (True, True, "nonpositive load"))
        self.assertEqual(Fq.tobytes(), F64.tobytes())
        pre4 = np.full(672, 2.0)                                                       # the boundary: last-3 mean close to -e
        for _ in range(6):                                                             # (s depends on those hours; iterate)
            s4 = float(np.mean(pre4))
            pre4[-3:] = -(0.01 * s4)
        a4 = C.anomalies(pre4, F, 0, [0] * 7)
        self.assertEqual(float(pre4[-3:].mean()) + a4["eps"] <= 0.0, a4["nonpositive_f"])
        L[o - 24:o] = -5.0
        Fz, rz = C.apply(F, L, types, o, constants=C.ZERO)                            # ZERO: F' = F, no error
        self.assertEqual(Fz.tobytes(), F64.tobytes())
        self.assertTrue(rz.nonpositive_fast and rz.nonpositive_day)
        Fs, rs = C.apply(F, L, types, o, state=True, constants=C.ZERO)                # a state window is never read
        self.assertEqual(Fs.tobytes(), F64.tobytes())
        self.assertEqual((rs.bypass, rs.nonpositive_fast), ("state window", False))
        pre, lt, dt = C.window_inputs(L, types, o)                                    # the batch info carries the flags
        Fb, info = C.apply_batch(np.stack([F, F]), np.stack([pre, load[o - 672:o]]), [lt, lt], [dt, dt], [False, False])
        self.assertEqual((info["nonpositive_f"].tolist(), info["nonpositive_d"].tolist()), ([True, False], [True, False]))
        self.assertEqual(Fb[0].tobytes(), F64.tobytes())
        L[o - 3:o] = -0.001                                                            # mean + e > 0: defined and used
        L[o - 24:o - 3] = load[o - 24:o - 3]
        Fq, rec = C.apply(F, L, types, o)
        self.assertTrue(np.isfinite(Fq).all() and rec.x_fast < 0 and not rec.nonpositive_fast)


def research_tool():
    """(path of the study's reference implementation, or None): ``ANKYRA_LITE_RESEARCH_ROOT`` (the research folder of
    the other tests; ``ANKYRA_LITE_RESEARCH_DIR`` is read too), else the sibling folder HKAD_V3 of the repository."""
    env = os.environ.get("ANKYRA_LITE_RESEARCH_ROOT") or os.environ.get("ANKYRA_LITE_RESEARCH_DIR")
    roots = [Path(env)] if env else [REPO.parent / "HKAD_V3", REPO.parents[2]]
    for r in roots:
        if (r / RESEARCH_TOOL).is_file():
            return r / RESEARCH_TOOL
    return None


def import_read_only(path):
    """Import a study module by path without leaving its folder on sys.path."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_study_continuation_reference", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def random_windows(rng, n=200):
    """n windows of different shapes: (F (n, 744), pre (n, 672), last_type (n,), day_types (n, 7), state (n,), rows)
    with scales 1e-5..1e4 kW, unobserved hours in and before the last 24 hours, all-zero loads and trajectories,
    clipped anomalies, holidays, both no-match cases, and state windows (some with a non-finite trajectory)."""
    Fs, rows_, origins, cal = [], [], [], []
    state = np.zeros(n, bool)
    for i in range(n):
        o = int(rng.integers(700, 2500))
        T = o + H + 24
        t = np.arange(T)
        scale = 10.0 ** rng.uniform(-5, 4)
        y = scale * (1 + 0.5 * np.sin(2 * np.pi * (t + rng.integers(24)) / 24) + rng.gamma(2.0, 0.3, T))
        ty = ((t // 24 + rng.integers(7)) % 7).astype(np.int64)
        ty[(t // 24) % int(rng.integers(5, 40)) == 0] = 7
        kind = i % 10
        if kind == 1:
            y[o - int(rng.integers(1, 4))] = np.nan
        elif kind == 2:
            y[o - int(rng.integers(4, 25))] = np.nan
        elif kind == 3:
            y[rng.random(T) < 0.05] = np.nan
        elif kind == 4:
            y[:] = 0.0
        elif kind == 5:
            y[o - 672:o] = np.nan
        elif kind == 6:
            ty[o - 24:o] = int(rng.integers(0, 5))                                     # a workday, then a week without one
            ty[o:o + 168] = rng.choice([5, 6, 7], 168)
        elif kind == 7:
            ty[o - 24:o] = int(rng.choice([5, 6, 7]))                                  # a non-workday, then only workdays
            ty[o:o + 168] = rng.integers(0, 5, 168)
        F = np.nan_to_num(np.abs(y[o:o + H]), nan=scale) * rng.lognormal(0, 0.5, H)
        F[rng.random(H) < 0.05] = 0.0
        if kind == 8:
            F *= 10.0 ** rng.choice([-3, 3])                                           # clipped anomalies
        if kind == 9:
            F[:] = 0.0
        if rng.random() < 0.1:
            state[i] = True
            if rng.random() < 0.5:
                F[:5] = np.nan
        Fs.append(F)
        rows_.append(y)
        origins.append(o)
        cal.append(ty)
    return np.stack(Fs), state, rows_, cal, origins


class TestEqualsTheStudyReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = research_tool()
        if path is None:
            raise unittest.SkipTest(f"{RESEARCH_TOOL} not found (set ANKYRA_LITE_RESEARCH_ROOT)")
        cls.sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cls.R = import_read_only(path)
        cls.path = path

    def test_the_reference_is_the_frozen_tool_with_the_same_constants(self):
        self.assertEqual(self.sha, RESEARCH_TOOL_SHA256)
        self.assertEqual(self.R.CONSTANTS, C.CONSTANTS.as_dict())
        self.assertEqual(self.R.ZERO, C.ZERO.as_dict())
        self.assertEqual(self.R.CONSTANTS_FIT_FULL_PRECISION, C.FULL_PRECISION.as_dict())
        self.assertEqual(self.R.RULE, C.RULE)
        self.assertEqual(self.R.digest(), C.DIGEST)
        frozen = Path(self.R.C.FROZEN_PATH)
        if frozen.is_file():
            fz = json.loads(frozen.read_text(encoding="utf-8"))["continuation"]
            self.assertEqual((fz["constants"], fz["rule"], fz["digest"]), (C.CONSTANTS.as_dict(), C.RULE, C.DIGEST))

    def test_200_random_windows_bit_for_bit(self):
        R = self.R
        rng = np.random.default_rng(20261020)
        F, st, rows_, cal, origins = random_windows(rng)
        n = len(F)
        w_mine = [C.window_inputs(rows_[i], cal[i], origins[i]) for i in range(n)]
        w_ref = [R.window_inputs(rows_[i], cal[i], origins[i]) for i in range(n)]
        for i, (a, b) in enumerate(zip(w_mine, w_ref)):
            self.assertEqual(a[0].tobytes(), b[0].tobytes(), i)
            self.assertEqual((a[1], list(a[2])), (b[1], list(b[2])), i)
        pre = np.stack([a for a, _, _ in w_mine])
        lt = np.array([b for _, b, _ in w_mine])
        dt = np.array([c for _, _, c in w_mine])
        self.assertGreater(int((~np.array([C.reference_day(lt[i], dt[i])[1] for i in range(n)])).sum()), 0)   # no-match cases present
        for dtype in (np.float64, np.float32):
            for cm, cr in ((C.CONSTANTS, R.CONSTANTS), (C.ZERO, R.ZERO), (C.FULL_PRECISION, R.CONSTANTS_FIT_FULL_PRECISION)):
                Fm, im = C.apply_batch(F.astype(dtype), pre, lt, dt, st, cm)
                Fr, ir = R.continue_batch(F.astype(dtype), pre, lt, dt, st, cr)
                self.assertEqual(Fm.tobytes(), Fr.tobytes(), (dtype, cm))
                for k in INFO_KEYS:
                    self.assertEqual(im[k].dtype, ir[k].dtype, k)
                    self.assertTrue(np.array_equal(im[k], ir[k], equal_nan=True), (k, dtype, cm))
                self.assertEqual(im["max_abs_delta"], ir["max_abs_delta"])
        Fm, im = C.apply_batch(F, pre, lt, dt, st)
        self.assertGreater(int(im["clipped_f"].sum()), 0)
        self.assertGreater(int(im["missing_f"].sum()), 0)
        self.assertGreater(int((im["missing_d"] & ~im["missing_f"]).sum()), 0)
        self.assertGreater(int(st.sum()), 0)
        for i in range(n):                                                             # one window at a time, and the anomalies
            Fq, rec = C.apply(F[i], rows_[i], cal[i], origins[i], state=bool(st[i]))
            Fw, iw = R.continue_window(F[i], *w_ref[i], bool(st[i]))
            self.assertEqual(Fq.tobytes(), Fw.tobytes(), i)
            self.assertEqual((rec.x_fast, rec.x_day), (iw["xf"], iw["xd"]), i)
            if not st[i]:
                am = C.anomalies(pre[i], F[i], lt[i], dt[i])
                ar = R.anomalies(pre[i], F[i], lt[i], dt[i])
                for k, v in ar.items():
                    self.assertTrue(v == am[k] or (isinstance(v, float) and math.isnan(v) and math.isnan(am[k])), (i, k))

    def test_parts_and_errors_match(self):
        R = self.R
        rng = np.random.default_rng(3)
        for a, b in zip(C.decay_curves(C.CONSTANTS), R.decay_curves(R.CONSTANTS)):
            self.assertEqual(a.tobytes(), b.tobytes())
        xf, xd = rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50)
        self.assertEqual(C.delta(xf, xd).tobytes(), R.delta(xf, xd, R.CONSTANTS).tobytes())
        self.assertEqual(C.delta(0.37, -0.81).tobytes(), R.delta(0.37, -0.81, R.CONSTANTS).tobytes())
        for _ in range(3000):
            last, days = int(rng.integers(0, 8)), [int(v) for v in rng.integers(0, 8, 7)]
            self.assertEqual(C.reference_day(last, days), R.reference_day(last, days))
        load, types, F = clean_window(rng)
        pre, lt, dt = C.window_inputs(load, types, 2000)
        bad = F.copy()
        bad[3] = np.nan
        for fn in (lambda: C.apply_batch(bad[None], pre[None], [lt], [dt], [False]),
                   lambda: R.continue_batch(bad[None], pre[None], [lt], [dt], [False])):
            with self.assertRaises(ValueError):
                fn()
        neg = pre.copy()
        neg[-3:] = -5.0
        with self.assertRaises(ValueError):                                           # the reference has no logarithm there
            R.continue_batch(F[None], neg[None], [lt], [dt], [False])
        Fm, im = C.apply_batch(F[None], neg[None], [lt], [dt], [False])               # the release sets x_f = 0 (flagged)
        self.assertTrue(bool(im["nonpositive_f"][0]) and im["xf"][0] == 0.0 and not im["nonpositive_d"][0])
        D = C.delta(0.0, im["xd"][0])
        self.assertEqual(Fm[0].tobytes(), (F.astype(np.float64) * np.exp(D)).tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
