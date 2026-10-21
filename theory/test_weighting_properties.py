"""Exact properties C11-C14 of the inverse-noise loss weighting of recipe Rw (stage 2;
theory/PROOFS.md, "Exact properties of the carrier network"), checked numerically, together with the counterexamples
that bound them.

The checks use synthetic training arrays in the layout of ``ankyra_lite.carrier.train`` (day tokens as
``ankyra_lite.carrier.features`` builds them), the package's own sampler, and untrained carriers (``build_carrier``
under a torch seed) for optimisation steps on the CPU.  No checkpoint and no study data are needed.  Run from the
repository root:

    python -m unittest discover -s theory -t .
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch.set_num_threads(1)

from ankyra_lite.anchor.synthetic import synthetic_history                             # noqa: E402
from ankyra_lite.carrier import train as T                                             # noqa: E402
from ankyra_lite.carrier.features import features                                      # noqa: E402
from ankyra_lite.carrier.model import build_carrier, forecast_loss, param_groups       # noqa: E402

RULE = T.NoiseWeights()
D0, D1 = 338, 366                                 # the day tokens the noise estimate reads (the last 28 days)
ORIGIN = 15000                                    # issue time used with the synthetic store (16,000 hours)


# ----------------------------------------------------------------------------- helpers
def day_tokens(n: int, seed: int, missing: float = 0.02, noise=(0.01, 0.5), dtype=np.float16) -> np.ndarray:
    """(n, 366, 69) day tokens in the layout of the features: 24 values arcsinh(load / s) (a daily and weekly cycle
    plus row-specific noise), 24 hourly observation flags (a share ``missing`` unobserved, their values 0), and
    random values in the other 21 columns."""
    rng = np.random.default_rng(seed)
    t = np.arange(366 * 24).reshape(366, 24)
    base = 1 + 0.5 * np.sin(2 * np.pi * t / 24) + 0.2 * np.cos(2 * np.pi * t / 168)
    y = np.maximum(base[None] + rng.uniform(*noise, (n, 1, 1)) * rng.normal(size=(n, 366, 24)), 0.0)
    obs = rng.random((n, 366, 24)) > missing
    tok = rng.normal(size=(n, 366, 69))
    tok[..., :24] = np.where(obs, np.arcsinh(y), 0.0)
    tok[..., 24:48] = obs
    return tok.astype(dtype)


def synthetic_pool(seed: int = 0, sizes=(("P", 60), ("Q", 40), ("S", 25)), undefined: int = 3):
    """A training pool of a target "P" and borrowed populations (unequal units and row counts): (pool rows, arrays,
    units).  Every population has rows outside the pool; ``undefined`` pool rows of each have too few observed pairs
    for a noise estimate."""
    rng = np.random.default_rng(seed)
    pool, arrays, units = {}, {}, {}
    for j, (q, n) in enumerate(sizes):
        tok = day_tokens(n, seed + 10 * j + 1)
        ix = np.sort(rng.choice(n, int(0.8 * n), replace=False))
        for r in ix[:undefined]:
            tok[r, D0:D1, 24:48] = 0                                               # no observed hour in the last 28 days
        arrays[q] = {"day_tokens": tok, "target": rng.random((n, 744)).astype(np.float32),
                     "observed": rng.random((n, 744)) > 0.05, "day_mask": np.ones((n, 366), bool),
                     "scale": rng.uniform(0.5, 5.0, n).astype(np.float32)}
        units[q] = np.sort(rng.integers(0, 4 + 2 * j, n))
        pool[q] = ix
    return pool, arrays, units


def weights(pool, arrays, units, target="P", rule=RULE):
    return T.noise_weights(target, pool, arrays, units, rule, draws=16000, own_fraction=0.5)


def batch(seed: int, size: int = 64) -> dict:
    """A training batch with the structure of ``make_batch``: nonnegative load tokens, 0/1 hourly observation flags,
    one-hot day types, masked tokens, targets with 5% unobserved hours."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    day = r(size, 366, 69)
    day[..., :24] = day[..., :24].abs()
    day[..., 24:48] = (torch.rand(size, 366, 24, generator=g) > 0.02).float()
    day[..., 52:60] = F.one_hot(torch.randint(0, 8, (size, 366), generator=g), 8).float()
    future = r(size, 31, 16)
    future[..., :8] = F.one_hot(torch.randint(0, 8, (size, 31), generator=g), 8).float()
    scale = torch.rand(size, generator=g) * 10 + 0.5
    return {"day_tokens": day, "day_mask": torch.rand(size, 366, generator=g) > 0.1, "month_tokens": r(size, 24, 16),
            "month_mask": torch.rand(size, 24, generator=g) > 0.3, "future_tokens": future, "scale": scale,
            "target": torch.rand(size, 744, generator=g) * 2 * scale[:, None], "observed": torch.rand(size, 744, generator=g) > 0.05}


def per_window(seed: int, size: int = 64):
    """The per-window loss of ``forecast_loss`` of an untrained carrier on a synthetic batch."""
    torch.manual_seed(seed)
    m = build_carrier().eval()
    b = batch(seed, size)
    with torch.no_grad():
        out = m({k: b[k] for k in T.INPUTS})
    L = forecast_loss(out, torch.where(b["observed"], b["target"], 0.0), b["scale"], b["observed"], kl_weight=0.1, compression=True)
    return L["per_window"], L["loss"]


# ============================================================================= C11 unit weights are recipe R
class UnitWeightTests(unittest.TestCase):
    def test_unit_weights_give_the_unweighted_loss_bit_for_bit(self):
        for seed, size in ((0, 64), (1, 64), (2, 5)):
            pw, loss = per_window(seed, size)
            self.assertTrue(torch.equal(T.weighted_log1p_loss(pw, torch.ones(size)), loss), (seed, size))

    def test_a_training_step_with_unit_weights_is_the_unweighted_step(self):
        unweighted = T.recipe("span_return", None)
        weighted = T.recipe("span_return_weighted", None)
        for seed in (0, 1):
            runs = []
            for cfg, w in ((unweighted, None), (weighted, np.ones(64))):
                torch.manual_seed(seed)
                m = build_carrier()
                opt = T.make_optimizer(m, cfg)
                groups = param_groups(m)
                m.train()
                losses = [T.train_step(m, opt, batch(100 * seed + step), groups, cfg, weights=w) for step in range(2)]
                runs.append((losses, {k: v.detach().clone() for k, v in m.state_dict().items()}))
            (la, sa), (lb, sb) = runs
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(la, lb)), seed)
            self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa), seed)

    def test_the_loss_is_invariant_to_rescaling_the_weights(self):
        pw, _ = per_window(3)
        w = torch.as_tensor(np.random.default_rng(3).uniform(0.2, 3.0, 64), dtype=pw.dtype)
        ref, ones = T.weighted_log1p_loss(pw, w), T.weighted_log1p_loss(pw, torch.ones(64))
        for c in (2.0, 0.5, 1024.0, 2.0 ** -6):                                    # bit for bit for powers of two
            self.assertTrue(torch.equal(T.weighted_log1p_loss(pw, c * w), ref), c)
            self.assertTrue(torch.equal(T.weighted_log1p_loss(pw, torch.full((64,), c)), ones), c)
        for c in (3.7, 0.013):                                                     # other factors: to float32 rounding
            torch.testing.assert_close(T.weighted_log1p_loss(pw, c * w), ref, rtol=1e-6, atol=0)
            torch.testing.assert_close(T.weighted_log1p_loss(pw, torch.full((64,), c)), ones, rtol=1e-6, atol=0)
        self.assertFalse(torch.equal(ref, ones))                                   # unequal weights do change the loss

    def test_without_weighting_the_recipes_and_their_records_are_unchanged(self):
        r, rw = T.recipe("span_return", "runs/{target}/seed{seed}.pt"), T.recipe("span_return_weighted", "runs/{target}/seed{seed}.pt")
        self.assertEqual(replace(rw, noise_weighting=None), r)                     # the loss weighting is the only difference
        self.assertEqual(rw.noise_weighting, T.NoiseWeights())
        self.assertEqual((T.recipe("R", "x"), T.recipe("Rw", "x")), (T.recipe("span_return", "x"), T.recipe("span_return_weighted", "x")))
        for name in T.RECIPES:
            self.assertIsNone(T.recipe(name, "x").noise_weighting)
            self.assertNotIn("noise_weighting", T.recipe(name, "x").describe())  # records of the 0.1 recipes keep their keys
        self.assertIn("noise_weighting", rw.describe())
        self.assertEqual((RULE.gamma, RULE.floor_quantile, RULE.min_pairs, RULE.context_days, RULE.lag_days),
                         (0.5, 0.25, 168, (338, 366), 7))
        cfg = replace(T.TrainConfig(), log1p_compression=False)
        m = build_carrier()
        with self.assertRaises(ValueError):                                        # the weights act on the log1p-compressed loss
            T.train_step(m, T.make_optimizer(m, cfg), batch(0, 4), param_groups(m), cfg, weights=np.ones(4))


# ============================================================================= C12 each source keeps its draw share
class SourceBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool, cls.arrays, cls.units = synthetic_pool()
        cls.p = T.draw_probabilities("P", cls.pool, cls.units, draws=16000, own_fraction=0.5)

    def test_draw_probabilities_follow_the_sampler_rule(self):
        p = self.p
        self.assertAlmostEqual(sum(float(v.sum()) for v in p.values()), 1.0, delta=1e-12)
        self.assertAlmostEqual(float(p["P"].sum()), 0.5, delta=1e-15)              # floor(16000 x 0.5) / 16000 = 1/2
        for q in ("Q", "S"):
            self.assertAlmostEqual(float(p[q].sum()), 0.25, delta=1e-15)           # the rest, split equally
        for q, ix in self.pool.items():                                            # units at equal total weight
            u = self.units[q][ix]
            tot = np.array([p[q][u == x].sum() for x in np.unique(u)])
            np.testing.assert_allclose(tot, tot.mean(), rtol=1e-12)
        alone = T.draw_probabilities("P", {"P": self.pool["P"]}, self.units, draws=16000, own_fraction=0.5)
        self.assertAlmostEqual(float(alone["P"].sum()), 1.0, delta=1e-12)         # without borrowed populations: all own

    def test_the_draw_probabilities_are_those_of_the_sampler(self):
        rng = np.random.default_rng(7)
        count = {}
        epochs = 10
        for _ in range(epochs):
            for q, r in T.sampler("P", self.pool, self.units, rng, draws=16000, own_fraction=0.5):
                key = (q, int(self.units[q][r]))
                count[key] = count.get(key, 0) + 1
        n = 16000 * epochs
        for q, ix in self.pool.items():
            u = self.units[q][ix]
            for x in np.unique(u):
                pu = float(self.p[q][u == x].sum())
                self.assertLess(abs(count.get((q, int(x)), 0) / n - pu), 5 * np.sqrt(pu * (1 - pu) / n), (q, x))

    def test_every_source_keeps_its_draw_share(self):
        W, rec = weights(self.pool, self.arrays, self.units)
        total = 0.0
        for q, ix in self.pool.items():
            share = float(self.p[q].sum())
            np.testing.assert_allclose((self.p[q] * W[q][ix]).sum(), share, rtol=1e-12)
            self.assertAlmostEqual(rec["sources"][q]["draw_share"], share, delta=1e-15)
            total += float((self.p[q] * W[q][ix]).sum())
        self.assertAlmostEqual(total, 1.0, delta=1e-12)                            # expected weight of a draw: 1
        W2, _ = weights(self.pool, {q: dict(a, day_tokens=self.noisier(a["day_tokens"])) if q == "Q" else a
                                    for q, a in self.arrays.items()}, self.units)
        for q, ix in self.pool.items():                                            # a noisier population: other weights, same share
            np.testing.assert_allclose((self.p[q] * W2[q][ix]).sum(), float(self.p[q].sum()), rtol=1e-12)
        self.assertFalse(np.array_equal(W2["Q"], W["Q"], equal_nan=True))

    @staticmethod
    def noisier(tok):
        out = tok.astype(np.float32)
        out[:, D0:D1, :24] *= np.random.default_rng(9).uniform(0.5, 1.5, out[:, D0:D1, :24].shape).astype(np.float32)
        return out.astype(tok.dtype)


# ============================================================================= C13 bounded weights and the floor
class WeightBoundTests(unittest.TestCase):
    def test_the_floor_bounds_every_weight_and_carries_a_quarter_of_the_draw_mass(self):
        for seed in (0, 1, 2):
            pool, arrays, units = synthetic_pool(seed)
            W, rec = weights(pool, arrays, units)
            p = T.draw_probabilities("P", pool, units, draws=16000, own_fraction=0.5)
            f, neutral = rec["floor_sigma2"], rec["neutral_raw_weight"]
            s2 = {q: T.pool_sigma2(arrays[q], ix) for q, ix in pool.items()}
            S = np.concatenate([s2[q] for q in pool])
            Pp = np.concatenate([p[q] for q in pool])
            ok = np.isfinite(S)
            self.assertTrue(0 < ok.sum() < len(S))                                 # defined and undefined rows
            self.assertGreaterEqual(Pp[ok][S[ok] <= f].sum(), 0.25 * Pp[ok].sum() - 1e-15)
            self.assertTrue(f > 0 and f == T.weighted_quantile(S[ok], Pp[ok], 0.25))
            raw_defined = np.maximum(S[ok], f) ** -0.5
            self.assertTrue(raw_defined.min() <= neutral <= raw_defined.max())
            self.assertAlmostEqual(raw_defined.max(), f ** -0.5, delta=1e-15 * f ** -0.5)
            smax = float(S[ok].max())
            for q, ix in pool.items():
                c = rec["sources"][q]["normaliser"]
                w = W[q][ix]
                self.assertTrue(np.isfinite(w).all() and (w > 0).all())            # every pool row: finite and positive
                outside = np.setdiff1d(np.arange(len(units[q])), ix)
                self.assertTrue(np.isnan(W[q][outside]).all())                    # no weight outside the pool
                d = np.isfinite(s2[q])
                np.testing.assert_allclose(w[d] / c, np.maximum(s2[q][d], f) ** -0.5, rtol=1e-14)
                np.testing.assert_allclose(w[~d] / c, neutral, rtol=1e-14)
                o = np.argsort(s2[q][d], kind="stable")
                self.assertTrue((np.diff(w[d][o]) <= 1e-14 * w.max()).all())       # nonincreasing in the noise
                at = w[d][s2[q][d] <= f]
                self.assertTrue((at == w.max()).all())                             # equal, and the largest, at the floor
                self.assertTrue((w >= c * smax ** -0.5 * (1 - 1e-14)).all() and (w <= c * f ** -0.5 * (1 + 1e-14)).all())
                self.assertLessEqual(w.max() / w.min(), np.sqrt(smax / f) * (1 + 1e-12))

    def test_a_quarter_of_noiseless_windows_sets_the_floor_to_zero_and_is_refused(self):
        # counterexample: rows whose last three weeks repeat exactly have sigma2 = 0; with a quarter of the draw mass on
        # them the floor is 0 and their raw weight 0^(-1/2) is infinite, so the pool is refused
        pool, arrays, units = synthetic_pool(4)
        tok = arrays["P"]["day_tokens"].copy()
        week = tok[:, D1 - 7:D1].copy()
        for k in range(D0, D1 - 7, 7):
            tok[:, k:k + 7] = week                                                 # the last 28 days repeat one week
        arrays = dict(arrays, P=dict(arrays["P"], day_tokens=tok))
        s2 = T.pool_sigma2(arrays["P"], pool["P"])
        self.assertTrue((s2[np.isfinite(s2)] == 0).all())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(ValueError):
                weights(pool, arrays, units)


# ============================================================================= C14 the weights read only each row's own recent context
class WeightInputTests(unittest.TestCase):
    def test_the_noise_estimate_reads_only_the_last_28_days_hourly_values_and_flags(self):
        tok = day_tokens(30, 11)
        s = T.noise_sigma2(tok)
        other = tok.copy()
        other[:, :D0] = np.float16(7.0)                                            # older days
        other[:, :, 48:] = np.float16(-3.0)                                       # daily statistics, temperature, calendar, ...
        self.assertTrue(np.array_equal(T.noise_sigma2(other), s, equal_nan=True))
        hit = tok.copy()
        hit[:, 360, 5] = np.where(hit[:, 360, 29] > 0.5, hit[:, 360, 5] + np.float16(1.0), hit[:, 360, 5])
        changed = np.isfinite(s) & (tok[:, 360, 29] > 0.5)
        self.assertTrue(changed.any())
        self.assertFalse(np.array_equal(T.noise_sigma2(hit)[changed], s[changed]))

    def test_the_noise_estimate_is_the_weekly_naive_relative_error_of_the_last_three_weeks(self):
        h = synthetic_history(16000)
        load = np.asarray(h.load_kw, dtype=np.float64) * (1 + 0.1 * np.random.default_rng(5).normal(size=16000))
        load[ORIGIN - 300:ORIGIN - 290] = np.nan                                   # a gap inside the paired hours
        for half, rtol in ((False, 1e-5), (True, 5e-3)):                          # float16 tokens: about 2-5e-4 observed
            f = features(load, h.temperature_c, h.day_types, "Office", h.start_timestamp, 0, ORIGIN, half=half)
            y = load[ORIGIN - 672:ORIGIN]
            ok = np.isfinite(y) & (y >= 0)
            d = (y[168:] - y[:-168]) / float(f["scale"])
            m = ok[168:] & ok[:-168]
            direct = float((d[m] ** 2).mean())
            np.testing.assert_allclose(T.noise_sigma2(f["day_tokens"][None])[0], direct, rtol=rtol, err_msg=str(half))

    def test_fewer_than_168_observed_pairs_leave_the_estimate_undefined(self):
        tok = day_tokens(2, 12, missing=0.0)
        tok[:, D0:D1, 24:48] = 0
        tok[:, D0:D0 + 14, 24:48] = 1                                              # days 338..351: 7 x 24 = 168 pairs
        tok[1, D0 + 13, 24 + 23] = 0                                               # one hour less: 167 pairs
        s = T.noise_sigma2(tok)
        self.assertTrue(np.isfinite(s[0]) and np.isnan(s[1]))

    def test_labels_masks_and_rows_outside_the_pool_do_not_enter(self):
        pool, arrays, units = synthetic_pool(5)
        W, rec = weights(pool, arrays, units)
        changed = {}
        for q, a in arrays.items():
            b = {k: np.array(v, copy=True) for k, v in a.items()}
            b["target"][:] = np.nan
            b["observed"] = ~b["observed"]
            b["day_mask"] = ~b["day_mask"]
            b["scale"] = b["scale"] * 3
            outside = np.setdiff1d(np.arange(len(units[q])), pool[q])
            b["day_tokens"][outside] = np.float16(np.nan)                          # rows outside the pool are never read
            changed[q] = b
        W2, rec2 = weights(pool, changed, units)
        for q in pool:
            self.assertTrue(np.array_equal(W2[q], W[q], equal_nan=True), q)
        self.assertEqual(rec2["floor_sigma2"], rec["floor_sigma2"])

    def test_the_weights_are_deterministic_use_no_random_stream_and_ignore_the_storage_of_the_rows(self):
        pool, arrays, units = synthetic_pool(6)
        np_state, torch_state = np.random.get_state(), torch.get_rng_state()
        W1, _ = weights(pool, arrays, units)
        W2, _ = weights(pool, arrays, units)
        after = np.random.get_state()
        self.assertTrue(np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        for q in pool:
            self.assertTrue(np.array_equal(W1[q], W2[q], equal_nan=True))
        # a population stored as built rows followed by extra rows (RowConcat), read in chunks of any size
        tok = arrays["P"]["day_tokens"]
        split = {"day_tokens": T.RowConcat(tok[:37], tok[37:])}
        for chunk in (1, 7, 512):
            np.testing.assert_array_equal(T.pool_sigma2(split, pool["P"], chunk=chunk), T.pool_sigma2(arrays["P"], pool["P"]))

    def test_power_of_two_scaling_of_the_load_leaves_the_noise_estimate_unchanged(self):
        # through the tokens of the features (C7): bit for bit for powers of two
        h = synthetic_history(16000)
        load = np.asarray(h.load_kw, dtype=np.float64) * (1 + 0.1 * np.random.default_rng(8).normal(size=16000))
        s = T.noise_sigma2(features(load, h.temperature_c, h.day_types, "Office", h.start_timestamp, 0, ORIGIN)["day_tokens"][None])
        for c in (2.0, 0.125, 1024.0):
            fc = features(c * load, h.temperature_c, h.day_types, "Office", h.start_timestamp, 0, ORIGIN)
            self.assertTrue(np.array_equal(T.noise_sigma2(fc["day_tokens"][None]), s), c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
