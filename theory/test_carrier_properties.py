"""Exact properties C1-C10 of the carrier network (theory/PROOFS.md, "Exact properties of the carrier network"),
checked numerically, together with the counterexamples that bound them.

The checks use untrained carriers (``build_carrier`` under several torch seeds) and the same carriers with perturbed
weights, so that the copy pointer's gates are non-zero; CPU only.  Statements about real arithmetic are checked in
float64; statements about the delivered carrier forecasts use the package's own float32 path
(``carrier_forecasts``).  No checkpoint and no study data are needed.  Run from the repository root:

    python -m unittest discover -s theory -t .
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch.set_num_threads(1)

from ankyra_lite.anchor import blocks                                                  # noqa: E402
from ankyra_lite.anchor.core import K_FM                                               # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                             # noqa: E402
from ankyra_lite.carrier import inference as I                                         # noqa: E402
from ankyra_lite.carrier.features import features                                      # noqa: E402
from ankyra_lite.carrier.model import INPUT_KEYS, StructuredLoadNetwork, build_carrier  # noqa: E402
from ankyra_lite.carrier.train import TrainConfig, cut_data, make_windows, split       # noqa: E402

SEEDS = (0, 1, 2, 3)
ORIGIN = 15000                                    # issue time used with the synthetic store (16,000 hours)
GAP = 6 * 744                                     # 186 days = 4,464 hours


# ----------------------------------------------------------------------------- helpers
def carrier(seed: int, perturb: bool = True, gate_std: float = 0.1, dtype=torch.float64):
    """build_carrier() under torch seed ``seed``; with ``perturb`` every weight is moved and the copy-pointer gates are
    set to non-zero random values (so that the pointer is active), reproducibly per seed."""
    torch.manual_seed(seed)
    m = build_carrier()
    if perturb:
        g = torch.Generator().manual_seed(10_000 + seed)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.startswith("copy_gate"):
                    p.copy_(torch.randn(p.shape, generator=g) * gate_std)
                elif name == "type_bonus":
                    p.fill_(0.7)
                else:
                    p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return m.to(dtype).eval()


def base_of(m):
    """The base network (no copy pointer) with the base weights of carrier ``m``."""
    b = StructuredLoadNetwork(m.config)
    b.load_state_dict({k: v for k, v in m.state_dict().items() if k.split(".")[0] not in m.POINTER_MODULES}, strict=True)
    return b.to(next(m.parameters()).dtype).eval()


def random_inputs(batch: int = 4, seed: int = 0, dtype=torch.float64, amp: float = 1.0, limit: float | None = None):
    """Random network inputs with the structure of the features: nonnegative load tokens, 0/1 hourly observation flags
    (2% missing hours), one-hot day types; masked tokens are present (10% of days, 30% of months)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    day = r(batch, 366, 69) * amp
    day[..., :24] = day[..., :24].abs()
    day[..., 24:48] = (torch.rand(batch, 366, 24, generator=g, dtype=torch.float64) > 0.02).double()
    day[..., 52:60] = F.one_hot(torch.randint(0, 8, (batch, 366), generator=g), 8).double()
    month = r(batch, 24, 16) * amp
    if limit is not None:
        day, month = day.clamp(-limit, limit), month.clamp(-limit, limit)
    future = r(batch, 31, 16)
    future[..., :8] = F.one_hot(torch.randint(0, 8, (batch, 31), generator=g), 8).double()
    return {"day_tokens": day.to(dtype), "day_mask": torch.rand(batch, 366, generator=g) > 0.1,
            "month_tokens": month.to(dtype), "month_mask": torch.rand(batch, 24, generator=g) > 0.3,
            "future_tokens": future.to(dtype),
            "scale": (torch.rand(batch, generator=g, dtype=torch.float64) * 10 + 0.5).to(dtype)}


def clone(x):
    return {k: v.clone() for k, v in x.items()}


def run(m, x):
    with torch.no_grad():
        return m(x)


def synthetic_store(rows=("a", "b"), hours: int = 16000, seed: int = 0):
    """Arrays in the layout of ankyra_lite.carrier.features; rows with the same letter hold the same series."""
    h = synthetic_history(hours)
    rng = np.random.default_rng(seed)
    series = {}
    for i, name in enumerate(sorted(set(rows))):
        series[name] = np.asarray(h.load_kw) * (1 + 0.1 * i) + rng.normal(0, 0.3, hours)
    load = np.stack([series[name] for name in rows])
    temp = np.stack([np.asarray(h.temperature_c, np.float32)] * len(rows))
    groups = ["Office" if name == "a" else "Residential" for name in rows]
    return {"load": load, "temp": temp, "types": np.asarray(h.day_types), "groups": groups, "start_iso": h.start_timestamp}


def cf(models, d, unit, origin=ORIGIN, cut=None, load=None, temp=None, types=None, groups=None):
    return I.carrier_forecasts(models, d["load"] if load is None else load, d["temp"] if temp is None else temp,
                               d["types"] if types is None else types, d["groups"] if groups is None else groups,
                               d["start_iso"], unit, origin, cut=cut)


def same(f, g):
    return sorted(f) == sorted(g) and all(np.array_equal(f[k], g[k]) for k in f)


def ensemble(seeds=(0, 1)):
    """A two-member ensemble of perturbed float32 carriers (the package's inference precision)."""
    return [carrier(s, dtype=torch.float32) for s in seeds]


# ============================================================================= C1 nonnegativity
class NonnegativityTests(unittest.TestCase):
    def test_trajectory_is_finite_and_nonnegative_for_random_weights_and_inputs(self):
        for seed in SEEDS:
            for dtype in (torch.float32, torch.float64):
                out = run(carrier(seed, dtype=dtype), random_inputs(seed=seed, dtype=dtype))
                tr = out["trajectory_kw"]
                self.assertTrue(torch.isfinite(tr).all() and (tr >= 0).all(), (seed, dtype))
                self.assertTrue((out["level_kw"] > 0).all())
                if dtype == torch.float64:                                    # strictly positive in real arithmetic
                    self.assertTrue((tr > 0).all(), seed)

    def test_extreme_logits_can_underflow_to_zero_but_never_below(self):
        # large tokens and large gates drive the softmax shares to underflow: exact zeros occur, negative values do not
        zeros = 0
        for gate_std, amp in ((3.0, 20.0), (10.0, 20.0), (10.0, 60.0)):
            m = carrier(3, gate_std=gate_std, dtype=torch.float32)
            tr = run(m, random_inputs(seed=7, dtype=torch.float32, amp=amp, limit=80.0))["trajectory_kw"]
            self.assertTrue(torch.isfinite(tr).all())
            self.assertGreaterEqual(float(tr.min()), 0.0)
            zeros += int((tr == 0).sum())
        self.assertGreater(zeros, 0)

    def test_feature_built_inputs_bound_the_pointer_exponential(self):
        # one large hour inside the last 56 days and almost no load elsewhere: load / scale stays at most 8,760
        d = synthetic_store(("a",))
        load = np.full_like(d["load"], 1e-3)
        load[0, ORIGIN - 1000] = 1e6                                          # in [o - 1344, o - 744): outside the 31-day mean
        f = features(load, d["temp"], d["types"], "Office", d["start_iso"], 0, ORIGIN)
        ratio = np.sinh(f["day_tokens"][-56:, :24].astype(np.float64))
        self.assertGreater(ratio.max(), 8000.0)
        self.assertLessEqual(ratio.max(), 8760.0 * 1.01)                      # float16 storage of the token adds < 1%
        pred = cf(ensemble(), d, 0, load=load, groups="Office")
        self.assertTrue(all(np.isfinite(v).all() and (v >= 0).all() for v in pred.values()))

    def test_inference_rejects_a_non_finite_forecast(self):
        d = synthetic_store(("a",))
        f = features(d["load"], d["temp"], d["types"], "Office", d["start_iso"], 0, ORIGIN)
        f = dict(f)
        f["day_tokens"] = f["day_tokens"].copy()
        f["day_tokens"][-1, :24] = 88.0                                       # 24 x sinh(88) overflows float32
        with self.assertRaises(FloatingPointError):
            I._run([carrier(0, dtype=torch.float32)], [f], "cpu", 64)


# ============================================================================= C2 energy budget
class EnergyBudgetTests(unittest.TestCase):
    def test_trajectory_mean_is_the_level_and_day_means_are_the_daily_budget(self):
        for seed in SEEDS:
            out = run(carrier(seed), random_inputs(seed=seed))
            cube = out["trajectory_kw"].reshape(-1, 31, 24)
            level, daily = out["level_kw"], out["daily_means_kw"]
            torch.testing.assert_close(cube.mean((1, 2)), level, rtol=1e-12, atol=0)
            torch.testing.assert_close(cube.mean(-1), daily, rtol=1e-12, atol=0)
            torch.testing.assert_close(daily, 31 * level[:, None] * out["day_fractions"], rtol=1e-14, atol=0)
            self.assertTrue(torch.equal(out["energy_kwh"], 744.0 * level))

    def test_budget_outputs_are_the_block_decomposition_of_the_trajectory(self):
        for seed in SEEDS:
            out = run(carrier(seed), random_inputs(seed=seed))
            tr = out["trajectory_kw"].numpy()
            tol = 1e-12 * float(np.abs(tr).max())
            np.testing.assert_allclose(blocks.level(tr), out["level_kw"].numpy(), rtol=1e-12, atol=0)
            np.testing.assert_allclose(blocks.daily_path(tr), out["daily_path_kw"].numpy(), rtol=0, atol=tol)
            np.testing.assert_allclose(blocks.within_day(tr), out["within_day_kw"].reshape(-1, 744).numpy(), rtol=0, atol=tol)
            np.testing.assert_allclose(out["daily_path_kw"].sum(-1).numpy(), 0.0, atol=tol * 31)

    def test_float32_path_and_ensemble_average(self):
        x = random_inputs(seed=5, dtype=torch.float32)
        outs = [run(carrier(s, dtype=torch.float32), x) for s in (0, 1, 2)]
        for out in outs:
            torch.testing.assert_close(out["trajectory_kw"].double().mean(1), out["level_kw"].double(), rtol=1e-5, atol=0)
        mean_traj = sum(o["trajectory_kw"].double() for o in outs) / 3                # the ensemble rule of inference._run
        torch.testing.assert_close(mean_traj.mean(1), sum(o["level_kw"].double() for o in outs) / 3, rtol=1e-5, atol=0)
        self.assertTrue((mean_traj >= 0).all())


# ============================================================================= C3 distributions
class ShareTests(unittest.TestCase):
    def test_day_and_hour_shares_are_distributions(self):
        for seed in SEEDS:
            out = run(carrier(seed), random_inputs(seed=seed))
            p, q = out["day_fractions"], out["hour_fractions"]
            self.assertEqual((tuple(p.shape[1:]), tuple(q.shape[1:])), ((31,), (31, 24)))
            self.assertTrue((p >= 0).all() and (q >= 0).all())
            torch.testing.assert_close(p.sum(-1), torch.ones_like(p[:, 0]), rtol=0, atol=1e-14)
            torch.testing.assert_close(q.sum(-1), torch.ones_like(q[..., 0]), rtol=0, atol=1e-14)

    def test_pointer_attention_is_a_distribution_over_complete_recent_days(self):
        for seed in SEEDS:
            x = random_inputs(seed=seed)
            x["day_tokens"][0, -56:, 30] = 0.0                                # sample 0: no complete day in the last 56
            out = run(carrier(seed), x)
            alpha = out["copy_alpha"]                                         # (B, 31 forecast days, 56 recent days)
            recent = x["day_tokens"][:, -56:]
            complete = x["day_mask"][:, -56:] & (recent[..., 24:48] > 0.5).all(-1)
            self.assertFalse(bool(complete[0].any()))
            self.assertTrue(bool(complete[1:].any(-1).all()))
            self.assertTrue((alpha >= 0).all())
            self.assertTrue(torch.equal(alpha[0], torch.zeros_like(alpha[0])))            # nothing to copy from
            torch.testing.assert_close(alpha[1:].sum(-1), torch.ones_like(alpha[1:, :, 0]), rtol=0, atol=1e-14)
            self.assertTrue(bool((alpha.masked_select(~complete[:, None, :].expand_as(alpha)) == 0).all()))


# ============================================================================= C4 causality in time
class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synthetic_store(("a", "b"))

    @staticmethod
    def garbage(a, start):
        """A copy of ``a`` whose hours at and after ``start`` hold NaN, +inf, a negative value and a huge value."""
        b = np.array(a, copy=True)
        fill = np.array([np.nan, np.inf, -5.0, 1e30], dtype=b.dtype)
        n = b.shape[-1] - start
        b[..., start:] = np.resize(fill, n)
        return b

    def test_features_read_no_load_or_temperature_at_or_after_the_issue_time(self):
        d = self.d
        for q in (ORIGIN, ORIGIN - 744, ORIGIN - 6 * 744, 3000, 500):
            types = d["types"].copy()
            types[q + 744:] = (types[q + 744:] + 3) % 8                       # calendar after the forecast month
            f1 = features(d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], 1, q)
            f2 = features(self.garbage(d["load"], q), self.garbage(d["temp"], q), types, d["groups"], d["start_iso"], 1, q)
            for k in f1:
                self.assertTrue(np.array_equal(np.asarray(f1[k]), np.asarray(f2[k])), (q, k))

    def test_carrier_forecasts_are_bitwise_unchanged_by_data_after_the_origin(self):
        d, models = self.d, ensemble()
        f = cf(models, d, 1)
        self.assertEqual(sorted(f), list(range(7)))
        types = d["types"].copy()
        types[ORIGIN + 744:] = 7
        g = cf(models, d, 1, load=self.garbage(d["load"], ORIGIN), temp=self.garbage(d["temp"], ORIGIN), types=types)
        self.assertTrue(same(f, g))
        # the pseudo-origin forecast issued at q = o - 744 k reads nothing at or after q either
        q = ORIGIN - 2 * 744
        load, temp = d["load"].copy(), d["temp"].copy()
        load[:, q:] *= 3.0
        temp[:, q:] += 5.0
        g = cf(models, d, 1, load=load, temp=temp)
        self.assertTrue(all(np.array_equal(f[k], g[k]) for k in (2, 3, 4, 5, 6)))
        self.assertFalse(np.array_equal(f[0], g[0]))

    def test_the_calendar_of_the_forecast_month_is_read(self):
        # where the claim stops: day types inside [o, o + 744) are an input (treated as known at the origin)
        d, models = self.d, ensemble()
        types = d["types"].copy()
        types[ORIGIN + 100:ORIGIN + 300] = 7
        self.assertFalse(np.array_equal(cf(models, d, 1)[0], cf(models, d, 1, types=types)[0]))


# ============================================================================= C5 token-order causality of the encoder
class EncoderOrderTests(unittest.TestCase):
    def test_day_and_month_memories_depend_only_on_earlier_tokens(self):
        for seed in SEEDS[:2]:
            m, x = carrier(seed), random_inputs(seed=seed)
            for monthly, key, positions in ((False, "day", (0, 200, 309, 364)), (True, "month", (0, 11, 22))):
                tok, mask = x[key + "_tokens"], x[key + "_mask"]
                with torch.no_grad():
                    e1 = m._encode(tok, mask, monthly)
                for j in positions:
                    tok2, mask2 = tok.clone(), mask.clone()
                    tok2[:, j + 1:] = torch.randn_like(tok2[:, j + 1:]) * 5
                    mask2[:, j + 1:] = ~mask2[:, j + 1:]
                    with torch.no_grad():
                        e2 = m._encode(tok2, mask2, monthly)
                    self.assertTrue(torch.equal(e1[:, :j + 1], e2[:, :j + 1]), (key, j))
                    self.assertFalse(torch.equal(e1[:, j + 1:], e2[:, j + 1:]), (key, j))


# ============================================================================= C6 masked tokens
class MaskTests(unittest.TestCase):
    def test_content_of_masked_tokens_is_ignored(self):
        for seed in SEEDS:
            m, x = carrier(seed), random_inputs(seed=seed)
            y = clone(x)
            y["day_tokens"][~y["day_mask"]] = float("nan")
            y["month_tokens"][~y["month_mask"]] = 1e6
            a, b = run(m, x), run(m, y)
            for key in ("trajectory_kw", "level_kw", "day_fractions", "hour_fractions", "copy_alpha", "copy_gate"):
                self.assertTrue(torch.equal(a[key], b[key]), (seed, key))


# ============================================================================= C7 scale equivariance
class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synthetic_store(("a", "b"))
        cls.models = ensemble()

    def test_power_of_two_scaling_is_bitwise_equivariant(self):
        d = self.d
        f = cf(self.models, d, 1)
        f0 = features(d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], 1, ORIGIN)
        self.assertFalse(f0["zero_scale"])
        for c in (2.0, 0.5, 8.0, 2.0 ** -6, 1024.0):
            fc = features(c * d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], 1, ORIGIN)
            for k in INPUT_KEYS:
                ref = np.asarray(f0[k]) * (c if k == "scale" else 1)
                self.assertTrue(np.array_equal(np.asarray(fc[k]), ref), (c, k))
            self.assertTrue(same(cf(self.models, d, 1, load=c * d["load"]), {k: c * v for k, v in f.items()}), c)

    def test_the_network_is_positively_homogeneous_in_the_scale_input(self):
        for seed in SEEDS:
            m, x = carrier(seed), random_inputs(seed=seed)
            a = run(m, x)
            for c in (3.7, 0.013, 250.0):
                y = clone(x)
                y["scale"] = y["scale"] * c
                b = run(m, y)
                for key in ("day_fractions", "hour_fractions", "copy_alpha"):
                    self.assertTrue(torch.equal(a[key], b[key]), (seed, c, key))
                torch.testing.assert_close(b["trajectory_kw"], c * a["trajectory_kw"], rtol=1e-13, atol=0)
                torch.testing.assert_close(b["level_kw"], c * a["level_kw"], rtol=1e-13, atol=0)

    def test_other_factors_are_equivariant_up_to_float32_and_float16_rounding(self):
        d = self.d
        f = cf(self.models, d, 1)
        g0 = features(d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], 1, ORIGIN, half=False)
        for c in (3.7, 0.3):
            gc = features(c * d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], 1, ORIGIN, half=False)
            for k in ("day_tokens", "month_tokens"):
                np.testing.assert_allclose(gc[k], g0[k], rtol=0, atol=1e-5)
            for k in ("day_mask", "month_mask", "future_tokens"):
                self.assertTrue(np.array_equal(gc[k], g0[k]), (c, k))
            self.assertAlmostEqual(float(gc["scale"]) / float(g0["scale"]), c, delta=1e-6 * c)
            fc = cf(self.models, d, 1, load=c * d["load"])                  # float16 tokens: rounding level only
            for k in f:
                self.assertLessEqual(float(np.max(np.abs(fc[k] - c * f[k]) / (c * f[k]))), 1e-3, (c, k))

    def test_the_zero_scale_fallback_breaks_equivariance(self):
        # counterexample: no positive load in the last 8,760 hours, older load positive -> scale 1 for every factor
        d = synthetic_store(("a",))
        load = d["load"].copy()
        load[0, ORIGIN - 9000:] = 0.0
        f1 = features(load, d["temp"], d["types"], "Office", d["start_iso"], 0, ORIGIN)
        f2 = features(2 * load, d["temp"], d["types"], "Office", d["start_iso"], 0, ORIGIN)
        self.assertTrue(f1["zero_scale"] and f2["zero_scale"] and f1["scale"] == f2["scale"] == 1.0)
        self.assertFalse(np.array_equal(f1["month_tokens"], f2["month_tokens"]))
        a = cf(self.models, d, 0, load=load, groups="Office")[0]
        b = cf(self.models, d, 0, load=2 * load, groups="Office")[0]
        self.assertFalse(np.allclose(b, 2 * a, rtol=1e-3))


# ============================================================================= C8 no unit identifier
class UnitIdentityTests(unittest.TestCase):
    def test_same_history_gives_the_same_forecast_at_any_row(self):
        d, models = synthetic_store(("a", "b", "a")), ensemble()
        f0, f2 = cf(models, d, 0), cf(models, d, 2)
        self.assertTrue(same(f0, f2))
        one = cf(models, d, 7, load=d["load"][0], temp=d["temp"][0], groups="Office")     # 1-D arrays: the index is ignored
        self.assertTrue(same(f0, one))
        load = d["load"].copy()
        load[1] = load[1][::-1] * 5.0                                         # other rows do not matter
        self.assertTrue(same(f0, cf(models, d, 0, load=load)))

    def test_the_network_has_no_per_unit_parameters(self):
        m = build_carrier()
        self.assertFalse(any(isinstance(mod, (torch.nn.Embedding, torch.nn.EmbeddingBag)) for mod in m.modules()))
        self.assertEqual(tuple(INPUT_KEYS), ("day_tokens", "day_mask", "month_tokens", "month_mask", "future_tokens", "scale"))

    def test_the_category_is_an_input(self):
        # where the claim stops: the category one-hot is read, so the same history in another category differs
        d, models = synthetic_store(("a",)), ensemble()
        a = cf(models, d, 0, groups="Office")[0]
        b = cf(models, d, 0, groups="Residential")[0]
        self.assertFalse(np.array_equal(a, b))


# ============================================================================= C9 copy pointer
class PointerTests(unittest.TestCase):
    def test_the_pointer_adds_nothing_at_initialisation(self):
        for seed in (0, 1, 2, 3, 4):
            torch.manual_seed(seed)
            m = build_carrier().eval()
            torch.manual_seed(seed)
            base = StructuredLoadNetwork(m.config).eval()                     # same seed: same base weights
            sb = base.state_dict()
            for k, v in m.state_dict().items():
                if k.split(".")[0] not in m.POINTER_MODULES:
                    self.assertTrue(torch.equal(v, sb[k]), k)
            self.assertTrue(torch.equal(m.copy_gate.weight, torch.zeros_like(m.copy_gate.weight)))
            self.assertTrue(torch.equal(m.copy_gate.bias, torch.zeros_like(m.copy_gate.bias)))
            x = random_inputs(seed=seed, dtype=torch.float32)
            a, b = run(m, x), run(base, x)
            self.assertTrue(torch.equal(a["copy_gate"], torch.zeros_like(a["copy_gate"])))
            for key in ("trajectory_kw", "level_kw", "day_fractions", "hour_fractions", "log_trajectory_normalized"):
                self.assertTrue(torch.equal(a[key], b[key]), (seed, key))

    def test_the_pointer_never_changes_the_level(self):
        for seed in SEEDS:
            m, x = carrier(seed), random_inputs(seed=seed)
            a, b = run(m, x), run(base_of(m), x)
            self.assertGreater(float(a["copy_gate"].abs().mean()), 0.1)
            self.assertTrue(torch.equal(a["level_kw"], b["level_kw"]), seed)
            self.assertTrue(torch.equal(a["energy_kwh"], b["energy_kwh"]), seed)
            self.assertGreater(float((a["trajectory_kw"] - b["trajectory_kw"]).abs().max()), 1e-3)   # the shape does change

    def test_the_pointer_is_inactive_without_a_complete_recent_day(self):
        for seed in SEEDS:
            m, x = carrier(seed), random_inputs(seed=seed)
            x["day_tokens"][:, -56:, 24 + seed] = 0.0                          # one unobserved hour on each of the last 56 days
            a, b = run(m, x), run(base_of(m), x)
            self.assertTrue(torch.equal(a["copy_alpha"], torch.zeros_like(a["copy_alpha"])))
            for key in ("trajectory_kw", "level_kw", "day_fractions", "hour_fractions"):
                self.assertTrue(torch.equal(a[key], b[key]), (seed, key))


# ============================================================================= C10 pseudo-origin out-of-sample guarantee
class PseudoOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synthetic_store(("a", "b"))
        cls.models = ensemble()

    def test_a_186_day_gap_keeps_every_pseudo_origin_at_or_after_the_cut(self):
        self.assertEqual((I.K, K_FM, I.STEP, GAP), (6, 6, 744, 186 * 24))
        for cut in (0, 2088, 10536, 123457):
            for o in range(cut + GAP, cut + GAP + 3000, 7):
                self.assertTrue(all(o - k * I.STEP >= cut for k in range(I.K + 1)))
            self.assertLess(cut + GAP - 1 - I.K * I.STEP, cut)                # the gap is tight: one hour less fails

    def test_carrier_forecasts_drop_nothing_after_the_gap_and_never_reach_before_the_cut(self):
        d = self.d
        cut = ORIGIN - GAP
        f = cf(self.models, d, 0)
        self.assertEqual(sorted(f), list(range(7)))
        self.assertTrue(same(cf(self.models, d, 0, cut=cut), f))             # origin = cut + 186 days: nothing dropped
        anchor_ks = I.pseudo_origins(d["load"], 0, ORIGIN)
        for o in range(cut, ORIGIN + 800, 97):
            ks = I.pseudo_origins(d["load"], 0, o, cut=cut)
            self.assertEqual(ks, [k for k in I.pseudo_origins(d["load"], 0, o) if o - k * I.STEP >= cut])
            self.assertTrue(all(o - k * I.STEP >= cut for k in ks))
            if o >= cut + GAP:
                self.assertEqual(ks, anchor_ks)
        self.assertEqual(I.pseudo_origins(d["load"], 0, ORIGIN - 1, cut=cut), list(range(6)))   # k = 6 is one hour early
        with self.assertRaises(ValueError):
            I.pseudo_origins(d["load"], 0, cut - 1, cut=cut)

    def test_training_and_validation_labels_end_at_or_before_the_cut(self):
        d = self.d
        cut = ORIGIN - GAP
        w = make_windows(d["load"], cut)
        self.assertGreater(len(w), 0)
        self.assertLessEqual(int(w[:, 1].max()) + 744, cut)
        labels, v0 = split(w, cut, TrainConfig())
        used = labels != ""
        self.assertTrue(set(labels[used]) == {"train", "val"})
        self.assertLessEqual(int(w[used, 1].max()) + 744, cut)
        self.assertLessEqual(int(w[labels == "train", 1].max()) + 744, v0)
        blanked = cut_data({"load": d["load"], "temp": d["temp"], "types": d["types"]}, cut)
        self.assertTrue(np.isnan(blanked["load"][:, cut:]).all() and np.isnan(blanked["temp"][:, cut:]).all())
        self.assertTrue(np.array_equal(blanked["load"][:, :cut], d["load"][:, :cut]))
        # so the target month [q, q + 744) of every pseudo-origin q >= cut is disjoint from every label hour < cut
        for k in range(7):
            self.assertGreaterEqual(ORIGIN - k * I.STEP, int(w[used, 1].max()) + 744)


if __name__ == "__main__":
    unittest.main(verbosity=2)
