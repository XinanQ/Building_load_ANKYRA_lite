"""Tests of the loss weighting of recipe Rw in ankyra_lite.carrier.train (CPU only; small synthetic data, no data store
and no checkpoint file is read, nothing is written).

Synthetic tests (always run):
1. Reduction to the unweighted training. On a tiny two-population span return (the target's returned span includes
   extra windows, so its training arrays are a row view), two continuation epochs of three batches each from the same
   start weights: (a) without weighting, ``train_one`` equals the version-0.1 training loop written out in this file
   (the unweighted loss, backward pass, clipping per parameter group, AdamW step) bit for bit; (b) the weighted path
   with every weight 1 equals (a) bit for bit (weights, training losses, validation criteria), with and without the
   weighting rule in the configuration; (c) the training record of the unweighted configuration has no
   loss-weighting key.
2. The weights rule on a synthetic pool of day tokens: the noise estimate against a direct count of the hour pairs
   (the 168-pair boundary included); per population the draw-weighted mean weight is 1; the floor, the raw-weight cap
   and the weight of windows without an estimate; NaN outside the pool; the result does not depend on the chunk size
   or on the row view; the draw probabilities have the sampler's per-population shares and equal unit masses, and
   agree with the sampler's draw frequencies.
3. Causality: the noise estimate of a window built by the package's feature builder does not change when the load at
   or after its origin changes (and does change when the load of its last 21 days changes); the weights of a pool do
   not change when the label arrays, the day tokens outside the block the rule reads, or the rows outside the pool
   change; end to end (stores -> sources -> pool -> weights), the weights do not change when the target's load at or
   after its latest pool origin changes (the labels of the pool's windows).
4. Recipe Rw: the named configurations ("Rw" = "span_return_weighted" = the span return with ``NoiseWeights()``,
   "R" = "span_return"), the frozen constants, the training record; a tiny weighted continuation (alignment check of
   the first batch of each epoch, finite losses, the recorded mean drawn weight), the weights computed once by
   ``train_ensemble`` and shared by the seeds, non-uniform weights change the trained weights, misaligned weights and
   weights missing on a pool row are refused, and the weighting needs the log1p-compressed loss.

Against the research code (skipped, with the reason, when it is absent): the weights equal those of the frozen rule
(tools_smallnet_pwformal_v1.weights_rule) and of the study's stage-2 rule (tools_lite_v2_train_v1.weights_rule) bit for
bit, on plain arrays and through the row view, with and without borrowed populations; the noise estimate, the draw
probabilities, the weighted quantile, the expected batch weight and the weighted loss equal the frozen functions of
tools_smallnet_pw_v1; the constants equal the frozen record of the study. The research folder is
ANKYRA_LITE_RESEARCH_ROOT (default: the sibling folder HKAD_V3 of this repository), imported read only.

Against version 0.1 (skipped unless ANKYRA_LITE_REFERENCE_TRAIN is the path of a version-0.1
ankyra_lite/carrier/train.py, or ANKYRA_LITE_REFERENCE_REPO a version-0.1 repository; imported read only): its
``train_one`` gives the same weights and history as both paths of test 1 bit for bit, and every recipe of version 0.1
has the same training record.

Run from the repository root:
    python -B -m unittest tests.test_train_rw -v
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True                       # never write bytecode next to the read-only research code
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""             # CPU only
import contextlib
import importlib
import importlib.util
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch                                          # noqa: E402

torch.set_num_threads(1)
from ankyra_lite.carrier import train as T          # noqa: E402
from ankyra_lite.carrier.model import build_carrier, forecast_loss, param_groups   # noqa: E402

RESEARCH = Path(os.environ.get("ANKYRA_LITE_RESEARCH_ROOT") or os.environ.get("ANKYRA_LITE_RESEARCH_DIR") or REPO.parent / "HKAD_V3")
HAVE_RESEARCH = (RESEARCH / "tools_smallnet_pw_v1.py").exists()
NO_RESEARCH = f"research code not available (set ANKYRA_LITE_RESEARCH_ROOT; looked in {RESEARCH})"
REFERENCE_TRAIN = os.environ.get("ANKYRA_LITE_REFERENCE_TRAIN") or (
    str(Path(os.environ["ANKYRA_LITE_REFERENCE_REPO"]) / "ankyra_lite" / "carrier" / "train.py")
    if os.environ.get("ANKYRA_LITE_REFERENCE_REPO") else None)   # a version-0.1 repository (staging only)
NO_REFERENCE = "no version-0.1 training module (set ANKYRA_LITE_REFERENCE_TRAIN to its train.py)"
FROZEN_JSON = RESEARCH / "analysis_outputs" / "lite_upgrade_v2" / "FROZEN_v2.json"
TINY = dict(train_cap=40, val_cap=4, draws_per_epoch=48, batch_size=16, torch_threads=1)   # val cap 4: extra span windows
EPOCHS = 2
_O, _S, _REF = {}, {}, {}


# ----------------------------------------------------------------------------- helpers
@contextlib.contextmanager
def research_path():
    """Research folder importable (appended); the research modules' own sys.path insertions are moved to the end, so
    this repository's packages always resolve first."""
    snap = list(sys.path)
    if str(RESEARCH) not in sys.path:
        sys.path.append(str(RESEARCH))
    try:
        yield
    finally:
        sys.path[:] = snap + [p for p in sys.path if p not in snap]


def research() -> dict:
    """The frozen weighting modules of the study (read-only import); skips the calling test when they do not import."""
    if "skip" in _O:
        raise unittest.SkipTest(_O["skip"])
    if not _O:
        try:
            with research_path():
                PW = importlib.import_module("tools_smallnet_pw_v1")
                PWF = importlib.import_module("tools_smallnet_pwformal_v1")
        except Exception as e:                                               # noqa: BLE001
            _O["skip"] = f"research weighting modules not importable: {type(e).__name__}: {e}"
            raise unittest.SkipTest(_O["skip"])
        _O.update(PW=PW, PWF=PWF)
    return _O


def reference() -> object:
    """The version-0.1 training module (``ANKYRA_LITE_REFERENCE_TRAIN``), imported read only under a private name;
    skips the calling test when the file is missing or is not version 0.1 (it has the loss weighting)."""
    if not _REF:
        if not Path(REFERENCE_TRAIN).is_file():
            raise unittest.SkipTest(f"no file {REFERENCE_TRAIN}")
        spec = importlib.util.spec_from_file_location("_ankyra_lite_train_reference_0_1", REFERENCE_TRAIN)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                  # dataclasses resolve their module through sys.modules
        spec.loader.exec_module(mod)
        _REF["module"] = mod
    if hasattr(_REF["module"], "NoiseWeights"):
        raise unittest.SkipTest(f"{REFERENCE_TRAIN} is not the version-0.1 training module")
    return _REF["module"]


def synthetic_store(n_units: int, hours: int, start_iso: str, seed: int) -> dict:
    """Hourly load with daily and weekly cycles, a long gap, negative values (= missing) and a holiday."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    from datetime import datetime
    wd0 = datetime.fromisoformat(start_iso).weekday()
    types = ((wd0 + t // 24) % 7).astype(np.int64)
    types[(t // 24) % 97 == 50] = 7
    daily = 1.0 + 0.5 * np.sin(2 * np.pi * (t % 24) / 24)
    week = np.where(types >= 5, 0.7, 1.0)
    noise = np.stack([0.02 + 0.08 * u for u in range(n_units)])[:, None]     # unit-specific noise: weights differ
    load = np.stack([(5.0 + 2 * u) * daily * week for u in range(n_units)]) * (1 + noise * rng.standard_normal((n_units, hours)))
    load[0, 2000:2700] = np.nan                       # makes some context and label windows ineligible
    load[-1, 3100:3130] = -1.0                        # negative load counts as missing
    load[-1, 4200:4210] = np.nan
    temp = (12 + 8 * np.sin(2 * np.pi * t / 8760) + rng.standard_normal(hours)).astype(np.float32)
    groups = ["Office", "Residential", "Industrial", "Public"][:n_units]
    return {"load": load, "temp": temp, "types": types, "names": [f"u{u}" for u in range(n_units)], "groups": groups, "start_iso": start_iso}


DATA = {"alpha": (synthetic_store(3, 6200, "2016-01-01T00:00:00", 1), 6000),
        "beta": (synthetic_store(3, 5200, "2015-12-01T00:00:00", 2), 5000)}       # beta starts 744 h earlier


def tiny(module, name: str, **overrides):
    """A named recipe of ``module`` at the tiny synthetic scale (two fixed epochs of three batches)."""
    return replace(module.recipe(name, None, **TINY), fixed_epochs=EPOCHS, **overrides)


def span_sources(module=T, data=None) -> dict:
    """Target 'alpha' (with its returned span) and the borrowed 'beta', prepared by ``module`` for the span return."""
    data = data or DATA
    cfg = tiny(module, "span_return")
    return {"alpha": module.prepare_source("alpha", data["alpha"][0], data["alpha"][1], cfg, seed_index=0),
            "beta": module.prepare_source("beta", data["beta"][0], data["beta"][1], cfg, seed_index=1, with_validation=False)}


def sources() -> dict:
    if not _S:
        _S.update(span_sources())
    return _S


def start_state() -> dict:
    """Start weights of the continuations: a seeded network with open copy gates (so every pointer parameter moves)."""
    torch.manual_seed(7)
    m = build_carrier()
    with torch.no_grad():
        m.copy_gate.weight.normal_(0.0, 0.05); m.copy_gate.bias.fill_(0.1); m.type_bonus.fill_(0.3)
    return {k: v.clone() for k, v in m.state_dict().items()}


def uniform_weights(pool, units) -> dict:
    """Weight 1 on every pool row, NaN elsewhere."""
    out = {}
    for q, ix in pool.rows.items():
        w = np.full(len(units[q]), np.nan)
        w[ix] = 1.0
        out[q] = w
    return out


def same_state(a, b) -> bool:
    sa = a if isinstance(a, dict) else a.state_dict()
    sb = b if isinstance(b, dict) else b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)


def without_seconds(history) -> list:
    return [{k: v for k, v in h.items() if k != "seconds"} for h in history]


def reference_loop(seed: int, S: dict, cfg, init: dict):
    """The training loop of version 0.1 written out (``train_one`` with fixed epochs): seeding, the start weights, a
    fresh AdamW, the sampler stream, per batch the unweighted log1p loss, backward pass, clipping of each parameter
    group and the optimiser step; the validation criterion at epoch 0 and after every epoch. Returns (model, epoch
    training losses, criteria)."""
    pool = T.build_pool("alpha", S, cfg)
    arrays, units = T.training_arrays(pool, S)
    T.configure(seed, cfg)
    model = build_carrier()
    model.load_state_dict(init, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    groups = param_groups(model)
    rng = np.random.default_rng(cfg.sampler_seed_offset + seed)
    v = S["alpha"].validation
    vals, losses = [T.validate(model, v.arrays, v.index, v.unit, "cpu", cfg.batch_size)], []
    for _ in range(cfg.fixed_epochs):
        order = T.sampler("alpha", pool, units, rng, draws=cfg.draws_per_epoch, own_fraction=cfg.own_fraction)
        model.train()
        total, seen = 0.0, 0
        for s0 in range(0, len(order), cfg.batch_size):
            part = order[s0:s0 + cfg.batch_size]
            b = T.make_batch(arrays, part, "cpu")
            out = model({k: b[k] for k in T.INPUTS})
            loss = forecast_loss(out, torch.where(b["observed"], b["target"], 0.0), b["scale"], b["observed"],
                                 kl_weight=cfg.kl_weight, compression=True)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for g in groups:
                if g:
                    torch.nn.utils.clip_grad_norm_(g, cfg.grad_clip, error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(part); seen += len(part)
        losses.append(total / seen)
        vals.append(T.validate(model, v.arrays, v.index, v.unit, "cpu", cfg.batch_size))
    return model, losses, vals


def batch_tokens(arrays: dict, pairs) -> np.ndarray:
    """The day tokens of a batch as ``make_batch`` stacks them (float16 rows cast to float32)."""
    rows = [np.asarray(arrays[q]["day_tokens"][int(r)]) for q, r in pairs]
    return np.ascontiguousarray(np.stack([v.astype(np.float32) if v.dtype == np.float16 else v for v in rows]))


def token_pool(seed: int = 20261020) -> dict:
    """A synthetic pool of day tokens: target 'tgt' (150 built rows + 40 extra rows behind a row view), borrowed 'src'
    (80 rows); row-specific noise levels, 5% unobserved hours, and rows without any observed hour in the read block
    (no noise estimate). Labels are random."""
    rng = np.random.default_rng(seed)

    def tok(n):
        t = np.zeros((n, 366, 69), np.float16)
        level = np.exp(rng.normal(0.0, 0.6, n))[:, None, None]
        t[:, :, :24] = np.arcsinh(level * rng.gamma(2.0, 0.5, (n, 366, 24))).astype(np.float16)
        t[:, :, 24:48] = (rng.random((n, 366, 24)) > 0.05).astype(np.float16)
        t[:, :, 48:] = rng.normal(0, 1, (n, 366, 21)).astype(np.float16)
        t[: max(1, n // 20), 338:366, 24:48] = 0
        return t

    def arrays(n):
        return {"day_tokens": tok(n), "target": rng.gamma(2.0, 1.0, (n, 744)).astype(np.float32),
                "observed": rng.random((n, 744)) > 0.1}
    a, b, ex = arrays(150), arrays(80), arrays(40)
    view = {k: T.RowConcat(a[k], ex[k]) for k in a}
    flat = {k: np.concatenate([a[k], ex[k]]) for k in a}
    units = {"tgt": rng.integers(0, 12, 190), "src": rng.integers(0, 9, 80)}
    pool = {"tgt": np.concatenate([np.sort(rng.choice(150, 90, replace=False)), 150 + np.arange(0, 40, 2)]),
            "src": np.sort(rng.choice(80, 60, replace=False))}
    return {"view": {"tgt": view, "src": b}, "flat": {"tgt": flat, "src": b}, "units": units, "pool": pool}


# ----------------------------------------------------------------------------- (1) reduction to the unweighted training
class TestReduction(unittest.TestCase):
    def test_unweighted_and_uniform_weights_equal_version_0_1_loop(self):
        S, init = sources(), start_state()
        cfg = tiny(T, "span_return")
        self.assertGreater(len(S["alpha"].span.extra), 0)                    # the pool includes the row view
        pool = T.build_pool("alpha", S, cfg)
        arrays, units = T.training_arrays(pool, S)
        self.assertIsInstance(arrays["alpha"]["day_tokens"], T.RowConcat)
        ref_model, ref_losses, ref_vals = reference_loop(0, S, cfg, init)
        off = T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None)
        ones = uniform_weights(pool, units)
        w1 = T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None, weights=ones)
        w1r = T.train_one(0, "alpha", S, replace(cfg, noise_weighting=T.NoiseWeights()), init=init, device="cpu", log=None,
                          weights=ones)
        self.assertFalse(same_state(off.model, init))                        # the continuation moved the weights
        for name, r in (("weighting off", off), ("w = 1", w1), ("w = 1, rule configured", w1r)):
            self.assertTrue(same_state(r.model, ref_model), name)            # bit for bit
            self.assertEqual([h["train_loss"] for h in r.history[1:]], ref_losses, name)
            self.assertEqual([h["val"] for h in r.history], ref_vals, name)
            self.assertEqual(r.epochs_run, EPOCHS)
        self.assertIsNone(off.weights_record)
        self.assertNotIn("mean_drawn_weight", off.history[1])
        self.assertEqual([h["mean_drawn_weight"] for h in w1.history[1:]], [1.0] * EPOCHS)
        self.assertEqual(w1.weights_record["rule"], "given")
        self.assertNotIn("noise_weighting", off.config)                      # the 0.1 training record
        self.assertEqual(w1r.config["noise_weighting"]["rule"], "NoiseWeights")
        print(f"\n  [reduction] {EPOCHS} epochs x {cfg.draws_per_epoch // cfg.batch_size} batches; losses {ref_losses}; "
              f"weighting off and w = 1 equal the version-0.1 loop bit for bit")

    def test_weighted_batch_loss_with_unit_weights(self):
        rng = np.random.default_rng(3)
        for n in (1, 16, 64):
            raw = torch.tensor(rng.normal(0.3, 0.5, n), dtype=torch.float32, requires_grad=True)
            a = torch.log1p(raw.clamp_min(0)).mean()                         # forecast_loss(..., compression=True)["loss"]
            ga, = torch.autograd.grad(a, raw)
            b = T.weighted_log1p_loss(raw, torch.ones(n, dtype=torch.float32))
            gb, = torch.autograd.grad(b, raw)
            self.assertTrue(torch.equal(a, b) and torch.equal(ga, gb), n)
        raw = torch.tensor([0.5, -0.2, 2.0, 0.1])
        w = torch.tensor([2.0, 1.0, 0.5, 4.0])
        ell = torch.log1p(raw.clamp_min(0))
        torch.testing.assert_close(T.weighted_log1p_loss(raw, w), (w * ell).sum() / w.sum(), rtol=0, atol=0)
        torch.testing.assert_close(T.weighted_log1p_loss(raw, 3 * w), T.weighted_log1p_loss(raw, w))   # self-normalised


# ----------------------------------------------------------------------------- (2) the weights rule
def sigma2_direct(tok: np.ndarray, rule=T.NoiseWeights()) -> float:
    """The noise estimate of one window counted pair by pair."""
    x = np.sinh(tok[:, :24].astype(np.float32)).astype(np.float64)
    ob = tok[:, 24:48].astype(np.float32) > rule.observed_threshold
    num, n = 0.0, 0
    for day in range(rule.context_days[0] + rule.lag_days, rule.context_days[1]):
        for h in range(24):
            if ob[day, h] and ob[day - rule.lag_days, h]:
                num += (x[day, h] - x[day - rule.lag_days, h]) ** 2
                n += 1
    return num / n if n >= rule.min_pairs else math.nan


class TestWeightsRule(unittest.TestCase):
    def test_noise_estimate(self):
        P = token_pool()
        tok = P["flat"]["tgt"]["day_tokens"][:30]
        s2 = T.noise_sigma2(tok)
        ref = np.array([sigma2_direct(t) for t in tok])
        np.testing.assert_array_equal(np.isnan(s2), np.isnan(ref))
        self.assertGreater(int(np.isnan(s2).sum()), 0)
        np.testing.assert_allclose(s2[~np.isnan(s2)], ref[~np.isnan(ref)], rtol=1e-6, atol=0)   # float32 sinh, any order
        np.testing.assert_allclose(T.noise_sigma2(tok.astype(np.float32)), s2, rtol=1e-12, atol=0)   # stored or batch tokens
        edge = np.zeros((2, 366, 69), np.float16)                                       # exactly 168 pairs, then 167
        edge[:, :, :24] = np.arcsinh(np.arange(24) / 10 + 1).astype(np.float16)
        edge[:, 338:366, 24:32] = 1
        edge[:, 345:366, :24] = np.arcsinh(np.arange(24) / 5 + 1).astype(np.float16)
        edge[1, 365, 24] = 0
        e = T.noise_sigma2(edge)
        self.assertTrue(math.isfinite(e[0]) and math.isnan(e[1]))
        self.assertAlmostEqual(float(e[0]), sigma2_direct(edge[0]), delta=1e-6 * float(e[0]))

    def test_weights_properties(self):
        P = token_pool()
        pl, units = P["pool"], P["units"]
        W, rec = T.noise_weights("tgt", pl, P["view"], units)
        p = T.draw_probabilities("tgt", pl, units)
        f, neutral = rec["floor_sigma2"], rec["neutral_raw_weight"]
        for q, ix in pl.items():
            w = W[q]
            self.assertTrue(np.isfinite(w[ix]).all() and (w[ix] > 0).all(), q)
            outside = np.setdiff1d(np.arange(len(w)), ix)
            self.assertTrue(np.isnan(w[outside]).all(), q)                       # NaN outside the pool
            self.assertAlmostEqual(float((p[q] * w[ix]).sum() / p[q].sum()), 1.0, places=12)   # draw-weighted mean 1
            c = rec["sources"][q]["normaliser"]
            s2 = T.pool_sigma2(P["flat"][q], ix)
            ok = np.isfinite(s2)
            np.testing.assert_allclose(w[ix][ok], c * np.maximum(s2[ok], f) ** -0.5, rtol=1e-15, atol=0)
            np.testing.assert_array_equal(w[ix][~ok], np.full(int((~ok).sum()), c * neutral))
            low = ok & (s2 <= f)
            self.assertTrue(low.any())
            self.assertEqual(len(set(w[ix][low].tolist())), 1)                   # capped at f^(-1/2)
            self.assertTrue(np.all(w[ix][ok] <= c * f ** -0.5 * (1 + 1e-12)))
            self.assertAlmostEqual(rec["sources"][q]["draw_share"], float(p[q].sum()), places=15)
            self.assertEqual(rec["sources"][q]["rows"], len(ix))
        S = np.concatenate([T.pool_sigma2(P["flat"][q], ix) for q, ix in pl.items()])
        Pp = np.concatenate([p[q] for q in pl])
        ok = np.isfinite(S)
        self.assertLess(float(Pp[ok][S[ok] < f].sum() / Pp[ok].sum()), 0.25)    # f is the weighted 25th percentile
        self.assertGreaterEqual(float(Pp[ok][S[ok] <= f].sum() / Pp[ok].sum()), 0.25)
        self.assertAlmostEqual(rec["undefined_sigma2_draw_share"], float(Pp[~ok].sum()), places=15)
        self.assertGreater(rec["undefined_sigma2_draw_share"], 0)
        self.assertTrue(0 < rec["kish_ess_ratio_draws"] <= 1)
        self.assertEqual(rec["rule"], {"rule": "NoiseWeights", "gamma": 0.5, "floor_quantile": 0.25, "min_pairs": 168,
                                       "context_days": (338, 366), "lag_days": 7, "observed_threshold": 0.5})
        json.dumps(rec)                                                         # JSON-ready record

    def test_chunks_and_row_view(self):
        P = token_pool()
        W0, r0 = T.noise_weights("tgt", P["pool"], P["flat"], P["units"])
        for arrays, chunk in ((P["view"], 7), (P["view"], 512), (P["flat"], 1), (P["flat"], 10 ** 6)):
            W1, r1 = T.noise_weights("tgt", P["pool"], arrays, P["units"], chunk=chunk)
            for q in W0:
                np.testing.assert_array_equal(W1[q], W0[q])
            self.assertEqual(r1, r0)
        ix = P["pool"]["tgt"][::-1].copy()                                       # any row order: values follow the rows
        np.testing.assert_array_equal(T.pool_sigma2(P["view"]["tgt"], ix, chunk=9)[::-1], T.pool_sigma2(P["flat"]["tgt"], ix[::-1]))

    def test_draw_probabilities_match_sampler(self):
        P = token_pool()
        units = P["units"]
        pools = (P["pool"], {"tgt": P["pool"]["tgt"]}, {"tgt": P["pool"]["tgt"], "src": P["pool"]["src"], "third": np.arange(0, 80, 5)})
        units = dict(units, third=units["src"])
        for pl in pools:
            p = T.draw_probabilities("tgt", pl, units)
            others = [q for q in pl if q != "tgt"]
            self.assertAlmostEqual(float(p["tgt"].sum()), 0.5 if others else 1.0, places=15)
            for q in others:
                self.assertAlmostEqual(float(p[q].sum()), 0.5 / len(others), places=15)
            for q, ix in pl.items():                                             # equal total probability per unit
                b = units[q][ix]
                mass = [float(p[q][b == u].sum()) for u in np.unique(b)]
                np.testing.assert_allclose(mass, np.full(len(mass), mass[0]), rtol=1e-12)
            rng, counts, n = np.random.default_rng(11), {q: np.zeros(len(ix)) for q, ix in pl.items()}, 0
            pos = {q: {int(r): i for i, r in enumerate(ix)} for q, ix in pl.items()}
            for _ in range(10):                                                  # 160,000 draws
                for q, r in T.sampler("tgt", pl, units, rng):
                    counts[q][pos[q][int(r)]] += 1; n += 1
            for q, ix in pl.items():                                             # per (population, unit), within 5 SD
                b = units[q][ix]
                for u in np.unique(b):
                    pr = float(p[q][b == u].sum())
                    self.assertLess(abs(counts[q][b == u].sum() / n - pr), 5 * math.sqrt(pr * (1 - pr) / n), (q, u))

    def test_expected_weight_and_alignment(self):
        P = token_pool()
        pl, units = P["pool"], P["units"]
        W, rec = T.noise_weights("tgt", pl, P["view"], units)
        pairs = [("tgt", int(r)) for r in pl["tgt"][:20]] + [("src", int(r)) for r in pl["src"][:12]]
        pairs += [("tgt", int(r)) for r in pl["tgt"][-6:]]                      # extra rows behind the row view
        ew = T.expected_weight(batch_tokens(P["view"], pairs), [q for q, _ in pairs], rec)   # float32, as in training
        w = np.array([W[q][r] for q, r in pairs])
        np.testing.assert_allclose(ew, w, rtol=1e-12, atol=0)
        self.assertFalse(np.allclose(ew, np.roll(w, 1), rtol=1e-9, atol=0))   # a shifted row is detected


# ----------------------------------------------------------------------------- (3) causality
class TestCausality(unittest.TestCase):
    def test_noise_estimate_reads_only_the_context(self):
        store, _ = DATA["alpha"]
        fn = T._default_feature_fn()
        rng = np.random.default_rng(5)
        for unit, origin in ((1, 4000), (2, 5000), (0, 4512)):
            tok = np.asarray(fn(store, unit, origin)["day_tokens"], np.float16)[None]
            s2 = T.noise_sigma2(tok)
            self.assertTrue(np.isfinite(s2).all())
            later = dict(store, load=store["load"].copy())
            later["load"][:, origin:] = rng.uniform(0, 50, later["load"][:, origin:].shape)   # labels and everything after
            self.assertTrue(np.array_equal(T.noise_sigma2(np.asarray(fn(later, unit, origin)["day_tokens"], np.float16)[None]), s2))
            recent = dict(store, load=store["load"].copy())
            recent["load"][unit, origin - 24 * 21:origin] *= rng.uniform(0.5, 1.5, 24 * 21)
            self.assertFalse(np.array_equal(T.noise_sigma2(np.asarray(fn(recent, unit, origin)["day_tokens"], np.float16)[None]), s2))

    def test_weights_read_only_the_context_block_of_pool_rows(self):
        P = token_pool()
        pl, units = P["pool"], P["units"]
        W0, r0 = T.noise_weights("tgt", pl, P["flat"], units)
        rng = np.random.default_rng(9)
        changed = {}
        for q, a in P["flat"].items():
            m = {k: np.array(v, copy=True) for k, v in a.items()}
            m["target"] = rng.gamma(1.0, 3.0, m["target"].shape).astype(np.float32)      # labels
            m["observed"] = ~m["observed"]
            m["day_tokens"][:, :338] = rng.normal(0, 1, m["day_tokens"][:, :338].shape)   # days outside the block
            m["day_tokens"][:, :, 48:] = rng.normal(0, 1, m["day_tokens"][:, :, 48:].shape)   # other token columns
            outside = np.setdiff1d(np.arange(len(units[q])), pl[q])
            m["day_tokens"][outside] = rng.normal(0, 1, m["day_tokens"][outside].shape)   # rows outside the pool
            changed[q] = m
        W1, r1 = T.noise_weights("tgt", pl, changed, units)
        for q in W0:
            np.testing.assert_array_equal(W1[q], W0[q])
        self.assertEqual(r1, r0)

    def test_pool_weights_do_not_read_labels(self):
        S = sources()
        cfg = tiny(T, "span_return")
        pool = T.build_pool("alpha", S, cfg)
        arrays, units = T.training_arrays(pool, S)
        W0, _ = T.noise_weights("alpha", pool, arrays, units)
        origins = np.concatenate([S["alpha"].rows.origin, S["alpha"].span.extra.origin])[pool.rows["alpha"]]
        o_max = int(origins.max())
        store, cut = DATA["alpha"]
        load = store["load"].copy()
        ok = np.isfinite(load) & (load >= 0)
        ok[:, :o_max] = False
        load[ok] *= np.random.default_rng(13).uniform(0.5, 2.0, int(ok.sum()))  # the labels of the pool's windows
        S2 = span_sources(T, dict(DATA, alpha=(dict(store, load=load), cut)))
        pool2 = T.build_pool("alpha", S2, cfg)
        arrays2, units2 = T.training_arrays(pool2, S2)
        for q in pool.rows:
            np.testing.assert_array_equal(pool2.rows[q], pool.rows[q])
        own = pool.rows["alpha"]
        self.assertFalse(np.array_equal(arrays2["alpha"]["target"][own], arrays["alpha"]["target"][own], equal_nan=True))
        W2, _ = T.noise_weights("alpha", pool2, arrays2, units2)
        for q in W0:
            np.testing.assert_array_equal(W2[q], W0[q])
        print(f"\n  [causality] pool {dict((q, len(ix)) for q, ix in pool.rows.items())}; labels at and after hour {o_max} "
              f"changed; weights unchanged")


# ----------------------------------------------------------------------------- (4) recipe Rw
class TestRecipeRw(unittest.TestCase):
    def test_configurations(self):
        rw = T.recipe("Rw", "ck/{target}/seed{seed}.pt")
        self.assertEqual(rw, T.recipe("span_return_weighted", "ck/{target}/seed{seed}.pt"))
        self.assertEqual(rw, replace(T.recipe("span_return", "ck/{target}/seed{seed}.pt"), noise_weighting=T.NoiseWeights()))
        self.assertEqual(T.recipe("R", "x"), T.recipe("span_return", "x"))
        self.assertEqual((rw.fixed_epochs, rw.sampler_seed_offset, rw.lr, rw.weight_decay, rw.batch_size, rw.draws_per_epoch),
                         (8, 10000, 3e-4, 1e-4, 64, 16000))
        self.assertIsInstance(rw.validation, T.ReturnSpan)
        self.assertEqual(T.RECIPES, ("from_scratch", "plain_continuation", "span_return", "proportional_validation"))
        self.assertEqual(T.WEIGHTED_RECIPES, ("span_return_weighted",))
        self.assertEqual(T.RECIPE_KEYS, {"R": "span_return", "Rw": "span_return_weighted"})
        nw = T.NoiseWeights()
        self.assertEqual((nw.gamma, nw.floor_quantile, nw.min_pairs, nw.context_days, nw.lag_days, nw.observed_threshold),
                         (0.5, 0.25, 168, (338, 366), 7, 0.5))
        for name in T.RECIPES:
            self.assertIsNone(T.recipe(name, "x").noise_weighting)
            self.assertNotIn("noise_weighting", T.recipe(name, "x").describe())
        d = rw.describe()
        self.assertEqual(d["noise_weighting"], {"rule": "NoiseWeights", "gamma": 0.5, "floor_quantile": 0.25, "min_pairs": 168,
                                                "context_days": (338, 366), "lag_days": 7, "observed_threshold": 0.5})
        json.dumps(d)
        with self.assertRaises(ValueError):
            T.recipe("Rx")

    def test_weighted_continuation(self):
        S, init = sources(), start_state()
        cfg = tiny(T, "Rw")
        pool = T.build_pool("alpha", S, cfg)
        arrays, units = T.training_arrays(pool, S)
        W, rec = T.noise_weights("alpha", pool, arrays, units, draws=cfg.draws_per_epoch, own_fraction=cfg.own_fraction)
        self.assertGreater(len(set(np.round(W["alpha"][pool.rows["alpha"]], 12).tolist())), 1)   # weights differ
        r = T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None)
        self.assertEqual(r.weights_record, rec)
        self.assertEqual(r.config["noise_weighting"]["rule"], "NoiseWeights")
        self.assertEqual(r.epochs_run, EPOCHS)
        for h in r.history[1:]:
            self.assertTrue(math.isfinite(h["train_loss"]) and math.isfinite(h["val"]) and h["mean_drawn_weight"] > 0, h)
        off = T.train_one(0, "alpha", S, tiny(T, "R"), init=init, device="cpu", log=None)
        self.assertFalse(same_state(r.model, off.model))                     # the weights change the training
        ens = T.train_ensemble("alpha", S, cfg, seeds=(0, 1), inits={0: init, 1: init}, device="cpu", log=None)
        self.assertEqual([e.weights_record for e in ens], [rec, rec])          # computed once, shared by the seeds
        self.assertTrue(same_state(ens[0].model, r.model))
        self.assertEqual(without_seconds(ens[0].history), without_seconds(r.history))
        print(f"\n  [Rw] floor {rec['floor_sigma2']:.4g}; Kish ratio {rec['kish_ess_ratio_draws']:.3f}; normalisers "
              + ", ".join(f"{q} {v['normaliser']:.4g}" for q, v in rec["sources"].items())
              + "; history " + ", ".join(f"e{h['epoch']} train {h.get('train_loss', float('nan')):.4f}" for h in r.history))

    def test_refusals(self):
        S, init = sources(), start_state()
        cfg = tiny(T, "Rw")
        pool = T.build_pool("alpha", S, cfg)
        arrays, units = T.training_arrays(pool, S)
        W, rec = T.noise_weights("alpha", pool, arrays, units, draws=cfg.draws_per_epoch, own_fraction=cfg.own_fraction)
        bad = {q: w.copy() for q, w in W.items()}
        ix = pool.rows["alpha"]
        bad["alpha"][ix] = np.roll(W["alpha"][ix], 1)                          # misaligned rows
        with self.assertRaises(RuntimeError):
            T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None, weights=bad, weights_record=rec)
        hole = {q: w.copy() for q, w in W.items()}
        hole["beta"][pool.rows["beta"][0]] = np.nan                             # a pool row without a weight
        with self.assertRaises(ValueError):
            T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None, weights=hole)
        with self.assertRaises(ValueError):                                     # the weighting needs the log1p loss
            T.train_one(0, "alpha", S, replace(cfg, log1p_compression=False), init=init, device="cpu", log=None)
        model = build_carrier()
        b = T.make_batch(arrays, [("alpha", int(r)) for r in ix[:4]])
        with self.assertRaises(ValueError):
            T.train_step(model, T.make_optimizer(model, cfg), b, param_groups(model), replace(cfg, log1p_compression=False),
                         weights=np.ones(4))


# ----------------------------------------------------------------------------- against the research code
@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class TestAgainstResearchWeights(unittest.TestCase):
    def _compare(self, W1, r1, W0, r0, label):
        self.assertEqual(sorted(W1), sorted(W0), label)
        for q in W0:
            np.testing.assert_array_equal(W1[q], W0[q], err_msg=f"{label} {q}")   # bit for bit (NaN outside the pool)
        for k in ("floor_sigma2", "neutral_raw_weight", "undefined_sigma2_draw_share", "sources", "kish_ess_ratio_draws",
                  "weight_max_over_median", "pool", "gamma", "floor_quantile"):
            self.assertEqual(r1[k], r0[k], f"{label} {k}")

    def test_weights_equal_frozen_rule(self):
        PWF = research()["PWF"]
        P = token_pool()
        cases = (("tgt", P["pool"]), ("src", P["pool"]), ("tgt", {"tgt": P["pool"]["tgt"]}))
        for target, pl in cases:
            W0, r0 = PWF.weights_rule(target, pl, P["flat"], P["units"])
            for arrays, chunk in ((P["flat"], 512), (P["view"], 7)):
                W1, r1 = T.noise_weights(target, pl, arrays, P["units"], chunk=chunk)
                self._compare(W1, r1, W0, r0, f"{target} {list(pl)} chunk {chunk}")

    def test_weights_equal_study_stage2_rule(self):
        try:
            with research_path():
                LT = importlib.import_module("tools_lite_v2_train_v1")
        except Exception as e:                                                   # noqa: BLE001
            self.skipTest(f"the study's training tool is not importable: {type(e).__name__}: {e}")
        P = token_pool()
        for target in ("tgt", "src"):
            W0, r0 = LT.weights_rule(target, P["pool"], P["flat"], P["units"])
            W1, r1 = T.noise_weights(target, P["pool"], P["view"], P["units"], chunk=13)
            self._compare(W1, r1, W0, r0, f"study rule, target {target}")

    def test_frozen_functions(self):
        PW = research()["PW"]
        P = token_pool()
        tok = P["flat"]["tgt"]["day_tokens"]
        np.testing.assert_array_equal(T.noise_sigma2(tok), PW.sigma2_from_tokens(tok))
        np.testing.assert_array_equal(T.noise_sigma2(tok.astype(np.float32)), PW.sigma2_from_tokens(tok.astype(np.float32)))
        for target, pl in (("tgt", P["pool"]), ("src", P["pool"]), ("tgt", {"tgt": P["pool"]["tgt"]})):
            p1, p0 = T.draw_probabilities(target, pl, P["units"]), PW.draw_probs(target, pl, P["units"])
            for q in pl:
                np.testing.assert_array_equal(p1[q], p0[q])
        rng = np.random.default_rng(4)
        for n in (1, 7, 300):
            v, w = rng.gamma(2.0, 1.0, n), rng.random(n)
            v[: n // 3] = v[0]                                                   # ties
            for q in (0.0, 0.25, 0.5, 0.999, 1.0):
                self.assertEqual(T.weighted_quantile(v, w, q), PW.wquantile(v, w, q))
        W, rec = T.noise_weights("tgt", P["pool"], P["view"], P["units"])
        pairs = [("tgt", int(r)) for r in P["pool"]["tgt"][-20:]] + [("src", int(r)) for r in P["pool"]["src"][:20]]
        bt = batch_tokens(P["view"], pairs)
        np.testing.assert_array_equal(T.expected_weight(bt, [q for q, _ in pairs], rec),
                                      PW.expected_weight(bt, [q for q, _ in pairs], rec))
        raw = torch.tensor(rng.normal(0.2, 0.6, 64), dtype=torch.float32)
        wt = torch.tensor(rng.gamma(2.0, 0.5, 64), dtype=torch.float32)
        torch.testing.assert_close(T.weighted_log1p_loss(raw, wt), PW.weighted_log1p_loss({"per_window": raw}, wt)[0], rtol=0, atol=0)

    def test_constants_equal_frozen_record(self):
        PW = research()["PW"]
        nw = T.NoiseWeights()
        self.assertEqual((nw.gamma, nw.floor_quantile, nw.min_pairs, nw.context_days), (PW.GAMMA, PW.FLOOR_Q, PW.MIN_PAIRS, PW.CTX_DAYS))
        if not FROZEN_JSON.exists():
            self.skipTest(f"no frozen record at {FROZEN_JSON}")
        fr = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))["carrier_recipe_Rw"]
        c = fr["pw_tool_constants"]
        self.assertEqual((nw.gamma, nw.floor_quantile, nw.min_pairs, list(nw.context_days)), (c["GAMMA"], c["FLOOR_Q"], c["MIN_PAIRS"], c["CTX_DAYS"]))
        self.assertEqual(list(range(nw.context_days[0] + nw.lag_days, nw.context_days[1]))[::20], fr["noise_estimate"]["day_marks"])
        self.assertEqual(nw.min_pairs, fr["noise_estimate"]["undefined_if_pairs_below"])
        rw = T.recipe("Rw", "x")
        o = fr["optimiser"]
        self.assertEqual((rw.lr, rw.weight_decay, rw.batch_size, rw.draws_per_epoch, rw.fixed_epochs, rw.sampler_seed_offset),
                         (o["lr"], o["weight_decay"], o["batch"], o["draws_per_epoch"], fr["epochs"], 10000))


# ----------------------------------------------------------------------------- against version 0.1
@unittest.skipUnless(REFERENCE_TRAIN, NO_REFERENCE)
class TestAgainstVersion01(unittest.TestCase):
    def test_same_training_as_version_0_1(self):
        R = reference()
        S, init = sources(), start_state()
        cfg = tiny(T, "span_return")
        pool = T.build_pool("alpha", S, cfg)
        _, units = T.training_arrays(pool, S)
        SR = span_sources(R)
        ref = R.train_one(0, "alpha", SR, tiny(R, "span_return"), init=init, device="cpu", log=None)
        off = T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None)
        w1 = T.train_one(0, "alpha", S, cfg, init=init, device="cpu", log=None, weights=uniform_weights(pool, units))
        self.assertTrue(same_state(off.model, ref.model))
        self.assertTrue(same_state(w1.model, ref.model))
        self.assertEqual(without_seconds(off.history), without_seconds(ref.history))
        self.assertEqual([h.get("train_loss") for h in w1.history], [h.get("train_loss") for h in ref.history])
        self.assertEqual([h["val"] for h in w1.history], [h["val"] for h in ref.history])
        self.assertEqual(json.dumps(off.config), json.dumps(ref.config))       # the training record is unchanged
        self.assertEqual((off.best_epoch, off.best_val, off.epochs_run, off.pool_table, off.returned_rows),
                         (ref.best_epoch, ref.best_val, ref.epochs_run, ref.pool_table, ref.returned_rows))

    def test_recipe_records_unchanged(self):
        R = reference()
        self.assertEqual(T.RECIPES, R.RECIPES)
        for name in R.RECIPES:
            for init in (None, "runs/{target}/seed{seed}.pt"):
                self.assertEqual(json.dumps(T.recipe(name, init).describe()), json.dumps(R.recipe(name, init).describe()), name)
        self.assertEqual(json.dumps(T.TrainConfig().describe()), json.dumps(R.TrainConfig().describe()))


if __name__ == "__main__":
    unittest.main()
