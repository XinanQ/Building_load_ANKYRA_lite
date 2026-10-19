"""Tests of ankyra_lite.carrier.train (CPU only).

Synthetic tests (small synthetic data; always run):
1. a tiny synthetic two-population data set trains for 2 epochs: finite losses, a finite validation criterion, the pool
   rules hold, and the criterion equals the original implementation's;
4. the continuation recipes: the named configurations; a plain continuation from a saved checkpoint (the start's
   criterion reproduced at epoch 0, fixed epochs, final weights kept, a wrong start rejected); the span return (rows
   returned at the population's own density, labels clear of the cut, extra span windows built when the built span
   rows are too few, the training arrays of the extended pool); the proportional validation (hold-out split, the start
   counted as epoch 0, at least 5 epochs); the row view over two arrays.

Tests against the research code (skipped, with the reason, when the research folder is absent):
2. the sampler reproduces the original sampler's draws exactly for the same pools and random stream (and the windows,
   split and caps reproduce the original candidate rules);
3. one optimisation step from the same initial weights on the same batch gives the same parameters as the original
   training-loop code path (tolerance 1e-6); same-seed initialisation is identical;
5. local-data tests on two study populations (Oslo and BDG2; the data stores and the outputs of the span-return /
   proportional-validation study are needed): from each population's store the package reproduces, row for row, the
   daily-grid rows, the returned span (chosen rows, extra rows, counts), the proportional hold-out split (training and
   validation rows, validation days, boundary) and both training pools (every population, order included); the extra
   span windows built by the package equal the research arrays bit for bit;
6. one optimisation step of the span return (Oslo; BDG2, whose returned span includes extra rows) and of the
   proportional validation (BDG2) from the trained start checkpoint, through the package code path (start weights,
   fresh optimiser, sampler stream, extended training arrays, one batch of ``run_epoch``) against the research
   training step: the same draws, the same batch and zero parameter difference; the start weights reproduce their
   recorded validation criterion at epoch 0 (Oslo, relative tolerance 1e-5, as the research check).

The research folder is ANKYRA_LITE_RESEARCH_ROOT (default: the sibling folder HKAD_V3 of this repository). It is
imported read only (no bytecode is written; the research code's own sys.path entries are moved behind this
repository's). Before the data stores are loaded (tests 5 and 6), the tests wait up to 30 minutes until
ANKYRA_LITE_TEST_MIN_FREE_GB (default 5) GB of memory are available, otherwise they are skipped. Run from the
repository root:
    python -B -m unittest tests.test_train -v
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
import json
import math
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch                                          # noqa: E402

torch.set_num_threads(1)
from ankyra_lite.carrier import train as T          # noqa: E402
from ankyra_lite.carrier.model import build_carrier, param_groups, parameter_count   # noqa: E402

RESEARCH = Path(os.environ.get("ANKYRA_LITE_RESEARCH_ROOT", REPO.parent / "HKAD_V3"))
HAVE_RESEARCH = (RESEARCH / "tools_smallnet_stageA_v1.py").exists()
NO_RESEARCH = f"research code not available (set ANKYRA_LITE_RESEARCH_ROOT; looked in {RESEARCH})"
MIN_FREE_GB = float(os.environ.get("ANKYRA_LITE_TEST_MIN_FREE_GB", "5"))
WAIT_S = 1800
REAL = ("oslo", "bdg2")                               # the two study populations of the local-data tests
_O = {}


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


def original():
    """The original modules (read-only import)."""
    if not _O:
        with research_path():
            import tools_smallnet_carrier_v1 as V
            import tools_smallnet_carrier_v3_pooled as T3
            import tools_smallnet_stageA_v1 as SA
            import tools_smallnet_loop2_v1 as L2
            from refinement_v2.model import forecast_loss
        _O.update(V=V, T3=T3, SA=SA, L2=L2, forecast_loss=forecast_loss)
    return _O


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
    load = np.stack([(5.0 + 2 * u) * daily * week * (1 + 0.05 * rng.standard_normal(hours)) for u in range(n_units)])
    load[0, 2000:2700] = np.nan                       # makes some context and label windows ineligible
    load[-1, 3100:3130] = -1.0                        # negative load counts as missing
    load[-1, 4200:4210] = np.nan
    temp = (12 + 8 * np.sin(2 * np.pi * t / 8760) + rng.standard_normal(hours)).astype(np.float32)
    groups = ["Office", "Residential", "Industrial", "Public"][:n_units]
    return {"load": load, "temp": temp, "types": types, "names": [f"u{u}" for u in range(n_units)], "groups": groups, "start_iso": start_iso}


CFG = T.TrainConfig(train_cap=40, val_cap=12, draws_per_epoch=48, batch_size=16, max_epochs=2, torch_threads=1)
DATA = {"alpha": (synthetic_store(2, 6200, "2016-01-01T00:00:00", 1), 6000),
        "beta": (synthetic_store(3, 5200, "2015-12-01T00:00:00", 2), 5000)}       # beta starts 744 h earlier
_S = {}


def sources():
    if not _S:
        _S["alpha"] = T.prepare_source("alpha", DATA["alpha"][0], DATA["alpha"][1], CFG, seed_index=0)
        _S["beta"] = T.prepare_source("beta", DATA["beta"][0], DATA["beta"][1], CFG, seed_index=1)
    return _S


# ----------------------------------------------------------------------------- (1) small training run
class TestTinyTraining(unittest.TestCase):
    def test_two_epochs_two_sources(self):
        S = sources()
        pool = T.build_pool("alpha", S, CFG)
        lim = S["alpha"].start_hour + S["alpha"].boundary
        self.assertEqual(lim, T.absolute_start_hour("2016-01-01T00:00:00") + max(1344 + 744, 6000 - 2232))
        for q, ix in pool.rows.items():                                     # absolute-time limit for every pooled row
            self.assertTrue(np.all(S[q].start_hour + S[q].rows.origin[ix] + 744 <= lim), q)
        self.assertTrue(set(pool.rows["alpha"].tolist()) <= set(np.flatnonzero(S["alpha"].rows.split == "train").tolist()))
        self.assertIn("beta", pool.rows)                                    # borrowed population present
        for q, (n_ref, n_used) in pool.table.items():                       # count matched to the weekly grid
            self.assertEqual(n_ref, int((S[q].start_hour + S[q].count_origins + 744 <= lim).sum()))
            self.assertEqual(n_used, min(n_ref, int((S[q].start_hour + S[q].rows.origin + 744 <= lim).sum())))
        res = T.train_ensemble("alpha", S, CFG, seeds=(0, 1), device="cpu", log=None)
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertEqual(r.epochs_run, 2)
            self.assertEqual(r.parameters, 61613)
            self.assertTrue(all(math.isfinite(h["train_loss"]) and math.isfinite(h["val"]) for h in r.history), r.history)
            self.assertTrue(math.isfinite(r.best_val) and r.best_epoch in (1, 2))
            self.assertAlmostEqual(r.best_val, min(h["val"] for h in r.history), places=12)
            v = S["alpha"].validation
            self.assertEqual(T.validate(r.model, v.arrays, v.index, v.unit, "cpu"), r.best_val)   # best weights restored
            if HAVE_RESEARCH:                                               # the criterion equals the original one
                self.assertEqual(original()["V"].validate(r.model, v.arrays, v.index, v.unit, "cpu"), r.best_val)
        v = S["alpha"].validation
        b = T.make_batch({"alpha": v.arrays}, [("alpha", int(i)) for i in v.index], "cpu")
        with torch.inference_mode():
            preds = [r.model({k: b[k] for k in T.INPUTS})["trajectory_kw"].double().numpy() for r in res]
        ens = T.average_seed_forecasts(preds)
        self.assertTrue(np.isfinite(ens).all() and (ens >= 0).all())
        np.testing.assert_allclose(ens, (preds[0] + preds[1]) / 2, rtol=0, atol=0)
        print(f"\n  [tiny run] pool {pool.table}; seed histories " +
              "; ".join(f"s{r.seed}: " + ", ".join(f"e{h['epoch']} train {h['train_loss']:.4f} val {h['val']:.4f}" for h in r.history) for r in res))

    def test_validation_rules(self):
        fx, pr = T.FixedSpan(), T.ProportionalSpan()
        self.assertEqual(fx.boundary(20000, 1344, 1344, 744), 20000 - 2232)            # 93 days before the cut
        self.assertEqual(fx.boundary(3000, 1344, 1344, 744), 1344 + 744)              # never before the first label end
        self.assertEqual(pr.validation_days(1344 + 24 * 2000, 1344), 93.0)            # long record: unchanged
        self.assertEqual(pr.validation_days(1344 + 24 * 300, 1344), 38.0)             # short record: lower clip
        self.assertAlmostEqual(pr.validation_days(1344 + 24 * 600, 1344), 60.0)
        self.assertEqual(pr.boundary(1344 + 24 * 600, 1344, 1344, 744), 1344 + 24 * 600 - 24 * 60)
        cfg = T.with_validation_rule(CFG, pr)
        w = T.make_windows(DATA["alpha"][0]["load"], 6000)
        lab, v0 = T.split(w, 6000, cfg)
        self.assertEqual(v0, 6000 - 24 * round(pr.validation_days(6000, int(w[:, 1].min()))))
        self.assertTrue(np.all(w[lab == "val", 1] + 744 <= 6000 - 24) and np.all(w[lab == "train", 1] + 744 <= v0))
        self.assertEqual(cfg.describe()["validation"]["rule"], "ProportionalSpan")


# ----------------------------------------------------------------------------- (2) sampler and window rules vs the original
@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class TestAgainstOriginalSampling(unittest.TestCase):
    def test_sampler_matches_original(self):
        SA = original()["SA"]
        rng0 = np.random.default_rng(99)
        units = {"P": rng0.integers(0, 7, 900), "Q1": rng0.integers(0, 3, 400), "Q2": rng0.integers(0, 40, 2500), "Q3": rng0.integers(0, 1, 50)}
        pools = {"P": np.sort(rng0.choice(900, 600, replace=False)), "Q1": np.arange(0, 400, 3), "Q2": np.sort(rng0.choice(2500, 1700, replace=False)),
                 "Q3": np.arange(50)}
        for target, pl in (("P", pools), ("P", {"Q2": pools["Q2"], "P": pools["P"]}), ("P", {"P": pools["P"]}), ("Q1", pools)):
            for seed in (0, 1, 2, 12345):
                a = T.sampler(target, pl, units, np.random.default_rng(seed))
                b = SA.sampler(target, pl, units, np.random.default_rng(seed))
                self.assertEqual(len(a), 16000)
                self.assertEqual([(q, int(r)) for q, r in a], [(q, int(r)) for q, r in b], (target, list(pl), seed))
        pool = T.Pool(target="P", limit=0, rows=pools, table={})                         # a Pool object works the same way
        self.assertEqual(T.sampler("P", pool, units, np.random.default_rng(5)), T.sampler("P", pools, units, np.random.default_rng(5)))

    def test_balanced_matches_original(self):
        V = original()["V"]
        rng0 = np.random.default_rng(3)
        p = np.stack([rng0.integers(0, 9, 3000), rng0.integers(0, 10 ** 6, 3000)], 1)
        for cap in (10, 500, 2999, 3000, 4000):
            for seed in (0, 7):
                np.testing.assert_array_equal(T.balanced(p, cap, np.random.default_rng(seed)), V.balanced(p, cap, np.random.default_rng(seed)))

    def test_windows_split_caps_match_original(self):
        O = original(); V, SA = O["V"], O["SA"]
        d0, cut = DATA["beta"]
        d = T.cut_data(d0, cut)
        # daily grid with binding caps: the original candidate loop (stage-A daily preparation) with small caps
        v0 = SA.v0_of(cut); cand = {"train": [], "val": []}
        for b in range(len(d["names"])):
            for o in range(V.CONTEXT, cut - V.HORIZON + 1, 24):
                sp = "train" if o + V.HORIZON <= v0 else "val" if o >= v0 else None
                if sp and V.eligible(d, b, o):
                    cand[sp].append((b, o))
        rng = np.random.default_rng(SA.SEED + 4)
        tr, va = V.balanced(cand["train"], 40, rng), V.balanced(cand["val"], 12, rng)
        w = T.make_windows(d["load"], cut, grid_hours=24)
        lab, v0_new = T.split(w, cut, CFG)
        self.assertEqual(v0_new, v0)
        self.assertEqual([tuple(x) for x in w[lab == "train"]], cand["train"])
        self.assertEqual([tuple(x) for x in w[lab == "val"]], cand["val"])
        rows, sp = T.select_rows(w, lab, train_cap=40, val_cap=12, rng=np.random.default_rng(CFG.row_seed(24) + 4))
        np.testing.assert_array_equal(rows, np.concatenate([tr, va]))
        self.assertEqual(sp.tolist(), ["train"] * len(tr) + ["val"] * len(va))
        # weekly grid: the original weekly pair builder itself (default caps)
        tr7, va7, v07, _ = V.pairs(d, cut, np.random.default_rng(1))
        w7 = T.make_windows(d["load"], cut, grid_hours=168)
        lab7, b7 = T.split(w7, cut, T.TrainConfig())
        rows7, _ = T.select_rows(w7, lab7, train_cap=16000, val_cap=2000, rng=np.random.default_rng(1))
        self.assertEqual(b7, v07)
        np.testing.assert_array_equal(rows7, np.concatenate([tr7, va7]))


# ----------------------------------------------------------------------------- (3) one optimisation step vs the original loop
@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class TestAgainstOriginalStep(unittest.TestCase):
    def test_same_seed_initialisation(self):
        L2 = original()["L2"]
        torch.manual_seed(0); a = L2.model_for("P")
        torch.manual_seed(0); b = build_carrier()
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(list(sa), list(sb))
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), k)
        self.assertEqual(parameter_count(b), 61613)

    def test_one_step_matches_original_loop(self):
        O = original(); V, T3, L2, forecast_loss = O["V"], O["T3"], O["L2"], O["forecast_loss"]
        S = sources()
        pool = T.build_pool("alpha", S, CFG)
        units = {q: S[q].rows.unit for q in pool.rows}
        arrs = {q: S[q].rows.arrays for q in pool.rows}
        order = T.sampler("alpha", pool, units, np.random.default_rng(0), draws=64)
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = L2.model_for("P")                                            # original network
        with torch.no_grad():                                               # open the copy gates so every pointer parameter moves
            model.copy_gate.weight.normal_(0.0, 0.05); model.copy_gate.bias.fill_(0.1); model.type_bonus.fill_(0.3)
        init = {k: v.clone() for k, v in model.state_dict().items()}
        new = build_carrier()
        new.load_state_dict(init, strict=True)

        # original code path (tools_smallnet_formal3_v1.train, one step)
        opt = torch.optim.AdamW(model.parameters(), lr=T3.LR, weight_decay=V.WD); base_p, ptr_p = L2.param_groups(model)
        model.train()
        b = T3._batch(arrs, order[0:V.BATCH], device); out = model({k: b[k] for k in V.INPUTS})
        loss = forecast_loss(out, torch.where(b["observed"], b["target"], 0.0), b["scale"], b["observed"], kl_weight=V.KL_WEIGHT, compression=True)["loss"]
        self.assertTrue(torch.isfinite(loss))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(base_p, 1.0, error_if_nonfinite=True)
        if ptr_p:
            torch.nn.utils.clip_grad_norm_(ptr_p, 1.0, error_if_nonfinite=True)
        opt.step()

        # package code path
        cfg = T.TrainConfig()
        opt2 = T.make_optimizer(new, cfg)
        new.train()
        b2 = T.make_batch(arrs, order[0:cfg.batch_size], device)
        self.assertEqual(list(b), list(b2))
        for k in b:                                                          # identical batches (targets are NaN where unobserved)
            self.assertEqual(b[k].dtype, b2[k].dtype, k)
            torch.testing.assert_close(b[k], b2[k], rtol=0, atol=0, equal_nan=True, msg=k)
        loss2 = T.train_step(new, opt2, b2, param_groups(new), cfg)
        self.assertLessEqual(abs(float(loss2) - float(loss.detach())), 1e-6)

        after_o, after_n = dict(model.named_parameters()), dict(new.named_parameters())
        self.assertEqual(list(after_o), list(after_n))
        worst, moved = 0.0, 0
        for k, p in after_o.items():
            worst = max(worst, float((p.detach() - after_n[k].detach()).abs().max()))
            moved += int(not torch.equal(p.detach(), init[k]))
        self.assertLessEqual(worst, 1e-6)
        self.assertGreater(moved, 0)
        for k in after_o:                                                    # the copy-pointer group took part in the step
            if k.split(".")[0] in model.POINTER_MODULES:
                self.assertFalse(torch.equal(after_o[k].detach(), init[k]), k)
        print(f"\n  [one step] loss original {float(loss.detach()):.9f} package {float(loss2):.9f}; max |parameter difference| {worst:.3g}; "
              f"{moved}/{len(after_o)} parameter tensors moved")


# ----------------------------------------------------------------------------- (4) continuation recipes (synthetic)
_C = {}


def start_checkpoints() -> dict:
    """A tiny from-scratch run of the target 'alpha' (2 epochs, seeds 0 and 1) saved as start checkpoints."""
    if not _C:
        tmp = tempfile.TemporaryDirectory(prefix="ankyra_lite_train_test_")
        _C["tmp"] = tmp
        res = T.train_ensemble("alpha", sources(), CFG, seeds=(0, 1), device="cpu", log=None, checkpoint_dir=Path(tmp.name) / "alpha")
        _C["results"] = {r.seed: r for r in res}
        _C["template"] = str(Path(tmp.name) / "{target}" / "seed{seed}.pt")
    return _C


def tearDownModule():
    if "tmp" in _C:
        _C["tmp"].cleanup()


def _states_equal(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)


def tiny(name: str, **overrides) -> T.TrainConfig:
    """A named recipe at the tiny synthetic scale (caps, draws and batch of CFG; 2 continuation epochs)."""
    base = dict(train_cap=CFG.train_cap, val_cap=CFG.val_cap, draws_per_epoch=CFG.draws_per_epoch, batch_size=CFG.batch_size,
                torch_threads=1)
    cfg = T.recipe(name, start_checkpoints()["template"], **base)
    if cfg.fixed_epochs is not None:
        cfg = replace(cfg, fixed_epochs=2)
    return replace(cfg, **overrides) if overrides else cfg


class TestContinuationRecipes(unittest.TestCase):
    def test_recipe_configurations(self):
        self.assertEqual(T.recipe("from_scratch"), T.TrainConfig())                    # the defaults are the from-scratch recipe
        d = T.TrainConfig()
        self.assertEqual((d.init_checkpoint, d.fixed_epochs, d.count_start_as_epoch0, d.sampler_seed_offset), (None, None, False, 0))
        self.assertIsInstance(d.validation, T.FixedSpan)
        f = T.recipe("plain_continuation", "ck/{target}/seed{seed}.pt")
        self.assertEqual((f.fixed_epochs, f.sampler_seed_offset, f.lr, f.weight_decay), (8, 10000, 3e-4, 1e-4))
        self.assertIsInstance(f.validation, T.FixedSpan)
        self.assertEqual(f.init_path("oslo", 2), "ck/oslo/seed2.pt")
        r = T.recipe("span_return", "x")
        self.assertEqual((r.fixed_epochs, r.sampler_seed_offset), (8, 10000))
        self.assertEqual((r.validation.days, r.validation.boundary_hours, r.validation.seed, r.validation.holdout), (93.0, 24, 20261019, None))
        self.assertEqual(r.validation.boundary(20000, 1344, 1344, 744), T.FixedSpan().boundary(20000, 1344, 1344, 744))
        v = T.recipe("proportional_validation", "x")
        self.assertEqual((v.fixed_epochs, v.count_start_as_epoch0, v.min_epochs, v.max_epochs, v.patience, v.sampler_seed_offset),
                         (None, True, 5, 30, 5, 10000))
        self.assertEqual(v.validation.holdout, T.ProportionalSpan(0.10, 38.0, 93.0, 24))
        self.assertEqual(v.describe()["validation"]["holdout"]["rule"], "ProportionalSpan")
        self.assertEqual(T.RECIPES, ("from_scratch", "plain_continuation", "span_return", "proportional_validation"))
        with self.assertRaises(ValueError):
            T.recipe("unknown")
        with self.assertRaises(ValueError):                                             # a returned span needs a fixed length
            T.train_one(0, "alpha", sources(), replace(r, fixed_epochs=None), device="cpu", log=None)

    def test_row_concat(self):
        a, b = np.arange(12).reshape(4, 3), 100 + np.arange(6).reshape(2, 3)
        rc = T.RowConcat(a, b)
        self.assertEqual((len(rc), rc.shape, rc.dtype), (6, (6, 3), a.dtype))
        np.testing.assert_array_equal(rc[4], b[0]); np.testing.assert_array_equal(rc[-1], b[1]); np.testing.assert_array_equal(rc[1], a[1])
        np.testing.assert_array_equal(rc[np.array([5, 0, 4, 3])], np.stack([b[1], a[0], b[0], a[3]]))
        np.testing.assert_array_equal(rc[1:5], np.concatenate([a, b])[1:5])
        with self.assertRaises(IndexError):
            rc[6]

    def test_plain_continuation(self):
        C = start_checkpoints()
        cfg = tiny("plain_continuation")
        self.assertEqual(cfg.init_path("alpha", 1), str(Path(C["template"].format(target="alpha", seed=1))))
        S = sources()
        with tempfile.TemporaryDirectory() as tmp:
            res = T.train_ensemble("alpha", S, cfg, seeds=(0, 1), device="cpu", log=None, checkpoint_dir=tmp)
            for r in res:
                start = C["results"][r.seed]
                self.assertEqual(r.history[0]["epoch"], 0)
                self.assertEqual(r.start_val, start.best_val)                      # start weights loaded: their criterion reproduced
                self.assertEqual(r.history[0]["recorded_val"], start.best_val)
                self.assertEqual((r.epochs_run, r.best_epoch, r.stopped_by_patience), (2, 2, False))
                self.assertEqual(r.best_val, r.history[-1]["val"])                 # final weights kept (no selection)
                v = S["alpha"].validation
                self.assertEqual(T.validate(r.model, v.arrays, v.index, v.unit, "cpu"), r.history[-1]["val"])
                ck = torch.load(Path(tmp) / f"seed{r.seed}.pt", map_location="cpu", weights_only=True)
                self.assertEqual(ck["epoch"], 2)
                saved = build_carrier(); saved.load_state_dict(ck["model"])
                self.assertTrue(_states_equal(saved, r.model))
                self.assertFalse(_states_equal(start.model, r.model))             # the continuation moved the weights
                self.assertEqual(r.config["sampler_seed_offset"], 10000)
        with self.assertRaises(ValueError):                                         # a start that does not reproduce its record
            T.train_one(0, "alpha", S, cfg, device="cpu", log=None, expected_start_val=C["results"][0].best_val * 1.01)
        # the same continuation without a validation set: no epoch-0 record, fixed epochs still run, final weights kept
        S2 = dict(S); S2["alpha"] = replace(S["alpha"], validation=None)
        r = T.train_one(0, "alpha", S2, cfg, device="cpu", log=None)
        self.assertEqual((r.epochs_run, r.history[0]["epoch"], r.start_val), (2, 1, None))
        self.assertTrue(math.isnan(r.best_val) and r.history[-1]["val"] is None)

    def test_span_return_rows_and_training(self):
        C = start_checkpoints()
        d0, cut = DATA["alpha"]
        d = T.cut_data(d0, cut)
        H = 744
        for val_cap in (12, 4):                                                   # with val_cap 4 the built span rows are too few
            cfg = tiny("span_return", val_cap=val_cap)
            P = T.prepare_source("alpha", d0, cut, cfg, seed_index=0)
            B = T.prepare_source("beta", DATA["beta"][0], DATA["beta"][1], cfg, seed_index=1, with_validation=False)
            base = T.prepare_source("alpha", d0, cut, replace(cfg, validation=T.FixedSpan()), seed_index=0)
            np.testing.assert_array_equal(P.rows.unit, base.rows.unit); np.testing.assert_array_equal(P.rows.origin, base.rows.origin)
            sp = P.span
            self.assertIsNotNone(sp); self.assertIsNone(B.span)
            n_own = len(T.build_pool("alpha", {"alpha": base}, replace(cfg, validation=T.FixedSpan())).rows["alpha"])
            cand = P.info["candidates"]
            self.assertEqual(sp.info["span_rows_target"], int(round(n_own / cand["train"] * cand["val"])))
            self.assertEqual(len(sp.returned), sp.info["span_rows_target"])
            n = len(P.rows)
            ro = np.concatenate([P.rows.origin, sp.extra.origin])[sp.returned]
            ru = np.concatenate([P.rows.unit, sp.extra.unit])[sp.returned]
            self.assertTrue(np.all(ro >= P.boundary) and np.all(ro + H <= cut - 24))          # in the span, labels 24 h clear of the cut
            self.assertTrue(np.all(P.rows.split[sp.returned[sp.returned < n]] == "val"))
            self.assertEqual(len(set(zip(ru.tolist(), ro.tolist()))), len(ro))              # no window returned twice
            if val_cap == 4:
                self.assertGreater(len(sp.extra), 0)
                built = set(zip(P.rows.unit[P.rows.split == "val"].tolist(), P.rows.origin[P.rows.split == "val"].tolist()))
                self.assertFalse(built & set(zip(sp.extra.unit.tolist(), sp.extra.origin.tolist())))
                ref = T.materialise(d, np.stack([sp.extra.unit, sp.extra.origin], 1))
                for k in T.ARRAY_KEYS:
                    np.testing.assert_array_equal(sp.extra.arrays[k], ref[k])
            S = {"alpha": P, "beta": B}
            pool = T.build_pool("alpha", S, cfg)
            pbase = T.build_pool("alpha", {"alpha": base, "beta": B}, replace(cfg, validation=T.FixedSpan()))
            np.testing.assert_array_equal(pool.rows["alpha"], np.concatenate([pbase.rows["alpha"], sp.returned]))
            np.testing.assert_array_equal(pool.rows["beta"], pbase.rows["beta"])
            self.assertEqual(pool.limit, pbase.limit)
            arrays, units = T.training_arrays(pool, S)
            if len(sp.extra):
                np.testing.assert_array_equal(arrays["alpha"]["target"][n], sp.extra.arrays["target"][0])
                self.assertEqual(int(units["alpha"][n]), int(sp.extra.unit[0]))
            if val_cap == CFG.val_cap:
                r = T.train_one(0, "alpha", S, cfg, device="cpu", log=None)
                self.assertEqual(r.start_val, C["results"][0].best_val)           # the monitor set is the start's validation set
            else:                                                                 # other caps: another validation set, so the
                with self.assertRaises(ValueError):                               # recorded criterion is not reproduced
                    T.train_one(0, "alpha", S, cfg, device="cpu", log=None)
                r = T.train_one(0, "alpha", S, replace(cfg, start_val_rtol=None), device="cpu", log=None)
            self.assertEqual((r.epochs_run, r.returned_rows), (2, len(sp.returned)))
            self.assertTrue(all(math.isfinite(h["train_loss"]) for h in r.history[1:]))
            self.assertGreater(sum(h["own_draws_returned"] for h in r.history[1:]), 0)
            print(f"\n  [span return, val cap {val_cap}] own pool {len(pbase.rows['alpha'])} + returned {len(sp.returned)} "
                  f"({len(sp.chosen)} built, {len(sp.extra)} extra); history " +
                  ", ".join(f"e{h['epoch']} val {h['val']:.4f}" for h in r.history))

    def test_proportional_validation(self):
        C = start_checkpoints()
        d0, cut = DATA["alpha"]
        H = 744
        cfg = tiny("proportional_validation", val_cap=4, max_epochs=6)
        P = T.prepare_source("alpha", d0, cut, cfg, seed_index=0)
        B = T.prepare_source("beta", DATA["beta"][0], DATA["beta"][1], cfg, seed_index=1, with_validation=False)
        sp, hold = P.span, cfg.validation.holdout
        first = int(P.rows.origin.min())
        b = hold.boundary(cut, first, 1344, H)
        self.assertEqual(sp.info["holdout_boundary"], b)
        self.assertEqual(sp.info["holdout_validation_days"], hold.validation_days(cut, first))
        allo = np.concatenate([P.rows.origin, sp.extra.origin])
        self.assertTrue(np.all(allo[sp.returned] + H <= b))                                  # returned windows end before the boundary
        val = P.validation
        self.assertIs(val, sp.validation)
        self.assertTrue(np.all(val.origin >= b) and np.all(val.origin + H <= cut - 24))
        n_val_built = int(((P.rows.split == "val") & (P.rows.origin >= b) & (P.rows.origin + H <= cut - 24)).sum())
        n_val_extra = int(((sp.extra.origin >= b) & (sp.extra.origin + H <= cut - 24)).sum())
        self.assertEqual(len(val.index), n_val_built + n_val_extra)
        S = {"alpha": P, "beta": B}
        r = T.train_one(0, "alpha", S, cfg, device="cpu", log=None)
        self.assertEqual(r.history[0]["epoch"], 0)
        self.assertNotIn("recorded_val", r.history[0])                                       # a new validation set: no start check
        self.assertEqual(r.start_val, T.validate(C["results"][0].model, val.arrays, val.index, val.unit, "cpu"))
        self.assertAlmostEqual(r.best_val, min(h["val"] for h in r.history), places=12)       # epoch 0 is a candidate
        self.assertGreaterEqual(r.epochs_run, 5)                                              # at least 5 epochs
        self.assertEqual(T.validate(r.model, val.arrays, val.index, val.unit, "cpu"), r.best_val)
        if r.best_epoch == 0:
            self.assertTrue(_states_equal(r.model, C["results"][0].model))
        print(f"\n  [proportional validation] boundary {b} ({sp.info['holdout_validation_days']:.1f} days), returned "
              f"{len(sp.returned)}, validation {len(val.index)}; best epoch {r.best_epoch}; history " +
              ", ".join(f"e{h['epoch']} {h['val']:.4f}" for h in r.history))


# ----------------------------------------------------------------------------- (5, 6) local-data tests on two study populations
_R = {}


def _report(msg):
    print("\n    " + msg, file=sys.stderr, flush=True)


def _wait_for_memory():
    try:
        import psutil
    except ImportError:
        _report("psutil is not installed: no memory check before loading the stores")
        return
    t0 = time.time()
    while psutil.virtual_memory().available < MIN_FREE_GB * 2 ** 30:
        if time.time() - t0 > WAIT_S:
            raise unittest.SkipTest(f"less than {MIN_FREE_GB} GB of memory available after {WAIT_S // 60} minutes")
        time.sleep(30)
    _report(f"memory check before loading the stores: {psutil.virtual_memory().available / 2 ** 30:.2f} GB available "
            f"(waited {time.time() - t0:.0f} s)")


def real():
    """The research modules of the span-return / proportional-validation study and, for Oslo and BDG2, the package's
    sources built from the data stores (rows only), the borrowed sources from the recorded research rows, and the BDG2
    store blanked at its cut (cached; skips the calling test class when anything is missing)."""
    if "skip" in _R:
        raise unittest.SkipTest(_R["skip"])
    if _R:
        return _R
    if not HAVE_RESEARCH:
        _R["skip"] = NO_RESEARCH
        raise unittest.SkipTest(NO_RESEARCH)
    try:
        original()
        with research_path():
            SF = importlib.import_module("tools_smallnet_sr_formal_v1")
            VV = importlib.import_module("tools_smallnet_sr_formal_v_v1")
    except Exception as e:                                                   # noqa: BLE001
        _R["skip"] = f"research modules of the span-return study not importable: {type(e).__name__}: {e}"
        raise unittest.SkipTest(_R["skip"])
    V, F2, F3, T3, SA = SF.V, SF.F2, SF.F3, SF.T3, SF.SA
    need = [SF.OUT / f / n for f in REAL for n in ("span_plan.json", "span_rows.npz", "V_plan.json", "V_rows.npz")]
    need += [F3.run_dir(f, "P3s", 0) / n for f in REAL for n in ("best.pt", "TRAINED.json", "STARTED.json")]
    need += [SA.WORK / q / "dn" / "rows.npz" for q in V.FACES] + [V.WORK / q / "formal" / "rows.npz" for q in V.FACES]
    need += [SF.WORK / "bdg2" / "span_extra" / "rows.npz"]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        _R["skip"] = "local-data test: missing " + ", ".join(miss[:3]) + (f" (+{len(miss) - 3} more)" if len(miss) > 3 else "")
        raise unittest.SkipTest(_R["skip"])
    try:
        _wait_for_memory()
    except unittest.SkipTest as exc:                     # cached: the next class skips at once instead of waiting again
        _R["skip"] = str(exc)
        raise
    cfgR, cfgV = T.recipe("span_return", "unused"), T.recipe("proportional_validation", "unused")
    own, cut_store = {}, {}
    try:
        for face in REAL:
            with research_path():
                d = V.raw(face, None)
            cut, idx = int(V.cuts(face)["Cprime"]), V.FACES.index(face)
            own[face] = {"R": T.prepare_source(face, d, cut, cfgR, seed_index=idx, build_arrays=False),
                         "V": T.prepare_source(face, d, cut, cfgV, seed_index=idx, build_arrays=False), "cut": cut,
                         "start_hour_research": T3.start_hours(face)}
            if face == "bdg2":
                cut_store[face] = T.cut_data(d, cut)
            del d
    except FileNotFoundError as e:
        _R["skip"] = f"local-data test: data store missing ({e})"
        raise unittest.SkipTest(_R["skip"])
    borrowed = {}
    for i, q in enumerate(V.FACES):                                          # every population as the research pools see it
        b, o, sp = F2.dn_rows(q)
        _, ow, _ = T3.rows_of(q)
        v0 = int(V.load_json(V.OUT / q / "prep_formal.json")["v0"])
        borrowed[q] = T.Source(name=q, start_hour=T3.start_hours(q), cut=int(V.cuts(q)["Cprime"]), boundary=v0,
                               rows=T.RowSet(unit=b, origin=o, split=sp.astype(str)), validation=None, count_origins=ow, seed_index=i)
    _R.update(SF=SF, VV=VV, own=own, borrowed=borrowed, cut_store=cut_store, cfg={"R": cfgR, "V": cfgV})
    return _R


def real_sources(face: str, arm: str, with_arrays: bool = False) -> dict:
    """Sources for the pool of ``face``: the target built by the package from its store, the others from the recorded
    rows; with ``with_arrays`` every population carries the research's daily-grid arrays (memory maps) and the
    target's extra span windows the research's extra arrays."""
    R = real()
    SA, SF = R["SF"].SA, R["SF"]
    S = dict(R["borrowed"])
    tgt = R["own"][face][arm]
    S[face] = replace(tgt, start_hour=int(R["own"][face]["start_hour_research"]))
    if with_arrays:
        for q in list(S):
            S[q] = replace(S[q], rows=replace(S[q].rows, arrays=SA.arm_arrays(q, "Dn")[0]))
        sp = S[face].span
        if len(sp.extra):
            ex = {k: np.load(SF.WORK / face / "span_extra" / f"{k}.npy", mmap_mode="r") for k in T.ARRAY_KEYS}
            S[face] = replace(S[face], span=replace(sp, extra=replace(sp.extra, arrays=ex)))
    return S


@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class TestSpanPlansAgainstResearch(unittest.TestCase):
    """(5) rows, returned span, hold-out split and pools from the stores of Oslo and BDG2, against the research study."""

    @classmethod
    def setUpClass(cls):
        cls.R = real()

    def test_rows_and_span_plans(self):
        R = self.R; SF = R["SF"]
        for face in REAL:
            with self.subTest(face=face):
                P, Pv = R["own"][face]["R"], R["own"][face]["V"]
                b, o, sp = SF.F2.dn_rows(face)
                for src in (P, Pv):                                          # the daily-grid rows of the fixed split
                    np.testing.assert_array_equal(src.rows.unit, b); np.testing.assert_array_equal(src.rows.origin, o)
                    np.testing.assert_array_equal(src.rows.split, sp.astype(str))
                self.assertEqual(P.start_hour, R["own"][face]["start_hour_research"])
                bw, ow, sw = SF.T3.rows_of(face)                             # the monitor set: the weekly validation rows
                vw = sw == "val"
                np.testing.assert_array_equal(P.validation.unit[P.validation.index], bw[vw])
                np.testing.assert_array_equal(P.validation.origin[P.validation.index], ow[vw])
                plan = json.loads((SF.OUT / face / "span_plan.json").read_text(encoding="utf-8"))
                z = np.load(SF.OUT / face / "span_rows.npz")
                extra = z["extra"].reshape(-1, 2) if "extra" in z.files else np.zeros((0, 2), np.int64)
                np.testing.assert_array_equal(P.span.chosen, z["chosen"])
                np.testing.assert_array_equal(np.stack([P.span.extra.unit, P.span.extra.origin], 1).reshape(-1, 2), extra)
                for mine, theirs in (("own_pool_rows", "own_pool_rows"), ("own_train_candidates", "own_train_candidates"),
                                     ("span_candidates", "span_candidates"), ("span_rows_target", "span_rows_target"),
                                     ("span_rows_from_built_rows", "span_rows_from_existing_dn_val"), ("span_rows_extra", "span_rows_extra_built"),
                                     ("built_span_rows_within_boundary", "existing_dn_val_rows_within_boundary"), ("v0", "v0"), ("cut", "Cprime")):
                    self.assertEqual(P.span.info[mine], plan[theirs], mine)
                self.assertAlmostEqual(P.span.info["sampling_rate"], plan["sampling_rate"], places=15)
                vplan = json.loads((SF.OUT / face / "V_plan.json").read_text(encoding="utf-8"))
                zv = np.load(SF.OUT / face / "V_rows.npz")
                for mine, theirs in (("train_rows", "train_dn"), ("train_extra", "train_extra"), ("val_rows", "val_dn"), ("val_extra", "val_extra")):
                    np.testing.assert_array_equal(Pv.span.holdout_rows[mine], zv[theirs], mine)
                self.assertEqual(Pv.span.info["holdout_validation_days"], vplan["validation_days"])
                self.assertEqual(Pv.span.info["record_days"], vplan["record_days_before_cut"])
                self.assertEqual(Pv.span.info["holdout_boundary"], vplan["boundary_index"])
                self.assertEqual(Pv.span.info["train_rows_returned"], vplan["train_rows_returned"])
                self.assertEqual(Pv.span.info["validation_rows"], vplan["validation_rows"])
                _report(f"[{face}] rows {len(P.rows)}, own pool {plan['own_pool_rows']}, returned span {len(P.span.returned)} "
                        f"({len(P.span.chosen)} built + {len(P.span.extra)} extra); proportional hold-out {vplan['validation_days']} days, "
                        f"{vplan['train_rows_returned']} returned, {vplan['validation_rows']} validation windows: identical")

    def test_pools_row_for_row(self):
        R = self.R; SF, VV = R["SF"], R["VV"]
        for face in REAL:
            for arm in ("R", "V"):
                with self.subTest(face=face, arm=arm):
                    pool = T.build_pool(face, real_sources(face, arm), R["cfg"][arm])
                    if arm == "R":
                        pl, lim, arrs, units, info = SF.pools(face, "R")
                    else:
                        pl, lim, arrs, units, va, vu = VV.setup(face)
                        val = R["own"][face]["V"].validation
                        np.testing.assert_array_equal(val.unit, vu)
                        del va
                    self.assertEqual(list(pool.rows), list(pl))
                    for q in pl:
                        np.testing.assert_array_equal(pool.rows[q], pl[q], err_msg=f"{face} {arm} {q}")
                    self.assertEqual(pool.limit, int(lim))
                    _report(f"[{face} {arm}] pool identical, order included: " + ", ".join(f"{q} {len(v)}" for q, v in pool.rows.items()))

    def test_extra_span_arrays(self):
        R = self.R; SF = R["SF"]
        sp = R["own"]["bdg2"]["R"].span
        self.assertGreater(len(sp.extra), 0)
        pick = np.unique(np.linspace(0, len(sp.extra) - 1, 6).astype(int))
        rows = np.stack([sp.extra.unit[pick], sp.extra.origin[pick]], 1)
        mine = T.materialise(R["cut_store"]["bdg2"], rows, token_dtype=T.TrainConfig().token_dtype)
        rz = np.load(SF.WORK / "bdg2" / "span_extra" / "rows.npz")
        np.testing.assert_array_equal(np.stack([rz["building"], rz["origin"]], 1)[pick], rows)
        for k in T.ARRAY_KEYS:
            ref = np.asarray(np.load(SF.WORK / "bdg2" / "span_extra" / f"{k}.npy", mmap_mode="r")[pick])
            self.assertEqual(ref.dtype, mine[k].dtype, k)
            np.testing.assert_array_equal(ref, mine[k], err_msg=k)
        _report(f"[bdg2] {len(pick)} extra span windows built by the package equal the research arrays bit for bit")


@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class TestContinuationStepAgainstResearch(unittest.TestCase):
    """(6) one optimisation step from the trained start checkpoint, package code path vs the research training step."""

    @classmethod
    def setUpClass(cls):
        cls.R = real()

    def template(self):
        return str(self.R["SF"].F3.OUT / "{target}" / "P3s_s{seed}" / "best.pt")

    def test_start_checkpoint_epoch0(self):
        R = self.R; SF = R["SF"]; PWF, T3, F3 = SF.PWF, SF.T3, SF.F3
        face, seed = "oslo", 0
        cfg = T.recipe("span_return", self.template(), torch_threads=1)
        model, start = T.initial_model(seed, face, cfg, device="cpu")
        ref, tr = PWF.load_p3s(face, seed)
        self.assertTrue(_states_equal(model, ref))
        self.assertEqual(start.val, tr["best_val"])
        self.assertEqual(start.origin, str(F3.run_dir(face, "P3s", seed) / "best.pt"))
        started = json.loads((F3.run_dir(face, "P3s", seed) / "STARTED.json").read_text(encoding="utf-8"))
        pool = T.build_pool(face, real_sources(face, "R"), replace(cfg, validation=T.FixedSpan()))      # the start's own pool
        self.assertEqual({q: int(len(v)) for q, v in pool.rows.items()}, started["pool"])
        self.assertEqual(pool.limit, started["abs_lim"])
        varr = T3._arrays(face); bW, _, sW = T3.rows_of(face); vai = np.flatnonzero(sW == "val")
        v0 = T.validate(model, varr, vai, bW, "cpu")
        rel = abs(v0 - tr["best_val"]) / abs(tr["best_val"])
        self.assertLessEqual(rel, cfg.start_val_rtol)
        _report(f"[oslo] start checkpoint: criterion at epoch 0 {v0:.9g} (CPU) vs recorded {tr['best_val']:.9g}; relative {rel:.2g}")

    def one_step(self, face: str, arm: str):
        R = self.R; SF, VV = R["SF"], R["VV"]
        O = original(); V, T3, SA, L2, forecast_loss = O["V"], O["T3"], O["SA"], O["L2"], O["forecast_loss"]
        PWF, L4 = SF.PWF, SF.L4
        seed, cpu = 0, torch.device("cpu")
        name = {"R": "span_return", "V": "proportional_validation"}[arm]
        cfg = T.recipe(name, self.template(), torch_threads=1)
        # research: start checkpoint, fresh optimiser, pool, sampler stream, one step of the training loop
        if arm == "R":
            pl, lim, arrs, units, info = SF.pools(face, "R")
        else:
            pl, lim, arrs, units, va, vu = VV.setup(face)
            del va
        m_o, tr = PWF.load_p3s(face, seed)
        opt_o = L4.optimizer_for(m_o); base_p, ptr_p = L2.param_groups(m_o)
        order_o = SA.sampler(face, pl, units, np.random.default_rng(L4.CONT_SEED + seed))
        b_o = T3._batch(arrs, order_o[0:V.BATCH], cpu)
        m_o.train(); out = m_o({k: b_o[k] for k in V.INPUTS})
        loss_o = forecast_loss(out, torch.where(b_o["observed"], b_o["target"], 0.0), b_o["scale"], b_o["observed"], kl_weight=V.KL_WEIGHT, compression=True)["loss"]
        opt_o.zero_grad(set_to_none=True); loss_o.backward(); torch.nn.utils.clip_grad_norm_(base_p, 1.0, error_if_nonfinite=True)
        if ptr_p:
            torch.nn.utils.clip_grad_norm_(ptr_p, 1.0, error_if_nonfinite=True)
        opt_o.step()
        # package: the same pieces train_one runs
        S = real_sources(face, arm, with_arrays=True)
        pool = T.build_pool(face, S, cfg)
        arrays, units_n = T.training_arrays(pool, S)
        model, start = T.initial_model(seed, face, cfg, device="cpu")
        init = {k: v.clone() for k, v in model.state_dict().items()}
        opt = T.make_optimizer(model, cfg)
        order = T.sampler(face, pool, units_n, np.random.default_rng(cfg.sampler_seed_offset + seed), draws=cfg.draws_per_epoch,
                          own_fraction=cfg.own_fraction)
        self.assertEqual([(q, int(r)) for q, r in order], [(q, int(r)) for q, r in order_o])
        b_n = T.make_batch(arrays, order[:cfg.batch_size], cpu)
        self.assertEqual(list(b_n), list(b_o))
        for k in b_o:
            self.assertEqual(b_o[k].dtype, b_n[k].dtype, k)
            torch.testing.assert_close(b_n[k], b_o[k], rtol=0, atol=0, equal_nan=True, msg=k)
        n_ret = int(pool.returned.get(face, 0))
        ret = set(pool.rows[face][len(pool.rows[face]) - n_ret:].tolist()) if n_ret else set()
        drawn_ret = sum(1 for q, r in order[:cfg.batch_size] if q == face and int(r) in ret)
        drawn_extra = sum(1 for q, r in order[:cfg.batch_size] if q == face and int(r) >= len(S[face].rows))
        total, seen = T.run_epoch(model, opt, order[:cfg.batch_size], arrays, param_groups(model), cfg, cpu)
        self.assertEqual(seen, cfg.batch_size)
        self.assertEqual(total, float(loss_o.detach()) * cfg.batch_size)
        after_o, after_n = dict(m_o.named_parameters()), dict(model.named_parameters())
        self.assertEqual(list(after_o), list(after_n))
        worst, moved = 0.0, 0
        for k, p in after_o.items():
            worst = max(worst, float((p.detach() - after_n[k].detach()).abs().max()))
            moved += int(not torch.equal(after_n[k].detach(), init[k]))
            self.assertTrue(torch.equal(p.detach(), after_n[k].detach()), k)
        self.assertGreater(moved, 0)
        _report(f"[{face} {name}] one step from the start checkpoint: loss {float(loss_o.detach()):.9f} (both); "
                f"max |parameter difference| {worst:.3g}; {moved}/{len(after_n)} tensors moved; batch: {drawn_ret} returned-span "
                f"draws ({drawn_extra} extra rows); pool {sum(len(v) for v in pool.rows.values())} rows")

    def test_one_step_span_return_oslo(self):
        self.one_step("oslo", "R")

    def test_one_step_span_return_bdg2(self):
        self.one_step("bdg2", "R")

    def test_one_step_proportional_validation_bdg2(self):
        self.one_step("bdg2", "V")


if __name__ == "__main__":
    unittest.main(verbosity=2)
