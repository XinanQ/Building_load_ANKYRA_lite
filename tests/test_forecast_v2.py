"""Tests of the forecast API with the origin-continuity layer: the layer switched off reproduces ankyra-lite 0.1.1
bit for bit (field by field in this process, and against a separate 0.1.1 process), zero constants reproduce it too,
the readout rule (energy, level and peak before the layer; the interval around the delivered trajectory), the
disclosure quantity, sign and zeros, the state-window bypass, the history-only configuration, the missing-hour
bypass, the absence of information from after the origin, ensembles of six carriers averaged at every issue time,
the release switches, the argument checks and the result type.

Synthetic data and untrained carriers (``build_carrier()``, seeds 0-5); no checkpoint or study data is needed.  The
comparison with 0.1.1 needs a checkout of ankyra-lite 0.1.1 (``ANKYRA_LITE_REFERENCE_REPO``) and is skipped without
it.  Run from the repository root:  python -m unittest tests.test_forecast_v2 -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import math
import pickle
import subprocess
import types as _types
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch.set_num_threads(1)

import ankyra_lite                                                                    # noqa: E402
from ankyra_lite import _release, build_carrier, carrier_forecasts, readouts          # noqa: E402
from ankyra_lite import anchor as A                                                   # noqa: E402
from ankyra_lite import continuation as C                                             # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                            # noqa: E402
from ankyra_lite.forecast import ForecastResult, _masked_load, forecast, peak         # noqa: E402

O, STEP, CONTEXT, HORIZON = 12000, 744, 1344, 744
GROUP, SIGMA, DST = "Office", 0.25, "EU"
REFERENCE_FORECAST_SHA256 = "a35ef9d63ac9af5ca1fcf5ff7121ace87bce198a829c3c8e673d6a7d5dd2198b"   # ankyra_lite/forecast.py, 0.1.1
FIELDS_0_1_1 = ["trajectory_kw", "level_kw", "peak_kw", "state_window", "state_rule", "anchored", "carrier"]
KW = dict(group=GROUP, temp_sigma_std=SIGMA, dst_region=DST)


def carrier(seed=0):
    torch.manual_seed(seed)
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


@functools.lru_cache(maxsize=None)
def shared():
    """The carrier, the record, its history and the forecasts with the layer off and on (computed once)."""
    m = carrier(0)
    load, temp, types, start = record()
    h = history_at(load, temp, types, start)
    off = forecast(h, m, continuation=False, **KW)
    on = forecast(h, m, continuation=True, **KW)
    return _types.SimpleNamespace(m=m, load=load, temp=temp, types=types, start=start, h=h, off=off, on=on)


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference_root():
    root = os.environ.get("ANKYRA_LITE_REFERENCE_REPO")
    if not root or not (Path(root) / "ankyra_lite" / "forecast.py").is_file():
        return None
    return Path(root)


def reference_forecast_module():
    """``ankyra_lite/forecast.py`` of the 0.1.1 checkout, loaded as ``ankyra_lite._forecast_0_1_1`` beside the package
    under test (it then runs on this package's carrier and anchoring modules, which 0.2.0 leaves unchanged)."""
    path = reference_root() / "ankyra_lite" / "forecast.py"
    if sha256(path) != REFERENCE_FORECAST_SHA256:
        raise AssertionError(f"{path} is not the forecast.py of ankyra-lite 0.1.1")
    name = "ankyra_lite._forecast_0_1_1"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def assert_same_result(tc, new, ref, msg):
    """Every 0.1.1 field of two results, the arrays bit for bit."""
    tc.assertEqual(np.asarray(new.trajectory_kw).dtype, np.asarray(ref.trajectory_kw).dtype, msg)
    tc.assertEqual(np.asarray(new.trajectory_kw).tobytes(), np.asarray(ref.trajectory_kw).tobytes(), msg)
    tc.assertEqual((new.level_kw, new.peak_kw, new.energy_kwh, new.state_window, new.state_rule),
                   (ref.level_kw, ref.peak_kw, ref.energy_kwh, ref.state_window, ref.state_rule), msg)
    tc.assertEqual(sorted(new.carrier), sorted(ref.carrier), msg)
    for k in ref.carrier:
        tc.assertEqual(new.carrier[k].tobytes(), ref.carrier[k].tobytes(), (msg, k))
    for f in dataclasses.fields(A.AnkyraForecast):
        np.testing.assert_equal(getattr(new.anchored, f.name), getattr(ref.anchored, f.name), err_msg=f"{msg}: {f.name}")


def configurations(s):
    """(label, history, carriers, keyword arguments) of the 0.1.1 comparison."""
    F = carrier_forecasts(s.m, s.load, s.temp, s.types, GROUP, s.start, 0, O)
    return [("carrier network", s.h, s.m, {}),
            ("precomputed mapping with a cut", s.h, F, {"cut": O - 3 * STEP}),
            ("history-only", s.h, None, {}),
            ("state window (off)", state_history("off"), s.m, {}),
            ("state window (micro-load), no state rule", state_history("micro"), s.m, {"state_rule": None})]


_REFERENCE_PROCESS = r"""
import os, pickle, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)
import numpy as np
import torch
torch.set_num_threads(1)
import ankyra_lite as L
from ankyra_lite import anchor as A
if L.__version__ != "0.1.1" or not os.path.abspath(L.__file__).startswith(root):
    raise SystemExit(f"not the 0.1.1 checkout: {L.__version__} {L.__file__}")
job = pickle.load(sys.stdin.buffer)
m = L.build_carrier()
m.load_state_dict({k: torch.from_numpy(v) for k, v in job["state_dict"].items()}, strict=True)
m.eval()
out = {}
for name, (hist, use_carrier, state_rule) in job["jobs"].items():
    r = L.forecast(A.History(*hist), m if use_carrier else None, state_rule=state_rule, **job["kw"])
    out[name] = {"trajectory_kw": r.trajectory_kw, "level_kw": r.level_kw, "peak_kw": r.peak_kw, "energy_kwh": r.energy_kwh,
                 "state_window": r.state_window, "state_rule": r.state_rule, "carrier": dict(r.carrier),
                 "daily_means_kw": r.daily_means_kw, "within_day_kw": r.within_day_kw, "level_weights": r.level_weights}
sys.stdout.buffer.write(pickle.dumps(out))
"""


class TestLayerOffIsVersion011(unittest.TestCase):
    def test_off_equals_the_0_1_1_forecast_field_by_field(self):
        if reference_root() is None:
            self.skipTest("no ankyra-lite 0.1.1 checkout (set ANKYRA_LITE_REFERENCE_REPO)")
        R = reference_forecast_module()
        self.assertEqual([f.name for f in dataclasses.fields(R.ForecastResult)], FIELDS_0_1_1)
        s = shared()
        for label, hist, carr, kw in configurations(s):
            ref = R.forecast(hist, carr, **KW, **kw)
            self.assertEqual(type(ref).__module__, "ankyra_lite._forecast_0_1_1")
            runs = {"False": forecast(hist, carr, continuation=False, **KW, **kw), "None": forecast(hist, carr, continuation=None, **KW, **kw)}
            if _release.CONTINUATION_DEFAULT is False:
                runs["release default"] = forecast(hist, carr, **KW, **kw)
            for how, new in runs.items():
                assert_same_result(self, new, ref, f"{label}, continuation={how}")
                self.assertIs(new.trajectory_before_continuation_kw, new.trajectory_kw, label)
                self.assertIsNone(new.continuation, label)

    def test_off_equals_a_separate_0_1_1_process(self):
        root = reference_root()
        if root is None:
            self.skipTest("no ankyra-lite 0.1.1 checkout (set ANKYRA_LITE_REFERENCE_REPO)")
        s = shared()
        hists = {"ordinary": s.h, "off": state_history("off"), "micro": state_history("micro")}
        jobs = {"carrier network": ("ordinary", True, "seasonal_naive"), "history-only": ("ordinary", False, "seasonal_naive"),
                "state window (off)": ("off", True, "seasonal_naive"), "state window (micro-load), no state rule": ("micro", True, None)}
        payload = {"state_dict": {k: v.detach().cpu().numpy().copy() for k, v in s.m.state_dict().items()}, "kw": KW,
                   "jobs": {n: ((hists[h].load_kw, hists[h].temperature_c, hists[h].day_types, hists[h].start_timestamp), c, r)
                            for n, (h, c, r) in jobs.items()}}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.pop("PYTHONPATH", None)
        p = subprocess.run([sys.executable, "-B", "-c", _REFERENCE_PROCESS, str(root)], input=pickle.dumps(payload), capture_output=True,
                           env=env, cwd=str(root), timeout=1800)
        self.assertEqual(p.returncode, 0, p.stderr.decode(errors="replace")[-3000:])
        ref = pickle.loads(p.stdout)
        for name, (h, c, r) in jobs.items():
            new = forecast(hists[h], s.m if c else None, state_rule=r, continuation=False, **KW)
            want = ref[name]
            self.assertEqual(new.trajectory_kw.tobytes(), want["trajectory_kw"].tobytes(), name)
            self.assertEqual((new.level_kw, new.peak_kw, new.energy_kwh, new.state_window, new.state_rule),
                             (want["level_kw"], want["peak_kw"], want["energy_kwh"], want["state_window"], want["state_rule"]), name)
            self.assertEqual(sorted(new.carrier), sorted(want["carrier"]), name)
            for k, v in want["carrier"].items():
                self.assertEqual(new.carrier[k].tobytes(), v.tobytes(), (name, k))
            self.assertEqual(np.asarray(new.daily_means_kw).tobytes(), np.asarray(want["daily_means_kw"]).tobytes(), name)
            self.assertEqual(np.asarray(new.within_day_kw).tobytes(), np.asarray(want["within_day_kw"]).tobytes(), name)
            self.assertEqual(new.level_weights, want["level_weights"], name)

    def test_zero_constants_reproduce_the_layer_off(self):
        s = shared()
        z = forecast(s.h, s.m, continuation=C.ZERO, **KW)
        self.assertEqual(z.trajectory_kw.tobytes(), s.off.trajectory_kw.tobytes())
        self.assertEqual(z.trajectory_before_continuation_kw.tobytes(), s.off.trajectory_kw.tobytes())
        self.assertEqual((z.level_kw, z.peak_kw, z.energy_kwh), (s.off.level_kw, s.off.peak_kw, s.off.energy_kwh))
        self.assertEqual(z.continuation.constants, C.ZERO)
        self.assertTrue((z.continuation.delta == 0).all())
        self.assertEqual(readouts.continuation_energy_change(z), 0.0)


class TestReadoutRule(unittest.TestCase):
    def test_level_energy_peak_and_the_readouts_before_the_layer_are_unchanged(self):
        s = shared()
        on, off = s.on, s.off
        self.assertEqual((on.level_kw, on.energy_kwh, on.peak_kw, on.state_window, on.state_rule),
                         (off.level_kw, off.energy_kwh, off.peak_kw, off.state_window, off.state_rule))
        self.assertEqual(on.trajectory_before_continuation_kw.tobytes(), off.trajectory_kw.tobytes())
        for k in off.carrier:
            self.assertEqual(on.carrier[k].tobytes(), off.carrier[k].tobytes(), k)
        for f in dataclasses.fields(A.AnkyraForecast):
            np.testing.assert_equal(getattr(on.anchored, f.name), getattr(off.anchored, f.name), err_msg=f.name)
        self.assertEqual(on.peak_kw, peak(on.trajectory_before_continuation_kw, s.h.load_kw, s.h.day_types))
        raw_on, raw_off = readouts.raw_trajectory(on), readouts.raw_trajectory(off)
        self.assertEqual(raw_on.tobytes(), raw_off.tobytes())
        self.assertTrue(np.array_equal(np.maximum(raw_on, 0.0), on.trajectory_before_continuation_kw))
        self.assertEqual(readouts.projection_energy_increase(on), readouts.projection_energy_increase(off))
        for a, b in zip(readouts.blocks(raw_on), readouts.blocks(raw_off)):
            self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())

    def test_the_delivered_trajectory_is_the_layer_on_the_trajectory_before_it(self):
        s = shared()
        on, off = s.on, s.off
        self.assertFalse(np.array_equal(on.trajectory_kw, off.trajectory_kw))           # the layer acts on this window
        tp, rec = C.apply(off.trajectory_kw, _masked_load(s.h), s.h.day_types, O)
        self.assertEqual(on.trajectory_kw.tobytes(), tp.tobytes())
        got = on.continuation
        self.assertIsInstance(got, C.ContinuationRecord)
        for name in C.ContinuationRecord._fields:
            np.testing.assert_equal(getattr(got, name), getattr(rec, name), err_msg=name)
        self.assertEqual((got.bypass, got.constants, got.digest), (None, C.CONSTANTS, C.DIGEST))
        self.assertTrue(np.isfinite(on.trajectory_kw).all() and (on.trajectory_kw >= 0).all())
        self.assertTrue(np.array_equal(on.trajectory_before_continuation_kw == 0, on.trajectory_kw == 0))
        E = C.envelope()
        ratio = on.trajectory_kw[off.trajectory_kw > 0] / off.trajectory_kw[off.trajectory_kw > 0]
        Eh = E[off.trajectory_kw > 0]
        self.assertTrue((ratio >= np.exp(-Eh) * (1 - 1e-15)).all() and (ratio <= np.exp(Eh) * (1 + 1e-15)).all())

    def test_disclosure_the_delivered_mean_differs_from_the_energy_readout(self):
        s = shared()
        on, off = s.on, s.off
        change = readouts.continuation_energy_change(on)
        self.assertEqual(change, float(on.trajectory_kw.sum() - on.trajectory_before_continuation_kw.sum()))
        self.assertNotEqual(change, 0.0)
        self.assertNotEqual(float(on.trajectory_kw.mean()), on.level_kw)
        self.assertEqual(readouts.continuation_energy_change(off), 0.0)
        self.assertEqual(readouts.continuation_energy_change(off.anchored), 0.0)
        F = on.trajectory_before_continuation_kw
        self.assertLessEqual(abs(change), float(np.sum(F * (np.exp(C.envelope()) - 1))) * (1 + 1e-12))
        total = on.energy_kwh + readouts.projection_energy_increase(on) + change
        self.assertAlmostEqual(float(on.trajectory_kw.sum()), total, delta=1e-9 * HORIZON * float(np.abs(F).max()))

    def test_the_interval_is_placed_around_the_delivered_trajectory(self):
        s = shared()
        bands, n = readouts.interval(s.h, s.on, group=GROUP, temp_sigma_std=SIGMA)
        load, types = np.asarray(s.h.load_kw, dtype=np.float64), np.asarray(s.h.day_types)
        Q, n_ref = A.readouts.residual_quantiles(load, s.h.temperature_c, types, s.h.start_timestamp, GROUP, SIGMA, O)
        args = (A.readouts.origin_scale(load[O - CONTEXT:O]), Q, (O + np.arange(HORIZON)) % 24, (types[O:O + HORIZON] < 5).astype(int))
        self.assertEqual(n, n_ref)
        self.assertEqual(bands.tobytes(), A.readouts.interval_bands(s.on.trajectory_kw, *args).tobytes())
        self.assertFalse(np.array_equal(bands, A.readouts.interval_bands(s.off.trajectory_kw, *args)))


class TestBypassAndScope(unittest.TestCase):
    def test_state_windows_bypass_the_layer(self):
        s = shared()
        for kind in ("off", "micro"):
            sh = state_history(kind)
            for rule in ("seasonal_naive", None):
                off = forecast(sh, s.m, state_rule=rule, continuation=False, **KW)
                on = forecast(sh, s.m, state_rule=rule, continuation=True, **KW)
                self.assertTrue(on.state_window, (kind, rule))
                self.assertEqual(on.trajectory_kw.tobytes(), off.trajectory_kw.tobytes(), (kind, rule))
                self.assertEqual(on.trajectory_before_continuation_kw.tobytes(), off.trajectory_kw.tobytes(), (kind, rule))
                self.assertEqual((on.level_kw, on.peak_kw, on.state_rule), (off.level_kw, off.peak_kw, off.state_rule), (kind, rule))
                self.assertEqual(on.continuation.bypass, "state window", (kind, rule))
                self.assertEqual(readouts.continuation_energy_change(on), 0.0)

    def test_the_history_only_configuration_is_never_continued(self):
        s = shared()
        off = forecast(s.h, None, continuation=False, **KW)
        on = forecast(s.h, None, continuation=True, **KW)
        self.assertEqual(on.trajectory_kw.tobytes(), off.trajectory_kw.tobytes())
        self.assertEqual((on.level_kw, on.peak_kw), (off.level_kw, off.peak_kw))
        self.assertIsNone(on.continuation)
        self.assertIs(on.trajectory_before_continuation_kw, on.trajectory_kw)

    def test_missing_last_hours_are_refused_by_the_history_and_bypassed_by_apply(self):
        s = shared()
        obs = np.ones(O, bool)
        obs[O - 2] = False
        h = history_at(s.load, s.temp, s.types, s.start, observed=obs)
        for carr in (s.m, None):
            with self.assertRaises(ValueError):                                       # the last 1,344 hours must be observed
                forecast(h, carr, continuation=True, **KW)
        tp, rec = C.apply(s.off.trajectory_kw, _masked_load(h), h.day_types, O)
        self.assertEqual(rec.bypass, "missing hours")
        self.assertEqual(tp.tobytes(), s.off.trajectory_kw.tobytes())

    def test_negative_last_hours_set_the_anomaly_to_zero_and_do_not_raise(self):
        # a net-metered record: the last 24 hours below -e have no logarithm; with the layer on the anomalies are 0
        # and flagged, so the delivered trajectory is the one before the layer (the release rule for negative loads)
        s = shared()
        load = s.load.copy()
        load[O - 24:O] = -5.0 * float(np.abs(s.load[O - 672:O]).mean())
        h = history_at(load, s.temp, s.types, s.start)
        off = forecast(h, s.m, continuation=False, **KW)
        for constants in (True, C.ZERO):
            on = forecast(h, s.m, continuation=constants, **KW)
            if on.state_window:
                self.assertEqual(on.continuation.bypass, "state window")
            else:
                self.assertEqual((on.continuation.nonpositive_fast, on.continuation.nonpositive_day), (True, True))
                self.assertEqual(on.continuation.bypass, "nonpositive load")
            self.assertEqual(on.trajectory_kw.tobytes(), off.trajectory_kw.tobytes())
            self.assertEqual((on.level_kw, on.peak_kw), (off.level_kw, off.peak_kw))
            self.assertEqual(readouts.continuation_energy_change(on), 0.0)

    def test_nothing_at_or_after_the_origin_reaches_the_continued_forecast(self):
        s = shared()
        rng = np.random.default_rng(7)
        load2, temp2 = s.load.copy(), s.temp.copy()
        load2[O:] = rng.normal(0.0, 100.0, len(load2) - O)
        temp2[O:] += rng.normal(0.0, 20.0, len(temp2) - O)
        out = []
        for ld, tp in ((s.load, s.temp), (load2, temp2)):
            F = carrier_forecasts(s.m, ld, tp, s.types, GROUP, s.start, 0, O)          # reads the arrays that run past the origin
            h = history_at(ld, tp, s.types, s.start)
            out.append((forecast(h, F, continuation=True, **KW), forecast(h, s.m, continuation=True, **KW)))
        for a, b in zip(out[0], out[1]):
            self.assertEqual(a.trajectory_kw.tobytes(), b.trajectory_kw.tobytes())
            self.assertEqual((a.level_kw, a.peak_kw), (b.level_kw, b.peak_kw))
            self.assertEqual(a.continuation.delta.tobytes(), b.continuation.delta.tobytes())
        self.assertEqual(out[0][0].trajectory_kw.tobytes(), out[0][1].trajectory_kw.tobytes())
        self.assertEqual(out[0][0].trajectory_kw.tobytes(), s.on.trajectory_kw.tobytes())


class TestEnsembles(unittest.TestCase):
    def test_six_carriers_are_averaged_at_every_issue_time(self):
        s = shared()
        members = [carrier(k) for k in range(6)]
        single = [carrier_forecasts(m, s.load, s.temp, s.types, GROUP, s.start, 0, O) for m in members]
        F6 = carrier_forecasts(members, s.load, s.temp, s.types, GROUP, s.start, 0, O)
        self.assertEqual(sorted(F6), list(range(7)))
        for k in F6:
            total = single[0][k]
            for f in single[1:]:
                total = total + f[k]                                                  # float64, in member order
            self.assertEqual(F6[k].tobytes(), (total / 6).tobytes(), k)
        r = forecast(s.h, members, continuation=True, **KW)
        rm = forecast(s.h, F6, continuation=True, **KW)
        self.assertEqual(r.trajectory_kw.tobytes(), rm.trajectory_kw.tobytes())
        self.assertEqual((r.level_kw, r.peak_kw), (rm.level_kw, rm.peak_kw))
        for k in F6:
            self.assertEqual(r.carrier[k].tobytes(), F6[k].tobytes(), k)
        cut = O - 3 * STEP                                                            # one cut for all six members
        Fc = carrier_forecasts(members, s.load, s.temp, s.types, GROUP, s.start, 0, O, cut=cut)
        self.assertEqual(sorted(Fc), [0, 1, 2, 3])
        rc = forecast(s.h, members, cut=cut, continuation=True, **KW)
        rcm = forecast(s.h, Fc, continuation=True, **KW)
        self.assertEqual(sorted(rc.carrier), [0, 1, 2, 3])
        self.assertEqual(rc.trajectory_kw.tobytes(), rcm.trajectory_kw.tobytes())
        self.assertEqual((rc.level_kw, rc.peak_kw), (rcm.level_kw, rcm.peak_kw))
        r6c = forecast(s.h, F6, cut=cut, continuation=True, **KW)                    # the cut drops k = 4..6 from a mapping
        self.assertEqual(sorted(r6c.carrier), [0, 1, 2, 3])
        for k in r6c.carrier:
            self.assertEqual(r6c.carrier[k].tobytes(), F6[k].tobytes(), k)
            self.assertTrue(np.allclose(Fc[k], F6[k], rtol=1e-6, atol=0), k)          # the same members on fewer issue times

    def test_release_switches(self):
        self.assertIs(inspect.signature(forecast).parameters["continuation"].default, _release.CONTINUATION_DEFAULT)
        self.assertIn(_release.CONTINUATION_DEFAULT, (False, True))
        self.assertEqual(ankyra_lite.__version__, _release.VERSION)
        self.assertIn(_release.CARRIER_RECIPE, ("R", "Rw"))
        self.assertEqual(_release.CHECKPOINT_SUBDIR, "" if _release.CARRIER_RECIPE == "R" else "rw")
        seeds = _release.CARRIER_SEEDS
        self.assertIn(len(seeds), (3, 6))
        self.assertEqual(tuple(seeds), tuple(range(len(seeds))))
        paths = _release.checkpoint_paths("cambridge", root="ck")
        self.assertEqual(paths, [Path("ck") / "cambridge" / _release.CHECKPOINT_SUBDIR / f"seed{k}.pt" for k in seeds])


class TestArgumentsAndResultType(unittest.TestCase):
    def test_invalid_continuation_values_are_refused(self):
        s = shared()
        for bad in ("yes", 1, 0, 1.0, np.True_, {"kappa1": 0.479}, (0.479, 12.0, 0.33, 168.0)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                forecast(s.h, s.m, continuation=bad, **KW)

    def test_the_result_type_is_backward_compatible(self):
        s = shared()
        self.assertEqual([f.name for f in dataclasses.fields(ForecastResult)],
                         FIELDS_0_1_1 + ["trajectory_before_continuation_kw", "continuation"])
        tr = s.off.trajectory_kw
        r = ForecastResult(tr, 1.5, 2.5, False, None, s.off.anchored)                 # the 0.1.1 positional form
        self.assertIs(r.trajectory_before_continuation_kw, tr)
        self.assertIsNone(r.continuation)
        self.assertEqual(r.energy_kwh, HORIZON * 1.5)
        self.assertEqual(r.daily_means_kw is s.off.anchored.daily_means_kw, True)      # read through to the anchoring layer
        text = repr(s.on)
        self.assertNotIn("trajectory_before_continuation_kw", text)
        self.assertNotIn("continuation=", text)
        self.assertEqual(s.on.energy_kwh, float(HORIZON * s.on.level_kw))

    def test_top_level_names(self):
        self.assertIs(ankyra_lite.continuation, C)
        self.assertIsInstance(C, _types.ModuleType)
        self.assertIn("continuation", ankyra_lite.__all__)
        self.assertIs(ankyra_lite.forecast, forecast)
        self.assertIs(ankyra_lite.checkpoint_paths, _release.checkpoint_paths)
        self.assertIn("checkpoint_paths", ankyra_lite.__all__)
        self.assertIn("continuation_energy_change", readouts.__all__)
        for name in ankyra_lite.__all__:
            self.assertTrue(hasattr(ankyra_lite, name), name)
        self.assertTrue(math.isfinite(C.CONSTANTS.kappa1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
