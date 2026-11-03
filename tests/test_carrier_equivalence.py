"""Equivalence of the carrier network, its features and its inference with the research code (local-data tests).

These tests need files that are not part of this repository: the research code and outputs (folder HKAD_V3), the
original ANKYRA 2.2.1 checkout (folder Building_load_ANKYRA) and the HEEW data store, which the research code locates.
They are local-data tests: every test class is skipped, with the reason, when its files are absent. The folders are
read only (no bytecode is written into them). Default locations are the siblings of this repository; override them
with the environment variables ANKYRA_LITE_RESEARCH_ROOT (HKAD_V3) and ANKYRA_ORIGINAL_ROOT (Building_load_ANKYRA).

  (a) the trained carrier checkpoint (HEEW, carrier recipe, seed 0) loads into build_carrier() with strict=True, and
      the carrier forecasts of the first 5 test windows reproduce the stored forecasts (carrier.npz, computed on a
      GPU) within a relative difference of 1e-6, with the identical availability mask; on the CPU, the batched
      inference is bit-identical to the research inference;
  (b) features() equals the research feature builder bit for bit on 20 random (unit, origin) pairs;
  (c) the network classes build the same initial weights for a seed and the same outputs as the research classes;
  (d) end to end: the forecast from the stored three-seed carrier average reproduces the stored anchored forecasts
      (trajectory, level and peak readouts) of the first 5 test windows.

Before the data store is loaded, the tests wait (up to 30 minutes) until at least ANKYRA_LITE_TEST_MIN_FREE_GB
(default 5) GB of memory is available, when psutil is installed; otherwise they are skipped.
Run from the repository root:  python -B -m unittest tests.test_carrier_equivalence -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True                       # the research folders are read only

import json
import time
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

RESEARCH = Path(os.environ.get("ANKYRA_LITE_RESEARCH_ROOT") or REPO.parent / "HKAD_V3")
ORIGINAL = Path(os.environ.get("ANKYRA_ORIGINAL_ROOT") or REPO.parent / "Building_load_ANKYRA")
FORMAL = RESEARCH / "analysis_outputs" / "smallnet_formal3_v1" / "heew"
RUN = FORMAL / "P3s_s0"
CODE_FILES = [RESEARCH / "tools_smallnet_carrier_v1.py", RESEARCH / "new_small_network_v1" / "data.py",
              RESEARCH / "refinement_v2" / "model.py", RESEARCH / "smallnet_loop2" / "pointer_model.py",
              ORIGINAL / "ankyra" / "core.py"]
DATA_FILES = CODE_FILES + [RUN / "best.pt", RUN / "carrier.npz"]    # the HEEW store itself is located by the research code
N_WINDOWS, N_PAIRS, REL_TOL, WAIT_S = 5, 20, 1e-6, 1800
MIN_FREE_GB = float(os.environ.get("ANKYRA_LITE_TEST_MIN_FREE_GB") or 5)   # memory needed before the store is loaded


def _missing(paths):
    return [str(p) for p in paths if not p.exists()]


def _skip_unless(paths):
    miss = _missing(paths)
    return unittest.skipIf(bool(miss), "local-data test: missing " + ", ".join(miss[:3]))


def _research_paths():
    for p in (str(RESEARCH), str(ORIGINAL)):
        if p not in sys.path:
            sys.path.append(p)                         # after the repository, so ankyra_lite always resolves here


def _anchor():
    """The package's copy of the anchoring layer."""
    from ankyra_lite import anchor
    return anchor


def _wait_for_memory():
    try:
        import psutil
    except ImportError:                                # not a dependency of the package; then no memory check
        _report("psutil is not installed: no memory check before loading the store")
        return
    t0 = time.time()
    while psutil.virtual_memory().available < MIN_FREE_GB * 2 ** 30:
        if time.time() - t0 > WAIT_S:
            raise unittest.SkipTest(f"less than {MIN_FREE_GB} GB of memory available after {WAIT_S // 60} minutes")
        time.sleep(30)
    _report(f"memory check before loading the store: {psutil.virtual_memory().available / 2 ** 30:.2f} GB available "
            f"(waited {time.time() - t0:.0f} s)")


_CACHE = {}


def heew():
    """(V, d, wb, wo, cut): the research module and the HEEW store as the research pipeline reads it (read only).
    Skips (once for the whole module) when the research code cannot find the store."""
    if "heew" not in _CACHE:
        _research_paths()
        try:
            _wait_for_memory()
        except unittest.SkipTest as exc:                 # cached: the other classes skip at once instead of waiting again
            _CACHE["heew"] = exc
            raise
        import tools_smallnet_carrier_v1 as V
        try:
            d = V.raw("heew", None)
            wb, wo = V.windows_late("heew")
        except FileNotFoundError as exc:
            _CACHE["heew"] = unittest.SkipTest(f"local-data test: the HEEW data store was not found ({exc})")
        else:
            _CACHE["heew"] = (V, d, wb, wo, int(V.cuts("heew")["Cprime"]))
    if isinstance(_CACHE["heew"], unittest.SkipTest):
        raise _CACHE["heew"]
    return _CACHE["heew"]


def _report(msg):
    print("\n    " + msg, file=sys.stderr, flush=True)


@_skip_unless(DATA_FILES)
class TestCarrierForecastsHEEW(unittest.TestCase):
    """(a) checkpoint, strict load, carrier forecasts of the first 5 test windows."""

    @classmethod
    def setUpClass(cls):
        from ankyra_lite.carrier.model import build_carrier
        cls.V, cls.d, wb, wo, cls.cut = heew()
        cls.wb, cls.wo = wb[:N_WINDOWS], wo[:N_WINDOWS]
        ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
        cls.model = build_carrier()
        cls.load_result = cls.model.load_state_dict(ck["model"], strict=True)
        cls.model.eval()
        z = np.load(RUN / "carrier.npz", allow_pickle=False)
        cls.stored = {k: z[k] for k in z.files}

    def args(self):
        d = self.d
        return d["load"], d["temp"], d["types"], d["groups"], d["start_iso"]

    def test_strict_checkpoint_load(self):
        self.assertEqual((list(self.load_result.missing_keys), list(self.load_result.unexpected_keys)), ([], []))
        from ankyra_lite.carrier.model import CARRIER_PARAMETERS, parameter_count
        self.assertEqual(parameter_count(self.model), CARRIER_PARAMETERS)

    def test_windows_are_the_stored_windows(self):
        self.assertTrue(np.array_equal(self.stored["building"][:N_WINDOWS], self.wb))
        self.assertTrue(np.array_equal(self.stored["origin"][:N_WINDOWS], self.wo))
        self.assertEqual(int(self.stored["cut"]), self.cut)

    def _compare(self, pred, have, label):
        ref, rh = self.stored["pred"][:N_WINDOWS].astype(np.float64), self.stored["have"][:N_WINDOWS]
        self.assertTrue(np.array_equal(have, rh), label + ": availability mask differs")
        a, b = pred[have], ref[rh]
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), np.finfo(np.float64).tiny)))
        _report(f"{label}: {int(have.sum())} forecasts x 744 h, max relative difference {rel:.3e} (tolerance {REL_TOL:g})")
        self.assertLessEqual(rel, REL_TOL)

    def test_carrier_forecasts_match_stored(self):
        from ankyra_lite.carrier.inference import K, carrier_forecasts
        pred = np.full((N_WINDOWS, K + 1, 744), np.nan)
        have = np.zeros((N_WINDOWS, K + 1), bool)
        for i, (b, o) in enumerate(zip(self.wb, self.wo)):
            f = carrier_forecasts(self.model, *self.args(), int(b), int(o), cut=self.cut)
            for k, v in f.items():
                self.assertEqual(v.shape, (744,))
                pred[i, k], have[i, k] = v, True
        self._compare(pred, have, "carrier_forecasts (per window)")

    def test_batched_windows_match_stored_and_research_cpu(self):
        from ankyra_lite.carrier.inference import forecast_windows
        pred, have = forecast_windows(self.model, *self.args(), self.wb, self.wo, cut=self.cut)
        self._compare(pred, have, "forecast_windows (batched)")
        p0, h0 = self.V.forecast_windows(self.model, self.d, self.wb, self.wo, self.cut, torch.device("cpu"))
        self.assertTrue(np.array_equal(h0, have))
        self.assertTrue(np.array_equal(p0[h0].astype(np.float64), pred[have]), "batched CPU inference differs from the research code")
        _report("forecast_windows (batched) vs research forecast_windows, both on the CPU: bit-identical")


@_skip_unless(DATA_FILES)
class TestFeaturesHEEW(unittest.TestCase):
    """(b) features() against the research feature builder, bit for bit."""

    @classmethod
    def setUpClass(cls):
        cls.V, cls.d, _, _, _ = heew()

    def test_random_pairs_bitwise(self):
        from ankyra_lite.carrier.features import features
        d = self.d
        n_units, hours = d["load"].shape
        rng = np.random.default_rng(20261018)
        pairs = list(zip(rng.integers(0, n_units, N_PAIRS).tolist(), rng.integers(0, hours + 1, N_PAIRS).tolist()))
        for b, o in pairs:
            ref = self.V.feats(d, b, o)
            got = features(d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], b, o)
            self.assertEqual(sorted(got), sorted(ref), (b, o))
            for k in ref:
                x, y = np.asarray(got[k]), np.asarray(ref[k])
                self.assertEqual((type(got[k]), x.dtype, x.shape), (type(ref[k]), y.dtype, y.shape), (b, o, k))
                self.assertEqual(x.tobytes(), y.tobytes(), (b, o, k))
        _report(f"features: {N_PAIRS} random (unit, origin) pairs, all 8 outputs bit-identical; origins {sorted(o for _, o in pairs)}")


@_skip_unless(CODE_FILES)
class TestNetworkMatchesResearchCode(unittest.TestCase):
    """(c) same construction (seeded initial weights, parameter names and order) and the same forward arithmetic."""

    def test_same_initialisation_and_outputs(self):
        _research_paths()
        from refinement_v2.model import ModelConfig as RConfig, StructuredLoadNetwork as RBase
        from smallnet_loop2.pointer_model import CopyPointerNetwork as RPointer
        from ankyra_lite.carrier.model import StructuredLoadNetwork, build_carrier
        cfg = RConfig(day_features=69, month_features=16, future_features=16, variant="full", calendar_month_head=False)
        for ours, theirs in ((build_carrier, lambda: RPointer(cfg)), (StructuredLoadNetwork, lambda: RBase(cfg))):
            torch.manual_seed(0); a = ours().eval()
            torch.manual_seed(0); b = theirs().eval()
            sa, sb = a.state_dict(), b.state_dict()
            self.assertEqual(list(sa), list(sb))
            self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))
            self.assertEqual([n for n, _ in a.named_parameters()], [n for n, _ in b.named_parameters()])
            with torch.no_grad():
                b.load_state_dict({k: v + 0.01 * torch.randn_like(v) for k, v in sb.items()})   # non-trivial weights
                a.load_state_dict(b.state_dict(), strict=True)
                g = torch.Generator().manual_seed(1)
                x = {"day_tokens": torch.randn(3, 366, 69, generator=g), "day_mask": torch.rand(3, 366, generator=g) > 0.1,
                     "month_tokens": torch.randn(3, 24, 16, generator=g), "month_mask": torch.rand(3, 24, generator=g) > 0.3,
                     "future_tokens": torch.randn(3, 31, 16, generator=g), "scale": torch.rand(3, generator=g) + 0.5}
                x["day_tokens"][:, :, 24:48] = (torch.rand(3, 366, 24, generator=g) > 0.05).float()
                oa, ob = a(x), b(x)
            self.assertEqual(sorted(oa), sorted(ob))
            for k, v in ob.items():
                self.assertTrue(torch.equal(oa[k], v), k)


@_skip_unless(DATA_FILES + [FORMAL / "arms_P3s.npz", FORMAL / "P3s_s1" / "carrier.npz", FORMAL / "P3s_s2" / "carrier.npz",
                            RESEARCH / "analysis_outputs" / "smallnet_carrier_v1" / "heew" / "prep_formal.json"])
class TestEndToEndHEEW(unittest.TestCase):
    """(d) forecast() with the stored three-seed carrier average reproduces the stored anchored forecasts."""

    @classmethod
    def setUpClass(cls):
        cls.V, cls.d, wb, wo, cls.cut = heew()
        cls.wb, cls.wo = wb[:N_WINDOWS], wo[:N_WINDOWS]
        cls.sigma = json.loads((RESEARCH / "analysis_outputs" / "smallnet_carrier_v1" / "heew" / "prep_formal.json").read_text(encoding="utf-8"))["sigma"]
        cls.region = staticmethod(cls.V._region("heew", cls.d))     # b -> dst_region, as in the research pipeline
        z = np.load(FORMAL / "arms_P3s.npz", allow_pickle=False)
        cls.arms = {k: z[k] for k in ("building", "origin", "traj", "level", "peak", "off", "micro")}
        sp, sh = None, None
        for s in range(3):                                   # the research ensemble: float64 sum of the members in seed order
            q = np.load(FORMAL / f"P3s_s{s}" / "carrier.npz", allow_pickle=False)
            sp = q["pred"][:N_WINDOWS].astype(np.float64) if sp is None else sp + q["pred"][:N_WINDOWS]
            sh = q["have"][:N_WINDOWS] if sh is None else sh & q["have"][:N_WINDOWS]
        cls.sp, cls.sh = sp / 3, sh

    def test_anchored_forecasts_match_stored(self):
        A = _anchor()
        from ankyra_lite.forecast import forecast
        d = self.d
        self.assertTrue(np.array_equal(self.arms["building"][:N_WINDOWS], self.wb) and np.array_equal(self.arms["origin"][:N_WINDOWS], self.wo))
        worst = 0.0
        for i, (b, o) in enumerate(zip(self.wb.tolist(), self.wo.tolist())):
            ty = d["types"] if d["types"].ndim == 1 else d["types"][b]
            tmp = d["temp"][b] if d["temp"].ndim == 2 else d["temp"]
            h = A.History(load_kw=d["load"][b, :o], temperature_c=tmp[:o], day_types=ty[:o + 744], start_timestamp=d["start_iso"],
                          observed=np.isfinite(d["load"][b, :o]))
            F = {k: self.sp[i, k] for k in range(7) if self.sh[i, k]}
            # the stored research forecasts precede the continuation layer (on by default from 0.2.0): compare with it off
            r = forecast(h, F, group=d["groups"][b], temp_sigma_std=self.sigma, dst_region=self.region(b), state_rule=None,
                         continuation=False)
            self.assertEqual(bool(r.off_state), bool(self.arms["off"][i]))
            self.assertEqual(bool(r.micro_load), bool(self.arms["micro"][i]))
            self.assertTrue(np.array_equal(r.trajectory_kw.astype(np.float32), self.arms["traj"][i]), i)   # stored in float32
            self.assertEqual(r.level_kw, float(self.arms["level"][i]), i)
            self.assertEqual(r.peak_kw, float(self.arms["peak"][i]), i)
            self.assertAlmostEqual(r.energy_kwh, 744 * float(self.arms["level"][i]), delta=1e-9 * abs(r.energy_kwh))
            worst = max(worst, float(np.max(np.abs(r.trajectory_kw - self.arms["traj"][i]) / np.abs(self.arms["traj"][i]))))
        _report(f"end to end: {N_WINDOWS} windows, trajectory equal at float32 storage precision (max relative {worst:.1e}), "
                f"level and peak readouts identical")


if __name__ == "__main__":
    unittest.main(verbosity=2)
