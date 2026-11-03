"""Self-contained tests of the carrier network, its features, its inference and the end-to-end forecast (synthetic data).

Run from the repository root:  python -m unittest tests.test_carrier_unit -v
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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch.set_num_threads(1)

from ankyra_lite import anchor as A                                                   # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                            # noqa: E402
from ankyra_lite.carrier import inference as I                                        # noqa: E402
from ankyra_lite.carrier.features import features                                    # noqa: E402
from ankyra_lite.carrier.model import (CARRIER_PARAMETERS, INPUT_KEYS, StructuredLoadNetwork, build_carrier,  # noqa: E402
                                       forecast_loss, load_carrier, param_groups, parameter_count)


def synthetic_store(units=2, hours=16000, seed=0):
    """Arrays in the layout of ankyra_lite.carrier.features (index 0 = 2019-01-01 00:00 UTC)."""
    h = synthetic_history(hours)
    rng = np.random.default_rng(seed)
    load = np.stack([np.asarray(h.load_kw) * (1 + 0.1 * u) + rng.normal(0, 0.3, hours) for u in range(units)])
    temp = np.stack([np.asarray(h.temperature_c, np.float32)] * units)
    return {"load": load, "temp": temp, "types": np.asarray(h.day_types), "groups": ["Office", "Residential"][:units],
            "start_iso": h.start_timestamp}


def random_inputs(batch=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = {"day_tokens": torch.randn(batch, 366, 69, generator=g), "day_mask": torch.rand(batch, 366, generator=g) > 0.1,
         "month_tokens": torch.randn(batch, 24, 16, generator=g), "month_mask": torch.rand(batch, 24, generator=g) > 0.3,
         "future_tokens": torch.randn(batch, 31, 16, generator=g), "scale": torch.rand(batch, generator=g) + 0.5}
    x["day_tokens"][:, :, 24:48] = (torch.rand(batch, 366, 24, generator=g) > 0.05).float()
    return x


class TestNetwork(unittest.TestCase):
    def test_parameter_count(self):
        m = build_carrier()
        self.assertEqual(parameter_count(m), CARRIER_PARAMETERS)
        self.assertEqual(CARRIER_PARAMETERS, 61613)
        base, ptr = param_groups(m)
        self.assertEqual(sum(p.numel() for p in base), 59850)
        self.assertEqual(sum(p.numel() for p in ptr), 61613 - 59850)

    def test_zero_gate_pointer_equals_base_decoder(self):
        torch.manual_seed(0)
        m = build_carrier().eval()
        base = StructuredLoadNetwork(m.config).eval()
        base.load_state_dict({k: v for k, v in m.state_dict().items() if k.split(".")[0] not in m.POINTER_MODULES}, strict=True)
        x = random_inputs()
        with torch.no_grad():
            a, b = m(x), base(x)
        for key in ("trajectory_kw", "log_trajectory_normalized", "level_kw"):
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_budget_positivity_and_masked_nan(self):
        torch.manual_seed(0)
        m = build_carrier().eval()
        with torch.no_grad():
            m.copy_gate.bias.fill_(1.0)
        x = random_inputs()
        x["day_tokens"][0, -3] = float("nan")
        x["day_mask"][0, -3] = False
        with torch.no_grad():
            out = m(x)
        tr = out["trajectory_kw"]
        self.assertTrue(torch.isfinite(tr).all() and (tr >= 0).all())
        self.assertTrue(torch.allclose(tr.reshape(-1, 31, 24).mean((1, 2)), out["level_kw"], rtol=1e-5))
        y = tr.detach() * 1.1
        loss = forecast_loss(out, y, x["scale"], torch.ones_like(y, dtype=torch.bool), kl_weight=0.1, compression=True)
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_checkpoint_round_trip_is_strict(self):
        import tempfile
        torch.manual_seed(3)
        m = build_carrier()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.pt"
            torch.save({"model": m.state_dict()}, p)
            r = load_carrier(p)
        self.assertTrue(all(torch.equal(v, r.state_dict()[k]) for k, v in m.state_dict().items()))
        self.assertFalse(r.training)
        bad = build_carrier().state_dict()
        bad.pop("type_bonus")
        with self.assertRaises(RuntimeError):
            build_carrier().load_state_dict(bad, strict=True)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synthetic_store()

    def f(self, d, b, o, **kw):
        return features(d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], b, o, **kw)

    def test_shapes_and_dtypes(self):
        f = self.f(self.d, 1, 12000)
        self.assertEqual((f["day_tokens"].shape, f["day_tokens"].dtype), ((366, 69), np.float16))
        self.assertEqual((f["month_tokens"].shape, f["month_tokens"].dtype), ((24, 16), np.float16))
        self.assertEqual((f["future_tokens"].shape, f["future_tokens"].dtype), ((31, 16), np.float32))
        self.assertEqual((f["day_mask"].shape, f["month_mask"].shape), ((366,), (24,)))
        self.assertIsInstance(f["scale"], np.float32)
        g = self.f(self.d, 1, 12000, half=False)
        self.assertEqual(g["day_tokens"].dtype, np.float32)
        self.assertTrue(np.array_equal(g["day_tokens"].astype(np.float16), f["day_tokens"]))
        self.assertEqual(int(np.argmax(f["future_tokens"][0, 11:16])), 3)          # group one-hot: Residential

    def test_no_value_at_or_after_the_origin_is_read(self):
        o = 12000
        d2 = dict(self.d)
        d2["load"] = self.d["load"].copy(); d2["load"][:, o:] = 999.0
        d2["temp"] = self.d["temp"].copy(); d2["temp"][:, o:] = 99.0
        f1, f2 = self.f(self.d, 0, o), self.f(d2, 0, o)
        for k in f1:
            self.assertTrue(np.array_equal(np.asarray(f1[k]), np.asarray(f2[k])), k)

    def test_short_record_and_one_dimensional_arrays(self):
        f = features(self.d["load"][0], self.d["temp"][0], self.d["types"], "Office", self.d["start_iso"], 0, 2000)
        g = self.f(self.d, 0, 2000)
        for k in f:
            self.assertTrue(np.array_equal(np.asarray(f[k]), np.asarray(g[k])), k)
        self.assertFalse(f["day_mask"][:280].any())                                 # virtual leading padding
        self.assertTrue(f["day_mask"][-83:].all())


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synthetic_store()
        torch.manual_seed(0)
        cls.m1 = build_carrier().eval()
        torch.manual_seed(1)
        cls.m2 = build_carrier().eval()
        with torch.no_grad():
            cls.m2.copy_gate.bias.fill_(0.5)

    def cf(self, models, b, o, cut=None, d=None):
        d = d or self.d
        return I.carrier_forecasts(models, d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], b, o, cut=cut)

    def test_pseudo_origin_rule(self):
        o = 15000
        self.assertEqual(I.pseudo_origins(self.d["load"], 0, o), list(range(7)))
        d2 = dict(self.d); d2["load"] = self.d["load"].copy()
        d2["load"][0, o - 2 * 744 - 100:o - 2 * 744 - 80] = np.nan                    # a 20-hour gap inside the contexts of k = 1, 2
        ks = I.pseudo_origins(d2["load"], 0, o)
        self.assertEqual(ks, [0, 3, 4, 5, 6])
        self.assertEqual(ks, sorted(A.pseudo_origin_contexts(d2["load"][0, :o]).keys()))
        self.assertEqual(I.pseudo_origins(self.d["load"], 0, o, cut=o - 3 * 744), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            I.pseudo_origins(self.d["load"], 0, o, cut=o + 1)
        d2["load"][0, o - 5] = np.nan
        with self.assertRaises(ValueError):
            I.pseudo_origins(d2["load"], 0, o)
        self.assertEqual(sorted(self.cf(self.m1, 0, o, cut=o - 3 * 744)), [0, 1, 2, 3])

    def test_forecasts_use_only_data_before_each_issue_time(self):
        o = 15000
        f = self.cf(self.m1, 1, o)
        q = o - 2 * 744
        d2 = dict(self.d); d2["load"] = self.d["load"].copy(); d2["temp"] = self.d["temp"].copy()
        d2["load"][:, q:] *= 3.0; d2["temp"][:, q:] += 5.0
        g = self.cf(self.m1, 1, o, d=d2)
        for k in (2, 3, 4, 5, 6):
            self.assertTrue(np.array_equal(f[k], g[k]), k)
        self.assertFalse(np.array_equal(f[0], g[0]))

    def test_ensemble_is_the_member_average(self):
        o = 15000
        a, b, e = self.cf(self.m1, 0, o), self.cf(self.m2, 0, o), self.cf([self.m1, self.m2], 0, o)
        for k in e:
            self.assertEqual(e[k].dtype, np.float64)
            self.assertTrue(np.array_equal(e[k], (a[k] + b[k]) / 2), k)

    def test_batched_windows_match_single_windows(self):
        units, origins = [0, 1, 0], [15000, 14000, 9000]
        pred, have = I.forecast_windows([self.m1, self.m2], self.d["load"], self.d["temp"], self.d["types"], self.d["groups"],
                                        self.d["start_iso"], units, origins)
        for i, (b, o) in enumerate(zip(units, origins)):
            f = self.cf([self.m1, self.m2], b, o)
            self.assertEqual(sorted(f), list(np.flatnonzero(have[i])))
            for k, v in f.items():                       # CPU results may depend on the batch composition in the last bits
                self.assertLessEqual(float(np.max(np.abs(pred[i, k] - v) / np.abs(v))), 1e-6)
        self.assertTrue(np.isnan(pred[~have]).all())


class TestForecast(unittest.TestCase):
    def test_end_to_end(self):
        from ankyra_lite.forecast import ForecastResult, forecast
        torch.manual_seed(0)
        m = build_carrier().eval()
        h = synthetic_history(16000)
        r = forecast(h, m, group="Office", temp_sigma_std=0.25)
        self.assertIsInstance(r, ForecastResult)
        self.assertEqual(r.trajectory_kw.shape, (744,))
        self.assertTrue(np.isfinite(r.trajectory_kw).all() and np.isfinite(r.peak_kw))
        self.assertAlmostEqual(r.energy_kwh, 744 * r.level_kw)
        self.assertFalse(r.state_window)
        self.assertEqual(sorted(r.carrier), list(range(7)))
        self.assertEqual(r.pseudo_pairs, r.anchored.pseudo_pairs)              # attributes of the anchoring result
        ref = A.forecast(h, foundation=r.carrier, group="Office", temp_sigma_std=0.25)
        # the anchoring layer's output is the trajectory before the continuation layer (on by default from 0.2.0)
        self.assertTrue(np.array_equal(ref.trajectory_kw, r.trajectory_before_continuation_kw))
        self.assertEqual(ref.level_kw, r.level_kw)
        off = forecast(h, r.carrier, group="Office", temp_sigma_std=0.25, continuation=False)
        self.assertTrue(np.array_equal(ref.trajectory_kw, off.trajectory_kw))
        self.assertIsNone(off.continuation)
        again = forecast(h, r.carrier, group="Office", temp_sigma_std=0.25)          # precomputed carrier forecasts
        self.assertTrue(np.array_equal(again.trajectory_kw, r.trajectory_kw))
        cut = 16000 - 3 * 744                                                       # cut= applies to a precomputed mapping too
        part = forecast(h, r.carrier, group="Office", temp_sigma_std=0.25, cut=cut)
        computed = forecast(h, m, group="Office", temp_sigma_std=0.25, cut=cut)
        self.assertEqual(sorted(part.carrier), [0, 1, 2, 3])
        self.assertEqual(sorted(computed.carrier), [0, 1, 2, 3])
        self.assertTrue(np.array_equal(part.trajectory_kw, computed.trajectory_kw))
        with self.assertRaises(ValueError):
            forecast(h, r.carrier, group="Office", temp_sigma_std=0.25, cut=16001)

    def test_peak_readout_uses_the_anchoring_layers_day_types(self):
        """The peak readout reads each origin-aligned day's type at its 12th hour, as the anchoring layer does. The
        origin of this record is at 16:00, where a day's first hour and its 12th hour fall on different calendar
        days; with irregular day types and spikes on one type the two readings give different peaks."""
        from ankyra_lite.forecast import peak
        h = synthetic_history(16000)
        load, o = np.array(h.load_kw, np.float64), 16000
        ty = np.repeat(np.random.default_rng(5).integers(0, 8, (o + 744) // 24 + 1), 24)[:o + 744]
        hrs = np.arange(o - 1344, o)
        load[hrs[(ty[hrs] == 3) & (hrs % 24 == 18)]] += 50.0
        tr = np.repeat(10.0 + np.arange(31.0), 24)
        self.assertEqual(o % 24, 16)
        mid = A.readouts.peak_readout(tr, load[o - 1344:o][None], ty[o - 1344 + 12 + 24 * np.arange(56)][None],
                                      ty[o + 12 + 24 * np.arange(31)][None])[0]
        first = A.readouts.peak_readout(tr, load[o - 1344:o][None], ty[o - 1344:o:24][None], ty[o:o + 744:24][None])[0]
        self.assertNotEqual(mid, first)
        self.assertEqual(peak(tr, load, ty), float(mid))
        o2 = o - 16                                                                   # an origin at midnight
        l2, t2 = load[:o2], ty[:o2 + 744]
        same = A.readouts.peak_readout(tr, l2[o2 - 1344:o2][None], t2[o2 - 1344:o2:24][None], t2[o2:o2 + 744:24][None])[0]
        self.assertEqual(peak(tr, l2, t2), float(same))                              # there the day's first hour gives the same types

    def test_state_window_rule(self):
        from ankyra_lite.forecast import forecast, seasonal_naive
        torch.manual_seed(0)
        m = build_carrier().eval()
        h = synthetic_history(16000, off_last_hours=200)
        load = np.asarray(h.load_kw, np.float64)
        h = A.History(load + np.where(np.arange(len(load)) >= len(load) - 200, 1e-7 * (np.arange(len(load)) % 3), 0.0),
                      h.temperature_c, h.day_types, h.start_timestamp)
        r = forecast(h, m, group="Office", temp_sigma_std=0.25)
        self.assertTrue(r.state_window and r.off_state)
        self.assertEqual(r.state_rule, "seasonal_naive")
        self.assertTrue(np.array_equal(r.trajectory_kw, seasonal_naive(h.load_kw)))
        self.assertEqual(r.level_kw, float(r.trajectory_kw.mean()))
        keep = forecast(h, m, group="Office", temp_sigma_std=0.25, state_rule=None)
        self.assertIsNone(keep.state_rule)
        self.assertTrue(np.allclose(keep.trajectory_kw, keep.carrier[0]))            # the anchoring layer returns the carrier
        with self.assertRaises(ValueError):
            forecast(h, m, group="Office", temp_sigma_std=0.25, state_rule="other")


if __name__ == "__main__":
    unittest.main(verbosity=2)
