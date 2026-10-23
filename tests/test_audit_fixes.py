"""Small synthetic regressions for the 2026-10-22 code-claim audit fixes (unittest; collected by ``unittest discover``)."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _feature_inputs():
    """One year of history plus one forecast month; the load varies hour by hour so that month tokens depend on which
    days are workdays (a constant or weekday-only load cannot tell a wrong calendar from the right one)."""
    hours = 366 * 24 + 31 * 24 + 744
    rng = np.random.default_rng(20261022)
    load = (10.0 + 4.0 * np.sin(2 * np.pi * np.arange(hours) / 24.0) + rng.uniform(0.0, 3.0, hours)).astype(np.float32)
    temp = np.full(hours, 10.0, dtype=np.float32)
    start = "2020-01-01"
    weekdays = (datetime.fromisoformat(start).weekday() + np.arange(hours) // 24) % 7
    return hours, load, temp, start, weekdays


class RankDirection(unittest.TestCase):
    def test_rank_direction_and_legacy_adapter(self):
        from evaluation import scoring
        from scipy import stats

        n = 100
        models = {"A": np.full(n, 10.0), "B": np.r_[np.full(90, 11.0), np.zeros(10)]}
        for j in range(20):
            models[f"M{j:02d}"] = np.r_[np.full(90, 100.0 + j), np.linspace(0.4, 8.0, 10)]
        result = scoring.rank_tests(models, reference="A")
        self.assertGreater(result["mean_rank"]["A"], result["mean_rank"]["B"])      # A ranks behind B on average ...
        self.assertIn("B", result["significantly_better_than"])                     # ... yet the paired signed ranks favour A
        self.assertNotIn("B", result["significantly_worse_than"])
        old = scoring.rank_block_from_result(result)
        self.assertEqual(old["ankyra_significantly_better_than"], result["significantly_better_than"])
        self.assertIn("ankyra_position", old)
        self.assertIn("wilcoxon_holm_p", old)
        raw = {m: float(stats.wilcoxon(models["A"], models[m], zero_method="wilcox").pvalue) for m in models if m != "A"}
        ordered = sorted(raw, key=raw.get)
        expected_holm, running = {}, 0.0
        for i, m in enumerate(ordered):
            running = max(running, min(1.0, (len(raw) - i) * raw[m]))
            expected_holm[m] = running
        self.assertEqual(result["wilcoxon_holm_p"], expected_holm)
        self.assertIn("A", scoring.rank_tests(models, reference="B")["significantly_worse_than"])
        scoring.rank_tests(dict(models, Added=np.full(n, 50.0)), reference="A")
        self.assertEqual(float(stats.wilcoxon(models["A"], models["B"], zero_method="wilcox").pvalue), raw["B"])
        tied = scoring.rank_tests({"A": np.zeros(4), "B": np.array([1.0, -1.0, 1.0, -1.0]), "C": np.full(4, 0.5)}, reference="A")
        self.assertEqual(tied["wilcoxon_holm_p"]["B"], 1.0)
        self.assertNotIn("B", tied["significantly_better_than"] + tied["significantly_worse_than"])
        same = scoring.rank_tests({"A": np.ones(8), "B": np.ones(8), "C": np.arange(8, dtype=float)}, reference="A")
        self.assertEqual(same["wilcoxon_holm_p"]["B"], 1.0)


class MonthlyWeekdayFallback(unittest.TestCase):
    # sums of the features with the complete weekday calendar, computed with the v0.2.0 code (6bcd706); the fix must not
    # change them.  Compared with a tolerance so that the check is portable across platforms and NumPy builds.
    V020_SUMS = {"day_tokens": 18242.012200092548, "future_tokens": 120.82276167348027, "month_tokens": 10.17628362774849}
    V020_SCALARS = {"low_load_threshold": 7.529876, "scale": 11.498381}

    def test_complete_calendar_unchanged_and_absent_calendar_falls_back(self):
        from ankyra_lite.carrier.features import features

        hours, load, temp, start, weekdays = _feature_inputs()
        explicit = features(load, temp, weekdays, "Residential", start, 0, hours - 744, half=False)
        for key, total in self.V020_SUMS.items():
            self.assertAlmostEqual(float(np.asarray(explicit[key], np.float64).sum()) / total, 1.0, places=6, msg=key)
        for key, value in self.V020_SCALARS.items():
            self.assertAlmostEqual(float(explicit[key]) / value, 1.0, places=6, msg=key)
        self.assertEqual(int(explicit["month_mask"].sum()), 1)
        absent = features(load, temp, None, "Residential", start, 0, hours - 744, half=False)
        np.testing.assert_array_equal(absent["month_tokens"][:, 4:6], explicit["month_tokens"][:, 4:6])

    def test_partial_calendar_falls_back_per_hour(self):
        from ankyra_lite.carrier.features import features

        hours, load, temp, start, weekdays = _feature_inputs()
        types_2d = np.tile(weekdays.astype(np.float64), (1, 1))
        types_2d[0, 10 * 24:11 * 24] = 7.0                                      # a holiday label is kept
        types_2d[0, 11 * 24 + 3] = np.nan                                       # one missing hour falls back to its weekday
        fallback_types = np.where(np.isfinite(types_2d), types_2d, weekdays[None, :])
        partial = features(load, temp, types_2d, "Residential", start, 0, hours - 744, half=True)
        complete = features(load, temp, fallback_types, "Residential", start, 0, hours - 744, half=True)
        np.testing.assert_array_equal(partial["month_tokens"], complete["month_tokens"])
        np.testing.assert_array_equal(partial["day_tokens"], complete["day_tokens"])


class ContinuationSwitch(unittest.TestCase):
    def test_off_layer_calls_capture_explicit_forecast_argument(self):
        from evaluation import cost_timing, robustness

        class Spy:
            def __init__(self):
                self.calls = []

            def forecast(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "called"

        for module, positional in ((cost_timing, ("history", "carrier_forecasts")), (robustness, ("history", "models"))):
            spy = Spy()
            self.assertEqual(module._forecast_with_layer(spy, positional[0], False, positional[1], cut=123), "called")
            self.assertEqual(spy.calls[-1], ((positional[0], positional[1]), {"continuation": False, "cut": 123}))
            module._forecast_with_layer(spy, positional[0], True, positional[1])
            self.assertIs(spy.calls[-1][1]["continuation"], True)


class ScorerAdapter(unittest.TestCase):
    def test_comparison_call_sites_use_new_scorer_adapter(self):
        from evaluation import compare_baselines as cb

        class OldTools:
            def unit_rmse(self, preds, y, units):
                return np.array([0, 1, 2]), {k: np.asarray(v) for k, v in preds.items()}

            def rank_block(self, *_args, **_kwargs):
                raise AssertionError("frozen legacy rank_block must not be called")

            def scaled(self, *_args, **_kwargs):
                return {"scaled": True}

        preds = {"ref": np.array([1.0, 2.0, 3.0]), "B": np.array([2.0, 1.0, 4.0]), "C": np.array([4.0, 3.0, 2.0])}
        T = SimpleNamespace(ST=OldTools())
        _, one, _ = cb.rank_scaled(T, {"yL": np.zeros(3), "uL": np.arange(3), "ctxL": np.ones((3, 2))}, preds, "ref")
        self.assertLessEqual({"ankyra_position", "ankyra_significantly_better_than", "ankyra_significantly_worse_than"}, set(one))
        captured = {}

        def fake_rank_rows(_T, _name, _tier, rb, *_args, **_kwargs):
            captured["rb"] = rb
            return []

        with mock.patch.object(cb, "rank_rows", fake_rank_rows):
            pooled = {f: {("ANKYRA" if k == "ref" else k): v for k, v in preds.items()} for f in cb.TEST6}
            _, two = cb.pooled_rank_rows(T, pooled, "ANKYRA", "ANKYRA")
        self.assertEqual(two, captured["rb"])
        self.assertIn("ankyra_position", two)


class CacheIdentity(unittest.TestCase):
    @staticmethod
    def _identity(lr, **changes):
        args = dict(label="R six-seed", face="toy", mode="s0", limit=8, spec=((0,), {}), layer=True,
                    seeds=(0, 1, 2), seed_groups=((0, 1, 2),), constants_digest="constants-a",
                    input_manifest=[{"role": "store", "path": "toy-store", "sha256": "input-a"}], implementation_digest="code-a")
        args.update(changes)
        return lr.cache_identity(**args)

    def test_cache_identity_namespace_and_validation(self):
        from evaluation import lite_reanchor as lr

        identity = self._identity(lr)
        with tempfile.TemporaryDirectory() as td:                               # the system temporary folder, not the repository
            root = Path(td)
            data_path, meta_path = lr.cache_paths("R six-seed", "toy", "s0", 8, identity, cache_root=root)
            data_path.parent.mkdir(parents=True)
            data_path.write_bytes(b"synthetic cache payload")
            metadata = {"mode": "s0", "limit": 8, "cache_identity": identity["payload"], "cache_identity_sha256": identity["sha256"],
                        "payload_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(), "product_sha256": "product-a",
                        "gate": {"pass": True}}
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(lr.validate_cache(data_path, meta_path, identity)["gate"]["pass"])
            alias_meta = data_path.with_name("s0_alias.npz").with_suffix(".json")
            alias_meta.write_text(json.dumps(dict(metadata, identical_to="ens")), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid identical-to"):  # only no_micro may alias another mode
                lr.validate_cache(data_path, alias_meta, identity, allow_identical=True)
            no_micro = self._identity(lr, mode="no_micro")
            no_micro_data, no_micro_meta = lr.cache_paths("R six-seed", "toy", "no_micro", 8, no_micro, cache_root=root)
            no_micro_meta.parent.mkdir(parents=True, exist_ok=True)
            no_micro_data = no_micro_data.with_name("no_micro_alias.npz")
            no_micro_meta = no_micro_data.with_suffix(".json")
            no_micro_alias = {"mode": "no_micro", "limit": 8, "cache_identity": no_micro["payload"],
                              "cache_identity_sha256": no_micro["sha256"], "identical_to": "ens"}
            no_micro_meta.write_text(json.dumps(no_micro_alias), encoding="utf-8")
            self.assertEqual(lr.validate_cache(no_micro_data, no_micro_meta, no_micro, allow_identical=True)["identical_to"], "ens")
            no_micro_alias["identical_to"] = "no_micro"
            no_micro_meta.write_text(json.dumps(no_micro_alias), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid identical-to"):  # a self-referential alias is rejected
                lr.validate_cache(no_micro_data, no_micro_meta, no_micro, allow_identical=True)

            claim_target = root / "claimed.npz"
            barrier = threading.Barrier(2)
            outcomes = []

            def attempt_claim():
                barrier.wait()
                try:
                    outcomes.append(("won", lr.claim_cache(claim_target, identity)))
                except RuntimeError:
                    outcomes.append(("blocked", None))

            contenders = [threading.Thread(target=attempt_claim) for _ in range(2)]
            for thread in contenders:
                thread.start()
            for thread in contenders:
                thread.join()
            self.assertEqual(sorted(state for state, _ in outcomes), ["blocked", "won"])
            won_path = next(path for state, path in outcomes if state == "won")
            self.assertEqual(json.loads(won_path.read_text(encoding="utf-8"))["status"], "claimed")
            with self.assertRaisesRegex(RuntimeError, "already claimed"):        # a failed run leaves its claim: no implicit retry
                lr.claim_cache(claim_target, identity)
            lr.complete_cache_claim(won_path)
            self.assertEqual(json.loads(won_path.read_text(encoding="utf-8"))["status"], "completed")

            legacy = root / "R six-seed" / "toy" / "s0_first8.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"legacy bytes")
            before = legacy.read_bytes()
            for changed in (self._identity(lr, implementation_digest="code-b"), self._identity(lr, spec=((1,), {})),
                            self._identity(lr, layer=False), self._identity(lr, seeds=(0, 2)),
                            self._identity(lr, input_manifest=[{"role": "store", "path": "toy-store", "sha256": "input-b"}])):
                self.assertNotEqual(lr.cache_paths("R six-seed", "toy", "s0", 8, changed, cache_root=root)[0], data_path)
            self.assertEqual(legacy.read_bytes(), before)                       # the legacy layout is never read or written
            self.assertEqual(lr.cached_gate_records(root, "R six-seed", "toy", "s", lambda _mode, _limit: identity),
                             {"s0_first8": {"pass": True}})
            data_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "damaged"):              # a damaged payload fails closed
                lr.validate_cache(data_path, meta_path, identity)

    def test_cache_source_paths_and_seed_gate_consumers_use_identity_api(self):
        from evaluation import extra_tables, lite_reanchor as lr
        import lite_reanchor as consumer_lr

        fake_sets = {"cineldi": ("C:/original/cineldi", None), "cambridge": ("C:/store/cambridge", None)}
        fake_p = SimpleNamespace(SETS=fake_sets, EXT={"bdg2": ("C:/ext/bdg2.npz", None, None)},
                                 EARLY={"oslo": (["C:/early/a.npz", "C:/early/b.npz"], None)}, GOI="C:/goi/store.npz")
        fake_r3 = SimpleNamespace(P=fake_p, PA=SimpleNamespace(SETS=fake_sets))
        with mock.patch.dict(sys.modules, {"tools_hcr_within_anchor_r3_v1": fake_r3}), \
                mock.patch.object(lr.CB, "corrected_store_info", lambda: ("C:/corrected/cineldi", {})):
            self.assertEqual([p.as_posix() for p in lr._source_store_paths("cambridge", {})], ["C:/store/cambridge/store.npz"])
            self.assertEqual([p.as_posix() for p in lr._source_store_paths("bdg2", {})], ["C:/ext/bdg2.npz"])
            self.assertEqual([p.as_posix() for p in lr._source_store_paths("oslo", {})], ["C:/early/a.npz", "C:/early/b.npz"])
            self.assertEqual([p.as_posix() for p in lr._source_store_paths("goiener_confirm", {})], ["C:/goi/store.npz"])
            self.assertEqual([p.as_posix() for p in lr._source_store_paths("cineldi", {"store": "corrected"})],
                             ["C:/original/cineldi/store.npz", "C:/corrected/cineldi/store.npz"])

        fake_z = dict(building=np.array([1]), origin=np.array([10]), traj=np.ones((1, 744)), level=np.ones(1), peak=np.ones(1),
                      off=np.zeros(1, bool), micro=np.zeros(1, bool))
        calls = []

        def fake_gates(_root, _label, _face, prefix, identity_factory):
            calls.append((prefix, identity_factory(f"{prefix}0", 8)))
            return {f"{prefix}0_first8": {"pass": True}}

        with mock.patch.object(extra_tables.np, "load", lambda *_a, **_k: fake_z), \
                mock.patch.object(consumer_lr, "cached_gate_records", fake_gates), \
                mock.patch.object(consumer_lr, "cache_key", lambda _prods: "synthetic-cache-label"), \
                mock.patch.object(consumer_lr, "_identity_for", lambda label, face, mode, entry, limit: (label, face, mode, limit)), \
                mock.patch.object(consumer_lr, "groups_of", lambda _entry: ((0, 1, 2), (3, 4, 5))):
            collector = extra_tables.Collector.__new__(extra_tables.Collector)
            collector.prods, collector.notes = {}, defaultdict(list)
            P = {"wbL": np.array([1]), "woL": np.array([10])}
            prod = {"pad": 0, "cont": None}
            _, seed_record = collector._seed_products("toy", {"seed_paths": ["seed0.npz"], "group_paths": [], "npz_path": "product.npz"},
                                                      P, prod, {})
            self.assertEqual(seed_record["gate"], {"s0_first8": {"pass": True}})
            group_entry = {"seed_paths": [], "group_paths": ["g0.npz", "g1.npz"], "group_continued_paths": [], "npz_path": "product.npz"}
            _, group_record = collector._seed_products("toy", group_entry, P, prod, {})
            self.assertEqual(group_record["gate"], {"g0_first8": {"pass": True}})
        self.assertEqual([prefix for prefix, _identity in calls], ["s", "g"])


class ContinuationFloatGuards(unittest.TestCase):
    def test_continuation_float_guards_and_default_equivalence(self):
        from ankyra_lite.continuation import CONSTANTS, Constants, ZERO, apply_batch, delta, envelope

        pre = np.full((1, 672), 100.0)
        last_type, days, state = [0], np.zeros((1, 7), dtype=int), [False]
        F = np.full((1, 744), 10.0)
        actual, info = apply_batch(F, pre, last_type, days, state)
        self.assertEqual((float(info["xf"][0]), float(info["xd"][0])), (1.0, 1.0))   # both anomalies clipped at +1
        h = np.arange(744, dtype=float)
        closed_form = 10.0 * np.exp((CONSTANTS.kappa1 * 1.0) * np.exp(-h / CONSTANTS.rho1)
                                    + (CONSTANTS.kappa2 * 1.0) * np.exp(-h / CONSTANTS.rho2))
        np.testing.assert_allclose(actual[0], closed_form, rtol=1e-12, atol=0.0)
        self.assertTrue(np.all(delta(0.1, -1.0) < 0))
        np.testing.assert_array_equal(envelope(ZERO), np.zeros(744))

        zeros = np.zeros((1, 744), dtype=np.float64)
        zeros[0, 0] = -0.0
        huge = Constants(1000.0, 12.0, 1000.0, 168.0)
        zero_out, _ = apply_batch(zeros, pre, last_type, days, state, huge)
        np.testing.assert_array_equal(zero_out, zeros)
        self.assertTrue(np.signbit(zero_out[0, 0]))                             # negative zero is kept
        state_input = np.ones((2, 744))
        state_input[0, 0] = np.nan
        state_input[1, 0] = -0.0
        state_out, state_info = apply_batch(state_input, np.full((2, 672), np.nan), [0, 0], np.zeros((2, 7), int), [True, True], huge)
        np.testing.assert_array_equal(state_out, state_input)                   # state windows are bypassed bit for bit
        self.assertTrue(np.signbit(state_out[1, 0]) and bool(state_info["bypass_state"].all()))
        with self.assertRaisesRegex(ValueError, "non-finite continuation output"):
            apply_batch(F, pre, last_type, days, state, huge)
        with self.assertRaisesRegex(ValueError, "non-finite continuation delta"):
            apply_batch(F, pre, last_type, days, state, Constants(1e308, 12.0, 1e308, 168.0))
        underflow, _ = apply_batch(np.ones((1, 744)), np.zeros((1, 672)), last_type, days, state, huge)
        self.assertTrue(np.isfinite(underflow).all() and underflow[0, 0] == 0.0 and underflow[0, -1] > 0.0)


if __name__ == "__main__":
    unittest.main()
