"""The evaluation module: scoring blocks, late subset, rank tests, scaled errors, energy, forecast days, error blocks,
peak contrasts, windows, baselines and the runnable example (artificial data only).

Run from the repository root:  python -m unittest tests.test_evaluation -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

from ankyra_lite.anchor import blocks, metrics, readouts      # noqa: E402
from evaluation import baselines, scoring, windows             # noqa: E402

REF = scoring.REFERENCE


def toy(n_units=8, per_unit=3, seed=1):
    rng = np.random.default_rng(seed); n = n_units * per_unit
    y = 10 + rng.random((n, 744)); unit = np.repeat([f"u{i}" for i in range(n_units)], per_unit); month = np.tile(["2020-01", "2020-02", "2020-03"], n_units)
    F = {REF: y + 0.1 * rng.standard_normal(y.shape), "A": y + 0.3 * rng.standard_normal(y.shape), "B": y + 0.2 * rng.standard_normal(y.shape)}
    return F, y, unit, month, 10 + rng.random((n, 1344))


class PackageTests(unittest.TestCase):
    def test_this_repositorys_evaluation_package_is_imported(self):
        for m in (scoring, windows, baselines):
            self.assertTrue(Path(m.__file__).resolve().is_relative_to(REPO / "evaluation"), m.__file__)
        self.assertEqual(REF, "ankyra-lite")
        self.assertIs(scoring.metrics, metrics)                       # the anchoring layer's estimands


class ScoringTests(unittest.TestCase):
    def test_block_matches_the_package_estimand_and_resolves_clear_differences(self):
        F, y, unit, month, _ = toy()
        b = scoring.block(F, y, unit, month, REF, n_boot=200)
        r = metrics.log_ratio(scoring.window_mse(F[REF], y), scoring.window_mse(F["A"], y), unit)
        self.assertEqual(b["pairwise"]["A"]["log_ratio"], r)
        self.assertEqual(b["pairwise"]["A"]["resolved"], "better")
        self.assertLess(b["pairwise"]["A"]["um_low"], b["pairwise"]["A"]["um_high"])
        self.assertEqual(min(b["mean_unit_rank"], key=b["mean_unit_rank"].get), REF)
        self.assertEqual((b["windows"], b["units"], b["months"]), (24, 8, 3))
        self.assertEqual(b["conventional"]["A"]["units_excluded_from_ratios"], 0)
        iv = metrics.um_interval(scoring.window_mse(F[REF], y), scoring.window_mse(F["B"], y), unit, month,
                                 np.random.default_rng(scoring.SEED), 200)["UM"]
        self.assertEqual([b["pairwise"]["B"]["um_low"], b["pairwise"]["B"]["um_high"]], iv)   # a fresh generator per contrast

    def test_intervals_are_reproducible_and_family_is_left_out_of_the_ranking(self):
        F, y, unit, month, _ = toy(); F["carrier alone"] = F[REF] + 0.01
        a = scoring.block(F, y, unit, month, REF, family=("carrier alone",), n_boot=100)
        b = scoring.block(F, y, unit, month, REF, family=("carrier alone",), n_boot=100)
        self.assertEqual(a["pairwise"], b["pairwise"])
        self.assertIn("carrier alone", a["pairwise"]); self.assertNotIn("carrier alone", a["mean_unit_rank"])

    def test_late_block_uses_the_windows_every_model_covers(self):
        F, y, unit, month, _ = toy(); F["trained"] = F["B"].copy(); F["trained"][month == "2020-01"] = np.nan
        r = scoring.score_population(F, y, unit, month, n_boot=50)
        self.assertNotIn("trained", r["full"]["pairwise"]); self.assertIn("trained", r["late"]["pairwise"])
        self.assertEqual(r["late"]["windows"], 16)
        np.testing.assert_array_equal(scoring.late_windows(F), month != "2020-01")
        with self.assertRaises(ValueError):
            scoring.score_population({**F, REF: F["trained"]}, y, unit, month, n_boot=10)

    def test_rank_tests_scaled_errors_energy_and_days(self):
        F, y, unit, month, ctx = toy(n_units=20)
        t = scoring.rank_tests(scoring.unit_rmse(F, y, unit)[1])
        self.assertEqual(t["units"], 20); self.assertEqual(set(t["significantly_better_than"]), {"A", "B"}); self.assertLess(t["friedman_p"], 0.01)
        self.assertEqual(t["reference_position"], 1)
        s = scoring.scaled_errors(F, y, ctx, unit)
        self.assertLess(s[REF]["RMSSE_geo_mean"], s["A"]["RMSSE_geo_mean"])
        self.assertTrue(0.0 <= s["A"]["units_below_1_share"] <= 1.0)
        e = scoring.energy_error(F, y, unit, month, n_boot=50); self.assertEqual(set(e), {"A", "B"})
        d = scoring.by_forecast_day(F, y, unit)
        self.assertEqual(d["gm_cv_rmse_pct"][REF].shape, (31,)); self.assertEqual(d["units_in_set"], 20)
        for key in ("rmse_kw", "mae_kw", "cv_rmse_pct", "nmae_pct"):
            self.assertEqual(d[key]["A"].shape, (31,), key)
        self.assertTrue((d["rmse_kw"]["A"] >= d["mae_kw"]["A"]).all())

    def test_energy_contrasts_share_one_stream_in_sorted_order(self):
        F, y, unit, month, _ = toy()
        e = scoring.energy_error(F, y, unit, month, n_boot=30)
        lev = {k: (y - p).mean(1) for k, p in F.items()}
        rng = np.random.default_rng(scoring.ENERGY_SEED)
        for k in ("A", "B"):                                                  # sorted order, one generator
            iv = metrics.um_interval(lev[REF] ** 2, lev[k] ** 2, unit, month, rng, 30)["UM"]
            self.assertEqual([e[k]["um_low"], e[k]["um_high"]], iv, k)
            self.assertEqual(e[k]["log_ratio"], metrics.log_ratio(lev[REF] ** 2, lev[k] ** 2, unit))
        self.assertAlmostEqual(e["A"]["median_energy_ape_pct"], scoring.energy_ape(lev["A"], y, unit))
        same = scoring.energy_error(F, y, unit, month, n_boot=30, reference_level=F[REF].mean(1))
        for k in ("A", "B"):
            self.assertAlmostEqual(same[k]["log_ratio"], e[k]["log_ratio"], places=10)

    def test_energy_by_forecast_day(self):
        F, y, unit, month, _ = toy()
        r = scoring.energy_by_forecast_day(F, y, unit)
        self.assertEqual(r["units_in_set"], 8)
        d, c = r["gm_cv_daily_energy_pct"]["A"], r["gm_cv_cumulative_energy_pct"]["A"]
        self.assertEqual((d.shape, c.shape), ((31,), (31,)))
        self.assertAlmostEqual(d[0], c[0])                                    # day 1: the cumulative error is the day's error
        e = (y - F["A"]).reshape(len(y), 31, 24).mean(2)
        uu = np.unique(unit); ybar = np.array([y[unit == u].mean() for u in uu])
        cum = np.stack([np.sqrt((e[unit == u].mean(1) ** 2).mean()) for u in uu])     # mean over the 31 days = window mean
        self.assertAlmostEqual(c[-1], float(np.exp(np.log(100 * cum / ybar).mean())), places=9)

    def test_block_shares_add_up_and_agree_with_the_block_operators(self):
        F, y, unit, month, _ = toy()
        r = scoring.block_shares(F, y, unit, against=("A",))
        for k, v in r["shares"].items():
            self.assertAlmostEqual(sum(v["pooled"].values()), 1.0, places=12, msg=k)
        lv, path, within = blocks.block_losses(F["A"], y)
        np.testing.assert_allclose(lv + path + within, scoring.window_mse(F["A"], y), rtol=1e-12)
        self.assertEqual(set(r["contrasts"]), {"A"})
        c = r["contrasts"]["A"]
        self.assertEqual(set(c), set(scoring.BLOCKS) | {"hourly"})
        hourly = metrics.log_ratio(scoring.window_mse(F[REF], y), scoring.window_mse(F["A"], y), unit)
        self.assertAlmostEqual(c["hourly"]["log_ratio"], hourly, places=12)  # equal window counts per unit
        self.assertEqual(c["level"]["units_in_ratio"], 8)

    def test_peak_readout_and_contrasts(self):
        F, y, unit, month, ctx = toy()
        n = len(y); ct = np.tile(np.arange(56) % 7, (n, 1)); tt = np.tile(np.arange(31) % 7, (n, 1))
        pk = scoring.peak_readout(F[REF], ctx, ct, tt)
        np.testing.assert_array_equal(pk, readouts.peak_readout(F[REF], ctx, ct, tt))
        np.testing.assert_array_equal(scoring.previous_window_peak(ctx), ctx[:, -744:].max(1))
        comps = {"exact": y.max(1), "trajectory maximum": F[REF].max(1), "same readout on A": scoring.peak_readout(F["A"], ctx, ct, tt)}
        r = scoring.peak_contrasts(pk, comps, y, unit, month, n_boot=50)
        self.assertNotIn("exact", r)                                          # an undefined log ratio is left out
        self.assertEqual(set(r), {"trajectory maximum", "same readout on A"})
        a = (pk - y.max(1)) ** 2; b = (F[REF].max(1) - y.max(1)) ** 2
        self.assertEqual(r["trajectory maximum"]["log_ratio"], metrics.log_ratio(a, b, unit))
        iv = metrics.um_interval(a, b, unit, month, np.random.default_rng(scoring.PEAK_SEED), 50)["UM"]
        self.assertEqual([r["trajectory maximum"]["um_low"], r["trajectory maximum"]["um_high"]], iv)
        self.assertGreater(scoring.peak_ape(F["A"].max(1), y, unit), 0.0)

    def test_plain_lists_give_the_same_results_as_arrays(self):
        F, y, unit, month, ctx = toy()
        Fl, yl, ul, ml, cl = {k: v.tolist() for k, v in F.items()}, y.tolist(), list(unit), list(month), ctx.tolist()
        self.assertEqual(scoring.energy_error(Fl, yl, ul, ml, n_boot=20), scoring.energy_error(F, y, unit, month, n_boot=20))
        self.assertEqual(scoring.scaled_errors(Fl, yl, cl, ul), scoring.scaled_errors(F, y, ctx, unit))
        a, b = scoring.by_forecast_day(Fl, yl, ul), scoring.by_forecast_day(F, y, unit)
        self.assertEqual(a["units_in_set"], b["units_in_set"]); np.testing.assert_array_equal(a["gm_cv_rmse_pct"]["A"], b["gm_cv_rmse_pct"]["A"])
        np.testing.assert_array_equal(scoring.unit_rmse(Fl, yl, ul)[1]["B"], scoring.unit_rmse(F, y, unit)[1]["B"])
        self.assertEqual(scoring.score_population(Fl, yl, ul, ml, n_boot=20), scoring.score_population(F, y, unit, month, n_boot=20))
        self.assertEqual(scoring.block_shares(Fl, yl, ul)["shares"], scoring.block_shares(F, y, unit)["shares"])


class WindowAndBaselineTests(unittest.TestCase):
    def test_usable_origins_need_complete_context_target_and_history(self):
        load = np.ones(12000); load[9000] = np.nan
        o = windows.usable_origins(load, [3000, 6000, 8500, 9500, 10400, 11500])
        self.assertEqual(list(o), [6000, 10400])                           # dropped: too little history, a gap in the target or the context, no room for the target
        self.assertEqual(list(windows.usable_origins(load, [6000, 10400], cut=1536)), [6000, 10400])   # 6000 - 4464 = 1536 >= cut
        self.assertEqual(list(windows.usable_origins(load, [6000, 10400], cut=1537)), [10400])
        self.assertEqual(list(windows.usable_origins(load, [6000, 10400], cut=5937)), [])
        self.assertEqual(windows.GAP_HOURS, 186 * 24)
        starts = windows.month_starts("2020-01-01T00:00:00+00:00", 2000)
        self.assertEqual(list(starts), [0, 744, 1440])
        ctx, y, month = windows.window_arrays(np.arange(12000.0), [6000], "2020-01-01T00:00:00+00:00")
        self.assertEqual((ctx.shape, y.shape), ((1, 1344), (1, 744))); self.assertEqual(ctx[0, -1], 5999.0); self.assertEqual(y[0, 0], 6000.0)

    def test_day_types_are_read_at_the_middle_of_each_origin_aligned_day(self):
        types = np.repeat(np.arange(1000) % 8, 24)
        ct, tt = windows.day_types(types, [2400, 2416])
        self.assertEqual((ct.shape, tt.shape), ((2, 56), (2, 31)))
        np.testing.assert_array_equal(tt[0], np.arange(100, 131) % 8)                # midnight origin: the calendar days
        np.testing.assert_array_equal(tt[1], np.arange(101, 132) % 8)                # origin at 16:00: the 12th hour is on the next day
        np.testing.assert_array_equal(ct[0], np.arange(44, 100) % 8)

    def test_baselines_repeat_the_context(self):
        ctx = np.arange(1344.0)[None]
        d = baselines.seasonal_naive(ctx, 24); w = baselines.seasonal_naive(ctx, 168); p = baselines.week_profile(ctx)
        self.assertEqual(d.shape, (1, 744)); np.testing.assert_array_equal(d[0, :24], ctx[0, -24:]); np.testing.assert_array_equal(d[0, 24:48], ctx[0, -24:])
        np.testing.assert_array_equal(w[0, :168], ctx[0, -168:])
        np.testing.assert_allclose(p[0, :168], ctx[0, -672:].reshape(4, 168).mean(0))

    def test_weekly_seasonal_naive_is_the_state_rule_forecast(self):
        from ankyra_lite.forecast import seasonal_naive
        ctx = np.random.default_rng(0).gamma(2.0, 3.0, (1, 1344))
        np.testing.assert_array_equal(baselines.seasonal_naive(ctx, 168)[0], seasonal_naive(ctx[0]))


class RunExampleTests(unittest.TestCase):
    def test_the_example_runs_end_to_end_with_the_untrained_stand_in(self):
        from evaluation import run_example
        models, description = run_example.carriers(untrained=True)
        self.assertEqual(len(models), 1); self.assertIn("untrained", description)
        out = run_example.run(n_units=2, n_boot=20, models=models, description=description)
        self.assertEqual((out["windows"], out["units"]), (6, 2))
        self.assertEqual(set(out["ankyra_lite_against"]), {"carrier alone", "SN-day", "SN-week", "WeekProf4"})
        self.assertNotIn("carrier alone", out["mean_unit_rank"])
        self.assertEqual(set(out["peak_improvement_pct"]), {"trajectory maximum", "previous-window peak", "same readout on SN-week"})
        self.assertTrue(all(math.isfinite(v) for v in out["rmsse_geo_mean"].values()))


if __name__ == "__main__":
    unittest.main()
