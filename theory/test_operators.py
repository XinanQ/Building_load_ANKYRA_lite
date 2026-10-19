"""Exact properties P1-P20 of theory/PROOFS.md, checked numerically, together with the counterexamples that bound them.

The anchoring layer is ``ankyra_lite.anchor`` (a byte-identical copy of the ANKYRA 2.2.1 package); these checks are the
ANKYRA 2.2.1 ``theory/test_operators.py`` with only the imports adapted.  Run from the repository root:

    python -m unittest discover -s theory -t .
"""
import itertools
import unittest

import numpy as np
import torch

from ankyra_lite.anchor import blocks, metrics
from ankyra_lite.anchor.history import estimate_from_history
from theory import operators as op
from ankyra_lite.anchor.synthetic import synthetic_history

torch.set_num_threads(1)
rng = np.random.default_rng(11)


class BlockGeometryTests(unittest.TestCase):                                     # P1-P3
    def test_four_projections_are_orthogonal_and_have_ranks_1_30_23_690(self):
        P = op.four_block_projections()
        np.testing.assert_allclose(sum(P), np.eye(744), atol=1e-12)
        for i, Pi in enumerate(P):
            np.testing.assert_allclose(Pi @ Pi, Pi, atol=1e-12)
            np.testing.assert_allclose(Pi, Pi.T, atol=1e-15)
            for Pj in P[i + 1:]:
                np.testing.assert_allclose(Pi @ Pj, 0.0, atol=1e-12)
        self.assertEqual([int(round(np.trace(Pi))) for Pi in P], [1, 30, 23, 690])

    def test_four_block_losses_refine_the_three_block_identity(self):
        F, y = rng.normal(10, 3, (20, 744)), rng.normal(10, 3, (20, 744))
        l4 = op.four_block_losses(F, y)
        l3 = blocks.block_losses(F, y)
        np.testing.assert_allclose(sum(l4), ((F - y) ** 2).mean(1), atol=1e-10)
        np.testing.assert_allclose(l4[0], l3[0], atol=1e-12)
        np.testing.assert_allclose(l4[1], l3[1], atol=1e-12)
        np.testing.assert_allclose(l4[2] + l4[3], l3[2], atol=1e-10)

    def test_the_24_hour_periodic_subspace_is_the_daily_harmonics_and_there_is_no_weekly_bin(self):
        x = rng.normal(0, 1, 744)
        periodic = np.tile(x.reshape(31, 24).mean(0), 31)                           # J_D (x) I_H applied to x
        X = np.fft.fft(x); keep = np.zeros(744, bool); keep[op.daily_harmonic_frequencies()] = True
        np.testing.assert_allclose(np.fft.ifft(np.where(keep, X, 0)).real, periodic, atol=1e-12)
        self.assertEqual(len(op.daily_harmonic_frequencies()), 24)
        self.assertTrue(op.has_fourier_bin(24) and not op.has_fourier_bin(168))
        self.assertEqual(np.gcd(7, 31), 1)

    def test_energy_error_is_744_times_the_level_error(self):
        F, y = rng.normal(10, 3, 744), rng.normal(10, 3, 744)
        self.assertAlmostEqual(F.sum() - y.sum(), 744 * (blocks.level(F) - blocks.level(y)), places=8)


class ReplacementTests(unittest.TestCase):                                        # P4-P6
    def test_replacement_changes_only_the_replaced_blocks(self):
        F, y = rng.normal(20, 4, 744), rng.normal(21, 4, 744)
        path = rng.normal(0, 1, 31); path -= path.mean()
        for kw in ({"level": 23.0}, {"daily_path": path}, {"level": 19.0, "daily_path": path}):
            G = op.replace_blocks(F, **kw)
            d = op.mse_change_by_block(F, G, y)
            self.assertAlmostEqual(sum(d), ((G - y) ** 2).mean() - ((F - y) ** 2).mean(), places=10)
            np.testing.assert_allclose(blocks.within_day(G), blocks.within_day(F), atol=1e-12)
            if "daily_path" not in kw:
                self.assertAlmostEqual(d[1], 0.0, places=12)
            if "level" not in kw:
                self.assertAlmostEqual(d[0], 0.0, places=12)

    def test_fixed_replacements_are_idempotent_and_commute(self):
        F = rng.normal(20, 4, 744); path = rng.normal(0, 1, 31); path -= path.mean()
        A = op.replace_blocks(F, level=17.0)
        np.testing.assert_allclose(op.replace_blocks(A, level=17.0), A, atol=1e-12)
        np.testing.assert_allclose(op.replace_blocks(op.replace_blocks(F, level=17.0), daily_path=path),
                                   op.replace_blocks(op.replace_blocks(F, daily_path=path), level=17.0), atol=1e-12)

    def test_level_only_replacement_is_bounded_by_the_level_share(self):
        F, y = rng.normal(20, 4, (40, 744)), rng.normal(22, 4, (40, 744))
        share = op.level_share(F, y); mse0 = ((F - y) ** 2).mean(1)
        oracle = op.replace_blocks(F, level=blocks.level(y))              # the best possible level
        np.testing.assert_allclose(1 - ((oracle - y) ** 2).mean(1) / mse0, share, atol=1e-12)
        for _ in range(5):
            G = op.replace_blocks(F, level=blocks.level(y) + rng.normal(0, 2, 40))
            self.assertTrue(((1 - ((G - y) ** 2).mean(1) / mse0) <= share + 1e-12).all())

    def test_ratio_form_of_the_mse_change(self):
        F, y = rng.normal(20, 4, 744), rng.normal(21, 4, 744)
        path = rng.normal(0, 2, 31); path -= path.mean()
        G = op.replace_blocks(F, level=20.5, daily_path=path)
        old, new = np.array(blocks.block_losses(F, y)), np.array(blocks.block_losses(G, y))
        self.assertAlmostEqual(op.mse_ratio_from_blocks(old / old.sum(), np.sqrt(new / old)),
                               ((G - y) ** 2).mean() / ((F - y) ** 2).mean(), places=12)

    def test_division_log_ratio_gains_are_bounded_and_losses_are_not(self):
        T, H, y = rng.normal(10, 3, (200, 744)), rng.normal(10, 3, (200, 744)), rng.normal(10, 3, (200, 744))
        div = op.replace_blocks(T, level=blocks.level(H), daily_path=blocks.daily_path(H))
        lt, bt, wt = blocks.block_losses(T, y); lh, bh, _ = blocks.block_losses(H, y)
        pi, r = (lt + bt) / (lt + bt + wt), (lh + bh) / (lt + bt)
        np.testing.assert_allclose(op.division_log_ratio(pi, r),
                                   0.5 * np.log(((div - y) ** 2).mean(1) / ((T - y) ** 2).mean(1)), atol=1e-12)
        self.assertTrue((op.division_log_ratio(pi, r) >= 0.5 * np.log1p(-pi) - 1e-12).all())
        self.assertGreater(op.division_log_ratio(0.5, 1e6), 6.0)                     # unbounded loss

    def test_blockwise_selection_is_the_pooled_optimum_of_two_sources(self):
        a, b = rng.gamma(2, 2, 3), rng.gamma(2, 2, 3)
        choice, best = op.blockwise_selection(a, b)
        allmse = [sum(b[k] if c[k] else a[k] for k in range(3)) for c in itertools.product((0, 1), repeat=3)]
        self.assertAlmostEqual(best, min(allmse), places=12)

    def test_correction_accounting(self):
        ref, y = rng.normal(10, 2, 744), rng.normal(10, 2, 744)
        cor = ref + rng.normal(0, 1, 744)
        size, align, change = op.correction_accounting(ref, cor, y)
        self.assertAlmostEqual(change, ((cor - y) ** 2).mean() - ((ref - y) ** 2).mean(), places=10)
        self.assertAlmostEqual(change, size - 2 * align, places=12)
        r, delta = 1.5, -1.0                                                         # one window: helps iff delta(2r+delta) < 0
        self.assertEqual((r + delta) ** 2 < r ** 2, delta * (2 * r + delta) < 0)


class SupportTests(unittest.TestCase):                                            # P7
    def test_completed_windows_formula_and_mask_monotonicity(self):
        for o in (1344, 2088, 2832, 4392, 8760, 10248, 12000):
            full = np.ones(o, bool)
            self.assertEqual(op.completed_windows(full, o), op.effective_pseudo_origins(o))
            self.assertEqual(op.completed_windows(full, o, lag=8760), max(0, min(12, (o - 8760) // 744)))
        self.assertEqual(1344 + 2 * 744, 2832)                                     # earliest origin with two errors
        self.assertEqual(op.annual_support_hours(2), 10248)
        for _ in range(50):
            o = int(rng.integers(2000, 12000)); mask = np.ones(o, bool)
            base = op.completed_windows(mask, o)
            for _ in range(3):
                s = int(rng.integers(0, o)); mask[s:s + int(rng.integers(1, 400))] = False
                now = op.completed_windows(mask, o)
                self.assertLessEqual(now, base); base = now

    def test_effective_pseudo_origins_and_shares(self):
        self.assertEqual([op.effective_pseudo_origins(m) for m in (1344, 4392, 6576, 8760, 10272, 20000)], [0, 4, 7, 9, 12, 12])
        self.assertEqual(op.data_share(1), 0.0)                                    # one error: equal-weight fallback
        self.assertAlmostEqual(op.data_share(2), 0.2)
        self.assertAlmostEqual(op.data_share(12), 0.6)

    def test_reference_estimator_matches_the_support_formula_and_the_weather_bounds(self):
        for hours in (2900, 4392, 6000, 7400):
            dg = estimate_from_history(synthetic_history(hours), group="Office", temp_sigma_std=0.25).diagnostics
            k = dg["level_k_eff"]
            self.assertEqual(k, op.effective_pseudo_origins(hours))
            lo, hi = op.weather_weight_bounds(k)
            self.assertTrue(lo - 1e-12 <= dg["weather_weight"] <= hi + 1e-12, (hours, dg["weather_weight"], lo, hi))


class ShrinkageTests(unittest.TestCase):                                          # P8-P10
    def test_mass_and_prediction_bounds_on_random_simplices(self):
        for _ in range(2000):
            m = int(rng.integers(2, 8)); p = np.full(m, 1 / m); q = rng.dirichlet(np.ones(m)); rho = rng.uniform()
            w = (1 - rho) * p + rho * q
            B = rng.random(m) < 0.5
            lo, hi = op.mixture_mass_bounds(p[B].sum(), rho)
            self.assertTrue(lo - 1e-15 <= w[B].sum() <= hi + 1e-15)
            h = rng.normal(0, 3, m)
            bound = op.prediction_movement_bound(h, rho, q, p)
            self.assertLessEqual(abs(h @ (w - p)), bound + 1e-12)
            self.assertLessEqual(bound, rho * (1 - 1 / m) * (h.max() - h.min()) + 1e-12)

    def test_weather_bounds(self):
        self.assertEqual(op.weather_weight_bounds(1), (0.5, 0.5))
        np.testing.assert_allclose(op.weather_weight_bounds(2), (0.4, 0.6))
        np.testing.assert_allclose(op.weather_weight_bounds(4), (1 / 3, 2 / 3))
        np.testing.assert_allclose(op.weather_weight_bounds(12), (0.2, 0.8))

    def test_deletion_renormalises_the_mixture_counterexample(self):
        p, q, rho, R = np.full(4, 0.25), np.array([0.0, 0.0, 1.0, 0.0]), 0.2, [0, 2]
        w = (1 - rho) * p + rho * q
        wR = w[R] / w[R].sum()
        rhoR = op.retained_data_share(rho, p[R].sum(), q[R].sum())
        self.assertAlmostEqual(rhoR, 1 / 3)
        np.testing.assert_allclose(wR, (1 - rhoR) * p[R] / p[R].sum() + rhoR * q[R] / q[R].sum(), atol=1e-15)
        self.assertAlmostEqual(wR[1], 2 / 3)                                         # weather mass 2/3 > old bound 0.6
        self.assertGreater(wR[1], op.weather_weight_bounds(2)[1])

    def test_support_threshold_is_discontinuous(self):
        uniform, two_errors = np.full(4, 0.25), 0.8 * np.full(4, 0.25) + 0.2 * np.eye(4)[0]
        self.assertAlmostEqual(np.abs(two_errors - uniform).sum(), 0.3)

    def test_shared_noise_pulls_inverse_mse_weights_to_one_half_but_not_least_squares(self):
        r = np.random.default_rng(3); n = 1_000_000; y = np.zeros(n)
        for common in (0.0, 3.0, 10.0):
            c = r.normal(0, common, n) if common else 0.0
            a, b = c + r.normal(0, 1.0, n), c + r.normal(0, 0.5, n)              # b is the better source
            ls, inv = op.two_source_weights(y, a, b)
            if common <= 3:
                self.assertAlmostEqual(ls, 1.0 / (1.0 + 0.25), delta=0.015)          # sigma_a^2 / (sigma_a^2 + sigma_b^2)
            expected_inv = (common ** 2 + 1.0) / (2 * common ** 2 + 1.25)            # MSE_a / (MSE_a + MSE_b)
            self.assertAlmostEqual(inv, expected_inv, delta=0.01)
        self.assertLess(abs(inv - 0.5), 0.01)                                        # large shared noise: close to 1/2


class FeasibilityTests(unittest.TestCase):                                        # P11-P13
    def test_clipping_never_increases_any_hours_error_and_is_positively_homogeneous(self):
        x, y = rng.normal(0.5, 2, 5000), np.abs(rng.normal(0, 2, 5000))
        self.assertTrue((np.abs(op.clip_nonnegative(x) - y) <= np.abs(x - y) + 1e-15).all())
        np.testing.assert_allclose(op.clip_nonnegative(3.7 * x), 3.7 * op.clip_nonnegative(x))

    def test_mean_preserving_projection_counterexample_and_impossibility(self):
        p, y = np.array([-1.0, 3.0]), np.array([0.0, 4.0])
        np.testing.assert_allclose(op.project_to_mean(p), [0.0, 2.0])
        self.assertAlmostEqual(np.sum((op.project_to_mean(p) - y) ** 2), 4.0)      # worse than the unprojected 2
        self.assertAlmostEqual(np.sum((p - y) ** 2), 2.0)
        self.assertAlmostEqual(np.sum((op.clip_nonnegative(p) - y) ** 2), 1.0)
        a = np.linspace(0, 2, 2001)                                                  # every nonnegative output with mean 1
        self.assertTrue(((a - y[0]) ** 2 + (2 - a - y[1]) ** 2 > np.sum((p - y) ** 2)).all())

    def test_simplex_projection_is_the_euclidean_projection(self):
        for _ in range(200):
            v, s = rng.normal(0, 2, 7), rng.uniform(0, 5)
            z = op.project_to_mean(v, s / 7)
            self.assertAlmostEqual(z.sum(), s, places=10); self.assertTrue((z >= 0).all())
            for _ in range(20):                                                      # no feasible point is closer
                t = rng.dirichlet(np.ones(7)) * s
                self.assertLessEqual(np.sum((z - v) ** 2), np.sum((t - v) ** 2) + 1e-12)

    def test_day_level_projection_guarantee_under_its_condition(self):
        r = np.random.default_rng(5); held = active = 0
        for _ in range(3000):
            y = np.abs(r.normal(1, 1, 744)) * (r.random(31) < 0.8).repeat(24)     # nonnegative, some zero days
            F = y + r.normal(0, 0.5, 744) + np.repeat(r.normal(0, 1.5, 31), 24)     # often negative daily means
            if F.mean() < 0 or not op.daily_projection_condition(F.mean(), blocks.daily_means(y)):
                continue
            held += 1
            m, ms = blocks.daily_means(F), blocks.daily_means(y)
            mp = op.project_daily_means(m)
            active += not np.allclose(mp, m)
            pre = F + np.repeat(mp - m, 24)                                          # before the final max(., 0)
            self.assertLessEqual(np.sum((mp - ms) ** 2), np.sum((m - ms) ** 2) + 1e-9)
            self.assertLessEqual(((pre - y) ** 2).mean(), ((F - y) ** 2).mean() + 1e-9)
            self.assertLessEqual(((op.daily_projection(F) - y) ** 2).mean(), ((pre - y) ** 2).mean() + 1e-12)
        self.assertGreater(held, 500); self.assertGreater(active, 100)
        # outside the condition the error can rise: the level is under-forecast by more than the smallest true daily mean
        m, ms = np.r_[-1.0, np.full(30, 3.0)], np.r_[0.0, np.full(30, 5.0)]
        self.assertFalse(op.daily_projection_condition(m.mean(), ms))
        self.assertGreater(np.sum((op.project_daily_means(m) - ms) ** 2), np.sum((m - ms) ** 2))


class PeakTests(unittest.TestCase):                                               # P14-P18
    def test_maximum_of_the_mean_path_underestimates_the_expected_maximum(self):
        Y = rng.gamma(2, 3, (500, 744))
        self.assertLessEqual(Y.mean(0).max(), Y.max(1).mean())

    def test_error_decomposition_is_exact_and_selection_is_nonnegative(self):
        for _ in range(300):
            F, y = rng.gamma(3, 3, 744), rng.gamma(3, 3, 744)
            A = rng.gamma(2, 3, 31); kap = rng.uniform(0.8, 1.3)
            dec = op.peak_error_decomposition(F, y, A, kappa=kap)
            self.assertAlmostEqual(dec["level"] + dec["between_day"] + dec["amplitude"] + dec["selection"], dec["U_minus_peak"], places=9)
            self.assertGreaterEqual(dec["selection"], -1e-12)
            P = F.mean() + kap * (dec["U"] - F.mean())
            self.assertAlmostEqual(P - y.max(), dec["U_minus_peak"] + dec["kappa_term"], places=9)

    def test_finite_support_median_property_and_counterexamples(self):
        s = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertTrue(op.envelope_is_median(4, 3) and not op.envelope_is_median(4, 2))
        self.assertEqual(op.max_lower_median([0, 0, 0], [s, s, s]), 3.0)             # median = envelope
        self.assertEqual(op.max_lower_median([0, 0], [s, s]), 2.0)                    # two days: median 2, envelope 3
        self.assertEqual(op.max_lower_median([0, -100, -100], [s, s, s]), 1.0)        # unequal levels: median 1
        for m in range(2, 9):
            for n in range(1, 8):
                med = op.max_lower_median(np.zeros(n), [np.arange(m, dtype=float)] * n)
                self.assertEqual(med == m - 1, op.envelope_is_median(m, n))

    def test_iid_joint_maximum_is_historical_with_probability_m_over_m_plus_n(self):
        m, n = 4, 3
        x = rng.random((200000, m + n))
        self.assertAlmostEqual((x.argmax(1) < m).mean(), op.historical_max_probability(m, n), delta=0.005)

    def test_asymmetric_cost_is_twice_the_pinball_loss(self):
        p, y = rng.gamma(3, 10, 500), rng.gamma(3, 10, 500)
        for lam in (0.5, 1.0, 2.0, 4.0):
            self.assertAlmostEqual(op.asymmetric_peak_cost(p, y, lam), 2 * op.pinball(p, y, lam / (1 + lam)), places=10)
        self.assertAlmostEqual(op.asymmetric_peak_cost(p, y, 1.0), float(np.abs(p - y).mean()), places=10)


class EstimandTests(unittest.TestCase):                                           # P19
    def test_pooled_ratio_identity_and_opposite_signs(self):
        unit = np.repeat(np.arange(30), 5)
        b = rng.gamma(2, 1, 150) * np.repeat(rng.gamma(1, 3, 30), 5)
        a = b * np.repeat(rng.uniform(0.5, 1.5, 30), 5)
        dec = metrics.pooled_ratio_decomposition(a, b, unit)
        self.assertAlmostEqual(dec["pooled_ratio"], a.sum() / b.sum(), places=12)
        self.assertAlmostEqual(dec["unit_equal_log_ratio"], metrics.log_ratio(a, b, unit), places=12)
        # nine small units improve by half, one large unit worsens by 50 %: unit-equal favours a, pooled favours b
        unit = np.arange(10); b = np.array([1.0] * 9 + [100.0]); a = b * np.array([0.25] * 9 + [2.25])
        dec = metrics.pooled_ratio_decomposition(a, b, unit)
        self.assertLess(dec["unit_equal_log_ratio"], 0); self.assertGreater(dec["pooled_ratio"], 1)


class EvidenceBoundTests(unittest.TestCase):                                      # P20
    def test_rates_for_the_study_sizes(self):
        self.assertAlmostEqual(op.aggregation_rate(7, 6), min(7 / 6, np.sqrt(np.log(7) / 6)), places=12)
        self.assertGreater(op.aggregation_rate(7, 6), 0.5); self.assertLess(op.aggregation_rate(7, 30), 0.25)
        self.assertGreater(op.selection_rate(7, 6), 0.3); self.assertLess(op.selection_rate(7, 30), 0.07)
        for T in (6, 12, 30, 120):
            self.assertGreater(op.aggregation_rate(7, T), op.aggregation_rate(7, T + 1))

    def test_exchangeable_errors_give_chance_hit_rate(self):
        rng = np.random.default_rng(20261006)
        self.assertAlmostEqual(op.exchangeable_hit_rate(7, 6, 40000, rng), 1 / 7, delta=0.01)
        self.assertAlmostEqual(op.exchangeable_hit_rate(6, 30, 40000, rng), 1 / 6, delta=0.01)


if __name__ == "__main__":
    unittest.main()
