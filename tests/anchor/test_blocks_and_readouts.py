# Ported from Building_load_ANKYRA/tests/test_blocks_and_readouts.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before).
"""Exact properties of the block decomposition and the readout operators."""
import unittest

import numpy as np

from ankyra_lite.anchor import blocks, readouts


def brute_force_envelope(pred, ctx, ct, tt):
    """Hour-wise envelope: max over days and hours of (daily mean + largest same-type excursion at that hour)."""
    days = ctx.reshape(56, 24)
    a = days - days.mean(1, keepdims=True)
    L = pred.reshape(31, 24).mean(1)
    best = -np.inf
    for d in range(31):
        ix = np.flatnonzero(ct == tt[d])[-4:]
        if len(ix) == 0:
            ix = np.flatnonzero(ct == 6)[-4:] if tt[d] == 7 and (ct == 6).any() else np.arange(56)
        best = max(best, (L[d] + a[ix].max(0)).max())
    return best


class BlockTests(unittest.TestCase):
    def test_block_losses_add_up_to_mse(self):
        rng = np.random.default_rng(7)
        F, y = rng.normal(10, 3, (50, 744)), rng.normal(10, 3, (50, 744))
        lv, dp, wd = blocks.block_losses(F, y)
        np.testing.assert_allclose(lv + dp + wd, ((F - y) ** 2).mean(1), rtol=0, atol=1e-10)

    def test_replacing_one_block_changes_only_that_blocks_loss(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(5, 2, 744), rng.normal(5, 2, 744)
        np.testing.assert_allclose(blocks.compose(blocks.level(x), blocks.daily_path(x), blocks.within_day(x)), x, atol=1e-12)
        x2 = blocks.compose(7.3, blocks.daily_path(x), blocks.within_day(x))
        before, after = blocks.block_losses(x, y), blocks.block_losses(x2, y)
        self.assertAlmostEqual(after[1], before[1], places=12)
        self.assertAlmostEqual(after[2], before[2], places=12)
        self.assertAlmostEqual(((x2 - y) ** 2).mean() - ((x - y) ** 2).mean(), after[0] - before[0], places=10)


class ReadoutTests(unittest.TestCase):
    def test_peak_scalar_form_equals_hourwise_envelope(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            ctx = rng.gamma(2.0, 5.0, 1344)
            ct = (np.arange(56) + rng.integers(0, 7)) % 7
            tt = (np.arange(31) + ct[-1] + 1) % 7
            tt[rng.integers(0, 31)] = 7
            pred = rng.gamma(2.0, 5.0, 744)
            u = readouts.peak_readout(pred, ctx[None], ct[None], tt[None])[0]
            self.assertAlmostEqual(u, brute_force_envelope(pred, ctx, ct, tt), places=9)

    def test_peak_readout_bounds(self):
        rng = np.random.default_rng(7)
        ctx = rng.gamma(2.0, 5.0, (10, 1344)); ct = np.tile(np.arange(56) % 7, (10, 1)); tt = np.tile(np.arange(31) % 7, (10, 1))
        pred = rng.gamma(2.0, 5.0, (10, 744))
        u = readouts.peak_readout(pred, ctx, ct, tt)
        L = pred.reshape(10, 31, 24).mean(2)
        self.assertTrue((u >= L.max(1) - 1e-12).all() and (L.max(1) >= pred.mean(1) - 1e-12).all())

    def test_excursion_is_the_largest_of_the_four_most_recent_days_of_the_type_and_a_holiday_borrows_sunday(self):
        rng = np.random.default_rng(7)
        ctx = rng.gamma(2.0, 5.0, 1344); ctx[5] += 1000.0                    # a large excursion on the oldest Monday
        days = ctx.reshape(56, 24); a = days.max(1) - days.mean(1)
        ct = np.arange(56) % 7                                              # eight days of each type, no holiday
        tt = np.array([0, 6, 7] + [1] * 28)
        A = readouts.historical_excursions(ctx[None], ct[None], tt[None])[0]
        self.assertEqual(A[0], a[ct == 0][-4:].max()); self.assertLess(A[0], a[0])   # the four most recent Mondays, not all eight
        self.assertEqual(A[2], A[1])                                        # no holiday in the context: Sunday's excursion
        ct[40] = 7                                                          # one holiday in the context: its own excursion
        self.assertEqual(readouts.historical_excursions(ctx[None], ct[None], tt[None])[0, 2], a[40])
        L = np.arange(31.0)[None]                                           # P_kappa = mean + kappa (U - mean)
        np.testing.assert_allclose(readouts.peak_from_daily_levels(L, A[None], 0.5), 15.0 + 0.5 * ((L[0] + A).max() - 15.0))

    def test_projection_never_increases_absolute_error(self):
        rng = np.random.default_rng(7)
        F, y = rng.normal(0.5, 2, 5000), np.abs(rng.normal(0, 2, 5000))
        P = np.maximum(F, 0)
        self.assertTrue((np.abs(P - y) <= np.abs(F - y) + 1e-15).all())
        self.assertAlmostEqual(readouts.energy_kwh(P) - readouts.energy_kwh(F), readouts.projection_energy_increase_kwh(F), places=9)

    def test_interval_bands_are_monotone_and_nonnegative(self):
        rng = np.random.default_rng(7)
        Q = np.sort(rng.normal(0, 1, (2, 24, 5)), axis=2)
        point = rng.gamma(2, 3, 744); hod = np.arange(744) % 24; work = (np.arange(744) // 24 % 7 < 5).astype(int)
        b = readouts.interval_bands(point, 2.0, Q, hod, work)
        self.assertTrue((b >= 0).all() and (np.diff(b, axis=0) >= 0).all())


if __name__ == "__main__":
    unittest.main()
