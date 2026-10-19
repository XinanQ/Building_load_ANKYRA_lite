# Ported from Building_load_ANKYRA/tests/test_gap_tolerance.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before).
"""Gap tolerance (2.2): short gaps interpolated and partially observed pseudo targets accepted, for the pseudo-origin bookkeeping only."""
import unittest

import numpy as np
import torch

from ankyra_lite import anchor
from ankyra_lite.anchor.core import CONTEXT, HORIZON, STEP, GAP_MAX_H, TARGET_MIN_OBS, fill_short_gaps, pseudo_origin_contexts
from ankyra_lite.anchor.history import History, InputError
from ankyra_lite.anchor.synthetic import synthetic_history

torch.set_num_threads(1)


def fm_for(load):
    """A foundation forecast at every pseudo-origin whose (raw or filled) context exists: the seasonal-naive week."""
    return {k: np.tile(ctx[-168:], 5)[:HORIZON] for k, ctx in pseudo_origin_contexts(load, gap_tolerance=True).items()}


def run(h, **kw):
    kw.setdefault("group", "Office"); kw.setdefault("temp_sigma_std", 0.25)
    kw.setdefault("foundation", fm_for(np.asarray(h.load_kw, float)))
    return anchor.forecast(h, **kw)


class FillTests(unittest.TestCase):
    def test_short_gap_inside_a_block_is_interpolated(self):
        x = np.arange(3000, dtype=float); x[100:104] = np.nan                        # 4 hours, far from any block boundary
        f = fill_short_gaps(x)
        self.assertTrue(np.allclose(f[100:104], np.arange(100, 104, dtype=float)))
        self.assertTrue(np.array_equal(np.isnan(f), np.zeros(3000, bool)))

    def test_long_gap_and_boundary_gap_stay_missing(self):
        o = 3000; x = np.arange(o, dtype=float)
        x[200:200 + GAP_MAX_H + 1] = np.nan                                           # one hour too long
        q = o - STEP; x[q - 2:q + 2] = np.nan                                         # straddles the pseudo-origin o - 744
        f = fill_short_gaps(x)
        self.assertTrue(np.isnan(f[200:200 + GAP_MAX_H + 1]).all()); self.assertTrue(np.isnan(f[q - 2:q + 2]).all())

    def test_no_value_after_a_pseudo_origin_enters_the_fill(self):
        o = 3000; x = np.arange(o, dtype=float); q = o - STEP
        x[q - 5:q - 2] = np.nan                                                       # gap inside the block ending at q
        f1 = fill_short_gaps(x); y = x.copy(); y[q:] = 999.0; f2 = fill_short_gaps(y)
        self.assertTrue(np.array_equal(f1[:q], f2[:q]))

    def test_complete_record_unchanged_and_untouched_at_the_origin(self):
        x = np.arange(3000, dtype=float); self.assertTrue(np.array_equal(fill_short_gaps(x), x))
        x[-3:] = np.nan; self.assertTrue(np.isnan(fill_short_gaps(x)[-3:]).all())    # a gap touching the origin has no right neighbour

    def test_pseudo_origin_contexts_origin_raw_pseudo_filled(self):
        h = synthetic_history(16000); load = np.array(h.load_kw, float); q = len(load) - 2 * STEP
        load[q - 700:q - 697] = np.nan
        c_old = pseudo_origin_contexts(load, gap_tolerance=False); c_new = pseudo_origin_contexts(load)
        self.assertNotIn(2, c_old); self.assertIn(2, c_new); self.assertTrue(np.array_equal(c_new[0], load[-CONTEXT:]))


class ForecastTests(unittest.TestCase):
    def test_complete_data_bit_identical_to_2_1(self):
        h = synthetic_history(16000)
        a = run(h); b = run(h, gap_tolerance=False)
        self.assertTrue(np.array_equal(a.trajectory_kw, b.trajectory_kw)); self.assertEqual(a.pseudo_pairs, b.pseudo_pairs)
        self.assertEqual(a.level_weights, b.level_weights)

    def test_gaps_in_pseudo_windows_recover_pseudo_pairs(self):
        h = synthetic_history(16000); load = np.array(h.load_kw, float); o = len(load)
        for k in (3, 4, 5):                                                           # a 3-hour gap in each pseudo context (outside the origin context) ...
            q = o - k * STEP; load[q - 500:q - 497] = np.nan
        q = o - 6 * STEP; load[q + 100:q + 160] = np.nan                                # ... and 60 missing target hours (92% observed) at k = 6
        hg = History(load, h.temperature_c, h.day_types, h.start_timestamp)
        a = run(hg); b = run(hg, gap_tolerance=False)
        self.assertGreater(a.pseudo_pairs, b.pseudo_pairs)
        self.assertTrue(np.isfinite(a.trajectory_kw).all()); self.assertGreaterEqual(a.pseudo_pairs, 4)

    def test_target_below_the_observed_share_is_not_used(self):
        h = synthetic_history(16000); load = np.array(h.load_kw, float); o = len(load)
        q = o - 4 * STEP; load[q + 100:q + 100 + int(HORIZON * (1 - TARGET_MIN_OBS)) + 24] = np.nan     # below 90 % observed
        hg = History(load, h.temperature_c, h.day_types, h.start_timestamp)
        a = run(hg); b = run(hg, gap_tolerance=False)
        self.assertEqual(a.pseudo_pairs, b.pseudo_pairs)

    def test_anchoring_record_is_reported_and_unused(self):
        h = synthetic_history(16000); a = run(h)
        rec = a.anchoring_record
        self.assertEqual(set(rec), {"carrier_level_weight", "history_vs_carrier_log_ratio", "pseudo_origins"})
        self.assertEqual(rec["carrier_level_weight"], a.level_weights["fm"]); self.assertTrue(np.isfinite(rec["history_vs_carrier_log_ratio"]))
        self.assertEqual(rec["pseudo_origins"], a.pseudo_pairs)
        off = run(History(np.r_[np.asarray(h.load_kw)[:-168], np.zeros(168)], h.temperature_c, h.day_types, h.start_timestamp), foundation={0: np.zeros(HORIZON)})
        self.assertTrue(off.off_state); self.assertEqual(off.anchoring_record, {})

    def test_origin_context_gap_still_refused(self):
        h = synthetic_history(16000); load = np.array(h.load_kw, float); load[-10:-7] = np.nan
        hg = History(load, h.temperature_c, h.day_types, h.start_timestamp)
        with self.assertRaises(InputError):
            run(hg, foundation={0: np.full(HORIZON, 1.0)})


if __name__ == "__main__":
    unittest.main()


class PseudoForecastValidationTests(unittest.TestCase):                           # 2.2.1
    def test_non_finite_pseudo_origin_forecast_is_refused(self):
        h = synthetic_history(16000); fm = fm_for(np.asarray(h.load_kw, float))
        bad = {k: v.copy() for k, v in fm.items()}; bad[4][100] = np.nan
        with self.assertRaises(ValueError):
            run(h, foundation=bad)
        bad = {k: v.copy() for k, v in fm.items()}; bad[2][:] = np.inf
        with self.assertRaises(ValueError):
            run(h, foundation=bad)


class NoFutureInformationUnderGapToleranceTests(unittest.TestCase):               # the 2.2 bookkeeping on a record with gaps
    def test_pseudo_origin_bookkeeping_reads_nothing_at_or_after_each_pseudo_origin(self):
        """A record with short gaps in pseudo contexts and partially observed pseudo targets: changing every value at or after
        a pseudo-origin o_k leaves the level candidates' errors at all later pseudo-origins (j > k) unchanged, and the handover
        and within-day evidence of the forecast made at o_k unchanged."""
        import ankyra_lite.anchor.core as C
        h = synthetic_history(20000); load = np.array(h.load_kw, float); temp = np.array(h.temperature_c, float); ty = np.asarray(h.day_types); o = len(load)
        for k in (3, 4, 5):                                                            # 3-hour gaps in three pseudo contexts
            q = o - k * STEP; load[q - 600:q - 597] = np.nan
        q = o - 6 * STEP; load[q + 50:q + 110] = np.nan                                 # a 92%-observed pseudo target
        rng = np.random.default_rng(7)
        for k in range(2, 7):
            cut = o - k * STEP
            load2 = load.copy(); load2[cut:] = np.where(np.isfinite(load[cut:]), load[cut:] * rng.uniform(0.5, 1.5, o - cut), np.nan)
            temp2 = temp.copy(); temp2[cut:] += rng.normal(0, 5, o - cut)
            # (a) the level candidates' pseudo-origin errors at j > k, with the 2.2 settings
            fill1 = C.fill_short_gaps(load); fill2 = C.fill_short_gaps(load2)
            np.testing.assert_array_equal(fill1[:cut], fill2[:cut])                      # the fill never reaches across a pseudo-origin
            h1 = History(fill1, temp, ty, h.start_timestamp, observed=np.isfinite(fill1)); h2 = History(fill2, temp2, ty, h.start_timestamp, observed=np.isfinite(fill2))
            e1 = C._level_parts(h1, "Office", 0.25, min_target_obs=C.TARGET_MIN_OBS)["err"]; e2 = C._level_parts(h2, "Office", 0.25, min_target_obs=C.TARGET_MIN_OBS)["err"]
            for name in e1:
                np.testing.assert_array_equal(e1[name][0, k:], e2[name][0, k:], err_msg=f"{name}, change at and after o_{k}")
            # (b) the forecast issued at o_k itself (handover pairs, trust, level) from the two records truncated at o_k
            hk1 = History(load[:cut], temp[:cut], ty[:cut + HORIZON], h.start_timestamp); hk2 = History(load2[:cut], temp2[:cut], ty[:cut + HORIZON], h.start_timestamp)
            np.testing.assert_array_equal(np.asarray(hk1.load_kw), np.asarray(hk2.load_kw))  # truncation at o_k removes every changed value
        f1 = run(History(load, temp, ty, h.start_timestamp)); self.assertTrue(np.isfinite(f1.trajectory_kw).all()); self.assertGreaterEqual(f1.pseudo_pairs, 5)

