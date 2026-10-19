# Ported from Building_load_ANKYRA/tests/test_history_reference.py at commit 487e0d2 (ANKYRA 2.2.1); the only change is the package name:
# imports point to the copied anchoring layer ankyra_lite.anchor, which the tests call as ``anchor`` (``ankyra`` before).
"""The frozen reference estimator (ankyra_lite.anchor.history) on the artificial building: a pinned level, its input checks and two
known properties (a temperature record without variation is rejected; Commercial has no prior curve)."""
import unittest

import numpy as np
import torch

from ankyra_lite import anchor
from ankyra_lite.anchor.history import estimate_from_history, frozen_config, InputError, SignatureDegeneracy
from ankyra_lite.anchor.synthetic import synthetic_history, seasonal_naive

torch.set_num_threads(1)


class HistoryReferenceTests(unittest.TestCase):
    def test_documented_synthetic_level(self):
        e = estimate_from_history(synthetic_history(16000), group="Office", temp_sigma_std=0.25)
        # 1e-6 kW: the last digits of the level differ across platforms' numerical libraries (Linux CI gives 22.5983831666)
        self.assertAlmostEqual(e.level_kw, 22.5983831772, places=6)
        self.assertLess(abs(e.daily_path_kw.sum()), 1e-9)
        self.assertAlmostEqual(sum(e.diagnostics["level_weights"].values()), 1.0, places=12)

    def test_annual_candidate_needs_two_error_records(self):
        before = estimate_from_history(synthetic_history(10247), group="Office", temp_sigma_std=0.25)
        at = estimate_from_history(synthetic_history(10248), group="Office", temp_sigma_std=0.25)
        self.assertEqual(before.diagnostics["level_error_counts"]["a_u"], 1)
        self.assertEqual(at.diagnostics["level_error_counts"]["a_u"], 2)

    def test_missing_context_is_rejected(self):
        h = synthetic_history(6000)
        h.load_kw[-10] = np.nan
        with self.assertRaises(InputError):
            estimate_from_history(h, group="Office", temp_sigma_std=0.25)

    def test_a_temperature_record_without_variation_is_rejected(self):
        h = synthetic_history(16000, constant_temperature=True)
        with self.assertRaisesRegex(SignatureDegeneracy, "does not vary"):
            estimate_from_history(h, group="Office", temp_sigma_std=0.25)
        with self.assertRaises(anchor.SignatureDegeneracy):                            # the forecast fails the same way
            anchor.forecast(h, group="Office", temp_sigma_std=0.25, foundation=seasonal_naive)
        e = estimate_from_history(h, group="Office", temp_sigma_std=0.25, boundary_policy="source_prior")   # the extension that was not evaluated
        self.assertTrue(e.diagnostics["boundary_extension_triggered"] and np.isfinite(e.level_kw))

    def test_commercial_has_no_prior_curve_and_a_zero_signature(self):
        """A known property of the frozen configuration; it is the evaluated behaviour and is kept."""
        curves = frozen_config()["source_curves"]
        self.assertNotIn("Commercial", curves); self.assertIn("group_4", curves)       # the fifth curve is stored but not read
        e = estimate_from_history(synthetic_history(16000), group="Commercial", temp_sigma_std=0.25)
        self.assertEqual({s["status"] for s in e.diagnostics["signature_fits"]}, {"unknown_group_zero_signature"})
        w = e.diagnostics["level_weights"]
        for c in "sla":                                                                # adjusted and unadjusted candidates coincide
            self.assertEqual(w[c + "_w"], w[c + "_u"])
        self.assertAlmostEqual(e.level_kw, 22.6236200637, places=8)


if __name__ == "__main__":
    unittest.main()
