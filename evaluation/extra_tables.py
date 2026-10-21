"""The result files of ankyra-lite beyond the ten tables of the main comparison (called by compare_baselines.py).

Each file mirrors the file of the same name in the first release repository (Building_load_ANKYRA/results) where a
counterpart exists, with ankyra-lite in ANKYRA's place; ``results/README.md`` describes the columns.

  datasets.csv                      populations, their tiers and counts, the training / validation / test split, the returned span
  example_window_cambridge.csv/json one Cambridge test window: load, ankyra-lite, the carrier alone, TimesFM
  cost_per_window.csv               parameters, device, CPU seconds per window (one thread)            (cost_timing.py)
  robustness.csv                    ankyra-lite and the carrier alone on corrupted contexts (CPU)      (robustness.py)
  intervals_by_population.csv       the release interval centred on ankyra-lite, and the native quantiles of TimesFM
  intervals_winkler_contrasts.csv   and Chronos-2; unit-equal Winkler contrasts
  intervals_households.json         the same on GoiEner households, with the interval around the carrier alone
  ablation.csv                      ankyra-lite against reduced and alternative versions of itself (incl. the recipe arms)
  component_contributions.csv       what each part adds (direct contrasts of ablation.csv)
  seed_spread.csv                   each seed alone against the three-seed ensemble, and the spread of the headline contrasts
  carriers_in_anchoring_layer.csv   ankyra-lite against other trained carriers inside the same anchoring layer
  contrast_ii_blocks.csv            ankyra-lite against the history-only configuration and the carrier alone, by error block
  contrast_ii_control.csv           ankyra-lite against the carrier-free control (the same procedure without a carrier), by block
  history_length.csv                ankyra-lite against the carrier alone, TimesFM and Chronos-2-X by history length at the origin
  bdg2_near_zero_sensitivity.csv    the BDG2 contrasts with and without the three near-zero meters
  anchoring_gain_deciles.csv        the anchoring record against the realised gain over the carrier alone
  training_curves.csv               per population and seed: both training stages (from scratch with early stopping, then
                                    the span return), validation criterion and training loss by epoch
  recipe_history.csv, constants.csv the recipe arms of the study and every constant                   (recipe_tables.py)
  continuation_layer.csv            version 0.2.0: the origin-continuity layer on and off on the same carrier, by lead time
  continuation_record.csv           version 0.2.0: what the layer did (bypasses, clipped spans, reference days, |Delta|) and
                                    the gap between the delivered trajectory's mean and the energy readout (readout rule A)

Files added on 19 October 2026 (the first release's analyses of the same name or purpose, with ankyra-lite's numbers; the
ones marked "fields" read the ``fields`` mode of lite_reanchor.py, the anchoring layer's own fields of every window):

  training_budget_sensitivity.csv   ankyra-lite against the four trained load-only baselines under the original and a
                                    relaxed training budget                                          (budget_tables.py)
  handover_granularity.csv          the handover arms: none, a fixed half weight, one weight per window, the weekly weights  (fields)
  lead_weeks.csv                    ankyra-lite and the fixed division against the carrier alone and TimesFM by forecast week,
                                    and the mean handover weight on the carrier                                      (fields)
  within_day_trust.csv              where the within-day anchoring acts: mean trust, windows with trust > 0, analog shape kept (fields)
  generic_combinations.csv          one whole-window combination weight of the carrier with the history side against the same
                                    with three generic forecasters, and equal-weight averages                         (fields)
  level_candidate_hit_rate.csv      how often the level candidate with the smallest pseudo-origin error is the best one at
                                    the origin, against chance (P20)                                                 (fields)
  long_context_baselines.csv        ankyra-lite against TimesFM and Chronos-2 with a year of context     (long_context_tables.py)
  energy_level_readout.csv          the energy contrasts with ankyra-lite's level readout in place of its trajectory mean
  rows added to ablation.csv, component_contributions.csv (fixed division, fixed half handover weight, one whole-window
  combination weight, half-and-half average; fields) and peak_readout.csv (the same readout on the carrier alone)

Interval estimand and contrasts are the release's: 2,000-draw unit x target-month bootstrap of the unit-equal log
ratio (tools_hcr_ext_drammen_score_v1), a fresh random stream per contrast.

Version 0.2.0 (a products mapping with a configuration and, possibly, the continuation layer).  ankyra-lite is the
delivered forecast: hourly errors read the trajectory after the layer, energy errors the trajectory before it (readout
rule A; the level and peak readouts are those of the product).  Re-anchored versions of the anchoring layer (the
switched-off parts, the handover arms, the whole-window weights) get the same layer, so that each row changes one part;
the carriers inside the anchoring layer and contrast (ii) compare the trajectories before the layer (the layer's
constants were fixed on this carrier's products), with one added row each for the layer.  The seed spread reads the
carriers of every seed and, anchored, the two stored three-seed groups.  With a 0.1.x mapping (no configuration, no
layer) every table is the 0.1.x table.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import compare_baselines as CB                                     # noqa: E402

import csv                                                         # noqa: E402
import io                                                          # noqa: E402
import json                                                        # noqa: E402
import math                                                        # noqa: E402
import subprocess                                                  # noqa: E402
import warnings                                                    # noqa: E402
from collections import defaultdict                                # noqa: E402
from datetime import datetime, timedelta                           # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

HORIZON, CONTEXT, HR, DAYS = CB.HORIZON, CB.CONTEXT, CB.HR, CB.DAYS
WEEK = np.repeat(np.array([0] * 7 + [1] * 7 + [2] * 7 + [3] * 10), 24)                 # forecast week of each hour (release)
DEV6 = ("cambridge", "heew", "eweld", "oslo", "drammen", "cineldi")                     # populations with development windows
STRATA_M = [("< 10,248 h (no annual candidate)", 0, 10248), ("10,248 h - 2 years", 10248, 17520), ("2-3 years", 17520, 26280), (">= 3 years", 26280, 10 ** 9)]
STRATA_K = [("K_eff 6-8", 6, 8), ("K_eff 9-11", 9, 11), ("K_eff 12", 12, 12)]           # tools_history_length_v1 (release)
INTERVAL_SEED, HISTORY_SEED = 20261003, 20261003                                       # the release's seeds of these tables
EXAMPLE_UNIT, EXAMPLE_ORIGIN_DATE = "UCam_Building_b37", "2011-09-01"                   # the release's example window
VARIANT_FILES = {                                                                        # saved anchored variants of the earlier P3s line
    "carrier without the copy pointer, from-scratch recipe (anchored)": "smallnet_formal3_v1/{face}/arms_D3s.npz",
    "earlier continuation study: unweighted loss (anchored)": "smallnet_pwformal_v1/B/{face}/arms_P3sS_ens.npz",
    "earlier continuation study: weighted loss (anchored)": "smallnet_pwformal_v1/B/{face}/arms_P3sW_ens.npz",
    "earlier continuation study: weighted loss and warm-up (anchored)": "smallnet_pwformal_v1/B/{face}/arms_P3sWU_ens.npz",
}
RECIPE_VARIANTS = {                                                                      # products mapping 'recipe_variants' (same windows and store)
    "P3s": "recipe: from-scratch stage alone, no continuation (anchored)",
    "F": "recipe: 8 continuation epochs, validation set held out, no span return (anchored)",
    "V": "recipe: continuation with a proportional validation set and early stopping (anchored)",
}
S3S_LABEL = "earlier carrier: structured network without the copy pointer, weekly training windows (anchored)"
MODE_ABLATION = {"no_within": "anchoring layer without the within-day anchoring",
                 "lead_trust": "within-day trust per lead block instead of one value per window",
                 "no_gap": "anchoring layer without the short-gap rule",
                 "no_micro": "anchoring layer without the micro-load rule"}
COMPONENT = {"carrier alone (no anchoring layer)": "the whole anchoring layer (carrier alone -> ankyra-lite)",
             "state rule off (the anchoring layer's own return on state windows)": "seasonal-naive state rule",
             "anchoring layer without the within-day anchoring": "within-day anchoring",
             "within-day trust per lead block instead of one value per window": "one within-day trust value per window",
             "anchoring layer without the short-gap rule": "short-gap rule",
             "anchoring layer without the micro-load rule": "micro-load rule",
             "history-only configuration (anchoring layer without a carrier)": "the carrier inside the anchoring layer (history only -> ankyra-lite)",
             RECIPE_VARIANTS["F"]: "span return (the validation span returned to the training set; both with 8 continuation epochs)",
             RECIPE_VARIANTS["P3s"]: "continuation with the span return (from-scratch stage -> released recipe)"}
DEEP_CARRIERS = ("LSTM", "PatchTST", "LSTM-T-KAN-M")
LITE_CARRIER = "small carrier network (ankyra-lite)"
BLOCKS = ("hourly", "level", "daily path", "within-day")
BLOCK_SEED = {"hourly": 9, "level": 11, "daily path": 13, "within-day": 15}           # tools_contrast_ii_blocks_v1 (release)
STATE_EXCLUDED = "off-state and micro-load windows excluded"
DRAMMEN_OLD_POOL = "Drammen: trained with the original CINELDI store in its pooled training set"
# ---- added on 19 October 2026 (tables that read the fields mode of lite_reanchor.py)
FIXED_DIVISION = "fixed division (history level and daily path, no handover)"
HALF_HANDOVER = "fixed half handover weight"
WHOLE_WINDOW = "one whole-window combination weight of the history side and the carrier"
HALF_AVERAGE = "half-and-half average of the history side and the carrier"
COMPONENT.update({WHOLE_WINDOW: "block-wise weights instead of one whole-window weight",
                  HALF_AVERAGE: "block-wise weights against a half-and-half average"})
ADDED_ABLATIONS = (FIXED_DIVISION, HALF_HANDOVER, WHOLE_WINDOW, HALF_AVERAGE)
# ---- version 0.2.0 (the continuation layer, six seeds, recipe Rw)
NO_LAYER = "without the continuation layer (the same carrier and anchoring layer)"
COMPONENT[NO_LAYER] = "origin-continuity layer"
LAYER_ROW = " with the continuation layer"                           # suffix of the added layer rows (carriers, contrast (ii))
CONT_SEGMENTS = (("lead hour 0", 0, 1), ("lead hours 0-3", 0, 3), ("forecast day 1", 0, 24), ("forecast days 2-7", 24, 168),
                 ("forecast days 8-31", 168, 744), ("whole window", 0, 744))           # "lead hours 0-3" = hours 0, 1, 2 (the study's definition)
CONT_LAYER_FIELDS = ["set", "tier", "segment", "hours", "error", "windows", "units", "state_windows", "log_ratio", "um_low", "um_high",
                     "improvement_pct", "resolved", "flag"]
CONT_RECORD_FIELDS = ["set", "tier", "windows", "state_windows_bypassed", "missing_fast_span", "missing_day_span", "nonpositive_fast_span",
                      "nonpositive_day_span", "clipped_fast_share", "clipped_day_share", "no_match_reference_day", "x_fast_median", "x_day_median",
                      "x_fast_mean_abs", "x_day_mean_abs", "max_abs_delta", "delivered_mean_vs_energy_readout_rel_p50",
                      "delivered_mean_vs_energy_readout_rel_p99", "delivered_mean_vs_energy_readout_rel_max", "layer_energy_change_pct_min",
                      "layer_energy_change_pct_max", "flag"]
RECIPE_WORDS = {"R": "recipe R (span return)", "Rw": "recipe Rw (span return with inverse-noise loss weighting)"}
FACTOR_COMPONENT = {"recipe": "inverse-noise loss weighting (recipe Rw instead of recipe R)", "seeds": "six seeds instead of three",
                    "continuation": "origin-continuity layer"}
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def config_of(key):
    """(recipe, seeds, layer) of a 0.2.0 recipe-variant key (R_g0, R_ens6_c, Rw_g0, ...), or None for another key."""
    import re
    m = re.fullmatch(r"(R|Rw)_(g0|ens6)(_c)?", key)
    if not m:
        return None
    return m.group(1), 6 if m.group(2) == "ens6" else 3, bool(m.group(3))


def config_label(key):
    """The ablation label of a 0.2.0 recipe variant."""
    r, n, c = config_of(key)
    if key == "R_g0":
        return "0.1.0 configuration: recipe R (span return), seeds 0-2, no continuation layer (anchored)"
    return (f"configuration: {RECIPE_WORDS[r]}, {'six seeds' if n == 6 else 'seeds 0-2'}, "
            f"{'with' if c else 'without'} the continuation layer (anchored)")


def config_component(key, parts):
    """The component a 0.2.0 recipe variant isolates against the release configuration ``parts``, or None: the variant
    differs from the release in one factor that the release has (recipe Rw, six seeds, the layer), or it is the 0.1.0
    configuration (every change together)."""
    r, n, c = config_of(key)
    if not parts:
        return None
    rel = (parts.get("recipe"), int(parts.get("seeds", 3)), bool(parts.get("continuation")))
    if key == "R_g0" and rel != ("R", 3, False):
        return "0.2.0 against 0.1.0 (every change of the release configuration together)"
    diff = [f for f, a, b in (("recipe", r, rel[0]), ("seeds", n, rel[1]), ("continuation", c, rel[2])) if a != b]
    if len(diff) != 1:
        return None
    f = diff[0]
    has = {"recipe": rel[0] == "Rw", "seeds": rel[1] == 6, "continuation": rel[2]}[f]
    return FACTOR_COMPONENT[f] if has else None


def n_words(n):
    return COUNT_WORDS.get(int(n), str(n))
ARMS = ("A0", "Ah", "AM", "AW")                     # handover arms (the first release's names; Ah = a fixed weight of one half)
ARM_WORDS = {"A0": "no handover (history-side daily means only)", "Ah": "fixed half handover weight",
             "AM": "one handover weight per unit and window", "AW": "weekly handover weights (ankyra-lite)"}
HANDOVER_CONTRASTS = (("C1 AW vs Ah", "AW", "Ah"), ("C2 AW vs AM", "AW", "AM"), ("C3 AM vs Ah", "AM", "Ah"), ("C4 Ah vs A0", "Ah", "A0"),
                      ("AW vs A0", "AW", "A0"), ("AM vs A0", "AM", "A0"))
WEEKS4 = ((0, 7), (7, 14), (14, 21), (21, 31))
CARRIER_PEAK = "same readout on the carrier alone"
GENERIC_X = ("SN-week", "WeekProf4", "LYR-profile")
EW_X = ("SN-week", "WeekProf4", "LYR-profile", "RIDGE-L", "MSTL")
HIT_SEED = 20261019                                  # unit bootstrap of the level-candidate hit rate
ADDED_FILES = {
    "handover_granularity.csv": ("handover_granularity", ["set", "tier", "contrast", "log_ratio", "um_low", "um_high", "improvement_pct", "pooled_mse_ratio", "flag"],
                                 "the handover arms (none, a fixed half weight, one weight per window, the weekly weights) against each other, the carrier alone, TimesFM and the fixed division"),
    "lead_weeks.csv": ("lead_weeks", ["set", "tier", "week", "days", "windows", "units",
                                      "ankyra_lite_vs_carrier_log_ratio", "ankyra_lite_vs_carrier_um_low", "ankyra_lite_vs_carrier_um_high",
                                      "ankyra_lite_vs_timesfm_log_ratio", "ankyra_lite_vs_timesfm_um_low", "ankyra_lite_vs_timesfm_um_high",
                                      "fixed_division_vs_carrier_log_ratio", "fixed_division_vs_carrier_um_low", "fixed_division_vs_carrier_um_high",
                                      "fixed_division_vs_timesfm_log_ratio", "fixed_division_vs_timesfm_um_low", "fixed_division_vs_timesfm_um_high",
                                      "mean_weight_on_carrier", "flag"],
                       "ankyra-lite and the fixed division against the carrier alone and TimesFM by forecast week, and the mean handover weight on the carrier"),
    "within_day_trust.csv": ("within_day_trust", ["set", "tier", "windows", "state_windows", "anchored_windows", "mean_trust", "median_trust", "share_trust_positive",
                                                  "share_analog_kept", "mean_triples", "share_three_triples", "flag"],
                             "where the within-day anchoring acts: trust on the analog-day shape, windows with a positive trust, analog shape kept"),
    "generic_combinations.csv": ("generic_combinations", ["set", "tier", "subset", "windows", "units", "contrast", "best", "candidates", "error", "log_ratio", "um_low",
                                                          "um_high", "improvement_pct", "resolved", "availability_note", "flag"],
                                 "one whole-window combination weight of the carrier with the history side against the same with generic forecasters, and equal-weight averages"),
    "level_candidate_hit_rate.csv": ("level_candidate_hit_rate", ["set", "tier", "candidates", "windows", "units", "windows_tied_excluded", "hit_rate", "chance_rate",
                                                                  "hit_minus_chance", "um_low", "um_high", "share_carrier_selected", "share_carrier_best", "flag"],
                                     "how often the level candidate with the smallest pseudo-origin error is the best one at the origin, against chance"),
    "energy_level_readout.csv": ("energy_level_readout", ["set", "tier", "windows", "units", "model", "model_label", "log_ratio", "um_low", "um_high", "improvement_pct",
                                                          "resolved", "median_energy_ape_pct", "windows_projection_binds", "flag"],
                                 "the energy contrasts with ankyra-lite's level readout (744 x level) in place of its trajectory's mean before the continuation layer"),
}

def _resolved(lo, hi, identical=False):
    if identical:
        return "identical forecasts"
    return "first better" if hi < 0 else ("second better" if lo > 0 else "not resolved")


class Collector:
    """Rows of the extra files, one population at a time (the data of the main comparison are reused)."""

    def __init__(self, T, prods):
        self.T, self.prods = T, prods
        self.rows = defaultdict(list)
        self.rec = {}
        self.hist_pool = []
        self.dec_pool = []
        self.hit_pool = []
        self.extra_json = {}
        self.faces = []
        self.notes = defaultdict(list)
        self.cont_pool = []                                          # per-population losses of continuation_layer.csv (pooled rows)
        self.parts = prods.get("configuration_parts") or {}          # version 0.2.0: recipe, seeds, continuation of the release
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tools_hcr_intervals_v1 as IV
        self.IV = IV

    # ------------------------------------------------------------------ helpers
    def ctr(self, a, b, unit, month, off):
        """Unit-equal log ratio of mean losses a over b (negative favours a), UM interval (fresh stream SEED + off)."""
        T = self.T
        if np.array_equal(a, b):
            return {"log_ratio": 0.0, "um_low": 0.0, "um_high": 0.0, "improvement_pct": 0.0, "resolved": "identical forecasts"}
        with np.errstate(all="ignore"):
            c = CB.contrast_um(T, a, b, unit, month, np.random.default_rng(T.DS.SEED + off))
        lr, (lo, hi) = c["log_ratio"], c["UM"]
        if not np.isfinite(lr):
            return {"log_ratio": None, "um_low": None, "um_high": None, "improvement_pct": None, "resolved": "undefined"}
        return {"log_ratio": float(lr), "um_low": float(lo), "um_high": float(hi), "improvement_pct": 100 * (1 - math.exp(lr)), "resolved": _resolved(lo, hi)}

    @staticmethod
    def losses(y, p, pe=None):
        """Hourly and monthly-energy losses; ``pe``: the trajectory the energy is read from when it is not ``p`` (ankyra-lite
        with the continuation layer: the trajectory before the layer, readout rule A)."""
        return {"hourly": ((p - y) ** 2).mean(1), "energy": ((p if pe is None else pe).mean(1) - y.mean(1)) ** 2}

    @staticmethod
    def layered(prod, raw, state):
        """The release continuation layer on a variant's trajectory before the layer (float32-stored values, as the study's
        continued products), with the product's pre-origin inputs and the variant's own state flags; ``raw`` unchanged
        when the product has no layer."""
        c = prod.get("cont")
        if c is None:
            return raw
        Fp, _ = CB.apply_layer(np.asarray(raw, np.float64).astype(np.float32), c["pre"], c["last_type"], c["day_types"], np.asarray(state, bool))
        return Fp.astype(np.float64)

    def deliver(self, prod, a, key="traj"):
        """A re-anchored or saved variant as delivered: with the layer when the product has one (the cache's layered
        trajectory ``traj_cont`` where it exists, else the release layer applied here), then the state rule.  Returns
        (hourly forecast, forecast the energy is read from)."""
        pre = self.state_rule(prod, a[key].astype(np.float64), a["off"], a["micro"])
        if prod.get("cont") is None:
            return pre, None
        tc = a["traj_cont"].astype(np.float64) if "traj_cont" in a else self.layered(prod, a[key], a["off"] | a["micro"])
        return self.state_rule(prod, tc, a["off"], a["micro"]), pre

    @staticmethod
    def rnd(c, nd=6):
        out = dict(c)
        for k in ("log_ratio", "um_low", "um_high"):
            out[k] = None if out[k] is None else round(out[k], nd) + 0.0          # + 0.0: no negative zero
        out["improvement_pct"] = None if out["improvement_pct"] is None else round(out["improvement_pct"], 3) + 0.0
        return out

    def state_rule(self, prod, traj, off, micro):
        """The delivered forecast of a variant: the weekly seasonal naive where the variant's anchoring call reported an
        off-state or micro-load window (the product's seasonal naive covers every state window of the product)."""
        st = off | micro
        if (st & ~prod["state"]).any():
            raise SystemExit("a variant has a state window the product does not have")
        return np.where(st[:, None], prod["SN"], traj)

    def align(self, z, P, pad=0, key="pred"):
        k = {(int(b), int(o) - pad): i for i, (b, o) in enumerate(zip(z["building"], z["origin"]))}
        idx = [k.get((int(b), int(o))) for b, o in zip(P["wbL"], P["woL"])]
        if any(i is None for i in idx):
            return None
        return np.asarray(z[key])[idx]

    # ------------------------------------------------------------------ one population
    def face(self, face, entry, P, prod, baselines, flag, hourly_ref=None):
        rec = self._face_main(face, entry, P, prod, baselines, flag)
        rec.update(self.added(face, entry, P, prod, baselines, flag, hourly_ref=hourly_ref))
        return rec

    def _face_main(self, face, entry, P, prod, baselines, flag):
        self.faces.append(face)
        rec = {"flag": flag or None}
        seeds, ens = CB.load_carriers(entry, prod["origins_product"], P["wbL"])
        if prod["C0"] is not None:
            rec["carrier_ensemble_vs_stored_C0_max_abs"] = float(np.abs(ens.astype(np.float32) - prod["C0"].astype(np.float32)).max())
        cache = self._cache(face, entry, P, prod)
        rec["reanchor"] = {m: v[1] for m, v in cache.items()}
        seedp, rec["seed_products"] = self._seed_products(face, entry, P, prod, cache)
        rec["ablation"] = self._ablation(face, entry, P, prod, ens, cache, seedp, flag)
        rec["study_scorer_crosscheck"] = self._scorer_check(face, rec["ablation"])
        rec["seed_spread"] = self._seeds(face, P, prod, seeds, ens, baselines, seedp, flag)
        rec["intervals"] = self._intervals(face, entry, P, prod, ens, flag)
        rec["carriers"] = self._carriers(face, entry, P, prod, ens, flag)
        rec["contrast_ii"] = self._contrast_ii(face, entry, P, prod, ens, flag, rec["carriers"])
        rec["history_length"] = self._history(face, P, prod, ens, baselines)
        rec["datasets"] = self._datasets(face, entry, P, prod)
        rec["training_curves"] = self._curves(face, entry, P)
        if face == "bdg2":
            rec["near_zero"] = self._near_zero(P, prod, baselines)
        if face == "cambridge":
            rec["example_window"] = self._example(P, prod, ens, baselines)
        if prod.get("cont") is not None:
            rec["continuation"] = self._continuation(face, P, prod, flag)
        if "ens" in cache and cache["ens"][0] is not None:
            self._deciles_collect(face, P, prod, ens, cache["ens"][0])
        self.rec[face] = rec
        return rec

    def _cache(self, face, entry, P, prod):
        """The re-anchored modes of lite_reanchor.py that belong to this product (full runs only); the single seeds are
        not read from here when the products mapping lists stored single-seed products."""
        import lite_reanchor as LR
        out = {}
        stored = bool(entry.get("seed_paths") or entry.get("group_paths"))
        for mode in LR.modes_for(entry):
            if (LR.is_seed_mode(mode) and stored) or LR.is_group_mode(mode) or mode in LR.FIELD_MODES:    # fields: read by added()
                continue
            z, m = LR.load_mode(LR.cache_key(self.prods), face, mode)
            if z is None:
                continue
            if m.get("product_sha256") != LR._sha(entry["npz_path"]) or (m.get("limit") is not None):
                self.notes[face].append(f"re-anchoring cache {mode} is stale or partial; not used")
                continue
            if m.get("error_count"):
                self.notes[face].append(f"re-anchoring {mode}: {m['error_count']} windows failed in the anchoring layer; mode not used")
                continue
            arr = {}
            keys = ("traj", "level", "peak", "off", "micro", "stats") + (("traj_cont",) if prod.get("cont") is not None else ())
            if prod.get("cont") is not None and "traj_cont" not in z:
                self.notes[face].append(f"re-anchoring cache {mode} has no layered trajectory; mode not used")
                continue
            for key in keys:
                a = self.align(z, P, pad=prod["pad"], key=key)
                if a is None:
                    break
                arr[key] = a
            else:
                out[mode] = (arr, {k: m.get(k) for k in ("label", "windows", "error_count", "gate", "identical_to", "wall_seconds")})
        if "ens" in out and not (out["ens"][1].get("gate") or {}).get("pass", False):
            self.notes[face].append("re-anchoring gate failed: the re-run ensemble does not reproduce the product; cache not used")
            return {}
        return out

    def _seed_products(self, face, entry, P, prod, cache):
        """{seed: arrays} of the stored single-seed anchored products (products mapping 'seed_npz'), aligned to the panel
        order, else of the re-anchored single seeds; and the record of the single-seed gate (lite_reanchor, first windows
        of seed 0 re-anchored and compared bit for bit with the stored product)."""
        import lite_reanchor as LR
        out, rec = {}, {"source": None}
        if entry.get("seed_paths"):
            rec["source"] = "stored single-seed products"
            rec["files"] = [CB.show(p) for p in entry["seed_paths"]]
            for s, f in enumerate(entry["seed_paths"]):
                z = np.load(f, allow_pickle=False)
                arr = {k: self.align(z, P, pad=prod["pad"], key=k) for k in ("traj", "level", "peak", "off", "micro")}
                if any(v is None for v in arr.values()):
                    self.notes[face].append(f"stored single-seed product of seed {s} does not cover the test windows; not used")
                    return {}, rec
                out[s] = arr
            gates = {}
            cdir = LR.cache_file(LR.cache_key(self.prods), face, "s0").parent
            for f in (sorted(cdir.glob("s*_first*.json")) if cdir.exists() else []):
                m = json.loads(f.read_text(encoding="utf-8"))
                if m.get("product_sha256") == LR._sha(entry["npz_path"]) and m.get("gate"):
                    gates[f.stem] = m["gate"]
            rec["gate"] = gates or None
            if not gates:
                self.notes[face].append("no single-seed gate in the re-anchoring cache (run with --reanchor)")
            elif not all(g.get("pass") for g in gates.values()):
                self.notes[face].append("single-seed gate failed: re-anchoring does not reproduce the stored single-seed product")
        elif entry.get("group_paths"):                                    # six seeds: the two stored three-seed groups
            layer = prod.get("cont") is not None
            rec["source"] = "stored three-seed group products" + (" (with the continuation layer)" if layer else "")
            paths = entry["group_continued_paths"] if layer else entry["group_paths"]
            rec["files"] = [CB.show(p) for p in paths]
            for j, (f, g) in enumerate(zip(paths, LR.groups_of(entry))):
                z = np.load(f, allow_pickle=False)
                zp = np.load(entry["group_paths"][j], allow_pickle=False)
                arr = {k: self.align(z, P, pad=prod["pad"], key=k) for k in ("traj", "level", "peak", "off", "micro")}
                arr["traj_pre"] = self.align(zp, P, pad=prod["pad"], key="traj")
                if any(v is None for v in arr.values()):
                    self.notes[face].append(f"stored group product {j} does not cover the test windows; not used")
                    return {}, rec
                arr["seeds"] = g
                out[f"g{j}"] = arr
            gates = {}
            cdir = LR.cache_file(LR.cache_key(self.prods), face, "g0").parent
            for f in (sorted(cdir.glob("g*_first*.json")) if cdir.exists() else []):
                m = json.loads(f.read_text(encoding="utf-8"))
                if m.get("product_sha256") == LR._sha(entry["npz_path"]) and m.get("gate"):
                    gates[f.stem] = m["gate"]
            rec["gate"] = gates or None
            if not gates:
                self.notes[face].append("no seed-group gate in the re-anchoring cache (run lite_reanchor.py)")
            elif not all(g.get("pass") for g in gates.values()):
                self.notes[face].append("seed-group gate failed: re-anchoring does not reproduce a stored group product")
        elif all(f"s{s}" in cache for s in LR.seeds_of(entry)):
            rec["source"] = "re-anchored single seeds"
            out = {s: cache[f"s{s}"][0] for s in LR.seeds_of(entry)}
        return out, rec

    def _ablation(self, face, entry, P, prod, ens, cache, seedp, flag):
        y, unit, month = P["yL"], P["uL"], P["mL"]
        lite = prod["traj"]
        layer = prod.get("cont") is not None
        L0 = self.losses(y, lite, prod["traj_pre"] if layer else None)
        nseeds = len(entry.get("seeds") or (0, 1, 2))
        variants = {"carrier alone (no anchoring layer)": (ens, f"the {n_words(nseeds)}-seed carrier forecast at the origin, without the anchoring layer and "
                                                                "without the state rule"),
                    "state rule off (the anchoring layer's own return on state windows)": (prod["raw"], f"{int(prod['state'].sum())} state windows")}
        if layer:
            variants[NO_LAYER] = (prod["traj_pre"], "the product before the origin-continuity layer (seasonal naive on state windows)")
        for s, a in seedp.items():
            if isinstance(s, str):                                         # a three-seed group of a six-seed configuration
                g = a["seeds"]
                pre = self.state_rule(prod, a["traj_pre"].astype(np.float64), a["off"], a["micro"]) if layer else None
                hourly = self.state_rule(prod, a["traj"].astype(np.float64), a["off"], a["micro"])
                variants[f"three-seed group {s[1:]} (seeds {g[0]}-{g[-1]}, anchored alone)"] = (
                    (hourly, pre) if layer else hourly, "stored group product" + (" with the continuation layer" if layer else ""))
                continue
            variants[f"single seed {s} (its carrier anchored alone)"] = (self.state_rule(prod, a["traj"].astype(np.float64), a["off"], a["micro"]),
                                                                         "stored single-seed product" if entry.get("seed_paths") else "re-anchored")
        for mode, lab in MODE_ABLATION.items():
            if mode in cache:
                a = cache[mode][0]
                note = "identical by construction (no micro-load window)" if cache[mode][1].get("identical_to") else "re-anchored"
                if layer:
                    variants[lab] = (self.deliver(prod, a), note + ", with the continuation layer")
                else:
                    variants[lab] = (self.state_rule(prod, a["traj"].astype(np.float64), a["off"], a["micro"]), note)
        ho = self._deep(face, entry, "history_only.npz", P)
        if ho is not None:
            variants["history-only configuration (anchoring layer without a carrier)"] = (ho, "the anchoring layer called without a carrier (saved study forecasts)")
        for arm, f in entry.get("variant_paths", {}).items():              # the study's other recipe arms and configurations, same windows and store
            if arm in RECIPE_VARIANTS:
                lab = RECIPE_VARIANTS[arm]
            elif config_of(arm) is not None:
                lab = config_label(arm)
            else:
                continue
            if f is None or not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            tr = self.align(z, P, pad=prod["pad"], key="traj")
            if tr is None:
                self.notes[face].append(f"recipe arm {arm} does not cover the test windows; not compared")
                continue
            off, mic = self.align(z, P, pad=prod["pad"], key="off"), self.align(z, P, pad=prod["pad"], key="micro")
            v = self.state_rule(prod, tr.astype(np.float64), off, mic)
            note = "saved study forecasts (three seeds), seasonal naive on state windows"
            cfg = config_of(arm)
            if cfg is not None:
                note = f"saved study product ({n_words(cfg[1])} seeds{', with the continuation layer' if cfg[2] else ''}), seasonal naive on state windows"
                if cfg[2] and "traj_anchored" in z.files:                    # energy read before the layer (readout rule A), as for ankyra-lite
                    ta = self.align(z, P, pad=prod["pad"], key="traj_anchored")
                    v = (v, self.state_rule(prod, ta.astype(np.float64), off, mic))
                comp = config_component(arm, self.parts)
                if comp:
                    COMPONENT.setdefault(lab, comp)
            variants[lab] = (v, note)
        if not CB.on_corrected_cineldi(face, entry):
            if prod.get("S3s") is not None:
                variants[S3S_LABEL] = (prod["S3s"]["traj"], "saved study forecasts, seasonal naive on state windows")
            for lab, rel in VARIANT_FILES.items():
                f = CB.AO / rel.format(face=face)
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=False)
                tr = self.align(z, P, key="traj")
                if tr is None:
                    continue
                off, mic = self.align(z, P, key="off"), self.align(z, P, key="micro")
                note = "saved study forecasts, seasonal naive on state windows"
                if face == "drammen" and entry["store"] == "corrected":
                    note += f"; {DRAMMEN_OLD_POOL}"
                variants[lab] = (self.state_rule(prod, tr.astype(np.float64), off, mic), note)
            if face == "drammen" and entry["store"] == "corrected" and prod.get("S3s") is not None:
                variants[S3S_LABEL] = (variants[S3S_LABEL][0], variants[S3S_LABEL][1] + f"; {DRAMMEN_OLD_POOL}")
        else:
            self.notes[face].append("the earlier carriers (from-scratch recipe and its earlier continuations) exist on the original CINELDI store only; not compared with a corrected-store product")
        out = {}
        for lab, (v, note) in variants.items():
            Lv = self.losses(y, *v) if isinstance(v, tuple) else self.losses(y, v)
            for err, off in (("hourly", 9), ("energy", 11)):
                c = self.ctr(L0[err], Lv[err], unit, month, off)
                out[f"{lab}|{err}"] = c
                r = self.rnd(c)
                self.rows["ablation"].append({"set": P["name"], "tier": P["tier"], "ablation": lab, "log_ratio": r["log_ratio"], "um_low": r["um_low"],
                                              "um_high": r["um_high"], "improvement_pct": r["improvement_pct"], "error": err, "windows": len(y),
                                              "units": int(len(np.unique(unit))), "resolved": c["resolved"], "note": note, "flag": flag})
                if lab in COMPONENT:
                    self.rows["component_contributions"].append({"set": P["name"], "component": COMPONENT[lab], "improvement_pct": r["improvement_pct"],
                                                                 "log_ratio": r["log_ratio"], "um_low": r["um_low"], "um_high": r["um_high"],
                                                                 "kind": "direct contrast", "error": err, "resolved": c["resolved"], "flag": flag})
        single = [k for k in seedp if not isinstance(k, str)]
        sk = [out.get(f"single seed {s} (its carrier anchored alone)|hourly") for s in single]
        if single and all(sk) and all(x["log_ratio"] is not None for x in sk):
            w = n_words(len(single))
            for err in ("hourly", "energy"):
                v = [out[f"single seed {s} (its carrier anchored alone)|{err}"]["log_ratio"] for s in single]
                lr = float(np.mean(v))
                self.rows["component_contributions"].append({"set": P["name"], "component": f"{w}-seed ensemble (against one seed, mean over the {w} seeds)",
                                                             "improvement_pct": round(100 * (1 - math.exp(lr)), 3), "log_ratio": round(lr, 6), "um_low": "", "um_high": "",
                                                             "kind": f"mean of {w} direct contrasts (point estimate)", "error": err, "resolved": "", "flag": flag})
        gl = [f"three-seed group {s[1:]} (seeds {a['seeds'][0]}-{a['seeds'][-1]}, anchored alone)" for s, a in seedp.items() if isinstance(s, str)]
        if len(gl) >= 2 and all((out.get(f"{g}|hourly") or {}).get("log_ratio") is not None for g in gl):
            for err in ("hourly", "energy"):
                lr = float(np.mean([out[f"{g}|{err}"]["log_ratio"] for g in gl]))
                self.rows["component_contributions"].append({"set": P["name"], "component": f"{n_words(nseeds)}-seed ensemble (against a three-seed group, mean over "
                                                                                             f"the {n_words(len(gl))} groups)",
                                                             "improvement_pct": round(100 * (1 - math.exp(lr)), 3), "log_ratio": round(lr, 6), "um_low": "", "um_high": "",
                                                             "kind": f"mean of {n_words(len(gl))} direct contrasts (point estimate)", "error": err, "resolved": "", "flag": flag})
        return out

    def _scorer_check(self, face, abl):
        """The product reading against the study's own scorer: the per-population points of the released recipe against
        the plain continuation (hourly, monthly energy) and against the from-scratch stage (hourly) in
        smallnet_sr_formal_v1/score_all.json must equal this run's ablation contrasts (same estimator, same losses).
        Only for the 0.1.x product that record scored (not for a 0.2.0 configuration)."""
        if self.prods.get("configuration") or not (str(self.prods.get("recipe", "")).startswith("span return") and CB.SR_SCORE.exists()):
            return None
        S = json.loads(CB.SR_SCORE.read_text(encoding="utf-8"))
        want = {(RECIPE_VARIANTS["F"], "hourly"): S["nine"]["R vs F"]["hourly"]["by_population_pct"].get(face),
                (RECIPE_VARIANTS["F"], "energy"): S["nine"]["R vs F"]["monthly (curve sum)"]["by_population_pct"].get(face),
                (RECIPE_VARIANTS["P3s"], "hourly"): S["descriptive vs un-continued P3s (unit x month only)"]["R_ens vs P3s | nine"]["by_population_pct"].get(face)}
        got, diffs = {}, []
        for (lab, err), w in want.items():
            c = abl.get(f"{lab}|{err}")
            if w is None or c is None or c.get("improvement_pct") is None:
                continue
            got[f"{lab}|{err}"] = {"study": w, "this_run": c["improvement_pct"]}
            diffs.append(abs(w - c["improvement_pct"]))
        if not diffs:
            return None
        ok = max(diffs) < 1e-8
        if not ok:
            self.notes[face].append(f"the product reading does not reproduce the study's scorer (largest difference {max(diffs):.3g} percentage points)")
        return {"compared": got, "max_abs_diff_pct_points": max(diffs), "pass": ok}

    def _seeds(self, face, P, prod, seeds, ens, baselines, seedp, flag):
        """Each seed alone against the ensemble of the same stage, and each seed's and the ensemble's hourly contrast
        against TimesFM and Chronos-2-X.  BDG2 also without the three near-zero meters: there the carrier alone has a
        nonzero error while some baselines' errors are near zero, which dominates the unit-equal mean.  A six-seed
        configuration has no single-seed anchored product: its anchored stage compares the two stored three-seed groups
        with the six-seed ensemble (with the continuation layer when the product has it; energy before the layer)."""
        out = {}
        ens_lab = f"{n_words(len(seeds))}-seed ensemble"
        layer = prod.get("cont") is not None
        stages = [("carrier alone", (ens, None), {s: (p, None) for s, p in enumerate(seeds)})]
        single = [k for k in seedp if not isinstance(k, str)]
        groups = [k for k in seedp if isinstance(k, str)]
        if single and len(single) == len(seeds):
            stages.append(("anchored (ankyra-lite)", (prod["traj"], None),
                           {s: (self.state_rule(prod, seedp[s]["traj"].astype(np.float64), seedp[s]["off"], seedp[s]["micro"]), None) for s in single}))
        elif groups:
            per = {}
            for k in groups:
                a = seedp[k]
                pre = self.state_rule(prod, a["traj_pre"].astype(np.float64), a["off"], a["micro"]) if layer else None
                per[f"group {k[1:]}"] = (self.state_rule(prod, a["traj"].astype(np.float64), a["off"], a["micro"]), pre, a["seeds"])
            stages.append(("anchored (ankyra-lite)", (prod["traj"], prod["traj_pre"] if layer else None), per))
        sets = [("all units", np.ones(len(P["yL"]), bool))]
        if face == "bdg2":
            sets.append(("without the three near-zero meters", ~np.isin(P["uL"], self.T.near_zero)))
        for uset, keep in sets:
            y, unit, month = P["yL"][keep], P["uL"][keep], P["mL"][keep]
            pre = "" if uset == "all units" else f"{uset}|"
            for stage, (ens_p, ens_e), per in stages:
                Le = self.losses(y, ens_p[keep], None if ens_e is None else ens_e[keep])
                Lb = {comp: self.losses(y, baselines[comp][keep]) for comp in ("TimesFM", "Chronos-2-X")}
                for comp in Lb:
                    c = self.ctr(Le["hourly"], Lb[comp]["hourly"], unit, month, 9)
                    out[f"{pre}{stage}|ensemble|{comp}"] = c
                    self._seed_row(P, uset, keep, stage, ens_lab, "hourly", comp, c, flag)
                for s_, item in per.items():
                    p, pe = item[0], item[1]
                    name = f"seed {s_}" if not isinstance(s_, str) else s_
                    shown = name if not isinstance(s_, str) else f"seeds {item[2][0]}-{item[2][-1]} (three-seed group)"
                    Ls = self.losses(y, p[keep], None if pe is None else pe[keep])
                    for err, off in (("hourly", 9), ("energy", 11)):
                        c = self.ctr(Ls[err], Le[err], unit, month, off)
                        out[f"{pre}{stage}|{name}|{ens_lab}|{err}"] = c
                        self._seed_row(P, uset, keep, stage, shown, err, ens_lab, c, flag)
                    for comp in Lb:
                        c = self.ctr(Ls["hourly"], Lb[comp]["hourly"], unit, month, 9)
                        out[f"{pre}{stage}|{name}|{comp}"] = c
                        self._seed_row(P, uset, keep, stage, shown, "hourly", comp, c, flag)
                for comp in Lb:
                    names = [f"seed {s_}" if not isinstance(s_, str) else s_ for s_ in per]
                    v = [out[f"{pre}{stage}|{nm}|{comp}"]["improvement_pct"] for nm in names]
                    out[f"{pre}{stage}|spread|{comp}"] = {"min": float(min(v)), "max": float(max(v)), "sd": float(np.std(v, ddof=1))} | (
                        {"members": names} if list(per) != [0, 1, 2] else {})
        return out

    def _seed_row(self, P, uset, keep, stage, seed, err, comp, c, flag):
        r = self.rnd(c)
        self.rows["seed_spread"].append({"set": P["name"], "tier": P["tier"], "unit_set": uset, "stage": stage, "seed": seed, "error": err, "comparator": comp,
                                         "windows": int(keep.sum()), "units": int(len(np.unique(P["uL"][keep]))), "log_ratio": r["log_ratio"], "um_low": r["um_low"],
                                         "um_high": r["um_high"], "improvement_pct": r["improvement_pct"], "resolved": c["resolved"], "flag": flag})

    def _interval_dir(self, face, entry):
        return (CB.SANDBOX / "intervals_v6") if CB.on_corrected_cineldi(face, entry) else (CB.AO / "intervals_v6")

    def _intervals(self, face, entry, P, prod, ens, flag):
        """The release interval (tools_intervals_v6_driver_v1.score): residual quantiles Q of the history-only estimator at
        the unit's pseudo-origins (independent of the carrier), scaled by s0 = origin_scale(context), by hour of day and
        workday, added to the point forecast; here centred on ankyra-lite (and the carrier alone)."""
        IV = self.IV
        y, unit, month = P["yL"], P["uL"], P["mL"]
        d = self._interval_dir(face, entry)
        Z = np.load(d / f"{face}_Q.npz", allow_pickle=False)
        NT = np.load(d / f"{face}_native_timesfm.npz", allow_pickle=False)
        NC = np.load(d / f"{face}_native_chronos.npz", allow_pickle=False)
        Q = self.align(Z, P, key="Q")
        used = self.align(Z, P, key="used")
        if Q is None:
            self.notes[face].append("interval quantiles do not cover the test windows; intervals not computed")
            return None
        s0 = np.array([IV.origin_scale(c) for c in P["ctxL"]])
        hod = np.stack([np.arange(o, o + HORIZON) % 24 for o in P["woL"]])
        work = (P["tyL"] < 5).astype(int)
        arms = {}
        for name, point in (("ankyra_lite", prod["traj"]), ("carrier", ens)):
            Bq = np.stack([IV.bands(point[i], s0[i], Q[i], hod[i], work[i]) for i in range(len(y))])
            arms[name] = {lv: Bq[:, j] for j, lv in enumerate(IV.LEVELS)}
        arms["timesfm_native"] = {lv: self.align(NT, P, key=f"q_timesfm_{lv}").astype(np.float64) for lv in (0.1, 0.5, 0.9)}
        arms["chronos_native"] = {lv: self.align(NC, P, key=f"q_chronos_{lv}").astype(np.float64) for lv in (0.05, 0.1, 0.5, 0.9, 0.95)}
        M, summ = {}, {}
        for arm, q in arms.items():
            m = IV.metrics(y, q, [lv for lv in IV.LEVELS if lv in q])
            M[arm] = m
            byw = {}
            for w in range(4):
                sl = WEEK == w
                lo, hi = q[0.1][:, sl], q[0.9][:, sl]
                yy = y[:, sl]
                byw[f"cov80_week{w + 1}"] = float(((yy >= lo) & (yy <= hi)).mean())
                byw[f"winkler80_week{w + 1}"] = float(((hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0)).mean())
                if 0.05 in q:
                    byw[f"cov90_week{w + 1}"] = float(((yy >= q[0.05][:, sl]) & (yy <= q[0.95][:, sl])).mean())
            summ[arm] = {**{k: float(v.mean()) for k, v in m.items()}, **byw}
            if arm != "carrier":
                self.rows["intervals_by_population"].append({"set": P["name"], "tier": P["tier"], "arm": arm, "windows": len(y), "units": int(len(np.unique(unit))),
                                                             **{k: round(v, 5) for k, v in summ[arm].items()}, "flag": flag})
        con = {}
        for other in ("timesfm_native", "chronos_native"):
            c = self._interval_ctr(M["ankyra_lite"]["winkler80"], M[other]["winkler80"], unit, month)
            con[other] = c
            r = self.rnd(c)
            self.rows["intervals_winkler_contrasts"].append({"version": self.prods["label"], "set": P["name"], "comparator": other, "log_ratio": r["log_ratio"],
                                                             "um_low": r["um_low"], "um_high": r["um_high"], "improvement_pct": r["improvement_pct"],
                                                             "resolved": c["resolved"], "flag": flag})
        out = {"arms": summ, "winkler80_ankyra_lite_vs": con, "pseudo_windows_used_mean": float(np.mean(used)), "quantile_source": CB.show(d / f"{face}_Q.npz")}
        if face == "households":
            hh = {"set": P["name"], "windows": int(len(y)), "units": int(len(np.unique(unit))),
                  "note": ("Test windows only. ours_around_ankyra_lite / ours_around_carrier: the release interval (residual quantiles of the history-only "
                           "estimator at the unit's pseudo-origins, scaled by the origin scale, by hour of day and workday) centred on ankyra-lite and on the "
                           "carrier alone; timesfm_native and chronos_native: the models' own quantiles. Coverage is a fraction; widths, Winkler scores and "
                           "pinball losses are means in kW. TimesFM has no 0.05 and 0.95 quantiles. The release file used another residual archive on all "
                           "1,258 windows; these quantiles are those of intervals_by_population.csv."),
                  "arms": {"ours_around_ankyra_lite": {k: v for k, v in summ["ankyra_lite"].items() if "week" not in k},
                           "ours_around_carrier": {k: v for k, v in summ["carrier"].items() if "week" not in k},
                           "timesfm_native": {k: v for k, v in summ["timesfm_native"].items() if "week" not in k},
                           "chronos_native": {k: v for k, v in summ["chronos_native"].items() if "week" not in k}}, "contrasts": {}}
            for other, key in (("ours_around_carrier", "carrier"), ("timesfm_native", "timesfm_native"), ("chronos_native", "chronos_native")):
                for met in ("winkler80", "pinball_mean_common"):
                    c = self._interval_ctr(M["ankyra_lite"][met], M[key][met], unit, month)
                    hh["contrasts"][f"{met}_ours_around_ankyra_lite_vs_{other}"] = {"log_ratio": c["log_ratio"], "UM": [c["um_low"], c["um_high"]], "pct": c["improvement_pct"]}
            self.extra_json["intervals_households.json"] = hh
        return out

    def _interval_ctr(self, a, b, unit, month):
        T = self.T
        with np.errstate(all="ignore"):
            c = CB.contrast_um(T, a, b, unit, month, np.random.default_rng(INTERVAL_SEED))
        lr, (lo, hi) = c["log_ratio"], c["UM"]
        return {"log_ratio": float(lr), "um_low": float(lo), "um_high": float(hi), "improvement_pct": 100 * (1 - math.exp(lr)), "resolved": _resolved(lo, hi)}

    def _deep_dir(self, face, entry):
        return (CB.SANDBOX / "deep_carrier_v1" / face) if CB.on_corrected_cineldi(face, entry) else (CB.AO / "deep_carrier_v1" / face)

    def _deep(self, face, entry, name, P, k0=False):
        f = self._deep_dir(face, entry) / name
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=False)
        a = self.align(z, P)
        if a is None:
            return None
        return (a[:, 0] if k0 else a).astype(np.float64)

    def _carriers(self, face, entry, P, prod, ens, flag):
        """trained_carriers.csv of the release, with the small carrier network as one of the carriers: each carrier inside
        the same anchoring layer (the deep carriers' saved anchored forecasts; their training cut is the small net's)."""
        y, unit, month = P["yL"], P["uL"], P["mL"]
        lite = prod["traj_pre"]                                             # before the continuation layer (= prod["traj"] without one)
        layer = prod.get("cont") is not None
        ho = self._deep(face, entry, "history_only.npz", P)
        out = {}
        rows = []

        def add(carrier, contrast, a, b):
            La, Lb = self.losses(y, a), self.losses(y, b)
            for err, off in (("hourly", 9), ("energy", 11)):
                c = self.ctr(La[err], Lb[err], unit, month, off)
                out[f"{carrier}|{contrast}|{err}"] = c
                r = self.rnd(c)
                rows.append({"set": P["name"], "tier": P["tier"], "subset": "late", "windows": len(y), "units": int(len(np.unique(unit))), "carrier": carrier,
                             "contrast": contrast, "error": err, "log_ratio": r["log_ratio"], "um_low": r["um_low"], "um_high": r["um_high"],
                             "improvement_pct": r["improvement_pct"], "resolved": c["resolved"], "flag": flag})

        add(LITE_CARRIER, "anchored vs carrier alone", lite, ens)
        if ho is not None:
            add(LITE_CARRIER, "anchored vs history only", lite, ho)
        if layer:
            add(LITE_CARRIER, f"anchored{LAYER_ROW} vs carrier alone", prod["traj"], ens)
        cuts = {}
        for X in DEEP_CARRIERS:
            an = self._deep(face, entry, f"anchored_{X}.npz", P)
            ca = self._deep(face, entry, f"carrier_{X}.npz", P, k0=True)
            if an is None or ca is None:
                self.notes[face].append(f"trained carrier {X}: no saved anchored forecast on the test windows")
                continue
            add(X, "ankyra-lite vs anchored carrier", lite, an)
            if layer:
                add(X, f"ankyra-lite{LAYER_ROW} vs anchored carrier", prod["traj"], an)
            add(X, "anchored vs carrier alone", an, ca)
            if ho is not None:
                add(X, "anchored vs history only", an, ho)
            z = np.load(self._deep_dir(face, entry) / f"carrier_{X}.npz", allow_pickle=False)
            cuts[X] = int(z["cutoff_Cprime"]) if "cutoff_Cprime" in z.files else None
        q = np.load(Path(entry["run_paths"][0]) / "carrier.npz", allow_pickle=False)
        own_cut = int(q["cut"]) if "cut" in q.files else None
        out["training_cuts"] = {"small carrier network": own_cut, **cuts}
        out["training_cut_axis_shift_h"] = int(prod["pad"])               # the deep carriers' cuts are on the original CINELDI hour axis
        out["same_training_cut"] = all(v is not None and v + int(prod["pad"]) == own_cut for v in cuts.values()) if cuts else None
        if CB.on_corrected_cineldi(face, entry) and cuts:
            out["carriers_trained_on"] = "original CINELDI store (anchored on the corrected store)"
            self.notes[face].append("the other trained carriers (" + ", ".join(cuts) + ") were trained on the original CINELDI store and "
                                    "anchored on the corrected store (only their anchoring was re-run there); LSTM and PatchTST read "
                                    "load only, LSTM-T-KAN-M as a carrier also reads its calendar channels")
        self.rows["carriers_in_anchoring_layer"] += rows
        return out

    @staticmethod
    def blocks(e):
        """Per-window squared error of each block (tools_contrast_ii_blocks_v1.blocks); the three blocks add up to the
        hourly mean squared error."""
        c = e.reshape(len(e), DAYS, HR)
        d = c.mean(2)
        lev = e.mean(1)
        b = {"level": lev ** 2, "daily path": ((d - lev[:, None]) ** 2).mean(1), "within-day": ((c - d[:, :, None]) ** 2).mean(axis=(1, 2)), "hourly": (e ** 2).mean(1)}
        assert np.allclose(b["level"] + b["daily path"] + b["within-day"], b["hourly"], rtol=1e-9, atol=1e-12), "blocks do not add up"
        return b

    def _control_path(self, face, entry):
        return (CB.SANDBOX if CB.on_corrected_cineldi(face, entry) else CB.AO) / "contrast_ii_control_v1" / face / "control.npz"

    def _contrast_ii(self, face, entry, P, prod, ens, flag, carriers):
        """contrast_ii_blocks.csv and contrast_ii_control.csv of the release with ankyra-lite as the carrier-enabled
        configuration: the error of every test window split into level, daily path and within-day blocks, each block with
        the release estimator (unit-equal log RMS ratio, unit x month bootstrap, seeds SEED + 9 / 11 / 13 / 15).  The
        history-only configuration and the two controls are saved study forecasts of the same anchoring layer without a
        carrier (they do not depend on the carrier)."""
        T = self.T
        y, unit, month = P["yL"], P["uL"], P["mL"]
        ho = self._deep(face, entry, "history_only.npz", P)
        if ho is None:
            self.notes[face].append("no saved history-only forecast on the test windows; contrast (ii) by block not computed")
            return None
        P_ = {"A": prod["traj_pre"], "C": ens, "H": ho}
        if prod.get("cont") is not None:
            P_["Ac"] = prod["traj"]
        cf = self._control_path(face, entry)
        if cf.exists():
            z = np.load(cf, allow_pickle=False)
            ah, h2, hc = (self.align(z, P, key=k) for k in ("pred_AH", "pred_H2", "pred_H"))
            if ah is not None and h2 is not None:
                P_["AH"], P_["H2"] = ah.astype(np.float64), h2.astype(np.float64)
                same_h = hc is not None and np.array_equal(hc.astype(np.float32), ho.astype(np.float32))
                if not same_h:
                    self.notes[face].append("the control file's history-only forecast differs from the saved history-only configuration")
        else:
            self.notes[face].append("no saved carrier-free control; contrast_ii_control rows not computed")
        L = {k: self.blocks(y - v) for k, v in P_.items()}
        subsets = {"all late windows": np.ones(len(y), bool)}
        if prod["state"].any():
            subsets[STATE_EXCLUDED] = ~prod["state"]
        specs = [("contrast_ii_blocks", LITE_CARRIER, "anchored vs history only", "A", "H"),
                 ("contrast_ii_blocks", LITE_CARRIER, "anchored vs carrier alone", "A", "C"),
                 ("contrast_ii_blocks", LITE_CARRIER, "carrier alone vs history only", "C", "H")]
        if "Ac" in P_:
            specs.append(("contrast_ii_blocks", LITE_CARRIER, f"anchored{LAYER_ROW} vs history only", "Ac", "H"))
        if "AH" in P_:
            specs += [("contrast_ii_control", "none", "carrier-free control vs history only", "AH", "H"),
                      ("contrast_ii_control", LITE_CARRIER, "anchored vs carrier-free control", "A", "AH"),
                      ("contrast_ii_control", LITE_CARRIER, "anchored vs history only with the full analog-day shape", "A", "H2"),
                      ("contrast_ii_control", "none", "history only with the full analog-day shape vs history only", "H2", "H")]
        out, checks = {}, []
        for table, carrier, contrast, a_, b_ in specs:
            for blk in BLOCKS:
                for sub, m in subsets.items():
                    a, b = L[a_][blk][m], L[b_][blk][m]
                    u, mo = unit[m], month[m]
                    identical = bool(np.array_equal(a, b))
                    if identical:
                        r_, lo, hi = 0.0, 0.0, 0.0
                    else:
                        with np.errstate(all="ignore"):
                            r_ = T.DS.log_ratio(a, b, u)
                            lo, hi = T.DS.boot(a, b, u, mo, np.random.default_rng(T.DS.SEED + BLOCK_SEED[blk]))["UM"]
                    finite = bool(np.isfinite(r_))
                    res = "identical forecasts" if identical else ("undefined" if not finite else _resolved(lo, hi))
                    out[f"{table}|{contrast}|{blk}|{sub}"] = {"log_ratio": float(r_), "um_low": float(lo), "um_high": float(hi), "resolved": res}
                    self.rows[table].append({"set": P["name"], "tier": P["tier"], "subset": "late", "analysis": sub, "windows": int(m.sum()),
                                             "units": int(len(np.unique(u))), "carrier": carrier, "contrast": contrast, "block": blk,
                                             "log_ratio": round(float(r_), 6) + 0.0 if finite else "", "um_low": round(float(lo), 6) + 0.0 if finite else "",
                                             "um_high": round(float(hi), 6) + 0.0 if finite else "",
                                             "improvement_pct": round(100 * (1 - math.exp(r_)), 4) + 0.0 if finite else "", "resolved": res, "flag": flag})
                    # the pre-layer hourly / level rows reproduce carriers_in_anchoring_layer.csv (the layer row has no counterpart there)
                    if sub == "all late windows" and blk in ("hourly", "level") and table == "contrast_ii_blocks"                             and contrast in ("anchored vs history only", "anchored vs carrier alone"):
                        cc = carriers.get(f"{LITE_CARRIER}|{contrast}|{'hourly' if blk == 'hourly' else 'energy'}")
                        if cc is not None and cc.get("log_ratio") is not None:
                            checks.append(abs(cc["log_ratio"] - r_) < 1e-12 and abs(cc["um_low"] - lo) < 1e-12 and abs(cc["um_high"] - hi) < 1e-12)
        out["reproduces_carriers_in_anchoring_layer"] = all(checks) if checks else None
        if checks and not all(checks):
            self.notes[face].append("contrast (ii) by block: the hourly / level rows do not reproduce carriers_in_anchoring_layer.csv")
        return out

    def _history(self, face, P, prod, ens, baselines):
        y, unit, month = P["yL"], P["uL"], P["mL"]
        m = P["histL"]
        keff = np.clip((m - 1344) // 744, 0, 12)
        comps = {"carrier alone": ens, "TimesFM": baselines["TimesFM"], "Chronos-2-X": baselines["Chronos-2-X"]}
        e = {k: ((p - y) ** 2).mean(1) for k, p in comps.items()}
        e["ankyra-lite"] = ((prod["traj"] - y) ** 2).mean(1)
        self.hist_pool.append({"face": face, "e": e, "unit": np.array([f"{face}:{u}" for u in unit]), "month": month, "m": m, "k": keff})
        out = {"history_hours_median": float(np.median(m)), "keff_mean": float(keff.mean()), "strata": {}}
        for lab, lo, hi in STRATA_M + STRATA_K:
            sel = ((keff >= lo) & (keff <= hi)) if lab.startswith("K_eff") else ((m >= lo) & (m < hi))
            s = {"windows": int(sel.sum()), "units": int(len(np.unique(unit[sel])))}
            if sel.sum() < 10 or s["units"] < 3:
                out["strata"][lab] = s
                continue
            for other in comps:
                c = self._hctr(e["ankyra-lite"][sel], e[other][sel], unit[sel], month[sel])
                s[other] = c
                self.rows["history_length"].append({"set": P["name"], "stratum": lab, "windows": s["windows"], "units": s["units"], "comparator": other,
                                                    **{k: v for k, v in self.rnd(c).items() if k in ("log_ratio", "um_low", "um_high", "improvement_pct")},
                                                    "resolved": c["resolved"], "flag": CB.FLAG_BY_NAME.get(P["name"], "")})
            out["strata"][lab] = s
        return out

    def _hctr(self, a, b, unit, month):
        T = self.T
        c = CB.contrast_um(T, a, b, unit, month, np.random.default_rng(HISTORY_SEED))
        lr, (lo, hi) = c["log_ratio"], c["UM"]
        return {"log_ratio": float(lr), "um_low": float(lo), "um_high": float(hi), "improvement_pct": 100 * (1 - math.exp(lr)), "resolved": _resolved(lo, hi)}

    def _near_zero(self, P, prod, baselines):
        """bdg2_near_zero_sensitivity.csv of the release: hourly contrasts (the stream of benchmark_pairwise) and
        ankyra-lite's mean unit rank among the 22 forecasters, with and without the three near-zero meters."""
        T = self.T
        y, unit, month = P["yL"], P["uL"], P["mL"]
        preds = {CB.REF_B: prod["traj"]} | baselines
        out = {}
        for lab, keep in (("all units", np.ones(len(y), bool)), ("without the three near-zero meters", ~np.isin(unit, T.near_zero))):
            yy, uu, mm = y[keep], unit[keep], month[keep]
            mr, rank, of, _ = CB.unit_rank({k: v[keep] for k, v in preds.items()}, yy, uu, CB.REF_B)
            a = ((prod["traj"][keep] - yy) ** 2).mean(1)
            for mname in sorted(baselines):
                with np.errstate(all="ignore"):
                    c = CB.contrast_um(T, a, ((baselines[mname][keep] - yy) ** 2).mean(1), uu, mm, np.random.default_rng(T.DS.SEED + 9))
                self.rows["bdg2_near_zero_sensitivity"].append({"unit_set": lab, "subset": "late", "model": mname, "log_ratio": round(c["log_ratio"], 6),
                                                                "um_low": round(c["UM"][0], 6), "um_high": round(c["UM"][1], 6),
                                                                "improvement_pct": round(100 * (1 - math.exp(c["log_ratio"])), 3), "ankyra_mean_unit_rank": round(mr, 4)})
            out[lab] = {"windows": int(keep.sum()), "units": int(len(np.unique(uu))), "mean_unit_rank": mr, "rank": rank, "of": of}
        return out

    def _example(self, P, prod, ens, baselines):
        """The release's example window (Cambridge, UCam_Building_b37, origin 2011-09-01) is not a test window; the
        nearest test window of the same building in time is used."""
        start = datetime.fromisoformat(P["start_iso"].replace("Z", "+00:00")).replace(tzinfo=None)
        target = datetime.fromisoformat(EXAMPLE_ORIGIN_DATE)
        cand = np.flatnonzero(P["uL"] == EXAMPLE_UNIT)
        if not len(cand):
            cand = np.arange(len(P["uL"]))
        dates = [start + timedelta(hours=int(P["woL"][i])) for i in cand]
        j = int(cand[int(np.argmin([abs((d - target).total_seconds()) for d in dates]))])
        o_date = start + timedelta(hours=int(P["woL"][j]))
        rows = [{"hour": h - CONTEXT, "load_kw": round(float(P["ctxL"][j, h]), 4), "timesfm_kw": "", "ankyra_kw": "", "carrier_kw": ""} for h in range(CONTEXT)]
        rows += [{"hour": h, "load_kw": round(float(P["yL"][j, h]), 4), "timesfm_kw": round(float(baselines["TimesFM"][j, h]), 4),
                  "ankyra_kw": round(float(prod["traj"][j, h]), 4), "carrier_kw": round(float(ens[j, h]), 4)} for h in range(HORIZON)]
        if prod.get("cont") is not None:                                    # the trajectory before the continuation layer
            for h, r_ in enumerate(rows):
                r_["ankyra_before_layer_kw"] = "" if h < CONTEXT else round(float(prod["traj_pre"][j, h - CONTEXT]), 4)
        self.rows["example_window_cambridge"] = rows
        js = {"set": "Cambridge", "building": str(P["uL"][j]), "target_month_midpoint": str(P["mL"][j]),
              "forecast_period": f"{o_date.date()} to {(o_date + timedelta(hours=HORIZON)).date()}",
              "context_day_types": [int(x) for x in P["ctL"][j]], "target_day_types": [int(x) for x in P["ttL"][j]],
              "day_type_codes": "Monday 0 ... Sunday 6, holiday 7",
              "source": "University of Cambridge estate building energy archive (Langtry & Choudhary 2024, Zenodo 10.5281/zenodo.10955332), CC BY 4.0",
              "note": ("hour 0 is the forecast origin; negative hours are the 1,344-hour context. ankyra_kw is ankyra-lite (the column keeps the "
                       f"release's name), carrier_kw the small carrier network alone ({n_words(len(self.prods.get('seeds', (0, 1, 2))))}-seed forecast at "
                       "the origin, before the anchoring layer), timesfm_kw zero-shot TimesFM 2.5"
                       + ("; ankyra_before_layer_kw ankyra-lite before the origin-continuity layer (energy, level and peak are read from it)"
                          if prod.get("cont") is not None else "")),
              "selection": (f"the release's example window ({EXAMPLE_UNIT}, origin {EXAMPLE_ORIGIN_DATE}) lies before the Cambridge test set; "
                            "this is the test window of the same building whose origin is nearest to it")}
        self.extra_json["example_window_cambridge.json"] = js
        return {"building": js["building"], "origin": str(o_date.date()), "window_index_in_test_set": j}

    def _first_training_origin(self, face, entry):
        """Earliest origin of the population's own windows in its pooled training set (hours of the record)."""
        if entry["store"] == "original":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import tools_smallnet_formal2_v1 as F2
            pl, _, _ = F2.pool(face)
            _, o, _ = F2.dn_rows(face)
            return int(o[pl[face]].min()), "formal pool of the study (tools_smallnet_formal2_v1.pool)"
        code = ("import sys, json; sys.dont_write_bytecode = True; sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[3]]; "
                "import compare_baselines; import tools_smallnet_sr_formal_dc_v1 as DC; "
                "pl, _, _ = DC.f2_pool(sys.argv[4]); _, o, _ = DC.dn_rows(sys.argv[4]); print(json.dumps(int(o[pl[sys.argv[4]]].min())))")
        r = subprocess.run([sys.executable, "-B", "-c", code, str(CB.HERE), str(CB.HKAD), str(CB.REL), face], capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=1800)
        if r.returncode:
            self.notes[face].append("first training origin on the corrected store not computed: " + CB.scrub(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "error"))
            return None, None
        return int(json.loads(r.stdout.strip().splitlines()[-1])), "formal pool on the corrected store (tools_smallnet_sr_formal_dc_v1.f2_pool)"

    def _datasets(self, face, entry, P, prod):
        prep = json.loads(Path(entry["prep_path"]).read_text(encoding="utf-8"))
        c = prep["cuts"]
        cp, C = int(c["Cprime"]), int(c["C"])
        start = datetime.fromisoformat(c["Cprime_label"]) - timedelta(hours=cp)                # the record's hour 0 (store calendar)
        shift = 0
        if face == "cineldi" and entry["store"] == "original":
            shift = CB.corrected_store_info()[1] - C                                          # original store: hour 0 is 59 days later than labelled
        true0 = start + timedelta(hours=shift)
        first, src = self._first_training_origin(face, entry)
        v0 = int(prep["v0"])
        ow = np.asarray(prod["origins_product"]).astype(int)
        odates = [true0 + timedelta(hours=int(o)) for o in ow]
        months = sorted({(d.year, d.month) for d in odates})
        cand = prep.get("candidates", {})
        share = 100 * cand["val"] / (cand["train"] + cand["val"]) if cand else None
        base = {}
        try:
            base = {r["set"]: r for r in csv.DictReader(io.StringIO(CB.published_text("datasets.csv")))}.get(P["name"], {})
        except OSError:
            pass
        d = lambda h: (true0 + timedelta(hours=int(h))).date().isoformat()                    # noqa: E731
        row = {"set": P["name"], "tier": P["tier"], "country": base.get("country", ""), "population": base.get("population", ""),
               "ankyra_design_status": "seen", "units": base.get("units", P["units_full"]), "windows": base.get("windows", P["windows_full"]),
               "units_late": int(len(np.unique(P["uL"]))), "windows_late": int(len(P["yL"])),
               "development_windows": "yes" if face in DEV6 else "no", "store": entry["store"],
               "first_training_origin": d(first) if first is not None else "", "validation_start": d(v0), "training_cut": d(cp),
               "test_start": d(C), "first_test_origin": min(odates).date().isoformat(), "last_test_origin": max(odates).date().isoformat(),
               "test_windows": int(len(ow)), "test_units": int(len(np.unique(P["uL"]))),
               "record_days_before_cut": round((cp - first) / 24) if first is not None else "",
               "validation_share_pct": round(share, 1) if share is not None else "", "test_origin_months": len(months),
               "test_months_spanned": (months[-1][0] - months[0][0]) * 12 + months[-1][1] - months[0][1] + 1,
               "gap_days_cut_to_test_start": round((C - cp) / 24), "calendar": "true dates" + (" (store labels + 59 days)" if shift else "")}
        sp = json.loads(Path(entry["span_path"]).read_text(encoding="utf-8")) if entry.get("span_path") and Path(entry["span_path"]).exists() else None
        epochs = ""
        if entry.get("start_paths"):
            hist = self._stage2_history(entry["run_paths"][0])
            epochs = max((int(h["epoch"]) for h in hist), default="")
        row.update({"own_training_windows": sp["own_pool_rows"] if sp else "", "returned_validation_windows": sp["span_rows_target"] if sp else "",
                    "returned_label_end": d(sp["label_end_max"]) if sp else "", "continuation_epochs": epochs})
        if self.prods.get("configuration"):                                   # version 0.2.0: the seeds of the configuration
            row["carrier_seeds"] = " ".join(str(x) for x in self.prods.get("seeds", []))
        self.rows["datasets"].append(row)
        return {k: row[k] for k in row} | {"first_training_origin_source": src}

    @staticmethod
    def _stage2_history(rd):
        """The stage-2 epochs of a run: COMPLETE.json (0.1.0 runs) or TRAINED.json (the 0.2.0 study's runs, whose
        COMPLETE.json keeps the last epoch only)."""
        c = json.loads((Path(rd) / "COMPLETE.json").read_text(encoding="utf-8"))
        hist = c.get("history")
        if not hist and (Path(rd) / "TRAINED.json").exists():
            hist = json.loads((Path(rd) / "TRAINED.json").read_text(encoding="utf-8")).get("history", [])
        return hist or []

    def _curves(self, face, entry, P):
        """Both training stages per seed: the from-scratch stage (validation criterion per epoch on the held-out validation
        set, best epoch, early stopping) and, for a continuation recipe, its epochs (epoch 0 = the selected from-scratch
        weights; training loss and the same criterion per epoch, which after the span return is no longer held out)."""
        out = {}
        flag = CB.FLAG_BY_NAME.get(P["name"], "")
        cont = bool(entry.get("start_paths"))
        for s, rd in enumerate(entry["run_paths"]):
            start = Path(entry["start_paths"][s]) if cont else Path(rd)
            f = start / "COMPLETE.json"
            if not f.exists() or "best_epoch" not in json.loads(f.read_text(encoding="utf-8")):
                f = start / "TRAINED.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            best, runs = r.get("best_epoch"), r.get("epochs_run")
            for h in r.get("history", []):
                self.rows["training_curves"].append({"set": P["name"], "tier": P["tier"], "seed": s, "stage": "from scratch (early stopping)", "epoch": h["epoch"],
                                                     "training_loss": "", "validation_criterion": round(float(h["val"]), 6), "validation_held_out": "yes",
                                                     "best_epoch": best, "epochs_run": runs, "selected": h["epoch"] == best, "returned_span_draws": "", "flag": flag})
            o = {"best_epoch": best, "epochs_run": runs, "best_validation_criterion": r.get("best_val")}
            if cont:
                hist = self._stage2_history(rd)
                last = max(int(h["epoch"]) for h in hist)
                st = json.loads((Path(rd) / "STARTED.json").read_text(encoding="utf-8")) if (Path(rd) / "STARTED.json").exists() else {}
                stage = "span return (fixed epochs)" if st.get("span_rows_returned", 1) else "continuation (fixed epochs)"
                if st.get("stage") == "Rw":
                    stage = "span return with inverse-noise loss weighting (fixed epochs)"
                for h in hist:
                    self.rows["training_curves"].append({"set": P["name"], "tier": P["tier"], "seed": s, "stage": stage, "epoch": h["epoch"],
                                                         "training_loss": round(float(h["train_loss"]), 6) if "train_loss" in h else "",
                                                         "validation_criterion": round(float(h["C1"]), 6),
                                                         "validation_held_out": "no (returned to the training set)" if st.get("span_rows_returned") else "yes",
                                                         "best_epoch": "", "epochs_run": last, "selected": int(h["epoch"]) == last,
                                                         "returned_span_draws": h.get("own_draws_from_returned_span", ""), "flag": flag})
                e0 = next((h for h in hist if int(h["epoch"]) == 0), None)
                o.update({"continuation_epochs": last, "returned_span_windows": st.get("span_rows_returned"),
                          "start_criterion_reproduced": (abs(e0["C1"] - r["best_val"]) / abs(r["best_val"]) < 1e-5) if e0 and r.get("best_val") else None,
                          "final_training_loss": hist[-1].get("train_loss"), "final_criterion": hist[-1].get("C1")})
            out[s] = o
        return out

    def _continuation(self, face, P, prod, flag):
        """continuation_layer.csv: the origin-continuity layer on against off on the same carrier and anchoring layer (the
        product after against before the layer, seasonal naive on state windows in both), hourly error by lead time and the
        delivered trajectory's energy, unit x month interval (fresh stream SEED + 9, energy + 11).  continuation_record.csv:
        what the layer did on the test windows (from the release layer's own record of every window, read with no target)
        and the gap between the delivered trajectory's mean and the energy readout 744 x level (readout rule A)."""
        y, unit, month = P["yL"], P["uL"], P["mL"]
        on, off, st = prod["traj"], prod["traj_pre"], prod["state"]
        base = {"set": P["name"], "tier": P["tier"], "windows": int(len(y)), "units": int(len(np.unique(unit))), "state_windows": int(st.sum()), "flag": flag or ""}
        out, losses = {"segments": {}}, {}
        for lab, a, b in CONT_SEGMENTS:
            la, lb = ((on[:, a:b] - y[:, a:b]) ** 2).mean(1), ((off[:, a:b] - y[:, a:b]) ** 2).mean(1)
            c = self.ctr(la, lb, unit, month, 9)
            losses[lab] = (la, lb)
            out["segments"][f"{lab}|hourly"] = c
            r = self.rnd(c)
            self.rows["continuation_layer"].append(base | {"segment": lab, "hours": f"{a}-{b - 1}", "error": "hourly",
                                                           **{k: r[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct")}, "resolved": c["resolved"]})
        ea, eb = (on.mean(1) - y.mean(1)) ** 2, (off.mean(1) - y.mean(1)) ** 2
        c = self.ctr(ea, eb, unit, month, 11)
        losses["whole window|energy"] = (ea, eb)
        out["segments"]["whole window|energy (delivered trajectory)"] = c
        r = self.rnd(c)
        self.rows["continuation_layer"].append(base | {"segment": "whole window", "hours": "0-743",
                                                       "error": "energy of the delivered trajectory (the energy readout is read before the layer)",
                                                       **{k: r[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct")}, "resolved": c["resolved"]})
        self.cont_pool.append({"face": face, "unit": np.array([f"{face}:{u}" for u in unit]), "month": month, "losses": losses})
        info = prod["cont"]["info"]
        ns = ~st
        CTm = CB._package()
        D = CTm.delta(info["xf"], info["xd"], CTm.CONSTANTS)
        lvl = prod["level"]
        dm = (np.abs(on.mean(1) - lvl) / np.maximum(np.abs(lvl), 1e-9))[ns]
        pos = ns & (off.mean(1) > 0)
        ech = 100 * (on.mean(1)[pos] / off.mean(1)[pos] - 1)
        q = lambda v, x: round(float(np.percentile(v, x)), 6) if len(v) else ""                    # noqa: E731
        row = {"set": P["name"], "tier": P["tier"], "windows": int(len(y)), "state_windows_bypassed": int(info["bypass_state"].sum()),
               "missing_fast_span": int(info["missing_f"].sum()), "missing_day_span": int(info["missing_d"].sum()),
               "nonpositive_fast_span": int(info["nonpositive_f"].sum()), "nonpositive_day_span": int(info["nonpositive_d"].sum()),
               "clipped_fast_share": round(float(info["clipped_f"][ns].mean()), 6) if ns.any() else "",
               "clipped_day_share": round(float(info["clipped_d"][ns].mean()), 6) if ns.any() else "",
               "no_match_reference_day": int((~info["matched"] & ~info["bypass_state"]).sum()),
               "x_fast_median": q(info["xf"][ns], 50), "x_day_median": q(info["xd"][ns], 50),
               "x_fast_mean_abs": round(float(np.abs(info["xf"][ns]).mean()), 6) if ns.any() else "",
               "x_day_mean_abs": round(float(np.abs(info["xd"][ns]).mean()), 6) if ns.any() else "",
               "max_abs_delta": round(float(np.abs(D[ns]).max()), 6) if ns.any() else 0.0,
               "delivered_mean_vs_energy_readout_rel_p50": q(dm, 50), "delivered_mean_vs_energy_readout_rel_p99": q(dm, 99),
               "delivered_mean_vs_energy_readout_rel_max": round(float(dm.max()), 6) if len(dm) else "",
               "layer_energy_change_pct_min": round(float(ech.min()), 4) if len(ech) else "", "layer_energy_change_pct_max": round(float(ech.max()), 4) if len(ech) else "",
               "flag": flag or ""}
        self.rows["continuation_record"].append(row)
        self.cont_pool[-1]["record"] = {"dm": dm, "ech": ech, "D": np.abs(D[ns]).max(1) if ns.any() else np.zeros(0), "info": {k: v[ns] for k, v in info.items()},
                                        "bypass": int(info["bypass_state"].sum()), "no_match": row["no_match_reference_day"]}
        out["record"] = {k: v for k, v in row.items() if k not in ("set", "tier", "flag")}
        return out

    def _continuation_pooled(self):
        """The pooled rows of both continuation files over the populations of the run (units of every population resampled)."""
        if len(self.cont_pool) < 2:
            return None
        name = "all nine populations" if sorted(x["face"] for x in self.cont_pool) == sorted(CB.NINE) else "all populations of this run"
        U = np.concatenate([x["unit"] for x in self.cont_pool])
        M = np.concatenate([x["month"] for x in self.cont_pool])
        out = {"set": name, "segments": {}}
        base = {"set": name, "tier": "", "windows": int(len(U)), "units": int(len(np.unique(U))), "state_windows": "", "flag": ""}
        for lab, a, b in CONT_SEGMENTS + (("whole window|energy", 0, 744),):
            la = np.concatenate([x["losses"][lab][0] for x in self.cont_pool])
            lb = np.concatenate([x["losses"][lab][1] for x in self.cont_pool])
            c = self.ctr(la, lb, U, M, 11 if lab.endswith("energy") else 9)
            out["segments"][lab] = c
            r = self.rnd(c)
            seg, err = (("whole window", "energy of the delivered trajectory (the energy readout is read before the layer)") if lab.endswith("energy")
                        else (lab, "hourly"))
            self.rows["continuation_layer"].append(base | {"segment": seg, "hours": f"{a}-{b - 1}", "error": err,
                                                           **{k: r[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct")}, "resolved": c["resolved"]})
        dm = np.concatenate([x["record"]["dm"] for x in self.cont_pool])
        ech = np.concatenate([x["record"]["ech"] for x in self.cont_pool])
        Dm = np.concatenate([x["record"]["D"] for x in self.cont_pool])
        inf = {k: np.concatenate([x["record"]["info"][k] for x in self.cont_pool]) for k in self.cont_pool[0]["record"]["info"]}
        row = {"set": name, "tier": "", "windows": int(len(U)), "state_windows_bypassed": sum(x["record"]["bypass"] for x in self.cont_pool),
               "missing_fast_span": int(inf["missing_f"].sum()), "missing_day_span": int(inf["missing_d"].sum()),
               "nonpositive_fast_span": int(inf["nonpositive_f"].sum()), "nonpositive_day_span": int(inf["nonpositive_d"].sum()),
               "clipped_fast_share": round(float(inf["clipped_f"].mean()), 6), "clipped_day_share": round(float(inf["clipped_d"].mean()), 6),
               "no_match_reference_day": sum(x["record"]["no_match"] for x in self.cont_pool),
               "x_fast_median": round(float(np.median(inf["xf"])), 6), "x_day_median": round(float(np.median(inf["xd"])), 6),
               "x_fast_mean_abs": round(float(np.abs(inf["xf"]).mean()), 6), "x_day_mean_abs": round(float(np.abs(inf["xd"]).mean()), 6),
               "max_abs_delta": round(float(Dm.max()), 6) if len(Dm) else 0.0,
               "delivered_mean_vs_energy_readout_rel_p50": round(float(np.percentile(dm, 50)), 6),
               "delivered_mean_vs_energy_readout_rel_p99": round(float(np.percentile(dm, 99)), 6),
               "delivered_mean_vs_energy_readout_rel_max": round(float(dm.max()), 6),
               "layer_energy_change_pct_min": round(float(ech.min()), 4), "layer_energy_change_pct_max": round(float(ech.max()), 4), "flag": ""}
        self.rows["continuation_record"].append(row)
        out["record"] = {k: v for k, v in row.items() if k not in ("set", "tier", "flag")}
        return out

    def _deciles_collect(self, face, P, prod, ens, ensc):
        y = P["yL"]
        ybar = y.mean(1)
        unit = P["uL"]
        keep = np.array([abs(ybar[unit == u]).mean() > 1e-6 for u in unit])
        A, Cr = prod["traj"], ens
        self.dec_pool.append({"face": face, "unit": np.array([f"{face}:{u}" for u in unit])[keep], "month": P["mL"][keep], "stats": ensc["stats"][keep],
                              "eA": (A.mean(1) - ybar)[keep] ** 2, "eC": (Cr.mean(1) - ybar)[keep] ** 2,
                              "mA": ((A - y) ** 2).mean(1)[keep], "mC": ((Cr - y) ** 2).mean(1)[keep]})

    # ------------------------------------------------------------------ files added on 19 October 2026
    def added(self, face, entry, P, prod, baselines, flag, hourly_ref=None, ens=None, parts=None):
        """The per-population part of the files added on 19 October 2026 (``compare_baselines.py --supplement`` runs this
        alone): training budget, long-context baselines, the carrier-alone peak row and, from the fields cache, the
        handover arms, lead weeks, within-day trust, whole-window combination, generic partners and the hit rate.
        ``parts=("budget",)``: the training budget only (``compare_baselines.py --budget``)."""
        import budget_tables as BT
        import long_context_tables as LC
        if face not in self.faces:
            self.faces.append(face)
        out = {}
        relaxed = BT.relaxed_forecasts(face, P, prod, entry)
        if relaxed:
            rows, out["budget"] = BT.rows_for(self.T, P, prod, baselines, relaxed, flag, hourly_ref=hourly_ref)
            self.rows["training_budget_sensitivity"] += rows
        if parts is not None and set(parts) <= {"budget"}:
            self.rec.setdefault(face, {}).update(out)
            return out
        if ens is None:
            _, ens = CB.load_carriers(entry, prod["origins_product"], P["wbL"])
        lc = LC.long_context(face, P, prod)
        if lc:
            rows, out["long_context"] = LC.rows_for(self, P, prod, baselines, lc, flag)
            self.rows["long_context_baselines"] += rows
        out["peak_carrier"] = self._peak_carrier(P, prod, ens, flag)
        F = self._fields(face, entry, P, prod)
        if F is None:
            self.notes[face].append("no complete fields cache of lite_reanchor.py (mode fields); the handover, lead-week, within-day, "
                                    "whole-window and hit-rate tables were not computed")
        else:
            out["fields_gate"] = F["_gate"]
            out["handover"] = self._handover(face, P, prod, ens, baselines, flag, F)
            out["combination"] = self._combination(face, P, prod, ens, baselines, flag, F)
            out["within_trust"] = self._within_trust(P, prod, flag, F)
            out["hit_rate"] = self._hit_rate(face, P, prod, flag, F)
        self.rec.setdefault(face, {}).update(out)
        return out

    def _fields(self, face, entry, P, prod):
        """The fields cache (lite_reanchor mode ``fields``) aligned to the panel order, or None when it is missing, stale,
        partial or did not pass its gate."""
        import lite_reanchor as LR
        z, m = LR.load_mode(LR.cache_key(self.prods), face, "fields")
        if z is None:
            return None
        if m.get("product_sha256") != LR._sha(entry["npz_path"]) or m.get("limit") is not None or m.get("error_count"):
            self.notes[face].append("fields cache is stale, partial or has failed windows; not used")
            return None
        if not (m.get("gate") or {}).get("pass"):
            self.notes[face].append("fields cache did not pass its gate (product bit for bit, rebuilt pairs, triples and level); not used")
            return None
        F = {}
        for key in ("traj", "level", "off", "micro") + tuple(LR.FIELD_ARRAYS):
            a = self.align(z, P, pad=prod["pad"], key=key)
            if a is None:
                self.notes[face].append("fields cache does not cover the test windows; not used")
                return None
            F[key] = a
        st = prod["state"]
        if not (np.array_equal(F["off"] | F["micro"], st) and np.array_equal(F["fields_ok"], ~st) and F["rebuild_ok"][~st].all()):
            self.notes[face].append("fields cache: state flags or fields do not match the product; not used")
            return None
        if not np.array_equal(F["traj"].astype(np.float32), prod["raw_pre"].astype(np.float32)):
            self.notes[face].append("fields cache: trajectory differs from the product; not used")
            return None
        F["_gate"] = {k: m["gate"].get(k) for k in ("traj_equal", "level_equal", "state_flags_equal", "fields_pass", "pass")} | {"fields": m.get("fields")}
        return F

    def _rows_ablation(self, P, flag, variants, out):
        """ablation.csv / component_contributions.csv rows of added variants: {label: (forecast, note)}, as _ablation."""
        y, unit, month = P["yL"], P["uL"], P["mL"]
        L0 = self.losses(y, self._lite, getattr(self, "_lite_e", None))
        for lab, (v, note) in variants.items():
            Lv = self.losses(y, *v) if isinstance(v, tuple) else self.losses(y, v)
            for err, off in (("hourly", 9), ("energy", 11)):
                c = self.ctr(L0[err], Lv[err], unit, month, off)
                out[f"{lab}|{err}"] = c
                r = self.rnd(c)
                self.rows["ablation"].append({"set": P["name"], "tier": P["tier"], "ablation": lab, "log_ratio": r["log_ratio"], "um_low": r["um_low"],
                                              "um_high": r["um_high"], "improvement_pct": r["improvement_pct"], "error": err, "windows": len(y),
                                              "units": int(len(np.unique(unit))), "resolved": c["resolved"], "note": note, "flag": flag})
                if lab in COMPONENT:
                    self.rows["component_contributions"].append({"set": P["name"], "component": COMPONENT[lab], "improvement_pct": r["improvement_pct"],
                                                                 "log_ratio": r["log_ratio"], "um_low": r["um_low"], "um_high": r["um_high"],
                                                                 "kind": "direct contrast", "error": err, "resolved": c["resolved"], "flag": flag})

    @staticmethod
    def _f32(x):
        return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)       # stored like the products (float32)

    def _handover(self, face, P, prod, ens, baselines, flag, F):
        """The first release's handover ablation with ankyra-lite's anchoring layer: four arms that share the level with the
        carrier candidate, the delivered within-day block and the projection, and differ only in the weight on the
        carrier's daily means (A0 none, Ah one half, AM one per window from the same pairs with the 31 days pooled, AW the
        weekly weights = ankyra-lite); the fixed division (history level and daily path, the carrier's own within-day
        block); seasonal naive on state windows in every arm."""
        import lite_reanchor as LR
        LR._paths()                                                                     # the repository root (ankyra_lite) on sys.path
        from ankyra_lite.anchor import core as AC
        y, unit, month = P["yL"], P["uL"], P["mL"]
        st, SN = prod["state"], prod["SN"]
        ns = ~st
        n = len(y)
        mT = ens.reshape(n, DAYS, HR).mean(2)
        mH, W, al, am = F["mh"], F["within"], F["alphas"], F["alpha_month"]
        aday = np.repeat(al, [7, 7, 7, 10], axis=1)                                     # weight of each forecast day
        rec_mh = (F["daily"] - aday * mT) / (1 - aday)                                   # m^H from the handover identity (alpha <= 7/8)
        rel = np.abs(rec_mh - mH)[ns].max() / max(1.0, np.abs(mH[ns]).max()) if ns.any() else 0.0
        daily = {"A0": mH, "Ah": 0.5 * mH + 0.5 * mT, "AM": (1 - am[:, None]) * mH + am[:, None] * mT}
        daily["AW"] = np.stack([AC.lead_week_transition(mH[i], mT[i], al[i]) if ns[i] else np.full(DAYS, np.nan) for i in range(n)])
        tr = {}
        for a in ARMS:
            raw = np.maximum(np.repeat(np.where(ns[:, None], daily[a], 0.0), HR, axis=1) + np.where(ns[:, None], W, 0.0), 0.0)
            tr[a] = np.where(st[:, None], SN, self._f32(raw))
        fixed = np.where(st[:, None], SN, self._f32(np.maximum(np.repeat(np.where(ns[:, None], F["fixed"], 0.0), HR, axis=1) + np.where(ns[:, None], F["fwithin"], 0.0), 0.0)))
        gate = {"aw_equals_the_product": bool(np.array_equal(tr["AW"], prod["traj_pre"])), "m_h_identity_max_rel_diff": float(rel),
                "daily_means_reproduced": bool(np.array_equal(np.where(ns[:, None], daily["AW"], 0.0), np.where(ns[:, None], F["daily"], 0.0)))}
        pre_arms = None
        if prod.get("cont") is not None:                                  # every arm through the continuation layer, as ankyra-lite
            pre_arms = dict(tr) | {"fixed division": fixed}
            tr = {a: np.where(st[:, None], SN, self.layered(prod, np.where(st[:, None], 0.0, v), st)) for a, v in tr.items()}
            fixed = np.where(st[:, None], SN, self.layered(prod, np.where(st[:, None], 0.0, fixed), st))
            gate["aw_with_the_layer_equals_the_product"] = bool(np.array_equal(tr["AW"], prod["traj"]))
            gate["arms"] = "with the continuation layer (energy read before it)"
        if not (gate["aw_equals_the_product"] and gate["daily_means_reproduced"] and rel < 1e-9 and gate.get("aw_with_the_layer_equals_the_product", True)):
            self.notes[face].append(f"handover arms: AW does not reproduce ankyra-lite ({gate}); not written")
            return {"gate": gate}
        tr["AW"] = prod["traj"]
        hm = {a: ((v - y) ** 2).mean(1) for a, v in tr.items()}
        hm |= {"carrier alone": ((ens - y) ** 2).mean(1), "TimesFM": ((baselines["TimesFM"] - y) ** 2).mean(1), "fixed division": ((fixed - y) ** 2).mean(1)}
        wk = {a: np.stack([((v - y) ** 2)[:, 24 * a0:24 * a1].mean(1) for a0, a1 in WEEKS4], 1) for a, v in (list(tr.items()) + [("carrier alone", ens),
              ("TimesFM", baselines["TimesFM"]), ("fixed division", fixed)])}
        out = {"gate": gate, "contrasts": {}, "weeks": {}}
        base = {"set": P["name"], "tier": P["tier"], "windows": int(n), "units": int(len(np.unique(unit))), "flag": flag or ""}

        def add(lab, a, b):
            c = self.ctr(a, b, unit, month, 9)
            c["pooled_mse_ratio"] = float(a.mean() / b.mean())
            out["contrasts"][lab] = c
            r = self.rnd(c)
            self.rows["handover_granularity"].append(base | {"contrast": lab, **{k: r[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct")},
                                                             "pooled_mse_ratio": round(c["pooled_mse_ratio"], 6)})
            return c
        for lab, a, b in HANDOVER_CONTRASTS:
            add(lab, hm[a], hm[b])
        for a in ARMS:
            for other in ("carrier alone", "TimesFM", "fixed division"):
                add(f"{a} vs {other}", hm[a], hm[other])
        for lab, a, b in HANDOVER_CONTRASTS[:2]:
            for w in range(4):
                add(f"{lab}, week {w + 1}", wk[a][:, w], wk[b][:, w])
        tot = out["contrasts"]["AW vs A0"]["log_ratio"]
        out["share_of_AW_gain"] = ({"Ah": out["contrasts"]["C4 Ah vs A0"]["log_ratio"] / tot, "AM": out["contrasts"]["AM vs A0"]["log_ratio"] / tot}
                                   if tot is not None and tot < 0 else None)
        out["alpha_week_mean"] = np.nanmean(al[ns], 0).tolist() if ns.any() else None
        out["alpha_month_mean"] = float(np.nanmean(am[ns])) if ns.any() else None
        out["pairs_mean"] = float(F["npairs"][ns].mean()) if ns.any() else None
        lw = {}
        for w, (a0, a1) in enumerate(WEEKS4):                                            # lead_weeks.csv
            row = base | {"week": w + 1, "days": f"{a0 + 1}-{a1}"}
            for first, fk in (("ankyra_lite", "AW"), ("fixed_division", "fixed division")):
                for second, sk in (("carrier", "carrier alone"), ("timesfm", "TimesFM")):
                    c = self.ctr(wk[fk][:, w], wk[sk][:, w], unit, month, 9)
                    lw[f"{first} vs {second}|week {w + 1}"] = c
                    r = self.rnd(c)
                    row |= {f"{first}_vs_{second}_log_ratio": r["log_ratio"], f"{first}_vs_{second}_um_low": r["um_low"],
                            f"{first}_vs_{second}_um_high": r["um_high"]}
            row["mean_weight_on_carrier"] = round(float(np.nanmean(al[ns, w])), 4) if ns.any() else ""
            self.rows["lead_weeks"].append(row)
        out["lead_weeks"] = lw
        out["mean_weight_on_carrier_by_week"] = out["alpha_week_mean"]
        self._lite, self._lite_e = prod["traj"], (prod["traj_pre"] if pre_arms is not None else None)
        abl = {}
        wl = "; with the continuation layer" if pre_arms is not None else ""
        self._rows_ablation(P, flag, {FIXED_DIVISION: ((fixed, pre_arms["fixed division"]) if pre_arms is not None else fixed,
                                                       "fixed-division daily means (historical level and daily path, no carrier candidate) plus the carrier's "
                                                       "within-day block, projected; seasonal naive on state windows; from the anchoring layer's own fields" + wl),
                                      HALF_HANDOVER: ((tr["Ah"], pre_arms["Ah"]) if pre_arms is not None else tr["Ah"],
                                                      "the carrier's daily means mixed in at a fixed weight of one half instead of the weekly weights; "
                                                      "everything else as ankyra-lite; seasonal naive on state windows" + wl)}, abl)
        out["ablation"] = abl
        return out

    def _combination(self, face, P, prod, ens, baselines, flag, F):
        """One whole-window combination weight (the first release's B2) and the half-and-half average (B1) of the history
        side H (fixed-division daily means plus the analog-day shape S0) and the carrier C at the origin; the same weight
        with three generic forecasters X in place of H, and equal-weight averages of the carrier with X.  The weight is the
        least-squares weight of C against the partner over the whole 744-hour trajectory at the unit's completed
        pseudo-origins (daily part: the handover pairs, 24 x the daily sums; within-day part: the trust triples; each part
        divided by its number of pseudo-origins), clipped to [0, 1] and shrunk to 1/2 by n/(n + 2), n = handover pairs."""
        y, unit, month = P["yL"], P["uL"], P["mL"]
        st, SN = prod["state"], prod["SN"]
        ns = ~st
        n = len(y)
        nd, nw = F["npairs"].astype(np.float64), F["ntrip"].astype(np.float64)

        def weight(s):                                                                    # s: (n, 4) daily num, den, within num, den
            num = np.where(nd > 0, 24 * s[:, 0] / np.maximum(nd, 1), 0.0) + np.where(nw > 0, s[:, 2] / np.maximum(nw, 1), 0.0)
            den = np.where(nd > 0, 24 * s[:, 1] / np.maximum(nd, 1), 0.0) + np.where(nw > 0, s[:, 3] / np.maximum(nw, 1), 0.0)
            with np.errstate(all="ignore"):
                lam = np.where(den > 0, np.clip(num / np.where(den > 0, den, 1.0), 0.0, 1.0), 0.5)
            return np.where(ns, (nd * lam + 1.0) / (nd + 2.0), np.nan)

        s = F["b2_sums"]
        sH = np.stack([s[:, 0], s[:, 1], s[:, 3] - s[:, 2], s[:, 3]], 1)                  # within-day: <y - S, C - S> = den - num of the trust sums
        aH = weight(sH)
        H = np.where(ns[:, None], np.repeat(np.where(ns[:, None], F["fixed"], 0.0), HR, axis=1) + np.where(ns[:, None], F["analog"], 0.0), 0.0)

        def comb(a, X):
            return np.where(st[:, None], SN, self._f32(np.maximum((1 - np.where(ns, a, 0.5))[:, None] * X + np.where(ns, a, 0.5)[:, None] * ens, 0.0)))
        B2, B1 = comb(aH, H), comb(np.full(n, 0.5), H)
        layer = prod.get("cont") is not None
        self._lite, self._lite_e = prod["traj"], (prod["traj_pre"] if layer else None)
        abl = {}
        if layer:                                                          # the ablation rows through the continuation layer, as ankyra-lite
            wl = "; with the continuation layer"
            B2c = np.where(st[:, None], SN, self.layered(prod, np.where(st[:, None], 0.0, B2), st))
            B1c = np.where(st[:, None], SN, self.layered(prod, np.where(st[:, None], 0.0, B1), st))
            v2, v1 = (B2c, B2), (B1c, B1)
        else:
            wl, v2, v1 = "", B2, B1
        self._rows_ablation(P, flag, {WHOLE_WINDOW: (v2, "one least-squares weight per window between the history side (fixed-division daily means plus the "
                                                         "analog-day shape) and the carrier, from the unit's handover pairs and trust triples, shrunk to 1/2; "
                                                         "seasonal naive on state windows" + wl),
                                      HALF_AVERAGE: (v1, "the history side and the carrier averaged with weight 1/2 each; seasonal naive on state windows" + wl)}, abl)
        out = {"ablation": abl, "weight_on_carrier": {"mean": float(np.nanmean(aH)), "quantiles_5_25_50_75_95": np.nanpercentile(aH, [5, 25, 50, 75, 95]).tolist()}}
        # generic partners
        gate = {}
        import lite_reanchor as LR
        for x in ("SN-week", "WeekProf4"):
            mine = np.stack([LR.x_forecast(x, c, CONTEXT) for c in P["ctxL"]])
            gate[x] = float((np.abs(mine - baselines[x]).max(1) / np.maximum(np.abs(baselines[x]).max(1), 1e-9)).max())
        out["generic_origin_reproduction_max_rel"] = gate
        cand = {"B2(H)": B2}
        for j, x in enumerate(GENERIC_X):
            cand[f"B2({x})"] = comb(weight(F["gen_sums"][:, j, :]), baselines[x])
        for x in EW_X:
            if x in baselines:
                cand[f"EW({x})"] = np.where(st[:, None], SN, np.maximum(0.5 * baselines[x] + 0.5 * ens, 0.0))
        L = {k: self.losses(y, v) for k, v in cand.items()}
        L["ankyra-lite"] = self.losses(y, prod["traj_pre"])                  # before the continuation layer, like the partners
        base = {"set": P["name"], "tier": P["tier"], "subset": "late", "windows": int(n), "units": int(len(np.unique(unit))), "availability_note": "", "flag": flag or ""}
        res = {}
        for err, off in (("hourly", 9), ("energy", 11)):
            got = {}
            for x in GENERIC_X:
                got[f"B2(H) vs B2({x})"] = ("", "", self.ctr(L["B2(H)"][err], L[f"B2({x})"][err], unit, month, off))
            b2x = [f"B2({x})" for x in GENERIC_X]
            ews = [k for k in cand if k.startswith("EW(")]
            for lab, first, pool in (("B2(H) vs best B2(X)", "B2(H)", b2x), ("B2(H) vs best equal-weight", "B2(H)", ews),
                                     ("ankyra-lite vs best generic", "ankyra-lite", ews + b2x)):
                cs = {k: self.ctr(L[first][err], L[k][err], unit, month, off) for k in pool}
                best = max(pool, key=lambda k: (cs[k]["log_ratio"] if cs[k]["log_ratio"] is not None else -np.inf))
                got[lab] = (best, "; ".join(pool), cs[best])
            for lab, (best, pool, c) in got.items():
                r = self.rnd(c)
                res[f"{lab}|{err}"] = c | {"best": best}
                self.rows["generic_combinations"].append(base | {"contrast": lab, "best": best, "candidates": pool, "error": err,
                                                                 **{k: r[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct")}, "resolved": c["resolved"]})
        out["generic"] = res
        out["generic_weights_mean"] = {x: float(np.nanmean(weight(F["gen_sums"][:, j, :]))) for j, x in enumerate(GENERIC_X)}
        return out

    def _within_trust(self, P, prod, flag, F):
        st = prod["state"]
        ns = ~st
        tr = F["trust"][:, 0]
        row = {"set": P["name"], "tier": P["tier"], "windows": int(len(st)), "state_windows": int(st.sum()), "anchored_windows": int(ns.sum()), "flag": flag or ""}
        if ns.any():
            row |= {"mean_trust": round(float(tr[ns].mean()), 4), "median_trust": round(float(np.median(tr[ns])), 4),
                    "share_trust_positive": round(float((tr[ns] > 0).mean()), 4), "share_analog_kept": round(float(F["kept"][ns].mean()), 4),
                    "mean_triples": round(float(F["ntrip"][ns].mean()), 3), "share_three_triples": round(float((F["ntrip"][ns] == 3).mean()), 4)}
        self.rows["within_day_trust"].append(row)
        return {k: v for k, v in row.items() if k not in ("set", "tier", "flag")}

    def _hit_rate(self, face, P, prod, flag, F):
        """P20: per window, is the level candidate with the largest weight (the smallest pseudo-origin error; ties excluded)
        the candidate whose window mean at the origin is closest to the realised window mean?  Seven candidates (six
        historical and the carrier's window mean) and the six historical alone; chance = the mean of 1 / (candidates
        available).  Descriptive."""
        y = P["yL"]
        ns = ~prod["state"]
        ybar = y.mean(1)
        out, rows = {}, []
        for lab, idx in (("seven (six historical and the carrier)", list(range(7))), ("six historical", list(range(6)))):
            w, c = F["lw"][:, idx], F["cand"][:, idx]
            avail = np.isfinite(c) & np.isfinite(w)
            ww = np.where(avail, w, -np.inf)
            top = ww.max(1)
            tied = (np.isclose(ww, top[:, None], rtol=0, atol=1e-12) & avail).sum(1) > 1
            sel = np.argmax(ww, 1)
            best = np.argmin(np.where(avail, np.abs(c - ybar[:, None]), np.inf), 1)
            use = ns & avail.any(1) & ~tied
            hit = (sel == best).astype(np.float64)
            chance = 1.0 / np.maximum(avail.sum(1), 1)
            self.hit_pool.append({"face": face, "set": lab, "unit": np.array([f"{face}:{u}" for u in P["uL"]])[use], "hit": hit[use], "chance": chance[use]})
            r = self._hit_row(P["name"], P["tier"], lab, P["uL"][use], hit[use], chance[use], int((ns & tied).sum()), flag)
            if 6 in idx:
                r["share_carrier_selected"] = round(float((sel[use] == 6).mean()), 4) if use.any() else ""
                r["share_carrier_best"] = round(float((best[use] == 6).mean()), 4) if use.any() else ""
            rows.append(r)
            out[lab] = {k: v for k, v in r.items() if k not in ("set", "tier", "candidates", "flag")}
        self.rows["level_candidate_hit_rate"] += rows
        return out

    @staticmethod
    def _hit_row(name, tier, lab, unit, hit, chance, tied, flag):
        r = {"set": name, "tier": tier, "candidates": lab, "windows": int(len(hit)), "units": int(len(np.unique(unit))), "windows_tied_excluded": tied,
             "share_carrier_selected": "", "share_carrier_best": "", "flag": flag or ""}
        if not len(hit):
            return r
        uu, inv = np.unique(unit, return_inverse=True)
        d = hit - chance
        su, cu = np.bincount(inv, d, len(uu)), np.bincount(inv, minlength=len(uu)).astype(np.float64)
        rng = np.random.default_rng(HIT_SEED)
        draws = []
        for _ in range(2000):
            k = rng.integers(0, len(uu), len(uu))
            draws.append(su[k].sum() / cu[k].sum())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        r |= {"hit_rate": round(float(hit.mean()), 4), "chance_rate": round(float(chance.mean()), 4), "hit_minus_chance": round(float(d.mean()), 4),
              "um_low": round(float(lo), 4), "um_high": round(float(hi), 4)}
        return r

    def _peak_carrier(self, P, prod, ens, flag):
        """peak_readout.csv row: ankyra-lite's peak readout against the same readout on the carrier alone (the three-seed
        carrier forecast at the origin, no anchoring layer, no state rule); fresh stream SEED + 13 as every peak row."""
        PK = self.T.PK
        y, ctx, unit, month = P["yL"], P["ctxL"], P["uL"], P["mL"]
        ypk = y.max(1)
        e_ref = np.abs(prod["peak"] - ypk)
        e_c = np.abs(PK.peak_readout_scalar(ens, ctx, P["ctL"], P["ttL"], 1.0) - ypk)
        with np.errstate(all="ignore"):
            c = CB.contrast_um(self.T, e_ref ** 2, e_c ** 2, unit, month, np.random.default_rng(self.T.DS.SEED + 13))
        ape_a, ape_p = CB.ape_mum(e_ref, ypk, unit), CB.ape_mum(np.abs(ctx[:, -HORIZON:].max(1) - ypk), ypk, unit)
        row = {"set": P["name"], "tier": P["tier"], "comparator": CARRIER_PEAK, "log_ratio": round(c["log_ratio"], 6), "um_low": round(c["UM"][0], 6),
               "um_high": round(c["UM"][1], 6), "improvement_pct": round(100 * (1 - math.exp(c["log_ratio"])), 3),
               "median_peak_ape_ankyra_pct": round(ape_a, 2), "median_peak_ape_previous_peak_pct": round(ape_p, 2), "flag": flag or ""}
        self.rows["peak_readout_added"].append(row)
        return {"log_ratio": float(c["log_ratio"]), "UM": [float(c["UM"][0]), float(c["UM"][1])], "median_peak_ape_carrier_readout_pct": CB.ape_mum(e_c, ypk, unit)}

    def _hit_pooled(self):
        """The hit rate pooled over the populations of the run (units of all populations resampled)."""
        rows = []
        for lab in ("seven (six historical and the carrier)", "six historical"):
            parts = [x for x in self.hit_pool if x["set"] == lab]
            if len(parts) < 2:
                continue
            name = "all nine populations" if sorted(x["face"] for x in parts) == sorted(CB.NINE) else "all populations of this run"
            rows.append(self._hit_row(name, "", lab, np.concatenate([x["unit"] for x in parts]), np.concatenate([x["hit"] for x in parts]),
                                      np.concatenate([x["chance"] for x in parts]), "", ""))
        self.rows["level_candidate_hit_rate"] += rows
        return rows

    def _summary_added(self, x, order, name, cell):
        """SUMMARY.md sections of the files added on 19 October 2026 (from the saved record only)."""
        import budget_tables as BT
        import long_context_tables as LC
        L = []
        R = {f: self.rec[f] for f in order}
        names = {f: name(f) for f in order}
        ho = [(f, r["handover"]) for f, r in R.items() if (r.get("handover") or {}).get("contrasts")]
        if ho:
            L += ["## What the handover contributes (`handover_granularity.csv`, `lead_weeks.csv`)", "",
                  "Four arms share the level with the carrier candidate, the delivered within-day block, the projection and the state rule; they",
                  "differ only in the weight on the carrier's daily means: none (A0), a fixed half (Ah), one weight per window from the handover",
                  "pairs (AM), the weekly weights (AW = ankyra-lite). Hourly improvement of the first-named arm in % (`*` resolved better, `(-)` resolved",
                  "worse); `share` = the part of AW's gain over A0 that the fixed half weight already gives (log-ratio share).", "",
                  "| Population | AW vs Ah | AW vs AM | Ah vs A0 | AW vs A0 | share (Ah) | ankyra-lite vs fixed division | mean weight on the carrier, weeks 1-4 |",
                  "|---|---|---|---|---|---|---|---|"]
            for f, h in ho:
                c = h["contrasts"]
                sh = h.get("share_of_AW_gain") or {}
                fd = (R[f].get("ablation") or {}).get(f"{FIXED_DIVISION}|hourly") or (h.get("ablation") or {}).get(f"{FIXED_DIVISION}|hourly")
                share_txt = f"{100 * sh['Ah']:.0f}%" if sh.get("Ah") is not None else "-"
                L.append(f"| {name(f)} | {cell(c['C1 AW vs Ah'])} | {cell(c['C2 AW vs AM'])} | {cell(c['C4 Ah vs A0'])} | {cell(c['AW vs A0'])} | "
                         f"{share_txt} | {cell(fd)} | "
                         f"{' / '.join(f'{v:.2f}' for v in (h.get('alpha_week_mean') or []))} |")
            L += ["", "By forecast week (`lead_weeks.csv`), hourly improvement of ankyra-lite and of the fixed division over the carrier alone:", "",
                  "| Population | ankyra-lite vs carrier, weeks 1-4 | fixed division vs carrier, weeks 1-4 |", "|---|---|---|"]
            for f, h in ho:
                lw = h.get("lead_weeks") or {}
                L.append(f"| {name(f)} | " + " ".join(cell(lw.get(f'ankyra_lite vs carrier|week {w}')) for w in range(1, 5)) + " | "
                         + " ".join(cell(lw.get(f'fixed_division vs carrier|week {w}')) for w in range(1, 5)) + " |")
            L += ["", "The arms are rebuilt from the anchoring layer's own fields (lite_reanchor.py, mode `fields`); AW reproduces the delivered",
                  "forecast bit for bit, and the history-side daily means before the handover are those of the layer's own level.", ""]
        wt = [(f, r["within_trust"]) for f, r in R.items() if (r.get("within_trust") or {}).get("mean_trust") is not None]
        if wt:
            L += ["## Within-day anchoring: where it acts (`within_day_trust.csv`)", "",
                  "| Population | Anchored windows | Mean trust on the analog shape | Windows with trust > 0 | Analog shape kept |", "|---|---:|---:|---:|---:|"]
            for f, w in wt:
                L.append(f"| {name(f)} | {w['anchored_windows']} | {w['mean_trust']:.3f} | {100 * w['share_trust_positive']:.0f}% | {100 * w['share_analog_kept']:.0f}% |")
            L += [""]
        cb = [(f, r["combination"]) for f, r in R.items() if (r.get("combination") or {}).get("ablation")]
        if cb:
            L += ["## Block-wise weights against one whole-window weight (`ablation.csv`, `component_contributions.csv`, `generic_combinations.csv`)", "",
                  "ankyra-lite against one whole-window combination weight of the same two forecasts (the history side: fixed-division daily",
                  "means plus the analog-day shape; the carrier at the origin; one least-squares weight per window from the unit's pseudo-origins,",
                  "shrunk to 1/2) and against their half-and-half average; improvement of ankyra-lite in %:", "",
                  "| Population | vs whole-window weight (hourly / energy) | vs half-and-half (hourly / energy) | mean weight on the carrier |", "|---|---|---|---:|"]
            for f, c in cb:
                a = c["ablation"]
                L.append(f"| {name(f)} | {cell(a.get(f'{WHOLE_WINDOW}|hourly'))} / {cell(a.get(f'{WHOLE_WINDOW}|energy'))} | "
                         f"{cell(a.get(f'{HALF_AVERAGE}|hourly'))} / {cell(a.get(f'{HALF_AVERAGE}|energy'))} | {c['weight_on_carrier']['mean']:.2f} |")
            gl = ("B2(H) vs B2(SN-week)", "B2(H) vs B2(WeekProf4)", "B2(H) vs B2(LYR-profile)", "B2(H) vs best B2(X)", "B2(H) vs best equal-weight",
                  "ankyra-lite vs best generic")
            L += ["", f"Generic partners (`generic_combinations.csv`), populations (of {len(cb)}) first-named resolved better / not resolved / resolved worse:", "",
                  "| Contrast | Hourly | Energy |", "|---|---|---|"]
            for lab in gl:
                cnt = []
                for err in ("hourly", "energy"):
                    v = [c["generic"].get(f"{lab}|{err}", {}).get("resolved") for _, c in cb]
                    cnt.append(f"{v.count('first better')} / {v.count('not resolved') + v.count('identical forecasts')} / {v.count('second better')}")
                L.append(f"| {lab} | {cnt[0]} | {cnt[1]} |")
            L += ["", "B2(H) is the whole-window weight with the history side, B2(X) the same with a generic forecaster X in its place (X computed",
                  "from the load at every pseudo-origin), EW(X) the equal-weight average of the carrier and X; `best` was picked after scoring.", ""]
        hr = [(f, r["hit_rate"]) for f, r in R.items() if r.get("hit_rate")]
        if hr:
            L += ["## Level candidates: hit rate of the pseudo-selected candidate (`level_candidate_hit_rate.csv`)", "",
                  "| Population | Seven candidates: hit rate / chance | Six historical: hit rate / chance |", "|---|---|---|"]
            for f, h in hr:
                a, b = h.get("seven (six historical and the carrier)", {}), h.get("six historical", {})
                L.append(f"| {name(f)} | {a.get('hit_rate', float('nan')):.3f} / {a.get('chance_rate', float('nan')):.3f} | "
                         f"{b.get('hit_rate', float('nan')):.3f} / {b.get('chance_rate', float('nan')):.3f} |")
            for r in x.get("level_candidate_hit_rate_pooled") or []:
                L.append(f"| {r['set']} ({r['candidates']}) | {r['hit_rate']:.3f} / {r['chance_rate']:.3f} (difference {r['hit_minus_chance']:+.3f}, "
                         f"95% unit bootstrap {r['um_low']:+.3f} to {r['um_high']:+.3f}) | |")
            L += ["", "Descriptive (P20): the candidate with the largest weight (smallest pseudo-origin error) against the candidate whose window",
                  "mean at the origin is closest to the realised mean; windows with tied weights (no evidence) are left out.", ""]
        pc = [(f, r["peak_carrier"]) for f, r in R.items() if r.get("peak_carrier")]
        if pc:
            L += ["Peak readout of ankyra-lite against the same readout on the carrier alone (`peak_readout.csv`, comparator `" + CARRIER_PEAK + "`): "
                  + "; ".join(f"{name(f)} {cell({'improvement_pct': 100 * (1 - math.exp(p['log_ratio'])), 'log_ratio': p['log_ratio'], 'um_low': p['UM'][0], 'um_high': p['UM'][1]})}"
                              for f, p in pc) + ".", ""]
        L += BT.summary(x.get("budget"), names)
        L += LC.summary(x.get("long_context"), names)
        return L

    def write_added(self, out_dir, rec, parts=None):
        """Write the files added on 19 October 2026 and return (written, files, record).  ``parts=("budget",)``: the
        training budget file only."""
        import budget_tables as BT
        import long_context_tables as LC
        written, files = [], {}
        only_budget = parts is not None and set(parts) <= {"budget"}
        hp = [] if only_budget else self._hit_pooled()
        if not only_budget:
            self.rows["energy_level_readout"] = energy_level_rows(rec, self.T)
        specs = {} if only_budget else dict(ADDED_FILES)
        specs[BT.FILE] = ("training_budget_sensitivity", BT.FIELDS, BT.DESCRIPTION)
        if not only_budget:
            specs[LC.FILE] = ("long_context_baselines", LC.FIELDS, LC.DESCRIPTION)
        for name, (key, fields, desc) in specs.items():
            rows = self.rows.get(key)
            if not rows:
                continue
            CB.write_rows(name, rows, fields)
            written.append(name)
            files[name] = desc
        x = {"budget": {"populations": {f: r["budget"] for f, r in self.rec.items() if r.get("budget")}}}
        if not only_budget:
            x |= {"long_context": {"populations": {f: r["long_context"] for f, r in self.rec.items() if r.get("long_context")}},
                  "level_candidate_hit_rate_pooled": [{k: v for k, v in r.items() if k not in ("flag",)} for r in hp]}
        if x["budget"]["populations"]:
            x["budget"]["counts"] = BT.counts(x["budget"]["populations"])
        return written, files, x

    # ------------------------------------------------------------------ after the last population
    def finish(self, out_dir, rec, cost=True, robustness=True, robustness_face=None, robustness_windows=None, cost_record=None):
        written, files = [], {}
        hist_pooled = self._history_pooled(rec)
        dec = self._deciles()
        cont_pooled = self._continuation_pooled()
        cost_rec = rob_rec = None
        if cost and cost_record:                                         # a saved timing record (cost_timing.py --out)
            import cost_timing as CT
            cost_rec, crows = CT.load_record(cost_record, self.prods)
            self.rows["cost_per_window"] = crows
        elif cost:
            import cost_timing as CT
            face = "oslo" if "oslo" in self.faces else (self.faces[0] if self.faces else None)
            if face:
                CB.wait_for_memory("cost timing")
                cost_rec, crows = CT.measure(self.T, face, self.prods["faces"][face])
                self.rows["cost_per_window"] = crows
        if robustness:
            import robustness as RB
            face = robustness_face or next((f for f in ("drammen", "oslo") if f in self.faces), self.faces[0] if self.faces else None)
            if face:
                CB.wait_for_memory("robustness")
                rob_rec, rrows = RB.measure(self.T, face, self.prods["faces"][face], n=robustness_windows or RB.N, log=CB.log)
                self.rows["robustness"] = rrows
        import recipe_tables as RT
        rt_written, rt_rec, rt_files = RT.write(out_dir, log=CB.log, prods=self.prods)
        written += rt_written
        files.update(rt_files)
        specs = {
            "datasets.csv": ("datasets", ["set", "tier", "country", "population", "ankyra_design_status", "units", "windows", "units_late", "windows_late",
                                          "development_windows", "store", "first_training_origin", "validation_start", "training_cut", "test_start",
                                          "first_test_origin", "last_test_origin", "test_windows", "test_units", "record_days_before_cut",
                                          "validation_share_pct", "test_origin_months", "test_months_spanned", "gap_days_cut_to_test_start", "calendar",
                                          "own_training_windows", "returned_validation_windows", "returned_label_end", "continuation_epochs"],
                             "populations, the training, validation and test split, and the returned validation span"),
            "robustness.csv": ("robustness", None, "ankyra-lite and the carrier alone on corrupted contexts (CPU re-forecast with the released checkpoints)"),
            "contrast_ii_blocks.csv": ("contrast_ii_blocks", ["set", "tier", "subset", "analysis", "windows", "units", "carrier", "contrast", "block", "log_ratio",
                                                              "um_low", "um_high", "improvement_pct", "resolved", "flag"],
                                       "ankyra-lite against the history-only configuration and the carrier alone, by error block"),
            "contrast_ii_control.csv": ("contrast_ii_control", ["set", "tier", "subset", "analysis", "windows", "units", "carrier", "contrast", "block", "log_ratio",
                                                                "um_low", "um_high", "improvement_pct", "resolved", "flag"],
                                        "ankyra-lite against the carrier-free control (the same procedure without a carrier), by error block"),
            "example_window_cambridge.csv": ("example_window_cambridge", ["hour", "load_kw", "timesfm_kw", "ankyra_kw", "carrier_kw"],
                                             "one Cambridge test window: load, TimesFM, ankyra-lite (`ankyra_kw`) and the carrier alone"),
            "cost_per_window.csv": ("cost_per_window", ["item", "value", "unit"], "parameters, device and CPU seconds per window (one thread)"),
            "intervals_by_population.csv": ("intervals_by_population", None, "the release interval centred on ankyra-lite and the native quantiles of TimesFM and Chronos-2 (test windows)"),
            "intervals_winkler_contrasts.csv": ("intervals_winkler_contrasts", ["version", "set", "comparator", "log_ratio", "um_low", "um_high", "improvement_pct", "resolved", "flag"],
                                                "unit-equal Winkler contrasts of the ankyra-lite interval against the native quantiles"),
            "ablation.csv": ("ablation", ["set", "tier", "ablation", "log_ratio", "um_low", "um_high", "improvement_pct", "error", "windows", "units", "resolved", "note", "flag"],
                             "ankyra-lite against reduced and alternative versions of itself (hourly and energy)"),
            "component_contributions.csv": ("component_contributions", ["set", "component", "improvement_pct", "log_ratio", "um_low", "um_high", "kind", "error", "resolved", "flag"],
                                            "what each part of ankyra-lite adds (direct contrasts)"),
            "seed_spread.csv": ("seed_spread", ["set", "tier", "unit_set", "stage", "seed", "error", "comparator", "windows", "units", "log_ratio", "um_low", "um_high", "improvement_pct", "resolved", "flag"],
                                "each seed's carrier alone (and, with six seeds, each anchored three-seed group) against the seed ensemble, and their contrasts against TimesFM and Chronos-2-X"),
            "carriers_in_anchoring_layer.csv": ("carriers_in_anchoring_layer", ["set", "tier", "subset", "windows", "units", "carrier", "contrast", "error", "log_ratio", "um_low", "um_high",
                                                                                "improvement_pct", "resolved", "flag"],
                                                "ankyra-lite against other trained carriers inside the same anchoring layer"),
            "history_length.csv": ("history_length", ["set", "stratum", "windows", "units", "comparator", "log_ratio", "um_low", "um_high", "improvement_pct", "resolved", "flag"],
                                   "ankyra-lite against the carrier alone, TimesFM and Chronos-2-X by the history length at the origin"),
            "bdg2_near_zero_sensitivity.csv": ("bdg2_near_zero_sensitivity", ["unit_set", "subset", "model", "log_ratio", "um_low", "um_high", "improvement_pct", "ankyra_mean_unit_rank"],
                                               "the BDG2 hourly contrasts with and without the three near-zero meters"),
            "anchoring_gain_deciles.csv": ("anchoring_gain_deciles", ["statistic", "decile", "lower_edge", "upper_edge", "windows", "populations", "level_improvement_pct", "level_um_low",
                                                                      "level_um_high", "hourly_improvement_pct", "hourly_um_low", "hourly_um_high"],
                                           "the anchoring record against the realised gain of ankyra-lite over its carrier alone (descriptive)"),
            "training_curves.csv": ("training_curves", ["set", "tier", "seed", "stage", "epoch", "training_loss", "validation_criterion", "validation_held_out",
                                                        "best_epoch", "epochs_run", "selected", "returned_span_draws", "flag"],
                                    "both training stages per population and seed: validation criterion (and training loss) by epoch, best epoch"),
        }
        if self.rows.get("continuation_layer"):                          # version 0.2.0: the continuation files
            specs["continuation_layer.csv"] = ("continuation_layer", CONT_LAYER_FIELDS,
                                               "the origin-continuity layer on against off on the same carrier, by lead time (hourly) and the delivered energy")
            specs["continuation_record.csv"] = ("continuation_record", CONT_RECORD_FIELDS,
                                                "what the layer did on the test windows, and the delivered mean against the energy readout (readout rule A)")
        if any("ankyra_before_layer_kw" in r for r in self.rows.get("example_window_cambridge", [])):
            k, f, d = specs["example_window_cambridge.csv"]
            specs["example_window_cambridge.csv"] = (k, f + ["ankyra_before_layer_kw"], d + " and ankyra-lite before the continuation layer")
        if any("carrier_seeds" in r for r in self.rows.get("datasets", [])):
            k, f, d = specs["datasets.csv"]
            specs["datasets.csv"] = (k, f + ["carrier_seeds"], d + ", and the carrier seeds")
        if dec is not None:
            self.rows["anchoring_gain_deciles"] = dec["rows"]
        for name, (key, fields, desc) in specs.items():
            rows = self.rows.get(key)
            if not rows:
                continue
            if key == "robustness":
                import robustness as RB
                fields = RB.FIELDS
            elif fields is None:                                            # the release's order: five identifiers, then alphabetical
                fields = sorted({k for r in rows for k in r} - {"flag"}, key=lambda k: (k not in ("set", "tier", "arm", "windows", "units"), k)) + ["flag"]
            CB.write_rows(name, rows, fields)
            written.append(name)
            files[name] = desc
        for name, obj in self.extra_json.items():
            CB.dump_json(out_dir / name, obj)
            written.append(name)
            files[name] = {"intervals_households.json": "the interval on GoiEner households: four arms and contrasts",
                           "example_window_cambridge.json": "the building, period, day types and source of the example window"}[name]
        w2, f2, x2 = self.write_added(out_dir, rec)
        written += w2
        files.update(f2)
        rec_out = {"populations": self.rec, "history_length_pooled": hist_pooled, "anchoring_gain_deciles": dec and dec["rec"],
                   "cost": cost_rec, "robustness": rob_rec, **rt_rec, **x2, "notes": dict(self.notes), "absent_files": ABSENT}
        if cont_pooled is not None:
            rec_out["continuation_pooled"] = cont_pooled
        return written, rec_out, self.summary(rec, rec_out), files

    def _history_pooled(self, rec):
        if len(self.hist_pool) < 2:
            return None
        name = "all nine populations" if sorted(x["face"] for x in self.hist_pool) == sorted(CB.NINE) else "all populations of this run"
        E = {k: np.concatenate([x["e"][k] for x in self.hist_pool]) for k in self.hist_pool[0]["e"]}
        U = np.concatenate([x["unit"] for x in self.hist_pool])
        M = np.concatenate([x["month"] for x in self.hist_pool])
        m = np.concatenate([x["m"] for x in self.hist_pool])
        keff = np.concatenate([x["k"] for x in self.hist_pool])
        out = {"set": name, "strata": {}}
        for lab, lo, hi in STRATA_M + STRATA_K:
            sel = ((keff >= lo) & (keff <= hi)) if lab.startswith("K_eff") else ((m >= lo) & (m < hi))
            s = {"windows": int(sel.sum()), "units": int(len(np.unique(U[sel])))}
            if sel.sum() >= 10:
                for other in ("carrier alone", "TimesFM", "Chronos-2-X"):
                    c = self._hctr(E["ankyra-lite"][sel], E[other][sel], U[sel], M[sel])
                    s[other] = c
                    self.rows["history_length"].append({"set": name, "stratum": lab, "windows": s["windows"], "units": s["units"], "comparator": other,
                                                        **{k: v for k, v in self.rnd(c).items() if k in ("log_ratio", "um_low", "um_high", "improvement_pct")},
                                                        "resolved": c["resolved"], "flag": ""})
            out["strata"][lab] = s
        return out

    def _deciles(self):
        if not self.dec_pool:
            return None
        T = self.T
        cat = lambda k: np.concatenate([x[k] for x in self.dec_pool])                           # noqa: E731
        faces = np.concatenate([[x["face"]] * len(x["unit"]) for x in self.dec_pool])
        unit, month, S = cat("unit"), cat("month"), cat("stats")
        eA, eC, mA, mC = cat("eA"), cat("eC"), cat("mA"), cat("mC")
        # "windows": the pooled windows (units with a mean load above 1e-6 kW); "windows_with_statistic": those with a finite
        # value of the statistic, which are the ones the deciles hold
        rows, rec = [], {"windows": int(len(unit)), "windows_with_statistic": {}, "populations": sorted(set(faces.tolist())), "deciles": {}}
        for j, s in enumerate(("carrier_level_weight", "history_vs_carrier_log_ratio")):
            v = S[:, j]
            ok = np.isfinite(v)
            rec["windows_with_statistic"][s] = int(ok.sum())
            if ok.sum() < 20:
                continue
            q = np.nanquantile(v[ok], np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(q[1:-1], v[ok], side="right"), 0, 9)
            idx = np.flatnonzero(ok)
            rec["deciles"][s] = []
            for dd in range(10):
                m = idx[dec == dd]
                if len(m) < 5:
                    continue
                cl = self._dctr(eA[m], eC[m], unit[m], month[m])
                ch = self._dctr(mA[m], mC[m], unit[m], month[m])
                rows.append({"statistic": s, "decile": dd + 1, "lower_edge": f"{q[dd]:.4f}", "upper_edge": f"{q[dd + 1]:.4f}", "windows": len(m),
                             "populations": len(set(faces[m])), "level_improvement_pct": f"{cl['improvement_pct']:.2f}", "level_um_low": f"{cl['um_low']:.4f}",
                             "level_um_high": f"{cl['um_high']:.4f}", "hourly_improvement_pct": f"{ch['improvement_pct']:.2f}", "hourly_um_low": f"{ch['um_low']:.4f}",
                             "hourly_um_high": f"{ch['um_high']:.4f}"})
                rec["deciles"][s].append({"decile": dd + 1, "edges": [float(q[dd]), float(q[dd + 1])], "windows": int(len(m)), "level": cl, "hourly": ch})
        return {"rows": rows, "rec": rec}

    def _dctr(self, a, b, unit, month):
        T = self.T
        m = np.isfinite(a) & np.isfinite(b)
        with np.errstate(all="ignore"):
            r = T.DS.log_ratio(a[m], b[m], unit[m])
            iv = T.DS.boot(a[m], b[m], unit[m], month[m], np.random.default_rng(T.DS.SEED))["UM"]
        return {"log_ratio": float(r), "um_low": float(iv[0]), "um_high": float(iv[1]), "improvement_pct": 100 * (1 - math.exp(r)), "windows": int(m.sum())}

    # ------------------------------------------------------------------ summary sections
    def summary(self, rec, x):
        pops = rec["populations"]
        order = [f for f in ("bdg2", "goiener_confirm", "households", "cambridge", "heew", "eweld", "oslo", "drammen", "cineldi") if f in self.rec]
        name = lambda f: pops[f]["name"]                                                        # noqa: E731

        def cell(c):
            if not c or c.get("log_ratio") is None:
                return "-"
            if c.get("resolved") == "identical forecasts":
                return "0 (identical)"
            return f"{c['improvement_pct']:+.1f}{' *' if c['um_high'] < 0 else (' (-)' if c['um_low'] > 0 else '')}"

        L = ["## Data splits", "",
             "| Population | Development windows | First training origin | Validation set from | Training cut | Test set (first ... last origin) | Test windows / units | Origin months | Days of training origins before the cut | Validation share | Returned validation windows |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
        for f in order:
            d = self.rec[f]["datasets"]
            L.append(f"| {name(f)} | {d['development_windows']} | {d['first_training_origin']} | {d['validation_start']} | {d['training_cut']} | "
                     f"{d['first_test_origin']} ... {d['last_test_origin']} | {d['test_windows']} / {d['test_units']} | {d['test_origin_months']} | "
                     f"{d['record_days_before_cut']} | {d['validation_share_pct']}% | {d.get('returned_validation_windows', '')} |")
        L += ["", "True calendar dates (`datasets.csv`, column `calendar`). The test set starts 186 days after the training cut, so that",
              "the carrier's forecasts at all six pseudo-origins are issued after the cut. The validation set is held out in the",
              "from-scratch stage only; the span return puts it back (`returned_validation_windows`: the population's own validation",
              "windows returned at its own sampling density, labels ending at least 24 h before the cut).", ""]
        if any(self.rec[f]["training_curves"] for f in order):
            cont = any("continuation_epochs" in v for f in order for v in self.rec[f]["training_curves"].values())
            sd = sorted({int(k) for f in order for k in self.rec[f]["training_curves"]})
            sdl = ", ".join(str(k) for k in sd) if len(sd) <= 3 else f"{sd[0]}-{sd[-1]}"
            L += ["## Training (`training_curves.csv`)", "",
                  f"| Population | From-scratch stage: best epoch (seeds {sdl}) | Epochs run | "
                  + ("Span return: epochs | Returned windows | Start criterion reproduced |" if cont else ""),
                  "|---|---|---|" + ("---|---|---|" if cont else "")]
            for f in order:
                tc = self.rec[f]["training_curves"]
                row = f"| {name(f)} | {', '.join(str(tc[s]['best_epoch']) for s in sorted(tc))} | {', '.join(str(tc[s]['epochs_run']) for s in sorted(tc))} |"
                if cont:
                    row += (f" {', '.join(str(tc[s].get('continuation_epochs', '-')) for s in sorted(tc))} |"
                            f" {', '.join(str(tc[s].get('returned_span_windows', '-')) for s in sorted(tc))} |"
                            f" {all(tc[s].get('start_criterion_reproduced') for s in tc)} |")
                L.append(row)
            L += ["", "The from-scratch stage records the validation criterion of every epoch, not the training loss (`training_loss` is",
                  "blank there). The span-return epochs record both; their validation criterion is computed on windows that are back in",
                  "the training set, so it is a description, not a held-out score. Epoch 0 of the span return is the selected from-scratch",
                  "weights (its criterion reproduces the best from-scratch value).", ""]
        if x.get("recipe_history"):
            import recipe_tables as RT
            L += RT.summary(x)
        sc = [(f, self.rec[f].get("study_scorer_crosscheck")) for f in order if self.rec[f].get("study_scorer_crosscheck")]
        if sc:
            L += [f"The product reading reproduces the study's own scorer on {sum(1 for _, v in sc if v['pass'])} of {len(sc)} populations: the "
                  "per-population contrasts against the plain continuation and against the from-scratch stage equal those of the score",
                  f"record (largest difference {max(v['max_abs_diff_pct_points'] for _, v in sc):.1e} percentage points).", ""]
        abl = [(f, self.rec[f]["ablation"]) for f in order]
        labs = []
        for _, a in abl:
            for k in a:
                lab = k.rsplit("|", 1)[0]
                if lab not in labs:
                    labs.append(lab)
        L += ["## Ablation (`ablation.csv`, `component_contributions.csv`)", "",
              "ankyra-lite against versions of itself, hourly error, improvement of ankyra-lite in % (`*` resolved better,",
              "`(-)` resolved worse). `-` = not available in this run.", "",
              "| Version | " + " | ".join(name(f) for f in order) + " |", "|---|" + "---|" * len(order)]
        for lab in labs:
            L.append(f"| {lab} | " + " | ".join(cell(a.get(f"{lab}|hourly")) for _, a in abl) + " |")
        L += ["", "The rows of switched-off parts of the anchoring layer need re-anchoring (`evaluation/lite_reanchor.py`); the single",
              "seeds are the stored single-seed products; the history-only configuration, the recipe arms and the earlier carriers are",
              "saved study forecasts. The earlier carriers (without the copy pointer, the earlier continuation study, the structured",
              "network) were trained with the from-scratch recipe or its earlier continuations, not with the span return.", ""]
        members = {}
        for f in order:
            for k, v in self.rec[f]["seed_spread"].items():
                if k.endswith("|spread|TimesFM"):
                    members[k] = v.get("members") or [f"seed {j}" for j in range(3)]
        generic = any(m != ["seed 0", "seed 1", "seed 2"] for m in members.values())
        if not generic:
            L += ["## Seeds (`seed_spread.csv`)", "", "Hourly improvement against TimesFM, per seed and for the three-seed ensemble (carrier alone, and anchored):", "",
                  "| Population | Stage | Seed 0 | Seed 1 | Seed 2 | Ensemble |", "|---|---|---|---|---|---|"]
        else:                                                              # six seeds: the carrier per seed, anchored per three-seed group
            nsd = len(self.prods.get("seeds", (0, 1, 2))) if hasattr(self, "prods") else 6
            L += ["## Seeds (`seed_spread.csv`)", "",
                  f"Hourly improvement against TimesFM: the carrier alone per seed and the {n_words(nsd)}-seed ensemble, and anchored per three-seed group",
                  "(the stored group products) and the ensemble:", "",
                  "| Population | Stage | Members (in order) | Ensemble |", "|---|---|---|---|"]
        for f in order:
            s = self.rec[f]["seed_spread"]
            for pre, lab in (("", ""), ("without the three near-zero meters|", " (without the three near-zero meters)")):
                for stage in ("carrier alone", "anchored (ankyra-lite)"):
                    if f"{pre}{stage}|ensemble|TimesFM" in s:
                        if not generic:
                            L.append(f"| {name(f)}{lab} | {stage} | " + " | ".join(cell(s.get(f"{pre}{stage}|seed {k}|TimesFM")) for k in (0, 1, 2))
                                     + f" | {cell(s[f'{pre}{stage}|ensemble|TimesFM'])} |")
                        else:
                            ms = members.get(f"{pre}{stage}|spread|TimesFM", [])
                            L.append(f"| {name(f)}{lab} | {stage} | " + " / ".join(f"{m}: {cell(s.get(f'{pre}{stage}|{m}|TimesFM'))}" for m in ms)
                                     + f" | {cell(s[f'{pre}{stage}|ensemble|TimesFM'])} |")
        L += ["", "The carrier alone has no state rule: on the three near-zero BDG2 meters its error is not zero while some baselines' errors",
              "are near zero, which dominates the unit-equal mean (hence the BDG2 row without those meters).", "",
              "## Prediction intervals (`intervals_by_population.csv`, `intervals_winkler_contrasts.csv`, `intervals_households.json`)", "",
              "The release interval (residual quantiles of the history-only estimator at the unit's pseudo-origins, independent of the",
              "carrier) centred on ankyra-lite, against the native quantiles of TimesFM and Chronos-2; test windows. Descriptive: the",
              "interval was not designed for this carrier.", "",
              "| Population | 80% coverage (ankyra-lite / TimesFM / Chronos-2) | 90% coverage (ankyra-lite / Chronos-2) | Winkler 80 vs TimesFM | vs Chronos-2 |", "|---|---|---|---|---|"]
        for f in order:
            iv = self.rec[f]["intervals"]
            if not iv:
                continue
            a = iv["arms"]
            L.append(f"| {name(f)} | {a['ankyra_lite']['cov80']:.3f} / {a['timesfm_native']['cov80']:.3f} / {a['chronos_native']['cov80']:.3f} | "
                     f"{a['ankyra_lite'].get('cov90', float('nan')):.3f} / {a['chronos_native'].get('cov90', float('nan')):.3f} | "
                     f"{cell(iv['winkler80_ankyra_lite_vs']['timesfm_native'])} | {cell(iv['winkler80_ankyra_lite_vs']['chronos_native'])} |")
        L += ["", "## Trained carriers inside the anchoring layer (`carriers_in_anchoring_layer.csv`)", "",
              "Hourly improvement in %: ankyra-lite against the same anchoring layer with each other trained carrier (`ankyra-lite vs",
              "anchored carrier`), and each carrier's anchored forecast against the carrier alone (contrast (i)).", ""]
        heads = [LITE_CARRIER] + list(DEEP_CARRIERS)
        L += ["| Population | " + " | ".join(f"vs anchored {X}" for X in DEEP_CARRIERS) + " | "
              + " | ".join(f"(i) {X if X != LITE_CARRIER else 'small net'}" for X in heads) + " |",
              "|---|" + "---|" * (len(DEEP_CARRIERS) + len(heads))]
        for f in order:
            c = self.rec[f]["carriers"]
            L.append(f"| {name(f)} | " + " | ".join(cell(c.get(f"{X}|ankyra-lite vs anchored carrier|hourly")) for X in DEEP_CARRIERS) + " | "
                     + " | ".join(cell(c.get(f"{X}|anchored vs carrier alone|hourly")) for X in heads) + " |")
        L += ["", "The other carriers were trained on their own population only, with the same training cut as the small network.",
              "On state windows the anchoring layer returns those carriers' own forecasts, whereas ankyra-lite delivers the seasonal naive.", ""]
        ci = [(f, self.rec[f].get("contrast_ii")) for f in order if self.rec[f].get("contrast_ii")]
        if ci:
            def cnt(key):
                v = [c.get(key) for _, c in ci]
                v = [z["resolved"] for z in v if z]
                return f"{v.count('first better')} / {v.count('not resolved') + v.count('identical forecasts')} / {v.count('second better')}"
            L += ["## Contrast (ii) by error block (`contrast_ii_blocks.csv`, `contrast_ii_control.csv`)", "",
                  f"Populations (of {len(ci)}) where the first-named forecast is resolved better / not resolved / resolved worse, all test windows:", "",
                  "| Contrast | Hourly | Level | Daily path | Within-day |", "|---|---|---|---|---|"]
            for table, contrast in (("contrast_ii_blocks", "anchored vs history only"), ("contrast_ii_blocks", "anchored vs carrier alone"),
                                    ("contrast_ii_blocks", "carrier alone vs history only"), ("contrast_ii_control", "anchored vs carrier-free control"),
                                    ("contrast_ii_control", "carrier-free control vs history only")):
                if not any(f"{table}|{contrast}|hourly|all late windows" in c for _, c in ci):
                    continue
                L.append(f"| {contrast} | " + " | ".join(cnt(f"{table}|{contrast}|{b}|all late windows") for b in BLOCKS) + " |")
            L += ["", "`anchored` is ankyra-lite. The history-only configuration and the carrier-free control (the anchoring layer called",
                  "with the history-only forecasts in the carrier's place) are saved forecasts of the same anchoring layer without a",
                  "carrier. On state windows ankyra-lite delivers the seasonal naive; the rows `" + STATE_EXCLUDED + "` leave those",
                  "windows out.", ""]
        if x.get("history_length_pooled"):
            hp = x["history_length_pooled"]
            L += [f"## History length at the origin (`history_length.csv`), {hp['set']}", "",
                  "| Stratum | Windows | vs carrier alone | vs TimesFM | vs Chronos-2-X |", "|---|---:|---|---|---|"]
            for lab, s in hp["strata"].items():
                L.append(f"| {lab} | {s['windows']} | {cell(s.get('carrier alone'))} | {cell(s.get('TimesFM'))} | {cell(s.get('Chronos-2-X'))} |")
            L += [""]
        if "bdg2" in self.rec and self.rec["bdg2"].get("near_zero"):
            nz = self.rec["bdg2"]["near_zero"]
            w = nz["without the three near-zero meters"]
            L += ["## BDG2 near-zero meters (`bdg2_near_zero_sensitivity.csv`)", "",
                  f"Mean unit rank of ankyra-lite: {nz['all units']['mean_unit_rank']:.3f} on all units ({nz['all units']['units']} units), "
                  f"{w['mean_unit_rank']:.3f} without the three near-zero meters ({w['units']} units; rank {w['rank']} of {w['of']}).", ""]
        if x.get("anchoring_gain_deciles") and x["anchoring_gain_deciles"]["deciles"]:
            d = x["anchoring_gain_deciles"]
            held = {s: sum(r["windows"] for r in rows) for s, rows in d["deciles"].items()}       # the windows the deciles hold
            n_dec = (f"{next(iter(held.values()))} windows" if len(set(held.values())) == 1 else
                     ", ".join(f"{n} windows ({s})" for s, n in held.items()))
            L += ["## Anchoring record (`anchoring_gain_deciles.csv`)", "",
                  "The two statistics of `anchoring_record` against the realised gain of ankyra-lite over its carrier alone (descriptive; the "
                  f"carrier weight is the anchoring layer's own weight). The deciles hold {n_dec} of {len(d['populations'])} "
                  f"population{'s' if len(d['populations']) != 1 else ''}: of the {d['windows']} pooled windows (units with a mean load above 1e-6 kW), "
                  "those with a finite statistic.", "",
                  "| Statistic | Level gain by decile (%) | Hourly gain by decile (%) |", "|---|---|---|"]
            for s, rows in d["deciles"].items():
                L.append(f"| `{s}` | " + " ".join(f"{r['level']['improvement_pct']:+.0f}" for r in rows) + " | "
                         + " ".join(f"{r['hourly']['improvement_pct']:+.0f}" for r in rows) + " |")
            L += [""]
        if x.get("cost"):
            c = x["cost"]
            L += ["## Cost (`cost_per_window.csv`)", "",
                  f"Measured on {c['windows']} {c['face_name']} test windows, CPU, one thread: carrier forecasts at 7 issue times x {c.get('seeds', 3)} seeds "
                  f"{c['carrier_s_per_window']:.3f} s/window, anchoring layer {c['anchor_s_per_window']:.3f} s/window, total {c['total_s_per_window']:.3f} s/window. "
                  f"The carrier forecasts match the stored evaluation forecasts to a relative difference of {c['carrier_vs_stored_max_rel']:.1e}.", ""]
        if x.get("robustness"):
            import robustness as RB
            L += RB.summary(x["robustness"])
        if x.get("constants"):
            L += ["## Constants (`constants.csv`)", "",
                  f"{x['constants']['carrier_rows']} carrier rows (training recipe, span return, network; read from the package defaults) and "
                  f"{x['constants']['anchoring_rows']} anchoring-layer rows (group, constant, value and where_fixed as in the first release; the layer",
                  "is a byte-identical copy). The `sensitivity` column holds the study's recipe readouts where a recipe constant was varied.", ""]
        cp = [(f, self.rec[f]["continuation"]) for f in order if self.rec[f].get("continuation")]
        if cp:
            pooled = x.get("continuation_pooled")
            heads = [lab for lab, _, _ in CONT_SEGMENTS]
            L += ["## Continuation layer (`continuation_layer.csv`, `continuation_record.csv`)", "",
                  "The origin-continuity layer on against off on the same carrier and anchoring layer (ankyra-lite after against before the",
                  "layer, seasonal naive on state windows in both), hourly error by lead time, improvement in % (`*` resolved better, `(-)`",
                  "resolved worse; unit x month interval, no seed layer: the same carrier on both sides):", "",
                  "| Population | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
            for f, c in cp:
                L.append(f"| {name(f)} | " + " | ".join(cell(c["segments"].get(f"{h}|hourly")) for h in heads) + " |")
            if pooled:
                L.append(f"| {pooled['set']} | " + " | ".join(cell(pooled["segments"].get(h)) for h in heads) + " |")
            L += ["", "Readout rule A: energy, level and peak are read before the layer, so the delivered trajectory's mean differs from the",
                  "energy readout (744 x level). Relative difference on the windows the layer acts on, and what the layer did:", "",
                  "| Population | Delivered mean vs energy readout, p50 / p99 / max | State windows bypassed | Clipped spans (fast / day) | No-match reference day | max abs Delta |",
                  "|---|---|---:|---|---:|---:|"]
            recs = [(name(f), c["record"]) for f, c in cp] + ([(pooled["set"], pooled["record"])] if pooled else [])
            for nm, r in recs:
                pct3 = " / ".join(f"{100 * float(r[k]):.2f}%" if r[k] != "" else "-" for k in ("delivered_mean_vs_energy_readout_rel_p50",
                                                                                                 "delivered_mean_vs_energy_readout_rel_p99",
                                                                                                 "delivered_mean_vs_energy_readout_rel_max"))
                cl = " / ".join(f"{100 * float(r[k]):.1f}%" if r[k] != "" else "-" for k in ("clipped_fast_share", "clipped_day_share"))
                L.append(f"| {nm} | {pct3} | {r['state_windows_bypassed']} | {cl} | {r['no_match_reference_day']} | {float(r['max_abs_delta']):.3f} |")
            L += [""]
        L += self._summary_added(x, order, name, cell)
        notes = [f"{name(f)}: {plain(n)}" for f, ns in x["notes"].items() for n in ns if f in pops]
        L += ["## Files of the first release without a counterpart here", "", "| File | Why there is none |", "|---|---|"]
        for k, v in ABSENT.items():
            L.append(f"| `{k}` | {v} |")
        if notes:
            L += ["", "Notes of this run: " + "; ".join(notes) + "."]
        L += [""]
        return L


# Wording of records saved before 19 October 2026 (compare_baselines.json) and the plain words the summary uses for it.
PLAIN_WORDS = {"(the start-fix sandbox re-ran the anchoring only)": "(only their anchoring was re-run there)",
               "the from-scratch stage (--recipe p3s)": "the from-scratch stage alone"}


def plain(text):
    """``text`` with the internal wording of older saved records replaced by plain words (PLAIN_WORDS)."""
    for old, new in PLAIN_WORDS.items():
        text = text.replace(old, new)
    return text


def summary_from_record(rec):
    """The summary sections of these files rebuilt from a saved ``compare_baselines.json`` record (its ``extra`` entry)
    with the current code; reads no data (``compare_baselines.py --summary-only``)."""
    c = Collector.__new__(Collector)                                # summary() reads only the per-population records
    c.rec = rec["extra"]["populations"]
    return c.summary(rec, rec["extra"])


def energy_level_rows(rec, T=None):
    """energy_level_readout.csv from the saved record: the sensitivity ``energy_level_readout`` of compare_baselines.json
    (ankyra-lite's level readout, 744 x level, in place of its trajectory's mean before the continuation layer; same estimator, random stream
    and model order as energy_error.csv).  ``energy_error.csv`` keeps the release definition."""
    rows = []
    for f, p in rec.get("populations", {}).items():
        el = p.get("energy_level_readout")
        if not el:
            continue
        for m in sorted(el["vs"]):
            v = el["vs"][m]
            lr, (lo, hi) = v["log_ratio"], v["UM"]
            rows.append({"set": p["name"], "tier": p["tier"], "windows": p["windows"], "units": p["units"], "model": m,
                         "model_label": (CB.TIDE_LABEL if m == "TiDE" else T.label(m, CB.REF_B, CB.REF_B_LABEL)) if T is not None else m,
                         "log_ratio": round(lr, 6) + 0.0, "um_low": round(lo, 6) + 0.0, "um_high": round(hi, 6) + 0.0,
                         "improvement_pct": round(100 * (1 - math.exp(lr)), 3) + 0.0, "resolved": _resolved(lo, hi),
                         "median_energy_ape_pct": round(el["median_energy_ape_pct"], 3),
                         "windows_projection_binds": p["product"]["level_vs_trajectory_mean_windows_over_1e-6"], "flag": p.get("flag") or ""})
    return rows


ABSENT = {
    "conventional_metrics.csv": "the release's file covers all windows; ankyra-lite forecasts exist on the test windows only (the earlier windows lie in or before its training period); `conventional_metrics_late.csv` holds the test windows",
    "constants_sensitivity.csv": "the release ran it on design windows without any test window; ankyra-lite forecasts exist on the test windows only, and the anchoring layer's constants are fixed in its byte-identical copy; the recipe constants the study varied are in `recipe_history.csv` and the `sensitivity` column of `constants.csv`",
    "contrast_ii_control_description.csv": "describes the calls of the carrier-free control, which does not involve the carrier (the same control as in the first release; `contrast_ii_control.csv` holds the contrasts)",
    "ankyra_2_vs_1x_by_day.csv, bdg2_micro_load_windows.csv, reevaluation_2_*.json, closeout_status.json": "records of ANKYRA's own versions and design decisions; ankyra-lite's own recipe history is `recipe_history.csv`",
    "carrier_swap*.csv, long_context_carriers.csv (the anchored rows)": "foundation models as ANKYRA's carrier are ANKYRA configurations; the two zero-shot long-context forecasts of `long_context_carriers.csv` are scored against ankyra-lite in `long_context_baselines.csv`, and trained carriers inside this anchoring layer are in `carriers_in_anchoring_layer.csv`",
    "trained_carriers_sensitivity.csv": "it measures what the earlier training cutoff of a trained carrier costs that carrier alone; ankyra-lite's carriers, and the other carriers of `carriers_in_anchoring_layer.csv`, are trained with that earlier cutoff already",
    "lcl_*, hkust_*, helsinki_*, unicon_*": "the external tests of this version are reported differently: carriers trained with the released recipe for each test set, LCL, UNICON and HKUST scored once (first read of this candidate on a previously read population) and a same-source out-of-time supplementary check (GoiEner households), all in `external_tests.csv` and docs/EXTERNAL_TESTS.md; Helsinki (retrospective) is trained and sealed, not yet scored; UCI370 was skipped",
    "ankyra_1x/, ankyra_2_0_0/, ankyra_2_0_1/, ankyra_2_1/ (version archives)": "ankyra-lite's version history is `recipe_history.csv` (the from-scratch stage and the recipe arms of the study)",
}
