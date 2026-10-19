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

Interval estimand and contrasts are the release's: 2,000-draw unit x target-month bootstrap of the unit-equal log
ratio (tools_hcr_ext_drammen_score_v1), a fresh random stream per contrast.
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
        self.extra_json = {}
        self.faces = []
        self.notes = defaultdict(list)
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
    def losses(y, p):
        return {"hourly": ((p - y) ** 2).mean(1), "energy": (p.mean(1) - y.mean(1)) ** 2}

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
    def face(self, face, entry, P, prod, baselines, flag):
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
        if "ens" in cache and cache["ens"][0] is not None:
            self._deciles_collect(face, P, prod, ens, cache["ens"][0])
        self.rec[face] = rec
        return rec

    def _cache(self, face, entry, P, prod):
        """The re-anchored modes of lite_reanchor.py that belong to this product (full runs only); the single seeds are
        not read from here when the products mapping lists stored single-seed products."""
        import lite_reanchor as LR
        out = {}
        for mode in LR.MODES:
            if mode in LR.SEED_MODES and entry.get("seed_paths"):
                continue
            z, m = LR.load_mode(self.prods["label"], face, mode)
            if z is None:
                continue
            if m.get("product_sha256") != LR._sha(entry["npz_path"]) or (m.get("limit") is not None):
                self.notes[face].append(f"re-anchoring cache {mode} is stale or partial; not used")
                continue
            if m.get("error_count"):
                self.notes[face].append(f"re-anchoring {mode}: {m['error_count']} windows failed in the anchoring layer; mode not used")
                continue
            arr = {}
            for key in ("traj", "level", "peak", "off", "micro", "stats"):
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
            cdir = LR.cache_file(self.prods["label"], face, "s0").parent
            for f in (sorted(cdir.glob("s*_first*.json")) if cdir.exists() else []):
                m = json.loads(f.read_text(encoding="utf-8"))
                if m.get("product_sha256") == LR._sha(entry["npz_path"]) and m.get("gate"):
                    gates[f.stem] = m["gate"]
            rec["gate"] = gates or None
            if not gates:
                self.notes[face].append("no single-seed gate in the re-anchoring cache (run with --reanchor)")
            elif not all(g.get("pass") for g in gates.values()):
                self.notes[face].append("single-seed gate failed: re-anchoring does not reproduce the stored single-seed product")
        elif all(f"s{s}" in cache for s in (0, 1, 2)):
            rec["source"] = "re-anchored single seeds"
            out = {s: cache[f"s{s}"][0] for s in (0, 1, 2)}
        return out, rec

    def _ablation(self, face, entry, P, prod, ens, cache, seedp, flag):
        y, unit, month = P["yL"], P["uL"], P["mL"]
        lite = prod["traj"]
        L0 = self.losses(y, lite)
        variants = {"carrier alone (no anchoring layer)": (ens, "the three-seed carrier forecast at the origin, without the anchoring layer and without the state rule"),
                    "state rule off (the anchoring layer's own return on state windows)": (prod["raw"], f"{int(prod['state'].sum())} state windows")}
        for s, a in seedp.items():
            variants[f"single seed {s} (its carrier anchored alone)"] = (self.state_rule(prod, a["traj"].astype(np.float64), a["off"], a["micro"]),
                                                                         "stored single-seed product" if entry.get("seed_paths") else "re-anchored")
        for mode, lab in MODE_ABLATION.items():
            if mode in cache:
                a = cache[mode][0]
                note = "identical by construction (no micro-load window)" if cache[mode][1].get("identical_to") else "re-anchored"
                variants[lab] = (self.state_rule(prod, a["traj"].astype(np.float64), a["off"], a["micro"]), note)
        ho = self._deep(face, entry, "history_only.npz", P)
        if ho is not None:
            variants["history-only configuration (anchoring layer without a carrier)"] = (ho, "the anchoring layer called without a carrier (saved study forecasts)")
        for arm, lab in RECIPE_VARIANTS.items():                       # the study's other recipe arms, same windows and store
            f = entry.get("variant_paths", {}).get(arm)
            if f is None or not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            tr = self.align(z, P, pad=prod["pad"], key="traj")
            if tr is None:
                self.notes[face].append(f"recipe arm {arm} does not cover the test windows; not compared")
                continue
            off, mic = self.align(z, P, pad=prod["pad"], key="off"), self.align(z, P, pad=prod["pad"], key="micro")
            variants[lab] = (self.state_rule(prod, tr.astype(np.float64), off, mic), "saved study forecasts (three seeds), seasonal naive on state windows")
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
            Lv = self.losses(y, v)
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
        sk = [out.get(f"single seed {s} (its carrier anchored alone)|hourly") for s in (0, 1, 2)]
        if all(sk) and all(x["log_ratio"] is not None for x in sk):
            for err in ("hourly", "energy"):
                v = [out[f"single seed {s} (its carrier anchored alone)|{err}"]["log_ratio"] for s in (0, 1, 2)]
                lr = float(np.mean(v))
                self.rows["component_contributions"].append({"set": P["name"], "component": "three-seed ensemble (against one seed, mean over the three seeds)",
                                                             "improvement_pct": round(100 * (1 - math.exp(lr)), 3), "log_ratio": round(lr, 6), "um_low": "", "um_high": "",
                                                             "kind": "mean of three direct contrasts (point estimate)", "error": err, "resolved": "", "flag": flag})
        return out

    def _scorer_check(self, face, abl):
        """The product reading against the study's own scorer: the per-population points of the released recipe against
        the plain continuation (hourly, monthly energy) and against the from-scratch stage (hourly) in
        smallnet_sr_formal_v1/score_all.json must equal this run's ablation contrasts (same estimator, same losses)."""
        if not (str(self.prods.get("recipe", "")).startswith("span return") and CB.SR_SCORE.exists()):
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
        """Each seed alone against the three-seed ensemble of the same stage, and each seed's and the ensemble's hourly
        contrast against TimesFM and Chronos-2-X.  BDG2 also without the three near-zero meters: there the carrier alone
        has a nonzero error while some baselines' errors are near zero, which dominates the unit-equal mean."""
        out = {}
        stages = [("carrier alone", ens, {s: p for s, p in enumerate(seeds)})]
        if all(s in seedp for s in (0, 1, 2)):
            stages.append(("anchored (ankyra-lite)", prod["traj"],
                           {s: self.state_rule(prod, seedp[s]["traj"].astype(np.float64), seedp[s]["off"], seedp[s]["micro"]) for s in (0, 1, 2)}))
        sets = [("all units", np.ones(len(P["yL"]), bool))]
        if face == "bdg2":
            sets.append(("without the three near-zero meters", ~np.isin(P["uL"], self.T.near_zero)))
        for uset, keep in sets:
            y, unit, month = P["yL"][keep], P["uL"][keep], P["mL"][keep]
            pre = "" if uset == "all units" else f"{uset}|"
            for stage, ens_p, per in stages:
                Le = self.losses(y, ens_p[keep])
                Lb = {comp: self.losses(y, baselines[comp][keep]) for comp in ("TimesFM", "Chronos-2-X")}
                for comp in Lb:
                    c = self.ctr(Le["hourly"], Lb[comp]["hourly"], unit, month, 9)
                    out[f"{pre}{stage}|ensemble|{comp}"] = c
                    self._seed_row(P, uset, keep, stage, "three-seed ensemble", "hourly", comp, c, flag)
                for s_, p in per.items():
                    Ls = self.losses(y, p[keep])
                    for err, off in (("hourly", 9), ("energy", 11)):
                        c = self.ctr(Ls[err], Le[err], unit, month, off)
                        out[f"{pre}{stage}|seed {s_}|three-seed ensemble|{err}"] = c
                        self._seed_row(P, uset, keep, stage, f"seed {s_}", err, "three-seed ensemble", c, flag)
                    for comp in Lb:
                        c = self.ctr(Ls["hourly"], Lb[comp]["hourly"], unit, month, 9)
                        out[f"{pre}{stage}|seed {s_}|{comp}"] = c
                        self._seed_row(P, uset, keep, stage, f"seed {s_}", "hourly", comp, c, flag)
                for comp in Lb:
                    v = [out[f"{pre}{stage}|seed {s_}|{comp}"]["improvement_pct"] for s_ in per]
                    out[f"{pre}{stage}|spread|{comp}"] = {"min": float(min(v)), "max": float(max(v)), "sd": float(np.std(v, ddof=1))}
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
        lite = prod["traj"]
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
        cuts = {}
        for X in DEEP_CARRIERS:
            an = self._deep(face, entry, f"anchored_{X}.npz", P)
            ca = self._deep(face, entry, f"carrier_{X}.npz", P, k0=True)
            if an is None or ca is None:
                self.notes[face].append(f"trained carrier {X}: no saved anchored forecast on the test windows")
                continue
            add(X, "ankyra-lite vs anchored carrier", lite, an)
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
        P_ = {"A": prod["traj"], "C": ens, "H": ho}
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
                    if sub == "all late windows" and blk in ("hourly", "level") and table == "contrast_ii_blocks" and contrast != "carrier alone vs history only":
                        k = {"anchored vs history only": "anchored vs history only", "anchored vs carrier alone": "anchored vs carrier alone"}[contrast]
                        cc = carriers.get(f"{LITE_CARRIER}|{k}|{'hourly' if blk == 'hourly' else 'energy'}")
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
        self.rows["example_window_cambridge"] = rows
        js = {"set": "Cambridge", "building": str(P["uL"][j]), "target_month_midpoint": str(P["mL"][j]),
              "forecast_period": f"{o_date.date()} to {(o_date + timedelta(hours=HORIZON)).date()}",
              "context_day_types": [int(x) for x in P["ctL"][j]], "target_day_types": [int(x) for x in P["ttL"][j]],
              "day_type_codes": "Monday 0 ... Sunday 6, holiday 7",
              "source": "University of Cambridge estate building energy archive (Langtry & Choudhary 2024, Zenodo 10.5281/zenodo.10955332), CC BY 4.0",
              "note": ("hour 0 is the forecast origin; negative hours are the 1,344-hour context. ankyra_kw is ankyra-lite (the column keeps the "
                       "release's name), carrier_kw the small carrier network alone (three-seed forecast at the origin, before the anchoring layer), "
                       "timesfm_kw zero-shot TimesFM 2.5"),
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
            hist = json.loads((Path(entry["run_paths"][0]) / "COMPLETE.json").read_text(encoding="utf-8")).get("history", [])
            epochs = max((int(h["epoch"]) for h in hist), default="")
        row.update({"own_training_windows": sp["own_pool_rows"] if sp else "", "returned_validation_windows": sp["span_rows_target"] if sp else "",
                    "returned_label_end": d(sp["label_end_max"]) if sp else "", "continuation_epochs": epochs})
        self.rows["datasets"].append(row)
        return {k: row[k] for k in row} | {"first_training_origin_source": src}

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
                c = json.loads((Path(rd) / "COMPLETE.json").read_text(encoding="utf-8"))
                hist = c.get("history", [])
                last = max(int(h["epoch"]) for h in hist)
                st = json.loads((Path(rd) / "STARTED.json").read_text(encoding="utf-8")) if (Path(rd) / "STARTED.json").exists() else {}
                stage = "span return (fixed epochs)" if st.get("span_rows_returned", 1) else "continuation (fixed epochs)"
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

    def _deciles_collect(self, face, P, prod, ens, ensc):
        y = P["yL"]
        ybar = y.mean(1)
        unit = P["uL"]
        keep = np.array([abs(ybar[unit == u]).mean() > 1e-6 for u in unit])
        A, Cr = prod["traj"], ens
        self.dec_pool.append({"face": face, "unit": np.array([f"{face}:{u}" for u in unit])[keep], "month": P["mL"][keep], "stats": ensc["stats"][keep],
                              "eA": (A.mean(1) - ybar)[keep] ** 2, "eC": (Cr.mean(1) - ybar)[keep] ** 2,
                              "mA": ((A - y) ** 2).mean(1)[keep], "mC": ((Cr - y) ** 2).mean(1)[keep]})

    # ------------------------------------------------------------------ after the last population
    def finish(self, out_dir, rec, cost=True, robustness=True, robustness_face=None, robustness_windows=None):
        written, files = [], {}
        hist_pooled = self._history_pooled(rec)
        dec = self._deciles()
        cost_rec = rob_rec = None
        if cost:
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
        rt_written, rt_rec, rt_files = RT.write(out_dir, log=CB.log)
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
                                "each seed alone against the three-seed ensemble, and each seed's contrast against TimesFM and Chronos-2-X"),
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
        rec_out = {"populations": self.rec, "history_length_pooled": hist_pooled, "anchoring_gain_deciles": dec and dec["rec"],
                   "cost": cost_rec, "robustness": rob_rec, **rt_rec, "notes": dict(self.notes), "absent_files": ABSENT}
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
            L += ["## Training (`training_curves.csv`)", "",
                  "| Population | From-scratch stage: best epoch (seeds 0, 1, 2) | Epochs run | "
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
        L += ["## Seeds (`seed_spread.csv`)", "", "Hourly improvement against TimesFM, per seed and for the three-seed ensemble (carrier alone, and anchored):", "",
              "| Population | Stage | Seed 0 | Seed 1 | Seed 2 | Ensemble |", "|---|---|---|---|---|---|"]
        for f in order:
            s = self.rec[f]["seed_spread"]
            for pre, lab in (("", ""), ("without the three near-zero meters|", " (without the three near-zero meters)")):
                for stage in ("carrier alone", "anchored (ankyra-lite)"):
                    if f"{pre}{stage}|ensemble|TimesFM" in s:
                        L.append(f"| {name(f)}{lab} | {stage} | " + " | ".join(cell(s.get(f"{pre}{stage}|seed {k}|TimesFM")) for k in (0, 1, 2))
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
                  f"Measured on {c['windows']} {c['face_name']} test windows, CPU, one thread: carrier forecasts at 7 issue times x 3 seeds "
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


ABSENT = {
    "conventional_metrics.csv": "the release's file covers all windows; ankyra-lite forecasts exist on the test windows only (the earlier windows lie in or before its training period); `conventional_metrics_late.csv` holds the test windows",
    "handover_granularity.csv": "needs other handover rules inside the anchoring layer, which must stay byte-identical to ANKYRA 2.2.1",
    "constants_sensitivity.csv": "the release ran it on design windows without any test window; ankyra-lite forecasts exist on the test windows only; the recipe constants the study varied are in `recipe_history.csv` and the `sensitivity` column of `constants.csv`",
    "contrast_ii_control_description.csv": "describes the calls of the carrier-free control, which does not involve the carrier (the same control as in the first release; `contrast_ii_control.csv` holds the contrasts)",
    "lead_weeks_first_read.csv, ankyra_2_vs_1x_by_day.csv, bdg2_micro_load_windows.csv, carrier_swap*.csv, long_context_carriers.csv, generic_combinations.csv, trained_carriers_sensitivity.csv, training_budget_sensitivity.csv":
        "records of ANKYRA's own versions, carriers or baselines; the comparable ankyra-lite content is in `ablation.csv`, `carriers_in_anchoring_layer.csv`, `seed_spread.csv` and `recipe_history.csv`",
    "lcl_*, hkust_*, helsinki_*, unicon_*": "the external tests of this version are reported differently: carriers trained with the released recipe for each test set, LCL, UNICON and HKUST scored once (first read of this candidate on a previously read population) and a same-source out-of-time supplementary check (GoiEner households), all in `external_tests.csv` and docs/EXTERNAL_TESTS.md; Helsinki (retrospective) is trained and sealed, not yet scored; UCI370 was skipped",
    "ankyra_1x/, ankyra_2_0_0/, ankyra_2_0_1/, ankyra_2_1/ (version archives)": "ankyra-lite's version history is `recipe_history.csv` (the from-scratch stage and the recipe arms of the study)",
}
