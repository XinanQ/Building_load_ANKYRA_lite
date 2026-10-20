"""Training budget of the trained baselines: ankyra-lite against the four trained load-only baselines retrained with a
relaxed budget (called by compare_baselines.py; ``--budget`` writes this file alone).

The first release's sensitivity analysis (results/training_budget_sensitivity.csv there) retrained DLinear,
iTransformer, LSTM and PatchTST on each population with at most 30 epochs and early-stopping patience 8 instead of 15 and
3; cutoff, training pairs, selection set, learning-rate grid, seeds and scoring unchanged (the study's
tools_hcr_deep_baselines_budget_v1.py, protocol fixed before training).  Its saved forecasts
(``analysis_outputs/deep_baselines_budget_v1/<population>/pred_<family>.npz``, with the selected learning rate, epoch
count and cap flags in ``deep_<family>.json``) are read here, read-only, and scored against ankyra-lite on the test
windows of the nine populations with the release estimand: unit-equal log RMS ratio, 95% unit x target-month bootstrap,
2,000 draws, a fresh generator per contrast seeded SEED + 9 (hourly; this reproduces benchmark_pairwise.csv) or SEED + 11
(monthly energy, 744 x the error of the window mean).

CINELDI.  ankyra-lite is scored on the corrected CINELDI store, whose hour h + 1,416 is hour h of the original store; the
relaxed forecasts were made on the original store.  The four families read load only (no date), so a forecast issued at
original-store hour o is the forecast of the corrected-store window at o + 1,416 h: windows are mapped by that shift and
the two stores' targets of every mapped window must be equal.  The original-budget forecasts of the corrected-store
scoring are the original-store forecasts themselves (checked array for array).

Nothing is trained and no forecast is changed; the file is a sensitivity of the comparison, not a new ranking.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import compare_baselines as CB                                     # noqa: E402

import json                                                        # noqa: E402
import math                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

FAMILIES = ("DLinear", "iTransformer", "LSTM", "PatchTST")
BUDGET_DIR = CB.AO / "deep_baselines_budget_v1"                    # relaxed budget: 30 epochs, patience 8
ORIGINAL_TRAIN_DIR = CB.AO / "hcr_baseline_panel_v1"               # training records of the original budget (15 epochs, patience 3)
ORIGINAL_STORE_CINELDI = Path(os.environ.get("ANKYRA_LITE_CINELDI_ORIGINAL_STORE", "D:/external/cineldi/store"))  # original CINELDI store (study default; override with the env var)
MAX_EPOCHS, PATIENCE, ORIG_MAX_EPOCHS, ORIG_PATIENCE = 30, 8, 15, 3
FIELDS = ["set", "tier", "subset", "windows", "units", "model", "error",
          "original_log_ratio", "original_um_low", "original_um_high", "original_improvement_pct", "original_resolved",
          "relaxed_log_ratio", "relaxed_um_low", "relaxed_um_high", "relaxed_improvement_pct", "relaxed_resolved",
          "relaxed_vs_original_log_ratio", "relaxed_vs_original_um_low", "relaxed_vs_original_um_high",
          "relaxed_vs_original_improvement_pct", "relaxed_vs_original_resolved",
          "lr_original", "epochs_original", "lr_relaxed", "epochs_relaxed", "selected_run_reached_30_epochs", "selected_epoch_is_30", "flag"]
OFFSET = {"hourly": 9, "energy": 11}
FILE = "training_budget_sensitivity.csv"
DESCRIPTION = ("ankyra-lite against DLinear, iTransformer, LSTM and PatchTST trained with the original budget (15 epochs, patience 3) and "
               "with the relaxed budget (30 epochs, patience 8), hourly and energy, and each relaxed model against its original")


def _losses(y, p):
    return {"hourly": ((p - y) ** 2).mean(1), "energy": (p.mean(1) - y.mean(1)) ** 2}


def _ctr(T, a, b, unit, month, off):
    if np.array_equal(a, b):
        return {"log_ratio": 0.0, "um_low": 0.0, "um_high": 0.0, "improvement_pct": 0.0, "identical": True}
    with np.errstate(all="ignore"):
        c = CB.contrast_um(T, a, b, unit, month, np.random.default_rng(T.DS.SEED + off))
    lr, (lo, hi) = float(c["log_ratio"]), (float(c["UM"][0]), float(c["UM"][1]))
    return {"log_ratio": lr, "um_low": lo, "um_high": hi, "improvement_pct": 100 * (1 - math.exp(lr)), "identical": False}


def _verdict(c, first, second):
    if c.get("identical"):
        return "identical forecasts"
    return f"{first} better" if c["um_high"] < 0 else (f"{second} better" if c["um_low"] > 0 else "not resolved")


def _r(x, nd=6):
    return round(float(x), nd) + 0.0


def relaxed_forecasts(face, P, prod, entry):
    """{family: (relaxed forecast on the test windows in panel order, record)}; CINELDI on the corrected store through the
    1,416-hour shift with the targets of both stores checked equal."""
    pad = int(prod["pad"])
    out = {}
    for fam in FAMILIES:
        f = BUDGET_DIR / face / f"pred_{fam}.npz"
        j = BUDGET_DIR / face / f"deep_{fam}.json"
        if not (f.exists() and j.exists()):
            continue
        z = np.load(f, allow_pickle=False)
        key = {(int(b), int(o) + pad): i for i, (b, o) in enumerate(zip(z["building"], z["origin"]))}     # origin on the scored store
        want = [(int(b), int(o)) for b, o in zip(P["wbL"], prod["origins_product"])]
        if any(k not in key for k in want):
            raise SystemExit(f"{face} {fam}: the relaxed forecast misses {sum(k not in key for k in want)} test windows")
        pred = z["pred"][[key[k] for k in want]].astype(np.float64)
        if not np.isfinite(pred).all():
            raise SystemExit(f"{face} {fam}: non-finite relaxed forecast")
        info = json.loads(j.read_text(encoding="utf-8"))
        if info.get("smoke") or int(info.get("max_epochs", -1)) != MAX_EPOCHS or int(info.get("patience", -1)) != PATIENCE:
            raise SystemExit(f"{face} {fam}: the relaxed record is not the 30-epoch, patience-8 run")
        orig = json.loads((ORIGINAL_TRAIN_DIR / face / f"deep_{fam}.json").read_text(encoding="utf-8"))
        same_setup = all(orig.get(k) == info.get(k) for k in ("cutoff_index", "train_pairs", "val_pairs", "val_first_origin", "late_windows"))
        rec = {"file": CB.show(f), "windows": int(len(want)), "extra_windows_in_file": int(len(key) - len(set(want))),
               "lr_original": orig.get("lr"), "epochs_original": orig.get("epochs"), "lr_relaxed": info.get("lr"), "epochs_relaxed": info.get("epochs"),
               "selected_run_reached_30_epochs": bool(info.get("selected_run_hit_cap")), "selected_epoch_is_30": bool(info.get("selected_epoch_is_cap")),
               "same_cutoff_pairs_and_selection_set_as_the_original": bool(same_setup), "origin_shift_h": pad}
        out[fam] = (pred, rec)
    if pad and out:                                                   # CINELDI: the two stores' targets of every mapped window
        store, _ = CB.corrected_store_info()
        zc = np.load(Path(store) / "store.npz", allow_pickle=False)["load"]
        zo = np.load(ORIGINAL_STORE_CINELDI / "store.npz", allow_pickle=False)["load"]
        yc = np.stack([zc[b, o:o + CB.HORIZON] for b, o in zip(P["wbL"], prod["origins_product"])])
        yo = np.stack([zo[b, o - pad:o - pad + CB.HORIZON] for b, o in zip(P["wbL"], prod["origins_product"])])
        eq = bool(np.array_equal(yc, yo) and np.array_equal(yc, P["yL"]))
        if not eq:
            raise SystemExit("cineldi: the corrected-store and original-store targets of the mapped windows differ")
        for fam in out:
            out[fam][1]["targets_equal_on_both_stores"] = eq
        del zc, zo
    return out


def rows_for(T, P, prod, baselines, relaxed, flag, hourly_ref=None):
    """Rows of training_budget_sensitivity.csv for one population and its record."""
    y, unit, month = P["yL"], P["uL"], P["mL"]
    Llite = _losses(y, prod["traj"])
    rows, rec = [], {"families": {}}
    for fam in FAMILIES:
        if fam not in relaxed:
            continue
        pred, info = relaxed[fam]
        Lo, Lr = _losses(y, baselines[fam]), _losses(y, pred)
        fr = {"train": info}
        for err in ("hourly", "energy"):
            o = _ctr(T, Llite[err], Lo[err], unit, month, OFFSET[err])
            n = _ctr(T, Llite[err], Lr[err], unit, month, OFFSET[err])
            s = _ctr(T, Lr[err], Lo[err], unit, month, OFFSET[err])
            vo, vn = _verdict(o, "ankyra-lite", "baseline"), _verdict(n, "ankyra-lite", "baseline")
            fr[err] = {"original": o, "relaxed": n, "relaxed_vs_original": s, "verdict_original": vo, "verdict_relaxed": vn,
                       "verdict_unchanged": vo == vn, "point_sign_changed": (o["log_ratio"] < 0) != (n["log_ratio"] < 0),
                       "relaxed_vs_original_verdict": _verdict(s, "relaxed", "original")}
            if err == "hourly" and hourly_ref is not None and fam in hourly_ref:
                ref = hourly_ref[fam]
                fr["reproduces_benchmark_pairwise"] = bool(abs(ref["log_ratio"] - o["log_ratio"]) < 1e-12
                                                           and abs(ref["UM"][0] - o["um_low"]) < 1e-12 and abs(ref["UM"][1] - o["um_high"]) < 1e-12)
            rows.append({"set": P["name"], "tier": P["tier"], "subset": "late", "windows": int(len(y)), "units": int(len(np.unique(unit))),
                         "model": fam, "error": err,
                         "original_log_ratio": _r(o["log_ratio"]), "original_um_low": _r(o["um_low"]), "original_um_high": _r(o["um_high"]),
                         "original_improvement_pct": _r(o["improvement_pct"], 3), "original_resolved": vo,
                         "relaxed_log_ratio": _r(n["log_ratio"]), "relaxed_um_low": _r(n["um_low"]), "relaxed_um_high": _r(n["um_high"]),
                         "relaxed_improvement_pct": _r(n["improvement_pct"], 3), "relaxed_resolved": vn,
                         "relaxed_vs_original_log_ratio": _r(s["log_ratio"]), "relaxed_vs_original_um_low": _r(s["um_low"]),
                         "relaxed_vs_original_um_high": _r(s["um_high"]), "relaxed_vs_original_improvement_pct": _r(s["improvement_pct"], 3),
                         "relaxed_vs_original_resolved": fr[err]["relaxed_vs_original_verdict"],
                         "lr_original": info["lr_original"], "epochs_original": info["epochs_original"], "lr_relaxed": info["lr_relaxed"],
                         "epochs_relaxed": info["epochs_relaxed"], "selected_run_reached_30_epochs": info["selected_run_reached_30_epochs"],
                         "selected_epoch_is_30": info["selected_epoch_is_30"], "flag": flag or ""})
        rec["families"][fam] = fr
    preds0 = {CB.REF_B: prod["traj"]} | baselines                       # ankyra-lite's rank with the relaxed versions substituted
    preds1 = dict(preds0) | {fam: relaxed[fam][0] for fam in relaxed}
    for lab, pr in (("original", preds0), ("relaxed_substituted", preds1)):
        mr, rank, of, allr = CB.unit_rank(pr, y, unit, CB.REF_B)
        rec[f"rank_{lab}"] = {"mean_unit_rank": mr, "rank": rank, "of": of, **{f"{fam}_mean_unit_rank": allr[fam] for fam in relaxed}}
    return rows, rec


def counts(rec):
    """Totals over the populations of a run: verdicts unchanged, sign changes, relaxed against original."""
    out = {}
    for err in ("hourly", "energy"):
        cells = [(f, fam, v[err]) for f, r in rec.items() for fam, v in r["families"].items()]
        out[err] = {"contrasts": len(cells), "verdict_unchanged": sum(c["verdict_unchanged"] for _, _, c in cells),
                    "point_sign_changed": sum(c["point_sign_changed"] for _, _, c in cells),
                    "changed": [{"population": f, "model": fam, "original": c["verdict_original"], "relaxed": c["verdict_relaxed"]}
                                for f, fam, c in cells if not c["verdict_unchanged"]],
                    "relaxed_vs_original": {k: sum(c["relaxed_vs_original_verdict"] == k for _, _, c in cells)
                                            for k in ("relaxed better", "not resolved", "original better", "identical forecasts")}}
    cap = [(f, fam, v["train"]) for f, r in rec.items() for fam, v in r["families"].items()]
    out["selected_run_reached_30_epochs"] = sorted(f"{f}|{fam}" for f, fam, t in cap if t["selected_run_reached_30_epochs"])
    out["selected_epoch_is_30"] = sorted(f"{f}|{fam}" for f, fam, t in cap if t["selected_epoch_is_30"])
    out["rank_unchanged"] = all(r["rank_original"]["rank"] == r["rank_relaxed_substituted"]["rank"] for r in rec.values())
    return out


def summary(budget, names):
    """SUMMARY.md section from the saved record (compare_baselines.json, key ``budget``)."""
    if not budget or not budget.get("populations"):
        return []
    rec, c = budget["populations"], budget["counts"]
    L = ["## Training budget of the trained baselines (`training_budget_sensitivity.csv`)", "",
         "ankyra-lite against DLinear, iTransformer, LSTM and PatchTST as trained for the main tables (at most 15 epochs, patience 3)",
         "and retrained with at most 30 epochs and patience 8 (the first release's sensitivity run; cutoff, pairs, selection set,",
         "learning-rate grid and seeds unchanged). Improvement of ankyra-lite in % (`*` resolved better, `(-)` resolved worse):", "",
         "| Population | " + " | ".join(f"{fam} original -> relaxed" for fam in FAMILIES) + " |", "|---|" + "---|" * len(FAMILIES)]

    def cell(x):
        return f"{x['improvement_pct']:+.1f}{' *' if x['um_high'] < 0 else (' (-)' if x['um_low'] > 0 else '')}"
    for f, r in rec.items():
        L.append(f"| {names.get(f, f)} | " + " | ".join(
            (f"{cell(r['families'][fam]['hourly']['original'])} -> {cell(r['families'][fam]['hourly']['relaxed'])}" if fam in r["families"] else "-")
            for fam in FAMILIES) + " |")
    h, e = c["hourly"], c["energy"]
    rv = h["relaxed_vs_original"]
    L += ["", f"Hourly: {h['verdict_unchanged']} of {h['contrasts']} verdicts unchanged, {h['point_sign_changed']} point estimates change sign; "
          f"energy: {e['verdict_unchanged']} of {e['contrasts']} unchanged, {e['point_sign_changed']} sign changes. "
          f"Relaxed against original (hourly): {rv['relaxed better']} better, {rv['not resolved']} not resolved, {rv['original better']} worse, "
          f"{rv['identical forecasts']} identical forecasts. Selected run at the 30-epoch cap: {len(c['selected_run_reached_30_epochs'])}; "
          f"selected epoch = 30: {len(c['selected_epoch_is_30'])}. ankyra-lite's rank among the 22 forecasters with the four relaxed versions "
          f"substituted is {'unchanged on every population' if c['rank_unchanged'] else 'changed on some population'}.", ""]
    return L
