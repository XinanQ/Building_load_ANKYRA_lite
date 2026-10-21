"""Zero-shot foundation models with a year of context: ankyra-lite against TimesFM 2.5 with an 8,736-hour context and
Chronos-2 with an 8,192-hour context (called by compare_baselines.py).

The first release scored these two long-context forecasts as baselines and as ANKYRA's carrier
(results/long_context_carriers.csv there).  Only the two zero-shot forecasts are read here (the study's
``analysis_outputs/v23_program/e1/{timesfm8736,chronos8192}_<population>.npz``, read-only); the anchored long-context
rows of the first release are ANKYRA configurations and are not used.  When a unit's record is shorter, or a gap falls
inside the span, the context is the longest completely observed run of hours ending at the origin (never shorter than
1,344 hours on these windows).  The two forecasters are a robustness check of the comparison, outside the 22
forecasters of the main tables and their ranks.

Windows are the test windows of the main tables.  CINELDI: the forecasts were made on the original store and are keyed
by its hours; the corrected store's hour h + 1,416 is the original store's hour h (both read no date), so windows are
mapped by that shift and the targets of the two stores must be equal on every mapped window (checked in
budget_tables.relaxed_forecasts for the same windows and here against the panel's targets).

Estimand: the release's unit-equal log RMS ratio, 95% unit x target-month bootstrap, 2,000 draws, a fresh generator per
contrast seeded SEED + 9 (hourly) or SEED + 11 (monthly energy).
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
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

E1 = CB.AO / "v23_program" / "e1"
MODELS = {"TimesFM-8736": "timesfm8736", "Chronos-2-8192": "chronos8192"}
SHORT = {"TimesFM-8736": "TimesFM", "Chronos-2-8192": "Chronos-2"}             # the same model with the 1,344-hour context
FILE = "long_context_baselines.csv"
FIELDS = ["set", "tier", "subset", "windows", "units", "contrast", "error", "log_ratio", "um_low", "um_high", "improvement_pct", "resolved",
          "median_context_hours", "flag"]
DESCRIPTION = ("ankyra-lite against TimesFM 2.5 with an 8,736-hour and Chronos-2 with an 8,192-hour context (zero-shot), and each against "
               "itself with the 1,344-hour context; outside the main ranks")


def long_context(face, P, prod):
    """{model: (forecast on the test windows in panel order, record)} of the two long-context forecasts, or {}."""
    pad = int(prod["pad"])
    out = {}
    want = [(int(b), int(o)) for b, o in zip(P["wbL"], prod["origins_product"])]
    for name, stem in MODELS.items():
        f = E1 / f"{stem}_{face}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        key = {(int(b), int(o) + pad): i for i, (b, o) in enumerate(z["keys"])}
        if any(k not in key for k in want):
            raise SystemExit(f"{face} {name}: the long-context forecast misses {sum(k not in key for k in want)} test windows")
        ix = [key[k] for k in want]
        pred = z["pred"][ix].astype(np.float64)
        if not np.isfinite(pred).all():
            raise SystemExit(f"{face} {name}: non-finite long-context forecast")
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        ctx = z["ctx_len"][ix] if "ctx_len" in z.files else None
        out[name] = (pred, {"file": CB.show(f), "windows": int(len(ix)), "extra_windows_in_file": int(len(key) - len(set(want))),
                            "origin_shift_h": pad, "context_hours_median": float(np.median(ctx)) if ctx is not None else None,
                            "context_hours_min": int(np.min(ctx)) if ctx is not None else None,
                            "windows_with_full_context": int((ctx >= int(stem[-4:])).sum()) if ctx is not None else None,
                            "max_context_requested": (meta.get("forecast_config_requested") or {}).get("max_context")})
    return out


def rows_for(xt, P, prod, baselines, lc, flag):
    """Rows of long_context_baselines.csv for one population (xt: the extra_tables Collector, for its contrast)."""
    y, unit, month = P["yL"], P["uL"], P["mL"]
    L = {"ankyra-lite": xt.losses(y, prod["traj"], prod.get("traj_pre") if prod.get("cont") is not None else None)}   # energy before the layer
    L |= {k: xt.losses(y, v[0]) for k, v in lc.items()}
    L |= {s: xt.losses(y, baselines[s]) for s in SHORT.values() if s in baselines}
    base = {"set": P["name"], "tier": P["tier"], "subset": "late", "windows": int(len(y)), "units": int(len(np.unique(unit))), "flag": flag or ""}
    specs = [(f"ankyra-lite vs {m}", "ankyra-lite", m) for m in lc]
    specs += [(f"{m} vs {SHORT[m]}", m, SHORT[m]) for m in lc if SHORT[m] in L]
    specs += [(f"ankyra-lite vs {SHORT[m]}", "ankyra-lite", SHORT[m]) for m in lc if SHORT[m] in L]
    rows, rec = [], {"forecasts": {k: v[1] for k, v in lc.items()}, "contrasts": {}}
    for lab, a, b in specs:
        for err, off in (("hourly", 9), ("energy", 11)):
            c = xt.ctr(L[a][err], L[b][err], unit, month, off)
            rec["contrasts"][f"{lab}|{err}"] = c
            r = xt.rnd(c)
            m = a if a in lc else (b if b in lc else None)
            rows.append(base | {"contrast": lab, "error": err, **{k: r[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct")},
                                "resolved": c["resolved"], "median_context_hours": lc[m][1]["context_hours_median"] if m else ""})
    return rows, rec


def summary(lcrec, names):
    """SUMMARY.md section from the saved record (key ``long_context``)."""
    pops = (lcrec or {}).get("populations") or {}
    if not pops:
        return []

    def cell(c):
        if not c or c.get("log_ratio") is None:
            return "-"
        return f"{c['improvement_pct']:+.1f}{' *' if c['um_high'] < 0 else (' (-)' if c['um_low'] > 0 else '')}"
    L = ["## Foundation models with a year of context (`long_context_baselines.csv`)", "",
         "Zero-shot TimesFM 2.5 with an 8,736-hour context and Chronos-2 with an 8,192-hour context, scored as two further baselines",
         "outside the main ranks. Improvement in % (`*` resolved better for the first-named, `(-)` resolved worse):", "",
         "| Population | ankyra-lite vs TimesFM-8736 (hourly / energy) | ankyra-lite vs Chronos-2-8192 | TimesFM-8736 vs TimesFM | Chronos-2-8192 vs Chronos-2 |",
         "|---|---|---|---|---|"]
    for f, r in pops.items():
        c = r["contrasts"]
        L.append(f"| {names.get(f, f)} | " + " | ".join(f"{cell(c.get(f'{lab}|hourly'))} / {cell(c.get(f'{lab}|energy'))}"
                                                     for lab in ("ankyra-lite vs TimesFM-8736", "ankyra-lite vs Chronos-2-8192",
                                                                 "TimesFM-8736 vs TimesFM", "Chronos-2-8192 vs Chronos-2")) + " |")
    L += [""]
    return L
