"""robustness.csv: ankyra-lite on corrupted contexts, with this repository's package on the CPU (one thread).

The counterpart of the first release's robustness.csv (its research tool tools_robustness_v1.py): the same eight
corruptions of the context at the origin, one level each, with the same random stream (default_rng(20261003), drawn in
the same order), the targets untouched.  The released checkpoints of one population (three seeds, the weights file of
the products mapping) forecast from the corrupted record at the origin and at every pseudo-origin the anchoring layer
reads (``ankyra_lite.forecast``: carrier forecasts, anchoring layer, state rule, peak readout), so every part of the
forecaster sees the corruption.  The comparator is the carrier alone (the three-seed carrier forecast at the origin,
from the same corrupted record), in the place of the first release's foundation model alone.

Corruptions: ``miss05`` / ``miss20`` 5% / 20% of the context hours removed and filled by linear interpolation (the
anchoring layer refuses an incomplete context at the origin), ``zero24`` a 24-hour block of zeros in the last week,
``spike`` one hour at 10 x the context maximum 36 hours before the origin, ``shift_plus1`` / ``shift_minus1`` the
context shifted by one hour, ``scale0.5`` / ``scale2`` the whole history multiplied by 0.5 or 2 (the forecast is scaled
back before comparing).

Sample: n test windows of the population spread evenly over its test windows in the product's order (default 64 Drammen
windows; the first release used the first 64 Drammen panel windows of its timing run; here only test windows qualify,
the windows whose carrier forecasts at all six pseudo-origins are issued after the carrier's training cut).  Measures as in the first release: unit-equal log ratio of the hourly error with the corrupted context to
the error with the clean one (positive = worse), share of windows whose error grew, median RMS change of the forecast
over the unit's scale s0, change of the level and of the peak readout, firing of the state rules, ankyra-lite against the
carrier alone on the corrupted context, and the largest deviation from exact scale equivariance.  The clean forecasts are
recomputed here (CPU) and compared with the stored evaluation product (computed on a GPU).

Version 0.2.0: every seed of the configuration forecasts (six seeds in a six-seed configuration) and, when the mapping
switches it on, the continuation layer acts on every forecast (``forecast(..., continuation=True)``): corrupting the last
hours then moves the layer as well, which is part of what is measured.  The hourly errors read the delivered trajectory;
the level and peak readouts are read before the layer (readout rule A).  The clean trajectories before the layer are
compared with the stored product and the delivered ones with the stored continued product.

Called by extra_tables.Collector.finish; ``python -B evaluation/robustness.py --face oslo --windows 64 --out <dir>``
runs it alone (CPU, one thread) and writes robustness.csv / robustness.json into <dir>.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import compare_baselines as CB                                     # noqa: E402

import argparse                                                    # noqa: E402
import importlib                                                   # noqa: E402
import json                                                        # noqa: E402
import time                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

FACE, N, CONTEXT, HORIZON, STEP = "drammen", 64, CB.CONTEXT, CB.HORIZON, 744
SEED = 20261003                                                    # the first release's stream
KINDS = ["clean", "miss05", "miss20", "zero24", "spike", "shift_plus1", "shift_minus1", "scale0.5", "scale2"]
FIELDS = ["corruption", "ankyra_error_vs_clean_log_ratio", "carrier_error_vs_clean_log_ratio", "ankyra_windows_worse_share",
          "carrier_windows_worse_share", "ankyra_displacement_over_s0_median", "carrier_displacement_over_s0_median",
          "level_change_pct_median", "level_change_pct_max", "peak_readout_change_pct_median", "peak_error_vs_clean_log_ratio",
          "off_state_fired", "micro_load_fired", "ankyra_vs_carrier_corrupted_log_ratio", "equivariance_max_rel_dev", "windows", "units", "set"]


def interp_nan(x):
    x = x.copy()
    m = np.isfinite(x)
    if m.all():
        return x
    idx = np.arange(len(x))
    x[~m] = np.interp(idx[~m], idx[m], x[m])
    return x


def corrupt(load_row, o, kind, rng):
    """(history load up to o with the corruption, scale factor) -- tools_robustness_v1.corrupt, line for line."""
    L = load_row[:o].astype(np.float64).copy()
    c = 1.0
    s, e = o - CONTEXT, o
    if kind in ("miss05", "miss20"):
        p = 0.05 if kind == "miss05" else 0.20
        idx = s + np.flatnonzero(rng.random(CONTEXT) < p)
        L[idx] = np.nan
        L[s:e] = interp_nan(L[s:e])
    elif kind == "zero24":
        L[o - 120:o - 96] = 0.0
    elif kind == "spike":
        L[o - 36] = 10.0 * np.nanmax(L[s:e])
    elif kind in ("shift_plus1", "shift_minus1"):
        seg = L[s:e].copy()
        L[s:e] = np.roll(seg, 1 if kind == "shift_plus1" else -1)
        if kind == "shift_plus1":
            L[s] = seg[0]
        else:
            L[e - 1] = seg[-1]
    elif kind in ("scale0.5", "scale2"):
        c = 0.5 if kind == "scale0.5" else 2.0
        L = L * c
    elif kind != "clean":
        raise ValueError(kind)
    return L, c


def measure(T, face, entry, n=N, log=print, layer=None):
    """Run the stress test on one population; returns (record, rows).  ``layer``: the continuation layer on or off
    (default: the mapping's)."""
    import torch
    torch.set_num_threads(1)
    if str(CB.REPO) not in sys.path:
        sys.path.append(str(CB.REPO))
    from ankyra_lite import load_carrier
    from ankyra_lite.anchor import History
    FM = importlib.import_module("ankyra_lite.forecast")
    if CB.on_corrected_cineldi(face, entry):
        raise SystemExit("robustness: choose a population whose own data are on their original store")
    t_all = time.perf_counter()
    V = T.pwf().V
    d = V.raw(face)
    region = V._region(face, d)
    q0 = np.load(entry["run_paths"][0] / "carrier.npz", allow_pickle=False)
    cut = int(q0["cut"])
    sigma = float(json.loads(Path(entry["prep_path"]).read_text(encoding="utf-8"))["sigma"])
    models = [load_carrier(Path(rd) / entry["checkpoint"]) for rd in entry["run_paths"]]
    zx = np.load(entry["npz_path"], allow_pickle=False)
    layer = bool(entry.get("layer")) if layer is None else bool(layer)
    zc = np.load(entry["continued_path"], allow_pickle=False) if (layer and entry.get("continued_path") is not None) else None
    kw = {"continuation": True} if layer else {}
    pick = np.unique(np.linspace(0, len(zx["building"]) - 1, n).round().astype(int))
    wb, wo = zx["building"][pick].astype(int), zx["origin"][pick].astype(int)
    load, temp, types = d["load"], d["temp"], d["types"]
    y = np.stack([load[b, o:o + HORIZON] for b, o in zip(wb, wo)]).astype(np.float64)
    if not np.isfinite(y).all():
        keep = np.isfinite(y).all(1)
        log(f"robustness {face}: {int((~keep).sum())} windows with missing target hours left out")
        pick, wb, wo, y = pick[keep], wb[keep], wo[keep], y[keep]
    m = len(wb)
    rng = np.random.default_rng(SEED)
    F, C, LEV, PK, OFF, MIC, CS, sec = {}, {}, {}, {}, {}, {}, {}, {}
    F_before = None
    for kind in KINDS:
        t0 = time.perf_counter()
        Fk, Ck = np.zeros((m, HORIZON)), np.zeros((m, HORIZON))
        Fb = np.zeros((m, HORIZON)) if (kind == "clean" and layer) else None
        lev, pk, off, mic, cs = np.zeros(m), np.zeros(m), np.zeros(m, bool), np.zeros(m, bool), np.ones(m)
        for i, (b, o) in enumerate(zip(wb, wo)):
            L, c = corrupt(load[b], int(o), kind, rng)
            ty = types if types.ndim == 1 else types[b]
            tmp = temp[b] if temp.ndim == 2 else temp
            h = History(load_kw=L, temperature_c=tmp[:o], day_types=ty[:o + HORIZON], start_timestamp=d["start_iso"], observed=np.isfinite(L))
            r = FM.forecast(h, models, group=d["groups"][b], temp_sigma_std=sigma, dst_region=region(b), cut=cut, **kw)
            Fk[i], Ck[i], lev[i], pk[i], cs[i] = r.trajectory_kw, r.carrier[0], r.level_kw, r.peak_kw, c
            if Fb is not None:
                Fb[i] = r.trajectory_before_continuation_kw
            off[i], mic[i] = bool(r.anchored.off_state), bool(r.anchored.micro_load)
        F[kind], C[kind], LEV[kind], PK[kind], OFF[kind], MIC[kind], CS[kind] = Fk, Ck, lev, pk, off, mic, cs
        if Fb is not None:
            F_before = Fb
        sec[kind] = time.perf_counter() - t0
        log(f"robustness {face} {kind}: {sec[kind]:.0f} s, off-state {int(off.sum())}, micro-load {int(mic.sum())}")
    stored = zx["traj"][pick].astype(np.float64)
    st = OFF["clean"] | MIC["clean"]
    Fc = F["clean"] if F_before is None else F_before                       # before the continuation layer, as the stored product
    clean_rel = float(np.max(np.abs(Fc[~st] - stored[~st]) / np.maximum(np.abs(stored[~st]).mean(1, keepdims=True), 1e-9))) if (~st).any() else float("nan")
    cont_rel = None
    if zc is not None and (~st).any():
        sc = zc["traj"][pick].astype(np.float64)
        cont_rel = float(np.max(np.abs(F["clean"][~st] - sc[~st]) / np.maximum(np.abs(sc[~st]).mean(1, keepdims=True), 1e-9)))
    s0 = np.array([max(float(load[b, o - HORIZON:o].std()), max(0.01 * abs(float(load[b, o - HORIZON:o].mean())), 1e-3)) for b, o in zip(wb, wo)])
    ypk = y.max(1)
    uu, inv = np.unique(wb, return_inverse=True)

    def unit_mean_log(a, b):                                      # tools_robustness_v1.unit_mean_log
        with np.errstate(all="ignore"):
            r = 0.5 * np.log(np.bincount(inv, a, len(uu)) / np.bincount(inv, b, len(uu)))
        return float(np.mean(r[np.isfinite(r)]))
    eA0, eC0 = ((F["clean"] - y) ** 2).mean(1), ((C["clean"] - y) ** 2).mean(1)
    rows, res = [], {}
    for kind in KINDS[1:]:
        c = CS[kind]
        FA, CA = F[kind] / c[:, None], C[kind] / c[:, None]
        eA, eC = ((FA - y) ** 2).mean(1), ((CA - y) ** 2).mean(1)
        with np.errstate(all="ignore"):
            lch = 100 * (LEV[kind] / c - LEV["clean"]) / np.abs(LEV["clean"])
            pch = 100 * (PK[kind] / c - PK["clean"]) / np.abs(PK["clean"])
        r = {"ankyra_error_vs_clean_log_ratio": unit_mean_log(eA, eA0), "carrier_error_vs_clean_log_ratio": unit_mean_log(eC, eC0),
             "ankyra_windows_worse_share": float((eA > eA0 * 1.0001).mean()), "carrier_windows_worse_share": float((eC > eC0 * 1.0001).mean()),
             "ankyra_displacement_over_s0_median": float(np.median(np.sqrt(((FA - F["clean"]) ** 2).mean(1)) / s0)),
             "carrier_displacement_over_s0_median": float(np.median(np.sqrt(((CA - C["clean"]) ** 2).mean(1)) / s0)),
             "level_change_pct_median": float(np.nanmedian(lch)), "level_change_pct_max": float(np.nanmax(np.abs(lch))),
             "peak_readout_change_pct_median": float(np.nanmedian(pch)),
             "peak_error_vs_clean_log_ratio": unit_mean_log((PK[kind] / c - ypk) ** 2, (PK["clean"] - ypk) ** 2),
             "off_state_fired": int(OFF[kind].sum()), "micro_load_fired": int(MIC[kind].sum()),
             "ankyra_vs_carrier_corrupted_log_ratio": unit_mean_log(eA, eC)}
        if kind.startswith("scale"):
            r["equivariance_max_rel_dev"] = float(np.max(np.abs(FA - F["clean"]) / (np.abs(F["clean"]) + 1e-9)))
        res[kind] = r
        rows.append({"corruption": kind, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()},
                     "windows": int(m), "units": int(len(uu)), "set": T.names[face]})
    total = time.perf_counter() - t_all
    rec = {"face": face, "face_name": T.names[face], "windows": int(m), "units": int(len(uu)), "seed": SEED, "checkpoint": entry["checkpoint"],
           "cut": cut, "corruptions": res, "clean": {"ankyra_vs_carrier_log_ratio": unit_mean_log(eA0, eC0), "off_state_fired": int(OFF["clean"].sum()),
                                                     "micro_load_fired": int(MIC["clean"].sum()),
                                                     "clean_forecast_vs_stored_product_max_rel": clean_rel},
           "seconds_by_corruption": sec, "seconds_total": total, "seconds_per_window_and_corruption": float(np.mean(list(sec.values()))) / max(m, 1),
           "device": "cpu", "threads": torch.get_num_threads(), "window_sample": "evenly spread over the test windows in the product's order"}
    if layer or len(models) != 3:                                            # version 0.2.0 fields
        rec |= {"seeds": len(models), "continuation_layer": layer}
        if layer:
            rec["clean"]["clean_delivered_vs_stored_continued_product_max_rel"] = cont_rel
    return rec, rows


def summary(rec):
    if not rec:
        return []
    c = rec["corruptions"]
    lab = {"miss05": "5% of hours missing, filled by linear interpolation", "miss20": "20% of hours missing, filled",
           "zero24": "a 24-hour zero-filled block in the last week", "shift_plus1": "context shifted by +1 hour",
           "shift_minus1": "context shifted by -1 hour", "spike": "one spike at 10x the context maximum, 36 hours before the origin",
           "scale0.5": "whole history scaled by 0.5", "scale2": "whole history scaled by 2"}
    pct = lambda v: f"{100 * (np.exp(v) - 1):+.1f}%"                                  # noqa: E731  (positive = worse)
    L = ["## Robustness to corrupted contexts (`robustness.csv`)", "",
         f"{rec['windows']} {rec['face_name']} test windows ({rec['units']} units), only the context corrupted, re-forecast on the CPU with the",
         "released checkpoints (carrier forecasts at every issue time, anchoring layer, state rule"
         + (", continuation layer" if rec.get("continuation_layer") else "") + "). Change of the hourly RMS error",
         "against the clean context (positive = worse); the comparator is the carrier alone from the same corrupted record.", "",
         "| Corruption of the context | ankyra-lite: hourly error vs clean | carrier alone | level change, median / max | peak readout change, median | ankyra-lite vs carrier alone (corrupted) |",
         "|---|---:|---:|---:|---:|---:|"]
    for k in KINDS[1:]:
        r = c[k]
        L.append(f"| {lab[k]} | {pct(r['ankyra_error_vs_clean_log_ratio'])} | {pct(r['carrier_error_vs_clean_log_ratio'])} | "
                 f"{r['level_change_pct_median']:+.2f}% / {r['level_change_pct_max']:.1f}% | {r['peak_readout_change_pct_median']:+.2f}% | "
                 f"{pct(r['ankyra_vs_carrier_corrupted_log_ratio'])} |")
    eq = max(c[k].get("equivariance_max_rel_dev", 0.0) for k in ("scale0.5", "scale2"))
    L += ["", f"Largest deviation from exact scale equivariance: {eq:.2e}. State rules fired: "
          + ", ".join(f"{k} {c[k]['off_state_fired'] + c[k]['micro_load_fired']}" for k in KINDS[1:]) + ". The clean forecasts match the stored "
          f"evaluation product to a relative difference of {rec['clean']['clean_forecast_vs_stored_product_max_rel']:.1e}. CPU time "
          f"{rec['seconds_total'] / 60:.1f} min ({rec['seconds_per_window_and_corruption']:.2f} s per window and corruption).", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", default=FACE)
    ap.add_argument("--windows", type=int, default=N)
    ap.add_argument("--recipe", default="R", choices=sorted(CB.RECIPE_PRODUCTS))
    ap.add_argument("--products", default=None, help="another products mapping (overrides --recipe)")
    ap.add_argument("--out", required=True, help="output directory (robustness.csv and robustness.json)")
    a = ap.parse_args()
    CB.RESULTS = Path(a.out).resolve()
    if CB._in_read_only(CB.RESULTS):
        raise SystemExit("the output directory lies in a read-only tree")
    CB.RESULTS.mkdir(parents=True, exist_ok=True)
    prods = CB.load_products(Path(a.products) if a.products else CB.RECIPE_PRODUCTS[a.recipe])
    CB.wait_for_memory("robustness")
    T = CB.Tools()
    rec, rows = measure(T, a.face, prods["faces"][a.face], n=a.windows, log=CB.log)
    CB.write_rows("robustness.csv", rows, FIELDS)
    CB.dump_json(CB.RESULTS / "robustness.json", rec)
    print("\n".join(summary(rec)))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    sys.modules.setdefault("compare_baselines", CB)
    main()
