"""cost_per_window.csv: what one ankyra-lite forecast costs, measured with this repository's package on the CPU, one thread.

On a handful of test windows of one population (Oslo by default) the three seeds' checkpoints of the evaluated product
(the weights file the products mapping names, ``final.pt`` of the span-return runs) are loaded with
``ankyra_lite.load_carrier``; per window, ``ankyra_lite.carrier_forecasts`` computes the carrier forecasts
at the origin and the pseudo-origins the anchoring layer reads (up to 7 issue times x 3 seeds), and
``ankyra_lite.forecast`` runs the anchoring layer, the state rule and the peak readout on them.  One untimed call warms
up first.  The carrier forecasts are compared with the stored evaluation forecasts (computed on a GPU) and the delivered
trajectory with the stored product.  Called by extra_tables.Collector.finish; no result is written here.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import compare_baselines as CB

if str(CB.REPO) not in sys.path:
    sys.path.append(str(CB.REPO))                                # this repository's package (ankyra_lite)


def measure(T, face, entry, n=6):
    import torch
    torch.set_num_threads(1)
    import ankyra_lite
    from ankyra_lite import load_carrier, carrier_forecasts
    from ankyra_lite.anchor import History
    from ankyra_lite.carrier.model import parameter_count
    import lite_reanchor as LR
    FM = importlib.import_module("ankyra_lite.forecast")
    if CB.on_corrected_cineldi(face, entry):
        raise SystemExit("cost timing: choose a population whose own data are on their original store")
    V = T.pwf().V
    d = V.raw(face)
    region = V._region(face, d)
    q0 = np.load(entry["run_paths"][0] / "carrier.npz", allow_pickle=False)
    cut = int(q0["cut"])
    sigma = float(json.loads(Path(entry["prep_path"]).read_text(encoding="utf-8"))["sigma"])
    zx = np.load(entry["npz_path"], allow_pickle=False)
    pick = np.unique(np.linspace(0, len(zx["building"]) - 1, n + 1).round().astype(int))
    wb, wo = zx["building"][pick].astype(int), zx["origin"][pick].astype(int)
    Sp, Sh = LR.carriers(entry, None, wb, wo)
    t0 = time.perf_counter()
    models = [load_carrier(Path(rd) / entry["checkpoint"]) for rd in entry["run_paths"]]
    t_load = time.perf_counter() - t0
    params = parameter_count(models[0])
    size_mb = float(np.mean([os.path.getsize(Path(rd) / entry["checkpoint"]) for rd in entry["run_paths"]])) / 1e6

    def one(i):
        b, o = int(wb[i]), int(wo[i])
        ty = d["types"] if d["types"].ndim == 1 else d["types"][b]
        tmp = d["temp"][b] if d["temp"].ndim == 2 else d["temp"]
        t1 = time.perf_counter()
        F = carrier_forecasts(models, d["load"], d["temp"], d["types"], d["groups"], d["start_iso"], unit=b, origin=o, cut=cut)
        t2 = time.perf_counter()
        h = History(load_kw=d["load"][b, :o], temperature_c=tmp[:o], day_types=ty[:o + CB.HORIZON], start_timestamp=d["start_iso"],
                    observed=np.isfinite(d["load"][b, :o]))
        t3 = time.perf_counter()
        r = FM.forecast(h, F, group=d["groups"][b], temp_sigma_std=sigma, dst_region=region(b), cut=cut)
        t4 = time.perf_counter()
        FM.peak(r.trajectory_kw, FM._masked_load(h), h.day_types)
        t5 = time.perf_counter()
        rel = max(float(np.abs(F[k] - Sp[i, k]).max() / max(np.abs(Sp[i, k]).mean(), 1e-9)) for k in F)
        same_k = sorted(F) == [k for k in range(Sp.shape[1]) if Sh[i, k]]
        stored = zx["traj"][pick[i]].astype(np.float64)
        trel = float(np.abs(np.asarray(r.trajectory_kw) - stored).max() / max(np.abs(stored).mean(), 1e-9)) if not r.state_window else float("nan")
        return {"carrier": t2 - t1, "anchor": t4 - t3, "peak": t5 - t4, "issue_times": len(F), "carrier_rel": rel, "same_issue_times": same_k, "traj_rel": trel}

    one(0)                                                                               # warm-up, not timed
    res = [one(i) for i in range(1, len(wb))]
    car = float(np.mean([r["carrier"] for r in res]))
    anc = float(np.mean([r["anchor"] - r["peak"] for r in res]))
    pk = float(np.mean([r["peak"] for r in res]))
    ki = float(np.mean([r["issue_times"] for r in res]))
    rec = {"face": face, "face_name": T.names[face], "windows": len(res), "device": "cpu", "threads": torch.get_num_threads(), "parameters": params,
           "checkpoint_mb": size_mb, "model_load_s": t_load, "carrier_s_per_window": car, "issue_times_mean": ki,
           "carrier_s_per_issue_time_and_seed": car / (ki * len(models)), "anchor_s_per_window": anc, "peak_s_per_window": pk,
           "total_s_per_window": car + anc + pk, "carrier_vs_stored_max_rel": max(r["carrier_rel"] for r in res),
           "issue_times_as_stored": all(r["same_issue_times"] for r in res),
           "trajectory_vs_stored_product_max_rel": float(np.nanmax([r["traj_rel"] for r in res])), "torch": torch.__version__,
           "package": getattr(ankyra_lite, "__version__", ""), "checkpoint": entry["checkpoint"]}
    rows = [{"item": "carrier network parameters", "value": params, "unit": "parameters"},
            {"item": "checkpoint size (one seed)", "value": round(size_mb, 4), "unit": "MB"},
            {"item": "device: CPU, one thread (no GPU)", "value": rec["threads"], "unit": "threads"},
            {"item": f"windows timed ({T.names[face]} test windows, after one untimed warm-up)", "value": len(res), "unit": "windows"},
            {"item": "ankyra-lite forecast (carrier forecasts, anchoring layer, peak readout)", "value": round(rec["total_s_per_window"], 6), "unit": "s/window"},
            {"item": f"component: carrier forecasts at the origin and the pseudo-origins ({ki:.1f} issue times on average) x 3 seeds", "value": round(car, 6), "unit": "s/window"},
            {"item": "component: carrier forecast, one issue time and one seed (mean)", "value": round(rec["carrier_s_per_issue_time_and_seed"], 6), "unit": "s"},
            {"item": "component: anchoring layer call with the carrier forecasts supplied (with the state rule)", "value": round(anc, 6), "unit": "s/window"},
            {"item": "component: peak readout", "value": round(pk, 6), "unit": "s/window"},
            {"item": "model load: three checkpoints", "value": round(t_load, 4), "unit": "s"},
            {"item": "check: carrier forecasts against the stored evaluation forecasts (largest relative difference)", "value": float(f"{rec['carrier_vs_stored_max_rel']:.3g}"), "unit": "ratio"},
            {"item": "check: delivered trajectory against the stored product (largest relative difference)", "value": float(f"{rec['trajectory_vs_stored_product_max_rel']:.3g}"), "unit": "ratio"}]
    return rec, rows
