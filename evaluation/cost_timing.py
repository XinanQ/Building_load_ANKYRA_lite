"""cost_per_window.csv: what one ankyra-lite forecast costs, measured with this repository's package on the CPU, one thread.

On a handful of test windows of one population (Oslo by default) the checkpoints of every seed of the evaluated product
(the weights file the products mapping names, ``final.pt`` of the stage-2 runs; three seeds in 0.1.x, six in a six-seed
0.2.0 configuration) are loaded with ``ankyra_lite.load_carrier``; per window, ``ankyra_lite.carrier_forecasts``
computes the carrier forecasts at the origin and the pseudo-origins the anchoring layer reads (up to 7 issue times x the
seeds), and ``ankyra_lite.forecast`` runs the anchoring layer, the state rule, the peak readout and, when the mapping
switches it on, the continuation layer on them.  One untimed call warms up first.  The carrier forecasts are compared
with the stored evaluation forecasts (computed on a GPU), the trajectory before the continuation layer with the stored
product and, with the layer, the delivered trajectory with the stored continued product (the study applied the layer to
the float32-rounded trajectory, so the two differ by about 1e-7 relative).  Called by extra_tables.Collector.finish; no
result is written there.

No target is read: every carrier and anchoring call gets the unit's load before the origin only (``load[b, :o]``); the
temperature and day-type rows are the unit's (the day types run through the forecast month, a calendar).

``python -B evaluation/cost_timing.py --products <mapping.json> --face oslo --out <file.json>`` times alone and saves
the record and the rows; ``compare_baselines.py --cost-record <file.json>`` then writes cost_per_window.csv from it.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import argparse                                                    # noqa: E402
import importlib                                                   # noqa: E402
import json                                                        # noqa: E402
import time                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

import compare_baselines as CB                                     # noqa: E402

if str(CB.REPO) not in sys.path:
    sys.path.append(str(CB.REPO))                                # this repository's package (ankyra_lite)

COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _forecast_with_layer(FM, history, layer, *args, **kwargs):
    """Call the public forecast API with an explicit layer setting, including when it is off."""
    return FM.forecast(history, *args, continuation=bool(layer), **kwargs)


def _study_loader():
    for p in (str(CB.REL), str(CB.HKAD)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tools_smallnet_carrier_v1 as V                     # the study's store loader (the module tools_smallnet_pwformal_v1 names V)
    return V


def _row(a, b):
    a = np.asarray(a)
    return a[b] if a.ndim == 2 else a


def measure(T, face, entry, n=6, layer=None):
    """Time ankyra-lite on n + 1 test windows of ``face`` (the first is the untimed warm-up).  ``T``: the study's tools
    (compare_baselines.Tools) or None (the study's store loader is imported directly; nothing else is needed).
    ``layer``: the continuation layer on or off (default: the mapping's)."""
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
    layer = bool(entry.get("layer")) if layer is None else bool(layer)
    V = T.pwf().V if T is not None else _study_loader()
    names = T.names if T is not None else CB.SHOW_NAMES
    d = LR.store_arrays(face)                                                            # the store files alone (no window arrays, no results file)
    region = V._region(face, d)
    q0 = np.load(entry["run_paths"][0] / "carrier.npz", allow_pickle=False)
    cut = int(q0["cut"])
    sigma = float(json.loads(Path(entry["prep_path"]).read_text(encoding="utf-8"))["sigma"])
    zx = np.load(entry["npz_path"], allow_pickle=False)
    zc = np.load(entry["continued_path"], allow_pickle=False) if (layer and entry.get("continued_path") is not None) else None
    pick = np.unique(np.linspace(0, len(zx["building"]) - 1, n + 1).round().astype(int))
    wb, wo = zx["building"][pick].astype(int), zx["origin"][pick].astype(int)
    d["load"] = LR.history_only(d["load"], zx["building"], zx["origin"])                 # no hour at or after a unit's last test origin stays
    Sp, Sh = LR.carriers(entry, None, wb, wo)
    t0 = time.perf_counter()
    models = [load_carrier(Path(rd) / entry["checkpoint"]) for rd in entry["run_paths"]]
    t_load = time.perf_counter() - t0
    params = parameter_count(models[0])
    size_mb = float(np.mean([os.path.getsize(Path(rd) / entry["checkpoint"]) for rd in entry["run_paths"]])) / 1e6

    def one(i):
        b, o = int(wb[i]), int(wo[i])
        pre_load = np.asarray(d["load"][b, :o])                                          # the only load a window reads
        ty = _row(d["types"], b)
        tmp = _row(d["temp"], b)
        t1 = time.perf_counter()
        F = carrier_forecasts(models, pre_load, tmp, ty, d["groups"][b], d["start_iso"], unit=0, origin=o, cut=cut)
        t2 = time.perf_counter()
        h = History(load_kw=pre_load, temperature_c=tmp[:o], day_types=ty[:o + CB.HORIZON], start_timestamp=d["start_iso"],
                    observed=np.isfinite(pre_load))
        t3 = time.perf_counter()
        r = _forecast_with_layer(FM, h, layer, F, group=d["groups"][b], temp_sigma_std=sigma, dst_region=region(b), cut=cut)
        t4 = time.perf_counter()
        FM.peak(getattr(r, "trajectory_before_continuation_kw", r.trajectory_kw), FM._masked_load(h), h.day_types)
        t5 = time.perf_counter()
        rel = max(float(np.abs(F[k] - Sp[i, k]).max() / max(np.abs(Sp[i, k]).mean(), 1e-9)) for k in F)
        same_k = sorted(F) == [k for k in range(Sp.shape[1]) if Sh[i, k]]
        stored = zx["traj"][pick[i]].astype(np.float64)
        before = np.asarray(getattr(r, "trajectory_before_continuation_kw", r.trajectory_kw))
        trel = float(np.abs(before - stored).max() / max(np.abs(stored).mean(), 1e-9)) if not r.state_window else float("nan")
        crel = float("nan")
        if zc is not None and not r.state_window:
            sc = zc["traj"][pick[i]].astype(np.float64)
            crel = float(np.abs(np.asarray(r.trajectory_kw) - sc).max() / max(np.abs(sc).mean(), 1e-9))
        return {"carrier": t2 - t1, "anchor": t4 - t3, "peak": t5 - t4, "issue_times": len(F), "carrier_rel": rel, "same_issue_times": same_k,
                "traj_rel": trel, "cont_rel": crel}

    one(0)                                                                               # warm-up, not timed
    res = [one(i) for i in range(1, len(wb))]
    car = float(np.mean([r["carrier"] for r in res]))
    anc = float(np.mean([r["anchor"] - r["peak"] for r in res]))
    pk = float(np.mean([r["peak"] for r in res]))
    ki = float(np.mean([r["issue_times"] for r in res]))
    m = len(models)
    rec = {"face": face, "face_name": names[face], "windows": len(res), "device": "cpu", "threads": torch.get_num_threads(), "parameters": params,
           "checkpoint_mb": size_mb, "model_load_s": t_load, "carrier_s_per_window": car, "issue_times_mean": ki,
           "carrier_s_per_issue_time_and_seed": car / (ki * m), "anchor_s_per_window": anc, "peak_s_per_window": pk,
           "total_s_per_window": car + anc + pk, "carrier_vs_stored_max_rel": max(r["carrier_rel"] for r in res),
           "issue_times_as_stored": all(r["same_issue_times"] for r in res),
           "trajectory_vs_stored_product_max_rel": float(np.nanmax([r["traj_rel"] for r in res])), "torch": torch.__version__,
           "package": getattr(ankyra_lite, "__version__", ""), "checkpoint": entry["checkpoint"]}
    if m != 3 or layer:                                                                  # version 0.2.0 fields
        rec |= {"seeds": m, "continuation_layer": layer}
    if layer:
        rec["delivered_vs_stored_continued_product_max_rel"] = float(np.nanmax([r["cont_rel"] for r in res])) if zc is not None else None
    w = COUNT_WORDS.get(m, str(m))
    rows = [{"item": "carrier network parameters", "value": params, "unit": "parameters"},
            {"item": "checkpoint size (one seed)", "value": round(size_mb, 4), "unit": "MB"},
            {"item": "device: CPU, one thread (no GPU)", "value": rec["threads"], "unit": "threads"},
            {"item": f"windows timed ({names[face]} test windows, after one untimed warm-up)", "value": len(res), "unit": "windows"},
            {"item": "ankyra-lite forecast (carrier forecasts, anchoring layer, " + ("continuation layer, " if layer else "") + "peak readout)",
             "value": round(rec["total_s_per_window"], 6), "unit": "s/window"},
            {"item": f"component: carrier forecasts at the origin and the pseudo-origins ({ki:.1f} issue times on average) x {m} seeds", "value": round(car, 6), "unit": "s/window"},
            {"item": "component: carrier forecast, one issue time and one seed (mean)", "value": round(rec["carrier_s_per_issue_time_and_seed"], 6), "unit": "s"},
            {"item": "component: anchoring layer call with the carrier forecasts supplied (with the state rule" + (" and the continuation layer" if layer else "") + ")",
             "value": round(anc, 6), "unit": "s/window"},
            {"item": "component: peak readout", "value": round(pk, 6), "unit": "s/window"},
            {"item": f"model load: {w} checkpoints", "value": round(t_load, 4), "unit": "s"},
            {"item": "check: carrier forecasts against the stored evaluation forecasts (largest relative difference)", "value": float(f"{rec['carrier_vs_stored_max_rel']:.3g}"), "unit": "ratio"},
            {"item": "check: " + ("trajectory before the continuation layer" if layer else "delivered trajectory") + " against the stored product (largest relative difference)",
             "value": float(f"{rec['trajectory_vs_stored_product_max_rel']:.3g}"), "unit": "ratio"}]
    if layer and rec.get("delivered_vs_stored_continued_product_max_rel") is not None:
        rows.append({"item": "check: delivered trajectory against the stored continued product (largest relative difference; float64 against the float32 product path)",
                     "value": float(f"{rec['delivered_vs_stored_continued_product_max_rel']:.3g}"), "unit": "ratio"})
    return rec, rows


def load_record(path, prods):
    """(record, rows) of a saved timing (``--out``), checked against the products mapping: the same product file and
    the same seeds."""
    x = json.loads(Path(path).read_text(encoding="utf-8"))
    e = prods["faces"][x["face"]]
    if x.get("product_sha256") != _sha256(e["npz_path"]) or x.get("mapping_label") != prods.get("label") or x.get("configuration") != prods.get("configuration"):
        raise SystemExit(f"{path}: the timing record belongs to another product or configuration")
    return x["record"], x["rows"]


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="time ankyra-lite on a few test windows (CPU, one thread) and save the record")
    ap.add_argument("--products", default=None, help="products mapping (default: evaluation/products.json)")
    ap.add_argument("--face", default="oslo")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--out", required=True, help="the JSON record to write (outside the research root)")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if CB._in_read_only(out):
        raise SystemExit("the output file lies in a read-only tree")
    prods = CB.load_products(Path(a.products) if a.products else CB.PRODUCTS)
    e = prods["faces"][a.face]
    CB.wait_for_memory("cost timing")
    rec, rows = measure(None, a.face, e, n=a.windows)
    x = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "face": a.face, "mapping_label": prods.get("label"),
         "configuration": prods.get("configuration"), "product_sha256": _sha256(e["npz_path"]), "record": rec, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(CB.scrub(json.dumps(x, indent=1, ensure_ascii=False)).encode("utf-8"))
    os.replace(tmp, out)
    print(json.dumps({k: rec[k] for k in ("windows", "total_s_per_window", "carrier_vs_stored_max_rel", "trajectory_vs_stored_product_max_rel")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
