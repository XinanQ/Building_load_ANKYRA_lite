"""Re-anchoring of ankyra-lite's saved carrier forecasts with the unchanged anchoring layer (no training, no new carrier
forecast).  Used by the ablation, seed-spread, component and anchoring-record tables of compare_baselines.py.

For each population of the products mapping and each mode, the carrier forecasts at the origin and the six
pseudo-origins (the run directories' carrier.npz, the same arrays the evaluated product was anchored with) are given to
``ankyra_lite.anchor.forecast`` window by window, exactly as the study's anchoring step did
(tools_smallnet_sr_formal_v1._anchor_work: same history record, temp_sigma_std, daylight-saving region, peak readout):

  ens         the three-seed ensemble with the default anchoring layer: the evaluated product itself, re-run.  Gate: it
              must reproduce the stored product bit for bit; it also records the anchoring record (``anchoring_record``)
  s0, s1, s2  one seed's carrier alone, default anchoring layer.  Where the mapping lists the stored single-seed
              products (``seed_npz``), the tables read those and these modes are not run in full; a run on the first
              windows only (``--limit``, cache name ``s<k>_first<n>``) is then a gate: it must reproduce the stored
              single-seed product bit for bit
  no_within   ensemble, ``within_anchor=False`` (the carrier's within-day block is kept)
  lead_trust  ensemble, ``single_trust=False`` (within-day trust per lead block instead of one value per window)
  no_gap      ensemble, ``gap_tolerance=False`` (no short-gap rule in the pseudo-origin bookkeeping)
  no_micro    ensemble, ``micro_load_rule=False`` (identical by construction where no window is a micro-load window;
              then not computed)

The seasonal-naive state rule is applied when the tables are scored, not here.  The cache lives in
ANKYRA_LITE_EVAL_CACHE (default: the system temporary directory), one npz per population and mode, keyed by the
product's label and checked against the product file's SHA-256.

Usage:  python -B evaluation/lite_reanchor.py --faces oslo [--modes ens,s0] [--limit 8] [--workers 2] [--recipe p3s]
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import compare_baselines as CB                                     # noqa: E402  (environment, read-only guard, paths)

import argparse                                                    # noqa: E402
import hashlib                                                     # noqa: E402
import json                                                        # noqa: E402
import math                                                        # noqa: E402
import re                                                          # noqa: E402
import time                                                        # noqa: E402
import warnings                                                    # noqa: E402
from concurrent.futures import ProcessPoolExecutor                 # noqa: E402
from pathlib import Path                                           # noqa: E402

import numpy as np                                                 # noqa: E402

MODES = {"ens": (None, {}), "s0": ((0,), {}), "s1": ((1,), {}), "s2": ((2,), {}),
         "no_within": (None, {"within_anchor": False}), "lead_trust": (None, {"single_trust": False}),
         "no_gap": (None, {"gap_tolerance": False}), "no_micro": (None, {"micro_load_rule": False})}
LABELS = {"ens": "three-seed ensemble (the product, re-run)", "s0": "carrier of seed 0 alone", "s1": "carrier of seed 1 alone",
          "s2": "carrier of seed 2 alone", "no_within": "anchoring layer without the within-day anchoring",
          "lead_trust": "within-day trust per lead block instead of one value", "no_gap": "anchoring layer without the short-gap rule",
          "no_micro": "anchoring layer without the micro-load rule"}
STATS = ("carrier_level_weight", "history_vs_carrier_log_ratio", "pseudo_origins")
SEED_MODES = ("s0", "s1", "s2")
K, HORIZON, CONTEXT, HR = 6, CB.HORIZON, CB.CONTEXT, CB.HR


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_file(label, face, mode, limit=None):
    name = mode if limit is None else f"{mode}_first{int(limit)}"
    return CB.CACHE / re.sub(r"[^A-Za-z0-9._-]+", "_", label) / face / f"{name}.npz"


def default_modes(entry):
    """All modes, except the single seeds when the mapping lists stored single-seed products (they are read instead)."""
    return [m for m in MODES if not (m in SEED_MODES and entry.get("seed_paths"))]


def _gate(traj, level, peak, off, micro, zx, n):
    g = {"traj_equal": bool(np.array_equal(traj, zx["traj"][:n])),
         "traj_max_abs_diff": float(np.nanmax(np.abs(traj.astype(np.float64) - zx["traj"][:n]))),
         "level_equal": bool(np.array_equal(level, zx["level"][:n])), "peak_equal": bool(np.array_equal(peak, zx["peak"][:n])),
         "state_flags_equal": bool(np.array_equal(off, zx["off"][:n]) and np.array_equal(micro, zx["micro"][:n])), "windows": int(n)}
    g["pass"] = bool(g["traj_equal"] and g["level_equal"] and g["state_flags_equal"])
    return g


def _paths():
    for p in (str(CB.REPO), str(CB.REL), str(CB.HKAD)):
        if p not in sys.path:
            sys.path.insert(0, p)


def face_data(face, entry):
    """The anchoring inputs of one population, as the study's anchoring step loaded them (tools_ankyra_simplify_v1.face);
    on the corrected CINELDI store, load, temperature, day types and start timestamp come from that store."""
    _paths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tools_ankyra_simplify_v1 as S
        d = S.face(face, "timesfm")
    if face == "cineldi" and entry["store"] == "corrected":
        store, _ = CB.corrected_store_info()
        z = np.load(Path(store) / "store.npz", allow_pickle=False)
        d = dict(d)
        d["load"] = z["load"].astype(np.float64)
        d["temp"] = z["temp"].astype(np.float32)
        d["types"] = z["types"].astype(np.int64)
        d["start_iso"] = str(z["start_iso"])
    sigma = float(json.loads(Path(entry["prep_path"]).read_text(encoding="utf-8"))["sigma"])
    return d, sigma


def carriers(entry, seeds, wb, wo):
    """Sum in seed order in float64 over the chosen seeds, divided by their number (the study's ensemble), and the
    pseudo-origins every chosen seed has."""
    Sp, Sh = None, None
    runs = entry["run_paths"] if seeds is None else [entry["run_paths"][s] for s in seeds]
    for rd in runs:
        q = np.load(Path(rd) / "carrier.npz", allow_pickle=False)
        if not (np.array_equal(q["building"], wb) and np.array_equal(q["origin"], wo)):
            key = {(int(b), int(o)): i for i, (b, o) in enumerate(zip(q["building"], q["origin"]))}
            ix = np.array([key[(int(b), int(o))] for b, o in zip(wb, wo)])
            pred, have = q["pred"][ix], q["have"][ix]
        else:
            pred, have = q["pred"], q["have"]
        Sp = pred.astype(np.float64) if Sp is None else Sp + pred
        Sh = have if Sh is None else Sh & have
    return Sp / len(runs), Sh


_D = {}


def _work(args):
    face, mode, idx, entry, wb, wo = args
    _paths()
    import torch
    from ankyra_lite import anchor as A
    from ankyra_lite.anchor import readouts as RO
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    if (face, mode) not in _D:
        _D.clear()
        d, sg = face_data(face, entry)
        seeds, _ = MODES[mode]
        Sp, Sh = carriers(entry, seeds, wb, wo)
        _D[(face, mode)] = (d, sg, Sp, Sh)
    d, sg, Sp, Sh = _D[(face, mode)]
    kw = MODES[mode][1]
    out = []
    for i in idx:
        b, o = int(wb[i]), int(wo[i])
        ty = d["types"] if d["types"].ndim == 1 else d["types"][b]
        tmp = d["temp"][b] if d["temp"].ndim == 2 else d["temp"]
        h = A.History(load_kw=d["load"][b, :o], temperature_c=tmp[:o], day_types=ty[:o + HORIZON], start_timestamp=d["start_iso"],
                      observed=np.isfinite(d["load"][b, :o]))
        F = {k: Sp[i, k] for k in range(K + 1) if Sh[i, k]}
        row = {"i": i, "err": ""}
        try:
            r = A.forecast(h, foundation=F, group=d["groups"][b], temp_sigma_std=sg, dst_region=d["region"](b), **kw)
        except Exception as e:                                                     # noqa: BLE001
            row["err"] = type(e).__name__
            out.append(row)
            continue
        ctx = d["load"][b, o - CONTEXT:o]
        ct = ty[o - CONTEXT:o:HR].astype(np.int64)
        tt = ty[o:o + HORIZON:HR].astype(np.int64)
        tr = np.asarray(r.trajectory_kw, np.float64)
        try:
            pk = float(RO.peak_readout(tr, ctx[None], ct[None], tt[None])[0])
        except Exception as e:                                                     # noqa: BLE001
            pk = float("nan")
            row["err"] += f"peak:{type(e).__name__};"
        rec = r.anchoring_record or {}
        row.update(traj=tr.astype(np.float32), level=float(r.level_kw), peak=pk, off=bool(r.off_state), micro=bool(r.micro_load),
                   stats=[float(rec.get(k, float("nan"))) for k in STATS])
        out.append(row)
    return out


def run_mode(face, mode, entry, label, workers=2, limit=None):
    """Anchor one population in one mode (cached).  Returns the record."""
    f = cache_file(label, face, mode, limit)
    meta = f.with_suffix(".json")
    psha = _sha(entry["npz_path"])
    if f.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("product_sha256") == psha and m.get("limit") == limit:
            return m
    zx = np.load(entry["npz_path"], allow_pickle=False)
    wb, wo = zx["building"].astype(np.int64), zx["origin"].astype(np.int64)
    if mode == "no_micro" and not zx["micro"].any():
        m = {"face": face, "mode": mode, "label": LABELS[mode], "identical_to": "ens", "reason": "no micro-load window in the product",
             "product_sha256": psha, "limit": limit}
        f.parent.mkdir(parents=True, exist_ok=True)
        meta.write_bytes(json.dumps(m, indent=1).encode("utf-8"))
        return m
    n = len(wb) if limit is None else min(int(limit), len(wb))
    e = {k: v for k, v in entry.items() if k in ("store", "prep_path", "run_paths", "npz_path")}
    e = {k: (str(v) if isinstance(v, Path) else ([str(x) for x in v] if k == "run_paths" else v)) for k, v in e.items()}
    t0, tc = time.time(), time.process_time()
    res = [None] * n
    if workers <= 1:
        parts = [_work((face, mode, list(range(n)), e, wb, wo))]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_work, [(face, mode, list(range(s, n, workers)), e, wb, wo) for s in range(workers)]))
    for part in parts:
        for r in part:
            res[r["i"]] = r
    errs = [(r["i"], r["err"]) for r in res if r["err"]]
    ok = [r for r in res if not r["err"]]
    traj = np.full((n, HORIZON), np.nan, np.float32)
    level, peak = np.full(n, np.nan), np.full(n, np.nan)
    off, micro = np.zeros(n, bool), np.zeros(n, bool)
    stats = np.full((n, len(STATS)), np.nan)
    for r in ok:
        i = r["i"]
        traj[i], level[i], peak[i], off[i], micro[i], stats[i] = r["traj"], r["level"], r["peak"], r["off"], r["micro"], r["stats"]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".part.npz")
    np.savez_compressed(tmp, building=wb[:n], origin=wo[:n], traj=traj, level=level, peak=peak, off=off, micro=micro, stats=stats)
    os.replace(tmp, f)
    m = {"face": face, "mode": mode, "label": LABELS[mode], "kwargs": MODES[mode][1], "seeds": MODES[mode][0], "windows": n,
         "errors": errs[:20], "error_count": len(errs), "wall_seconds": time.time() - t0, "workers": workers,
         "product_sha256": psha, "limit": limit, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if mode == "ens":                                                    # gate: the stored product, bit for bit
        m["gate"] = _gate(traj, level, peak, off, micro, zx, n)
    elif mode in SEED_MODES and entry.get("seed_paths"):                 # gate: the stored single-seed product, bit for bit
        zs = np.load(entry["seed_paths"][int(mode[1:])], allow_pickle=False)
        if not (np.array_equal(zs["building"][:n], wb[:n]) and np.array_equal(zs["origin"][:n], wo[:n])):
            raise SystemExit(f"{face} {mode}: the stored single-seed product has other windows")
        m["gate"] = _gate(traj, level, peak, off, micro, zs, n)
        m["gate"]["against"] = CB.show(entry["seed_paths"][int(mode[1:])])
    meta.write_bytes(json.dumps(m, indent=1).encode("utf-8"))
    return m


def load_mode(label, face, mode, limit=None):
    """(arrays, record) of a cached mode, or (None, None); a mode identical to another returns the other's arrays."""
    f = cache_file(label, face, mode, limit)
    meta = f.with_suffix(".json")
    if not meta.exists():
        return None, None
    m = json.loads(meta.read_text(encoding="utf-8"))
    if m.get("identical_to"):
        z, _ = load_mode(label, face, m["identical_to"])
        return z, m
    if not f.exists():
        return None, None
    return dict(np.load(f, allow_pickle=False)), m


def run(faces, modes=None, workers=2, prods=None, limit=None, seed_gate=8):
    """Re-anchor the populations in the given modes (default: default_modes).  ``seed_gate`` > 0: where stored
    single-seed products are used, seed 0 is also re-anchored on the first ``seed_gate`` windows and must reproduce its
    stored product bit for bit (one worker; cache name s0_first<n>)."""
    prods = prods or CB.load_products()
    rec = {}
    for face in faces:
        entry = prods["faces"].get(face)
        if entry is None or not entry["npz_path"].exists():
            continue
        jobs = [(m, limit, workers) for m in (modes or default_modes(entry))]
        if seed_gate and entry.get("seed_paths") and not any(m in SEED_MODES for m, _, _ in jobs):
            jobs.append(("s0", int(seed_gate), 1))
        for mode, lim, wk in jobs:
            CB.wait_for_memory(f"re-anchor {face} {mode}")                     # ANKYRA_LITE_MIN_FREE_GB (default 5)
            t0 = time.time()
            m = run_mode(face, mode, entry, prods["label"], workers=wk, limit=lim)
            rec[f"{face}|{mode}" + ("" if lim is None else f"|first{lim}")] = m
            CB.log(f"re-anchor {face} {mode}{'' if lim is None else f' (first {lim} windows)'}: windows {m.get('windows')} "
                   f"errors {m.get('error_count')} gate {m.get('gate')} "
                   f"{m.get('identical_to') and 'identical to ' + m['identical_to'] or ''} {time.time() - t0:.0f} s")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N windows only (development, or the single-seed gate)")
    ap.add_argument("--recipe", default="R", choices=sorted(CB.RECIPE_PRODUCTS), help="products mapping (R: the release)")
    ap.add_argument("--seed-gate", type=int, default=8, help="windows of the single-seed gate (0: none)")
    a = ap.parse_args()
    out = run([f for f in a.faces.split(",") if f], a.modes.split(",") if a.modes else None, a.workers,
              prods=CB.load_products(CB.RECIPE_PRODUCTS[a.recipe]), limit=a.limit, seed_gate=a.seed_gate)
    print(json.dumps({k: {x: v.get(x) for x in ("windows", "error_count", "gate", "wall_seconds", "identical_to")} for k, v in out.items()}, indent=1))
