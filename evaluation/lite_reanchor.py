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
  fields      the ensemble call of ``ens`` again, keeping the anchoring layer's fields of every window
              (``daily_means_kw``, ``fixed_division_daily_means_kw``, ``within_day_kw``, ``foundation_within_day_kw``,
              ``analog_shape_kw``, ``lead_week_weights``, ``pseudo_pairs``, ``within_trust``, ``within_pseudo_pairs``,
              ``analog_kept``, ``level_weights``) and quantities rebuilt from the layer's public functions and the
              private helpers ``_masked`` and ``_level_parts`` (read only): the six handover pairs (fixed division,
              realised and carrier daily means at the completed pseudo-origins, ``history.estimate_from_history``), the
              three within-day triples (``analog.AnalogShapes``), the history-side daily means before the handover m^H,
              the seven level candidates at the origin, and the pseudo-origin sums of one whole-window combination weight
              for the history side and for three generic forecasters.  Gate: the delivered trajectory, level and state
              flags equal the stored product bit for bit, and the rebuilt pairs, triples and level reproduce the call's
              ``lead_week_weights``, ``within_trust``, ``level_weights`` and ``daily_means_kw`` exactly.  It is not
              an ablation; the tables of ``extra_tables.py`` read it (handover, lead weeks, within-day trust, whole-window
              weight, generic combinations, level-candidate hit rate)

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
FIELD_MODES = {"fields": (None, {})}             # not an ablation: read by the handover and combination tables only
SPECS = MODES | FIELD_MODES                      # every mode run_mode accepts
LABELS = {"ens": "three-seed ensemble (the product, re-run)", "s0": "carrier of seed 0 alone", "s1": "carrier of seed 1 alone",
          "s2": "carrier of seed 2 alone", "no_within": "anchoring layer without the within-day anchoring",
          "lead_trust": "within-day trust per lead block instead of one value", "no_gap": "anchoring layer without the short-gap rule",
          "no_micro": "anchoring layer without the micro-load rule",
          "fields": "three-seed ensemble (the product, re-run) with the anchoring layer's fields and the rebuilt pseudo-origin quantities"}
STATS = ("carrier_level_weight", "history_vs_carrier_log_ratio", "pseudo_origins")
SEED_MODES = ("s0", "s1", "s2")
K, HORIZON, CONTEXT, HR = 6, CB.HORIZON, CB.CONTEXT, CB.HR
DAYS = CB.DAYS
CANDIDATES = ("s_u", "l_u", "a_u", "s_w", "l_w", "a_w", "fm")   # the seven level candidates (level_weights keys)
GENERIC = ("SN-week", "WeekProf4", "LYR-profile")             # generic partners of the whole-window combination
YEAR_LAG = 364 * 24                                            # the last-year profile's lag (the panel's definition)
# per-window arrays of the fields mode: name -> (shape after the window axis, dtype, fill)
FIELD_ARRAYS = {
    "daily": ((DAYS,), np.float64, np.nan),              # daily_means_kw (after the handover)
    "fixed": ((DAYS,), np.float64, np.nan),              # fixed_division_daily_means_kw
    "mh": ((DAYS,), np.float64, np.nan),                 # history-side daily means before the handover (level with the carrier candidate + daily path)
    "within": ((HORIZON,), np.float64, np.nan),          # within_day_kw (delivered within-day block)
    "fwithin": ((HORIZON,), np.float64, np.nan),         # foundation_within_day_kw (the carrier's within-day block)
    "analog": ((HORIZON,), np.float64, np.nan),          # analog_shape_kw (S0; the carrier's block when the guard fell back)
    "alphas": ((4,), np.float64, np.nan),                # lead_week_weights
    "alpha_month": ((), np.float64, np.nan),             # one weight per window from the same pairs (the 31 days pooled)
    "npairs": ((), np.int64, 0), "trust": ((4,), np.float64, np.nan), "ntrip": ((), np.int64, 0),
    "kept": ((), bool, False), "lw": ((len(CANDIDATES),), np.float64, np.nan),
    "cand": ((len(CANDIDATES),), np.float64, np.nan),    # each level candidate's window mean at the origin (kW)
    "b2_sums": ((4,), np.float64, np.nan),               # daily num, den (handover pairs), within-day num, den (trust triples)
    "gen_sums": ((len(GENERIC), 4), np.float64, np.nan),  # the same sums with a generic forecaster in place of the history side
    "fields_ok": ((), bool, False),                      # a non-state window whose fields were recorded
    "rebuild_ok": ((), bool, False),                     # the rebuilt quantities reproduce the call's fields exactly
}


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
        seeds, _ = SPECS[mode]
        Sp, Sh = carriers(entry, seeds, wb, wo)
        _D[(face, mode)] = (d, sg, Sp, Sh)
    d, sg, Sp, Sh = _D[(face, mode)]
    kw = SPECS[mode][1]
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
        if mode in FIELD_MODES:
            try:
                row["fields"] = window_fields(h, F, r, d["groups"][b], sg, d["region"](b))
            except Exception as e:                                                 # noqa: BLE001
                row["ferr"] = f"{type(e).__name__}: {e}"
        out.append(row)
    return out


def x_forecast(name, load, q):
    """A generic forecaster issued at hour q from load[:q] only (the panel's definitions: seasonal naive of the last
    week, the mean of the last four weeks, the 744 hours one year (364 days) earlier, else the seasonal naive)."""
    ctx = load[q - CONTEXT:q]
    if name == "SN-week":
        return np.tile(ctx[-168:], math.ceil(HORIZON / 168))[:HORIZON].astype(np.float64)
    if name == "WeekProf4":
        return np.tile(ctx[-4 * 168:].reshape(4, 168).mean(0), math.ceil(HORIZON / 168))[:HORIZON].astype(np.float64)
    if name == "LYR-profile":
        s = q - YEAR_LAG
        if s >= 0:
            seg = load[s:s + HORIZON]
            if np.isfinite(seg).all():
                return seg.astype(np.float64)
        return x_forecast("SN-week", load, q)
    raise KeyError(name)


def window_fields(h, F, r, group, sigma, region):
    """The fields of one default anchoring call ``r`` (``ankyra_lite.anchor.forecast(h, foundation=F, ...)``) and the
    quantities rebuilt from the layer's own functions, exactly as ``core.forecast`` computes them: the masked record with
    short gaps filled, ``_level_parts`` (level candidates and the completed pseudo-origins), the six handover pairs
    (``history.estimate_from_history`` at each completed pseudo-origin) and the three within-day triples
    (``analog.AnalogShapes``).  ``rebuild_ok``: the rebuilt pairs give ``lead_week_weights``, the rebuilt triples
    ``within_trust``, the rebuilt level ``level_weights`` and the handover of m^H ``daily_means_kw``, all exactly.
    Reads no hour at or after the origin except the pseudo-origins' completed targets (all before the origin)."""
    import warnings as _w
    from dataclasses import replace
    from ankyra_lite.anchor import core as C, analog as AN, blocks as BL
    from ankyra_lite.anchor.history import History, estimate_from_history
    out = {k: (np.full(s, f, dt) if s else (dt(f) if dt is not bool else bool(f))) for k, (s, dt, f) in FIELD_ARRAYS.items()}
    if r.off_state or r.micro_load:
        return out
    hm = C._masked(h)
    load = C.fill_short_gaps(np.asarray(hm.load_kw, dtype=np.float64))
    obs = np.isfinite(load)
    hf = replace(hm, load_kw=load, observed=obs)
    temp = np.asarray(hf.temperature_c, dtype=np.float32)
    types = np.asarray(hf.day_types)
    o = len(load)
    T0 = np.asarray(F[0], dtype=np.float64)
    mT = T0.reshape(DAYS, HR).mean(1)
    parts = C._level_parts(hf, group, sigma, min_target_obs=C.TARGET_MIN_OBS)
    fm_ps = {k: float(F[k].mean()) for k in range(1, K + 1) if k in parts["pseudo"] and k in F}
    lvl, wl = C.level_with_model_candidate(parts, float(T0.mean()), fm_ps)
    fixed = np.asarray(r.fixed_division_daily_means_kw, dtype=np.float64)
    mh = fixed - fixed.mean() + lvl
    hh = parts["h"]
    cand = []
    for n in CANDIDATES[:-1]:
        v = float(hh[n][0]) if n in parts["names"] else float("nan")
        cand.append(float(parts["l0"] + parts["s0"] * (v + float(hh["c_bar"][0]) + (float(hh["g_h"][0].mean()) if n in ("s_w", "l_w", "a_w") else 0.0))))
    cand.append(float(T0.mean()))
    pairs, ks = [], []
    for k in range(1, K + 1):
        p = parts["pseudo"].get(k)
        if p is None or k not in F:
            continue
        ok = p["o"]
        ek = estimate_from_history(History(load_kw=load[:ok], temperature_c=temp[:ok], day_types=types[:ok + HORIZON],
                                           start_timestamp=hf.start_timestamp, observed=obs[:ok]), group=group, temp_sigma_std=sigma)
        yk = load[ok:ok + HORIZON].reshape(DAYS, HR)
        nk = np.isfinite(yk).sum(1)
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            yday = np.where(nk >= 12, np.nanmean(yk, axis=1), np.nan)
        pairs.append((yday, ek.level_kw + ek.daily_path_kw, np.asarray(F[k], dtype=np.float64).reshape(DAYS, HR).mean(1)))
        ks.append(k)
    aw = C.lead_week_weights(pairs) if pairs else [0.5] * 4
    num = den = 0.0
    for y_, hd, fd in pairs:                                           # the 31 days pooled (one weight per window)
        m = np.isfinite(y_)
        num += float(((y_ - hd)[m] * (fd - hd)[m]).sum())
        den += float(((fd - hd)[m] ** 2).sum())
    lam = min(max(num / den, 0.0), 1.0) if den > 0 else 0.5
    am = (len(pairs) * lam + C.K0_WEEK * 0.5) / (len(pairs) + C.K0_WEEK)
    wT0 = BL.within_day(T0)
    shapes = AN.AnalogShapes(load, temp, types, hf.start_timestamp, region, origin=o)
    S0, kept = shapes.shape_with_sanity(o, wT0)
    triples, tks = [], []
    for k in range(1, AN.K_WITHIN + 1):
        q = o - k * HORIZON
        if k not in F or q - CONTEXT < 0 or not (obs[q - CONTEXT:q + HORIZON].all() and np.isfinite(load[q - CONTEXT:q + HORIZON]).all()):
            continue
        wTq = BL.within_day(np.asarray(F[k], dtype=np.float64))
        Sq, _ = shapes.shape_with_sanity(q, wTq)
        triples.append((Sq, wTq, BL.within_day(load[q:q + HORIZON])))
        tks.append(k)
    wtr, n_w = AN.within_trust(triples, single=True)
    b2 = [num, den, sum(float((np.asarray(s_) - f_) @ (y_ - f_)) for s_, f_, y_ in triples),
          sum(float((np.asarray(s_) - f_) @ (np.asarray(s_) - f_)) for s_, f_, _ in triples)]
    gen = np.zeros((len(GENERIC), 4))
    for j, name in enumerate(GENERIC):
        for (y_, _, fd), k in zip(pairs, ks):
            xd = x_forecast(name, load, o - k * HORIZON).reshape(DAYS, HR).mean(1)
            m = np.isfinite(y_)
            gen[j, 0] += float(((y_ - xd)[m] * (fd - xd)[m]).sum())
            gen[j, 1] += float(((fd - xd)[m] ** 2).sum())
        for (_, f_, y_), k in zip(triples, tks):
            xw = BL.within_day(x_forecast(name, load, o - k * HORIZON))
            gen[j, 2] += float((y_ - xw) @ (f_ - xw))
            gen[j, 3] += float((f_ - xw) @ (f_ - xw))
    rebuild = (tuple(float(a) for a in aw) == tuple(r.lead_week_weights) and tuple(float(x) for x in wtr) == tuple(r.within_trust)
               and n_w == r.within_pseudo_pairs and len(pairs) == r.pseudo_pairs and dict(wl) == dict(r.level_weights)
               and np.array_equal(C.lead_week_transition(mh, mT, aw), np.asarray(r.daily_means_kw))
               and np.array_equal(np.asarray(S0), np.asarray(r.analog_shape_kw)) and bool(kept) == bool(r.analog_kept))
    out.update(daily=np.asarray(r.daily_means_kw, dtype=np.float64), fixed=fixed, mh=mh, within=np.asarray(r.within_day_kw, dtype=np.float64),
               fwithin=np.asarray(r.foundation_within_day_kw, dtype=np.float64),
               analog=np.asarray(r.analog_shape_kw, dtype=np.float64) if r.analog_shape_kw is not None else np.full(HORIZON, np.nan),
               alphas=np.asarray(r.lead_week_weights, dtype=np.float64), alpha_month=float(am), npairs=int(r.pseudo_pairs),
               trust=np.asarray(r.within_trust, dtype=np.float64), ntrip=int(r.within_pseudo_pairs), kept=bool(r.analog_kept),
               lw=np.array([float(r.level_weights.get(n, np.nan)) for n in CANDIDATES]), cand=np.array(cand), b2_sums=np.array(b2), gen_sums=gen,
               fields_ok=True, rebuild_ok=bool(rebuild))
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
    extra, ferrs = {}, []
    if mode in FIELD_MODES:
        extra = {k: np.full((n,) + s, fill, dt) for k, (s, dt, fill) in FIELD_ARRAYS.items()}
        for r in ok:
            if r.get("ferr"):
                ferrs.append((r["i"], r["ferr"]))
                continue
            for k in FIELD_ARRAYS:
                extra[k][r["i"]] = r["fields"][k]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".part.npz")
    np.savez_compressed(tmp, building=wb[:n], origin=wo[:n], traj=traj, level=level, peak=peak, off=off, micro=micro, stats=stats, **extra)
    os.replace(tmp, f)
    m = {"face": face, "mode": mode, "label": LABELS[mode], "kwargs": SPECS[mode][1], "seeds": SPECS[mode][0], "windows": n,
         "errors": errs[:20], "error_count": len(errs), "wall_seconds": time.time() - t0, "workers": workers,
         "product_sha256": psha, "limit": limit, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if mode == "ens" or mode in FIELD_MODES:                            # gate: the stored product, bit for bit
        m["gate"] = _gate(traj, level, peak, off, micro, zx, n)
    if mode in FIELD_MODES:                                              # and the rebuilt quantities: the call's own fields
        st = off | micro
        fo, rb = extra["fields_ok"], extra["rebuild_ok"]
        m["fields"] = {"field_errors": ferrs[:20], "field_error_count": len(ferrs), "non_state_windows": int((~st).sum()),
                       "windows_with_fields": int(fo.sum()), "fields_on_every_non_state_window": bool(np.array_equal(fo, ~st & np.isfinite(level))),
                       "rebuilt_reproduce_the_call": int(rb.sum()), "rebuilt_mismatches": int((fo & ~rb).sum())}
        m["gate"]["fields_pass"] = bool(not ferrs and m["fields"]["fields_on_every_non_state_window"] and m["fields"]["rebuilt_mismatches"] == 0)
        m["gate"]["pass"] = bool(m["gate"]["pass"] and m["gate"]["fields_pass"])
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


def run(faces, modes=None, workers=2, prods=None, limit=None, seed_gate=8, fields=True):
    """Re-anchor the populations in the given modes (default: default_modes, and with ``fields`` the fields mode).
    ``seed_gate`` > 0: where stored single-seed products are used, seed 0 is also re-anchored on the first
    ``seed_gate`` windows and must reproduce its stored product bit for bit (one worker; cache name s0_first<n>)."""
    prods = prods or CB.load_products()
    rec = {}
    for face in faces:
        entry = prods["faces"].get(face)
        if entry is None or not entry["npz_path"].exists():
            continue
        jobs = [(m, limit, workers) for m in (modes or default_modes(entry) + (list(FIELD_MODES) if fields else []))]
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
