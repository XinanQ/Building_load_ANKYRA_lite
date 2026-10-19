"""Extracted frozen arithmetic; see provenance.json. Do not edit in place."""
from __future__ import annotations
import numpy as np
from . import _span as span
from . import _level as v24
HYPS_C = ("cur", "c_w8", "c_w4", "c_w2", "c_ann_type", "p_ann_cal", "g_w")
ANNUAL_C = ("c_ann_type", "p_ann_cal")
_SPAN_INDEX = {h: i for i, h in enumerate(span.HYPS)}


def hypotheses_c(store, batcher, sigs, system, b, o, wW):
    """(N, 7, 31) centred day paths in HYPS_C order, centred truth (N, 31), lag availability (N,)."""
    d = span.day_path_hypotheses(store, batcher, sigs, system, np.asarray(b), np.asarray(o))
    H = d["H"]
    cur = H[:, _SPAN_INDEX["c_w8"]] + np.asarray(wW, dtype=np.float64)[:, None] * H[:, _SPAN_INDEX["g_w"]]
    stack = [cur] + [H[:, _SPAN_INDEX[h]] for h in HYPS_C[1:]]
    return np.stack(stack, axis=1), d["truth"], d["lag_ok"]


def pseudo_errors_c(store, batcher, sigs, system, b, o, wW, K):
    """err[h] (N, K): RMS between-day error of hypothesis h at o - k*744 (NaN where unusable)."""
    b, o, wW = np.asarray(b, dtype=np.int64), np.asarray(o, dtype=np.int64), np.asarray(wW, dtype=np.float64)
    N = len(b)
    err = {h: np.full((N, K), np.nan) for h in HYPS_C}
    for k in range(1, K + 1):
        ok = o - k * v24.STEP
        usable = np.array([batcher.observed(bi, oi) for bi, oi in zip(b, ok)])
        rows = np.flatnonzero(usable)
        if not len(rows):
            continue
        H, T, lag = hypotheses_c(store, batcher, sigs, system, b[rows], ok[rows], wW[rows])
        e = np.sqrt(((H - T[:, None, :]) ** 2).mean(axis=2))                       # (n, 7)
        for i, h in enumerate(HYPS_C):
            v = e[:, i].copy()
            if h in ANNUAL_C:
                v[~lag] = np.nan
            err[h][rows, k - 1] = v
    return err


def weights_c(err, K, K0, scale="rms", prior="equal", include_annual=True, min_valid=2):
    """Per-window simplex weights over the day-path hypotheses; mirrors tools_eo_v24_level_v1.weights."""
    names = [h for h in HYPS_C if include_annual or h not in ANNUAL_C]
    E = {h: err[h][:, :K] for h in names}
    valid = {h: np.isfinite(E[h]) for h in names}
    m = {}
    for h in names:
        a = np.abs(np.where(valid[h], E[h], np.nan))
        with np.errstate(all="ignore"):
            m[h] = np.nanmedian(a, axis=1) if scale == "mad" else np.sqrt(np.nanmean(a ** 2, axis=1))
        m[h] = np.where(valid[h].sum(axis=1) >= min_valid, m[h], np.nan)
    avail = np.stack([np.isfinite(m[h]) for h in names], axis=1)
    inv = np.stack([np.where(np.isfinite(m[h]), 1.0 / np.maximum(m[h], 1e-6) ** 2, 0.0) for h in names], axis=1)
    what = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-300)
    cur_i = names.index("cur")
    if prior == "equal":
        w0 = avail / np.maximum(avail.sum(axis=1, keepdims=True), 1)
    elif prior == "current":
        w0 = np.zeros_like(what)
        w0[:, cur_i] = 1.0
    else:
        raise ValueError(prior)
    k_eff = valid["cur"].sum(axis=1).astype(np.float64)[:, None]
    w = (k_eff * what + K0 * w0) / (k_eff + K0)
    none = avail.sum(axis=1) == 0
    w[none] = 0.0
    w[none, cur_i] = 1.0                                                          # no evidence: arm A's own path
    return names, w, {"k_eff": k_eff[:, 0], "no_evidence": none}


def arm_c_level(LA, H, names, w, lag_ok=None):
    """L_C (N, 31): arm A's window mean plus the weighted centred day path.  Where the real origin's lag window
    is not fully observed (lag_ok False), the annual weights are zeroed and the remaining weights renormalised."""
    w = np.array(w, dtype=np.float64, copy=True)
    if lag_ok is not None:
        ann = [i for i, n in enumerate(names) if n in ANNUAL_C]
        if ann:
            w[np.ix_(~np.asarray(lag_ok, bool), ann)] = 0.0
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    idx = [HYPS_C.index(n) for n in names]
    P = np.einsum("nh,nhd->nd", w, H[:, idx, :])
    return LA.mean(axis=1, keepdims=True) + P
