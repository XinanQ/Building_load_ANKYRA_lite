"""Extracted frozen arithmetic; see provenance.json. Do not edit in place."""
from __future__ import annotations
import math
import numpy as np
import torch
from types import SimpleNamespace
from . import _eo as eo
from . import _climate as base
v1 = SimpleNamespace(HISTORY_CAP=8760)
YEAR, STEP = 8760, 744
HYP = ("s_u", "l_u", "a_u", "s_w", "l_w", "a_w")
W_SET = {"s_w", "l_w", "a_w"}


def annual_hypothesis(store, pairs, sigs, lam, s0, l0):
    """l_a and its de-weathered form, in origin-normalised units."""
    b, o = pairs[:, 0], pairs[:, 1]
    la_kw = np.array([np.nanmean(store.load_np[bi, oi - YEAR:oi - YEAR + eo.HORIZON]) for bi, oi in zip(b, o)])
    l_a = (la_kw - l0) / s0
    day_cum = store.day_cum.cpu().numpy()
    d_o = o // 24
    f_a = (day_cum[b, d_o - 365 + 31] - day_cum[b, d_o - 365]) / 31.0
    W = np.stack([sigs.fit(int(bi), int(oi) // 24, float(si), lam) for bi, oi, si in zip(b, o, s0)])
    return l_a, l_a - (W * f_a).sum(axis=1)


def pad(v, n):
    """Groups beyond the four in-scope ones (Commercial) take the mean, the frozen v2.1 convention."""
    v = np.asarray(v, dtype=np.float64)
    return v if len(v) >= n else np.concatenate([v, np.full(n - len(v), float(v.mean()))])


pad_n = pad


class PseudoBatcher:
    """eo.Batch for any (b, o) whose context is fully observed; mirrors ed.Store.batch without the position index."""

    def __init__(self, store):
        self.store = store
        self.temp_np = store.temp.cpu().numpy()                      # float32, as stored
        self.types_np = store.types.cpu().numpy()
        self.csum = store.csum.cpu().numpy()
        self.ccnt = store.ccnt.cpu().numpy()
        self.day_cum = store.day_cum.cpu().numpy()
        self.groups = store.groups_np
        self._clim = {}

    def clim(self, b, o):
        key = (int(b), int(o))
        if key not in self._clim:
            self._clim[key] = base.climatology(self.temp_np[b].astype(np.float64), int(o))
        return self._clim[key]

    def observed(self, b, o, need_lag=False):
        """context + target fully observed (and the annual lag window, when asked)."""
        ld = self.store.load_np
        if o - eo.CONTEXT < 0 or o + eo.HORIZON > ld.shape[1]:
            return False
        ok = np.isfinite(ld[b, o - eo.CONTEXT:o + eo.HORIZON]).all()
        if need_lag:
            ok = ok and o - YEAR >= 0 and np.isfinite(ld[b, o - YEAR:o - YEAR + eo.HORIZON]).all()
        return bool(ok)

    def batch(self, b_arr, o_arr, dtype=torch.float64):
        b, o = np.asarray(b_arr, dtype=np.int64), np.asarray(o_arr, dtype=np.int64)
        ld = self.store.load_np
        ctx = np.stack([ld[bi, oi - eo.CONTEXT:oi] for bi, oi in zip(b, o)])
        tgt = np.stack([ld[bi, oi:oi + eo.HORIZON] for bi, oi in zip(b, o)])
        if not np.isfinite(ctx).all():
            raise ValueError("pseudo-origin with a missing context hour: the zero-fill path must never be taken")
        temp_h = np.stack([self.temp_np[bi, oi - eo.CONTEXT:oi] for bi, oi in zip(b, o)])
        ctx_mid = o[:, None] - eo.CONTEXT + 12 + 24 * np.arange(eo.CTX_DAYS)
        tgt_mid = o[:, None] + 12 + 24 * np.arange(eo.DAYS)
        start = np.clip(o - v1.HISTORY_CAP, 0, None)
        hist = (self.csum[b, o] - self.csum[b, start]) / np.maximum(self.ccnt[b, o] - self.ccnt[b, start], 1.0)
        hour_t = torch.tensor(o % 24, dtype=torch.float32)                      # float32 first, as Store.batch
        phase = torch.stack([torch.sin(2 * math.pi * hour_t / 24), torch.cos(2 * math.pi * hour_t / 24)], dim=1).to(dtype)
        clim = np.stack([self.clim(bi, oi) for bi, oi in zip(b, o)])
        batch = eo.Batch(context=torch.tensor(ctx, dtype=dtype),
                         # daily temperature means in float32 first, exactly as ed.Store.batch does
                         ctx_temp_day=torch.tensor(temp_h, dtype=torch.float32).view(-1, eo.CTX_DAYS, 24).mean(dim=2).to(dtype),
                         ctx_types=torch.tensor(self.types_np[b[:, None], ctx_mid]),
                         tgt_types=torch.tensor(self.types_np[b[:, None], tgt_mid]),
                         clim_temp=torch.tensor(clim, dtype=dtype), hist_mean=torch.tensor(hist, dtype=dtype),
                         group=torch.tensor(self.groups[b]), phase=phase,
                         target=torch.tensor(np.where(np.isfinite(tgt), tgt, 0.0), dtype=dtype))
        # the v2.1 long-window temperature features (ed.Store.batch)
        d0 = np.clip(-((-(o - v1.HISTORY_CAP)) // 24), 0, None)
        d1 = o // 24
        span = np.maximum(d1 - d0, 1).astype(np.float64)[:, None]
        batch.hist_temp_feat = torch.tensor((self.day_cum[b, d1] - self.day_cum[b, d0]) / span, dtype=dtype)
        batch.temp_sigma = torch.full((len(b),), float(self.store.temp_sigma_std), dtype=dtype)
        batch.target_finite = np.isfinite(tgt).all(axis=1)
        return batch


@torch.no_grad()
def hypotheses(store, sigs, system, batcher, b_arr, o_arr, lam, batch_size=256):
    """The six level hypotheses (origin-normalised), the shared terms and the realised window mean.

    Returns arrays over the given pairs:  h[name] (N,), c_bar (N,), g_h (N,31), c_d (N,31), D0 (N,744), S (N,8,24),
    l0, s0, group, truth (N, normalised window mean of the target; NaN when the target is not fully observed),
    lag_ok (N, annual lag window observed).
    """
    b, o = np.asarray(b_arr, dtype=np.int64), np.asarray(o_arr, dtype=np.int64)
    out = {k: [] for k in HYP + ("c_bar", "g_h", "c_d", "D0", "S", "l0", "s0", "group", "truth", "lag_ok")}
    for s in range(0, len(b), batch_size):
        bb, oo = b[s:s + batch_size], o[s:s + batch_size]
        batch = batcher.batch(bb, oo)
        l0, s0, flat, z, m, dev = eo.encode(batch)
        parts = eo.z1_parts(batch, l0, s0, m, dev)
        f_recent = eo.hinge_features(eo.std_temp(batch.ctx_temp_day[:, -7:])).mean(dim=1).numpy()
        f_long = batch.hist_temp_feat.numpy()
        f_h = eo.expected_hinge_features(eo.std_temp(batch.clim_temp), batch.temp_sigma).numpy()
        scales = s0.numpy()
        W = np.stack([sigs.fit(int(bi), int(oi) // 24, float(sc), lam) for bi, oi, sc in zip(bb, oo, scales)])
        g_recent, g_long = (W * f_recent).sum(axis=1), (W * f_long).sum(axis=1)
        g_h = np.einsum("bdf,bf->bd", f_h, W)
        ell_s, ell_l = parts["ell_s"].numpy(), parts["ell_l"].numpy()
        lag_ok = np.array([batcher.observed(bi, oi, need_lag=True) for bi, oi in zip(bb, oo)])
        la_u, la_w = np.full(len(bb), np.nan), np.full(len(bb), np.nan)
        if lag_ok.any():
            pairs = np.stack([bb[lag_ok], oo[lag_ok]], axis=1)
            la_u[lag_ok], la_w[lag_ok] = annual_hypothesis(store, pairs, sigs, lam, scales[lag_ok], l0.numpy()[lag_ok])
        c_d = parts["c_d"].numpy()
        truth = (batch.target.numpy().mean(axis=1) - l0.numpy()) / scales
        truth[~batch.target_finite] = np.nan
        vals = {"s_u": ell_s, "l_u": ell_l, "a_u": la_u,
                "s_w": ell_s - g_recent, "l_w": ell_l - g_long, "a_w": la_w}
        for k in HYP:
            out[k].append(vals[k])
        out["c_bar"].append(c_d.mean(axis=1)), out["g_h"].append(g_h), out["c_d"].append(c_d)
        out["D0"].append(parts["D0"].numpy()), out["S"].append(parts["S"].numpy())
        out["l0"].append(l0.numpy()), out["s0"].append(scales), out["group"].append(batch.group.numpy())
        out["truth"].append(truth), out["lag_ok"].append(lag_ok)
    return {k: np.concatenate(v) for k, v in out.items()}


def pseudo_errors(store, sigs, system, batcher, b_arr, o_arr, lam, K, batch_size=256):
    """e[h] (N, K) errors of hypothesis h at o - k*744 (NaN where the pseudo-window is not usable)."""
    b, o = np.asarray(b_arr, dtype=np.int64), np.asarray(o_arr, dtype=np.int64)
    N = len(b)
    err = {h: np.full((N, K), np.nan) for h in HYP}
    for k in range(1, K + 1):
        ok = o - k * STEP
        usable = np.array([batcher.observed(bi, oi) for bi, oi in zip(b, ok)])
        if not usable.any():
            continue
        rows = np.flatnonzero(usable)
        h = hypotheses(store, sigs, system, batcher, b[rows], ok[rows], lam, batch_size)
        for name in HYP:
            pred = h[name] + h["c_bar"] + (h["g_h"].mean(axis=1) if name in W_SET else 0.0)
            e = pred - h["truth"]
            if name in ("a_u", "a_w"):
                e[~h["lag_ok"]] = np.nan
            err[name][rows, k - 1] = e
    return err


def weights(err, K, K0, scale="mad", prior="equal", prior_w=None, annual_in_w=True, min_valid=2):
    """Per-window simplex weights over the six hypotheses (registration section 1)."""
    names = list(HYP) if annual_in_w else [h for h in HYP if h != "a_w"]
    N = next(iter(err.values())).shape[0]
    E = {h: err[h][:, :K] for h in names}
    valid = {h: np.isfinite(E[h]) for h in names}
    counts = {h: valid[h].sum(axis=1) for h in names}
    m = {}
    for h in names:
        a = np.abs(np.where(valid[h], E[h], np.nan))
        with np.errstate(all="ignore"):
            m[h] = np.nanmedian(a, axis=1) if scale == "mad" else np.sqrt(np.nanmean(a ** 2, axis=1))
        m[h] = np.where(counts[h] >= min_valid, m[h], np.nan)
    avail = np.stack([np.isfinite(m[h]) for h in names], axis=1)                       # (N, H)
    inv = np.stack([np.where(np.isfinite(m[h]), 1.0 / np.maximum(m[h], 1e-6) ** 2, 0.0) for h in names], axis=1)
    what = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-300)
    if prior == "equal":
        w0 = avail / np.maximum(avail.sum(axis=1, keepdims=True), 1)
    else:                                                                              # frozen v2.1 weights, restricted to the available hypotheses
        pw = np.asarray(prior_w, dtype=np.float64)
        w0 = (pw[None, :] if pw.ndim == 1 else pw) * avail                            # (H,) or per-window (N,H)
        w0 = w0 / np.maximum(w0.sum(axis=1, keepdims=True), 1e-300)
    # K_eff: pseudo-origins where the two short/long hypotheses are both valid (the annual ones may have fewer)
    k_eff = (valid["s_u"] & valid["l_u"]).sum(axis=1).astype(np.float64)[:, None]
    w = (k_eff * what + K0 * w0) / (k_eff + K0)
    none = avail.sum(axis=1) == 0                                                      # no usable pseudo-origin at all
    if prior == "equal":
        w[none] = 1.0 / len(names)                                                     # equal over all candidates (NaN ones drop in level_path)
    else:
        pw = np.asarray(prior_w, dtype=np.float64)
        w[none] = (pw[None, :] if pw.ndim == 1 else pw)[none]                          # the frozen v2.1 weights themselves
    return names, w, {"k_eff": k_eff[:, 0], "avail": avail, "scale": m, "no_evidence": none}


def level_path(h, names, w):
    """L (N,31) in normalised units for the real-origin hypotheses h and weights w; NaN hypotheses get zero weight."""
    H = np.stack([np.where(np.isfinite(h[n]), h[n], 0.0) for n in names], axis=1)
    w = np.where(np.isfinite(np.stack([h[n] for n in names], axis=1)), w, 0.0)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    lvl = (H * w).sum(axis=1)
    w_w = w[:, [i for i, n in enumerate(names) if n in W_SET]].sum(axis=1)
    return lvl[:, None] + w_w[:, None] * h["g_h"] + h["c_d"], w
