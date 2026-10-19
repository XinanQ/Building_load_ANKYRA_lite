"""Extracted frozen arithmetic; see provenance.json. Do not edit in place."""
from __future__ import annotations
import numpy as np
import torch
from . import _eo as eo
from . import _level as v24
YEAR = 8760
HYPS = ("c_w8", "c_w4", "c_w2", "c_ann_type", "p_ann_cal", "g_w", "zero")


def offsets_from_weeks(m, ctx_types, weeks):
    """Day-type offsets (B,8) from the last `weeks` context weeks only; eo.type_tables' rule restricted in time."""
    B = m.shape[0]
    mw = m[:, -7 * weeks:]
    tw = ctx_types[:, -7 * weeks:]
    onehot = torch.nn.functional.one_hot(tw.long(), eo.N_TYPES).to(m.dtype)
    block = mw.view(B, weeks, 7).mean(dim=2, keepdim=True).expand(B, weeks, 7).reshape(B, 7 * weeks)
    count = onehot.sum(dim=1)
    c = ((mw - block)[..., None] * onehot).sum(dim=1) / count.clamp_min(1.0)
    seen = count > 0
    c = torch.where(seen, c, torch.zeros_like(c))
    borrow = torch.zeros_like(seen)
    borrow[:, eo.HOLIDAY] = ~seen[:, eo.HOLIDAY]
    src = torch.arange(eo.N_TYPES).expand(B, eo.N_TYPES).clone()
    src[:, eo.HOLIDAY] = eo.SUNDAY
    return torch.where(borrow, c.gather(1, src), c)


def centre(x):
    return x - x.mean(axis=1, keepdims=True)


@torch.no_grad()
def day_path_hypotheses(store, batcher, sigs, system, b, o, batch_size=256):
    """H (N, 7, 31) centred day paths in origin-normalised units, truth (N, 31), s0 (N,), group, v2.1 path."""
    load = store.load_np
    types = store.types.cpu().numpy()
    out = {"H": [], "truth": [], "s0": [], "group": [], "v21": [], "lag_ok": []}
    mu = v24.pad_n(system.mu, int(np.asarray(store.groups_np).max()) + 1)
    for s in range(0, len(b), batch_size):
        bb, oo = b[s:s + batch_size], o[s:s + batch_size]
        batch = batcher.batch(bb, oo)
        l0, s0, flat, z, m, dev = eo.encode(batch)
        parts = eo.z1_parts(batch, l0, s0, m, dev)
        tgt_types = batch.tgt_types
        c8 = parts["c_d"].numpy()                                             # current term
        c4 = eo.gather_types(offsets_from_weeks(m, batch.ctx_types, 4), tgt_types).numpy()
        c2 = eo.gather_types(offsets_from_weeks(m, batch.ctx_types, 2), tgt_types).numpy()
        l0n, s0n = l0.numpy(), s0.numpy()
        # weather path with the frozen signature
        f_h = eo.expected_hinge_features(eo.std_temp(batch.clim_temp), batch.temp_sigma).numpy()
        W = np.stack([sigs.fit(int(bi), int(oi) // 24, float(sc), system.lam) for bi, oi, sc in zip(bb, oo, s0n)])
        g_h = np.einsum("bdf,bf->bd", f_h, W)
        # annual hypotheses from the fully observed lag window
        ann_type = np.zeros_like(c8)
        ann_cal = np.zeros_like(c8)
        lag_ok = np.zeros(len(bb), bool)
        for j, (bi, oi) in enumerate(zip(bb, oo)):
            lo = oi - YEAR
            if lo < 0:
                continue
            seg = load[bi, lo:lo + eo.HORIZON]
            if not np.isfinite(seg).all():
                continue
            lag_ok[j] = True
            dm = (seg.reshape(eo.DAYS, 24).mean(axis=1) - l0n[j]) / s0n[j]
            dm = dm - dm.mean()
            ann_cal[j] = dm
            lt = types[bi, lo + 12 + 24 * np.arange(eo.DAYS)]
            tbl = np.zeros(eo.N_TYPES)
            for k in range(eo.N_TYPES):
                sel = lt == k
                tbl[k] = dm[sel].mean() if sel.any() else np.nan
            if np.isnan(tbl[eo.HOLIDAY]):
                tbl[eo.HOLIDAY] = tbl[eo.SUNDAY]
            tbl = np.where(np.isnan(tbl), 0.0, tbl)
            ann_type[j] = tbl[tgt_types[j].numpy()]
        tgt = batch.target.numpy()
        truth = (tgt.reshape(len(bb), eo.DAYS, 24).mean(axis=2) - l0n[:, None]) / s0n[:, None]
        grp = batch.group.numpy().astype(int)
        H = np.stack([centre(c8), centre(c4), centre(c2), centre(ann_type), centre(ann_cal), centre(g_h),
                      np.zeros_like(c8)], axis=1)
        v21 = centre(c8) + (1.0 - mu[grp])[:, None] * centre(g_h)
        out["H"].append(H), out["truth"].append(centre(truth)), out["s0"].append(s0n)
        out["group"].append(grp), out["v21"].append(v21), out["lag_ok"].append(lag_ok)
    return {k: np.concatenate(v) for k, v in out.items()}
