"""Extracted frozen arithmetic; see provenance.json. Do not edit in place."""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear
from . import _eo as eo
HISTORY_DAYS, NFEAT = 365, 12
import torch

class Signatures:
    """Per-building convex temperature responses, fitted on the days before each origin."""

    def __init__(self, store, group_curves):
        self.load_day = store.load_np.reshape(len(store.buildings), -1, 24).mean(axis=2)
        temp_day = store.temp.cpu().numpy().reshape(len(store.buildings), -1, 24).mean(axis=2)
        self.types_day = store.types.cpu().numpy().reshape(len(store.buildings), -1, 24)[:, :, 0].astype(int)
        self.feat = eo.hinge_features(eo.std_temp(torch.tensor(temp_day))).numpy()
        self.groups = store.groups_np
        self.curves = group_curves
        self.cache = {}

    def fit(self, b, origin_day, scale, lam):
        key = (int(b), int(origin_day), float(lam))
        if key in self.cache:
            return self.cache[key]
        lo = max(0, origin_day - HISTORY_DAYS)
        days = np.arange(lo, origin_day)
        y = self.load_day[b, days]
        ok = np.isfinite(y)
        days, y = days[ok], y[ok].astype(np.float64)
        w_g = self.curves.get(int(self.groups[b]))
        if len(days) < 60 or w_g is None:
            w = np.zeros(NFEAT) if w_g is None else w_g.copy()
            self.cache[key] = w
            return w
        y = y / max(scale, 1e-9)                                  # window-normalised units, as the level is
        t = self.types_day[b, days]
        for k in np.unique(t):                                    # remove day-type means: Z1 models them with c
            y[t == k] -= y[t == k].mean()
        f = self.feat[b, days]
        f = f - f.mean(axis=0)
        n = f.shape[1]
        gram = f.T @ f
        # knots the history never reaches leave their column at zero, so a floor keeps the problem definite and
        # pulls those directions to the group curve, which is what governs the horizon outside the seen range
        ridge = max(lam, 1e-3) * np.trace(gram) / n
        G = gram + ridge * np.eye(n)
        c = f.T @ y + ridge * w_g
        R = np.linalg.cholesky(G).T
        bounds = (np.concatenate([[-np.inf], np.zeros(n - 1)]), np.full(n, np.inf))
        w = lsq_linear(R, np.linalg.solve(R.T, c), bounds=bounds).x
        self.cache[key] = w
        return w


class ScaledSignatures(Signatures):
    """Signatures whose cache is keyed by the window scale as well (v2.4 audit C0/C1; corrected by the v2.5 audit).

    Signatures.fit divides the history by the window's s0 before the ridge fit, but caches on
    (building, origin_day, lam) only.  A pseudo-origin at 00:00 shares the day with a real 23:00 origin on
    GoiEner (month-end origins), so it would inherit a curve normalised by the real window's s0 - which ends
    23 h after the pseudo-origin.  Real origins never share a day, so the frozen v2.1 path is unchanged.

    The first version of this class (2026-09-17 morning) still called the parent with the parent's day-level key
    in place, so the parent answered from the first curve fitted that day whatever the scale: the fix never took
    effect.  This version keeps its own dictionary and removes the day-level key around every parent call, so the
    parent can only ever fit.  assert_scale_isolation() checks it on real data."""

    def __init__(self, store, group_curves):
        super().__init__(store, group_curves)
        self._scaled = {}

    def fit(self, b, origin_day, scale, lam):
        key = (int(b), int(origin_day), float(lam), float(scale))
        hit = self._scaled.get(key)
        if hit is not None:
            return hit
        day_key = (int(b), int(origin_day), float(lam))
        saved = self.cache.pop(day_key, None)             # the parent must not answer from another scale's curve
        w = Signatures.fit(self, b, origin_day, scale, lam)
        self.cache.pop(day_key, None)
        if saved is not None:
            self.cache[day_key] = saved
        self._scaled[key] = w
        return w
