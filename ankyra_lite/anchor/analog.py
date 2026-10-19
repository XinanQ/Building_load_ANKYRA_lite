"""Analog-day within-day shapes and the error-weighted within-day anchoring (ANKYRA 2.0; one trust value since 2.1).

Analog days.  For a target day of the horizon, its analog days are the complete pre-origin days of the unit's own
record that have the same calendar type (Monday ... Sunday, holiday), lie within 14 days of the same day of year, are in
the same daylight-saving state, and are preceded by a complete 744-hour window.  Up to eight analogs are kept, the
closest in daily-mean temperature to the frozen climatology of the target day (ties: more recent first).  Each analog
day's shape is its hourly load minus its own daily mean, divided by the raw standard deviation of the 744 hours before
it; the target day's shape is the mean of the kept analog shapes times the origin scale s0 (raw standard deviation of
the 744 hours before the origin, floored at max(0.01 |mean|, 1e-3)).  A day with fewer than four analogs keeps the
foundation model's shape; a window whose analog shape exceeds three times the largest absolute context value keeps the
foundation model's shape entirely.

Within-day anchoring.  With wT the foundation model's within-day block at the origin and S the analog shape, the
delivered within-day block is  W = wT + w (S - wT).  Since 2.1 the trust w is one value for the whole window: the
least-squares weight of (S_q - wT_q) against the realised within-day error (y_q - wT_q), pooled over all 744 hours of
the unit's completed pseudo-origin windows q = o - 744 k', k' = 1..3, clipped to [0, 1], shrunk towards zero by
n/(n + 2) and capped at 1/2:  w = min(clip(lambda*, 0, 1) n / (n + 2), 1/2).  Up to 2.0.1 the same weight was estimated
separately for each lead block k (days 1-7, 8-14, 15-21, 22-31); ``within_trust(..., single=False)`` gives those four
weights.  Nothing is trained; both shapes and the trust use only data before the origin (and, at a pseudo-origin, only
data before that pseudo-origin).

Both within-day blocks have zero daily means, so the pre-projection daily means are unchanged by the anchoring, and with
them the level, the daily path, the energy readout (744 x level) and any readout computed from those daily means.  The
delivered trajectory max(daily + W, 0) can differ where the projection binds, and so can a readout computed from it (the
peak readout as evaluated in the study and in examples/quickstart.py).  The analog shape of day d is
S_dh = s0 / |A_d| * sum_j (x_jh - mean_j) / s_j over the kept analog days j.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

from .blocks import within_day, CONTEXT, HORIZON, D, HR
from .history._climate import climatology

WIN, KMAX, KMIN = 14, 8, 4           # day-of-year window, analogs kept, analogs required
SANITY = 3.0                         # whole-window fallback when max|S| > SANITY * max|context|
K_WITHIN, K0_WITHIN, W_CAP = 3, 2.0, 0.5
LEAD_BLOCKS = ((0, 7), (7, 14), (14, 21), (21, 31))
REGIONS = ("EU", "US", "none")


def _nth_sunday(y, m, n):
    d0 = date(y, m, 1); first = d0 + timedelta(days=(6 - d0.weekday()) % 7)
    return first + timedelta(weeks=n - 1)


def _last_sunday(y, m):
    d0 = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return d0 - timedelta(days=(d0.weekday() - 6) % 7)


def dst_state(day: date, region: str) -> bool:
    """True inside daylight-saving time for the region's rule (EU: last Sundays of March/October; US: second Sunday of
    March to first Sunday of November); always False for 'none'."""
    if region == "EU":
        return _last_sunday(day.year, 3) <= day < _last_sunday(day.year, 10)
    if region == "US":
        return _nth_sunday(day.year, 3, 2) <= day < _nth_sunday(day.year, 11, 1)
    if region == "none":
        return False
    raise ValueError(f"dst_region must be one of {REGIONS}")


def floor_of(m):
    """Floor of the normalisation scale (kW) for a window with mean m: max(0.01 |m|, 1e-3)."""
    return max(0.01 * abs(m), 1e-3)


class AnalogShapes:
    """Per-day quantities of one unit's record, and analog shapes at any origin whose data lie inside the record."""

    def __init__(self, load_kw, temperature_c, day_types, start_timestamp: str, dst_region: str = "none", origin: Optional[int] = None):
        """Days are the origin-aligned 24-hour blocks of the record: block j covers hours offset + 24 j ... + 23, where
        offset = origin mod 24 (0 for the month-start origins of the study, so the blocks are the record's calendar days)."""
        if dst_region not in REGIONS:
            raise ValueError(f"dst_region must be one of {REGIONS}")
        x = np.asarray(load_kw, dtype=np.float64); self.load = x
        T = np.asarray(temperature_c, dtype=np.float64)
        fill = np.nanmean(np.where(np.isfinite(T), T, np.nan)) if np.isfinite(T).any() else 15.0
        self.temp = np.where(np.isfinite(T), T, fill)
        self.types = np.asarray(day_types); self.region = dst_region
        self.t0 = datetime.fromisoformat(start_timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
        self.offset = int(len(x) if origin is None else origin) % HR; off = self.offset
        nd = (len(x) - off) // HR; self.nd = nd; L = x[off:off + nd * HR].reshape(nd, HR)
        self.tday = self.temp[off:off + nd * HR].reshape(nd, HR).mean(1)
        self.ty = self.types[off + 12:off + nd * HR:HR]
        days = [(self.t0 + timedelta(hours=off, days=int(j))).date() for j in range(nd)]
        self.doy = np.array([d.timetuple().tm_yday for d in days])
        self.dst = np.array([dst_state(d, dst_region) for d in days])
        ok = np.isfinite(L).all(1)
        c1 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))]); c2 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x) ** 2)])
        cf = np.concatenate([[0], np.cumsum(np.isfinite(x))]); s0 = np.full(nd, np.nan)
        for j in range(nd):
            e = off + j * HR; s = e - HORIZON
            if s >= 0 and cf[e] - cf[s] == HORIZON:
                m = (c1[e] - c1[s]) / HORIZON; v = max((c2[e] - c2[s]) / HORIZON - m * m, 0.0)
                s0[j] = max(np.sqrt(v), floor_of(m))
        with np.errstate(invalid="ignore", divide="ignore"):
            self.dev = (L - L.mean(1, keepdims=True)) / s0[:, None]
        self.usable = ok & np.isfinite(s0)
        self.fallback_days = 0; self.days_built = 0

    def shape(self, origin: int, fallback) -> np.ndarray:
        """(744,) analog within-day shape for the 31 days after `origin` (an hour index on the record's day grid, at least
        744 hours into the record); days without four analogs take `fallback` (744,) for that day."""
        o = int(origin)
        if (o - self.offset) % HR or o < HORIZON:
            raise ValueError("origin must lie on the record's day grid and at least 744 hours into the record")
        if len(self.types) < o + HORIZON:
            raise ValueError("day types are needed through the 744-hour horizon")
        last = self.load[o - HORIZON:o]; m = float(np.nanmean(last)); s0 = max(float(np.nanstd(last)), floor_of(m))
        theta = climatology(self.temp[:o].astype(np.float32), o); od = (o - self.offset) // HR
        out = np.asarray(fallback, dtype=np.float64).reshape(D, HR).copy()
        for d in range(D):
            day = (self.t0 + timedelta(hours=self.offset, days=int(od + d))).date(); tau = self.types[o + HR * d + 12]
            dd = np.abs(self.doy[:od] - day.timetuple().tm_yday); dd = np.minimum(dd, 365 - dd)
            pool = np.flatnonzero(self.usable[:od] & (self.ty[:od] == tau) & (dd <= WIN) & (self.dst[:od] == dst_state(day, self.region)))
            self.days_built += 1
            if pool.size < KMIN:
                self.fallback_days += 1; continue
            sel = pool[np.lexsort((-pool, np.abs(self.tday[pool] - theta[d])))[:KMAX]]
            out[d] = s0 * self.dev[sel].mean(0)
        return out.reshape(HORIZON)

    def shape_with_sanity(self, origin: int, foundation_within: np.ndarray):
        """Analog shape with the whole-window fallback; returns (shape, kept) where kept is False when it fell back."""
        o = int(origin); fw = np.asarray(foundation_within, dtype=np.float64)
        s = self.shape(o, fw); ctx = self.load[o - CONTEXT:o]
        if np.abs(s).max() > SANITY * np.nanmax(np.abs(ctx)):
            return fw.copy(), False
        return s, True


def within_trust(pairs, single: bool = True) -> tuple[np.ndarray, int]:
    """Trust on the analog shape from completed pseudo-origin triples (analog shape, foundation within-day, realised
    within-day), each (744,).  Returns ((4,) weights for the lead blocks, number of pairs).

    single True (2.1): one weight from the sums over the four lead blocks, repeated four times.
    single False (2.0.0 - 2.0.1): one weight per lead block."""
    n = len(pairs); w = np.zeros(4)
    if n == 0:
        return w, 0
    lead = np.repeat(np.array([0] * 7 + [1] * 7 + [2] * 7 + [3] * 10), HR)
    num, den = np.zeros(4), np.zeros(4)
    for k in range(4):
        m = lead == k
        for s, f, y in pairs:
            dl = (np.asarray(s) - np.asarray(f))[m]; r = (np.asarray(y) - np.asarray(f))[m]
            num[k] += float(dl @ r); den[k] += float(dl @ dl)

    def shrunk(nu, de):
        lam = nu / de if de > 1e-12 else 0.0
        return min(float(np.clip(lam, 0.0, 1.0)) * n / (n + K0_WITHIN), W_CAP)
    if single:
        w[:] = shrunk(float(num.sum()), float(den.sum()))
    else:
        for k in range(4):
            w[k] = shrunk(num[k], den[k])
    return w, n


def anchored_within_day(foundation_within, analog_shape, weights):
    """W = wT + w_k (S - wT) with the block weights repeated over the hours of each day."""
    fw = np.asarray(foundation_within, dtype=np.float64); s = np.asarray(analog_shape, dtype=np.float64)
    w = np.zeros(HORIZON)
    for (a0, a1), wk in zip(LEAD_BLOCKS, weights):
        w[a0 * HR:a1 * HR] = wk
    return fw + w * (s - fw)


__all__ = ["AnalogShapes", "within_trust", "anchored_within_day", "dst_state"]
