"""Readouts from the same computation: energy, a historical-envelope peak operator and a pseudo-origin interval.

Energy.  Window energy is 744 * level (kWh for loads in kW) and is read before the nonnegativity projection.  The
projection max(F, 0) cannot increase any hour's absolute error when the truth is nonnegative; it raises delivered energy
by sum(max(-F, 0)), which is known at the origin.  No map can keep the energy readout, return nonnegative load and
guarantee no increase in error at the same time, so ANKYRA keeps energy from the level and projects only the trajectory.

Peak.  For each context day j let a_j = max_h x_{j,h} - mean_h x_{j,h} (its excursion above its own mean).  For target
day d of calendar type t_d (Monday 0 ... Sunday 6, holiday 7) let A_{t_d} be the largest excursion among the four most
recent context days of that type (all days if the type is absent; holidays borrow Sunday).  With forecast daily means
L_d and window mean F_bar,

    U = max_d (L_d + A_{t_d}),     P_kappa = F_bar + kappa * (U - F_bar)        (ANKYRA uses kappa = 1).

Because maxima commute this equals the hour-wise envelope construction; U >= max_d L_d >= F_bar.

Interval.  Hourly residuals of historical pseudo-forecasts (level + daily path + same-day-type within-day default) at
up to 12 completed pseudo-origins, normalised by each pseudo-window's origin scale and pooled by hour of day and
workday / non-workday, give empirical quantiles that are added to the point trajectory, scaled by the origin scale,
clipped at zero and made monotone across levels.  ``examples/quickstart.py`` shows the three calls
(``residual_quantiles``, ``origin_scale``, ``interval_bands``).
"""
from __future__ import annotations

import numpy as np
import torch

from .history import History, estimate_from_history
from .history import _eo
from .blocks import CONTEXT, HORIZON, D, HR

LEVELS = (0.05, 0.1, 0.5, 0.9, 0.95)      # quantile levels of the interval readout; rows 1 and 3 bound the 80% interval
K_INTERVAL = 12                          # pseudo-origins o - 744k searched for interval residuals


# ----------------------------------------------------------------------------- energy
def energy_kwh(trajectory_kw) -> float:
    """Sum of an hourly kW trajectory, in kWh.

    ANKYRA's energy readout is ``AnkyraForecast.energy_kwh`` (744 x level, read before the nonnegativity projection);
    this function gives the energy of any trajectory, for example the delivered one."""
    return float(np.asarray(trajectory_kw, dtype=np.float64).sum())


def projection_energy_increase_kwh(raw_trajectory_kw) -> float:
    """Energy added by max(F, 0); known at the origin."""
    return float(np.maximum(-np.asarray(raw_trajectory_kw, dtype=np.float64), 0.0).sum())


# ----------------------------------------------------------------------------- peak operator
def historical_excursions(context, context_types, target_types, *, k=4):
    """A_{t_d} for every target day: (N, 31).  context (N, 1344) kW, context_types (N, 56), target_types (N, 31)."""
    c = np.asarray(context, dtype=np.float64)
    ct = np.asarray(context_types)
    tt = np.asarray(target_types)
    if c.ndim != 2 or c.shape[1] != CONTEXT or ct.shape != (len(c), 56) or tt.shape != (len(c), 31):
        raise ValueError("Expected context (N,1344), context_types (N,56), target_types (N,31)")
    if not np.isfinite(c).all() or not np.issubdtype(ct.dtype, np.integer) or not np.issubdtype(tt.dtype, np.integer):
        raise ValueError("Finite context and integer calendar labels required")
    if np.any((ct < 0) | (ct > 7)) or np.any((tt < 0) | (tt > 7)) or not isinstance(k, int) or not 1 <= k <= 56:
        raise ValueError("Invalid calendar or k")
    days = c.reshape(-1, 56, 24)
    a = days.max(2) - days.mean(2)
    table = np.empty((len(c), 8))
    for i in range(len(c)):
        for t in range(8):
            ix = np.flatnonzero(ct[i] == t)[-k:]
            table[i, t] = a[i, ix].max() if len(ix) else a[i].max()
        if not (ct[i] == 7).any():
            table[i, 7] = table[i, 6]
    return table[np.arange(len(c))[:, None], tt]


def peak_from_daily_levels(levels, excursions, kappa=1.0):
    """P_kappa = mean + kappa (max_d (L_d + A_d) - mean), in kW, for (N, 31) daily means L and excursions A."""
    L = np.asarray(levels, dtype=np.float64)
    A = np.asarray(excursions, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] != 31 or A.shape != L.shape:
        raise ValueError("Expected paired (N,31) inputs")
    kap = np.broadcast_to(np.asarray(kappa, dtype=float), (len(L),))
    if not np.isfinite(L).all() or not np.isfinite(A).all() or not np.isfinite(kap).all() or (kap < 0).any():
        raise ValueError("Finite levels/excursions and nonnegative kappa required")
    m = L.mean(1)
    return m + kap * ((L + A).max(1) - m)


def peak_readout(prediction, context, context_types, target_types, kappa=1.0):
    """Monthly peak (kW) read from a 744-hour trajectory's daily means and the context's day-type excursions."""
    p = np.asarray(prediction, dtype=np.float64)
    if p.ndim == 1:
        p = p[None]
    if p.ndim != 2 or p.shape[1] != HORIZON or not np.isfinite(p).all():
        raise ValueError("Expected finite prediction (N,744)")
    return peak_from_daily_levels(p.reshape(-1, 31, 24).mean(2), historical_excursions(context, context_types, target_types), kappa)


# ----------------------------------------------------------------------------- interval
def origin_scale(context_kw) -> float:
    """Scale s0 (kW) of the last 744 context hours: their standard deviation, floored at max(0.01 |mean|, 1e-3)."""
    last = np.asarray(context_kw, dtype=np.float64)[-HORIZON:]
    return max(float(last.std()), max(0.01 * abs(float(last.mean())), 1e-3))


def within_day_default(context, context_types, target_types):
    """Same-day-type historical within-day default s0 * D0 (eight most recent days of each type), (N, 744) kW."""
    out = []
    for s in range(0, len(context), 256):
        c = torch.tensor(np.asarray(context[s:s + 256], dtype=np.float64), dtype=torch.float64)
        l0, s0, _ = _eo.origin_scale(c)
        zz = (c - l0[:, None]) / s0[:, None]
        m, dev = _eo.deviations(zz)
        _, Dt = _eo.type_tables(m, dev, torch.tensor(np.asarray(context_types[s:s + 256])))
        D0 = _eo.gather_types(Dt, torch.tensor(np.asarray(target_types[s:s + 256])))
        out.append((s0[:, None, None] * D0).reshape(len(c), HORIZON).numpy())
    return np.concatenate(out)


def _historical_trajectory(load, temp, types, start_timestamp, group, sigma, o):
    if o - CONTEXT < 0 or not np.isfinite(load[o - CONTEXT:o]).all():
        return None
    h = History(load_kw=load[:o], temperature_c=temp[:o], day_types=types[:o + HORIZON], start_timestamp=start_timestamp,
                observed=np.isfinite(load[:o]))
    e = estimate_from_history(h, group=group, temp_sigma_std=sigma)
    ctx = load[o - CONTEXT:o][None]
    ct = types[o - CONTEXT + 12 + 24 * np.arange(56)][None]
    tt = types[o + 12 + 24 * np.arange(D)][None]
    wd = within_day_default(ctx, ct, tt)[0]
    return np.maximum(e.level_kw + np.repeat(e.daily_path_kw, HR) + wd, 0.0)


def residual_quantiles(load, temp, types, start_timestamp, group, sigma, o):
    """Pooled normalised residual quantiles Q (2 workday classes, 24 hours, len(LEVELS)) and the pseudo-windows used.

    load, temp, types   the unit's hourly arrays as in ``History``: load (kW) and temperature (degC) from index 0 of
                        the record, day types (Monday 0 ... Sunday 6, holiday 7) through o + 744; nothing at or after
                        o is read from load or temp
    start_timestamp     ``History.start_timestamp``
    group, sigma        the ``group`` and ``temp_sigma_std`` arguments of ``ankyra.forecast``
    o                   origin index (the length of the pre-origin record)

    Q[1] is the workday class (day type < 5), Q[0] the other days (weekends and holidays).  The second return value
    is the number of completed pseudo-windows that supplied residuals (at most 12).  A class and hour with fewer than
    five residuals takes the quantiles of the other class; if neither has five, that entry is NaN (always the case
    when no pseudo-window is complete)."""
    load = np.asarray(load, dtype=np.float64)
    types = np.asarray(types)
    res = [[[] for _ in range(24)] for _ in range(2)]
    used = 0
    for k in range(1, K_INTERVAL + 1):
        ok = o - k * HORIZON
        if ok - CONTEXT < 0 or not np.isfinite(load[ok - CONTEXT:ok + HORIZON]).all():
            continue
        Hf = _historical_trajectory(load, temp, types, start_timestamp, group, sigma, ok)
        if Hf is None:
            continue
        s0 = origin_scale(load[ok - CONTEXT:ok])
        r = (load[ok:ok + HORIZON] - Hf) / s0
        hod = np.arange(ok, ok + HORIZON) % 24
        work = (types[ok:ok + HORIZON] < 5).astype(int)
        for w in (0, 1):
            for hh in range(24):
                res[w][hh].extend(r[(hod == hh) & (work == w)].tolist())
        used += 1
    Q = np.full((2, 24, len(LEVELS)), np.nan)
    for w in (0, 1):
        for hh in range(24):
            v = np.array(res[w][hh])
            if len(v) >= 5:
                Q[w, hh] = np.quantile(v, LEVELS)
    for w in (0, 1):
        bad = ~np.isfinite(Q[w]).all(1)
        Q[w][bad] = Q[1 - w][bad]
    return Q, used


def interval_bands(point_kw, s0, Q, hour_of_day, workday):
    """Quantile trajectories (len(LEVELS), 744): point + s0 * q, clipped at zero and monotone across levels.

    point_kw      (744,) point trajectory, e.g. ``AnkyraForecast.trajectory_kw``
    s0            ``origin_scale`` of the 1,344-hour context
    Q             from ``residual_quantiles``
    hour_of_day   (744,) integers: (o + arange(744)) % 24
    workday       (744,) integers 0 / 1: (types[o:o + 744] < 5)

    Rows follow ``LEVELS`` (0.05, 0.1, 0.5, 0.9, 0.95), so rows 1 and 3 bound the 80% interval and row 2 is the median."""
    point = np.asarray(point_kw, dtype=np.float64)
    out = np.stack([point + s0 * Q[workday, hour_of_day, i] for i in range(len(LEVELS))])
    out = np.maximum(out, 0.0)
    return np.maximum.accumulate(out, axis=0)
