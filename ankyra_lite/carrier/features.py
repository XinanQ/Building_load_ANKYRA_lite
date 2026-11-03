"""Input features of the carrier network, built only from data before the forecast origin and the known calendar.

``features(load, temp, day_types, groups_or_group, start_iso, unit, origin)`` returns the six network inputs of one
forecast (plus two diagnostics). The arithmetic equals the research feature builder bit for bit, including the
half-precision storage of the history tokens that the carrier was trained on.

Conventions
  * index 0 of every array is the hour ``start_iso``; arrays are hourly, either (units, hours) or (hours,)
  * missing load is NaN (negative load also counts as missing); missing temperature is NaN
  * day types are integers Monday=0 .. Sunday=6, holiday=7; where they are unknown (outside the array) the weekday of
    the calendar is used
  * load and temperature are read only at hours before ``origin``; day types are read up to ``origin + 744`` (the
    calendar of the forecast month is known at the origin)
  * hours before index 0 are virtual missing hours (leading padding), so any origin >= 0 is accepted

Feature layout
  day tokens   (366, 69): 24 arcsinh(load/scale), 24 hourly observation flags, arcsinh(daily mean/scale),
               arcsinh(daily std/scale), arcsinh(daily mean temperature/30), temperature coverage, 8 day-type one-hot,
               2 annual phase (sin, cos), relative position, load coverage, 5 group one-hot; unusable days are zero
  month tokens (24, 16): 744-hour blocks older than the daily span: arcsinh(mean, std, 10th and 90th percentile,
               workday mean, weekend mean / scale), zero-load share, coverage, arcsinh(temperature/30), temperature
               coverage, annual phase (2), relative position, constant 1, arcsinh(change of the mean/scale), validity
  future tokens (31, 16): 8 day-type one-hot, annual phase (2), relative day position, 5 group one-hot
  scale        max(mean of the observed last 744 hours, mean of the observed last 8,760 hours); 1 if not positive
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Sequence

import numpy as np

__all__ = ["features", "GROUPS", "DAYS", "MONTHS", "MONTH_HOURS", "HORIZON", "DAY_FEATURES", "MONTH_FEATURES",
           "FUTURE_FEATURES"]

GROUPS = ("Industrial", "Office", "Public", "Residential", "Commercial")
HORIZON, DAYS, MONTHS, MONTH_HOURS = 744, 366, 24, 744
DAY_FEATURES, MONTH_FEATURES, FUTURE_FEATURES = 69, 16, 16


def _take(a, b, start, end, fill=np.nan):
    """Hours [start, end) of row ``b`` (or of the single series) as float32; hours outside the array are ``fill``."""
    n = end - start
    result = np.full(n, fill, np.float32)
    if a is None or n <= 0:
        return result
    aa = a[b] if a.ndim == 2 else a
    lo, hi = max(start, 0), min(end, len(aa))
    if hi > lo:
        result[lo - start:hi - start] = aa[lo:hi]
    return result


def _calendar(start_iso, hours):
    """Annual phase (sin, cos) of the calendar day of each hour index."""
    t0 = datetime.fromisoformat(start_iso).replace(tzinfo=None)
    date = np.datetime64(t0.isoformat(), "h") + np.asarray(hours).astype("timedelta64[h]")
    day = date.astype("datetime64[D]")
    year = day.astype("datetime64[Y]")
    phase = (day - year).astype(int) / 365.25
    return np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], -1).astype(np.float32)


def _stats(x, mask, axis=-1):
    """Masked mean, standard deviation and count along ``axis``."""
    count = mask.sum(axis)
    value = np.where(mask, x, 0.0)
    mean = value.sum(axis) / np.maximum(count, 1)
    second = (value.astype(np.float64) ** 2).sum(axis) / np.maximum(count, 1)
    std = np.sqrt(np.maximum(second - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32), count


def _group_name(groups_or_group, b):
    if groups_or_group is None:
        return "Public"
    if isinstance(groups_or_group, str):
        return groups_or_group
    return str(groups_or_group[b])


def features(load, temp, day_types, groups_or_group: str | Sequence[str] | None, start_iso: str, unit: int, origin: int,
             half: bool = True) -> dict:
    """Carrier inputs for the forecast of unit ``unit`` issued at hour index ``origin``.

    load            (units, hours) or (hours,) hourly load in kW, NaN where missing
    temp            (units, hours) or (hours,) hourly temperature in degC, NaN where missing, or None
    day_types       (units, hours) or (hours,) integer day types (0..7), or None (calendar weekdays are used)
    groups_or_group the unit's category (one of GROUPS; anything else counts as "Public"), a sequence indexed by unit,
                    or None ("Public")
    start_iso       ISO timestamp of index 0
    unit            row of the 2-D arrays (ignored for 1-D arrays)
    origin          hour index of the first forecast hour
    half            store the day and month tokens in float16, as the carrier saw them in training (default)

    Returns {"day_tokens", "day_mask", "month_tokens", "month_mask", "future_tokens", "scale",
    "low_load_threshold", "zero_scale"}.
    """
    b, o = int(unit), int(origin)
    load = np.asarray(load)
    temp = None if temp is None else np.asarray(temp)
    types = None if day_types is None else np.asarray(day_types)
    weekday0 = datetime.fromisoformat(start_iso).weekday()

    # Load features never read a value at or after the origin.
    raw = _take(load, b, o - DAYS * 24, o)
    good = np.isfinite(raw) & (raw >= 0)
    recent = raw[-744:]; rm = good[-744:]
    long = raw[-8760:]; lm = good[-8760:]
    mean_recent = float(np.where(rm, recent, 0).sum() / max(int(rm.sum()), 1))
    mean_long = float(np.where(lm, long, 0).sum() / max(int(lm.sum()), 1))
    s = max(mean_recent, mean_long)
    zero_scale = not (math.isfinite(s) and s > 0)
    if zero_scale:
        s = 1.0
    y = raw.reshape(DAYS, 24); obs = good.reshape(DAYS, 24)
    mean, std, count = _stats(y, obs)
    tmp = _take(temp, b, o - DAYS * 24, o).reshape(DAYS, 24)
    tm = np.isfinite(tmp)
    tmean, _, tc = _stats(tmp, tm)
    hour = o - DAYS * 24 + np.arange(DAYS) * 24 + 12
    ty = _take(types, b, int(hour[0]), o)
    dt = ty[::24][:DAYS]
    fallback = (weekday0 + np.floor_divide(hour, 24)) % 7
    dt = np.where(np.isfinite(dt), dt, fallback).astype(int)
    onehot = np.eye(8, dtype=np.float32)[np.clip(dt, 0, 7)]
    group = _group_name(groups_or_group, b)
    gi = GROUPS.index(group) if group in GROUPS else 2
    gout = np.repeat(np.eye(5, dtype=np.float32)[gi][None], DAYS, 0)
    token = np.concatenate([
        np.arcsinh(np.where(obs, y / s, 0)), obs.astype(np.float32),
        np.arcsinh(mean / s)[:, None], np.arcsinh(std / s)[:, None],
        np.arcsinh(tmean / 30)[:, None], (tc / 24)[:, None], onehot,
        _calendar(start_iso, hour), ((np.arange(DAYS) - DAYS + 1) / DAYS)[:, None],
        (count / 24)[:, None], gout,
    ], 1).astype(np.float32)
    day_mask = count > 0
    token[~day_mask] = 0
    if token.shape != (DAYS, DAY_FEATURES) or not np.isfinite(token).all():
        raise ValueError("non-finite day tokens: load and temperature must be finite or NaN (no infinity)")

    end = o - DAYS * 24
    old = _take(load, b, end - MONTHS * MONTH_HOURS, end).reshape(MONTHS, MONTH_HOURS)
    om = np.isfinite(old) & (old >= 0)
    av, sd, n = _stats(old, om)
    oldtmp = _take(temp, b, end - MONTHS * MONTH_HOURS, end).reshape(MONTHS, MONTH_HOURS)
    otm = np.isfinite(oldtmp); ot, _, otn = _stats(oldtmp, otm)
    midpoint = end - MONTHS * MONTH_HOURS + np.arange(MONTHS) * MONTH_HOURS + MONTH_HOURS // 2
    month = np.zeros((MONTHS, MONTH_FEATURES), np.float32)
    month[:, 0], month[:, 1] = np.arcsinh(av / s), np.arcsinh(sd / s)
    valid = n > 0
    if valid.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            quant = np.nanpercentile(np.where(om[valid], old[valid], np.nan), [10, 90], axis=1)
        month[valid, 2:4] = np.arcsinh(quant.T / s)
    old_hours = np.arange(end - MONTHS * MONTH_HOURS, end)
    oldtype = _take(types, b, int(old_hours[0]), end).reshape(MONTHS, MONTH_HOURS)
    old_weekday = (weekday0 + np.floor_divide(old_hours, 24)) % 7
    oldtype = np.where(np.isfinite(oldtype), oldtype, old_weekday.reshape(MONTHS, MONTH_HOURS))
    for k, cond in ((4, oldtype < 5), (5, oldtype >= 5)):
        mv, _, _ = _stats(old, om & np.isfinite(oldtype) & cond)
        month[:, k] = np.arcsinh(mv / s)
    month[:, 6] = ((old == 0) & om).sum(1) / np.maximum(n, 1)
    month[:, 7] = n / MONTH_HOURS
    month[:, 8], month[:, 9] = np.arcsinh(ot / 30), otn / MONTH_HOURS
    month[:, 10:12] = _calendar(start_iso, midpoint)
    month[:, 12] = (np.arange(MONTHS) - MONTHS + 1) / MONTHS
    month[:, 13] = 1
    previous_valid = np.r_[False, valid[:-1]]
    month[:, 14] = np.where(valid & previous_valid, np.arcsinh(np.r_[0, np.diff(av)] / s), 0)
    month[:, 15] = valid
    month[~valid] = 0

    futurehour = o + np.arange(31) * 24 + 12
    fty = _take(types, b, o + 12, o + HORIZON)[::24][:31]
    fallback = (weekday0 + np.floor_divide(futurehour, 24)) % 7
    fty = np.where(np.isfinite(fty), fty, fallback).astype(int)
    future = np.concatenate([np.eye(8, dtype=np.float32)[np.clip(fty, 0, 7)], _calendar(start_iso, futurehour),
                             (np.arange(31) / 30)[:, None], np.repeat(np.eye(5, dtype=np.float32)[gi][None], 31, 0)],
                            1).astype(np.float32)
    threshold = float(np.quantile(long[lm], .1)) if lm.any() else 0.0
    if half:
        token, month = token.astype(np.float16), month.astype(np.float16)
    return {"day_tokens": token, "day_mask": day_mask, "month_tokens": month, "month_mask": valid,
            "future_tokens": future, "scale": np.float32(s), "low_load_threshold": np.float32(threshold),
            "zero_scale": zero_scale}
