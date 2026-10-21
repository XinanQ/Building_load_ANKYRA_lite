"""The origin-continuity layer: a correction of the anchored trajectory's first hours and days toward the last
observed hours before the origin.

    from ankyra_lite import continuation
    Fp, rec = continuation.apply(F, load_kw, day_types)          # F (744,) issued at the origin len(load_kw)
    Fp, info = continuation.apply_batch(F, pre, last_type, day_types, state)    # n windows at once

``ankyra_lite.forecast(..., continuation=True)`` applies it to the delivered trajectory after the anchoring layer and
the state rule; energy, level and peak are read before it (see ``ankyra_lite.forecast.ForecastResult``).

Layer, one window (origin o; F(h), h = 0..743, the anchored trajectory after the nonnegativity projection)
  bypass   state windows (the anchoring layer reported off_state or micro_load; the state rule delivers the weekly
           seasonal naive): F' = F, untouched
  scale    s = mean of the observed load in the 672 hours before o (floor 1e-3 kW); e = 0.01 s
  d*       the first forecast day d = 0..6 whose workday class equals that of the last observed day; 6 when none does.
           Class: day type < 5 is a workday; 5, 6 and 7 (Saturday, Sunday, holiday) are not.  The type of a day is
           read at its 12th hour: types[o - 12] for the last observed day, types[o + 12 + 24 d] for forecast day d
           (the calendar passed to the anchoring layer)
  x_f      clip(ln[(mean y(o-3..o-1) + e) / (max(mean F(24d*+21..24d*+23), 0) + e)], -1, 1): the last 3 hours
           against the same clock hours of the forecast; 0 if one of the last 3 hours is unobserved, or if
           mean y(o-3..o-1) + e <= 0 (negative loads, see below)
  x_d      clip(ln[(mean y(o-24..o-1) + e) / (max(mean F(24d*..24d*+23), 0) + e)], -1, 1): the last day against
           the forecast's day of the same class; 0 if one of the last 24 hours is unobserved, or if
           mean y(o-24..o-1) + e <= 0
  Delta(h) = (kappa1 x_f) exp(-h / rho1) + (kappa2 x_d) exp(-h / rho2);   F'(h) = F(h) exp(Delta(h))

Constants (:data:`CONSTANTS`): kappa1 = 0.479, rho1 = 12 h, kappa2 = 0.330, rho2 = 168 h.  One global set, fitted
once on a development set (six populations pooled, all windows, unit-equal whole-window squared-error ratio), kappa
rounded to three decimals (:data:`FULL_PRECISION` keeps the fitted values for provenance), and never refitted: every
population and every external dataset uses the same four numbers.  :data:`RULE` holds the remaining fixed settings
and :data:`DIGEST` the sha256 that identifies constants and rule (:func:`digest`).

Exact properties (``tests/test_continuation.py`` checks each)
  sign      F >= 0 gives F' >= 0; F(h) = 0 gives F'(h) = 0, and a positive normal float stays positive
  bounds    |Delta(h)| <= E(h) = kappa1 exp(-h / 12) + kappa2 exp(-h / 168), decreasing in h: E(0) = 0.809,
            E(168) = 0.12140 (< 0.1215), E(743) = 0.00396 (< 0.004); F'/F lies in [0.4453, 2.2457] overall, in
            [0.8856, 1.1291] from h = 168 on.  E(h) is reached when both anomalies are clipped to the same sign
  causal    F' depends on F, the 672 load hours before o (and which of them are observed) and the calendar at
            o - 12 and o + 12 + 24 d (d = 0..6) only; no load at or after the origin is read (the calendar of the
            first forecast week is an input known in advance)
  bypass    state windows: F' = F bit for bit.  An unobserved hour among the last 3 sets x_f = x_d = 0 (F' = F bit for
            bit); one among the last 24 only sets x_d = 0.  A nonpositive span mean (below) sets that span's x to 0
  zero      the constants :data:`ZERO` (kappa1 = kappa2 = 0) give F' = F bit for bit on every window with a finite
            trajectory (``forecast(..., continuation=False)`` switches the layer off without computing anything)
  scale     F'(c y, c F) = c F'(y, F) for c > 0 while s stays above its 1e-3 kW floor (bit for bit when c is a power
            of two)

Negative loads (a release rule, not part of :data:`RULE`).  The logarithm is defined only when the span mean plus e is
positive.  When the mean of the last 3 (or 24) hours plus e is at or below 0 (negative loads, e.g. a net-metered
record), that anomaly is treated like a span with an unobserved hour: x = 0, raw = NaN, and the window is flagged
(``nonpositive_f`` / ``nonpositive_d`` in the batch info, ``nonpositive_fast`` / ``nonpositive_day`` in the record).
The study's reference implementation raised an error there instead; no study window reached the case, so the rule
changes no study output, and :data:`RULE` and :data:`DIGEST` keep identifying the frozen layer.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Optional, Union

import numpy as np

__all__ = ["apply", "apply_batch", "anomalies", "delta", "decay_curves", "envelope", "reference_day", "window_inputs",
           "digest", "Constants", "ContinuationRecord", "CONSTANTS", "ZERO", "FULL_PRECISION", "RULE", "DIGEST"]

HORIZON = 744
FAST_HOURS, DAY_HOURS, SCALE_HOURS = 3, 24, 672
SCALE_FLOOR, EPS_FRACTION, CLIP = 1e-3, 0.01, 1.0
WORKDAY_TYPE_BELOW, TYPE_HOUR, REFERENCE_DAYS, NO_MATCH_DAY = 5, 12, 7, 6

RULE = {"fast_hours": FAST_HOURS, "day_hours": DAY_HOURS, "scale_hours": SCALE_HOURS, "scale_floor": SCALE_FLOOR,
        "eps_fraction": EPS_FRACTION, "clip": CLIP, "workday_type_below": WORKDAY_TYPE_BELOW, "type_hour": TYPE_HOUR,
        "reference_days": REFERENCE_DAYS, "no_match_day": NO_MATCH_DAY, "horizon": HORIZON,
        "bypass": "state windows (off_state or micro_load) and missing last hours (x = 0)"}
DIGEST = "d514e3ec93a745b14950671297bf2f922fd512424a804e292377e1b0c4cc1dce"   # digest(CONSTANTS, RULE)


@dataclass(frozen=True)
class Constants:
    """The four constants of the layer: Delta(h) = (kappa1 x_f) exp(-h / rho1) + (kappa2 x_d) exp(-h / rho2).

    kappa1, kappa2 >= 0 (weights of the fast and the day anomaly); rho1, rho2 > 0 (decay times in hours).  Values are
    stored as Python floats."""
    kappa1: float
    rho1: float
    kappa2: float
    rho2: float

    def __post_init__(self):
        for name in ("kappa1", "rho1", "kappa2", "rho2"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"continuation constant {name} must be finite")
            object.__setattr__(self, name, v)
        if not (self.kappa1 >= 0.0 and self.kappa2 >= 0.0):
            raise ValueError("kappa1 and kappa2 must be nonnegative")
        if not (self.rho1 > 0.0 and self.rho2 > 0.0):
            raise ValueError("rho1 and rho2 must be positive")

    def as_dict(self) -> dict:
        return {"kappa1": self.kappa1, "rho1": self.rho1, "kappa2": self.kappa2, "rho2": self.rho2}


CONSTANTS = Constants(kappa1=0.479, rho1=12.0, kappa2=0.330, rho2=168.0)
ZERO = Constants(kappa1=0.0, rho1=12.0, kappa2=0.0, rho2=168.0)          # F' = F bit for bit on every finite trajectory
FULL_PRECISION = Constants(kappa1=0.4789714953611109, rho1=12.0, kappa2=0.3303470261128675, rho2=168.0)   # provenance only


class ContinuationRecord(NamedTuple):
    """What the layer did on one window (``ForecastResult.continuation``).

    bypass          "state window" (F' = F; nothing below is computed), "missing hours" (an unobserved hour among the
                    last 3, so x_fast = x_day = 0 and F' = F), "nonpositive load" (the last 3 hours' mean + e <= 0 and
                    the day anomaly unobserved or nonpositive too, so x_fast = x_day = 0 and F' = F) or None
    d_star          the reference forecast day 0..6 (None on a state window)
    matched         False when no forecast day 0..6 shares the last observed day's workday class (d_star = 6)
    scale_kw        s, the mean observed load of the 672 hours before the origin (floor 1e-3); eps_kw = 0.01 s
    x_fast, x_day   the clipped anomalies (0 when their hours are not all observed, when their mean + e <= 0, or on a
                    state window)
    raw_fast, raw_day   the anomalies before clipping (NaN when not computed)
    missing_fast, missing_day, clipped_fast, clipped_day
                    an hour of the span was unobserved / the anomaly was clipped to [-1, 1]
    delta           (744,) Delta(h); F'(h) = F(h) exp(delta[h])
    max_abs_delta   max |delta| (0 on a state window)
    constants, digest   the constants used and :func:`digest` of them with :data:`RULE`
    nonpositive_fast, nonpositive_day
                    the span's mean load + e was at or below 0 (negative loads), so its anomaly was set to 0
    """
    bypass: Optional[str]
    d_star: Optional[int]
    matched: Optional[bool]
    scale_kw: Optional[float]
    eps_kw: Optional[float]
    x_fast: float
    x_day: float
    raw_fast: float
    raw_day: float
    missing_fast: bool
    missing_day: bool
    clipped_fast: bool
    clipped_day: bool
    delta: np.ndarray
    max_abs_delta: float
    constants: Constants
    digest: str
    nonpositive_fast: bool = False
    nonpositive_day: bool = False


def _constants(constants) -> Constants:
    if isinstance(constants, Constants):
        return constants
    if isinstance(constants, Mapping):
        return Constants(**{k: constants[k] for k in ("kappa1", "rho1", "kappa2", "rho2")})
    raise TypeError("constants must be a continuation.Constants (or a mapping with kappa1, rho1, kappa2, rho2)")


def digest(constants: Union[Constants, Mapping] = CONSTANTS, rule: Mapping = RULE) -> str:
    """sha256 of the canonical JSON of {"constants": ..., "rule": ...} (sorted keys, separators "," and ":",
    ASCII): the identity of the layer.  ``digest() == DIGEST``."""
    text = json.dumps({"constants": _constants(constants).as_dict(), "rule": dict(rule)}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def decay_curves(constants: Union[Constants, Mapping] = CONSTANTS, horizon: int = HORIZON):
    """(exp(-h / rho1), exp(-h / rho2)) for h = 0..horizon-1."""
    c = _constants(constants)
    hh = np.arange(horizon, dtype=float)
    return np.exp(-hh / float(c.rho1)), np.exp(-hh / float(c.rho2))


def envelope(constants: Union[Constants, Mapping] = CONSTANTS, horizon: int = HORIZON) -> np.ndarray:
    """E(h) = kappa1 exp(-h / rho1) + kappa2 exp(-h / rho2), the bound of |Delta(h)| (reached with both anomalies
    clipped to the same sign)."""
    c = _constants(constants)
    g1, g2 = decay_curves(c, horizon)
    return c.kappa1 * g1 + c.kappa2 * g2


def reference_day(last_type, day_types) -> tuple[int, bool]:
    """(d*, matched): the first forecast day d = 0..6 whose workday class (day type < 5) equals that of
    ``last_type``, the last observed day's type; (6, False) when none does.  ``day_types`` holds the types of
    forecast days 0..6."""
    cls = int(last_type) < WORKDAY_TYPE_BELOW
    for d in range(REFERENCE_DAYS):
        if (int(day_types[d]) < WORKDAY_TYPE_BELOW) == cls:
            return d, True
    return int(NO_MATCH_DAY), False


def window_inputs(load_kw, day_types, origin: int):
    """The only data the layer reads for an origin: (pre, last_type, day_types7).

    pre         load_kw[origin - 672:origin] as float64 (NaN = unobserved)
    last_type   day_types[origin - 12], the type of the last observed day
    day_types7  (day_types[origin + 12 + 24 d] for d = 0..6), the types of the first seven forecast days

    ``load_kw`` and ``day_types`` are hourly rows on the same axis (they may run past the origin; nothing at or after
    the origin is read from ``load_kw``)."""
    o = int(origin)
    if o < SCALE_HOURS:
        raise ValueError(f"origin {o} has fewer than {SCALE_HOURS} hours before it")
    pre = np.asarray(load_kw[o - SCALE_HOURS:o], dtype=np.float64)
    last_type = int(day_types[o - 24 + TYPE_HOUR])
    day_types7 = tuple(int(day_types[o + TYPE_HOUR + 24 * d]) for d in range(REFERENCE_DAYS))
    return pre, last_type, day_types7


def anomalies(pre, F, last_type, day_types) -> dict:
    """x_f, x_d and their bookkeeping for one window.

    pre         the load before the origin (at least 672 values, the last one at o - 1; NaN = unobserved)
    F           (744,) the anchored trajectory issued at o (read as float64)
    last_type, day_types
                as returned by :func:`window_inputs`

    Returns {"xf", "xd", "raw_f", "raw_d", "s", "eps", "missing_f", "missing_d", "clipped_f", "clipped_d",
    "nonpositive_f", "nonpositive_d", "d_star", "matched"}.  A span whose mean load + e is at or below 0 gets x = 0,
    raw = NaN and nonpositive = True (negative loads; see the module docstring)."""
    pre = np.asarray(pre, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    if len(pre) < SCALE_HOURS:
        raise ValueError(f"pre-origin record shorter than {SCALE_HOURS} hours")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sm = float(np.nanmean(pre[-SCALE_HOURS:])) if np.isfinite(pre[-SCALE_HOURS:]).any() else float("nan")
    s = max(sm, SCALE_FLOOR) if math.isfinite(sm) else SCALE_FLOOR
    e = EPS_FRACTION * s
    d, matched = reference_day(last_type, day_types)
    out = {"d_star": d, "matched": matched, "s": s, "eps": e}
    for name, num, den in (("f", pre[-FAST_HOURS:], F[24 * d + 24 - FAST_HOURS:24 * d + 24]), ("d", pre[-DAY_HOURS:], F[24 * d:24 * d + 24])):
        if not np.isfinite(num).all():
            out.update({f"x{name}": 0.0, f"raw_{name}": float("nan"), f"missing_{name}": True, f"clipped_{name}": False,
                        f"nonpositive_{name}": False})
            continue
        dm = float(den.mean())
        if not math.isfinite(dm):
            raise ValueError("non-finite anchored trajectory in the reference day")
        ratio = (float(num.mean()) + e) / (max(dm, 0.0) + e)
        if not ratio > 0.0:                                 # mean + e <= 0 (negative loads): no logarithm; x = 0, as for
            out.update({f"x{name}": 0.0, f"raw_{name}": float("nan"), f"missing_{name}": False,   # an unobserved span
                        f"clipped_{name}": False, f"nonpositive_{name}": True})
            continue
        raw = math.log(ratio)
        out.update({f"x{name}": float(np.clip(raw, -CLIP, CLIP)), f"raw_{name}": raw, f"missing_{name}": False,
                    f"clipped_{name}": abs(raw) > CLIP, f"nonpositive_{name}": False})
    return out


def delta(x_fast, x_day, constants: Union[Constants, Mapping] = CONSTANTS, curves=None) -> np.ndarray:
    """Delta(h) = (kappa1 x_f) exp(-h / rho1) + (kappa2 x_d) exp(-h / rho2), in this order of operations.

    Scalars give (744,); (n,) arrays give (n, 744).  ``curves`` = :func:`decay_curves` of the same constants."""
    c = _constants(constants)
    g1, g2 = curves if curves is not None else decay_curves(c)
    xf = np.asarray(x_fast, dtype=np.float64)
    xd = np.asarray(x_day, dtype=np.float64)
    if xf.ndim == 0:
        return (c.kappa1 * float(xf)) * g1 + (c.kappa2 * float(xd)) * g2
    return (c.kappa1 * xf)[:, None] * g1[None, :] + (c.kappa2 * xd)[:, None] * g2[None, :]


_INFO_KEYS = ("xf", "xd", "raw_f", "raw_d", "s", "eps", "missing_f", "missing_d", "clipped_f", "clipped_d", "nonpositive_f",
              "nonpositive_d", "matched", "d_star")


def apply_batch(F, pre, last_type, day_types, state, constants: Union[Constants, Mapping] = CONSTANTS,
                return_delta: bool = False):
    """The layer on n windows: (F' (n, 744) float64, info).

    F           (n, 744) anchored trajectories (any float dtype; finite on every window that is not a state window)
    pre         (n, >= 672) load before each origin (NaN = unobserved); last_type (n,); day_types (n, 7)
    state       (n,) bool: state windows are passed through unchanged (F' = F as float64, bit for bit)
    constants   :data:`CONSTANTS` (default), :data:`ZERO` or another :class:`Constants`

    info holds (n,) arrays "xf", "xd", "raw_f", "raw_d", "s", "eps", "missing_f", "missing_d", "clipped_f",
    "clipped_d", "nonpositive_f", "nonpositive_d", "matched", "d_star" and "bypass_state" (state windows: x = 0,
    raw = NaN, s = eps = 0, d_star = 0, matched False, nonpositive False), the float "max_abs_delta" over the other
    windows, and with ``return_delta`` the (n, 744) "delta".  A non-finite trajectory on a window that is not a state
    window raises ValueError."""
    c = _constants(constants)
    F = np.asarray(F)
    n = len(F)
    if F.ndim != 2 or F.shape[1] != HORIZON:
        raise ValueError(f"trajectory shape {F.shape}, expected (n, {HORIZON})")
    F64 = F.astype(np.float64)
    st = np.asarray(state, dtype=bool)
    if st.shape != (n,):
        raise ValueError(f"state shape {st.shape}, expected ({n},)")
    info = {k: np.zeros(n, dtype=np.float64) for k in ("xf", "xd", "raw_f", "raw_d", "s", "eps")}
    info.update({k: np.zeros(n, dtype=bool) for k in ("missing_f", "missing_d", "clipped_f", "clipped_d", "nonpositive_f", "nonpositive_d",
                                                      "matched", "bypass_state")})
    info["d_star"] = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if st[i]:
            info["bypass_state"][i] = True
            info["raw_f"][i] = info["raw_d"][i] = np.nan
            continue
        if not np.isfinite(F64[i]).all():
            raise ValueError(f"window {i}: non-finite anchored trajectory on a window that is not a state window")
        a = anomalies(pre[i], F64[i], last_type[i], day_types[i])
        for k in _INFO_KEYS:
            info[k][i] = a[k]
    D = delta(info["xf"], info["xd"], c)
    Fp = F64 * np.exp(D)
    if st.any():
        Fp[st] = F64[st]                                    # an exact copy (x = 0 there anyway)
    info["max_abs_delta"] = float(np.abs(D[~st]).max()) if (~st).any() else 0.0
    if return_delta:
        info["delta"] = D
    return Fp, info


def apply(trajectory_kw, load_kw, day_types, origin: Optional[int] = None, *, state: bool = False,
          constants: Union[Constants, Mapping] = CONSTANTS):
    """The layer on one window: (F' (744,) float64, :class:`ContinuationRecord`).

    trajectory_kw   (744,) the anchored trajectory issued at the origin (after the state rule)
    load_kw         the hourly load up to the origin (NaN = unobserved), e.g. the masked ``History.load_kw``
    day_types       the hourly calendar on the same axis, through the first week of the horizon at least
    origin          the hour index of the origin; default len(load_kw)
    state           True on a state window (bypass: F' = F)

    The same arithmetic as :func:`apply_batch` (it is called with n = 1)."""
    c = _constants(constants)
    o = len(load_kw) if origin is None else int(origin)
    pre, last_type, day_types7 = window_inputs(load_kw, day_types, o)
    Fp, info = apply_batch(np.asarray(trajectory_kw)[None], pre[None], [last_type], [day_types7], [bool(state)], c, return_delta=True)
    st = bool(info["bypass_state"][0])
    missing = bool(info["missing_f"][0])
    npf, npd = bool(info["nonpositive_f"][0]), bool(info["nonpositive_d"][0])
    nonpositive = npf and (npd or bool(info["missing_d"][0]))
    rec = ContinuationRecord(
        bypass="state window" if st else ("missing hours" if missing else ("nonpositive load" if nonpositive else None)),
        d_star=None if st else int(info["d_star"][0]),
        matched=None if st else bool(info["matched"][0]),
        scale_kw=None if st else float(info["s"][0]),
        eps_kw=None if st else float(info["eps"][0]),
        x_fast=float(info["xf"][0]), x_day=float(info["xd"][0]),
        raw_fast=float(info["raw_f"][0]), raw_day=float(info["raw_d"][0]),
        missing_fast=missing, missing_day=bool(info["missing_d"][0]),
        clipped_fast=bool(info["clipped_f"][0]), clipped_day=bool(info["clipped_d"][0]),
        delta=info["delta"][0], max_abs_delta=float(info["max_abs_delta"]), constants=c, digest=digest(c),
        nonpositive_fast=npf, nonpositive_day=npd)
    return Fp[0], rec
