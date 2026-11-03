"""ANKYRA: a unit's own history and a time-series foundation model, divided by blocks and weighted by the unit's errors.

For a forecast origin o the 744-hour trajectory is assembled from three orthogonal blocks (see ``ankyra.blocks``):

* within-day shape    the day-demeaned foundation-model forecast from the last 1,344 hours of load;
* level               six historical candidates (recent / long-history / annual means, each with and without weather
                      adjustment) plus the foundation model's own window mean, weighted by the inverse mean squared
                      window-mean error of each candidate at earlier pseudo-origins o - 744k and shrunk towards equal
                      weights (K0 = 8);
* centred daily path  seven historical candidate paths with inverse-square weights shrunk towards equal weights
                      (K0 = 2).  The rule differs from the level's in three places: a candidate's score is the median
                      over the pseudo-origins of its 31-day RMS error, the evidence count is the number of errors of
                      the path implied by the level construction (``cur``), and without evidence all weight goes to
                      ``cur`` instead of equal weights.

Since 2.0 the within-day block is anchored as well (``ankyra.analog``): the unit's own analog-day shape competes with
the foundation model's shape, with a weight set by the unit's own errors at three completed pseudo-origins, shrunk
towards the foundation model and capped at one half.  Since 2.1 that weight is one value for the whole window (up to
2.0.1 it was estimated separately for days 1-7, 8-14, 15-21 and 22-31).  Both shapes have zero daily means, so the
pre-projection daily means, the level, the daily path, the energy readout and any readout computed from those daily
means are exactly those of the 1.x division of labour; the delivered trajectory max(., 0), and a readout computed from it,
can differ where the projection binds.

The history-side daily means (level with the model candidate, plus the daily path) are then mixed with the foundation
model's daily means week by week (days 1-7, 8-14, 15-21, 22-31).  The weekly weights are least-squares combination
weights shrunk towards 1/2 (K0 = 2).  They are estimated on up to six completed pseudo-origin pairs of the *fixed
division* (six-candidate level plus daily path, without the model candidate) against the model's daily means, and
applied at the origin to the history-side daily means that include the model candidate.  Using the fixed division at
the pseudo-origins avoids nested pseudo-origins.  If the last 168 hours are all at or below 1e-6 kW the foundation-model trajectory is returned
unchanged (off-state rule).  Since 2.0.1 the same is done when the whole 1,344-hour context stays within 1e-3 kW of
zero (micro-load rule): that value is the floor of the scale in which the historical estimator normalises load, and a
record that never leaves it for eight weeks is treated as switched off.  Otherwise the delivered trajectory is
projected onto nonnegative load.

Since 2.2 (``gap_tolerance``, the short-gap rule) the estimator works on a copy of the pre-origin record in which gaps
of at most six hours inside one 744-hour block are interpolated.  The interpolated hours belong to that record for
everything computed after the off-state and micro-load checks: the pseudo-origin contexts and targets, the origin's
own historical candidates and the analog-day pool.  In addition, a pseudo-origin's target month qualifies for the
seven-candidate level and for the handover pairs when at least 90% of its hours are observed.  The daily-path weights,
the six-candidate level of the fixed division and the within-day triples still need completely observed target months
of the interpolated record.  The context at the origin is never filled: it must be complete in the supplied record.
Package 2.2.1 adds a check that the pseudo-origin foundation forecasts are finite; on finite inputs its forecasts are
those of 2.2.0.

Nothing is trained on the target series.  The historical estimator is the frozen reference implementation shipped in
``ankyra.history``; this module adds the foundation-model level candidate, the week-by-week handover, the off-state
rule, the micro-load rule and the short-gap rule exactly as evaluated in the study.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Union

import numpy as np
import torch

from .history import History, estimate_from_history
from .history import api as _api
from .history import _level
from .blocks import within_day, CONTEXT, HORIZON, D, HR     # CONTEXT: the 1,344 hours of load given to the foundation model
from . import analog as _analog

STEP = 744                # spacing of pseudo-origins
K_FM = 6                  # pseudo-origins with foundation-model forecasts
GAP_MAX_H = 6             # gap tolerance (2.2): gaps of at most this many hours are interpolated in the copy of the pre-origin record the estimator reads
TARGET_MIN_OBS = 0.9      # gap tolerance (2.2): a pseudo-origin's target month qualifies (seven-candidate level, handover pairs) when this share of its hours is observed
K0_WEEK = 2.0             # shrinkage of the lead-week weights towards 1/2
ZERO_KW = 1e-6            # off-state threshold (kW)
MICRO_KW = 1e-3           # micro-load threshold (kW, on |load|, compared in float64): the floor of the estimator's normalisation scale
WEEKS = ((0, 7), (7, 14), (14, 21), (21, 31))
GROUPS = ("Industrial", "Office", "Public", "Residential", "Commercial")
# The frozen configuration holds a temperature-signature prior curve for the first four categories only.  The fifth curve
# is stored under the key "group_4", which the lookup by category name never reads, so a Commercial unit gets a zero
# signature.  This is the evaluated behaviour and is kept (see the ``group`` argument of ``forecast``).

Foundation = Union[Mapping[int, np.ndarray], Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class AnkyraForecast:
    """Output of :func:`forecast`.  All quantities in kW; the energy readout is ``744 * level_kw`` kWh.

    ``level_weights`` maps each level candidate to its weight.  ``s``, ``l`` and ``a`` are the mean of the last seven
    context days, the long-history mean (at most one year) and the mean of the window one year earlier; ``_u`` is the
    unadjusted candidate and ``_w`` the weather-adjusted one; ``fm`` is the foundation model's window mean.  A
    candidate that could not be formed (for example ``a_u`` and ``a_w`` on a record shorter than a year) has weight 0.
    On an off-state or micro-load return the foundation-model forecast is delivered unchanged, and the weights, the
    fixed division and the analog fields keep their defaults.
    """
    trajectory_kw: np.ndarray                 # (744,) delivered trajectory (nonnegative unless off-state or micro-load)
    level_kw: float                           # window mean before the nonnegativity projection
    daily_means_kw: np.ndarray                # (31,) after the week-by-week handover
    within_day_kw: np.ndarray                 # (744,) delivered within-day block: the foundation model's shape, anchored to the analog-day shape since 2.0
    off_state: bool                           # True when the last 168 h were at or below 1e-6 kW and the model was used unchanged
    lead_week_weights: tuple = ()             # alpha_w on the model's daily means, weeks 1-4 (days 1-7, 8-14, 15-21, 22-31)
    level_weights: dict = field(default_factory=dict)      # candidate -> weight: s_u, l_u, a_u, s_w, l_w, a_w, fm (see the class docstring)
    fixed_division_daily_means_kw: Optional[np.ndarray] = None   # fixed division: historical level + daily path, before the handover (for ablation)
    pseudo_pairs: int = 0                     # completed pseudo-origin pairs behind the lead-week weights (at most 6)
    foundation_within_day_kw: Optional[np.ndarray] = None  # (744,) the foundation model's own within-day block
    analog_shape_kw: Optional[np.ndarray] = None           # (744,) the unit's analog-day shape (None when not built)
    within_trust: tuple = (0.0, 0.0, 0.0, 0.0)             # weight on the analog shape for lead blocks 1-7, 8-14, 15-21, 22-31 (one value repeated since 2.1)
    within_pseudo_pairs: int = 0              # completed pseudo-origin triples behind the within-day trust (at most 3)
    analog_kept: bool = False                 # False when the whole-window guard returned the model's shape (or no analog shape was built)
    micro_load: bool = False                  # True when the whole 1,344-hour context stayed within 1e-3 kW of zero and the model was used unchanged
    anchoring_record: dict = field(default_factory=dict)   # 2.2: the unit's own pseudo-forecast record on the level, read before the forecast (see anchoring_record_doc)

    @property
    def energy_kwh(self) -> float:
        """Energy readout (kWh): 744 x ``level_kw``, read before the nonnegativity projection."""
        return float(HORIZON * self.level_kw)


# ----------------------------------------------------------------------------- level with the model candidate
def _level_parts(history: History, group: str, sigma: float, min_target_obs: float = 1.0):
    """Level hypotheses, their pseudo-origin errors and the pseudo-origin bookkeeping of the reference estimator."""
    cfg = _api.frozen_config()
    if group not in GROUPS:
        raise _api.InputError("group must be one of: " + ", ".join(GROUPS))
    gi = GROUPS.index(group)
    curves = {i: np.asarray(cfg["source_curves"][g], dtype=np.float64) for i, g in enumerate(GROUPS) if g in cfg["source_curves"]}
    store = _api._store(history, gi, sigma)
    batcher = _api._PredictionBatcher(store, min_target_obs=min_target_obs)
    sigs = _api._BoundarySignatures(store, curves, "strict")
    system = SimpleNamespace(lam=cfg["signature_lambda"], mu=np.zeros(len(GROUPS)))
    b, o = np.array([0]), np.array([store.origin])
    a = cfg["level"]
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        h = _level.hypotheses(store, sigs, system, batcher, b, o, system.lam)
        err = _level.pseudo_errors(store, sigs, system, batcher, b, o, system.lam, a["K"])
        names, _, _ = _level.weights(err, a["K"], a["K0"], scale=a["scale"], prior=a["prior"], annual_in_w=a["annual_in_w"])
        pseudo = {}
        for k in range(1, K_FM + 1):
            ok = int(o[0] - k * STEP)
            if batcher.observed(0, ok):
                hk = _level.hypotheses(store, sigs, system, batcher, b, np.array([ok]), system.lam)
                pseudo[k] = {"o": ok, "l0": float(hk["l0"][0]), "s0": float(hk["s0"][0]), "truth": float(hk["truth"][0])}
    return {"h": h, "err": err, "names": names, "cfg": a, "l0": float(h["l0"][0]), "s0": float(h["s0"][0]),
            "c_bar": float(h["c_bar"][0]), "pseudo": pseudo}


def _weights(err, names, K, K0, scale, min_valid=2):
    """Inverse-MSE weights over the candidates, shrunk towards equal weights by K0 (the formula of the reference estimator in ``ankyra.history``)."""
    E = {n: err[n][:, :K] for n in names}
    valid = {n: np.isfinite(E[n]) for n in names}
    counts = {n: valid[n].sum(axis=1) for n in names}
    m = {}
    for n in names:
        a = np.abs(np.where(valid[n], E[n], np.nan))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-missing rows are handled by min_valid below
            m[n] = np.nanmedian(a, axis=1) if scale == "mad" else np.sqrt(np.nanmean(a ** 2, axis=1))
        m[n] = np.where(counts[n] >= min_valid, m[n], np.nan)
    avail = np.stack([np.isfinite(m[n]) for n in names], axis=1)
    inv = np.stack([np.where(np.isfinite(m[n]), 1.0 / np.maximum(m[n], 1e-6) ** 2, 0.0) for n in names], axis=1)
    what = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-300)
    w0 = avail / np.maximum(avail.sum(axis=1, keepdims=True), 1)
    k_eff = (valid["s_u"] & valid["l_u"]).sum(axis=1).astype(np.float64)[:, None]
    w = (k_eff * what + K0 * w0) / (k_eff + K0)
    none = avail.sum(axis=1) == 0
    w[none] = 1.0 / len(names)
    return w


def _level_path(h, names, w):
    H = np.stack([np.where(np.isfinite(h[n]), h[n], 0.0) for n in names], axis=1)
    w = np.where(np.isfinite(np.stack([h[n] for n in names], axis=1)), w, 0.0)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    lvl = (H * w).sum(axis=1)
    w_w = w[:, [i for i, n in enumerate(names) if n in _level.W_SET]].sum(axis=1)
    return lvl[:, None] + w_w[:, None] * h["g_h"] + h["c_d"], w


def anchoring_record(parts, fm_means_pseudo: Mapping[int, float], w_level: Mapping[str, float]) -> dict:
    """Two statistics of the unit's own pseudo-forecast record on the monthly level, computable before the forecast (2.2).

    carrier_level_weight          the weight the level block gives the foundation model's window mean (``level_weights['fm']``);
    history_vs_carrier_log_ratio  log of (the smallest pseudo-origin RMS error among the six historical level candidates) over
                                  (the pseudo-origin RMS error of the foundation model's window mean), both in the estimator's
                                  normalised units over the completed pseudo-origins with a model forecast; negative = the unit's
                                  history beat the model at its earlier origins; NaN with fewer than two such pseudo-origins;
    pseudo_origins                the number of pseudo-origins behind the ratio.
    Across the fourteen populations of the study the realised gain of ANKYRA over the foundation model fell overall, but not
    strictly monotonically, with both statistics (results/anchoring_gain_deciles.csv in the repository); the forecast itself
    does not use them."""
    names = list(parts["names"]); K = parts["cfg"]["K"]; err = parts["err"]
    efm = []
    for k in range(1, K + 1):
        pk = parts["pseudo"].get(k)
        if pk is not None and k in fm_means_pseudo:
            efm.append(((fm_means_pseudo[k] - pk["l0"]) / pk["s0"] - pk["truth"], k))
    ratio = float("nan")
    if len(efm) >= 2:
        ks = [k - 1 for _, k in efm]; rms_fm = float(np.sqrt(np.mean([e ** 2 for e, _ in efm])))
        hist = []
        for n in names:
            e = np.asarray(err[n][0, ks], dtype=np.float64); e = e[np.isfinite(e)]
            if len(e) >= 2:
                hist.append(float(np.sqrt(np.mean(e ** 2))))
        if hist and rms_fm > 0:
            ratio = float(np.log(max(min(hist), 1e-12) / rms_fm))
    return {"carrier_level_weight": float(w_level.get("fm", float("nan"))), "history_vs_carrier_log_ratio": ratio, "pseudo_origins": int(len(efm))}


def level_with_model_candidate(parts, fm_mean_origin: float, fm_means_pseudo: Mapping[int, float]):
    """Level (kW) with the foundation model's window mean as an extra candidate, and the candidate weights."""
    h, err, names, a = parts["h"], parts["err"], parts["names"], parts["cfg"]
    K = a["K"]
    err_x = {n: err[n].copy() for n in names}
    err_x["fm"] = np.full((1, K), np.nan)
    for k in range(1, K + 1):
        p = parts["pseudo"].get(k)
        if p is None or k not in fm_means_pseudo:
            continue
        err_x["fm"][0, k - 1] = (fm_means_pseudo[k] - p["l0"]) / p["s0"] - p["truth"]
    names_x = list(names) + ["fm"]
    h_x = dict(h)
    h_x["fm"] = np.array([(fm_mean_origin - parts["l0"]) / parts["s0"] - parts["c_bar"]])
    w = _weights(err_x, names_x, K, a["K0"], a["scale"])
    la, used = _level_path(h_x, names_x, w)
    return float(parts["l0"] + parts["s0"] * la[0].mean()), dict(zip(names_x, used[0].tolist()))


# ----------------------------------------------------------------------------- week-by-week handover
def lead_week_weights(pairs, k0: float = K0_WEEK):
    """Least-squares weight on the model's daily means for each lead week, shrunk towards 1/2.

    pairs: list of (realised daily means, fixed-division daily means, model daily means), each (31,), from completed
    pseudo-origins; the fixed division is the six-candidate historical level plus the daily path.
    Returns four weights alpha_w in [0, 1]."""
    out = []
    for a0, a1 in WEEKS:
        num = den = 0.0
        n = 0
        for y, hh, ff in pairs:
            e_h = hh[a0:a1] - y[a0:a1]
            d = ff[a0:a1] - hh[a0:a1]
            m = np.isfinite(e_h)                                      # 2.2: days of a partially observed pseudo target drop out
            num += float((-e_h[m] * d[m]).sum())
            den += float((d[m] * d[m]).sum())
            n += 1
        lam = min(max(num / den, 0.0), 1.0) if den > 0 else 0.5
        out.append((n * lam + k0 * 0.5) / (n + k0))
    return out


def lead_week_transition(daily_hist, daily_fm, alphas):
    """Week-wise convex combination of the historical and model daily means."""
    out = np.array(daily_hist, dtype=np.float64, copy=True)
    for (a0, a1), al in zip(WEEKS, alphas):
        out[a0:a1] = al * daily_fm[a0:a1] + (1 - al) * daily_hist[a0:a1]
    return out


# ----------------------------------------------------------------------------- driver
def fill_short_gaps(load_kw: np.ndarray, max_hours: int = GAP_MAX_H) -> np.ndarray:
    """Gap tolerance (2.2): a copy of the pre-origin load in which every gap of at most ``max_hours`` missing hours is linearly
    interpolated between its two observed neighbours, provided the gap and both neighbours lie inside one 744-hour block counted
    back from the origin (the end of the array).  A pseudo-origin o - 744k is a block boundary, so no interpolated hour before a
    pseudo-origin uses a value at or after it.  Gaps that touch the origin, the start, a block boundary, or are longer than
    ``max_hours`` are left as NaN.  A record without such gaps is returned unchanged."""
    load = np.array(load_kw, dtype=np.float64); o = len(load)
    fin = np.isfinite(load)
    if fin.all() or o == 0:
        return load
    block = (o - 1 - np.arange(o)) // STEP                                        # block index of each hour, 0 = the last 744 h
    i = 0
    while i < o:
        if fin[i]:
            i += 1; continue
        j = i
        while j < o and not fin[j]:
            j += 1
        if 1 <= j - i <= max_hours and i > 0 and j < o and block[i - 1] == block[j]:
            load[i:j] = load[i - 1] + (load[j] - load[i - 1]) * (np.arange(1, j - i + 1) / (j - i + 1))
        i = j
    return load


def pseudo_origin_contexts(load_kw: np.ndarray, k_max: int = K_FM, gap_tolerance: bool = True):
    """Contexts the foundation model must forecast from: k = 0 (the origin) and pseudo-origins o - 744k.

    Returns {k: (1344,) context} for every k whose context is completely observed; with ``gap_tolerance`` (2.2, default) the
    pseudo-origin contexts (k >= 1) are taken from :func:`fill_short_gaps`, as the forecaster's own bookkeeping does; the
    context at the origin (k = 0) is always the raw record."""
    load = np.asarray(load_kw, dtype=np.float64)
    o = len(load)
    filled = fill_short_gaps(load) if gap_tolerance else load
    out = {}
    for k in range(0, k_max + 1):
        ok = o - k * STEP; src = load if k == 0 else filled
        if ok - CONTEXT >= 0 and np.isfinite(src[ok - CONTEXT:ok]).all():
            out[k] = src[ok - CONTEXT:ok]
    return out


def _foundation_map(foundation: Foundation, load_kw) -> dict:
    """{k: forecast issued at o - 744k} from either form of the ``foundation`` argument of :func:`forecast`."""
    if callable(foundation):
        ctx = pseudo_origin_contexts(load_kw)
        if 0 not in ctx:          # without this the model would be called on the pseudo-origins only, or on nothing
            raise _api.InputError("the final 1344 load hours must be completely observed "
                                  "(load_kw needs at least 1344 finite hourly values before the origin)")
        keys = sorted(ctx)
        pred = np.asarray(foundation(np.stack([ctx[k] for k in keys])), dtype=np.float64)
        if pred.shape != (len(keys), HORIZON):
            raise ValueError(f"foundation forecaster must return ({len(keys)}, {HORIZON}), got {pred.shape}")
        return dict(zip(keys, pred))
    if not hasattr(foundation, "items"):
        raise TypeError("foundation must be a mapping {k: (744,) forecast}, a callable (N, 1344) -> (N, 744), or None")
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in foundation.items() if v is not None}


def _check_pseudo_forecasts(T: dict, used, within_anchor: bool) -> None:
    """A clear error for a pseudo-origin forecast of the wrong shape, raised only where the computation would fail on it.

    ``used`` are the pseudo-origins k >= 1 that are both completed in the record and present in ``T``.  An entry of the
    wrong size fails in the handover; an entry of the right size but another shape fails in the within-day anchoring,
    which reads k = 1..3.  Entries that are never read are not checked."""
    for k in used:
        shape = T[k].shape
        if shape != (HORIZON,) and (T[k].size != HORIZON or (within_anchor and k <= _analog.K_WITHIN)):
            raise ValueError(f"foundation forecast k={k} must have shape ({HORIZON},), got {shape}")
        if not np.isfinite(np.asarray(T[k], dtype=np.float64)).all():                       # 2.2.1: a NaN here used to drop the
            raise ValueError(f"foundation forecast k={k} must be finite")                     # pseudo-origin's evidence silently


def forecast(history: History, *, group: str, temp_sigma_std: float, foundation: Optional[Foundation], dst_region: str = "none",
             within_anchor: bool = True, micro_load_rule: bool = True, single_trust: bool = True, gap_tolerance: bool = True) -> AnkyraForecast:
    """ANKYRA forecast for the 744 hours after the end of ``history``.

    history         pre-origin hourly load and temperature and the calendar through the horizon (``ankyra.History``)
    group           category of the unit: Industrial, Office, Public, Residential or Commercial.  It selects the fixed
                    temperature-signature prior.  Commercial has no prior curve: its temperature signature is zero, so
                    its weather-adjusted candidates equal the unadjusted ones.  (The fifth curve of
                    ``frozen_config.json`` is stored under the key ``group_4`` and is not read.)  This is the behaviour
                    that was evaluated - 13 BDG2 units and one Suzhou series were Commercial - and it is kept.
    temp_sigma_std  fixed scale of the daily temperature anomaly, in units of 10 degC (0.25 means 2.5 degC).  The study
                    computed it once per population, before the first origin, and kept it fixed: standardise the
                    temperature as (T - 15 degC) / 10 degC, take daily means, subtract a centred 31-day moving mean;
                    temp_sigma_std is the standard deviation of that anomaly, pooled over the units of the population,
                    on a fit period that ends at or before the population's first scored origin (15 days dropped at
                    each end): the first 244 days of the record for BDG2, the GoiEner non-household sets and the
                    Suzhou park, and the complete days before the first scored origin for the other seven
                    populations (docs/DATA.md lists the rule of each population).  The per-population values are not
                    listed in the repository.
    foundation      either {k: (744,) forecast issued at o - 744k} for k = 0..6 (k = 0 required, finite), or a callable
                    that maps an (N, 1344) array of contexts to (N, 744) point forecasts (e.g.
                    ``ankyra.timesfm_adapter``), or None for the history-only configuration (no foundation model:
                    within-day default, no model candidate, no handover; a reduced configuration kept for ablation and
                    offline use)
    dst_region      daylight-saving rule used to match analog days: "EU", "US" or "none".  It is read only when the
                    within-day anchoring runs (not with within_anchor False or foundation None, and not on an off-state
                    or micro-load return)
    within_anchor   False leaves the foundation model's within-day block unchanged
    micro_load_rule False switches off the hand-over of contexts that stay within 1e-3 kW of zero
    single_trust    False estimates the within-day trust separately for each lead block (as up to 2.0.1) instead of one
                    value per window (2.1)
    gap_tolerance   the short-gap rule of 2.2.  True (default): the estimator reads a copy of the pre-origin record in
                    which gaps of at most 6 hours inside one 744-hour block are interpolated (``fill_short_gaps``),
                    and a pseudo-origin's target month counts for the seven-candidate level and the handover pairs
                    when at least 90% of its hours are observed.  The context at the origin is never filled.  False
                    gives the 2.1 forecast; on a record without gaps the two are identical.

    Earlier versions are reproduced by these combinations (all other switches at their defaults):
        2.1    gap_tolerance=False
        2.0.1  gap_tolerance=False, single_trust=False
        2.0.0  gap_tolerance=False, single_trust=False, micro_load_rule=False
        1.x    gap_tolerance=False, within_anchor=False, micro_load_rule=False

    Returns an :class:`AnkyraForecast` (all values in kW).

    Raises ``ankyra.InputError`` (a ``ValueError``) for inputs the estimator rejects: a category outside the five, a
    temp_sigma_std that is not a finite positive scalar, arrays of the wrong length, a gap in the last 1,344 load
    hours, a start timestamp that is not 1 January 00:00 UTC.  Raises ``ValueError`` for a missing, non-finite or
    mis-shaped foundation forecast and for an unknown dst_region, and ``TypeError`` for a ``foundation`` that is
    neither a mapping, a callable nor None.  Raises ``ankyra.SignatureDegeneracy`` when the temperature record does
    not vary over the year before the origin (for example a constant fill): the temperature signature cannot be
    fitted then.  If the last 168 hours are at or below 1e-6 kW (off-state rule) the foundation-model forecast is
    returned before the estimator's input checks run, as in every evaluated version.
    """
    history = _masked(history)                                       # one masked record for every branch (2.0.2)
    load = np.asarray(history.load_kw, dtype=np.float64)
    temp = np.asarray(history.temperature_c, dtype=np.float32)
    types = np.asarray(history.day_types)
    obs = np.isfinite(load) if history.observed is None else np.asarray(history.observed, dtype=bool)
    o = len(load)
    if foundation is None:
        return _history_only(history, load, temp, types, group, temp_sigma_std)
    T = _foundation_map(foundation, load)
    if 0 not in T or T[0].shape != (HORIZON,):
        raise ValueError("a 744-hour foundation-model forecast at the origin (k = 0) is required, as an array of shape (744,)")
    T0 = T[0]
    if not np.isfinite(T0).all():
        raise ValueError("the foundation-model forecast at the origin (k = 0) must be finite")
    wT0 = within_day(T0)

    off = bool(np.nanmax(load[o - 168:o]) <= ZERO_KW)                                 # off-state rule
    ctx = load[max(o - CONTEXT, 0):o]                                                 # micro-load rule (2.0.1): a completely observed
    micro = bool(micro_load_rule and len(ctx) == CONTEXT and obs.shape == load.shape and obs[o - len(ctx):o].all()   # context that stays within
                 and np.isfinite(ctx).all() and np.abs(ctx).max() <= MICRO_KW)        # the scale floor of zero
    if micro and not off:                                                            # the inputs must be ones the estimator would accept
        _check_inputs(history, group, temp_sigma_std)
    if off or micro:
        return AnkyraForecast(trajectory_kw=T0.copy(), level_kw=float(T0.mean()), daily_means_kw=T0.reshape(D, HR).mean(1),
                              within_day_kw=wT0, off_state=off, micro_load=micro, foundation_within_day_kw=wT0)

    _check_inputs(history, group, temp_sigma_std)                    # the estimator's own checks, before any computation
    if within_anchor and dst_region not in _analog.REGIONS:
        raise ValueError(f"dst_region must be one of {_analog.REGIONS}")
    if gap_tolerance:                                                # 2.2: short gaps interpolated in the whole pre-origin record that the
        load = fill_short_gaps(load); obs = np.isfinite(load)         # estimator reads; the origin context has no gap (checked above)
        history = replace(history, load_kw=load, observed=obs)
    parts = _level_parts(history, group, temp_sigma_std, min_target_obs=TARGET_MIN_OBS if gap_tolerance else 1.0)
    _check_pseudo_forecasts(T, [k for k in range(1, K_FM + 1) if k in parts["pseudo"] and k in T], within_anchor)
    est = estimate_from_history(history, group=group, temp_sigma_std=temp_sigma_std)
    daily_f0 = est.level_kw + np.asarray(est.daily_path_kw, dtype=np.float64)          # fixed division of labour

    fm_ps = {k: float(T[k].mean()) for k in range(1, K_FM + 1) if k in parts["pseudo"] and k in T}
    lvl, w_level = level_with_model_candidate(parts, float(T0.mean()), fm_ps)
    record = anchoring_record(parts, fm_ps, w_level)
    daily_level = daily_f0 - daily_f0.mean() + lvl                                     # model candidate in the level

    pairs = []                                   # fixed division vs the model at completed pseudo-origins (see module docstring)
    for k in range(1, K_FM + 1):
        p = parts["pseudo"].get(k)
        if p is None or k not in T:
            continue
        ok = p["o"]
        ek = estimate_from_history(History(load_kw=load[:ok], temperature_c=temp[:ok], day_types=types[:ok + HORIZON],
                                           start_timestamp=history.start_timestamp, observed=obs[:ok]),
                                   group=group, temp_sigma_std=temp_sigma_std)
        yk = load[ok:ok + HORIZON].reshape(D, HR); nk = np.isfinite(yk).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yday = np.where(nk >= 12, np.nanmean(yk, axis=1), np.nan)    # 2.2: a day needs 12 observed hours; complete data: plain means
        pairs.append((yday, ek.level_kw + ek.daily_path_kw, T[k].reshape(D, HR).mean(1)))
    alphas = lead_week_weights(pairs) if pairs else [0.5] * 4
    daily = lead_week_transition(daily_level, T0.reshape(D, HR).mean(1), alphas)

    W, S0, trust, n_w, kept = wT0, None, (0.0, 0.0, 0.0, 0.0), 0, False
    if within_anchor:                                                 # within-day anchoring (2.0)
        shapes = _analog.AnalogShapes(load, temp, types, history.start_timestamp, dst_region, origin=o)
        S0, kept = shapes.shape_with_sanity(o, wT0)
        triples = []
        for k in range(1, _analog.K_WITHIN + 1):
            q = o - k * STEP
            if k not in T or q - CONTEXT < 0 or not (obs[q - CONTEXT:q + HORIZON].all() and np.isfinite(load[q - CONTEXT:q + HORIZON]).all()):
                continue
            wTq = within_day(T[k]); Sq, _ = shapes.shape_with_sanity(q, wTq)
            triples.append((Sq, wTq, within_day(load[q:q + HORIZON])))
        w, n_w = _analog.within_trust(triples, single=single_trust); trust = tuple(float(x) for x in w)
        W = _analog.anchored_within_day(wT0, S0, w)

    raw = np.repeat(daily, HR) + W
    return AnkyraForecast(trajectory_kw=np.maximum(raw, 0.0), level_kw=float(daily.mean()), daily_means_kw=daily,
                          within_day_kw=W, off_state=False, lead_week_weights=tuple(float(a) for a in alphas),
                          level_weights=w_level, fixed_division_daily_means_kw=daily_f0, pseudo_pairs=len(pairs), anchoring_record=record,
                          foundation_within_day_kw=wT0, analog_shape_kw=S0, within_trust=trust, within_pseudo_pairs=n_w, analog_kept=kept)


def _masked(history: History) -> History:
    """The history with every load value marked unobserved (``observed`` False) replaced by NaN.

    The historical estimator always masked these hours; before 2.0.2 the foundation-model contexts, the off-state and
    micro-load rules and the analog days read the raw array, so a finite placeholder at an unobserved hour could change
    the forecast.  With the mask applied once here, every branch sees the same record and a placeholder value has no
    effect.  A malformed mask is rejected with the estimator's own message.  When ``observed`` is None, or equals
    ``isfinite(load_kw)`` (every evaluation in the study), the record is unchanged."""
    if history.observed is None:
        return history
    load = np.asarray(history.load_kw, dtype=np.float64); obs = np.asarray(history.observed)
    if obs.dtype != np.bool_ or obs.shape != load.shape:
        raise _api.InputError("observed must be a boolean mask of the load history")
    return replace(history, load_kw=np.where(obs, load, np.nan))


def _check_inputs(history, group, temp_sigma_std) -> None:
    """The estimator's own input checks, in its order (scale, category, history), run before any computation.

    They are the checks of ``estimate_from_history``; running them first gives the same readable error on the
    micro-load return, which never reaches the estimator, and on the ordinary path, where the level computation would
    otherwise fail first with a less clear message."""
    if not np.isscalar(temp_sigma_std) or not np.isfinite(temp_sigma_std) or temp_sigma_std <= 0:
        raise _api.InputError("temp_sigma_std must be a finite, positive scalar (the fixed temperature-anomaly scale, in units of 10 degC)")
    if group not in GROUPS:
        raise _api.InputError("group must be one of: " + ", ".join(GROUPS))
    _api._validate_history(history)


def _history_only(history, load, temp, types, group, temp_sigma_std) -> AnkyraForecast:
    """History-only configuration: fixed division with the same-day-type within-day default (no foundation model)."""
    from .readouts import within_day_default
    o = len(load)
    est = estimate_from_history(history, group=group, temp_sigma_std=temp_sigma_std)
    daily = est.level_kw + np.asarray(est.daily_path_kw, dtype=np.float64)
    ctx = load[o - CONTEXT:o]
    if not np.isfinite(ctx).all():
        raise ValueError("the history-only configuration needs a complete 1,344-hour context")
    ct = types[o - CONTEXT + 12 + HR * np.arange(CONTEXT // HR)]; tt = types[o + 12 + HR * np.arange(D)]
    W = within_day_default(ctx[None], ct[None], tt[None])[0]
    raw = np.repeat(daily, HR) + W
    return AnkyraForecast(trajectory_kw=np.maximum(raw, 0.0), level_kw=float(daily.mean()), daily_means_kw=daily, within_day_kw=W,
                          off_state=False, fixed_division_daily_means_kw=daily)
