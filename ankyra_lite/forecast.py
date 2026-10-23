"""End-to-end month-ahead forecast: the small carrier network feeding the ANKYRA anchoring layer, optionally followed
by the origin-continuity layer.

    from ankyra_lite import forecast, load_carrier
    from ankyra_lite.anchor import History
    h = History(load_kw=..., temperature_c=..., day_types=..., start_timestamp="2016-01-01T00:00:00+00:00")
    r = forecast(h, [load_carrier(p) for p in checkpoint_paths], group="Office", temp_sigma_std=0.25)
    r.trajectory_kw, r.level_kw, r.energy_kwh, r.peak_kw

Steps
  1. carrier forecasts at the origin and at the pseudo-origins the anchoring layer reads
     (:func:`ankyra_lite.carrier.inference.carrier_forecasts`); an ensemble averages its members at every issue time;
  2. the anchoring layer, ``ankyra_lite.anchor.forecast(history, foundation={k: forecast}, ...)``;
  3. the state rule of the evaluated recipe: on a state window (the anchoring layer reports ``off_state`` or
     ``micro_load``, i.e. the last 168 hours are at or below 1e-6 kW, or the whole 1,344-hour context stays within
     1e-3 kW of zero) the anchoring layer hands back the carrier forecast unchanged; with
     ``state_rule="seasonal_naive"`` (default) the delivered forecast is instead the last observed week repeated
     (missing hours as 0), and the level readout is its mean. ``state_rule=None`` keeps the anchoring layer's output;
  4. the monthly peak readout of the anchoring layer (``anchor.readouts.peak_readout``) on the trajectory of step 3;
  5. with ``continuation`` on, the origin-continuity layer (:mod:`ankyra_lite.continuation`) on the trajectory of
     step 3: F'(h) = F(h) exp((kappa1 x_f) exp(-h / 12) + (kappa2 x_d) exp(-h / 168)), where x_f and x_d compare the
     last 3 and 24 observed hours with the forecast's own reference day.  State windows bypass it.

Readout rule.  Energy, level and peak are read before step 5: ``level_kw``, ``energy_kwh`` and ``peak_kw`` are bit for
bit the same with the layer on and off, and ``trajectory_before_continuation_kw`` keeps the trajectory of step 3.
Only the hourly trajectory ``trajectory_kw`` (and an interval placed around it) changes.  With the layer on, the mean
of ``trajectory_kw`` therefore differs from ``level_kw``; ``ankyra_lite.readouts.continuation_energy_change`` gives
sum(trajectory_kw) - sum(trajectory_before_continuation_kw), known at the origin.

``forecast(history, None, ...)`` is the history-only configuration: ``ankyra_lite.anchor.forecast(...,
foundation=None)``, the historical level and daily path with the same-day-type within-day default (no carrier, no
carrier level candidate, no handover, no within-day anchoring). The anchoring layer applies neither the off-state nor
the micro-load rule to it, so no state rule is applied either; step 4 reads the peak from its trajectory.  The
continuation layer is not applied to it (that configuration was not evaluated with the layer).

The interval, the block split and the energy added by the projection are in :mod:`ankyra_lite.readouts`.

Only the public API of the anchoring layer is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from . import _release
from . import continuation as _continuation
from .carrier.inference import CONTEXT, HORIZON, STEP, carrier_forecasts

__all__ = ["forecast", "ForecastResult", "seasonal_naive", "peak"]

HR, WEEK = 24, 168


def _anchor():
    from . import anchor
    return anchor


@dataclass(frozen=True)
class ForecastResult:
    """Delivered month-ahead forecast. All values in kW; ``energy_kwh`` = 744 x ``level_kw``.

    trajectory_kw       (744,) delivered hourly forecast: after the origin-continuity layer when it is on, otherwise
                        the same array as ``trajectory_before_continuation_kw``
    level_kw            delivered window mean readout (the anchoring layer's level, before its nonnegativity
                        projection; on a state window with the seasonal-naive rule, the mean of that forecast); read
                        before the continuation layer
    peak_kw             monthly peak readout of ``trajectory_before_continuation_kw``
    state_window        the anchoring layer reported off_state or micro_load (always False for the history-only
                        configuration, to which neither rule applies)
    state_rule          the state rule that was applied ("seasonal_naive") or None
    anchored            the anchoring layer's own result (``ankyra_lite.anchor.AnkyraForecast``)
    carrier             {k: (744,)} the carrier forecasts passed to the anchoring layer ({} for the history-only
                        configuration)
    trajectory_before_continuation_kw
                        (744,) the trajectory before the continuation layer: the anchoring layer's output after the
                        state rule (the delivered trajectory of ankyra-lite 0.1.x).  Defaults to ``trajectory_kw``
    continuation        the ``ankyra_lite.continuation.ContinuationRecord`` of the layer (its anomalies, reference day,
                        Delta(h) and bypass), or None when the layer is off and for the history-only configuration

    Readout rule: energy, level and peak are read before the continuation layer and do not change when it is switched
    on; with it on, ``trajectory_kw.mean()`` differs from ``level_kw`` and ``trajectory_kw.sum()`` from
    ``energy_kwh`` (``ankyra_lite.readouts.continuation_energy_change``).  On a state window the layer is bypassed and
    ``trajectory_kw`` equals ``trajectory_before_continuation_kw`` bit for bit.

    Any other attribute (``off_state``, ``micro_load``, ``level_weights``, ``lead_week_weights``,
    ``anchoring_record``, ``daily_means_kw``, ``within_day_kw``, ``foundation_within_day_kw`` (the carrier's own
    within-day block), ``analog_shape_kw``, ``fixed_division_daily_means_kw``, ...) is read from ``anchored``.  On a
    state window these describe the anchoring layer's return (the carrier forecast at the origin, with the weights,
    the fixed division and the analog fields at their defaults), not the seasonal-naive trajectory delivered by the
    default state rule.
    """
    trajectory_kw: np.ndarray
    level_kw: float
    peak_kw: float
    state_window: bool
    state_rule: Optional[str]
    anchored: Any = field(repr=False)
    carrier: Mapping[int, np.ndarray] = field(repr=False, default_factory=dict)
    trajectory_before_continuation_kw: Optional[np.ndarray] = field(repr=False, default=None)
    continuation: Any = field(repr=False, default=None)

    def __post_init__(self):
        if self.trajectory_before_continuation_kw is None:
            object.__setattr__(self, "trajectory_before_continuation_kw", self.trajectory_kw)

    @property
    def energy_kwh(self) -> float:
        return float(HORIZON * self.level_kw)

    def __getattr__(self, name):
        if name.startswith("__") or name == "anchored":
            raise AttributeError(name)
        return getattr(self.anchored, name)


def _masked_load(history) -> np.ndarray:
    load = np.asarray(history.load_kw, dtype=np.float64)
    if history.observed is not None:
        load = np.where(np.asarray(history.observed, dtype=bool), load, np.nan)
    return load


def _continuation_constants(continuation):
    """None (layer off) or the ``ankyra_lite.continuation.Constants`` to apply."""
    if continuation is None or continuation is False:
        return None
    if continuation is True:
        return _continuation.CONSTANTS
    if isinstance(continuation, _continuation.Constants):
        return continuation
    raise ValueError("continuation must be True, False, None or an ankyra_lite.continuation.Constants")


def seasonal_naive(load_kw) -> np.ndarray:
    """The last 168 pre-origin hours repeated over the 744-hour horizon (missing hours as 0)."""
    last = np.nan_to_num(np.asarray(load_kw, dtype=np.float64)[-WEEK:], nan=0.0)
    return last[np.arange(HORIZON) % WEEK]


def peak(trajectory_kw, load_kw, day_types) -> float:
    """Monthly peak readout of a 744-hour trajectory from its daily means and the context's day-type excursions.

    The type of each origin-aligned day is read at its 12th hour, as everywhere in the anchoring layer
    (``anchor.readouts.residual_quantiles``, the history-only configuration). For an origin at midnight this is the
    calendar day's type; for an origin after noon the day's 24 hours lie mostly on the next calendar day."""
    load = np.nan_to_num(np.asarray(load_kw, dtype=np.float64), nan=0.0)
    o = len(load)
    ty = np.asarray(day_types)
    ctx = load[o - CONTEXT:o]
    ct = ty[o - CONTEXT + 12 + HR * np.arange(CONTEXT // HR)].astype(np.int64)
    tt = ty[o + 12 + HR * np.arange(HORIZON // HR)].astype(np.int64)
    return float(_anchor().readouts.peak_readout(np.asarray(trajectory_kw, np.float64), ctx[None], ct[None], tt[None])[0])


def forecast(history, carriers, *, group: str, temp_sigma_std: float, dst_region: str = "none", cut: Optional[int] = None,
             state_rule: Optional[str] = "seasonal_naive", continuation=_release.CONTINUATION_DEFAULT, device: str = "cpu",
             **anchor_kwargs) -> ForecastResult:
    """Anchored month-ahead forecast for the 744 hours after the end of ``history``.

    history          ``ankyra_lite.anchor.History`` (pre-origin load and temperature, day types through the horizon,
                     start timestamp January 1 00:00 UTC, optional observed mask)
    carriers         a carrier network, a sequence of carrier networks, a precomputed mapping {k: (744,) carrier
                     forecast issued at origin - 744 k}, or None for the history-only configuration
                     (``ankyra_lite.anchor.forecast(..., foundation=None)``; no state rule and no continuation layer
                     are applied, ``carrier`` is {}, and ``cut`` and ``device`` are not used).  A sequence is an
                     ensemble: at every issue time the members' forecasts are summed in float64 in the order given
                     and divided by their number; pass the seeds of a release configuration (three or six) in
                     ascending order
    group, temp_sigma_std, dst_region, **anchor_kwargs
                     passed to ``ankyra_lite.anchor.forecast``; ``group`` is also the carrier's category feature
    cut              the carrier's training cut (hour index), shared by all members of an ensemble: pseudo-origins
                     before it are not used (they are not computed, or are dropped from a precomputed mapping), and an
                     origin before it raises ValueError
    state_rule       "seasonal_naive" (the evaluated recipe) or None
    continuation     True (the origin-continuity layer with ``ankyra_lite.continuation.CONSTANTS``), False or None
                     (off), or an ``ankyra_lite.continuation.Constants`` (``continuation.ZERO`` reproduces the layer
                     switched off bit for bit).  When the mean load of the last 3 or 24 hours plus 0.01 x scale is at
                     or below 0 (negative loads), that anomaly is set to 0 and flagged in the record
                     (``nonpositive_fast`` / ``nonpositive_day``), as for an unobserved span.  The default is the
                     release setting ``ankyra_lite._release.CONTINUATION_DEFAULT``
    device           torch device for the carrier

    With ``continuation=False``, identical inputs, and the original R three-seed checkpoints, the result matches
    ankyra-lite 0.1.1 bit for bit. With it on, ``trajectory_kw`` is the
    corrected trajectory and every readout (level, energy, peak) is unchanged, so ``trajectory_kw.sum()`` differs from
    ``energy_kwh`` by a further ``ankyra_lite.readouts.continuation_energy_change`` (see :class:`ForecastResult`)."""
    if state_rule not in ("seasonal_naive", None):
        raise ValueError("state_rule must be 'seasonal_naive' or None")
    constants = _continuation_constants(continuation)
    anchor = _anchor()
    load = _masked_load(history)
    o = len(load)
    if carriers is None:                                     # history-only configuration: neither state rule applies to it
        r = anchor.forecast(history, foundation=None, group=group, temp_sigma_std=temp_sigma_std, dst_region=dst_region,
                            **anchor_kwargs)
        tr = np.asarray(r.trajectory_kw, dtype=np.float64)
        return ForecastResult(trajectory_kw=tr, level_kw=float(r.level_kw), peak_kw=peak(tr, load, history.day_types),
                              state_window=False, state_rule=None, anchored=r, carrier={},
                              trajectory_before_continuation_kw=tr, continuation=None)
    if isinstance(carriers, Mapping):
        F = {int(k): np.asarray(v, dtype=np.float64) for k, v in carriers.items()}
        if cut is not None:                                  # the rule of carrier_forecasts(..., cut=cut), on the mapping
            if o < int(cut):
                raise ValueError(f"the origin {o} lies before the carrier's training cut {int(cut)}")
            F = {k: v for k, v in F.items() if o - k * STEP >= int(cut)}
    else:
        F = carrier_forecasts(carriers, load, np.asarray(history.temperature_c), np.asarray(history.day_types), group,
                              history.start_timestamp, 0, o, cut=cut, device=device)
    r = anchor.forecast(history, foundation=F, group=group, temp_sigma_std=temp_sigma_std, dst_region=dst_region,
                        **anchor_kwargs)
    state = bool(r.off_state or r.micro_load)
    if state and state_rule == "seasonal_naive":
        tr = seasonal_naive(load)
        lv = float(tr.mean())
    else:
        tr = np.asarray(r.trajectory_kw, dtype=np.float64)
        lv = float(r.level_kw)
    pk = peak(tr, load, history.day_types)                   # readout rule: level, energy and peak before the layer
    if constants is None:
        tp, rec = tr, None
    else:
        tp, rec = _continuation.apply(tr, load, history.day_types, o, state=state, constants=constants)
    return ForecastResult(trajectory_kw=tp, level_kw=lv, peak_kw=pk, state_window=state,
                          state_rule=state_rule if state else None, anchored=r, carrier=F,
                          trajectory_before_continuation_kw=tr, continuation=rec)
