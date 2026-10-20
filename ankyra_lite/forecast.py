"""End-to-end month-ahead forecast: the small carrier network feeding the ANKYRA anchoring layer.

    from ankyra_lite import forecast, load_carrier
    from ankyra_lite.anchor import History
    h = History(load_kw=..., temperature_c=..., day_types=..., start_timestamp="2016-01-01T00:00:00+00:00")
    r = forecast(h, [load_carrier(p) for p in checkpoint_paths], group="Office", temp_sigma_std=0.25)
    r.trajectory_kw, r.level_kw, r.energy_kwh, r.peak_kw

Steps
  1. carrier forecasts at the origin and at the pseudo-origins the anchoring layer reads
     (:func:`ankyra_lite.carrier.inference.carrier_forecasts`); an ensemble averages its members;
  2. the anchoring layer, ``ankyra_lite.anchor.forecast(history, foundation={k: forecast}, ...)``;
  3. the state rule of the evaluated recipe: on a state window (the anchoring layer reports ``off_state`` or
     ``micro_load``, i.e. the last 168 hours are at or below 1e-6 kW, or the whole 1,344-hour context stays within
     1e-3 kW of zero) the anchoring layer hands back the carrier forecast unchanged; with
     ``state_rule="seasonal_naive"`` (default) the delivered forecast is instead the last observed week repeated
     (missing hours as 0), and the level readout is its mean. ``state_rule=None`` keeps the anchoring layer's output;
  4. the monthly peak readout of the anchoring layer (``anchor.readouts.peak_readout``) on the delivered trajectory.

``forecast(history, None, ...)`` is the history-only configuration: ``ankyra_lite.anchor.forecast(...,
foundation=None)``, the historical level and daily path with the same-day-type within-day default (no carrier, no
carrier level candidate, no handover, no within-day anchoring). The anchoring layer applies neither the off-state nor
the micro-load rule to it, so no state rule is applied either; step 4 reads the peak from its trajectory.

The interval, the block split and the energy added by the projection are in :mod:`ankyra_lite.readouts`.

Only the public API of the anchoring layer is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .carrier.inference import CONTEXT, HORIZON, STEP, carrier_forecasts

__all__ = ["forecast", "ForecastResult", "seasonal_naive", "peak"]

HR, WEEK = 24, 168


def _anchor():
    from . import anchor
    return anchor


@dataclass(frozen=True)
class ForecastResult:
    """Delivered month-ahead forecast. All values in kW; ``energy_kwh`` = 744 x ``level_kw``.

    trajectory_kw       (744,) delivered hourly forecast
    level_kw            delivered window mean readout (the anchoring layer's level, before its nonnegativity
                        projection; on a state window with the seasonal-naive rule, the mean of that forecast)
    peak_kw             monthly peak readout of the delivered trajectory
    state_window        the anchoring layer reported off_state or micro_load (always False for the history-only
                        configuration, to which neither rule applies)
    state_rule          the state rule that was applied ("seasonal_naive") or None
    anchored            the anchoring layer's own result (``ankyra_lite.anchor.AnkyraForecast``)
    carrier             {k: (744,)} the carrier forecasts passed to the anchoring layer ({} for the history-only
                        configuration)

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
             state_rule: Optional[str] = "seasonal_naive", device: str = "cpu", **anchor_kwargs) -> ForecastResult:
    """Anchored month-ahead forecast for the 744 hours after the end of ``history``.

    history          ``ankyra_lite.anchor.History`` (pre-origin load and temperature, day types through the horizon,
                     start timestamp January 1 00:00 UTC, optional observed mask)
    carriers         a carrier network, a sequence of carrier networks (ensemble average), a precomputed mapping
                     {k: (744,) carrier forecast issued at origin - 744 k}, or None for the history-only
                     configuration (``ankyra_lite.anchor.forecast(..., foundation=None)``; no state rule is applied,
                     ``carrier`` is {}, and ``cut`` and ``device`` are not used)
    group, temp_sigma_std, dst_region, **anchor_kwargs
                     passed to ``ankyra_lite.anchor.forecast``; ``group`` is also the carrier's category feature
    cut              the carrier's training cut (hour index): pseudo-origins before it are not used (they are not
                     computed, or are dropped from a precomputed mapping), and an origin before it raises ValueError
    state_rule       "seasonal_naive" (the evaluated recipe) or None
    device           torch device for the carrier
    """
    if state_rule not in ("seasonal_naive", None):
        raise ValueError("state_rule must be 'seasonal_naive' or None")
    anchor = _anchor()
    load = _masked_load(history)
    o = len(load)
    if carriers is None:                                     # history-only configuration: neither state rule applies to it
        r = anchor.forecast(history, foundation=None, group=group, temp_sigma_std=temp_sigma_std, dst_region=dst_region,
                            **anchor_kwargs)
        tr = np.asarray(r.trajectory_kw, dtype=np.float64)
        return ForecastResult(trajectory_kw=tr, level_kw=float(r.level_kw), peak_kw=peak(tr, load, history.day_types),
                              state_window=False, state_rule=None, anchored=r, carrier={})
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
    return ForecastResult(trajectory_kw=tr, level_kw=lv, peak_kw=peak(tr, load, history.day_types), state_window=state,
                          state_rule=state_rule if state else None, anchored=r, carrier=F)
