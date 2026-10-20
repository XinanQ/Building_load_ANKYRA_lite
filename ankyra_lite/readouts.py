"""Readouts of an ankyra-lite forecast: the prediction interval, the block split, the energy added by the
nonnegativity projection, and the monthly peak.

    from ankyra_lite import forecast, readouts
    r = forecast(history, carriers, group="Office", temp_sigma_std=0.25)
    bands, n = readouts.interval(history, r, group="Office", temp_sigma_std=0.25)   # rows: 5, 10, 50, 90, 95 %
    b = readouts.blocks(readouts.raw_trajectory(r))      # level, centred daily path, within-day path
    readouts.projection_energy_increase(r)               # kWh that max(F, 0) added to the delivered trajectory
    readouts.peak(r.trajectory_kw, load_kw, day_types)   # the peak readout behind r.peak_kw

Every function calls the public API of the anchoring layer (``ankyra_lite.anchor.readouts`` and
``ankyra_lite.anchor.blocks``) and nothing else; none of them changes the forecast.

Energy.  ``ForecastResult.energy_kwh`` is 744 x ``level_kw``, the level read before the nonnegativity projection.  The
projection max(F, 0) cannot increase any hour's absolute error when the truth is nonnegative; it raises the delivered
trajectory's energy by sum(max(-F, 0)), which is known at the origin (:func:`projection_energy_increase`).

Peak.  U = max_d (L_d + A_{t_d}): the largest, over the 31 forecast days, of the day's forecast mean L_d plus the
largest excursion above its own mean among the four most recent context days of the same day type (all context days
when the type is absent from the context; a holiday borrows Sunday when the context holds no holiday).
``ForecastResult.peak_kw`` is this readout of the delivered trajectory (kappa = 1).

Interval.  Hourly residuals of the unit's own history-only pseudo-forecasts (historical level and daily path with a
same-day-type within-day default, no carrier) at up to 12 completed pseudo-origins, normalised by each pseudo-window's
origin scale and pooled by hour of day and workday / other day, give empirical quantiles; they are added to the
point trajectory, scaled by the origin scale, clipped at zero and made monotone across levels.  The residuals are not
those of ankyra-lite's own forecasts and are not centred.  The interval reads the record with unobserved hours as
missing and applies neither the short-gap rule nor its 90% target rule: a pseudo-window counts only when its context
and target month are completely observed.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .forecast import HORIZON, HR, _anchor, _masked_load, peak
from .carrier.inference import CONTEXT

__all__ = ["interval", "blocks", "Blocks", "raw_trajectory", "projection_energy_increase", "peak", "LEVELS"]

LEVELS = (0.05, 0.1, 0.5, 0.9, 0.95)        # rows of ``interval``: rows 1 and 3 bound the 80% interval, 0 and 4 the 90% one


class Blocks(NamedTuple):
    """The three orthogonal blocks of a 744-hour trajectory (``ankyra_lite.anchor.blocks``)."""
    level_kw: np.ndarray            # window mean, shape (...)
    daily_path_kw: np.ndarray       # centred daily path, shape (..., 31); sums to zero
    within_day_kw: np.ndarray       # within-day path, shape (..., 744); every day sums to zero


def _point(result) -> np.ndarray:
    """The point trajectory of a ``ForecastResult`` or ``AnkyraForecast``, or a (744,) array given directly."""
    return np.asarray(getattr(result, "trajectory_kw", result), dtype=np.float64)


def _state_window(result) -> bool:
    s = getattr(result, "state_window", None)
    if s is None:                                       # an AnkyraForecast of the anchoring layer
        s = bool(getattr(result, "off_state", False) or getattr(result, "micro_load", False))
    return bool(s)


def interval(history, result, *, group: str, temp_sigma_std: float):
    """Prediction-interval readout around a forecast: (bands (5, 744) in kW, pseudo_windows).

    history          the ``ankyra_lite.anchor.History`` the forecast was made from
    result           the ``ForecastResult`` of ``ankyra_lite.forecast`` (or an ``AnkyraForecast``, or a (744,) point
                     trajectory); the bands are placed around its ``trajectory_kw``
    group, temp_sigma_std
                     the arguments of the forecast (the history-only pseudo-forecasts use the same estimator)

    Rows follow ``LEVELS`` (0.05, 0.1, 0.5, 0.9, 0.95): rows 1 and 3 bound the 80% interval, rows 0 and 4 the 90%
    interval, row 2 is the median band.  ``pseudo_windows`` is the number of completed pseudo-windows that supplied
    residuals (at most 12); with none, every band is NaN.  The three calls are those of
    ``ankyra_lite.anchor.readouts`` (``residual_quantiles``, ``origin_scale``, ``interval_bands``) with hour of day
    (o + arange(744)) % 24 and workday = day type < 5 of each forecast hour.  Of the forecast month only the calendar
    is read; the residuals come from pseudo-windows that end at or before the origin."""
    R = _anchor().readouts
    load = _masked_load(history)
    o = len(load)
    types = np.asarray(history.day_types)
    Q, n_windows = R.residual_quantiles(load, history.temperature_c, types, history.start_timestamp, group, temp_sigma_std, o)
    hour_of_day = (o + np.arange(HORIZON)) % HR
    workday = (types[o:o + HORIZON] < 5).astype(int)    # day types 0-4 are Monday to Friday; 5, 6 and 7 are weekend and holiday
    bands = R.interval_bands(_point(result), R.origin_scale(load[o - CONTEXT:o]), Q, hour_of_day, workday)
    return bands, n_windows


def blocks(trajectory_kw) -> Blocks:
    """Level, centred daily path and within-day path of a (744,) or (N, 744) trajectory.

    The three are orthogonal projections (ranks 1, 30 and 713), so ``ankyra_lite.anchor.blocks.compose`` of them gives
    the trajectory back, and the hourly mean squared error of a forecast splits exactly into the three blocks' losses
    (``ankyra_lite.anchor.blocks.block_losses``).  Pass :func:`raw_trajectory` for the blocks before the
    nonnegativity projection."""
    B = _anchor().blocks
    x = np.asarray(trajectory_kw, dtype=np.float64)
    return Blocks(B.level(x), B.daily_path(x), B.within_day(x))


def raw_trajectory(result) -> np.ndarray:
    """The forecast before the nonnegativity projection, (744,) kW.

    Off a state window this is the anchoring layer's daily means repeated over the hours plus its within-day block
    (``np.repeat(daily_means_kw, 24) + within_day_kw``), so ``max(raw, 0)`` is the delivered trajectory; for the
    history-only configuration likewise.  On a state window no projection is applied and the delivered trajectory is
    returned (a copy)."""
    if _state_window(result):
        return np.array(result.trajectory_kw, dtype=np.float64, copy=True)
    return np.repeat(np.asarray(result.daily_means_kw, dtype=np.float64), HR) + np.asarray(result.within_day_kw, dtype=np.float64)


def projection_energy_increase(result) -> float:
    """Energy (kWh) that the nonnegativity projection max(F, 0) added to the delivered trajectory; known at the origin.

    ``sum(trajectory_kw) - 744 * level_kw`` up to rounding; 0 on a state window, where no projection is applied."""
    if _state_window(result):
        return 0.0
    return float(_anchor().readouts.projection_energy_increase_kwh(raw_trajectory(result)))
