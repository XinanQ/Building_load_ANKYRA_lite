"""An artificial building history for examples and tests (no observed data)."""
from __future__ import annotations

import numpy as np

from .history import History


def synthetic_history(hours: int = 16000, constant_temperature: bool = False, off_last_hours: int = 0) -> History:
    """Hourly load with daily and weekly cycles, a slow trend and a heating response; calendar through the horizon.

    Index 0 is 2019-01-01 00:00 UTC (a Tuesday).  Every 173rd day is a holiday (type 7).  ``off_last_hours`` sets the
    last hours before the origin to zero (a switched-off unit).  ``constant_temperature`` replaces the temperature by
    a constant 15 degC: a record on which the temperature signature cannot be fitted, so the forecaster raises
    ``SignatureDegeneracy``."""
    t = np.arange(hours, dtype=float)
    temp = 15 - 9 * np.cos(2 * np.pi * (t / 24 - 19.5) / 365.25) + 1.1 * np.sin(t / 100)
    if constant_temperature:
        temp[:] = 15
    load = 20 + 0.0002 * t + 2 * np.cos(2 * np.pi * t / 168) + 3 * np.sin(2 * np.pi * (t - 6) / 24) + 0.5 * np.maximum(10 - temp, 0)
    if off_last_hours:
        load[-off_last_hours:] = 0.0
    day = (np.arange(hours + 744) // 24 + 1) % 7
    day[(np.arange(hours + 744) // 24) % 173 == 0] = 7
    return History(load, temp, day.astype(np.int64), "2019-01-01T00:00:00+00:00")


def seasonal_naive(contexts):
    """Stand-in for a foundation model: repeat the last week.  (N, 1344) -> (N, 744)."""
    c = np.asarray(contexts, dtype=np.float64)
    return np.tile(c[:, -168:], (1, 5))[:, :744]
