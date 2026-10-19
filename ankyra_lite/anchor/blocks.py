"""Orthogonal block decomposition of a 744-hour trajectory (31 origin-aligned days x 24 hours).

For any trajectory v the level l(v), the centred daily path b(v) and the within-day path w(v) are orthogonal
projections of ranks 1, 30 and 713 under the uniform hourly inner product, so for an error e = F - y

    mean(e**2) = l(e)**2 + mean(b(e)**2) + mean(w(e)**2).

Replacing one block of a trajectory therefore changes its mean squared error by exactly the change in that block's
loss.  All functions accept a single trajectory (744,) or a batch (N, 744).
"""
from __future__ import annotations

import numpy as np

D, HR, HORIZON = 31, 24, 744
CONTEXT = 1344            # hours of load before the origin that the foundation model and the readouts use (56 days)


def _cube(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != HORIZON:
        raise ValueError(f"expected trailing dimension {HORIZON}, got {x.shape}")
    return x.reshape(x.shape[:-1] + (D, HR))


def level(x):
    """Window mean, shape (...)."""
    return _cube(x).mean(axis=(-2, -1))


def daily_means(x):
    """31 daily means, shape (..., 31)."""
    return _cube(x).mean(axis=-1)


def daily_path(x):
    """Centred daily path b_d = daily mean - level, shape (..., 31); sums to zero."""
    dm = daily_means(x)
    return dm - dm.mean(axis=-1, keepdims=True)


def within_day(x):
    """Within-day path w = hourly value - its day's mean, shape (..., 744); every day sums to zero."""
    c = _cube(x)
    return (c - c.mean(axis=-1, keepdims=True)).reshape(c.shape[:-2] + (HORIZON,))


def compose(level_kw, daily_path_kw, within_kw):
    """level + centred daily path (repeated over the hours of each day) + within-day path."""
    lv = np.asarray(level_kw, dtype=np.float64)
    bp = np.asarray(daily_path_kw, dtype=np.float64)
    wd = np.asarray(within_kw, dtype=np.float64)
    return lv[..., None] + np.repeat(bp, HR, axis=-1) + wd


def block_losses(forecast, truth):
    """(level, daily-path, within-day) squared-error blocks whose sum equals the hourly MSE exactly."""
    e = np.asarray(forecast, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return level(e) ** 2, (daily_path(e) ** 2).mean(axis=-1), (within_day(e) ** 2).mean(axis=-1)
