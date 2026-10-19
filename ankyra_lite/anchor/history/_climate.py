"""Extracted frozen arithmetic; see provenance.json. Do not edit in place."""
from __future__ import annotations
import math
import numpy as np
CLIM_PRIOR_AMPLITUDE, CLIM_PRIOR_STRENGTH, CLIM_MIN_DAY = 10.0, 60.0, 19.5

def climatology(temp_hourly: np.ndarray, origin: int) -> np.ndarray:
    """Fixed-phase annual harmonic with a conjugate ridge prior on the amplitude; pre-origin days only."""
    days = origin // 24
    daily = temp_hourly[: days * 24].reshape(days, 24).mean(axis=1)
    t = np.arange(days) + 0.5
    cos = np.cos(2 * math.pi * (t - CLIM_MIN_DAY) / 365.25)
    # T = c0 - c1 * cos ; minimise sum (T - c0 + c1 cos)^2 + lam (c1 - prior)^2
    lam = CLIM_PRIOR_STRENGTH
    n, sc, scc = days, cos.sum(), (cos * cos).sum()
    sy, syc = daily.sum(), (daily * cos).sum()
    # normal equations in (c0, c1) for residual T - c0 + c1 cos
    a11, a12, a22 = n, -sc, scc + lam
    b1, b2 = sy, -syc + lam * CLIM_PRIOR_AMPLITUDE
    det = a11 * a22 - a12 * a12
    c1 = float(np.clip((a11 * b2 - a12 * b1) / det, 0.0, 25.0))
    c0 = (sy + c1 * sc) / n                      # re-solved for the (possibly clipped) amplitude
    centres = (origin + 12 + 24 * np.arange(31)) / 24.0
    return (c0 - c1 * np.cos(2 * math.pi * (centres - CLIM_MIN_DAY) / 365.25)).astype(np.float32)
