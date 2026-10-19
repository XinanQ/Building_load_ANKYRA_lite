"""The untrained reference forecasters of the comparison that need nothing but the 1,344-hour load context.

    seasonal_naive(context, 24)      the last day repeated            ("Seasonal naive (day)")
    seasonal_naive(context, 168)     the last week repeated           ("Seasonal naive (week)")
    week_profile(context)            the mean week of the last four   ("4-week profile")

``seasonal_naive(context, 168)`` is also the forecast ankyra-lite delivers on state windows (off-state or micro-load),
with missing hours set to 0 (``ankyra_lite.forecast.seasonal_naive``).  The other 18 of the 21 baselines of the
comparison (the zero-shot foundation models and their covariate-informed variants, the seven trained models of the
first release, LSTM-T-KAN-M (model and training code of its authors' public repository with their exp2 feature list),
Holt-Winters, MSTL, the per-unit ridge, the zero-shot gradient-boosting model and the last-year and previous-month
profiles) are described in docs/EVALUATION.md; their code is not part of this repository.
"""
from __future__ import annotations

import math

import numpy as np

HORIZON = 744


def seasonal_naive(context, period):
    """Repeat the last ``period`` hours of each context over the horizon.  context: (n, >= period)."""
    context = np.asarray(context, dtype=np.float64)
    return np.tile(context[:, -period:], (1, math.ceil(HORIZON / period)))[:, :HORIZON]


def week_profile(context, weeks=4):
    """Repeat the mean of the last ``weeks`` weeks, hour of week by hour of week."""
    context = np.asarray(context, dtype=np.float64)
    mean_week = context[:, -weeks * 168:].reshape(len(context), weeks, 168).mean(1)
    return np.tile(mean_week, (1, math.ceil(HORIZON / 168)))[:, :HORIZON]
