"""ANKYRA: anchoring a time-series foundation model to each unit's own history for month-ahead load forecasting.

2.0 anchors all three blocks: level, daily path (1.x) and, with ``ankyra.analog``, the within-day shape.
2.0.1 hands a context whose 1,344 hours all stay within 1e-3 kW of zero to the foundation model (micro-load rule).
Package 2.0.2 masks hours marked unobserved once for every branch; 2.0.3 changes documentation only (model 2.0.1).
2.1 estimates the within-day trust as one value per window instead of one per lead block.
2.2 adds the short-gap rule (``gap_tolerance``): the estimator reads a copy of the pre-origin record with gaps of at most
six hours interpolated, and a pseudo-origin's target month counts when 90% of its hours are observed.
``forecast(..., gap_tolerance=False)`` reproduces 2.1 and ``gap_tolerance=False, single_trust=False`` reproduces 2.0.1.
Package 2.2.1 adds a finite-value check of the pseudo-origin foundation forecasts (model 2.2; forecasts of 2.2.0)."""
from .history import History, InputError, SignatureDegeneracy
from .core import AnkyraForecast, forecast, pseudo_origin_contexts, fill_short_gaps, lead_week_weights, lead_week_transition
from . import blocks, readouts, metrics, analog

__version__ = "2.2.1"
__all__ = ["History", "InputError", "SignatureDegeneracy", "AnkyraForecast", "forecast", "pseudo_origin_contexts", "fill_short_gaps", "lead_week_weights",
           "lead_week_transition", "blocks", "readouts", "metrics", "analog"]
