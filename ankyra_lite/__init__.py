"""ankyra-lite: a small self-developed carrier network feeding the ANKYRA 2.2.1 anchoring layer, for month-ahead
(744-hour) hourly building-load forecasting. No foundation model is needed.

  ankyra_lite.carrier   the carrier network (61,613 parameters), its input features, inference and training recipe
  ankyra_lite.anchor    a byte-identical copy of the ANKYRA 2.2.1 anchoring layer
  ankyra_lite.forecast  the end-to-end forecast (carrier forecasts at the origin and pseudo-origins -> anchoring layer)
  ankyra_lite.readouts  readouts of a forecast: the prediction interval, the block split, the energy added by the
                        nonnegativity projection, the monthly peak

Top-level names:

  forecast            end-to-end anchored forecast (``ankyra_lite.forecast.forecast``); ``forecast(history, None, ...)``
                      is the history-only configuration
  interval            the prediction-interval readout around a forecast (``ankyra_lite.readouts.interval``)
  readouts            the module ``ankyra_lite.readouts``
  build_carrier       an untrained carrier network in the published configuration
  load_carrier        a carrier network from a checkpoint file (strict load, eval mode)
  carrier_forecasts   carrier forecasts at the origin and at the pseudo-origins the anchoring layer reads
  features            the carrier's input features for one forecast origin
  ForecastResult      the result type of ``forecast``
  History, InputError, SignatureDegeneracy, fill_short_gaps, pseudo_origin_contexts
                      the anchoring layer's input record, its two exceptions, its short-gap helper and the contexts it
                      reads (``ankyra_lite.anchor``)

Importing ``forecast`` here rebinds the package attribute ``ankyra_lite.forecast`` to the function. The submodule is
still importable by name, e.g. ``from ankyra_lite.forecast import ForecastResult, seasonal_naive``.
"""
from .anchor import History, InputError, SignatureDegeneracy, fill_short_gaps, pseudo_origin_contexts
from .carrier import build_carrier, carrier_forecasts, features, load_carrier
from .forecast import ForecastResult, forecast
from . import readouts
from .readouts import interval

__version__ = "0.1.1"
__all__ = ["forecast", "interval", "readouts", "build_carrier", "load_carrier", "carrier_forecasts", "features", "ForecastResult",
           "History", "InputError", "SignatureDegeneracy", "fill_short_gaps", "pseudo_origin_contexts", "__version__"]
