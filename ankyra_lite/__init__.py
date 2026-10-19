"""ankyra-lite: a small self-developed carrier network feeding the ANKYRA 2.2.1 anchoring layer, for month-ahead
(744-hour) hourly building-load forecasting. No foundation model is needed.

  ankyra_lite.carrier   the carrier network (61,613 parameters), its input features, inference and training recipe
  ankyra_lite.anchor    a byte-identical copy of the ANKYRA 2.2.1 anchoring layer
  ankyra_lite.forecast  the end-to-end forecast (carrier forecasts at the origin and pseudo-origins -> anchoring layer)

Top-level names:

  forecast            end-to-end anchored forecast (``ankyra_lite.forecast.forecast``)
  build_carrier       an untrained carrier network in the published configuration
  load_carrier        a carrier network from a checkpoint file (strict load, eval mode)
  carrier_forecasts   carrier forecasts at the origin and at the pseudo-origins the anchoring layer reads
  features            the carrier's input features for one forecast origin
  ForecastResult      the result type of ``forecast``

Importing ``forecast`` here rebinds the package attribute ``ankyra_lite.forecast`` to the function. The submodule is
still importable by name, e.g. ``from ankyra_lite.forecast import ForecastResult, seasonal_naive``.
"""
from .carrier import build_carrier, carrier_forecasts, features, load_carrier
from .forecast import ForecastResult, forecast

__version__ = "0.1.0"
__all__ = ["forecast", "build_carrier", "load_carrier", "carrier_forecasts", "features", "ForecastResult", "__version__"]
