"""ankyra-lite: a small self-developed carrier network feeding the ANKYRA 2.2.1 anchoring layer, for month-ahead
(744-hour) hourly building-load forecasting. No foundation model is needed.

  ankyra_lite.carrier       the carrier network (61,613 parameters), its input features, inference and training recipe
  ankyra_lite.anchor        a byte-identical copy of the ANKYRA 2.2.1 anchoring layer
  ankyra_lite.continuation  the origin-continuity layer: a correction of the first hours and days of the anchored
                            trajectory toward the last observed hours (four frozen constants; energy and peak are read
                            before it)
  ankyra_lite.forecast      the end-to-end forecast (carrier forecasts at the origin and pseudo-origins -> anchoring
                            layer -> state rule -> optional continuation layer)
  ankyra_lite.readouts      readouts of a forecast: the prediction interval, the block split, the energy added by the
                            nonnegativity projection, the energy changed by the continuation layer, the monthly peak

Top-level names:

  forecast            end-to-end anchored forecast (``ankyra_lite.forecast.forecast``); ``forecast(history, None, ...)``
                      is the history-only configuration
  interval            the prediction-interval readout around a forecast (``ankyra_lite.readouts.interval``)
  readouts            the module ``ankyra_lite.readouts``
  continuation        the module ``ankyra_lite.continuation``
  build_carrier       an untrained carrier network in the published configuration
  load_carrier        a carrier network from a checkpoint file (strict load, eval mode)
  checkpoint_paths    the release's checkpoint files of a population, in ascending seed order
                      (``ankyra_lite._release.checkpoint_paths``)
  carrier_forecasts   carrier forecasts at the origin and at the pseudo-origins the anchoring layer reads
  features            the carrier's input features for one forecast origin
  ForecastResult      the result type of ``forecast``
  History, InputError, SignatureDegeneracy, fill_short_gaps, pseudo_origin_contexts
                      the anchoring layer's input record, its two exceptions, its short-gap helper and the contexts it
                      reads (``ankyra_lite.anchor``)

Importing ``forecast`` here rebinds the package attribute ``ankyra_lite.forecast`` to the function. The submodule is
still importable by name, e.g. ``from ankyra_lite.forecast import ForecastResult, seasonal_naive``.  The release
configuration (default of the continuation layer, carrier recipe and seeds of the default checkpoints, version) is set
in ``ankyra_lite._release``.
"""
from . import _release
from ._release import checkpoint_paths
from .anchor import History, InputError, SignatureDegeneracy, fill_short_gaps, pseudo_origin_contexts
from .carrier import build_carrier, carrier_forecasts, features, load_carrier
from .forecast import ForecastResult, forecast
from . import continuation, readouts
from .readouts import interval

__version__ = _release.VERSION
__all__ = ["forecast", "interval", "readouts", "continuation", "build_carrier", "load_carrier", "checkpoint_paths",
           "carrier_forecasts", "features", "ForecastResult", "History", "InputError", "SignatureDegeneracy", "fill_short_gaps",
           "pseudo_origin_contexts", "__version__"]
