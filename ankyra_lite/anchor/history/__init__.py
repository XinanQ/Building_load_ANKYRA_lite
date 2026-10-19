"""Frozen reference estimator of the historical level and centred daily path (the history side of ANKYRA).

The modules whose names start with an underscore are extracted unchanged from the study's code and keep its internal
names; ``provenance.json`` records where each definition came from.  Use the entry points of ``api.py``, re-exported
here.  Interface version 1.1.0."""
from .api import History, Estimate, InputError, SignatureDegeneracy, estimate_from_history, frozen_config, score_against_observations
__version__ = '1.1.0'   # version of this frozen reference interface, not of the ankyra package (ankyra.__version__)
__all__ = ['History','Estimate','InputError','SignatureDegeneracy','estimate_from_history','frozen_config','score_against_observations']
