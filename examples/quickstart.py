"""End-to-end ankyra-lite forecast on an artificial building; no data or downloads needed.

    python examples/quickstart.py                          # the shipped checkpoints checkpoints/cambridge/seed*.pt
    python examples/quickstart.py --population households  # the shipped checkpoints of another population
    python examples/quickstart.py --checkpoints DIR        # every seed*.pt in DIR (an ensemble of the seeds)
    python examples/quickstart.py --untrained              # an untrained network of the same shape instead

The script computes the carrier network's forecasts at the origin and at the six pseudo-origins, anchors them with the
anchoring layer, and prints the readouts (energy, peak, an 80% interval) and the weights behind the forecast.  By
default it loads the three released Cambridge carriers shipped in ``checkpoints/cambridge/`` (seeds 0, 1 and 2,
averaged as an ensemble).  They belong to the Cambridge population and its training cut, and the artificial building is
almost exactly periodic, so the output shows the format, not typical values.  With ``--untrained``, or when the folder
holds no ``seed*.pt``, the carrier is an untrained network of the published shape (``build_carrier()``, seed 0), and
the output shows the data flow only.  The interval is ``ankyra_lite.readouts.interval``, the anchoring layer's readout
(residual quantiles of the unit's own history-only pseudo-forecasts, added to the delivered trajectory); its coverage
on the nine populations is described in the README (section Prediction intervals).  ``examples/own_unit.py`` shows how
to prepare a record of your own unit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                               # also runs from a checkout without ``pip install -e .``
    sys.path.insert(0, str(REPO))

from ankyra_lite import anchor, build_carrier, carrier_forecasts, forecast, load_carrier, readouts   # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                           # noqa: E402
from ankyra_lite.carrier import CARRIER_PARAMETERS                                   # noqa: E402
CONTEXT = 1344                                          # hours of load before the origin that the readouts use


def carriers(population, checkpoints=None, untrained=False):
    """(list of carrier networks, description): the checkpoints seed*.pt of ``checkpoints`` (default: the shipped
    checkpoints/<population>/ under the repository root); one untrained network (seed 0) with ``untrained`` or when
    the folder holds no seed*.pt."""
    folder = Path(checkpoints) if checkpoints else REPO / "checkpoints" / population
    paths = [] if untrained else sorted(folder.glob("seed*.pt"))
    if paths:
        return [load_carrier(p) for p in paths], f"{len(paths)} checkpoint(s) in {folder.name}/: " + ", ".join(p.name for p in paths)
    torch.manual_seed(0)
    reason = "--untrained given" if untrained else f"no seed*.pt in {folder.name}/"
    return [build_carrier().eval()], f"untrained stand-in (build_carrier(), seed 0; {reason})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", default="cambridge", help="checkpoints/<population>/seed*.pt are used when present")
    ap.add_argument("--checkpoints", default=None, help="folder with seed*.pt (overrides --population)")
    ap.add_argument("--untrained", action="store_true", help="use the untrained stand-in even if checkpoints exist")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)

    models, description = carriers(args.population, args.checkpoints, args.untrained)
    history = synthetic_history(16000)                      # 16,000 hours before the origin; day types through the horizon
    group, sigma = "Office", 0.25                           # category of the unit; fixed temperature-anomaly scale (units of 10 degC)
    t0 = time.perf_counter()

    # step 1: the carrier's forecasts at the origin (k = 0) and at the pseudo-origins origin - 744 k (k = 1..6)
    load = np.asarray(history.load_kw, dtype=np.float64)
    o = len(load)
    F = carrier_forecasts(models, load, history.temperature_c, history.day_types, group, history.start_timestamp, unit=0, origin=o)
    # step 2: the anchoring layer on those forecasts (one call does both: forecast(history, models, ...))
    f = forecast(history, F, group=group, temp_sigma_std=sigma, dst_region="EU")
    seconds = time.perf_counter() - t0

    # peak readout: daily means of the delivered trajectory plus the largest recent excursion of the same day type
    types = np.asarray(history.day_types)
    context = load[o - CONTEXT:o]
    ctx_types = types[o - CONTEXT + 12 + 24 * np.arange(56)]
    tgt_types = types[o + 12 + 24 * np.arange(31)]
    peak = anchor.readouts.peak_readout(f.trajectory_kw, context[None], ctx_types[None], tgt_types[None])[0]

    # interval readout: residual quantiles of the unit's own pseudo-origin forecasts, added to the point trajectory
    # (anchor.readouts.residual_quantiles, origin_scale and interval_bands, by hour of day and workday)
    bands, n_windows = readouts.interval(history, f, group=group, temp_sigma_std=sigma)   # rows: 5, 10, 50, 90, 95 %

    print(json.dumps({
        "carrier": description,
        "carrier_parameters": CARRIER_PARAMETERS,
        "carrier_issue_times_k": sorted(F),                 # k: forecast issued at origin - 744 k
        "state_window": f.state_window,
        "state_rule": f.state_rule,
        "energy_kwh": round(f.energy_kwh, 3),
        "level_kw": round(f.level_kw, 4),
        "peak_kw": round(float(peak), 3),                 # equals f.peak_kw
        "interval_80pct_first_hour_kw": [round(float(bands[1, 0]), 2), round(float(bands[3, 0]), 2)],
        "interval_pseudo_windows": n_windows,
        "lead_week_weights_on_carrier": [round(a, 3) for a in f.lead_week_weights],
        "level_weights": {k: round(v, 3) for k, v in f.level_weights.items()},
        "pseudo_origin_pairs": f.pseudo_pairs,
        "within_day_trust_on_analog_shape": [round(w, 3) for w in f.within_trust],       # lead blocks 1-7, 8-14, 15-21, 22-31; one value repeated
        "within_day_pseudo_origin_triples": f.within_pseudo_pairs,
        "analog_shape_kept": f.analog_kept,
        "trajectory_first_day_kw": np.round(f.trajectory_kw[:24], 2).tolist(),
        "seconds": round(seconds, 2),
    }, indent=1))


if __name__ == "__main__":
    main()
