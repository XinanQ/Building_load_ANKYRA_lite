"""A record of your own unit, prepared for ankyra-lite; runs on artificial data, no data or downloads needed.

    python examples/own_unit.py                          # the shipped checkpoints checkpoints/cambridge/seed*.pt
    python examples/own_unit.py --population households  # the shipped checkpoints of another population
    python examples/own_unit.py --checkpoints DIR        # every seed*.pt in DIR (an ensemble of the seeds)
    python examples/own_unit.py --untrained              # an untrained network of the same shape instead

The artificial meter starts in the middle of a year (18 March 2019, 09:00), has public holidays and three gaps, and
the forecast month starts on 1 October 2020, 00:00.  The steps are those you take with your own meter:

  1. pad the load back to 1 January 00:00 of the first year with NaN, and give the temperature for the padded hours
     too (archive temperature; finite everywhere);
  2. day types from the dates, for the history and the 744 forecast hours: Monday 0 ... Sunday 6, public holiday 7;
  3. the start timestamp is that 1 January 00:00 written with a ``+00:00`` offset (a label: no conversion is made,
     so keep the record on one fixed-offset grid, UTC or local standard time);
  4. gaps: the 1,344 hours before the origin must be complete.  ``fill_short_gaps`` interpolates every gap of at
     most 6 hours that lies inside one 744-hour block counted back from the origin, and leaves longer gaps, gaps that
     touch the origin or a block boundary, and the padding missing.  The anchoring layer applies the same rule itself
     to the copy of the record its estimator reads, but it first requires the context at the origin to be complete
     and never fills it.  Filling the record before the call also hands the interpolated hours to the carrier, which
     otherwise reads them as missing;
  5. the forecast with the carrier, the history-only configuration (``forecast(history, None, ...)``), the 80%
     interval (``readouts.interval``) and the anchoring record.

``cut=None``: the unit's record is not in the carrier's training data, so no pseudo-origin is dropped.  The shipped
checkpoints were trained and evaluated on their own population only (README, "When not to use it"); for a unit of
another population, train a carrier with the recipe (``ankyra_lite.carrier.train``).  Here they show the format, not
typical values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                               # also runs from a checkout without ``pip install -e .``
    sys.path.insert(0, str(REPO))

from ankyra_lite import History, build_carrier, fill_short_gaps, forecast, load_carrier, readouts   # noqa: E402
from ankyra_lite.anchor.synthetic import synthetic_history                                          # noqa: E402

HORIZON, CONTEXT = 744, 1344
FIRST = datetime(2019, 3, 18, 9)            # the meter's first hourly reading
ORIGIN = datetime(2020, 10, 1)              # the first forecast hour
HOLIDAYS = {date(2019, 4, 19), date(2019, 4, 22), date(2019, 5, 1), date(2019, 12, 25), date(2019, 12, 26),
            date(2020, 1, 1), date(2020, 4, 10), date(2020, 4, 13), date(2020, 5, 1), date(2020, 12, 25),
            date(2020, 10, 12)}             # example dates; one lies in the forecast month


def carriers(population, checkpoints=None, untrained=False):
    """(list of carrier networks, description), as in examples/quickstart.py."""
    folder = Path(checkpoints) if checkpoints else REPO / "checkpoints" / population
    paths = [] if untrained else sorted(folder.glob("seed*.pt"))
    if paths:
        return [load_carrier(p) for p in paths], f"{len(paths)} checkpoint(s) in {folder.name}/: " + ", ".join(p.name for p in paths)
    torch.manual_seed(0)
    reason = "--untrained given" if untrained else f"no seed*.pt in {folder.name}/"
    return [build_carrier().eval()], f"untrained stand-in (build_carrier(), seed 0; {reason})"


def meter():
    """The artificial meter: hourly load (kW) and temperature (degC) from FIRST up to the hour before ORIGIN, and the
    archive temperature from 1 January 00:00 up to FIRST.  It stands in for your own data."""
    jan1 = datetime(FIRST.year, 1, 1)
    n = int((ORIGIN - jan1).total_seconds() // 3600)
    art = synthetic_history(n)                              # index 0 is 2019-01-01 00:00, like FIRST.year
    pad = int((FIRST - jan1).total_seconds() // 3600)
    noise = np.random.default_rng(0).normal(0.0, 0.4, n - pad)
    load_meter = np.asarray(art.load_kw, dtype=np.float64)[pad:] + noise
    temp_meter = np.asarray(art.temperature_c, dtype=np.float64)[pad:]
    temp_before_first = np.asarray(art.temperature_c, dtype=np.float64)[:pad]
    at = lambda when: int((when - FIRST).total_seconds() // 3600)        # noqa: E731  (index into the meter arrays)
    load_meter[at(datetime(2019, 7, 10, 13)):at(datetime(2019, 7, 10, 16))] = np.nan    # 3 hours: interpolated
    load_meter[at(datetime(2019, 11, 5, 6)):at(datetime(2019, 11, 6, 12))] = np.nan     # 30 hours: stays missing
    load_meter[at(datetime(2020, 9, 21, 10)):at(datetime(2020, 9, 21, 14))] = np.nan    # 4 hours in the last 1,344
    return load_meter, temp_meter, temp_before_first


def prepare(load_meter, temp_meter, temp_before_first, first=FIRST, holidays=HOLIDAYS):
    """Steps 1-3: the record padded back to 1 January, day types from the dates, the +00:00 anchor."""
    anchor = datetime(first.year, 1, 1)
    pad = int((first - anchor).total_seconds() // 3600)
    load_kw = np.concatenate([np.full(pad, np.nan), load_meter])
    temperature_c = np.concatenate([temp_before_first, temp_meter])      # finite, same length as load_kw
    days = [(anchor + timedelta(hours=h)).date() for h in range(len(load_kw) + HORIZON)]
    day_types = np.array([7 if d in holidays else d.weekday() for d in days], dtype=np.int64)
    return load_kw, temperature_c, day_types, f"{first.year}-01-01T00:00:00+00:00", pad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", default="cambridge", help="checkpoints/<population>/seed*.pt are used when present")
    ap.add_argument("--checkpoints", default=None, help="folder with seed*.pt (overrides --population)")
    ap.add_argument("--untrained", action="store_true", help="use the untrained stand-in even if checkpoints exist")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    models, description = carriers(args.population, args.checkpoints, args.untrained)
    group, sigma, dst = "Office", 0.25, "EU"               # category; temperature-anomaly scale (units of 10 degC); DST rule of the region

    load_kw, temperature_c, day_types, start, pad = prepare(*meter())
    o = len(load_kw)                                       # the origin: hours from 1 January 00:00 of the first year

    # step 4: the context at the origin has a 4-hour gap, so the forecast refuses the record as it is
    try:
        forecast(History(load_kw, temperature_c, day_types, start), models, group=group, temp_sigma_std=sigma, dst_region=dst)
        refused = None
    except ValueError as e:                                # ankyra_lite.InputError is a ValueError
        refused = str(e)
    filled = fill_short_gaps(load_kw)
    history = History(filled, temperature_c, day_types, start)

    # step 5: the forecast (cut=None: the unit is not in the carrier's training data)
    t0 = time.perf_counter()
    f = forecast(history, models, group=group, temp_sigma_std=sigma, dst_region=dst, cut=None)
    seconds = time.perf_counter() - t0
    bands, n_windows = readouts.interval(history, f, group=group, temp_sigma_std=sigma)    # rows: 5, 10, 50, 90, 95 %
    h = forecast(history, None, group=group, temp_sigma_std=sigma, dst_region=dst)         # history-only configuration

    rec = f.anchoring_record
    print(json.dumps({
        "carrier": description,
        "record": {
            "start_timestamp": start,
            "padded_hours": pad,                           # = hour index of the first reading
            "origin_hour_index": o,
            "origin": ORIGIN.isoformat(),
            "missing_hours_after_padding": int(np.isnan(load_kw[pad:]).sum()),
            "missing_hours_after_fill_short_gaps": int(np.isnan(filled[pad:]).sum()),
            "complete_hours_before_origin": int(np.isfinite(filled).sum()),
            "holiday_days_in_forecast_month": int((day_types[o:o + HORIZON:24] == 7).sum()),
        },
        "refused_without_filling": refused,
        "carrier_issue_times_k": sorted(f.carrier),
        "state_window": f.state_window,
        "energy_kwh": round(f.energy_kwh, 3),
        "level_kw": round(f.level_kw, 4),
        "peak_kw": round(f.peak_kw, 3),
        "interval_80pct_first_hour_kw": [round(float(bands[1, 0]), 2), round(float(bands[3, 0]), 2)],
        "interval_pseudo_windows": n_windows,
        "projection_energy_increase_kwh": round(readouts.projection_energy_increase(f), 3),
        "lead_week_weights_on_carrier": [round(a, 3) for a in f.lead_week_weights],
        "pseudo_origin_pairs": f.pseudo_pairs,
        "within_day_trust_on_analog_shape": round(f.within_trust[0], 3),
        "anchoring_record": {"carrier_level_weight": round(rec["carrier_level_weight"], 3),
                             "history_vs_carrier_log_ratio": round(rec["history_vs_carrier_log_ratio"], 3),
                             "pseudo_origins": rec["pseudo_origins"]},
        "history_only": {"energy_kwh": round(h.energy_kwh, 3), "level_kw": round(h.level_kw, 4), "peak_kw": round(h.peak_kw, 3),
                         "state_window": h.state_window, "carrier": sorted(h.carrier)},
        "trajectory_first_day_kw": np.round(f.trajectory_kw[:24], 2).tolist(),
        "seconds": round(seconds, 2),
    }, indent=1))


if __name__ == "__main__":
    main()
