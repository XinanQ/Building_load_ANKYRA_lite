"""End-to-end example of the evaluation on artificial buildings: windows, forecasts, scoring.

    python -m evaluation.run_example                          # the release's Cambridge checkpoints (ankyra_lite.checkpoint_paths)
    python -m evaluation.run_example --population households  # the release's checkpoints of another population
    python -m evaluation.run_example --checkpoints DIR        # every seed*.pt in DIR
    python -m evaluation.run_example --untrained              # an untrained carrier network of the same shape instead

Twelve artificial buildings are forecast at their last three usable month-start origins by ankyra-lite (the carrier
network's forecasts at the origin and the six pseudo-origins, anchored by the anchoring layer, with the seasonal-naive
state rule and, when the release switches it on, the origin-continuity layer) and by three untrained baselines; the
carrier's own forecast at the origin is scored as the reference's reduced version (a pairwise contrast, left out of the
ranking).  The population is then scored as in the study: hourly contrasts and ranks, conventional metrics, rank tests,
scaled errors, energy contrasts, errors by forecast day, error blocks and the peak readout.

Release configuration (``ankyra_lite._release``).  By default the carrier is the release's checkpoints of the
population, ``ankyra_lite.checkpoint_paths(population)``: checkpoints/<population>/<CHECKPOINT_SUBDIR>/seed<k>.pt for k
in CARRIER_SEEDS, averaged as an ensemble at every issue time; ``forecast`` applies the continuation layer when
CONTINUATION_DEFAULT is on.  Readout rule A: the hourly tables, the conventional metrics, the scaled errors, the errors
by forecast day and the error blocks score the delivered trajectory, while the energy tables, the peak readout and the
trajectory-maximum comparator read the trajectory before the continuation layer (``trajectory_before_continuation_kw``),
as ``forecast`` reads level, energy and peak.  With the layer off both are the same array.  With ``--untrained``, or
when not every release checkpoint is present (or ``--checkpoints DIR`` holds no ``seed*.pt``), the carrier is an
untrained network of the published shape (``build_carrier()`` with a fixed seed).  Checkpoints of a real population do not belong to
artificial buildings, so in both cases the numbers mean nothing; the example shows the data flow and the output
format.  Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ankyra_lite import _release, build_carrier, checkpoint_paths, forecast, load_carrier
from ankyra_lite.anchor import History
from ankyra_lite.anchor.synthetic import synthetic_history
from evaluation import baselines, scoring, windows

REPO = Path(__file__).resolve().parents[1]
REFERENCE, REDUCED = scoring.REFERENCE, "carrier alone"
GROUP, SIGMA = "Office", 0.25                    # category of the artificial buildings; temperature-anomaly scale


def carriers(population="cambridge", checkpoints=None, untrained=False):
    """(list of carrier networks, description): with ``checkpoints``, every seed*.pt in that folder; by default the
    release's checkpoints of the population, ``ankyra_lite.checkpoint_paths(population)`` (the seeds
    ``_release.CARRIER_SEEDS`` in ascending order, in the subfolder ``_release.CHECKPOINT_SUBDIR``), used when all of
    them are present; one untrained network (seed 0) with ``untrained`` or when no checkpoint is used."""
    if checkpoints:
        folder = Path(checkpoints)
        paths = sorted(folder.glob("seed*.pt"))
        shown, missing = folder.name, f"no seed*.pt in {folder.name}/"
    else:
        paths = checkpoint_paths(population, REPO / "checkpoints")
        shown = "/".join(x for x in (population, _release.CHECKPOINT_SUBDIR) if x)
        missing = f"the release's checkpoints are not all in {shown}/"
        paths = paths if all(p.is_file() for p in paths) else []
    if paths and not untrained:
        return [load_carrier(p) for p in paths], f"{len(paths)} checkpoint(s) in {shown}/: " + ", ".join(p.name for p in paths)
    torch.manual_seed(0)
    reason = "--untrained given" if untrained else missing
    return [build_carrier().eval()], f"untrained stand-in (build_carrier(), seed 0; {reason})"


def population(n_units=12, hours=16000, seed=0):
    rng = np.random.default_rng(seed); units = []
    for i in range(n_units):
        h = synthetic_history(hours)
        scale = float(rng.uniform(0.5, 3.0)); noise = rng.normal(0.0, 0.05, len(h.load_kw))
        units.append(History(np.maximum(h.load_kw * scale * (1.0 + noise), 0.0), h.temperature_c, h.day_types, h.start_timestamp))
    return units


def run(n_units=12, n_boot=200, models=None, description=""):
    """Forecast and score the artificial population; returns the summary printed by ``main``."""
    models, description = (models, description) if models is not None else carriers()
    t0 = time.perf_counter()
    names = (REFERENCE, REDUCED, "SN-day", "SN-week", "WeekProf4")
    F = {k: [] for k in names}; pre = []; peak, ys, ctxs, cts, tts, unit, month = [], [], [], [], [], [], []
    for i, h in enumerate(population(n_units)):
        usable = h.load_kw[:len(h.load_kw) - windows.HORIZON]                      # keep the last month as a target
        origins = windows.usable_origins(h.load_kw, windows.month_starts(h.start_timestamp, len(usable) + 1))[-3:]
        ctx, y, mo = windows.window_arrays(h.load_kw, origins, h.start_timestamp)
        ct, tt = windows.day_types(h.day_types, origins)
        for o in origins:                                                          # the forecaster sees nothing at or after o
            past = History(h.load_kw[:o], h.temperature_c[:o], h.day_types[:o + windows.HORIZON], h.start_timestamp)
            r = forecast(past, models, group=GROUP, temp_sigma_std=SIGMA)
            F[REFERENCE].append(r.trajectory_kw[None]); F[REDUCED].append(r.carrier[0][None]); peak.append(r.peak_kw)
            pre.append(r.trajectory_before_continuation_kw[None])                  # readout rule A (energy, peak)
        F["SN-day"].append(baselines.seasonal_naive(ctx, 24)); F["SN-week"].append(baselines.seasonal_naive(ctx, 168))
        F["WeekProf4"].append(baselines.week_profile(ctx))
        ys.append(y); ctxs.append(ctx); cts.append(ct); tts.append(tt); unit += [f"building_{i:02d}"] * len(origins); month += list(mo)
    F = {k: np.vstack(v) for k, v in F.items()}; y = np.vstack(ys); ctx = np.vstack(ctxs); ct = np.vstack(cts); tt = np.vstack(tts)
    unit = np.array(unit); month = np.array(month); peak = np.array(peak)
    E = dict(F); E[REFERENCE] = np.vstack(pre)                   # energy readouts: before the continuation layer (rule A)
    t_fc = time.perf_counter() - t0

    res = scoring.score_population(F, y, unit, month, reference=REFERENCE, family=(REDUCED,), n_boot=n_boot)["full"]
    external = {k: v for k, v in F.items() if k != REDUCED}
    tests = scoring.rank_tests(scoring.unit_rmse(external, y, unit)[1], reference=REFERENCE)
    energy = scoring.energy_error(E, y, unit, month, reference=REFERENCE, n_boot=n_boot)
    days = scoring.by_forecast_day(external, y, unit)
    edays = scoring.energy_by_forecast_day({k: v for k, v in E.items() if k != REDUCED}, y, unit)
    blk = scoring.block_shares(external, y, unit, reference=REFERENCE, against=("SN-week",))
    comps = {"trajectory maximum": E[REFERENCE].max(1), "previous-window peak": scoring.previous_window_peak(ctx),
             "same readout on SN-week": scoring.peak_readout(F["SN-week"], ctx, ct, tt)}
    pk = scoring.peak_contrasts(peak, comps, y, unit, month, n_boot=n_boot)
    r2 = lambda v: round(float(v), 2)                                                    # noqa: E731
    return {
        "note": "artificial buildings" + (" and an untrained carrier" if description.startswith("untrained") or not description else "")
                + ": the numbers show the output format only",
        "carrier": description,
        "release": {"version": _release.VERSION, "carrier_recipe": _release.CARRIER_RECIPE,
                    "carrier_seeds": sorted(_release.CARRIER_SEEDS), "continuation_layer": bool(_release.CONTINUATION_DEFAULT),
                    "readout_rule": "energy, level and peak before the continuation layer (readout rule A)"},
        "windows": res["windows"], "units": res["units"], "seconds_forecasting": round(t_fc, 1),
        "ankyra_lite_against": {k: {"improvement_pct": r2(v["improvement_pct"]), "interval_log": [round(v["um_low"], 4), round(v["um_high"], 4)],
                                    "resolved": v["resolved"]} for k, v in res["pairwise"].items()},
        "mean_unit_rank": {k: round(v, 3) for k, v in res["mean_unit_rank"].items()},
        "cv_rmse_pct": {k: r2(v["CV_RMSE_pct"]) for k, v in res["conventional"].items()},
        "friedman_p": tests["friedman_p"], "nemenyi_cd": round(tests["nemenyi_cd"], 3),
        "rmsse_geo_mean": {k: round(v["RMSSE_geo_mean"], 3) for k, v in scoring.scaled_errors(external, y, ctx, unit).items()},
        "energy_improvement_pct": {k: r2(v["improvement_pct"]) for k, v in energy.items()},
        "gm_cv_rmse_pct_day_1_and_31": {k: [r2(v[0]), r2(v[-1])] for k, v in days["gm_cv_rmse_pct"].items()},
        "gm_cv_cumulative_energy_pct_day_31": {k: r2(v[-1]) for k, v in edays["gm_cv_cumulative_energy_pct"].items()},
        "pooled_block_shares_ankyra_lite": {k: round(v, 3) for k, v in blk["shares"][REFERENCE]["pooled"].items()},
        "peak_improvement_pct": {k: r2(v["improvement_pct"]) for k, v in pk.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", default="cambridge", help="the release's checkpoints of this population (ankyra_lite.checkpoint_paths) are used when all are present")
    ap.add_argument("--checkpoints", default=None, help="folder with seed*.pt (overrides --population)")
    ap.add_argument("--untrained", action="store_true", help="use the untrained stand-in even if checkpoints exist")
    ap.add_argument("--units", type=int, default=12, help="number of artificial buildings")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap replicates (the study used 2,000)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    models, description = carriers(args.population, args.checkpoints, args.untrained)
    out = run(args.units, args.n_boot, models, description)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
