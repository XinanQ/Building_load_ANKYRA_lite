"""Scoring of one population: the tables of the comparison from forecasts and realised load.

Inputs are plain arrays, one row per forecast window (see ``evaluation/README.md``):

    forecasts   {model name: (n, 744) hourly forecast in kW}; a model that exists only on part of the windows (the
                trained baselines, which exist after their training cutoff) has NaN rows elsewhere
    y           (n, 744) realised load in kW
    unit        (n,) unit label of each window
    month       (n,) label of the month that holds the window's midpoint (the bootstrap resamples these)
    context     (n, 1344) load before the origin; needed for the scaled errors and the peak readout only

``score_population`` returns two blocks with the same content: ``full`` (all windows, the models that cover them all)
and ``late`` (the windows every model covers, all models; the test set of the comparison).  Each block holds

    pairwise        the reference forecaster against every other model: unit-equal log RMS ratio (negative favours the
                    reference), its 95% unit-and-month bootstrap interval, the improvement 100[1 - exp(r)] and the
                    pooled MSE ratio
    mean_unit_rank  mean over units of each model's rank by that unit's RMSE; the reference's own reduced versions
                    (``family``) are left out of the ranking
    conventional    RMSE, MAE, CV(RMSE), NMBE, WAPE per model

The other functions give the remaining tables: ``rank_tests`` (Friedman, Nemenyi, Holm-corrected Wilcoxon),
``scaled_errors`` (RMSSE, MASE), ``energy_error`` (the estimand on the monthly energy error), ``by_forecast_day``
(hourly loss by forecast day), ``energy_by_forecast_day`` (daily and cumulative energy error by forecast day),
``block_shares`` (the exact split of the hourly squared error into level, daily path and within-day blocks) and
``peak_contrasts`` (the monthly peak readout against other peak estimates).

The estimands are those of ``ankyra_lite.anchor.metrics`` (the anchoring layer's copy of ``ankyra.metrics``); this
module only arranges them into the tables.  Every interval uses 2,000 replicates.  The random streams are those of the
study's scoring: a fresh generator seeded ``SEED`` for every hourly contrast, one generator seeded ``ENERGY_SEED`` for
all energy contrasts of a population (models in sorted order), and a fresh generator seeded ``PEAK_SEED`` for every
peak contrast.
"""
from __future__ import annotations

import math

import numpy as np

from ankyra_lite.anchor import blocks as _blocks
from ankyra_lite.anchor import metrics
from ankyra_lite.anchor import readouts as _readouts

REFERENCE = "ankyra-lite"   # default name of the reference forecaster
SEED = 20260935             # hourly contrasts: a fresh generator with this seed for every contrast
ENERGY_SEED = 20260937      # energy contrasts: one generator per population, models in sorted order
PEAK_SEED = 20260939        # peak contrasts: a fresh generator with this seed for every comparator
SEASON = 168                # weekly seasonal-naive lag of the scaled errors
NEAR_ZERO_KW = 1e-6         # units whose mean load is below this are left out of the relative metrics
DAYS, HR, HORIZON = 31, 24, 744


def window_mse(pred, y):
    """Mean squared error of each window, (n,)."""
    return ((np.asarray(pred, dtype=np.float64) - y) ** 2).mean(1)


def _resolved(low, high):
    return "better" if high < 0 else "worse" if low > 0 else "no"


def block(forecasts, y, unit, month, reference=REFERENCE, family=(), seed=SEED, n_boot=metrics.N_BOOT):
    """Pairwise contrasts, mean unit ranks and conventional metrics on one set of windows."""
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit); month = np.asarray(month)
    for name, p in forecasts.items():
        if np.shape(p) != y.shape or not np.isfinite(p).all():
            raise ValueError(f"forecast {name!r} must be finite with the shape of y on these windows")
    mse = {k: window_mse(p, y) for k, p in forecasts.items()}
    excluded = int(sum(abs(float(y[unit == u].mean())) < NEAR_ZERO_KW for u in np.unique(unit)))
    out = {"windows": int(len(y)), "units": int(len(np.unique(unit))), "months": int(len(np.unique(month))), "reference": reference,
           "pairwise": {},
           "conventional": {k: {**metrics.conventional(np.asarray(p, dtype=np.float64), y, unit), "units_excluded_from_ratios": excluded}
                            for k, p in forecasts.items()}}
    for k in forecasts:
        if k == reference:
            continue
        r = metrics.log_ratio(mse[reference], mse[k], unit)
        iv = metrics.um_interval(mse[reference], mse[k], unit, month, np.random.default_rng(seed), n_boot)["UM"]
        out["pairwise"][k] = {"log_ratio": r, "um_low": iv[0], "um_high": iv[1], "improvement_pct": 100 * (1 - math.exp(r)),
                              "pooled_mse_ratio": float(mse[reference].mean() / mse[k].mean()), "resolved": _resolved(*iv)}
    out["mean_unit_rank"] = metrics.mean_unit_rank({k: v for k, v in mse.items() if k not in family}, unit)
    return out


def score_population(forecasts, y, unit, month, reference=REFERENCE, family=(), seed=SEED, n_boot=metrics.N_BOOT):
    """The full-window and late-window blocks of one population (see the module docstring)."""
    y = np.asarray(y, dtype=np.float64)
    covered = {k: np.isfinite(np.asarray(p, dtype=np.float64)).all(1) for k, p in forecasts.items()}
    if not covered[reference].all():
        raise ValueError("the reference forecaster must cover every window")
    everywhere = [k for k, c in covered.items() if c.all()]
    late = np.logical_and.reduce(list(covered.values()))
    out = {"full": block({k: forecasts[k] for k in everywhere}, y, unit, month, reference, family, seed, n_boot)}
    if late.any() and len(everywhere) < len(forecasts):
        out["late"] = block({k: np.asarray(p)[late] for k, p in forecasts.items()}, y[late], np.asarray(unit)[late], np.asarray(month)[late],
                            reference, family, seed, n_boot)
    return out


def late_windows(forecasts):
    """Boolean (n,): the windows on which every model has a forecast (the late subset)."""
    return np.logical_and.reduce([np.isfinite(np.asarray(p, dtype=np.float64)).all(1) for p in forecasts.values()])


def unit_rmse(forecasts, y, unit):
    """{model: per-unit RMSE over the unit's windows}, with the unit labels."""
    y = np.asarray(y, dtype=np.float64)
    uu, inv = np.unique(unit, return_inverse=True); cnt = np.bincount(inv)
    return uu, {k: np.sqrt(np.bincount(inv, window_mse(p, y), len(uu)) / cnt) for k, p in forecasts.items()}


def rank_tests(unit_rmse_by_model, reference=REFERENCE, alpha=0.05):
    """Friedman test, mean ranks, Nemenyi critical difference, and Holm-corrected Wilcoxon signed-rank tests of the
    reference against every other model, with units as blocks (Demsar 2006; Garcia and Herrera 2008).  Per-unit RMSE
    arrays of several populations can be concatenated model by model for a pooled test."""
    from scipy import stats
    models = sorted(unit_rmse_by_model); X = np.stack([unit_rmse_by_model[m] for m in models], 1); n, k = X.shape
    mean_rank = dict(zip(models, stats.rankdata(X, axis=1).mean(0)))
    fr = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    cd = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n)))
    a = X[:, models.index(reference)]; raw = {}
    for j, m in enumerate(models):
        if m != reference:
            raw[m] = float(stats.wilcoxon(a, X[:, j], zero_method="wilcox").pvalue) if np.any(a != X[:, j]) else 1.0
    holm, run = {}, 0.0
    for i, m in enumerate(sorted(raw, key=raw.get)):
        run = max(run, min(1.0, (len(raw) - i) * raw[m])); holm[m] = run
    ref = mean_rank[reference]
    return {"units": int(n), "models": int(k), "friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue), "nemenyi_cd": cd,
            "mean_rank": {m: float(v) for m, v in mean_rank.items()},
            "reference_position": int(1 + sum(v < ref - 1e-12 for v in mean_rank.values())),
            "wilcoxon_holm_p": holm,
            "significantly_better_than": [m for m in holm if holm[m] < alpha and ref < mean_rank[m]],
            "significantly_worse_than": [m for m in holm if holm[m] < alpha and ref > mean_rank[m]],
            "within_nemenyi_cd": [m for m in models if m != reference and abs(mean_rank[m] - ref) <= cd]}


def scaled_errors(forecasts, y, context, unit):
    """RMSSE and MASE per model: errors scaled by the in-sample weekly seasonal-naive error over the context, per unit,
    summarised by the median and the geometric mean over units, and the share of units with RMSSE below 1.  Windows
    with a zero scale are left out."""
    y = np.asarray(y, dtype=np.float64); context = np.asarray(context)        # the context keeps its own precision
    d = context[:, SEASON:] - context[:, :-SEASON]; s2 = (d ** 2).mean(1); s1 = np.abs(d).mean(1); ok = (s2 > 0) & (s1 > 0)
    uu, inv = np.unique(np.asarray(unit)[ok], return_inverse=True); cnt = np.bincount(inv); out = {}
    for k, p in forecasts.items():
        e = (np.asarray(p, dtype=np.float64) - y)[ok]
        rmsse = np.sqrt(np.bincount(inv, (e ** 2).mean(1) / s2[ok], len(uu)) / cnt); mase = np.bincount(inv, np.abs(e).mean(1) / s1[ok], len(uu)) / cnt
        out[k] = {"RMSSE_median_unit": float(np.median(rmsse)), "RMSSE_geo_mean": float(np.exp(np.log(rmsse[rmsse > 0]).mean())),
                  "MASE_median_unit": float(np.median(mase)), "MASE_geo_mean": float(np.exp(np.log(mase[mase > 0]).mean())), "units": int(len(uu)),
                  "units_below_1_share": float((rmsse < 1).mean())}
    return out


# ----------------------------------------------------------------------------- energy
def energy_ape(level_error, y, unit):
    """Median over units of the unit's median absolute percentage error of the window mean (the monthly energy error
    relative to the realised energy), in percent; windows with a realised mean below 1e-6 kW are left out."""
    level_error = np.asarray(level_error, dtype=np.float64); y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit)
    ybar = y.mean(1); ok = ybar >= NEAR_ZERO_KW
    return float(np.median([np.median(100 * np.abs(level_error[(unit == u) & ok]) / ybar[(unit == u) & ok]) for u in np.unique(unit[ok])]))


def energy_error(forecasts, y, unit, month, reference=REFERENCE, seed=ENERGY_SEED, n_boot=metrics.N_BOOT, reference_level=None):
    """The primary estimand applied to the monthly energy error (744 x the error of the window mean) instead of the
    hourly error, for every model other than the reference, with the median energy APE (``energy_ape``) of that model.

    ``reference_level`` (n,), optional: the reference's own level readout in kW, used in place of the mean of its
    trajectory (the two differ only where the nonnegativity projection binds).  One generator seeded ``seed`` serves
    all contrasts, in sorted model order."""
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit); month = np.asarray(month)
    err = {k: (y - np.asarray(p, dtype=np.float64)).mean(1) for k, p in forecasts.items()}       # mean of the hourly errors
    if reference_level is not None:
        err[reference] = y.mean(1) - np.asarray(reference_level, dtype=np.float64)
    a = err[reference] ** 2
    rng = np.random.default_rng(seed); out = {}
    for k in sorted(m for m in forecasts if m != reference):
        r = metrics.log_ratio(a, err[k] ** 2, unit)
        iv = metrics.um_interval(a, err[k] ** 2, unit, month, rng, n_boot)["UM"]
        out[k] = {"log_ratio": r, "um_low": iv[0], "um_high": iv[1], "improvement_pct": 100 * (1 - math.exp(r)), "resolved": _resolved(*iv),
                  "median_energy_ape_pct": energy_ape(err[k], y, unit)}
    return out


# ----------------------------------------------------------------------------- by forecast day
def _unit_mean_load(y, unit, uu):
    return np.array([y[unit == u].mean() for u in uu])


def by_forecast_day(forecasts, y, unit):
    """Hourly loss by forecast day 1-31.

    gm_cv_rmse_pct   for each model the geometric mean, over one fixed set of units, of the day's CV(RMSE) (the unit's
                     RMSE over that day's 24 hours in all its windows, over the unit's mean load).  The unit set is the
                     units with a mean load of at least 1e-6 kW and a nonzero error on every day for every model.
    rmse_kw, mae_kw  the unit mean of the day's RMSE and MAE (all units)
    cv_rmse_pct, nmae_pct
                     the median over units (mean load at least 1e-6 kW) of the day's RMSE and MAE over the unit's mean load
    """
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit); uu = np.unique(unit); days = y.shape[1] // HR
    ybar = _unit_mean_load(y, unit, uu); r, a = {}, {}
    for k, p in forecasts.items():
        e = (y - np.asarray(p, dtype=np.float64)).reshape(len(y), days, HR)
        r[k] = np.stack([np.sqrt((e[unit == u] ** 2).mean(axis=(0, 2))) for u in uu])
        a[k] = np.stack([np.abs(e[unit == u]).mean(axis=(0, 2)) for u in uu])
    ok = np.abs(ybar) >= NEAR_ZERO_KW
    keep = ok & np.logical_and.reduce([(v > 0).all(1) for v in r.values()])
    return {"units_in_set": int(keep.sum()),
            "gm_cv_rmse_pct": {k: np.exp(np.log(100 * v[keep] / ybar[keep, None]).mean(0)) for k, v in r.items()},
            "rmse_kw": {k: v.mean(0) for k, v in r.items()}, "mae_kw": {k: v.mean(0) for k, v in a.items()},
            "cv_rmse_pct": {k: np.median(100 * v[ok] / ybar[ok, None], axis=0) for k, v in r.items()},
            "nmae_pct": {k: np.median(100 * v[ok] / ybar[ok, None], axis=0) for k, v in a.items()}}


def energy_by_forecast_day(forecasts, y, unit):
    """Energy error by forecast day 1-31: for each model the geometric mean over one fixed set of units of the unit's
    RMS daily-energy error (error of the day's mean load) and of its RMS cumulative-energy error (error of the mean over
    days 1..d), each over the unit's mean load, in percent.  The unit set is the units with a mean load of at least
    1e-6 kW and nonzero daily and cumulative errors on every day for every model."""
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit); uu = np.unique(unit); days = y.shape[1] // HR
    ybar = _unit_mean_load(y, unit, uu); stats_ = {}
    for k, p in forecasts.items():
        e = (y - np.asarray(p, dtype=np.float64)).reshape(len(y), days, HR).mean(2)          # daily mean errors (n, 31)
        c = np.cumsum(e, 1) / np.arange(1, days + 1)                                          # error of the mean over days 1..d
        stats_[k] = (np.stack([np.sqrt((e[unit == u] ** 2).mean(0)) for u in uu]), np.stack([np.sqrt((c[unit == u] ** 2).mean(0)) for u in uu]))
    keep = (np.abs(ybar) >= NEAR_ZERO_KW) & np.logical_and.reduce([(d_ > 0).all(1) & (c_ > 0).all(1) for d_, c_ in stats_.values()])
    gm = lambda v: np.exp(np.log(100 * v[keep] / ybar[keep, None]).mean(0))                   # noqa: E731
    return {"units_in_set": int(keep.sum()),
            "gm_cv_daily_energy_pct": {k: gm(d_) for k, (d_, _) in stats_.items()},
            "gm_cv_cumulative_energy_pct": {k: gm(c_) for k, (_, c_) in stats_.items()}}


# ----------------------------------------------------------------------------- error blocks
BLOCKS = ("level", "daily path", "within-day")


def block_shares(forecasts, y, unit, reference=REFERENCE, against=None):
    """The exact split of each model's hourly squared error into the level, daily-path and within-day blocks
    (``ankyra_lite.anchor.blocks.block_losses``; the three add up to the hourly MSE of every window).

    shares[model]     {"pooled": block share of the summed squared error over all windows,
                       "median_unit": median over units (nonzero error) of the unit's block share}
    contrasts[model]  for every model in ``against`` (default: every other model): by block and for the hourly error,
                      the unit-equal log ratio of the reference's per-unit block MSE over the model's (units where both
                      are positive), its improvement 100[1 - exp(r)] and the number of units used
    """
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit)
    uu, inv = np.unique(unit, return_inverse=True); cnt = np.bincount(inv, minlength=len(uu)); per = {}
    for k, p in forecasts.items():
        b = dict(zip(BLOCKS, _blocks.block_losses(np.asarray(p, dtype=np.float64), y)))
        hourly = window_mse(p, y)
        ub = {n: np.bincount(inv, v, len(uu)) / cnt for n, v in b.items()}
        tot = sum(ub.values()); pos = tot > 0
        per[k] = {"unit_blocks": ub, "unit_total": tot, "pooled": {n: float(v.sum() / hourly.sum()) for n, v in b.items()},
                  "median_unit": {n: float(np.median(ub[n][pos] / tot[pos])) for n in BLOCKS}}
    against = [m for m in forecasts if m != reference] if against is None else list(against)
    contrasts = {}
    for other in against:
        contrasts[other] = {}
        for n in BLOCKS + ("hourly",):
            a = per[reference]["unit_total"] if n == "hourly" else per[reference]["unit_blocks"][n]
            o = per[other]["unit_total"] if n == "hourly" else per[other]["unit_blocks"][n]
            ok = (a > 0) & (o > 0)
            r = float(np.mean(0.5 * np.log(a[ok] / o[ok])))
            contrasts[other][n] = {"log_ratio": r, "improvement_pct": 100 * (1 - math.exp(r)), "units_in_ratio": int(ok.sum())}
    return {"windows": int(len(y)), "units": int(len(uu)),
            "shares": {k: {"pooled": v["pooled"], "median_unit": v["median_unit"]} for k, v in per.items()}, "contrasts": contrasts}


# ----------------------------------------------------------------------------- peak
def peak_readout(trajectories, context, context_types, target_types, kappa=1.0):
    """The monthly peak readout of the anchoring layer (``ankyra_lite.anchor.readouts.peak_readout``) applied to any
    forecaster's trajectories: (n,) kW from (n, 744) trajectories, (n, 1344) contexts and the day types of the 56
    context days and the 31 target days (each day's type at its 12th hour)."""
    return _readouts.peak_readout(np.asarray(trajectories, dtype=np.float64), np.nan_to_num(np.asarray(context, dtype=np.float64), nan=0.0),
                                  np.asarray(context_types), np.asarray(target_types), kappa)


def previous_window_peak(context):
    """The observed peak of the last 744 context hours, (n,) kW."""
    return np.asarray(context, dtype=np.float64)[:, -HORIZON:].max(1)


def peak_ape(peak, y, unit):
    """Median over units of the unit's median absolute percentage peak error, in percent (windows with a realised
    peak above 1e-6 kW)."""
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit); ypk = y.max(1); ok = ypk > NEAR_ZERO_KW
    a = np.abs(np.asarray(peak, dtype=np.float64) - ypk)[ok] / ypk[ok]; u = unit[ok]
    return float(100 * np.median([np.median(a[u == x]) for x in np.unique(u)]))


def peak_contrasts(reference_peak, comparators, y, unit, month, seed=PEAK_SEED, n_boot=metrics.N_BOOT):
    """The reference's peak readout against other peak estimates of the same windows.

    reference_peak  (n,) kW, e.g. the stored peak readout of the delivered trajectory
    comparators     {label: (n,) kW}, e.g. the reference's own trajectory maximum, the previous window's peak
                    (``previous_window_peak``), and for each baseline the same readout on its trajectory
                    (``peak_readout``) or its trajectory maximum
    The estimand is the unit-equal log ratio of the squared peak errors against the realised monthly peak, with its
    unit-and-month interval (a fresh generator seeded ``seed`` for every comparator); each entry also carries the
    comparator's median peak APE (``peak_ape``).  Comparators whose log ratio is undefined are left out."""
    y = np.asarray(y, dtype=np.float64); unit = np.asarray(unit); month = np.asarray(month); ypk = y.max(1)
    a = (np.asarray(reference_peak, dtype=np.float64) - ypk) ** 2; out = {}
    with np.errstate(all="ignore"):
        for label, pk in comparators.items():
            b = (np.asarray(pk, dtype=np.float64) - ypk) ** 2
            r = metrics.log_ratio(a, b, unit)
            if not np.isfinite(r):
                continue
            iv = metrics.um_interval(a, b, unit, month, np.random.default_rng(seed), n_boot)["UM"]
            out[label] = {"log_ratio": r, "um_low": iv[0], "um_high": iv[1], "improvement_pct": 100 * (1 - math.exp(r)),
                          "resolved": _resolved(*iv), "median_peak_ape_pct": peak_ape(pk, y, unit)}
    return out
