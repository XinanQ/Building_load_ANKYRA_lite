"""Evaluation estimands used in the study.

Primary: the unit-equal log RMS ratio r = mean_i log(R_i^A / R_i^B), where R_i is the RMS of unit i's errors over its
windows; reported as the improvement 100 [1 - exp(r)] (positive favours A).  Uncertainty: 2,000 paired bootstrap
replicates resampling units and target-midpoint months independently (UM).  Also: the mean per-unit rank (models ranked
within each unit by RMSE, ties averaged), which stays defined when some units have exactly zero error, and the pooled
MSE ratio, whose relation to the unit-equal estimand is given by ``pooled_ratio_decomposition`` (P19 in theory/PROOFS.md).
"""
from __future__ import annotations

import math

import numpy as np

N_BOOT = 2000


def log_ratio(a, b, unit, wwin=None, wunit=None):
    """Unit-equal log RMS ratio of per-window mean squared errors a over b (negative favours a)."""
    uu, inv = np.unique(unit, return_inverse=True)
    w = np.ones(len(a)) if wwin is None else wwin
    na, nb = np.bincount(inv, w * a, len(uu)), np.bincount(inv, w * b, len(uu))
    da = np.bincount(inv, w, len(uu))
    wu = np.ones(len(uu)) if wunit is None else wunit
    ok = (da > 0) & (na > 0) & (nb > 0) & (wu > 0)
    return float(np.sum(wu[ok] * 0.5 * np.log(na[ok] / nb[ok])) / np.sum(wu[ok]))


def um_interval(a, b, unit, month, rng=None, n_boot=N_BOOT):
    """95 % intervals resampling units (U), months (M) and both independently (UM)."""
    rng = np.random.default_rng(0) if rng is None else rng
    uu = np.unique(unit)
    mm, minv = np.unique(month, return_inverse=True)
    out = {"U": [], "M": [], "UM": []}
    for _ in range(n_boot):
        cu = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu)).astype(float)
        cm = np.bincount(rng.integers(0, len(mm), len(mm)), minlength=len(mm)).astype(float)
        out["U"].append(log_ratio(a, b, unit, None, cu))
        out["M"].append(log_ratio(a, b, unit, cm[minv], None))
        out["UM"].append(log_ratio(a, b, unit, cm[minv], cu))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def contrast(a, b, unit, month, rng=None):
    """Point estimate, improvement in percent and U / M / UM intervals for per-window MSEs a (model A) and b (model B)."""
    r = log_ratio(a, b, unit)
    return {"log_ratio": r, "improvement_pct": 100 * (1 - math.exp(r)), "pooled_ratio": float(np.mean(a) / np.mean(b)),
            **um_interval(a, b, unit, month, rng)}


def pooled_ratio_decomposition(a, b, unit):
    """Why pooled and unit-equal summaries can disagree in sign.

    With n_i windows and reference RMS R_i^b in unit i, the pooled MSE ratio equals sum_i omega_i rho_i^2 with
    omega_i proportional to n_i (R_i^b)^2 and rho_i = R_i^a / R_i^b, whereas the unit-equal estimand is mean_i log rho_i.
    Units with a positive reference error are used; rho_i carries R_i^b in its denominator, so weights and ratios are
    coupled and the sign of a disagreement does not identify which units benefit.
    """
    uu, inv = np.unique(unit, return_inverse=True)
    n = np.bincount(inv, minlength=len(uu)).astype(float)
    ra2, rb2 = np.bincount(inv, a, len(uu)) / n, np.bincount(inv, b, len(uu)) / n
    ok = rb2 > 0
    omega = n[ok] * rb2[ok] / np.sum(n[ok] * rb2[ok])
    rho2 = ra2[ok] / rb2[ok]
    pos = rho2 > 0
    return {"weights": omega, "ratio_sq": rho2, "pooled_ratio": float(omega @ rho2),
            "unit_equal_log_ratio": float(np.mean(0.5 * np.log(rho2[pos])))}


def mean_unit_rank(window_mse: dict, unit) -> dict:
    """Mean over units of each model's rank by that unit's RMSE (ties averaged)."""
    from scipy.stats import rankdata
    uu, inv = np.unique(unit, return_inverse=True)
    names = list(window_mse)
    counts = np.bincount(inv, minlength=len(uu))
    R = np.stack([np.sqrt(np.bincount(inv, window_mse[k], len(uu)) / counts) for k in names], 1)
    rk = np.apply_along_axis(lambda r: rankdata(np.round(r, 12), method="average"), 1, R).mean(0)
    return {k: float(v) for k, v in zip(names, rk)}


def conventional(pred, y, unit):
    """Per-unit RMSE and MAE (unit mean, kW); CV(RMSE), NMBE and WAPE (unit median, %); pooled WAPE.

    pred, y: (n, 744) arrays in kW; unit: (n,) labels.  Units with a mean load below 1e-6 kW in absolute value are
    left out of the three relative metrics."""
    unit = np.asarray(unit)
    e = y - pred
    rows = {"RMSE": [], "MAE": [], "CV(RMSE)": [], "NMBE": [], "WAPE": []}
    for u in np.unique(unit):
        m = unit == u
        eu, yu = e[m].ravel(), y[m].ravel()
        ybar = yu.mean()
        rows["RMSE"].append(math.sqrt(np.mean(eu ** 2)))
        rows["MAE"].append(np.mean(np.abs(eu)))
        if abs(ybar) < 1e-6:
            continue
        rows["CV(RMSE)"].append(100 * rows["RMSE"][-1] / ybar)
        rows["NMBE"].append(100 * eu.sum() / (len(eu) * ybar))
        rows["WAPE"].append(100 * np.abs(eu).sum() / np.abs(yu).sum())
    return {"RMSE_kW": float(np.mean(rows["RMSE"])), "MAE_kW": float(np.mean(rows["MAE"])),
            "CV_RMSE_pct": float(np.median(rows["CV(RMSE)"])), "NMBE_pct": float(np.median(rows["NMBE"])),
            "WAPE_pct": float(np.median(rows["WAPE"])), "WAPE_pooled_pct": float(100 * np.abs(e).sum() / np.abs(y).sum())}
