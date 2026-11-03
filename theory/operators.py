"""Exact properties of the blocks, the historical weights and the readouts, written as operators.

Each function states one property that the anchoring layer's design relies on.  ``PROOFS.md`` in this folder numbers
them P1-P20 and gives the proofs, and ``test_operators.py`` checks every one numerically, including the counterexamples
that limit what is claimed.  The operators do not depend on the carrier: they apply to any forecaster's 744-hour
trajectory and to the historical estimator.  The anchoring layer is ``ankyra_lite.anchor``, a byte-identical copy of
the ANKYRA 2.2.1 package, so this module is the ANKYRA 2.2.1 ``theory/operators.py`` with the import and two docstrings
adapted; the functions are unchanged.  It is kept apart from the forecaster (``ankyra_lite/``), which does not import
it.  The exact properties of the carrier network (C1-C10) are checked in ``test_carrier_properties.py``.

    Block geometry         four_block_losses, four_block_projections, daily_harmonic_frequencies, has_fourier_bin
    Replacement calculus   replace_blocks, mse_change_by_block, level_share, mse_ratio_from_blocks,
                           division_log_ratio, blockwise_selection, correction_accounting
    Support and shrinkage  completed_windows, effective_pseudo_origins, data_share, annual_support_hours,
                           mixture_mass_bounds, weather_weight_bounds, prediction_movement_bound,
                           retained_data_share, two_source_weights
    Feasibility            clip_nonnegative, project_to_mean, project_daily_means, daily_projection,
                           daily_projection_condition
    Peak                   peak_error_decomposition, envelope_is_median, max_lower_median,
                           historical_max_probability, asymmetric_peak_cost, pinball
"""
from __future__ import annotations

import numpy as np

from ankyra_lite.anchor import blocks

CONTEXT, STEP, HORIZON, K_MAX, K0_LEVEL, MIN_ERRORS = 1344, 744, 744, 12, 8, 2
ANNUAL_LAG = 8760
D, HR = blocks.D, blocks.HR


# ============================================================================= block geometry (P1-P3)
def four_block_losses(forecast, truth):
    """(level, between-day, hour-of-day mean, day-by-hour interaction) squared-error blocks; ranks 1, 30, 23, 690.

    The last two add up to the within-day block of ``blocks.block_losses``; all four add up to the hourly MSE.
    """
    e = np.asarray(forecast, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    c = e.reshape(e.shape[:-1] + (D, HR))
    lv = c.mean(axis=(-2, -1))
    dm = c.mean(axis=-1)
    w = c - dm[..., None]
    wbar = w.mean(axis=-2)                                          # hour-of-day mean of the within-day error
    u = w - wbar[..., None, :]
    return lv ** 2, ((dm - lv[..., None]) ** 2).mean(-1), (wbar ** 2).mean(-1), (u ** 2).mean(axis=(-2, -1))


def four_block_projections():
    """The four projections J_D(x)J_H, (I-J_D)(x)J_H, J_D(x)(I-J_H), (I-J_D)(x)(I-J_H) on R^744 (day index outermost)."""
    JD, JH = np.full((D, D), 1.0 / D), np.full((HR, HR), 1.0 / HR)
    ID, IH = np.eye(D), np.eye(HR)
    return np.kron(JD, JH), np.kron(ID - JD, JH), np.kron(JD, IH - JH), np.kron(ID - JD, IH - JH)


def daily_harmonic_frequencies(horizon=HORIZON, period=HR):
    """DFT indices f of the 744-hour window that are 24-hour periodic (31 | f); they span J_D (x) I_H."""
    return np.flatnonzero(np.arange(horizon) % (horizon // period) == 0)


def has_fourier_bin(period_hours, horizon=HORIZON):
    """Whether a sinusoid of this period is a single DFT bin of the window (the weekly period is not: 7 and 31 are coprime)."""
    return (horizon / period_hours).is_integer()


# ============================================================================= replacement calculus (P4-P6)
def replace_blocks(base, level=None, daily_path=None):
    """Replace the level and/or the centred daily path of a base trajectory (744,) or (N, 744); keeps its within-day block."""
    base = np.asarray(base, dtype=np.float64)
    lv = blocks.level(base) if level is None else np.asarray(level, dtype=np.float64)
    bp = blocks.daily_path(base) if daily_path is None else np.asarray(daily_path, dtype=np.float64)
    if daily_path is not None and np.max(np.abs(np.sum(bp, axis=-1))) > 1e-8 * max(1.0, float(np.max(np.abs(bp)))):
        raise ValueError("the daily path must be centred (sum to zero over the 31 days)")
    return blocks.compose(lv, bp, blocks.within_day(base))


def mse_change_by_block(base, new, truth):
    """Exact change of hourly MSE from `base` to `new`, split into (level, daily path, within-day); the parts add up."""
    b0, b1 = blocks.block_losses(base, truth), blocks.block_losses(new, truth)
    return tuple(x1 - x0 for x0, x1 in zip(b0, b1))


def level_share(base, truth):
    """Share of the hourly MSE in the level block: the largest relative reduction any level-only replacement can give."""
    lv, dp, wd = blocks.block_losses(base, truth)
    total = lv + dp + wd
    return np.where(total > 0, lv / np.where(total > 0, total, 1.0), 0.0)


def mse_ratio_from_blocks(shares, block_ratios):
    """MSE_new / MSE_old = 1 + sum_k pi_k (rho_k^2 - 1) for block shares pi_k of the old MSE and block RMS ratios rho_k."""
    pi, rho = np.asarray(shares, dtype=np.float64), np.asarray(block_ratios, dtype=np.float64)
    return 1.0 + np.sum(pi * (rho ** 2 - 1.0), axis=-1)


def division_log_ratio(share, ratio):
    """Log RMS ratio of a division to the source it modifies: 0.5 log(1 + pi (r - 1)).

    pi is the modified source's share of its own MSE in the replaced blocks, r the ratio of the donor's squared error
    to the source's in those blocks.  Gains are bounded below by 0.5 log(1 - pi); losses are unbounded in r.
    """
    pi, r = np.asarray(share, dtype=np.float64), np.asarray(ratio, dtype=np.float64)
    return 0.5 * np.log1p(pi * (r - 1.0))


def blockwise_selection(pooled_losses_a, pooled_losses_b):
    """Choose each block from the source with the smaller pooled block loss; returns (choice per block, pooled MSE).

    Because pooled block losses add, this is the minimum pooled MSE over all 2^3 block-wise divisions of two sources.
    It is not the optimum of a unit-equal log estimand.
    """
    a, b = np.asarray(pooled_losses_a, dtype=np.float64), np.asarray(pooled_losses_b, dtype=np.float64)
    choice = (b < a).astype(int)
    return choice, float(np.where(choice == 1, b, a).sum())


def correction_accounting(reference, corrected, truth):
    """MSE change of a correction D = corrected - reference: returns (size term E[D^2], alignment term E[eD], change)."""
    ref, cor, y = (np.asarray(x, dtype=np.float64) for x in (reference, corrected, truth))
    e, D_ = y - ref, cor - ref
    size, align = float(np.mean(D_ ** 2)), float(np.mean(e * D_))
    return size, align, size - 2 * align


# ============================================================================= support and shrinkage (P7-P10)
def completed_windows(observed, origin, *, context=CONTEXT, step=STEP, k_max=K_MAX, lag=0):
    """Pseudo-origins o - j*step (j = 1..k_max) whose context, target and optional lag window are fully observed.

    With a complete record this equals min{k_max, floor((origin - max(context, lag)) / step)}_+; marking hours as
    missing can only lower it.
    """
    obs = np.asarray(observed, dtype=bool)
    n = 0
    for j in range(1, k_max + 1):
        oj = origin - j * step
        spans = [(oj - context, oj + step)] + ([(oj - lag, oj - lag + step)] if lag else [])
        if all(s >= 0 and e <= len(obs) and obs[s:e].all() for s, e in spans):
            n += 1
    return n


def effective_pseudo_origins(history_hours, k_max=K_MAX, context=CONTEXT, step=STEP):
    """Pseudo-origins with a complete context inside a complete record of `history_hours` hours."""
    return int(max(0, min(k_max, (int(history_hours) - context) // step)))


def data_share(k_eff, k0=K0_LEVEL, min_errors=MIN_ERRORS):
    """Nominal weight of the error evidence against the equal prior; zero below two errors (equal-weight fallback)."""
    return k_eff / (k_eff + k0) if k_eff >= min_errors else 0.0


def annual_support_hours(n_errors=2, lag=ANNUAL_LAG, step=STEP):
    """History needed before the annual candidate has n completed pseudo-origin errors (complete records)."""
    return lag + n_errors * step


def mixture_mass_bounds(prior_mass, rho):
    """For w = (1 - rho) p + rho q and any candidate subset B: (1 - rho) p(B) <= w(B) <= (1 - rho) p(B) + rho."""
    return (1 - rho) * prior_mass, (1 - rho) * prior_mass + rho


def weather_weight_bounds(k, k0=K0_LEVEL):
    """Bounds on the total weather weight for four non-annual candidates with matched support k >= 2."""
    if k < MIN_ERRORS:
        return 0.5, 0.5                          # the specified fallback gives equal weights
    return mixture_mass_bounds(0.5, k / (k + k0))


def prediction_movement_bound(values, rho, q, p):
    """|h.(w - p)| <= rho * range(h) * TV(q, p): how far shrunk weights can move a combined prediction from the prior."""
    h = np.asarray(values, dtype=np.float64)
    tv = 0.5 * np.abs(np.asarray(q, dtype=np.float64) - np.asarray(p, dtype=np.float64)).sum()
    return rho * (h.max() - h.min()) * tv


def retained_data_share(rho, prior_mass, error_mass):
    """Effective data share after deleting candidates at the origin (retained prior mass P, error-weight mass Q)."""
    den = (1 - rho) * prior_mass + rho * error_mass
    if den <= 0:
        raise ValueError("no retained mass")
    return rho * error_mass / den


def two_source_weights(truth, a, b):
    """Weight on source b: least squares <y - a, b - a> / ||b - a||^2 (clipped to [0, 1]) and inverse MSE.

    With a shared error component the inverse-MSE weight is pulled towards 1/2; the least-squares weight depends only
    on the part in which the two sources differ.  The anchoring layer's weekly handover uses the least-squares form.
    """
    y, a, b = (np.asarray(x, dtype=np.float64) for x in (truth, a, b))
    d = b - a
    ls = float(np.clip(np.sum((y - a) * d) / np.sum(d ** 2), 0.0, 1.0)) if np.sum(d ** 2) > 0 else 0.5
    ma, mb = np.mean((y - a) ** 2), np.mean((y - b) ** 2)
    return ls, float(ma / (ma + mb)) if ma + mb > 0 else 0.5


# ============================================================================= feasibility (P11-P13)
def clip_nonnegative(x):
    """Euclidean projection onto nonnegative load; for nonnegative truth it cannot increase any hour's absolute error."""
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def _project_to_sum(v, total):
    """Euclidean projection of v (last axis) onto {z >= 0, sum z = total}, total >= 0."""
    v = np.asarray(v, dtype=np.float64)
    t = np.broadcast_to(np.asarray(total, dtype=np.float64), v.shape[:-1])
    if (t < 0).any():
        raise ValueError("the target sum must be nonnegative")
    u = -np.sort(-v, axis=-1)
    css = np.cumsum(u, axis=-1) - t[..., None]
    j = np.arange(1, v.shape[-1] + 1)
    k = np.sum(u - css / j > 0, axis=-1)                       # number of positive entries at the solution
    theta = np.take_along_axis(css, np.maximum(k - 1, 0)[..., None], axis=-1)[..., 0] / np.maximum(k, 1)
    return np.maximum(v - theta[..., None], 0.0)


def project_to_mean(x, mean=None):
    """Projection onto {z >= 0, mean z = mean} (default: the mean of x).  It keeps energy but has no error guarantee."""
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(-1) if mean is None else np.asarray(mean, dtype=np.float64)
    return _project_to_sum(x, m * x.shape[-1])


def project_daily_means(daily_means):
    """Day-level projection (P13): project the 31 daily means onto {x >= 0 : mean x = max(level, 0)}."""
    m = np.asarray(daily_means, dtype=np.float64)
    return _project_to_sum(m, np.maximum(m.mean(-1), 0.0) * m.shape[-1])


def daily_projection(trajectory):
    """Trajectory delivered under the day-level projection: projected daily means plus the unchanged within-day
    deviations, then max(., 0)."""
    x = np.asarray(trajectory, dtype=np.float64)
    m = blocks.daily_means(x)
    return clip_nonnegative(x + np.repeat(project_daily_means(m) - m, HR, axis=-1))


def daily_projection_condition(forecast_level, true_daily_means):
    """Sufficient condition of the day-level guarantee: level >= 0 and the level is not under-forecast by more than the
    smallest true daily mean (true level - forecast level <= min_d true daily mean)."""
    m = np.asarray(true_daily_means, dtype=np.float64)
    lv = np.asarray(forecast_level, dtype=np.float64)
    return (lv >= 0) & (m.mean(-1) - lv <= m.min(-1))


# ============================================================================= peak (P14-P18)
def peak_error_decomposition(forecast, truth, excursions, kappa=1.0):
    """Split U - M(y) into level, between-day, amplitude and selection terms (selection >= 0).

    forecast, truth: (744,) kW; excursions: (31,) A_{tau_d} for the target days (readouts.historical_excursions).
    """
    F, y = np.asarray(forecast, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    A = np.asarray(excursions, dtype=np.float64)
    L, my = blocks.daily_means(F), blocks.daily_means(y)
    ay = (y.reshape(D, HR) - my[:, None]).max(1)                  # realised excursion above each day's mean
    d_hat, d_star = int(np.argmax(L + A)), int(np.argmax(my + ay))
    U, M = float((L + A)[d_hat]), float(y.max())
    out = {"level": float(F.mean() - y.mean()),
           "between_day": float((L[d_star] - F.mean()) - (my[d_star] - y.mean())),
           "amplitude": float(A[d_star] - ay[d_star]),
           "selection": float((L + A)[d_hat] - (L + A)[d_star]),
           "U": U, "observed_peak": M, "U_minus_peak": U - M, "peak_day": d_star, "selected_day": d_hat}
    out["kappa_term"] = float((kappa - 1.0) * (U - F.mean()))
    return out


def envelope_is_median(m, n):
    """Sufficient condition ((m - 1)/m)^n < 1/2 for the envelope to be the median of the future maximum (m samples,
    n future days of the type that attains it, constant levels within type, independent draws)."""
    return ((m - 1) / m) ** n < 0.5


def max_lower_median(levels, samples):
    """Exact lower median of max_d (L_d + X_d) with independent X_d uniform on the empirical sample of day d."""
    L = np.asarray(levels, dtype=np.float64)
    S = [np.asarray(s, dtype=np.float64) for s in samples]
    cand = np.unique(np.concatenate([l + s for l, s in zip(L, S)]))
    cdf = np.array([np.prod([np.mean(l + s <= x) for l, s in zip(L, S)]) for x in cand])
    return float(cand[np.argmax(cdf >= 0.5 - 1e-15)])


def historical_max_probability(m, n):
    """Under continuous iid sampling, the probability that the joint maximum of m historical and n future values is historical."""
    return m / (m + n)


def asymmetric_peak_cost(pred_peak, true_peak, under_cost):
    """Normalised cost with `under_cost` per kW of under-forecast and 1 per kW of over-forecast (equals MAE at 1)."""
    p, y = np.asarray(pred_peak, dtype=np.float64), np.asarray(true_peak, dtype=np.float64)
    u, o = np.maximum(y - p, 0).mean(), np.maximum(p - y, 0).mean()
    return 2 * (under_cost * u + o) / (1 + under_cost)


def pinball(pred, truth, tau):
    """Mean pinball (quantile) loss at level tau."""
    d = np.asarray(truth, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


# ----------------------------------------------------------------------------- P20 evidence bound
def aggregation_rate(K, T):
    """Order psi(K, T) = min(K/T, sqrt(log K / T)) of the minimax rate of convex aggregation of K candidates from T samples
    (a simplified form of the rate in Tsybakov 2003; constants unspecified)."""
    return float(min(K / T, np.sqrt(np.log(K) / T)))


def selection_rate(K, T):
    """Minimax rate log(K)/T of model-selection aggregation (choosing one of K candidates from T samples)."""
    return float(np.log(K) / T)


def exchangeable_hit_rate(K, T, draws, rng):
    """Monte Carlo check of the null benchmark: with exchangeable candidate errors, the candidate with the smallest pseudo RMS
    over T pseudo-origins is the target's best with probability 1/K.  Returns the simulated hit rate."""
    e = rng.standard_normal((draws, T + 1, K))
    pick = np.argmin(np.sqrt((e[:, :T] ** 2).mean(1)), axis=1); best = np.argmin(np.abs(e[:, T]), axis=1)
    return float(np.mean(pick == best))
