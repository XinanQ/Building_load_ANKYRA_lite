"""Carrier forecasts at a forecast origin and at the pseudo-origins the anchoring layer reads.

The anchoring layer needs the carrier's 744-hour forecast issued at the origin o (k = 0) and at the pseudo-origins
q = o - 744 k, k = 1..6, whose 1,344-hour context is completely observed (``anchor.pseudo_origin_contexts``, with its
default short-gap rule). The carrier forecast at q uses features built from the data before q only. When the carrier's
training cut is given, a pseudo-origin before the cut is never used (its target month overlaps the training period);
the origin itself must not lie before the cut.

An ensemble (a sequence of carriers) is the plain average of the members' forecasts, accumulated in float64 in member
order.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .features import features
from .model import INPUT_KEYS

__all__ = ["carrier_forecasts", "forecast_windows", "pseudo_origins", "batch_inputs", "STEP", "K", "HORIZON", "CONTEXT"]

STEP, K, HORIZON, CONTEXT = 744, 6, 744, 1344
BATCH = 64


def _anchor():
    from .. import anchor            # the package's copy of the anchoring layer (public API only)
    return anchor


def _members(model_or_models) -> list:
    if isinstance(model_or_models, torch.nn.Module):
        return [model_or_models]
    members = list(model_or_models)
    if not members or not all(isinstance(m, torch.nn.Module) for m in members):
        raise TypeError("model_or_models must be a carrier network or a nonempty sequence of carrier networks")
    return members


def _row(a, unit):
    a = np.asarray(a)
    return a[int(unit)] if a.ndim == 2 else a


def pseudo_origins(load, unit: int, origin: int, cut: int | None = None) -> list[int]:
    """Sorted k (0 = the origin) whose context the anchoring layer reads, restricted to q = origin - 744 k >= cut."""
    o = int(origin)
    if cut is not None and o < int(cut):
        raise ValueError(f"the origin {o} lies before the carrier's training cut {int(cut)}")
    pre = np.asarray(_row(load, unit)[:o], dtype=np.float64)
    ks = sorted(_anchor().pseudo_origin_contexts(pre).keys())
    if 0 not in ks:
        raise ValueError("the final 1344 load hours before the origin must be completely observed")
    if cut is not None:
        ks = [k for k in ks if o - k * STEP >= int(cut)]
    return ks


def batch_inputs(feature_dicts: Sequence[Mapping], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Stack feature dictionaries into the six network inputs (float16 tokens are widened to float32)."""
    out = {}
    for k in INPUT_KEYS:
        v = np.stack([f[k] for f in feature_dicts])
        out[k] = torch.from_numpy(v.astype(np.float32) if v.dtype == np.float16 else v).to(device)
    return out


def _run(members, jobs_features, device, batch_size) -> np.ndarray:
    """(N, 744) float64: the ensemble mean of the members' forecasts for each feature dictionary."""
    n = len(jobs_features)
    total = None
    for m in members:
        m.eval()
        out = np.empty((n, HORIZON), np.float64)
        with torch.inference_mode():
            for s in range(0, n, batch_size):
                p = m(batch_inputs(jobs_features[s:s + batch_size], device))["trajectory_kw"].double().cpu().numpy()
                if not (np.isfinite(p).all() and (p >= 0).all()):
                    raise FloatingPointError("the carrier returned a non-finite or negative forecast")
                out[s:s + batch_size] = p
        total = out if total is None else total + out
    return total if len(members) == 1 else total / len(members)


def carrier_forecasts(model_or_models, load, temp, day_types, groups_or_group, start_iso: str, unit: int, origin: int,
                      cut: int | None = None, device: str | torch.device = "cpu",
                      batch_size: int = BATCH) -> dict[int, np.ndarray]:
    """{k: (744,) float64 carrier forecast issued at origin - 744 k} for every k the anchoring layer reads.

    The arrays follow :func:`ankyra_lite.carrier.features.features`; only data before each issue time is read (and
    the day types through its forecast month). ``model_or_models`` is one carrier or a sequence (ensemble average).
    """
    members = _members(model_or_models)
    ks = pseudo_origins(load, unit, origin, cut)
    fs = [features(load, temp, day_types, groups_or_group, start_iso, unit, int(origin) - k * STEP) for k in ks]
    pred = _run(members, fs, device, batch_size)
    return {k: pred[j] for j, k in enumerate(ks)}


def forecast_windows(model_or_models, load, temp, day_types, groups, start_iso: str, units: Iterable[int],
                     origins: Iterable[int], cut: int | None = None, device: str | torch.device = "cpu",
                     batch_size: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """Carrier forecasts for many windows at once: pred (n, 7, 744) float64 (NaN where absent) and have (n, 7) bool.

    Row i holds :func:`carrier_forecasts` for (units[i], origins[i]); the pseudo-origin forecasts of all windows are
    run through the network in batches of ``batch_size``.
    """
    members = _members(model_or_models)
    units, origins = [int(b) for b in units], [int(o) for o in origins]
    if len(units) != len(origins):
        raise ValueError("units and origins must have the same length")
    n = len(units)
    pred = np.full((n, K + 1, HORIZON), np.nan, np.float64)
    have = np.zeros((n, K + 1), bool)
    jobs = [(i, k, b, o - k * STEP) for i, (b, o) in enumerate(zip(units, origins)) for k in pseudo_origins(load, b, o, cut)]
    for s in range(0, len(jobs), batch_size):
        part = jobs[s:s + batch_size]
        fs = [features(load, temp, day_types, groups, start_iso, b, q) for _, _, b, q in part]
        p = _run(members, fs, device, batch_size)
        for (i, k, _, _), row in zip(part, p):
            pred[i, k] = row
            have[i, k] = True
    return pred, have
