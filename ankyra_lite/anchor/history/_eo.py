"""Extracted frozen arithmetic; see provenance.json. Do not edit in place."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
CONTEXT, HORIZON, DAYS, CTX_DAYS, WEEKS = 1344, 744, 31, 56, 8
N_TYPES, SUNDAY, HOLIDAY, N_GROUPS = 8, 6, 7, 4
TEMP_CENTER, TEMP_SCALE = 15.0, 10.0
K_RECENT, K_ENVELOPE = 8, 4
SIG_LO, SIG_HI, SIG_KNOTS = -3.0, 3.0, 13


def origin_scale(context: torch.Tensor):
    """l0, s0 from the last 744 context hours; s0 floored so flat windows stay finite."""
    last = context[:, -HORIZON:]
    l0 = last.mean(dim=1)
    s0 = last.std(dim=1, unbiased=False)
    floor = torch.clamp(0.01 * l0.abs(), min=1e-3)
    return l0, torch.maximum(s0, floor), s0 < floor


def gather_types(table: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """table (B,8,...) indexed by per-day types (B,31)."""
    idx = types.long()
    if table.dim() == 2:
        return table.gather(1, idx)
    return table.gather(1, idx[..., None].expand(-1, -1, table.shape[-1]))


class Batch:
    """Tensors for one mini-batch; all load values in kW, temperatures in degC."""

    def __init__(self, context, ctx_temp_day, ctx_types, tgt_types, clim_temp, hist_mean, group, phase,
                 target=None, hourly_cal=None):
        self.context = context            # (B,1344)
        self.ctx_temp_day = ctx_temp_day  # (B,56)
        self.ctx_types = ctx_types        # (B,56)
        self.tgt_types = tgt_types        # (B,31)
        self.clim_temp = clim_temp        # (B,31)
        self.hist_mean = hist_mean        # (B,)
        self.group = group                # (B,)
        self.phase = phase                # (B,2) sin/cos of origin hour
        self.target = target              # (B,744) or None
        self.hourly_cal = hourly_cal      # (B,1344,6) for the A3 arm: temp, tod sin/cos, weekday sin/cos, holiday


def interior_knots(dtype=torch.float64, device=None) -> torch.Tensor:
    return torch.linspace(SIG_LO, SIG_HI, SIG_KNOTS, dtype=dtype, device=device)[1:-1]


def std_temp(t_c: torch.Tensor) -> torch.Tensor:
    return (t_c - TEMP_CENTER) / TEMP_SCALE


def hinge_features(t_std: torch.Tensor) -> torch.Tensor:
    """[t, relu(t - k_1), ..., relu(t - k_11)]; window means of these make window-mean signatures exact."""
    k = interior_knots(t_std.dtype, t_std.device)
    return torch.cat([t_std[..., None], torch.relu(t_std[..., None] - k)], dim=-1)


def expected_hinge_features(t_std: torch.Tensor, sigma_std: torch.Tensor) -> torch.Tensor:
    """Thom (1954): E[relu(t + e - k)] = s phi(u) + (t - k) Phi(u), u = (t - k)/s, e ~ N(0, s^2)."""
    k = interior_knots(t_std.dtype, t_std.device)
    s = sigma_std.reshape(sigma_std.shape + (1,) * (t_std.dim() - sigma_std.dim())).unsqueeze(-1)
    diff = t_std[..., None] - k
    u = diff / s
    phi = torch.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(u / math.sqrt(2)))
    return torch.cat([t_std[..., None], s * phi + diff * Phi], dim=-1)


def deviations(z: torch.Tensor):
    cube = z.view(z.shape[0], CTX_DAYS, 24)
    m = cube.mean(dim=2)
    return m, cube - m[..., None]


def type_tables(m, dev, ctx_types, k=K_RECENT):
    """Day-type level offsets c (all occurrences) and zero-mean deviation profiles D (k most recent)."""
    B = m.shape[0]
    onehot = F.one_hot(ctx_types.long(), N_TYPES).to(m.dtype)
    block = m.view(B, WEEKS, 7).mean(dim=2, keepdim=True).expand(B, WEEKS, 7).reshape(B, CTX_DAYS)
    count = onehot.sum(dim=1)
    c = ((m - block)[..., None] * onehot).sum(dim=1) / count.clamp_min(1.0)
    after = torch.flip(torch.cumsum(torch.flip(onehot, [1]), dim=1), [1])
    recent = onehot * (after <= k).to(m.dtype)
    rc = recent.sum(dim=1)
    D = torch.einsum("btk,bth->bkh", recent, dev) / rc.clamp_min(1.0)[..., None]
    seen = rc > 0
    D = torch.where(seen[..., None], D, dev.mean(dim=1, keepdim=True).expand(B, N_TYPES, 24))
    c = torch.where(seen, c, torch.zeros_like(c))
    borrow = torch.zeros_like(seen)
    borrow[:, HOLIDAY] = ~seen[:, HOLIDAY]
    src = torch.arange(N_TYPES, device=m.device).expand(B, N_TYPES).clone()
    src[:, HOLIDAY] = SUNDAY
    c = torch.where(borrow, c.gather(1, src), c)
    D = torch.where(borrow[..., None], D.gather(1, src[..., None].expand(-1, -1, 24)), D)
    return c, D


def recent_envelope(dev, ctx_types, k=K_ENVELOPE):
    """Hour-wise maximum of the k most recent same-type deviation profiles (B,8,24); fallbacks as type_tables."""
    B = dev.shape[0]
    onehot = F.one_hot(ctx_types.long(), N_TYPES).to(dev.dtype)
    after = torch.flip(torch.cumsum(torch.flip(onehot, [1]), dim=1), [1])
    recent = (onehot * (after <= k).to(dev.dtype)) > 0                              # (B,56,8)
    masked = torch.where(recent[..., None], dev[:, :, None, :], torch.full_like(dev[:, :, None, :], -1e9))
    env = masked.amax(dim=1)                                                        # (B,8,24)
    seen = recent.any(dim=1)
    env = torch.where(seen[..., None], env, dev.amax(dim=1, keepdim=True).expand(B, N_TYPES, 24))
    borrow = torch.zeros_like(seen)
    borrow[:, HOLIDAY] = ~seen[:, HOLIDAY]
    src = torch.arange(N_TYPES, device=dev.device).expand(B, N_TYPES).clone()
    src[:, HOLIDAY] = SUNDAY
    return torch.where(borrow[..., None], env.gather(1, src[..., None].expand(-1, -1, 24)), env)


def z1_parts(batch: Batch, l0, s0, m, dev, eta=0.5):
    c, D = type_tables(m, dev, batch.ctx_types)
    ell_s = m[:, -7:].mean(dim=1)
    ell_l = (batch.hist_mean - l0) / s0
    c_d = gather_types(c, batch.tgt_types)
    level0 = eta * ell_s + (1.0 - eta) * ell_l
    return {"c": c, "D": D, "ell_s": ell_s, "ell_l": ell_l, "c_d": c_d, "level0": level0,
            "L0": level0[:, None] + c_d, "D0": gather_types(D, batch.tgt_types),
            "S": gather_types(recent_envelope(dev, batch.ctx_types), batch.tgt_types)}


def encode(batch: Batch):
    l0, s0, flat = origin_scale(batch.context)
    z = (batch.context - l0[:, None]) / s0[:, None]
    m, dev = deviations(z)
    return l0, s0, flat, z, m, dev
