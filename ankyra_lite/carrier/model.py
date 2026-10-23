"""The small carrier network: a structured load network with a copy-pointer decoder (61,613 parameters).

The network forecasts the 744 hours (31 days x 24 hours) after a forecast origin from the unit's own pre-origin history
and the known calendar. It has no pretrained weights and no unit-identifier embeddings.

Structure
  * an ordered, masked encoder over 366 daily tokens and 24 monthly tokens (two shared gated causal blocks, reused with
    a dilation schedule), and one attention read-out from 31 calendar-conditioned day queries plus one global query;
  * a positive energy-budget decoder: a monthly level (scale x softplus), day fractions (softmax over 31 days) and
    hour fractions (softmax over 24 hours), so the forecast is nonnegative and its mean equals the level exactly in
    real arithmetic;
  * a copy pointer: every forecast day attends over complete days within the unit's last 56 day positions (keys from the encoder's day
    memory, queries from the forecast-day embedding, plus one learned bonus for a matching day type) and adds the
    attended within-day share profile to the hourly logits and the attended relative daily mean to the daily logits,
    through learned gates that start at zero (so an untrained pointer equals the base decoder).

Arithmetic and parameter names are identical to the research code this package was extracted from, so its
checkpoints load with ``load_state_dict(strict=True)``. Only the configuration used by the carrier (variant "full",
no calendar month head) is implemented.

Inputs (one batch mapping, or six tensors in this order):
  day_tokens     (B, 366, 69)  oldest to newest; features 0-23 arcsinh(load/scale), 24-47 hourly observation mask,
                               52-59 day type one-hot (Monday..Sunday, holiday)
  day_mask       (B, 366) bool usable daily token
  month_tokens   (B, 24, 16)   monthly summaries older than the daily span
  month_mask     (B, 24) bool  usable monthly token
  future_tokens  (B, 31, 16)   calendar of the forecast days, known at the origin; features 0-7 the day type one-hot
  scale          (B,)          finite, strictly positive causal kW scale
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi, sqrt
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["ModelConfig", "StructuredLoadNetwork", "CopyPointerNetwork", "build_carrier", "load_carrier", "forecast_loss",
           "decode_budget", "decompose_trajectory", "zero_mean_basis", "parameter_count", "param_groups", "INPUT_KEYS",
           "CARRIER_PARAMETERS"]

INPUT_KEYS = ("day_tokens", "day_mask", "month_tokens", "month_mask", "future_tokens", "scale")
CARRIER_PARAMETERS = 61613          # trainable parameters of build_carrier()


@dataclass(frozen=True)
class ModelConfig:
    """Network configuration. The field set and validation equal the research configuration, so stored configuration
    dictionaries remain readable; the networks of this package accept only ``variant="full"``."""
    day_features: int = 69
    month_features: int = 16
    future_features: int = 16
    width: int = 48
    rank: int = 8
    variant: str = "full"
    recent_days: int = 56
    day_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    month_dilations: tuple[int, ...] = (1, 2, 4, 8)
    heads: int = 4
    calendar_month_head: bool = False
    dropout: float = 0.0
    dictionary_delta: float = 0.1
    days: int = 31
    hours: int = 24

    def __post_init__(self):
        if self.variant not in {"full", "short_history", "static_dictionary", "direct_hour"}:
            raise ValueError("unknown model variant")
        if min(self.day_features, self.month_features, self.future_features, self.width, self.recent_days) < 1:
            raise ValueError("feature counts, width and recent_days must be positive")
        if self.days != 31 or self.hours != 24:
            raise ValueError("this network uses the fixed 31 x 24 task")
        if not 1 <= self.rank < self.hours:
            raise ValueError("rank must lie between 1 and 23")
        if self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1 or self.dictionary_delta <= 0:
            raise ValueError("invalid dropout or dictionary_delta")
        if not self.day_dilations or not self.month_dilations or min(*self.day_dilations, *self.month_dilations) < 1:
            raise ValueError("positive dilation schedules are required")

    def to_dict(self):
        return asdict(self)


def parameter_count(model: nn.Module) -> int:
    """Registered trainable parameters; every registered layer is on the forward path."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def forecast_loss(output: Mapping, target: Tensor, scale: Tensor, observed: Tensor | None = None,
                  kl_weight: float = 0.1, compression: bool = False) -> dict[str, Tensor]:
    """Dimensionless trajectory MSE plus a masked generalized Kullback-Leibler term.

    Each window has equal weight after averaging its observed hours. ``scale`` is the causal kW scale of each window
    (finite, positive). Targets may be NaN only where ``observed`` is False; missing hours are never zero targets.
    The predicted log intensities come from the budget decoder, so zero-load days or months need no division of their
    totals into proportions. With ``compression`` the per-window loss is log1p-compressed before averaging (the
    carrier's training objective uses ``kl_weight=0.1, compression=True``).
    """
    prediction = output["trajectory_kw"]
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 744:
        raise ValueError("target and prediction must have shape (B, 744)")
    if target.device != prediction.device or not target.is_floating_point():
        raise ValueError("target must be floating point on the prediction device")
    if observed is None:
        observed = torch.ones_like(target, dtype=torch.bool)
    if observed.shape != target.shape or observed.dtype != torch.bool or observed.device != target.device:
        raise ValueError("observed must be a Boolean target mask")
    if not torch.isfinite(target[observed]).all() or not (target[observed] >= 0).all():
        raise ValueError("observed targets must be finite and nonnegative")
    if scale.shape not in {(len(target),), (len(target), 1)} or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("loss scale must be finite, positive and match the batch")
    if scale.device != target.device or not observed.any(-1).all():
        raise ValueError("each training window needs observed targets and a scale on its device")
    if not 0 <= kl_weight < float("inf"):
        raise ValueError("kl_weight must be finite and nonnegative")
    s = scale.reshape(-1, 1).to(prediction.dtype)
    y = torch.where(observed, target, 0.0).to(prediction.dtype) / s
    f = prediction / s
    log_f = output["log_trajectory_normalized"] + output["scale_kw"].reshape(-1, 1).log() - s.log()
    tiny = torch.finfo(y.dtype).tiny
    y_log_y = torch.where(y > 0, y * y.clamp_min(tiny).log(), 0.0)
    mask = observed.to(prediction.dtype)
    counts = mask.sum(-1)
    mse_i = ((f - y).square() * mask).sum(-1) / counts
    mse = mse_i.mean()
    kl_i = torch.where(observed, y_log_y - y * log_f - y + f, 0.0).sum(-1) / counts
    kl = kl_i.mean()
    raw = mse_i + kl_weight * kl_i
    loss = torch.log1p(raw.clamp_min(0)).mean() if compression else mse + kl_weight * kl
    return {"loss": loss, "mse": mse, "generalized_kl": kl, "per_window": raw,
            "negative_rounding_count": (raw < 0).sum()}


def zero_mean_basis(hours: int = 24) -> Tensor:
    """Orthonormal Helmert coordinates of the hourly zero-mean subspace (all 23 zero-mean directions)."""
    basis = torch.zeros(hours, hours - 1)
    for j in range(hours - 1):
        c = 1.0 / sqrt((j + 1) * (j + 2))
        basis[:j + 1, j] = c
        basis[j + 1, j] = -(j + 1) * c
    return basis


def decompose_trajectory(trajectory: Tensor) -> dict[str, Tensor]:
    """The orthogonal level / daily path / within-day decomposition of a (B, 744) or (B, 31, 24) trajectory."""
    if trajectory.shape[-1] == 744 and trajectory.ndim == 2:
        cube = trajectory.reshape(-1, 31, 24)
    elif trajectory.ndim == 3 and trajectory.shape[-2:] == (31, 24):
        cube = trajectory
    else:
        raise ValueError("trajectory must have shape (B, 744) or (B, 31, 24)")
    daily = cube.mean(-1)
    level = daily.mean(-1)
    return {"level_kw": level, "daily_means_kw": daily,
            "daily_path_kw": daily - level[:, None],
            "within_day_kw": cube - daily[:, :, None]}


def decode_budget(level_kw: Tensor, day_logits: Tensor, hour_logits: Tensor) -> dict[str, Tensor]:
    """Positive allocation of a fixed predicted monthly mean over days (softmax) and hours (softmax)."""
    if day_logits.ndim != 2 or day_logits.shape[1] != 31:
        raise ValueError("day_logits must have shape (B, 31)")
    if hour_logits.shape != (*day_logits.shape, 24) or level_kw.shape != (day_logits.shape[0],):
        raise ValueError("invalid budget tensor shapes")
    p = F.softmax(day_logits, dim=-1)
    q = F.softmax(hour_logits, dim=-1)
    daily = 31.0 * level_kw[:, None] * p
    cube = 24.0 * daily[:, :, None] * q
    return {"trajectory_kw": cube.flatten(1), "level_kw": level_kw,
            "daily_means_kw": daily, "daily_path_kw": daily - level_kw[:, None],
            "within_day_kw": cube - daily[:, :, None], "energy_kwh": 744.0 * level_kw,
            "day_fractions": p, "hour_fractions": q,
            "log_day_fractions": F.log_softmax(day_logits, -1),
            "log_hour_fractions": F.log_softmax(hour_logits, -1),
            "day_logits": day_logits, "hour_logits": hour_logits}


class SharedCausalBlock(nn.Module):
    """One reusable, masked gated temporal block; the dilation is a call argument."""
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise_weight = nn.Parameter(torch.empty(width, 1, 3))
        self.depthwise_bias = nn.Parameter(torch.zeros(width))
        self.gate = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.depthwise_weight, a=sqrt(5))

    def forward(self, x: Tensor, mask: Tensor, dilation: int) -> Tensor:
        valid = mask[..., None]
        clean = torch.where(valid, x, 0.0)
        z = torch.where(valid, self.norm(clean), 0.0)
        z = F.pad(z.transpose(1, 2), (2 * dilation, 0))
        z = F.conv1d(z, self.depthwise_weight, self.depthwise_bias,
                     dilation=dilation, groups=x.shape[-1]).transpose(1, 2)
        z = torch.where(valid, z, 0.0)
        a, b = self.gate(z).chunk(2, -1)
        z = self.output(torch.tanh(a) * torch.sigmoid(b))
        return torch.where(valid, clean + self.dropout(z), 0.0)


def _unpack(day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale):
    if isinstance(day_tokens, Mapping) or hasattr(day_tokens, "keys"):
        if any(x is not None for x in (day_mask, month_tokens, month_mask, future_tokens, scale)):
            raise ValueError("provide a batch mapping or six tensors, not both")
        batch = day_tokens
        return tuple(batch[k] for k in INPUT_KEYS)
    if any(x is None for x in (day_mask, month_tokens, month_mask, future_tokens, scale)):
        raise ValueError("all six inputs are required")
    return day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale


class StructuredLoadNetwork(nn.Module):
    """Shared ordered encoder and one calendar-conditioned budget decoder (the base network of the carrier).

    The module construction order equals the research code, so a seeded construction gives the same initial weights
    and the parameter names match its checkpoints.
    """
    INPUT_KEYS = INPUT_KEYS

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("provide config or keyword options, not both")
        self.config = config or ModelConfig(**kwargs)
        c = self.config
        if c.variant != "full" or c.calendar_month_head:
            raise ValueError("ankyra-lite implements only the carrier configuration: variant 'full', no calendar month head")
        w = c.width
        self.input_features = max(c.day_features, c.month_features)
        self.history_projection = nn.Linear(self.input_features, w)
        self.age_projection = nn.Linear(6, w, bias=False)
        # One shared signed resolution embedding (+1 daily, -1 monthly tokens).
        self.scale_embedding = nn.Parameter(torch.zeros(w))
        self.blocks = nn.ModuleList([SharedCausalBlock(w, c.dropout) for _ in range(2)])
        self.memory_norm = nn.LayerNorm(w)
        self.future_projection = nn.Sequential(nn.Linear(c.future_features, w), nn.SiLU(), nn.Linear(w, w))
        self.global_query = nn.Parameter(torch.zeros(1, 1, w))
        nn.init.normal_(self.global_query, std=0.02)
        self.attention = nn.MultiheadAttention(w, c.heads, dropout=c.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(w)
        self.query_mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.SiLU(), nn.Linear(2 * w, w))
        self.output_norm = nn.LayerNorm(w)
        self.level_head = nn.Sequential(nn.Linear(w, 64), nn.SiLU(), nn.Linear(64, 1))

        # Low-rank hourly dictionary: fixed zero-mean coordinates, a learned base dictionary (initialised with the
        # first `rank` cosine modes) and a bounded origin-conditioned change.
        hidden = 64
        coordinates = zero_mean_basis(c.hours)
        self.register_buffer("hourly_coordinates", coordinates)
        h = torch.arange(c.hours, dtype=torch.float32) + 0.5
        k = torch.arange(1, c.rank + 1, dtype=torch.float32)
        initial = sqrt(2 / c.hours) * torch.cos(pi * h[:, None] * k[None, :] / c.hours)
        self.base_dictionary = nn.Parameter(coordinates.T @ initial)
        decoder = [nn.Linear(2 * w, hidden), nn.SiLU()]
        self.dictionary_generator = nn.Linear(w, (c.hours - 1) * c.rank)
        decoder.append(nn.Linear(hidden, c.rank + 1))
        self.day_decoder = nn.Sequential(*decoder)

    def architecture(self) -> dict:
        return {"config": self.config.to_dict(), "trainable_parameters": parameter_count(self),
                "complete_forecast_members": 1, "pretrained": False}

    @staticmethod
    def _age_features(length: int, like: Tensor, monthly: bool) -> Tensor:
        age = torch.arange(length - 1, -1, -1, dtype=like.dtype, device=like.device)
        if monthly:
            age = 366.0 + 31.0 * age
        periods = like.new_tensor((7.0, 31.0, 366.0))
        angles = 2.0 * pi * age[:, None] / periods
        return torch.cat((angles.sin(), angles.cos()), -1)

    def _encode(self, tokens: Tensor, mask: Tensor, monthly: bool) -> Tensor:
        if tokens.shape[1] == 0:
            return tokens.new_empty(tokens.shape[0], 0, self.config.width)
        valid = mask[..., None]
        x = torch.where(valid, tokens, 0.0)
        x = F.pad(x, (0, self.input_features - x.shape[-1]))
        x = self.history_projection(x) + self.age_projection(self._age_features(x.shape[1], x, monthly))
        x = x + (-1.0 if monthly else 1.0) * self.scale_embedding
        x = torch.where(valid, x, 0.0)
        schedule = self.config.month_dilations if monthly else self.config.day_dilations
        for i, dilation in enumerate(schedule):
            x = self.blocks[i % len(self.blocks)](x, mask, dilation)
        return torch.where(valid, self.memory_norm(x), 0.0)

    def _validate(self, day: Tensor, dm: Tensor, month: Tensor, mm: Tensor, future: Tensor, scale: Tensor):
        c = self.config
        if day.ndim != 3 or day.shape[-1] != c.day_features or day.shape[1] < 1:
            raise ValueError("invalid day_tokens shape")
        b = day.shape[0]
        if b < 1:
            raise ValueError("a nonempty batch is required")
        if month.ndim != 3 or month.shape[0] != b or month.shape[-1] != c.month_features:
            raise ValueError("invalid month_tokens shape")
        if future.shape != (b, c.days, c.future_features):
            raise ValueError("future_tokens must cover exactly 31 days")
        for tokens, mask, name in ((day, dm, "day"), (month, mm, "month")):
            if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
                raise ValueError(name + " mask must be a Boolean token mask")
            if tokens.device != day.device or tokens.dtype != day.dtype or mask.device != day.device:
                raise ValueError("all tokens and masks must share the input device and token dtype")
            if not tokens.is_floating_point() or not torch.isfinite(tokens[mask]).all():
                raise ValueError("observed token values must be finite floating-point values")
        if future.device != day.device or future.dtype != day.dtype or not torch.isfinite(future).all():
            raise ValueError("future_tokens must be finite and have the history dtype/device")
        if scale.shape not in {(b,), (b, 1)} or scale.device != day.device:
            raise ValueError("scale must have shape (B,) or (B, 1) on the input device")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("scale must be finite and strictly positive")
        if not (dm.any(-1) | mm.any(-1)).all():
            raise ValueError("every sample needs usable history")

    def _trunk(self, day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale):
        """Shared encoder, attention read-out and base decoder (level, day logits, dictionary hour logits)."""
        self._validate(day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale)
        c = self.config
        month_memory = self._encode(month_tokens, month_mask, True)
        day_memory = self._encode(day_tokens, day_mask, False)
        memory = torch.cat((month_memory, day_memory), 1)
        usable = torch.cat((month_mask, day_mask), 1)
        queries = self.future_projection(future_tokens)
        queries = torch.cat((self.global_query.expand(day_tokens.shape[0], -1, -1), queries), 1)
        attended, _ = self.attention(queries, memory, memory, key_padding_mask=~usable, need_weights=False)
        z = self.query_norm(queries + attended)
        z = self.output_norm(z + self.query_mlp(z))
        v, daily = z[:, 0], z[:, 1:]
        raw_level = self.level_head(v).squeeze(-1)
        level = scale.reshape(-1).to(day_tokens.dtype) * F.softplus(raw_level)
        dec_in = torch.cat((daily, v[:, None, :].expand(-1, c.days, -1)), -1)
        decoded = self.day_decoder(dec_in)
        day_logits = decoded[..., 0]
        coordinates = self.base_dictionary[None].expand(day_tokens.shape[0], -1, -1)
        change = self.dictionary_generator(v).reshape(-1, c.hours - 1, c.rank)
        coordinates = coordinates + c.dictionary_delta * torch.tanh(change)
        dictionary = torch.einsum("hr,brk->bhk", self.hourly_coordinates, coordinates)
        dictionary = dictionary - dictionary.mean(1, keepdim=True)
        coefficients = decoded[..., 1:]
        hour_logits = torch.einsum("bhr,bdr->bdh", dictionary, coefficients)
        return dict(day_memory=day_memory, daily=daily, v=v, raw_level=raw_level, level=level, dec_in=dec_in,
                    day_logits=day_logits, hour_logits=hour_logits, dictionary=dictionary, coefficients=coefficients)

    @staticmethod
    def _finish(t: dict, day_logits: Tensor, hour_logits: Tensor, scale: Tensor, log744: Tensor | float) -> dict:
        result = decode_budget(t["level"], day_logits, hour_logits)
        raw_level = t["raw_level"]
        # log(softplus(x)) is asymptotically x for large negative x; the clamp keeps the unused branch finite.
        log_level_normalized = torch.where(raw_level < -20.0, raw_level, F.softplus(raw_level.clamp_min(-20.0)).log())
        log_trajectory_normalized = (log744 + log_level_normalized[:, None, None]
                                     + result["log_day_fractions"][:, :, None] + result["log_hour_fractions"])
        result.update(dictionary=t["dictionary"], shape_coefficients=t["coefficients"], scale_kw=scale.reshape(-1),
                      log_trajectory_normalized=log_trajectory_normalized.flatten(1))
        return result

    def forward(self, day_tokens: Tensor | Mapping, day_mask: Tensor | None = None, month_tokens: Tensor | None = None,
                month_mask: Tensor | None = None, future_tokens: Tensor | None = None,
                scale: Tensor | None = None) -> dict[str, Tensor]:
        """The base decoder without the copy pointer."""
        x = _unpack(day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale)
        t = self._trunk(*x)
        return self._finish(t, t["day_logits"], t["hour_logits"], x[5], log(744.0))


class CopyPointerNetwork(StructuredLoadNetwork):
    """The carrier network: the base network plus the copy pointer over the unit's own last ``copy_days`` complete days.

      hour_logits_d += g_h,d * l_h,d     (l_h: zero-mean log of the attended within-day share profile)
      day_logits_d  += g_d,d * l_d,d     (l_d: zero-mean log of the attended relative daily mean)

    The gates g are a linear map of the decoder input, initialised at zero. The level head, the budget decoder and its
    positivity and exact energy decomposition are unchanged.
    """
    POINTER_MODULES = ("pointer_query", "pointer_key", "copy_gate", "type_bonus")

    def __init__(self, config: ModelConfig | None = None, copy_days: int = 56, key_dim: int = 16, copy_hour: bool = True,
                 copy_day: bool = True, **kwargs):
        super().__init__(config, **kwargs)
        if not (copy_hour or copy_day):
            raise ValueError("the pointer must copy the hourly shape, the daily means or both")
        w = self.config.width
        self.copy_days, self.key_dim, self.copy_hour, self.copy_day = copy_days, key_dim, copy_hour, copy_day
        self.pointer_query = nn.Linear(w, key_dim)
        self.pointer_key = nn.Linear(w, key_dim)
        self.type_bonus = nn.Parameter(torch.zeros(1))
        self.copy_gate = nn.Linear(2 * w, int(copy_hour) + int(copy_day))
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.zeros_(self.copy_gate.bias)

    def architecture(self) -> dict:
        a = super().architecture()
        a.update(pointer={"copy_days": self.copy_days, "key_dim": self.key_dim, "copy_hour": self.copy_hour,
                          "copy_day": self.copy_day})
        return a

    def forward(self, day_tokens: Tensor | Mapping, day_mask: Tensor | None = None, month_tokens: Tensor | None = None,
                month_mask: Tensor | None = None, future_tokens: Tensor | None = None,
                scale: Tensor | None = None) -> dict[str, Tensor]:
        day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale = _unpack(
            day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale)
        t = self._trunk(day_tokens, day_mask, month_tokens, month_mask, future_tokens, scale)
        day_logits, hour_logits, daily, dec_in = t["day_logits"], t["hour_logits"], t["daily"], t["dec_in"]

        # Copy pointer over the unit's own last `copy_days` complete days.
        L = min(self.copy_days, day_tokens.shape[1])
        dm = day_mask[:, -L:]
        tok = torch.where(dm[..., None], day_tokens[:, -L:], torch.zeros_like(day_tokens[:, -L:]))  # masked tokens may hold NaN
        valid = dm & (tok[..., 24:48] > 0.5).all(-1)                                       # complete days (zero-load days included)
        load = torch.sinh(tok[..., :24]).clamp_min(0.0)                                   # load / scale, hourly
        mean = load.mean(-1)
        shares = torch.where((mean > 1e-6)[..., None], load / (24.0 * mean.clamp_min(1e-6))[..., None],
                             torch.full_like(load, 1.0 / 24))
        match = torch.einsum("bdt,bjt->bdj", future_tokens[..., 0:8], tok[..., 52:60])
        score = torch.einsum("bdk,bjk->bdj", self.pointer_query(daily),
                             self.pointer_key(t["day_memory"][:, -L:])) / sqrt(self.key_dim)
        score = score + self.type_bonus * match
        score = score.masked_fill(~valid[:, None, :], float("-inf"))
        any_valid = valid.any(-1)
        alpha = torch.softmax(torch.where(any_valid[:, None, None], score, torch.zeros_like(score)), -1)
        alpha = torch.where(any_valid[:, None, None], alpha, torch.zeros_like(alpha))
        gate = self.copy_gate(dec_in) * any_valid[:, None, None].to(dec_in.dtype)
        j = 0
        if self.copy_hour:
            ls = torch.log(torch.einsum("bdj,bjh->bdh", alpha, shares).clamp_min(1e-4))
            ls = ls - ls.mean(-1, keepdim=True)
            hour_logits = hour_logits + gate[..., j:j + 1] * torch.where(any_valid[:, None, None], ls, torch.zeros_like(ls))
            j += 1
        if self.copy_day:
            lm = torch.log(torch.einsum("bdj,bj->bd", alpha, mean).clamp_min(1e-4))
            lm = lm - lm.mean(-1, keepdim=True)
            day_logits = day_logits + gate[..., j] * torch.where(any_valid[:, None], lm, torch.zeros_like(lm))

        level = t["level"]
        result = self._finish(t, day_logits, hour_logits, scale, torch.log(torch.tensor(744.0, dtype=level.dtype, device=level.device)))
        result.update(copy_gate=gate, copy_alpha=alpha)
        return result


def build_carrier() -> CopyPointerNetwork:
    """The carrier configuration: base width 48, rank 8, 366 daily + 24 monthly tokens, copy pointer over 56 days with
    16-dimensional keys copying both the hourly shape and the daily means (61,613 trainable parameters)."""
    return CopyPointerNetwork(ModelConfig(day_features=69, month_features=16, future_features=16, variant="full",
                                          calendar_month_head=False),
                              copy_days=56, key_dim=16, copy_hour=True, copy_day=True)


def load_carrier(path: str | Path, map_location: str | torch.device = "cpu") -> CopyPointerNetwork:
    """A carrier from a checkpoint file ({"model": state_dict, ...} or a bare state_dict), strict, in eval mode."""
    ck = torch.load(Path(path), map_location=map_location, weights_only=True)
    state = ck["model"] if isinstance(ck, Mapping) and "model" in ck else ck
    model = build_carrier()
    model.load_state_dict(state, strict=True)
    return model.to(map_location).eval()


def param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(base parameters, pointer parameters). Training clips the gradient norm of each group separately (max norm 1),
    so the base parameters see the same clipping as the network without a pointer."""
    names = getattr(model, "POINTER_MODULES", ())
    base = [p for n, p in model.named_parameters() if n.split(".")[0] not in names]
    ptr = [p for n, p in model.named_parameters() if n.split(".")[0] in names]
    return base, ptr
