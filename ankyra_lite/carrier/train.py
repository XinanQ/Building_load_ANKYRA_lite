"""Training recipe of the ankyra-lite carrier network.

The carrier (``ankyra_lite.carrier.model.build_carrier``: a structured load network with a copy pointer, 61,613
parameters) forecasts the 744 hours after a forecast origin from the unit's own history before the origin. One carrier
is trained per target population; its forecasts (averaged over the seeds) are then passed to the anchoring layer
(``ankyra_lite.anchor``). Every constant of the recipe is a field of :class:`TrainConfig`. The released recipe of
version 0.1.0 (the study's R recipe) has two stages: stage 1 is ``TrainConfig()`` (the defaults below; the study's P3s
recipe: training from scratch with early stopping on a fixed 93-day validation set) and stage 2 is
``recipe("span_return", <stage-1 checkpoint>)`` (the validation span returned to the training set, exactly 8 more
epochs, final weights kept; see "Continuation recipes"). Recipe Rw differs from R in stage 2 only:
``recipe("span_return_weighted", <stage-1 checkpoint>)`` is the same continuation with an inverse-noise weighting of
the training windows in the loss (see "Loss weighting"). ``RECIPE_KEYS`` maps the keys "R" and "Rw" to these stage-2
configurations.

Stage 1 of the released recipe (defaults: training from scratch)
----------------------------------------------------------------
1. Windows. A window is a pair (unit, origin o): context hours [o - 1344, o), label hours [o, o + 744). Only data
   before the training cut is used (o + 744 <= cut; load and temperature at or after the cut are blanked before any
   feature is computed). A window is eligible when at least 75% of its context hours and at least 75% of its label
   hours are observed (finite and nonnegative). Training origins lie on a daily grid (every 24 h from hour 1344).
2. Split. The validation span is the last 93 days (3 x 744 h) before the cut: boundary v0 = max(1344 + 744,
   cut - 2232). Training windows end at or before v0 (o + 744 <= v0); validation windows start at or after v0
   (o >= v0, o + 744 <= cut); windows across v0 are dropped. The rule is pluggable (:class:`FixedSpan`, stage 1 of
   the released recipe; :class:`ReturnSpan`, its stage 2; or :class:`ProportionalSpan`).
3. Caps. At most 16,000 training and 2,000 validation windows per population, chosen unit-balanced (randomised round
   robin across units, :func:`balanced`), training windows first, then validation windows, from one random stream.
4. Validation set. Early stopping uses the target population's own validation windows on the weekly grid (every
   168 h from hour 1344; same span, eligibility and caps).
5. Pool. For the target P, with validation start T = start(P) + v0(P) in absolute time (hours since 2000-01-01 UTC),
   every population Q (P included) contributes its daily-grid windows whose labels end at or before T in absolute time
   (start(Q) + o + 744 <= T). For P this is exactly its training split; windows of other populations ("borrowed"
   windows) may come from their training or their validation split, provided they end before T. Per population the
   number of windows is matched to the weekly grid: it equals the number of Q's capped weekly-grid windows (step 3)
   that pass the same time limit; when more of Q's capped daily-grid rows pass it, a unit-balanced subsample of those
   capped daily-grid rows is drawn (not of all eligible daily windows), so the daily grid adds origin-day coverage,
   not windows.
6. Sampler. Each epoch draws 16,000 windows with replacement: half from P (each unit with equal total weight),
   half from the borrowed populations (population chosen uniformly per draw, then each unit with equal total weight);
   the draws are shuffled. Without borrowed populations all draws come from P.
7. Optimisation. AdamW (learning rate 3e-4, weight decay 1e-4), batch 64, loss ``forecast_loss`` (scale-normalised
   trajectory MSE + 0.1 x generalised KL, log1p-compressed per window), gradient-norm clipping at 1.0 separately for
   the base parameters and for the copy-pointer parameters.
8. Early stopping. After each epoch the validation criterion (:func:`validate`) is computed; the best weights are
   kept; training stops after 5 epochs without improvement, or after 30 epochs.
9. Seeds and ensemble. Seeds 0, 1, 2 (the seed sets the weight initialisation and the sampler stream); the carrier
   forecast is the average of the three seeds' forecasts (:func:`average_seed_forecasts`). Any seed list can be
   trained the same way (``train_ensemble(..., seeds=range(6))`` for six seeds; the forecast is then the average over
   all of them, in ascending seed order).

Continuation recipes
--------------------
Three recipes continue trained weights (one start checkpoint per population and seed, ``init_checkpoint``) with a
fresh AdamW optimiser (same learning rate and weight decay) and the sampler stream ``default_rng(10000 + seed)``;
windows, caps, borrowed windows and their time limit are those of steps 1-5 (:func:`recipe` builds the
configurations; ``"span_return_weighted"``, the span return with loss weighting, is described under "Loss
weighting"):

* ``"plain_continuation"`` (control): the same training pool, exactly 8 epochs, the final weights are kept (no
  selection). The validation criterion on the step-4 validation set is recorded per epoch as a description only.
* ``"span_return"`` (stage 2 of the released recipe): as the plain continuation, plus the 93-day validation span
  returned to the target's training pool (:class:`ReturnSpan`): the target's own daily-grid span windows whose labels
  end at least 24 h before the cut, at the population's own sampling density, n = round(own pool windows / own
  training candidates x span candidates).
* ``"proportional_validation"``: the span windows of the span return, split at a proportional boundary
  (``ReturnSpan(holdout=ProportionalSpan())``): validation days = clip(10% of the population's own record before the
  cut, 38, 93); returned windows whose labels end at or before the boundary join the training pool, and the
  validation set is every own daily-grid span window from the boundary on (labels ending at least 24 h before the
  cut). Early stopping as step 8 with the start weights' own validation score counted as epoch 0, at least 5 epochs,
  at most 30.

Before the first continuation epoch the criterion of the start weights is computed (epoch 0). When the start
checkpoint records the criterion it was selected with (``"val"``) and the validation set is the step-4 set, it must
be reproduced within ``start_val_rtol`` (relative), otherwise the start does not belong to these data.

Loss weighting (stage 2 of recipe Rw)
-------------------------------------
``recipe("span_return_weighted", ...)`` is the span return with ``noise_weighting=NoiseWeights()``; pool, sampler,
optimiser, epochs and clipping are unchanged. Each training window i gets a weight w_i computed once per pool, before
training, from the windows' own context only (:func:`noise_weights`; :class:`NoiseWeights` holds the constants):

1. Noise estimate. With x = sinh(day token[:24]) = load / s_i (s_i the window's causal scale), sigma2_i is the mean of
   (x_t - x_{t-168 h})^2 over the hours of day tokens 345..365 (the last 21 days before the origin) paired with the
   same hour 7 days earlier, both observed; undefined with fewer than 168 such pairs.
2. Floor f: the 25th percentile of sigma2 over the pool, weighted by the sampler's exact draw probabilities p
   (:func:`draw_probabilities`: the target's share and the borrowed populations' share of the draws, units at equal
   weight within each population).
3. Raw weight omega_i = max(sigma2_i, f)^(-1/2); a window with undefined sigma2 gets the p-weighted median raw weight.
4. Per-population normalisation: w_i = c_Q omega_i, with c_Q such that the p-weighted mean weight of every population
   Q is 1 (each population keeps its share of the draws).
5. Loss: per batch sum_i w_i l_i / sum_i w_i, l_i = log1p(clamp_0(per-window loss)) (the per-window term of
   ``forecast_loss`` is unchanged; :func:`weighted_log1p_loss`).

The weights depend only on the day tokens of the pool's windows (context hours before each origin) and on the sampler
rule; no label hour, no validation window and no per-unit or per-population constant is used. With every weight equal
to 1 the weighted loss equals the unweighted one bit for bit on the CPU; with ``noise_weighting=None`` (every other
recipe) the code path is that of version 0.1.

Typical use::

    cfg = TrainConfig()                                             # stage 1; stage 2: recipe("span_return", "runs/{target}/stage1/seed{seed}.pt")
    sources = {name: prepare_source(name, data[name], cut[name], cfg, seed_index=i) for i, name in enumerate(names)}
    results = train_ensemble("target_name", sources, cfg)          # one TrainResult per seed
    models = [r.model for r in results]

For stage 2, prepare the target with ``with_validation=True`` and the other populations with
``with_validation=False`` under the span-return configuration (docs/METHOD.md, "Seeds and ensemble"); recipe Rw uses
the same sources (``recipe("Rw", ...)`` changes the loss only). The released
weights, checkpoints/<population>/seed<s>.pt, are stage-2 final weights, not stage-1 checkpoints: they are not a
start point (``init_checkpoint``) for stage 2.

``data[name]`` is a store mapping with ``load`` (units x hours, kW, NaN where missing), ``temp`` (hours or units x
hours, degC), ``day_types`` (or ``types``; 0..6 Monday..Sunday, 7 holiday), ``groups`` (one category per unit) and
``start_iso`` (timestamp of hour 0), i.e. the arrays read by ``ankyra_lite.carrier.features.features``; any other
feature builder can be passed as ``feature_fn(data, unit, origin)``. The per-population seeds use ``seed_index``; to
reproduce the development record exactly, give the populations in the original order with their index in the original
population list.

Validation rule: ``TrainConfig(validation=ProportionalSpan())`` switches to the proportional split from scratch;
anything with ``boundary(cut, first_origin, context_hours, horizon_hours)``, ``validation_days(cut, first_origin)``
and ``end_gap_hours`` can be used as a rule.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")           # deterministic cuBLAS (only effective before CUDA starts)

__all__ = ["TrainConfig", "FixedSpan", "ProportionalSpan", "ReturnSpan", "RowSet", "ValidationSet", "Source", "SpanPlan",
           "Pool", "RowConcat", "StartWeights", "TrainResult", "make_windows", "split", "balanced", "select_rows", "cut_data",
           "materialise", "prepare_source", "plan_span", "build_pool", "training_arrays", "sampler", "make_batch",
           "configure", "make_optimizer", "train_step", "run_epoch", "validate", "start_weights", "initial_model",
           "train_one", "train_ensemble", "average_seed_forecasts", "absolute_start_hour", "with_validation_rule", "recipe",
           "RECIPES", "CONTINUATION_SEED_OFFSET", "CONTINUATION_EPOCHS", "INPUTS", "ARRAY_KEYS", "NoiseWeights",
           "noise_sigma2", "pool_sigma2", "draw_probabilities", "weighted_quantile", "noise_weights", "expected_weight",
           "weighted_log1p_loss", "WEIGHTED_RECIPES", "RECIPE_KEYS"]

INPUTS = ("day_tokens", "day_mask", "month_tokens", "month_mask", "future_tokens", "scale")
ARRAY_KEYS = INPUTS + ("target", "observed")
ABSOLUTE_EPOCH = datetime(2000, 1, 1)                                  # absolute hours count from 2000-01-01 00:00 UTC
_TOKEN_KEYS = ("day_tokens", "month_tokens")                           # stored at TrainConfig.token_dtype
_BOOL_KEYS = ("day_mask", "month_mask", "observed")
CONTINUATION_SEED_OFFSET = 10000                                       # sampler stream of the continuation recipes
CONTINUATION_EPOCHS = 8                                                # fixed length of the plain continuation and the span return
RECIPES = ("from_scratch", "plain_continuation", "span_return", "proportional_validation")
WEIGHTED_RECIPES = ("span_return_weighted",)                          # recipes with the inverse-noise loss weighting
RECIPE_KEYS = {"R": "span_return", "Rw": "span_return_weighted"}       # release recipe key -> its stage-2 configuration
NOISE_CHUNK_ROWS = 512                                                 # rows per token read when the weights are computed


# ----------------------------------------------------------------------------- validation rules
@dataclass(frozen=True)
class FixedSpan:
    """Validation span of fixed length immediately before the training cut (stage 1 of the released recipe: 93 days =
    3 x 744 h; stage 2, :class:`ReturnSpan`, uses the same split and then returns the span).

    boundary v0 = max(context + horizon, cut - round(24 * days)); validation windows also end ``end_gap_hours`` or
    more before the cut."""
    days: float = 93.0
    end_gap_hours: int = 0

    def validation_days(self, cut: int, first_origin: int) -> float:
        return float(self.days)

    def boundary(self, cut: int, first_origin: int, context_hours: int, horizon_hours: int) -> int:
        return max(context_hours + horizon_hours, int(cut) - int(round(24 * self.days)))


@dataclass(frozen=True)
class ProportionalSpan:
    """Validation span proportional to the population's own record before the cut.

    validation days = clip(fraction x record days, min_days, max_days), with record days = (cut - first origin) / 24;
    boundary b = max(context + horizon, cut - 24 x round(validation days)); validation windows end ``end_gap_hours``
    or more before the cut. The defaults (10%, 38 to 93 days, 24 h) are the rule of the proportional-validation
    recipe: 38 days = one 31-day label window plus 7 daily origins, 93 days = the fixed span, so long records are
    unchanged.

    Two uses. As ``TrainConfig(validation=ProportionalSpan())`` it is a from-scratch split (record measured from the
    first eligible training-grid window). As ``ReturnSpan(holdout=ProportionalSpan())`` it is the hold-out rule of the
    proportional-validation continuation: there the record is measured from the first built training-grid window
    (equal to the first eligible window whenever the training cap does not bind) and only the returned span is split
    (see :class:`ReturnSpan`)."""
    fraction: float = 0.10
    min_days: float = 38.0
    max_days: float = 93.0
    end_gap_hours: int = 24

    def validation_days(self, cut: int, first_origin: int) -> float:
        record_days = (int(cut) - int(first_origin)) / 24.0
        return float(min(self.max_days, max(self.min_days, self.fraction * record_days)))

    def boundary(self, cut: int, first_origin: int, context_hours: int, horizon_hours: int) -> int:
        return max(context_hours + horizon_hours, int(cut) - 24 * int(round(self.validation_days(cut, first_origin))))


@dataclass(frozen=True)
class ReturnSpan:
    """Span return: the validation span of the fixed split goes back into the target's training pool.

    The base split is :class:`FixedSpan` with the same ``days`` and ``end_gap_hours``, so windows, caps, the pool and
    the time limit of the borrowed windows are those of the from-scratch recipe. Then the target's own training-grid
    windows of the span [v0, cut) whose labels end at least ``boundary_hours`` before the cut are returned to its
    training pool at the population's own sampling density: n = round(own pool windows / own training candidates x
    span candidates) windows (candidates: eligible training-grid windows before the caps). They are a unit-balanced
    subsample (stream ``default_rng(seed + seed_index)``) of the built span windows; when fewer are built, all of them
    are returned and the rest is a unit-balanced subsample, from the same stream, of the other eligible span windows,
    which are then built with the same feature path (store blanked at the cut). The boundary keeps every returned
    label 24 h clear of the cut, so no cleaning step can depend on data at or after the cut.

    ``holdout`` (a :class:`ProportionalSpan`) turns this into the proportional-validation recipe: with the boundary b
    of the hold-out rule (record measured from the first built training-grid window), only the returned windows whose
    labels end at or before b join the training pool, and the validation set is every own span window (built rows and
    extra rows) with origin >= b whose labels end ``holdout.end_gap_hours`` or more before the cut. Without
    ``holdout`` there is no validation set for early stopping: train a fixed number of epochs
    (``TrainConfig.fixed_epochs``)."""
    days: float = 93.0
    boundary_hours: int = 24
    seed: int = 20261019
    holdout: ProportionalSpan | None = None
    end_gap_hours: int = 0                    # of the base split (as FixedSpan)

    def validation_days(self, cut: int, first_origin: int) -> float:
        return float(self.days)

    def boundary(self, cut: int, first_origin: int, context_hours: int, horizon_hours: int) -> int:
        return FixedSpan(self.days, self.end_gap_hours).boundary(cut, first_origin, context_hours, horizon_hours)


# ----------------------------------------------------------------------------- loss weighting rule
@dataclass(frozen=True)
class NoiseWeights:
    """Inverse-noise weighting of the training windows in the loss (stage 2 of recipe Rw; module docstring, "Loss
    weighting"). The defaults are the frozen constants of the recipe.

    The noise estimate reads day tokens ``context_days`` (days 338..365 of the 366 daily tokens, i.e. the last 28
    days before the origin), columns 0..23 (arcsinh(load / scale)) and 24..47 (hourly observation flags): pairs
    (t, t - ``lag_days``) on days 345..365, both hours observed (flag > ``observed_threshold``), at least
    ``min_pairs`` pairs. floor = weighted ``floor_quantile`` of sigma2 over the pool; raw weight
    max(sigma2, floor)^(-``gamma``)."""
    gamma: float = 0.5
    floor_quantile: float = 0.25
    min_pairs: int = 168
    context_days: tuple[int, int] = (338, 366)
    lag_days: int = 7
    observed_threshold: float = 0.5


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TrainConfig:
    """Every constant of the training recipe; the defaults are stage 1 of the released recipe (training from scratch
    with early stopping); :func:`recipe` builds stage 2 (``"span_return"``) and the other continuation recipes."""
    # windows
    context_hours: int = 1344                 # 56 days of context
    horizon_hours: int = 744                  # 31-day label window
    min_observed_fraction: float = 0.75       # eligibility: observed share of the context and of the label hours
    train_origin_grid_hours: int = 24         # daily training origins
    val_origin_grid_hours: int = 168          # validation origins (weekly grid)
    pool_count_grid_hours: int | None = 168   # per-population pool count matched to this grid; None = all capped daily-grid rows before the limit
    # split and caps
    validation: FixedSpan | ProportionalSpan | ReturnSpan = field(default_factory=FixedSpan)
    train_cap: int = 16000                    # training windows per population (unit-balanced)
    val_cap: int = 2000                       # validation windows per population (unit-balanced)
    row_seeds: tuple[tuple[int, int], ...] = ((24, 20261015), (168, 20261014))   # (grid hours, base seed) for the caps
    pool_seed: int = 20261215                 # base seed of the per-population pool subsample (daily seed + 200)
    # sampler
    draws_per_epoch: int = 16000
    own_fraction: float = 0.5                 # share of the draws from the target population (when others exist)
    sampler_seed_offset: int = 0              # sampler stream = default_rng(offset + seed)
    # optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    grad_clip: float = 1.0                    # max gradient norm, per parameter group (base, copy pointer)
    kl_weight: float = 0.1
    log1p_compression: bool = True
    # early stopping and seeds
    max_epochs: int = 30
    patience: int = 5
    min_epochs: int = 0                       # no stop before this many epochs (0: patience alone decides)
    seeds: tuple[int, ...] = (0, 1, 2)
    # continuation (defaults: training from scratch)
    init_checkpoint: str | None = None        # start weights; "{target}" and "{seed}" are filled in (one file per population and seed)
    fixed_epochs: int | None = None           # train exactly this many epochs and keep the final weights (no selection)
    count_start_as_epoch0: bool = False       # early stopping: the start weights are a candidate (epoch 0)
    start_val_rtol: float | None = 1e-5       # epoch-0 check against the start checkpoint's recorded criterion; None = off
    # numerics
    token_dtype: str = "float16"              # storage precision of day and month tokens (cast to float32 per batch)
    torch_threads: int | None = 2             # torch.set_num_threads in configure(); None leaves it unchanged
    deterministic: bool = True
    # loss weighting (stage 2 of recipe Rw); None: every window has weight 1 (all other recipes)
    noise_weighting: NoiseWeights | None = None

    def row_seed(self, grid_hours: int) -> int:
        for g, s in self.row_seeds:
            if int(g) == int(grid_hours):
                return int(s)
        raise KeyError(f"no row seed for a {grid_hours}-h origin grid")

    def init_path(self, target: str, seed: int) -> str | None:
        """The start checkpoint of one population and seed (``init_checkpoint`` with ``{target}`` and ``{seed}``
        filled in), or None when training from scratch."""
        return None if self.init_checkpoint is None else str(self.init_checkpoint).format(target=target, seed=int(seed))

    def describe(self) -> dict:
        """The configuration as a JSON-ready mapping (training records); without loss weighting the key
        ``noise_weighting`` is left out, so the records of the unweighted recipes are those of version 0.1."""
        d = asdict(self)
        d["validation"] = {"rule": type(self.validation).__name__, **asdict(self.validation)}
        if isinstance(self.validation, ReturnSpan) and self.validation.holdout is not None:
            d["validation"]["holdout"] = {"rule": type(self.validation.holdout).__name__, **asdict(self.validation.holdout)}
        if self.noise_weighting is None:
            del d["noise_weighting"]
        else:
            d["noise_weighting"] = {"rule": type(self.noise_weighting).__name__, **asdict(self.noise_weighting)}
        return d


def recipe(name: str = "from_scratch", init_checkpoint: str | None = None, **overrides) -> TrainConfig:
    """The configuration of a named recipe (``RECIPES``, ``WEIGHTED_RECIPES``, or a key of ``RECIPE_KEYS``), with
    field ``overrides``.

    from_scratch             TrainConfig() (stage 1 of the released recipe: training from scratch, fixed 93-day
                             validation set, early stopping); with ``init_checkpoint``, the same schedule from those weights
    plain_continuation       continue ``init_checkpoint`` for exactly 8 epochs (fresh AdamW, stream 10000 + seed), final weights
    span_return              stage 2 of recipe R: plain continuation plus the 93-day span returned to the training pool
                             (``ReturnSpan()``)
    proportional_validation  continuation with ``ReturnSpan(holdout=ProportionalSpan())``: early stopping on the
                             proportional hold-out, the start counted as epoch 0, at least 5 and at most 30 epochs
    span_return_weighted     stage 2 of recipe Rw: the span return with the inverse-noise loss weighting
                             (``noise_weighting=NoiseWeights()``)
    R, Rw                    the stage-2 configurations of recipes R and Rw (``RECIPE_KEYS``; stage 1 of both is
                             ``TrainConfig()``)

    ``init_checkpoint`` may contain ``{target}`` and ``{seed}`` (e.g. ``"runs/{target}/stage1/seed{seed}.pt"``, the
    stage-1 weights of a continuation; the released checkpoints/<population>/seed<s>.pt are stage-2 final weights)."""
    cont = dict(init_checkpoint=init_checkpoint, sampler_seed_offset=CONTINUATION_SEED_OFFSET)
    name = RECIPE_KEYS.get(name, name)
    if name == "from_scratch":
        cfg = TrainConfig(init_checkpoint=init_checkpoint)
    elif name == "plain_continuation":
        cfg = TrainConfig(fixed_epochs=CONTINUATION_EPOCHS, **cont)
    elif name == "span_return":
        cfg = TrainConfig(fixed_epochs=CONTINUATION_EPOCHS, validation=ReturnSpan(), **cont)
    elif name == "proportional_validation":
        cfg = TrainConfig(validation=ReturnSpan(holdout=ProportionalSpan()), count_start_as_epoch0=True, min_epochs=5,
                          max_epochs=30, patience=5, **cont)
    elif name == "span_return_weighted":
        cfg = TrainConfig(fixed_epochs=CONTINUATION_EPOCHS, validation=ReturnSpan(), noise_weighting=NoiseWeights(), **cont)
    else:
        raise ValueError(f"unknown recipe {name!r}; one of {RECIPES + WEIGHTED_RECIPES} or {tuple(RECIPE_KEYS)}")
    return replace(cfg, **overrides) if overrides else cfg


def _holdout(config: TrainConfig) -> ProportionalSpan | None:
    return config.validation.holdout if isinstance(config.validation, ReturnSpan) else None


# ----------------------------------------------------------------------------- windows, split, caps
def absolute_start_hour(start_iso: str) -> int:
    """Absolute hour (hours since 2000-01-01 00:00 UTC) of a record's first hour; naive timestamps are read as UTC."""
    t = datetime.fromisoformat(str(start_iso).replace("Z", "+00:00"))
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    return int(round((t - ABSOLUTE_EPOCH).total_seconds() / 3600))


def make_windows(load: np.ndarray, cut: int, *, grid_hours: int = 24, context_hours: int = 1344, horizon_hours: int = 744,
                 min_observed: float = 0.75) -> np.ndarray:
    """All eligible windows before ``cut`` as an (n, 2) int64 array of (unit, origin), unit-major, origins ascending.

    Origins run from ``context_hours`` in steps of ``grid_hours`` with ``origin + horizon_hours <= cut``. A window is
    eligible when the observed share (finite and >= 0, checked in float32) of its context hours and of its label hours
    are both at least ``min_observed``."""
    L = np.asarray(load)
    L = L[None] if L.ndim == 1 else L
    cut = int(cut)
    if cut > L.shape[1]:
        raise ValueError("cut lies beyond the end of the record")
    origins = np.arange(context_hours, cut - horizon_hours + 1, grid_hours, dtype=np.int64)
    out = []
    if not len(origins):
        return np.zeros((0, 2), np.int64)
    for b in range(L.shape[0]):
        x = L[b, :cut].astype(np.float32)
        good = np.concatenate([[0], np.cumsum(np.isfinite(x) & (x >= 0), dtype=np.int64)])
        ctx = (good[origins] - good[origins - context_hours]) / context_hours
        tgt = (good[origins + horizon_hours] - good[origins]) / horizon_hours
        ok = origins[(ctx >= min_observed) & (tgt >= min_observed)]
        out.append(np.stack([np.full(len(ok), b, np.int64), ok], 1))
    return np.concatenate(out).astype(np.int64).reshape(-1, 2)


def split(windows: np.ndarray, cut: int, config: TrainConfig = TrainConfig(), boundary: int | None = None) -> tuple[np.ndarray, int]:
    """Labels 'train' / 'val' / '' (dropped: across the boundary) for each window, and the validation boundary v0.

    v0 comes from ``config.validation`` (first origin = the first window's origin) unless ``boundary`` is given."""
    w = np.asarray(windows, np.int64).reshape(-1, 2)
    if boundary is None:
        first = int(w[:, 1].min()) if len(w) else int(config.context_hours)
        boundary = config.validation.boundary(int(cut), first, config.context_hours, config.horizon_hours)
    o, H = w[:, 1], config.horizon_hours
    lab = np.full(len(w), "", dtype="<U5")
    lab[o + H <= boundary] = "train"
    lab[(o >= boundary) & (o + H <= int(cut) - int(config.validation.end_gap_hours))] = "val"
    return lab, int(boundary)


def balanced(pairs, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-balanced subsample of at most ``cap`` rows of an (n, 2) array whose first column is the unit: units are
    shuffled into a randomised round robin (each unit's rows shuffled; every round visits the units in a new random
    order), so all units contribute equally until they run out. Returns the input unchanged when it fits."""
    p = np.asarray(pairs, np.int64).reshape(-1, 2)
    if len(p) <= cap:
        return p
    bins = [p[p[:, 0] == b] for b in np.unique(p[:, 0])]
    for pp in bins:
        rng.shuffle(pp)
    sel, depth = [], 0
    while len(sel) < cap:
        added = False
        for k in rng.permutation(len(bins)):
            if depth < len(bins[k]):
                sel.append(bins[k][depth]); added = True
                if len(sel) == cap:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(sel, np.int64).reshape(-1, 2)


def select_rows(windows: np.ndarray, labels: np.ndarray, *, train_cap: int, val_cap: int, rng: np.random.Generator):
    """Capped rows: unit-balanced training windows first, then validation windows (one stream). Returns (rows, split)."""
    w = np.asarray(windows, np.int64).reshape(-1, 2)
    tr = balanced(w[labels == "train"], train_cap, rng)
    va = balanced(w[labels == "val"], val_cap, rng)
    rows = np.concatenate([tr, va]).reshape(-1, 2)
    return rows, np.array(["train"] * len(tr) + ["val"] * len(va), dtype="<U5")


# ----------------------------------------------------------------------------- feature arrays
def _default_feature_fn() -> Callable:
    """``fn(data, unit, origin)`` -> the carrier inputs of ``ankyra_lite.carrier.features.features`` for a store mapping
    (tokens are returned in float32 here; :func:`materialise` stores them at ``token_dtype``)."""
    from ankyra_lite.carrier.features import features

    def fn(data, unit, origin):
        types = data.get("day_types", data.get("types"))
        return features(data["load"], data.get("temp"), types, data.get("groups"), data["start_iso"], unit, origin, half=False)
    return fn


def cut_data(data: Mapping, cut: int) -> dict:
    """A copy of a store mapping with load (float64) and temperature (float32) blanked at and after ``cut``; the
    calendar (``types``) is known in advance and kept."""
    d = dict(data)
    load = np.array(d["load"], np.float64, copy=True)
    load[..., int(cut):] = np.nan
    d["load"] = load
    if d.get("temp") is not None:
        temp = np.array(d["temp"], np.float32, copy=True)
        temp[..., int(cut):] = np.nan
        d["temp"] = temp
    d["cut"] = int(cut)
    return d


def materialise(data: Mapping, rows: np.ndarray, feature_fn: Callable | None = None, *, horizon_hours: int = 744,
                token_dtype: str = "float16", out_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    """Model inputs and targets for each (unit, origin) row: the six input arrays from ``feature_fn(data, unit,
    origin)``, ``target`` (744 h of load, NaN where unobserved) and ``observed`` (finite and >= 0). Day and month
    tokens are stored at ``token_dtype``. With ``out_dir`` the arrays are written as .npy memory maps there."""
    feature_fn = feature_fn or _default_feature_fn()
    rows = np.asarray(rows, np.int64).reshape(-1, 2)
    n = len(rows)
    load = np.asarray(data["load"])
    load = load[None] if load.ndim == 1 else load
    arr: dict[str, np.ndarray] = {}
    for i, (b, o) in enumerate(rows):
        f = feature_fn(data, int(b), int(o))
        if not arr:
            for k in INPUTS:
                v = np.asarray(f[k])
                dt = np.dtype(token_dtype) if k in _TOKEN_KEYS else np.dtype(bool) if k in _BOOL_KEYS else np.dtype(np.float32)
                arr[k] = _alloc(out_dir, k, (n,) + v.shape, dt)
            arr["target"] = _alloc(out_dir, "target", (n, horizon_hours), np.dtype(np.float32))
            arr["observed"] = _alloc(out_dir, "observed", (n, horizon_hours), np.dtype(bool))
        for k in INPUTS:
            arr[k][i] = f[k]
        y = load[int(b), int(o):int(o) + horizon_hours].astype(np.float32)
        if len(y) != horizon_hours:
            raise ValueError(f"row {(int(b), int(o))}: the label window runs past the record")
        ob = np.isfinite(y) & (y >= 0)
        arr["target"][i] = np.where(ob, y, np.nan)
        arr["observed"][i] = ob
    for a in arr.values():
        if isinstance(a, np.memmap):
            a.flush()
    return arr


def _alloc(out_dir, key, shape, dtype):
    if out_dir is None:
        return np.zeros(shape, dtype)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(Path(out_dir) / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)


class RowConcat:
    """Rows of ``first`` followed by rows of ``second`` without copying them (e.g. a population's built windows followed
    by its extra span windows); supports integer and integer-array indexing along the first axis."""

    def __init__(self, first, second):
        self.first, self.second, self.n = first, second, int(len(first))
        self.shape = (self.n + int(len(second)),) + tuple(np.shape(first)[1:])
        self.dtype = np.asarray(first[:0]).dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            i = int(i) + (len(self) if int(i) < 0 else 0)
            if not 0 <= i < len(self):
                raise IndexError(i)
            return self.first[i] if i < self.n else self.second[i - self.n]
        idx = np.arange(len(self))[i] if isinstance(i, slice) else np.asarray(i)
        idx = np.flatnonzero(idx) if idx.dtype == bool else idx.astype(np.int64).reshape(-1)
        out = np.empty((len(idx),) + self.shape[1:], self.dtype)
        a = idx < self.n
        if a.any():
            out[a] = np.asarray(self.first[idx[a]])
        if (~a).any():
            out[~a] = np.asarray(self.second[idx[~a] - self.n])
        return out


# ----------------------------------------------------------------------------- sources and pool
@dataclass
class RowSet:
    """Windows of one population on one origin grid: unit, origin and split per row, and (optionally) their arrays."""
    unit: np.ndarray
    origin: np.ndarray
    split: np.ndarray
    arrays: Mapping[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return int(len(self.unit))


@dataclass
class ValidationSet:
    """Rows ``index`` of ``arrays`` with the unit (and, when known, the origin) of every row of ``arrays`` (the
    unit-equal validation criterion averages within units first). ``arrays`` is None when the source was prepared
    without arrays (``build_arrays=False``)."""
    arrays: Mapping[str, np.ndarray] | None
    index: np.ndarray
    unit: np.ndarray
    origin: np.ndarray | None = None


@dataclass
class SpanPlan:
    """The target's own span windows returned to its training pool (:class:`ReturnSpan`).

    chosen        indices into the source's ``rows`` of the built span windows selected for return (ascending)
    extra         further span windows that were not among the built rows (unit-major, origins ascending), with their
                  arrays when built; in the training pool row ``len(rows) + i`` is ``extra`` row i
    returned      the row indices appended to the target's own training pool, in pool order
    validation    with a hold-out rule: the held-out span windows (the early-stopping set); otherwise None
    holdout_rows  with a hold-out rule: {"train_rows", "train_extra", "val_rows", "val_extra"} (indices into ``rows``
                  and into ``extra``); otherwise None
    info          counts, density and boundaries of the plan"""
    chosen: np.ndarray
    extra: RowSet
    returned: np.ndarray
    validation: ValidationSet | None
    holdout_rows: dict | None
    info: dict


@dataclass
class Source:
    """One population prepared for training.

    rows           daily-grid windows (training and validation split) with arrays: what the population contributes to
                   any pool (its own or another target's)
    validation     its weekly-grid validation windows: the early-stopping set when it is the target (with a hold-out
                   rule: the held-out span windows instead)
    count_origins  origins of its count-reference windows (weekly grid, both splits), or None (no count matching)
    start_hour     absolute hour of the record's first hour; boundary: its own validation start v0 (record hour)
    seed_index     index used in the per-population seeds
    span           the returned span (:class:`SpanPlan`) when prepared as a target with a :class:`ReturnSpan` rule"""
    name: str
    start_hour: int
    cut: int
    boundary: int
    rows: RowSet
    validation: ValidationSet | None
    count_origins: np.ndarray | None
    seed_index: int = 0
    info: dict = field(default_factory=dict)
    span: SpanPlan | None = None


def _grid_rows(load, cut, grid, config, boundary, seed_index):
    w = make_windows(load, cut, grid_hours=grid, context_hours=config.context_hours, horizon_hours=config.horizon_hours,
                     min_observed=config.min_observed_fraction)
    lab, _ = split(w, cut, config, boundary=boundary)
    rows, sp = select_rows(w, lab, train_cap=config.train_cap, val_cap=config.val_cap,
                           rng=np.random.default_rng(config.row_seed(grid) + int(seed_index)))
    return rows, sp, {"train": int((lab == "train").sum()), "val": int((lab == "val").sum())}


def prepare_source(name: str, data: Mapping, cut: int, config: TrainConfig = TrainConfig(), *, feature_fn: Callable | None = None,
                   start_hour: int | None = None, seed_index: int = 0, out_dir: str | Path | None = None,
                   with_validation: bool = True, build_arrays: bool = True) -> Source:
    """Windows, split, caps and arrays of one population at its training cut (recipe steps 1-4).

    ``start_hour`` defaults to :func:`absolute_start_hour` of ``data["start_iso"]``. ``with_validation=False`` skips the
    validation arrays (a population that is only ever borrowed from). ``out_dir`` stores the arrays as memory maps.
    ``build_arrays=False`` computes the windows only (no features; for planning and inspection, not for training).
    With a :class:`ReturnSpan` rule and ``with_validation``, the returned span is planned as well (``Source.span``,
    :func:`plan_span`); with its hold-out rule, ``Source.validation`` is the held-out span."""
    cut = int(cut)
    d = cut_data(data, cut)
    start = absolute_start_hour(d["start_iso"]) if start_hour is None else int(start_hour)
    tg, vg, cg = config.train_origin_grid_hours, config.val_origin_grid_hours, config.pool_count_grid_hours
    w = make_windows(d["load"], cut, grid_hours=tg, context_hours=config.context_hours, horizon_hours=config.horizon_hours,
                     min_observed=config.min_observed_fraction)
    first = int(w[:, 1].min()) if len(w) else int(config.context_hours)
    v0 = config.validation.boundary(cut, first, config.context_hours, config.horizon_hours)
    rows, sp, cand = _grid_rows(d["load"], cut, tg, config, v0, seed_index)
    sub = (lambda k: None) if out_dir is None else (lambda k: Path(out_dir) / k)
    arrays = materialise(d, rows, feature_fn, horizon_hours=config.horizon_hours, token_dtype=config.token_dtype,
                         out_dir=sub("rows")) if build_arrays else None
    rs = RowSet(unit=rows[:, 0].copy(), origin=rows[:, 1].copy(), split=sp, arrays=arrays)
    info = {"candidates": cand, "validation_days": config.validation.validation_days(cut, first), "first_origin": first}
    span_rule = isinstance(config.validation, ReturnSpan) and with_validation
    weekly_val = with_validation and not (span_rule and config.validation.holdout is not None)
    needed = ({vg} if weekly_val else set()) | ({cg} if cg is not None else set())
    other = {g: _grid_rows(d["load"], cut, g, config, v0, seed_index)[:2] for g in sorted(needed - {tg})}   # grid -> (rows, split)
    val = None
    if weekly_val:
        if vg == tg:
            val = ValidationSet(arrays=arrays, index=np.flatnonzero(sp == "val"), unit=rs.unit, origin=rs.origin)
        else:
            vrows, vsp = other[vg]
            vr = vrows[vsp == "val"]
            va = materialise(d, vr, feature_fn, horizon_hours=config.horizon_hours, token_dtype=config.token_dtype,
                             out_dir=sub("validation")) if build_arrays else None
            val = ValidationSet(arrays=va, index=np.arange(len(vr)), unit=vr[:, 0].copy(), origin=vr[:, 1].copy())
    count = None if cg is None else (rs.origin.copy() if cg == tg else other[cg][0][:, 1].copy())
    src = Source(name=name, start_hour=start, cut=cut, boundary=int(v0), rows=rs, validation=val, count_origins=count,
                 seed_index=int(seed_index), info=info)
    if span_rule:
        src.span = plan_span(src, config, windows=w, data=d, feature_fn=feature_fn, build_arrays=build_arrays,
                             out_dir=sub("span_extra"))
        if config.validation.holdout is not None:
            src.validation = src.span.validation
        info["span"] = src.span.info
    info.update(rows={"train": int((sp == "train").sum()), "val": int((sp == "val").sum())},
                validation_rows=None if src.validation is None else int(len(src.validation.index)),
                count_rows=None if count is None else int(len(count)))
    return src


def plan_span(source: Source, config: TrainConfig, *, windows: np.ndarray | None = None, data: Mapping | None = None,
              feature_fn: Callable | None = None, build_arrays: bool = True, out_dir: str | Path | None = None) -> SpanPlan:
    """The returned span of a target population (``config.validation`` must be a :class:`ReturnSpan`).

    ``source`` is the population prepared with that rule (its ``rows``, ``info["candidates"]``, ``count_origins``).
    ``windows`` (all eligible training-grid windows, :func:`make_windows` on the blanked store) and ``data`` (the store
    blanked at the cut, :func:`cut_data`) are needed only when the built span windows are fewer than the density
    requires: then the extra windows are chosen from ``windows`` and, with ``build_arrays``, built from ``data``."""
    rule = config.validation
    if not isinstance(rule, ReturnSpan):
        raise TypeError("plan_span needs a ReturnSpan validation rule")
    H, cut, v0, bh = int(config.horizon_hours), int(source.cut), int(source.boundary), int(rule.boundary_hours)
    u, o, sp = source.rows.unit, source.rows.origin, source.rows.split
    n_rows = len(source.rows)
    cand = source.info["candidates"]
    own = _base_pool(source.name, [source], config)[0][source.name]
    n_own = int(len(own))
    rate = n_own / cand["train"]
    n_target = int(round(rate * cand["val"]))
    ok = np.flatnonzero((sp == "val") & (o + H <= cut - bh))
    rng = np.random.default_rng(int(rule.seed) + int(source.seed_index))
    extra = np.zeros((0, 2), np.int64)
    if n_target <= len(ok):
        chosen = np.sort(balanced(np.stack([u[ok], ok], 1), n_target, rng)[:, 1]) if n_target < len(ok) else ok
    else:
        chosen, need = ok, n_target - len(ok)
        if windows is None:
            raise ValueError(f"{source.name}: {need} extra span windows are needed; pass the eligible training-grid windows")
        w = np.asarray(windows, np.int64).reshape(-1, 2)
        is_val = sp == "val"
        have = set(zip(u[is_val].tolist(), o[is_val].tolist()))
        m = (w[:, 1] >= v0) & (w[:, 1] + H <= cut - bh)
        cl = np.asarray([x for x in map(tuple, w[m].tolist()) if x not in have], np.int64).reshape(-1, 2)
        extra = balanced(cl, need, rng) if len(cl) > need else cl
        extra = extra[np.lexsort((extra[:, 1], extra[:, 0]))]
    ex_arrays = None
    if len(extra) and build_arrays:
        if data is None:
            raise ValueError(f"{source.name}: extra span windows must be built; pass the store blanked at the cut")
        ex_arrays = materialise(data, extra, feature_fn, horizon_hours=H, token_dtype=config.token_dtype, out_dir=out_dir)
    ex = RowSet(unit=extra[:, 0].copy(), origin=extra[:, 1].copy(), split=np.full(len(extra), "val", dtype="<U5"), arrays=ex_arrays)
    info = {"own_pool_rows": n_own, "own_train_candidates": int(cand["train"]), "span_candidates": int(cand["val"]),
            "sampling_rate": rate, "span_rows_target": n_target, "span_rows_from_built_rows": int(len(chosen)),
            "span_rows_extra": int(len(extra)), "built_span_rows_within_boundary": int(len(ok)), "boundary_hours": bh,
            "v0": v0, "cut": cut}
    hold = rule.holdout
    if hold is None:
        returned = np.concatenate([chosen, n_rows + np.arange(len(extra), dtype=np.int64)]).astype(np.int64)
        return SpanPlan(chosen=chosen, extra=ex, returned=returned, validation=None, holdout_rows=None, info=info)
    first = int(o.min())                                                  # the record is measured from the first built window
    vd = hold.validation_days(cut, first)
    b = hold.boundary(cut, first, config.context_hours, H)
    end = cut - int(hold.end_gap_hours)
    tr_rows = chosen[o[chosen] + H <= b]
    tr_ex = np.flatnonzero(extra[:, 1] + H <= b)
    va_rows = np.flatnonzero((sp == "val") & (o >= b) & (o + H <= end))
    va_ex = np.flatnonzero((extra[:, 1] >= b) & (extra[:, 1] + H <= end))
    returned = np.concatenate([tr_rows, n_rows + tr_ex]).astype(np.int64)
    vunit = np.concatenate([u[va_rows], extra[va_ex, 0]]).astype(u.dtype)
    vorigin = np.concatenate([o[va_rows], extra[va_ex, 1]]).astype(o.dtype)
    varr = None
    if source.rows.arrays is not None and (ex_arrays is not None or not len(va_ex)):
        varr = {k: np.concatenate([np.asarray(source.rows.arrays[k][va_rows])] +
                                  ([np.asarray(ex_arrays[k][va_ex])] if len(va_ex) else [])) for k in ARRAY_KEYS}
    if not len(vunit):
        raise ValueError(f"{source.name}: the hold-out rule leaves no validation windows")
    info.update(record_days=(cut - first) / 24.0, holdout_validation_days=vd, holdout_boundary=int(b),
                train_rows_returned=int(len(returned)), validation_rows=int(len(vunit)))
    return SpanPlan(chosen=chosen, extra=ex, returned=returned,
                    validation=ValidationSet(arrays=varr, index=np.arange(len(vunit)), unit=vunit, origin=vorigin),
                    holdout_rows={"train_rows": tr_rows, "train_extra": tr_ex, "val_rows": va_rows, "val_extra": va_ex}, info=info)


@dataclass
class Pool:
    """Training rows for one target: {population: row indices into that population's ``rows``} (dict order =
    population order, which the sampler uses), the absolute time limit, and per population (reference count, rows).

    The rows of each population are ascending; with a returned span the target's rows are followed by the returned
    ``returned[target]`` row indices (rows ``len(rows) + i`` are ``extra[target]`` row i)."""
    target: str
    limit: int
    rows: dict[str, np.ndarray]
    table: dict[str, tuple[int, int]]
    extra: dict[str, RowSet] = field(default_factory=dict)
    returned: dict[str, int] = field(default_factory=dict)


def _base_pool(target: str, srcs: Sequence[Source], config: TrainConfig):
    P = next(s for s in srcs if s.name == target)
    H, limit = config.horizon_hours, int(P.start_hour) + int(P.boundary)
    rows: dict[str, np.ndarray] = {}
    table: dict[str, tuple[int, int]] = {}
    for Q in srcs:
        ix = np.flatnonzero(int(Q.start_hour) + Q.rows.origin + H <= limit)
        if config.pool_count_grid_hours is None:
            n = len(ix)
        else:
            if Q.count_origins is None:
                raise ValueError(f"{Q.name}: count matching needs count_origins")
            n = int((int(Q.start_hour) + np.asarray(Q.count_origins) + H <= limit).sum())
            if n == 0:                                                         # no reference window before the limit
                continue
        if not len(ix):
            table[Q.name] = (n, 0)
            continue
        if len(ix) > n:
            rng = np.random.default_rng(config.pool_seed + int(Q.seed_index))
            ix = balanced(np.stack([Q.rows.unit[ix], ix], 1), n, rng)[:, 1]
        rows[Q.name] = np.sort(ix)
        table[Q.name] = (n, int(len(ix)))
    if target not in rows:
        raise ValueError(f"{target}: no own training windows")
    if not set(rows[target].tolist()) <= set(np.flatnonzero(P.rows.split == "train").tolist()):
        raise AssertionError(f"{target}: own pool must consist of training-split windows")
    return rows, limit, table, P


def build_pool(target: str, sources: Mapping[str, Source] | Sequence[Source], config: TrainConfig = TrainConfig()) -> Pool:
    """Recipe step 5: every population's windows whose labels end, in absolute time, at or before the target's
    validation start; per population count-matched to ``config.pool_count_grid_hours`` (a unit-balanced subsample of
    the population's capped daily-grid rows, ``Source.rows``).
    With a :class:`ReturnSpan` rule the target's returned span (``Source.span``) is appended to its own rows."""
    srcs = list(sources.values()) if isinstance(sources, Mapping) else list(sources)
    rows, limit, table, P = _base_pool(target, srcs, config)
    pool = Pool(target=target, limit=limit, rows=rows, table=table)
    if isinstance(config.validation, ReturnSpan):
        if P.span is None:
            raise ValueError(f"{target}: no returned span; prepare the target with this ReturnSpan rule (with_validation=True)")
        rows[target] = np.concatenate([rows[target], P.span.returned]).astype(np.int64)
        pool.returned[target] = int(len(P.span.returned))
        if len(P.span.extra):
            pool.extra[target] = P.span.extra
    return pool


def training_arrays(pool: Pool, sources: Mapping[str, Source]) -> tuple[dict, dict]:
    """({population: arrays}, {population: unit of every row}) for the sampler and :func:`make_batch`; a population
    with extra span windows gets its built rows followed by the extra rows (:class:`RowConcat`, no copy)."""
    arrays = {q: sources[q].rows.arrays for q in pool.rows}
    units = {q: sources[q].rows.unit for q in pool.rows}
    for q, ex in pool.extra.items():
        if not len(ex):
            continue
        if arrays[q] is None or ex.arrays is None:
            raise ValueError(f"{q}: arrays were not built (prepare_source(..., build_arrays=True))")
        arrays[q] = {k: RowConcat(arrays[q][k], ex.arrays[k]) for k in ARRAY_KEYS}
        units[q] = np.concatenate([np.asarray(units[q]), np.asarray(ex.unit, np.asarray(units[q]).dtype)])
    for q, a in arrays.items():
        if a is None:
            raise ValueError(f"{q}: arrays were not built (prepare_source(..., build_arrays=True))")
    return arrays, units


# ----------------------------------------------------------------------------- sampler and batches
def sampler(target: str, pool: Pool | Mapping[str, np.ndarray], units: Mapping[str, np.ndarray], rng: np.random.Generator, *,
            draws: int = 16000, own_fraction: float = 0.5) -> list[tuple[str, int]]:
    """One epoch of (population, row) draws with replacement (recipe step 6).

    ``own_fraction`` of the draws (floor) come from the target, each of its units with equal total weight; the rest pick a
    borrowed population uniformly per draw, then a row with each of that population's units at equal total weight.
    ``units[name]`` is the unit of every row of that population. Without borrowed populations all draws are own."""
    pl = pool.rows if isinstance(pool, Pool) else pool
    own = pl[target]
    bP = units[target][own]
    cP = {u: n for u, n in zip(*np.unique(bP, return_counts=True))}
    wP = np.array([1.0 / cP[u] for u in bP])
    wP /= wP.sum()
    oth = []
    for Q in [q for q in pl if q != target]:
        bQ = units[Q][pl[Q]]
        cQ = {u: n for u, n in zip(*np.unique(bQ, return_counts=True))}
        w = np.array([1.0 / cQ[u] for u in bQ])
        oth.append((Q, pl[Q], w / w.sum()))
    n_own = draws if not oth else int(math.floor(draws * own_fraction))
    out = [(target, r) for r in rng.choice(own, n_own, replace=True, p=wP)]
    if oth:
        qs = rng.integers(0, len(oth), draws - n_own)
        for j, cnt in zip(*np.unique(qs, return_counts=True)):
            Q, ix, w = oth[j]
            out += [(Q, r) for r in rng.choice(ix, cnt, replace=True, p=w)]
    return [out[i] for i in rng.permutation(len(out))]


def make_batch(arrays: Mapping[str, Mapping[str, np.ndarray]], pairs: Sequence[tuple[str, int]], device="cpu") -> dict:
    """Stack the rows ``pairs`` = [(population, row), ...] of ``arrays[population][key]`` into tensors on ``device``
    (float16 tokens are cast to float32), in the given order."""
    import torch
    out = {k: [] for k in ARRAY_KEYS}
    for Q, r in pairs:
        for k in out:
            v = np.asarray(arrays[Q][k][int(r)])
            out[k].append(v.astype(np.float32) if v.dtype == np.float16 else v)
    return {k: torch.from_numpy(np.ascontiguousarray(np.stack(v))).to(device) for k, v in out.items()}


# ----------------------------------------------------------------------------- loss weighting (stage 2 of recipe Rw)
def _noise_sigma2_block(block, rule: NoiseWeights) -> np.ndarray:
    """sigma2 per row of a token block (rows, days ``rule.context_days``, columns 0..47); NaN below ``min_pairs``."""
    t = np.asarray(block, dtype=np.float32)
    x, ob = np.sinh(t[:, :, :24]).astype(np.float64), t[:, :, 24:48] > rule.observed_threshold
    k = int(rule.lag_days)
    d, m = x[:, k:] - x[:, :-k], ob[:, k:] & ob[:, :-k]
    n = m.sum((1, 2))
    return np.where(n >= rule.min_pairs, ((d ** 2) * m).sum((1, 2)) / np.maximum(n, 1), np.nan)


def noise_sigma2(day_tokens, rule: NoiseWeights = NoiseWeights()) -> np.ndarray:
    """Noise estimate of each window from its own day tokens ((n, 366, >= 48), as stored or as a batch): the mean of
    (x_t - x_{t-168 h})^2, x = sinh(token[:24]) = load / scale, over the hours of days 345..365 paired with the same
    hour 7 days earlier, both observed; NaN with fewer than ``rule.min_pairs`` pairs. Only the context before the
    origin enters (the day tokens end at the origin)."""
    d0, d1 = rule.context_days
    return _noise_sigma2_block(np.asarray(day_tokens[:, d0:d1, :48]), rule)


def _token_block(a, rows: np.ndarray, rule: NoiseWeights) -> np.ndarray:
    """Days ``rule.context_days``, columns 0..47 of the day tokens of ``rows`` (ascending), read from an array, a memory
    map or a :class:`RowConcat` (only this block is read, never the full rows)."""
    rows = np.asarray(rows, np.int64)
    d0, d1 = rule.context_days
    if isinstance(a, RowConcat):
        m = rows < a.n
        out = np.zeros((len(rows), d1 - d0, 48), a.dtype)
        if m.any():
            out[m] = _token_block(a.first, rows[m], rule)
        if (~m).any():
            out[~m] = _token_block(a.second, rows[~m] - a.n, rule)
        return out
    return np.asarray(a[rows, d0:d1, :48])


def pool_sigma2(arrays: Mapping[str, np.ndarray], rows: np.ndarray, rule: NoiseWeights = NoiseWeights(), *,
                chunk: int = NOISE_CHUNK_ROWS) -> np.ndarray:
    """:func:`noise_sigma2` of the rows ``rows`` of one population's arrays (``arrays["day_tokens"]``), returned in the
    order of ``rows``; read in chunks of ``chunk`` ascending rows (the value of a row does not depend on the chunk)."""
    ix = np.asarray(rows, np.int64)
    v = np.empty(len(ix))
    o = np.argsort(ix, kind="stable")
    s = ix[o]
    for c0 in range(0, len(s), int(chunk)):
        v[o[c0:c0 + chunk]] = _noise_sigma2_block(_token_block(arrays["day_tokens"], s[c0:c0 + chunk], rule), rule)
    return v


def draw_probabilities(target: str, pool: Pool | Mapping[str, np.ndarray], units: Mapping[str, np.ndarray], *,
                       draws: int = 16000, own_fraction: float = 0.5) -> dict[str, np.ndarray]:
    """{population: probability of each pool row per draw} of :func:`sampler`: the target's share of the draws
    (floor(draws x own_fraction) / draws; all draws without borrowed populations), the rest split equally across the
    borrowed populations, and within each population every unit at equal total weight."""
    pl = pool.rows if isinstance(pool, Pool) else pool
    others = [q for q in pl if q != target]
    n_own = draws if not others else int(math.floor(draws * own_fraction))
    p = {}
    for q, ix in pl.items():
        b = np.asarray(units[q])[ix]
        uq, cnt = np.unique(b, return_counts=True)
        cu = dict(zip(uq.tolist(), cnt.tolist()))
        pu = np.array([1.0 / cu[u] for u in b.tolist()])
        pu /= pu.sum()
        p[q] = (n_own / draws if q == target else (draws - n_own) / draws / len(others)) * pu
    return p


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The ``q`` quantile of ``values`` under ``weights``: the smallest value whose cumulative weight share (values in
    ascending order, ties in input order) reaches ``q``."""
    v, p = np.asarray(values), np.asarray(weights)
    o = np.argsort(v, kind="stable")
    c = np.cumsum(p[o]) / p.sum()
    return float(v[o][min(int(np.searchsorted(c, q)), len(v) - 1)])


def noise_weights(target: str, pool: Pool | Mapping[str, np.ndarray], arrays: Mapping[str, Mapping[str, np.ndarray]],
                  units: Mapping[str, np.ndarray], rule: NoiseWeights = NoiseWeights(), *, draws: int = 16000,
                  own_fraction: float = 0.5, chunk: int = NOISE_CHUNK_ROWS) -> tuple[dict[str, np.ndarray], dict]:
    """The loss weights of a training pool (module docstring, "Loss weighting").

    ``arrays`` and ``units`` are those of :func:`training_arrays` (returned span rows included); ``draws`` and
    ``own_fraction`` are the sampler's. Returns ({population: weight of every row of its arrays, NaN outside the pool},
    record) with the record holding the floor, the weight of windows without a noise estimate, per population the
    normaliser c_Q, draw share and rows, and the Kish effective sample size ratio under the draw probabilities.
    Deterministic (no random stream is used); raises ValueError when a pool row gets no finite positive weight."""
    pl = pool.rows if isinstance(pool, Pool) else pool
    s2 = {q: pool_sigma2(arrays[q], ix, rule, chunk=chunk) if len(ix) else np.zeros(0) for q, ix in pl.items()}
    p = draw_probabilities(target, pl, units, draws=draws, own_fraction=own_fraction)
    S = np.concatenate([s2[q] for q in pl])
    Pp = np.concatenate([p[q] for q in pl])
    ok = np.isfinite(S)
    if not ok.any():
        raise ValueError(f"{target}: no pool window has a noise estimate (fewer than {rule.min_pairs} observed pairs)")
    floor = weighted_quantile(S[ok], Pp[ok], rule.floor_quantile)
    om = np.full(len(S), np.nan)
    om[ok] = np.maximum(S[ok], floor) ** -rule.gamma
    neutral = weighted_quantile(om[ok], Pp[ok], 0.5)
    om[~ok] = neutral
    W, k, src = {}, 0, {}
    for q, ix in pl.items():
        o_, pq = om[k:k + len(ix)], Pp[k:k + len(ix)]
        c = pq.sum() / (pq * o_).sum()
        w = np.full(len(units[q]), np.nan)
        w[ix] = c * o_
        W[q] = w
        src[q] = {"normaliser": float(c), "draw_share": float(pq.sum()), "rows": int(len(ix))}
        k += len(ix)
    Wp = np.concatenate([W[q][ix] for q, ix in pl.items()])
    if not (np.isfinite(Wp).all() and (Wp > 0).all()):
        raise ValueError(f"{target}: a pool row has no finite positive weight")
    rec = {"target": target, "rule": {"rule": type(rule).__name__, **asdict(rule)}, "pool": {q: int(len(ix)) for q, ix in pl.items()},
           "draws": int(draws), "own_fraction": float(own_fraction), "gamma": rule.gamma, "floor_quantile": rule.floor_quantile,
           "floor_sigma2": floor, "neutral_raw_weight": neutral, "undefined_sigma2_draw_share": float(Pp[~ok].sum()),
           "sources": src, "kish_ess_ratio_draws": float((Pp * Wp).sum() ** 2 / (Pp * Wp ** 2).sum()),
           "weight_max_over_median": float(np.nanmax(Wp) / weighted_quantile(Wp, Pp, 0.5))}
    return W, rec


def expected_weight(day_tokens, populations: Sequence[str], record: Mapping, rule: NoiseWeights = NoiseWeights()) -> np.ndarray:
    """The weight of each window of a batch implied by its own day tokens and the weights ``record`` of
    :func:`noise_weights` (``populations``: the population of each row); the row-alignment check of the first batch of
    every weighted epoch."""
    s2 = noise_sigma2(day_tokens, rule)
    om = np.where(np.isfinite(s2), np.maximum(s2, record["floor_sigma2"]) ** -rule.gamma, record["neutral_raw_weight"])
    return np.array([record["sources"][q]["normaliser"] for q in populations]) * om


def weighted_log1p_loss(per_window, weights):
    """Self-normalised weighted batch loss sum_i w_i l_i / sum_i w_i, l_i = log1p(clamp_0(per_window_i)) (the
    log1p-compressed per-window loss of ``forecast_loss``; with every w_i = 1 it is the unweighted batch mean)."""
    ell = per_window.clamp_min(0).log1p()
    return (weights * ell).sum() / weights.sum()


# ----------------------------------------------------------------------------- optimisation
def configure(seed: int, config: TrainConfig = TrainConfig()) -> None:
    """Seed every generator (Python, NumPy global, torch, CUDA) and select deterministic kernels."""
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if config.torch_threads is not None:
        torch.set_num_threads(int(config.torch_threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(model, config: TrainConfig = TrainConfig()):
    """AdamW over all parameters (one group: learning rate and weight decay of the recipe). A continuation starts
    with a fresh optimiser (no optimiser state is carried over)."""
    import torch
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def train_step(model, optimizer, batch: Mapping, groups: Sequence[Sequence], config: TrainConfig = TrainConfig(), loss_fn: Callable | None = None,
               weights: np.ndarray | None = None):
    """One optimisation step on one batch; returns the (detached) loss. Raises FloatingPointError on a non-finite loss
    and RuntimeError on a non-finite gradient norm. Each non-empty parameter group is clipped separately.

    ``weights`` (one positive weight per batch row) switches to the weighted loss :func:`weighted_log1p_loss` of the
    per-window loss (requires ``config.log1p_compression``); None is the unweighted loss of every other recipe."""
    import torch
    if loss_fn is None:
        from ankyra_lite.carrier.model import forecast_loss as loss_fn
    if weights is not None and not config.log1p_compression:
        raise ValueError("the loss weighting weights the log1p-compressed per-window loss; set log1p_compression=True")
    out = model({k: batch[k] for k in INPUTS})
    if weights is None:
        loss = loss_fn(out, torch.where(batch["observed"], batch["target"], 0.0), batch["scale"], batch["observed"],
                       kl_weight=config.kl_weight, compression=config.log1p_compression)["loss"]
    else:
        per_window = loss_fn(out, torch.where(batch["observed"], batch["target"], 0.0), batch["scale"], batch["observed"],
                             kl_weight=config.kl_weight, compression=config.log1p_compression)["per_window"]
        loss = weighted_log1p_loss(per_window, torch.as_tensor(weights, dtype=per_window.dtype, device=per_window.device))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for g in groups:
        if g:
            torch.nn.utils.clip_grad_norm_(g, config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def run_epoch(model, optimizer, order: Sequence[tuple[str, int]], arrays: Mapping, groups: Sequence[Sequence],
              config: TrainConfig = TrainConfig(), device="cpu", loss_fn: Callable | None = None,
              weights: Mapping[str, np.ndarray] | None = None, weights_record: Mapping | None = None) -> tuple[float, int]:
    """One pass over the draws ``order`` in batches of ``config.batch_size`` in training mode; returns (sum over
    batches of loss x batch size, number of draws).

    ``weights`` ({population: weight of every row}, :func:`noise_weights`) switches to the weighted loss; every drawn
    row must have a finite positive weight. With ``weights_record`` the first batch is checked against the weights
    implied by its own day tokens (:func:`expected_weight`, relative tolerance 1e-9), which catches misaligned rows."""
    model.train()
    total, seen = 0.0, 0
    for s0 in range(0, len(order), config.batch_size):
        part = order[s0:s0 + config.batch_size]
        if weights is None:
            loss = train_step(model, optimizer, make_batch(arrays, part, device), groups, config, loss_fn)
        else:
            batch = make_batch(arrays, part, device)
            w = np.array([weights[q][int(r)] for q, r in part], dtype=np.float64)
            if not (np.isfinite(w).all() and (w > 0).all()):
                raise ValueError("a drawn window has no finite positive weight (not a row of the weighted pool?)")
            if weights_record is not None and s0 == 0:
                ew = expected_weight(batch["day_tokens"].detach().cpu().numpy(), [q for q, _ in part], weights_record,
                                     config.noise_weighting or NoiseWeights())
                if not np.allclose(ew, w, rtol=1e-9, atol=0):
                    raise RuntimeError("the weights do not belong to the drawn rows (row-alignment check of the first batch)")
            loss = train_step(model, optimizer, batch, groups, config, loss_fn, weights=w)
        total += float(loss) * len(part); seen += len(part)
    return total, seen


def validate(model, arrays: Mapping[str, np.ndarray], index: np.ndarray, unit: np.ndarray, device="cpu", batch_size: int = 64) -> float:
    """Validation criterion (lower is better): per window the mean over its observed label hours of ((F - y) / s)^2,
    the squared forecast error divided by the squared scale s of the window (its ``scale`` input); per unit the mean
    over its windows; the criterion is the mean over units of log1p(unit mean). ``unit`` gives the unit of every row of
    ``arrays``; ``index`` selects the validation rows."""
    import torch
    idx = np.sort(np.asarray(index, np.int64))
    if not len(idx):
        raise ValueError("empty validation set")
    model.eval()
    per = np.zeros(len(idx))
    with torch.inference_mode():
        for s in range(0, len(idx), batch_size):
            ii = idx[s:s + batch_size]
            b = {}
            for k in ARRAY_KEYS:
                v = np.asarray(arrays[k][ii])
                b[k] = torch.from_numpy(np.ascontiguousarray(v.astype(np.float32) if v.dtype == np.float16 else v)).to(device)
            o = model({k: b[k] for k in INPUTS})
            m = b["observed"]
            sc = b["scale"][:, None]
            e = torch.where(m, (o["trajectory_kw"] - torch.where(m, b["target"], 0)) / sc, 0)
            per[s:s + len(ii)] = (e.square().sum(-1) / m.sum(-1).clamp_min(1)).double().cpu().numpy()
    u = np.asarray(unit)[idx]
    v = float(np.mean([math.log1p(float(per[u == x].mean())) for x in np.unique(u)]))
    if not math.isfinite(v):
        raise FloatingPointError("non-finite validation criterion")
    return v


# ----------------------------------------------------------------------------- start weights
@dataclass
class StartWeights:
    """Weights a continuation starts from: the state dict, the criterion recorded with them (``"val"`` of the
    checkpoint, or None) and where they came from."""
    state: dict
    val: float | None = None
    origin: str = "state dict"


def start_weights(init) -> StartWeights | None:
    """Start weights from a checkpoint path (``{"model": state_dict, "val": ...}`` as written by :func:`train_one`, or a
    bare state dict), a checkpoint mapping, a state dict, a :class:`StartWeights`, or None (training from scratch)."""
    if init is None or isinstance(init, StartWeights):
        return init
    origin = "state dict"
    if isinstance(init, (str, Path)):
        import torch
        origin = str(init)
        init = torch.load(Path(init), map_location="cpu", weights_only=True)
    if isinstance(init, Mapping) and "model" in init and isinstance(init["model"], Mapping):
        val = init.get("val")
        return StartWeights(state=dict(init["model"]), val=None if val is None else float(val), origin=origin)
    if isinstance(init, Mapping):
        return StartWeights(state=dict(init), val=None, origin=origin)
    raise TypeError(f"cannot read start weights from {type(init).__name__}")


def initial_model(seed: int, target: str, config: TrainConfig = TrainConfig(), *, device="cpu", init=None,
                  model_factory: Callable | None = None):
    """Seed everything (:func:`configure`), build the network, and load the start weights: ``init`` when given (see
    :func:`start_weights`), else ``config.init_path(target, seed)``, else none (training from scratch). Returns
    (model on ``device``, :class:`StartWeights` or None)."""
    import torch
    from ankyra_lite.carrier import model as M
    configure(int(seed), config)                                                # the model is initialised right after seeding
    model = (model_factory or M.build_carrier)().to(torch.device(device))
    start = start_weights(init if init is not None else config.init_path(target, int(seed)))
    if start is not None:
        model.load_state_dict(start.state, strict=True)
    return model, start


# ----------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    """One trained seed: the model with its kept weights loaded (eval mode) and the training record.

    With early stopping the kept weights are the best ones (``best_epoch``, ``best_val``; epoch 0 = the start weights
    when they are counted); with ``fixed_epochs`` they are the final ones and ``best_epoch``/``best_val`` are the final
    epoch and its criterion (NaN without a validation set). ``history`` starts with epoch 0 (the start weights'
    criterion) for a continuation; ``epochs_run`` counts training epochs only. ``weights_record`` is the record of the
    loss weights (:func:`noise_weights`; ``{"rule": "given", ...}`` for weights passed in), None without weighting."""
    seed: int
    model: object
    best_epoch: int
    best_val: float
    history: list
    epochs_run: int
    stopped_by_patience: bool
    seconds: float
    parameters: int
    pool_table: dict
    config: dict
    start_val: float | None = None
    start_origin: str | None = None
    returned_rows: int = 0
    weights_record: dict | None = None


def _state(model) -> dict:
    return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}


def _save(checkpoint, state, epoch, val, seed):
    import torch
    ck = Path(checkpoint); ck.parent.mkdir(parents=True, exist_ok=True); tmp = ck.with_name(ck.name + ".part")
    torch.save({"model": state, "epoch": epoch, "val": val, "seed": seed}, tmp); os.replace(tmp, ck)


def train_one(seed: int, target: str, sources: Mapping[str, Source], config: TrainConfig = TrainConfig(), *, pool: Pool | None = None,
              validation: ValidationSet | None = None, device=None, checkpoint: str | Path | None = None,
              log: Callable[[str], None] | None = print, model_factory: Callable | None = None, loss_fn: Callable | None = None,
              init=None, expected_start_val: float | None = None, weights: Mapping[str, np.ndarray] | None = None,
              weights_record: Mapping | None = None) -> TrainResult:
    """Train one seed of the target's carrier (recipe steps 6-8, or a continuation recipe).

    ``sources`` maps population name to :class:`Source`; ``pool`` defaults to :func:`build_pool`, ``validation`` to the
    target's own validation set. Start weights: ``init`` (path, checkpoint mapping, state dict) or
    ``config.init_checkpoint``; none = from scratch. For a continuation the start weights' criterion is computed first
    (epoch 0) and, when ``expected_start_val`` is given or the checkpoint records ``"val"`` and the validation set is
    the target's default one (not a hold-out), it must agree within ``config.start_val_rtol``. The kept weights are
    held in memory and, with ``checkpoint``, also saved there (``{"model": state_dict, "epoch", "val", "seed"}``,
    written atomically: the best weights with early stopping, the final ones with ``fixed_epochs``). ``device``
    defaults to CUDA when available.

    Loss weighting: with ``config.noise_weighting`` the weights are computed on the pool (:func:`noise_weights`, no
    random stream is used); ``weights`` ({population: weight of every row of its training arrays}) with an optional
    ``weights_record`` passes them in instead (:func:`train_ensemble` computes them once per pool; uniform weights
    give the unweighted training bit for bit on the CPU). Neither: the unweighted loss."""
    import torch
    from ankyra_lite.carrier import model as M
    loss_fn = loss_fn or M.forecast_loss
    seed = int(seed)
    fixed = config.fixed_epochs is not None
    if isinstance(config.validation, ReturnSpan) and config.validation.holdout is None and not fixed:
        raise ValueError("a returned span without a hold-out rule leaves no validation set for early stopping; set fixed_epochs")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    pool = pool if pool is not None else build_pool(target, sources, config)
    default_validation = validation is None
    validation = validation if validation is not None else sources[target].validation
    if validation is None and not fixed:
        raise ValueError(f"{target}: no validation set")
    if validation is not None and validation.arrays is None:
        raise ValueError(f"{target}: the validation arrays were not built")
    arrays, units = training_arrays(pool, sources)
    W, wrec = weights, weights_record
    if W is None and config.noise_weighting is not None:
        W, wrec = noise_weights(target, pool, arrays, units, config.noise_weighting, draws=config.draws_per_epoch,
                                own_fraction=config.own_fraction)
    if W is not None:
        _check_pool_weights(target, pool, W, config)
    model, start = initial_model(seed, target, config, device=device, init=init, model_factory=model_factory)
    opt = make_optimizer(model, config)
    groups = M.param_groups(model)                                              # (base, copy pointer), clipped separately
    rng = np.random.default_rng(config.sampler_seed_offset + seed)
    n_ret = int(pool.returned.get(target, 0))
    returned = set(pool.rows[target][len(pool.rows[target]) - n_ret:].tolist()) if n_ret else set()
    best, best_ep, bad, hist, best_state, start_val = math.inf, 0, 0, [], None, None
    t0 = time.monotonic()
    if validation is not None and (start is not None or config.count_start_as_epoch0 or expected_start_val is not None):
        start_val = validate(model, validation.arrays, validation.index, validation.unit, device, config.batch_size)
        expected = expected_start_val
        if expected is None and start is not None and start.val is not None and default_validation and _holdout(config) is None:
            expected = start.val
        rec0 = {"epoch": 0, "val": start_val, "seconds": time.monotonic() - t0}
        if expected is not None:
            rec0["recorded_val"] = float(expected)
            rel = abs(start_val - float(expected)) / max(abs(float(expected)), 1e-300)
            if config.start_val_rtol is not None and rel > config.start_val_rtol:
                raise ValueError(f"{target} seed {seed}: the start weights give {start_val!r} on the validation set, not the "
                                 f"recorded {float(expected)!r} (relative difference {rel:.3g}); wrong start checkpoint or data")
        hist.append(rec0)
        if config.count_start_as_epoch0 and not fixed:
            best, best_ep, best_state = start_val, 0, _state(model)
            if checkpoint is not None:
                _save(checkpoint, best_state, 0, start_val, seed)
        if log:
            log(f"{target} seed {seed}: epoch 0 (start) val {start_val:.5g}")
    n_epochs = int(config.fixed_epochs) if fixed else int(config.max_epochs)
    v = None
    for ep in range(1, n_epochs + 1):
        te = time.monotonic()
        order = sampler(target, pool, units, rng, draws=config.draws_per_epoch, own_fraction=config.own_fraction)
        if W is None:
            total, seen = run_epoch(model, opt, order, arrays, groups, config, device, loss_fn)
        else:
            total, seen = run_epoch(model, opt, order, arrays, groups, config, device, loss_fn, weights=W, weights_record=wrec)
        v = validate(model, validation.arrays, validation.index, validation.unit, device, config.batch_size) if validation is not None else None
        rec = {"epoch": ep, "train_loss": total / max(seen, 1), "val": v, "seconds": time.monotonic() - te}
        if W is not None:
            rec["mean_drawn_weight"] = float(np.mean([W[q][int(r)] for q, r in order]))
        if hasattr(model, "copy_gate"):
            with torch.no_grad():
                rec["copy_gate_weight_norm"] = float(model.copy_gate.weight.norm()); rec["type_bonus"] = float(model.type_bonus)
        if n_ret:
            rec["own_draws_returned"] = int(sum(1 for q, r in order if q == target and int(r) in returned))
        hist.append(rec)
        if fixed:
            if log:
                log(f"{target} seed {seed}: epoch {ep}/{n_epochs} train {rec['train_loss']:.5g}"
                    + ("" if v is None else f" val {v:.5g}") + f" ({time.monotonic() - t0:.0f} s)")
            continue
        if v < best:
            best, best_ep, bad, best_state = v, ep, 0, _state(model)
            if checkpoint is not None:
                _save(checkpoint, best_state, ep, v, seed)
        else:
            bad += 1
        if log:
            log(f"{target} seed {seed}: epoch {ep} train {rec['train_loss']:.5g} val {v:.5g} best {best:.5g}@{best_ep} ({time.monotonic() - t0:.0f} s)")
        if bad >= config.patience and ep >= config.min_epochs:
            break
    if fixed:
        best_state, best_ep, best = _state(model), n_epochs, (math.nan if v is None else v)
        if checkpoint is not None:
            _save(checkpoint, best_state, n_epochs, None if v is None else v, seed)
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(seed=seed, model=model, best_epoch=best_ep, best_val=best, history=hist,
                       epochs_run=sum(1 for h in hist if h["epoch"] >= 1), stopped_by_patience=(not fixed) and bad >= config.patience,
                       seconds=time.monotonic() - t0, parameters=int(M.parameter_count(model)),
                       pool_table={q: list(map(int, t)) for q, t in pool.table.items()}, config=config.describe(),
                       start_val=start_val, start_origin=None if start is None else start.origin, returned_rows=n_ret,
                       weights_record=None if W is None else (dict(wrec) if wrec is not None else
                                                              {"rule": "given", "pool": {q: int(len(ix)) for q, ix in pool.rows.items()}}))


def _check_pool_weights(target: str, pool: Pool, weights: Mapping[str, np.ndarray], config: TrainConfig) -> None:
    """Loss weights must cover every pool row with a finite positive value; the weighting needs the log1p loss."""
    if not config.log1p_compression:
        raise ValueError("the loss weighting weights the log1p-compressed per-window loss; set log1p_compression=True")
    for q, ix in pool.rows.items():
        if q not in weights:
            raise ValueError(f"{target}: no loss weights for population {q}")
        w = np.asarray(weights[q], np.float64)[np.asarray(ix, np.int64)]
        if not (np.isfinite(w).all() and (w > 0).all()):
            raise ValueError(f"{target}: a pool row of {q} has no finite positive loss weight")


def train_ensemble(target: str, sources: Mapping[str, Source], config: TrainConfig = TrainConfig(), *, seeds: Sequence[int] | None = None,
                   device=None, checkpoint_dir: str | Path | None = None, log: Callable[[str], None] | None = print,
                   inits: Mapping[int, object] | None = None, **kwargs) -> list[TrainResult]:
    """Train the target's carrier once per seed (default ``config.seeds``) on one pool; checkpoints go to
    ``checkpoint_dir/seed{s}.pt`` when given. Start weights per seed: ``inits[seed]`` when given, else
    ``config.init_checkpoint`` with the seed filled in. Average the seeds' forecasts with :func:`average_seed_forecasts`.
    With ``config.noise_weighting`` the loss weights are computed once for the pool and shared by the seeds."""
    pool = build_pool(target, sources, config)
    if config.noise_weighting is not None and kwargs.get("weights") is None:
        arrays, units = training_arrays(pool, sources)
        kwargs["weights"], kwargs["weights_record"] = noise_weights(target, pool, arrays, units, config.noise_weighting,
                                                                    draws=config.draws_per_epoch, own_fraction=config.own_fraction)
        del arrays, units
    out = []
    for s in (config.seeds if seeds is None else seeds):
        ck = None if checkpoint_dir is None else Path(checkpoint_dir) / f"seed{int(s)}.pt"
        init = None if inits is None else inits[int(s)]
        out.append(train_one(int(s), target, sources, config, pool=pool, device=device, checkpoint=ck, log=log, init=init, **kwargs))
    return out


def average_seed_forecasts(preds: Sequence[np.ndarray], haves: Sequence[np.ndarray] | None = None):
    """Ensemble of the seeds' carrier forecasts: the float64 mean of the predictions and, when availability masks are
    given, their logical AND (a forecast is used only where every seed has one)."""
    if not len(preds):
        raise ValueError("no forecasts")
    mean = np.mean([np.asarray(p, np.float64) for p in preds], 0)
    if haves is None:
        return mean
    have = np.asarray(haves[0], bool).copy()
    for h in haves[1:]:
        have &= np.asarray(h, bool)
    return mean, have


def with_validation_rule(config: TrainConfig, rule: FixedSpan | ProportionalSpan | ReturnSpan) -> TrainConfig:
    """``config`` with another validation rule (e.g. ``with_validation_rule(TrainConfig(), ProportionalSpan())``)."""
    return replace(config, validation=rule)
