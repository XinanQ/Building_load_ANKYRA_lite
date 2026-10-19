# Method

This document describes ankyra-lite 0.1.0: a small carrier network, trained per population, whose forecasts at the
origin and at six pseudo-origins go into an unchanged copy of the ANKYRA 2.2.1 anchoring layer. Every number below was
checked against the code: `ankyra_lite/carrier/model.py`, `features.py`, `inference.py` and `train.py`,
`ankyra_lite/forecast.py`, and the anchoring layer `ankyra_lite/anchor/` (a byte-identical copy;
[PROVENANCE.md](../ankyra_lite/anchor/PROVENANCE.md)). The exact properties of both parts, with proofs and numerical
checks, are in [theory/PROOFS.md](../theory/PROOFS.md) (P1–P20 for the anchoring layer, C1–C10 for the carrier).

Contents: [Information at the origin](#information-at-the-origin) · [The carrier network](#the-carrier-network)
([inputs](#inputs) · [encoder](#encoder) · [read-out](#read-out) · [energy-budget decoder](#energy-budget-decoder) ·
[copy pointer](#copy-pointer) · [parameters](#parameters)) · [Training](#training) ([windows and split](#windows-and-split) ·
[pooled training set](#pooled-training-set) · [sampler](#sampler) · [loss](#loss) · [optimisation](#optimisation) ·
[early stopping](#stage-1-early-stopping) · [span return](#stage-2-the-validation-span-returned) ·
[seeds](#seeds-and-ensemble) · [released weights](#released-weights) · [other recipes](#other-recipes-in-the-package)) ·
[How the anchoring layer uses the carrier](#how-the-anchoring-layer-uses-the-carrier) ·
[State windows](#state-windows-and-the-seasonal-naive-rule) · [Readouts](#readouts) · [Inference cost](#inference-cost) ·
[Constants](#constants) · [What is not claimed](#what-is-not-claimed)

## Information at the origin

A forecast at origin o is a function of the following, and nothing else:

- load and temperature at hours t < o (the carrier reads at most the last 366 days for its daily tokens and the 24 blocks
  of 744 hours before them; the anchoring layer reads the whole pre-origin record);
- the day types (weekday or public holiday) of the 31 forecast days, treated as known at the origin;
- the unit's category: an input of the carrier, and the selector of the anchoring layer's fixed temperature-signature
  prior;
- the population's fixed temperature-anomaly scale `temp_sigma_std`, fitted on the days before the training cut
  ([DATA.md](DATA.md#the-input-format)), and the daylight-saving rule of the unit's region (analog days only);
- the carrier's weights, fitted on labels that end before the training cut, at least 186 days before the first test
  origin;
- the carrier's forecasts issued at o and at each pseudo-origin q = o − 744k (k = 1 … 6), each built from data before q
  only.

The carrier's weights are fitted once per population and are not changed by a forecast. The anchoring layer estimates
its per-unit quantities (candidate weights, handover weights, within-day trust, temperature signature) at every origin
from the unit's own pre-origin record and its completed pseudo-origins.

Checks in the code: [`tests/test_carrier_unit.py`](../tests/test_carrier_unit.py) (no read at or after the origin, the
pseudo-origin rule), the anchoring layer's own [`tests/anchor/test_no_future_information.py`](../tests/anchor/test_no_future_information.py)
and properties C4 and C10 in [theory](../theory/README.md). These establish input isolation in the implementation. They
do not show that the inputs carry no later information: two calendars announced holidays after some test origins
([DATA.md](DATA.md#calendar-caveat-holidays-announced-after-the-origin)), and providers' gap filling and later revisions
of weather archives lie upstream of the package.

## The carrier network

`ankyra_lite.carrier.model.build_carrier()` returns a `CopyPointerNetwork`: a structured load network (encoder,
attention read-out, energy-budget decoder) with a copy pointer over the unit's own recent days. It forecasts the
744 hours (31 days × 24 hours) after an issue time from the unit's own history and the known calendar. It has no
pretrained weights and no unit-identifier embedding, and it is identical to the research network: parameter names,
construction order and therefore seeded initialisation are the same (`tests/test_carrier_equivalence.py`).

Figure [1](../figures/fig1_architecture.png) shows the inputs, the network and the anchoring layer.

### Inputs

`ankyra_lite.carrier.features.features(load, temp, day_types, group, start_iso, unit, origin)` builds six arrays for one
issue time o. Load and temperature are read only at hours before o; day types are read through o + 744. Day and month
tokens are stored in half precision (float16), as during training, and widened to float32 for the network.

**Scale.** s = max(mean of the observed hours among the last 744, mean of the observed hours among the last 8,760); if s
is not a finite positive number, s = 1. It is the larger of the last-month mean and the last-year mean. Every load
feature is divided by s, and the decoder multiplies its level by s, so the forecast scales with the load (property C7).

**Daily tokens** (366 × 69), the last 366 days before o, oldest first; a day with no observed load hour is masked out
and its token set to zero:

| Features | Content |
|---|---|
| 0–23 | arcsinh(load / s) of the 24 hours (0 where unobserved) |
| 24–47 | hourly observation flags (finite and nonnegative load) |
| 48, 49 | arcsinh(daily mean / s), arcsinh(daily standard deviation / s), over the observed hours |
| 50, 51 | arcsinh(daily mean temperature / 30), temperature coverage (observed hours / 24) |
| 52–59 | day type one-hot: Monday … Sunday, holiday (read at the day's 12th hour; the calendar weekday where unknown) |
| 60, 61 | annual phase of the day: sin and cos of 2π × (day of year) / 365.25 |
| 62 | relative position, from −365/366 (oldest) to 0 (newest) |
| 63 | load coverage (observed hours / 24) |
| 64–68 | category one-hot: Industrial, Office, Public, Residential, Commercial (any other label counts as Public) |

**Monthly tokens** (24 × 16), 24 blocks of 744 hours older than the daily span (together with it, about three years of
history); a block with no observed hour is masked out and set to zero:

| Features | Content |
|---|---|
| 0–3 | arcsinh of the block's mean, standard deviation, 10th and 90th percentile, each / s |
| 4, 5 | arcsinh(mean of workday hours / s) (day types 0–4), arcsinh(mean of weekend and holiday hours / s) (5–7) |
| 6, 7 | share of observed hours at exactly zero load; load coverage (observed hours / 744) |
| 8, 9 | arcsinh(mean temperature / 30); temperature coverage |
| 10, 11 | annual phase of the block's midpoint (sin, cos) |
| 12, 13 | relative position, from −23/24 to 0; a constant 1 |
| 14 | arcsinh(change of the block mean from the previous block / s), where both are usable |
| 15 | validity (1) |

**Future-day tokens** (31 × 16), the calendar of the forecast days, known at the origin: the day type one-hot (8), the
annual phase (2), the relative day position d / 30 (1) and the category one-hot (5).

### Encoder

Width 48. Daily and monthly tokens are encoded separately by the same modules:

1. **Embedding.** A linear map of the token (69 features; monthly tokens are zero-padded to 69), plus a linear map of six
   age features (sin and cos of 2π × age / 7, 31 and 366 days; a monthly token's age is 366 + 31 × its block index from
   the newest), plus a learned resolution embedding added for daily and subtracted for monthly tokens. Masked tokens are
   zero.
2. **Two shared gated causal blocks**, reused along a dilation schedule: 1, 2, 4, …, 128 over the 366 daily tokens
   (eight applications, the two blocks alternating) and 1, 2, 4, 8 over the 24 monthly tokens. A block applies a layer
   normalisation, a depthwise causal convolution with kernel 3 at the given dilation (left padding only, so a token sees
   itself and earlier tokens), a gate tanh(a) · σ(b) from one linear map to 2 × 48, a linear output map and a residual
   connection; masked positions stay zero (C5, C6).
3. **Memory.** A final layer normalisation; the monthly and daily memories are concatenated (390 positions).

### Read-out

The queries are one learned global query and 31 day queries, each a two-layer map (16 → 48 → 48, SiLU) of a future-day
token. One multi-head attention (4 heads, width 48) reads the memory, ignoring masked positions; then
z = LayerNorm(q + attended) and z ← LayerNorm(z + MLP(z)) with a 48 → 96 → 48 SiLU MLP. The global output v summarises
the history; the 31 day outputs z_d carry the calendar of each forecast day.

### Energy-budget decoder

- **Level.** L = s · softplus(h(v)), with h a 48 → 64 → 1 SiLU map: a positive monthly mean in kW.
- **Day logits and shape coefficients.** For each day d, a 96 → 64 → 9 SiLU map of [z_d, v] gives one day logit and 8
  coefficients c_d.
- **Rank-8 hourly dictionary.** The 23 zero-mean directions of the 24 hours are given orthonormal Helmert coordinates H
  (24 × 23, fixed). A learned base dictionary B (23 × 8, initialised with the first eight cosine modes) plus a bounded,
  origin-conditioned change 0.1 · tanh(G v) (G: 48 → 23 × 8) gives the dictionary D = H (B + 0.1 tanh(G v)), re-centred
  over the hours. The hour logits of day d are D c_d: the 31 days share one dictionary per window and differ in their
  coefficients.
- **Budget.** p = softmax of the 31 day logits, q_d = softmax of the 24 hour logits of day d, and

  F(d, h) = 744 · L · p_d · q_d(h),

  so the daily mean is 31 · L · p_d and the trajectory's mean is exactly L in real arithmetic. The forecast is
  nonnegative (C1, C2), and its level, daily path and within-day shape are the P1 blocks of its trajectory.

### Copy pointer

Every forecast day attends over the unit's own **last 56 complete days** (all 24 hours observed; zero-load days
included):

- for each such day j, its hourly shares (load / (24 × daily mean); uniform where the daily mean is at most 10⁻⁶ of the
  scale) and its daily mean, both read back from the day token;
- scores: (W_q z_d) · (W_k m_j) / 4 with 16-dimensional queries and keys (m_j: the encoder's memory of day j), plus a
  learned bonus when day d and day j have the same day type; a softmax over the complete days gives the attention α_d;
- the attended share profile and the attended daily mean enter as zero-mean logs:
  hour logits_d += g_h,d · centred log(Σ_j α_dj shares_j) and day logits_d += g_d,d · centred log(Σ_j α_dj mean_j),
  each log clamped at 10⁻⁴;
- the gates (g_h, g_d) are a linear map of [z_d, v], initialised at zero, and are zero when the unit has no complete day
  among its last 56.

An untrained pointer equals the base decoder, the pointer never changes the level (it acts on shares only), and the
copied profile is a mixture of the unit's own complete days (C3, C9). The direction of the pointer (copy recent own days
into the shape and the daily means) came from an analysis of the errors on the test windows; see
[EVALUATION.md](EVALUATION.md#identity-of-the-evidence).

### Parameters

| Part | Trainable parameters |
|---|---:|
| token and age embeddings, resolution embedding | 3,696 |
| two shared causal blocks | 14,688 |
| memory normalisation | 96 |
| future-day projection | 3,168 |
| global query, attention, query normalisation, read-out MLP, output normalisation | 19,008 |
| level head | 3,201 |
| day decoder (day logit and 8 coefficients) | 6,793 |
| hourly dictionary (base dictionary and generator) | 9,200 |
| copy pointer (query, key, gates, type bonus) | 1,763 |
| **total** | **61,613** |

`CARRIER_PARAMETERS = 61613` in `model.py`; `tests/test_carrier_unit.py` checks it. A checkpoint is about 0.27 MB.

## Training

One carrier is trained per target population P, with three seeds. Every constant of the recipe is a field of
`ankyra_lite.carrier.train.TrainConfig`; `recipe(name, init_checkpoint)` builds the named configurations. The released
recipe of version 0.1.0 has two stages: stage 1 is `recipe("from_scratch")` (= `TrainConfig()`; labelled `P3s` in
`results/recipe_history.csv`, where it is the starting point of every arm) and stage 2 is
`recipe("span_return", <stage-1 checkpoint>)` (labelled `R` there). Stage 1 alone is not the released recipe.

### Windows and split

- A window is (unit, origin o): context [o − 1344, o), labels [o, o + 744).
- Only data before the training cut C′ is used: o + 744 ≤ C′, and load and temperature at and after C′ are set to
  missing before any feature is computed (`cut_data`).
- Training origins lie on a daily grid (every 24 hours from hour 1,344); a window is eligible when at least 75% of its
  context hours and of its label hours are observed (`make_windows`).
- Validation span: the last 93 days before the cut, from v0 = max(1344 + 744, C′ − 2,232) (`FixedSpan`). Training
  windows end at or before v0; validation windows start at or after v0 and end by C′; windows across v0 are dropped.
- Caps: at most 16,000 training and 2,000 validation windows per population, chosen unit-balanced (a randomised round
  robin across units, each unit's windows shuffled), training windows first, from one random stream (`balanced`,
  `select_rows`).
- The validation set used for early stopping is the target's own validation windows on a weekly grid (every 168 hours),
  with the same span, eligibility and caps.

The per-population split, with dates, is in [DATA.md](DATA.md#training-validation-and-test-sets).

### Pooled training set

For the target P, with T = start(P) + v0(P) its validation start in absolute time (hours since 2000-01-01 UTC), every
study source Q (P included; the nine populations and the Suzhou park) contributes from its **capped daily-grid rows**
(the at most 16,000 training and 2,000 validation windows chosen above) those whose labels end at or before T:
start(Q) + o + 744 ≤ T (`build_pool`). For P these are its training windows; other sources' windows are "borrowed" and
come from their own training or validation span. Per source, the number of windows is the number of its capped
weekly-grid windows (the count reference) that pass the same limit; when more of its capped daily-grid rows pass it,
a unit-balanced subsample of those rows is drawn (stream `default_rng(20261215 + the source's index)`; `_base_pool`),
not a subsample of all its eligible daily windows. Which sources enter each pool is listed in
[DATA.md](DATA.md#windows-borrowed-from-other-populations).

### Sampler

Each epoch draws 16,000 windows with replacement (`sampler`):

- half (8,000) from P, each of P's units with equal total weight (a unit's windows share its weight equally);
- half from the borrowed sources: for each draw a source chosen uniformly, then a window with each of that source's
  units at equal total weight;
- the draws are shuffled; without borrowed sources (Cambridge) all 16,000 come from P.

With batches of 64, an epoch is 250 optimisation steps.

### Loss

For a window with causal scale s (the carrier's input scale), observed label hours O, label y_t and forecast F_t, let
f_t = F_t / s and ỹ_t = y_t / s. With MSE = mean over t ∈ O of (f_t − ỹ_t)² and the generalised Kullback–Leibler term
KL = mean over t ∈ O of [ỹ_t log ỹ_t − ỹ_t log f_t − ỹ_t + f_t] (0 log 0 = 0; log f_t comes from the decoder's
log-intensities, so zero-load days need no special case), the training loss of a batch is

  mean over windows of log(1 + MSE + 0.1 · KL)

(`forecast_loss(..., kl_weight=0.1, compression=True)`). Each window has equal weight after averaging its observed
hours, the log1p compression limits the weight of windows with large scaled errors, and missing label hours are never
treated as zero.

### Optimisation

AdamW over all parameters, learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, batch 64. The gradient norm is clipped at 1.0
separately for the base parameters and for the copy-pointer parameters (`param_groups`), so the base network sees the
same clipping as a network without a pointer. Deterministic kernels are requested; the study trained on a GPU, where
training is not bitwise repeatable across runs.

### Stage 1: early stopping

After every epoch the validation criterion is computed on the target's weekly validation windows (`validate`):

- per window: the mean over its observed label hours of ((F_t − y_t) / s)², the squared error divided by the squared
  scale;
- per unit: the mean over its validation windows;
- criterion: the mean over units of log(1 + unit mean). Lower is better.

The weights of the best epoch are kept; training stops after 5 epochs without improvement or after 30 epochs.

### Stage 2: the validation span returned

Starting from the kept stage-1 weights of the same seed:

1. **Check.** The stage-1 criterion of the start weights is recomputed on the same validation windows and must agree
   with the recorded value within a relative 10⁻⁵ (otherwise the checkpoint does not belong to these data).
2. **Returned span** (`ReturnSpan`). The target's own daily-grid windows of the span [v0, C′) whose labels end at least
   24 hours before the cut are added to its training windows, at the population's own sampling density:
   n = round(own pool windows / own training candidates × span candidates), where the candidates are the eligible
   daily-grid windows before the caps. They are a unit-balanced subsample (stream 20261019 + the population's index) of
   the built span windows; where fewer are built (BDG2, both GoiEner populations and, by two windows, EWELD), the
   remainder is a unit-balanced subsample of the other eligible span windows, built with the same feature path. The 24-hour boundary keeps every
   returned label clear of the cut, so no cleaning step can depend on data at or after it.
3. **Training.** A fresh AdamW (same learning rate and weight decay), the sampler stream `default_rng(10000 + seed)`, the
   same borrowed windows and time limit, and exactly 8 epochs. The returned windows are part of P's own half of each
   epoch's draws. The final weights are kept; there is no selection. The stage-1 criterion is recorded per epoch as a
   description only (its windows now lie inside the training data).

Why the second stage exists, and what it changed and did not change on the test windows, is in
[EVALUATION.md](EVALUATION.md#why-the-recipe-returns-the-validation-span).

### Seeds and ensemble

Seeds 0, 1 and 2. A seed sets Python's, NumPy's and PyTorch's generators, and through them the weight initialisation
and, in stage 1, the sampler stream `default_rng(seed)`; the caps, the pool subsample and the returned span use fixed
seeds per population. The carrier forecast is the plain average of the three seeds' forecasts, accumulated in float64 in
seed order (`average_seed_forecasts`, `carrier_forecasts` with a list of carriers).

```python
from ankyra_lite.carrier.train import TrainConfig, recipe, prepare_source, train_ensemble

stage1 = TrainConfig()                                                    # from scratch, early stopping
sources = {n: prepare_source(n, data[n], cut[n], stage1, seed_index=i) for i, n in enumerate(names)}
first = train_ensemble(target, sources, stage1, checkpoint_dir=f"runs/{target}/stage1")

stage2 = recipe("span_return", "runs/{target}/stage1/seed{seed}.pt")     # validation span returned, 8 epochs
sources = {n: prepare_source(n, data[n], cut[n], stage2, seed_index=i, with_validation=(n == target))
           for i, n in enumerate(names)}
final = train_ensemble(target, sources, stage2, checkpoint_dir=f"runs/{target}")
```

The example writes into `runs/` (ignored by git) so that it never overwrites the released checkpoints below.

The package reproduces the research training on small runs: the sampler's draws, the windows, splits and caps, the
returned span and both pools row for row (Oslo and BDG2), and one optimisation step of each stage from the same weights
with zero parameter difference (`tests/test_train.py`; the data-dependent tests need the study's local data). The
released forecasts come from the study's research runs on a GPU, not from a full-size run of this package.

### Released weights

The repository ships the 27 carriers of the released recipe (9 populations × 3 seeds): the stage-2 final weights of
the study's research runs, copied byte for byte by `tools/export_checkpoints.py --out checkpoints`:

```
checkpoints/<population>/seed<s>.pt     population in {bdg2, cambridge, heew, eweld, goiener_confirm, households,
                                                       oslo, drammen, cineldi}; s in {0, 1, 2}
checkpoints/MANIFEST.json               per file: SHA-256, size, parameters, training cut (hour index and date), store,
                                        source run and its record's SHA-256, the stage-1 run it continued
```

Each file is `{"model": state_dict, "epoch": 8}` (about 0.27 MB) and loads strictly with
`ankyra_lite.load_carrier(path)`. The export checks every copy against the SHA-256 in its run record and, for three test
windows of Oslo and three of CINELDI (corrected store) per seed, reproduces the stored GPU carrier forecasts on the CPU at
the same issue times; the rule is a relative difference below 10⁻⁴, and the largest one recorded in
`checkpoints/MANIFEST.json` (`verification`) is 2.1 × 10⁻⁵ (CINELDI, seed 2). A checkpoint belongs to one population
and its training cut: pass that cut as `cut=` and use origins at least 186 days after it
([DATA.md](DATA.md#windows-borrowed-from-other-populations)); it has not been evaluated on any other population.

### Other recipes in the package

`recipe("plain_continuation", ...)` (stage 2 without the returned span) and `recipe("proportional_validation", ...)` (a
validation span of 10% of the population's record, clipped to 38–93 days, with early stopping) are the two other arms
of the study that chose the recipe; `TrainConfig()` alone is stage 1 without stage 2. They are kept for reproduction and
are not the released recipe; their readings are in [EVALUATION.md](EVALUATION.md#why-the-recipe-returns-the-validation-span).

## How the anchoring layer uses the carrier

`ankyra_lite.anchor.forecast(history, foundation={k: forecast issued at o − 744k}, group=..., temp_sigma_std=...,
dst_region=...)` is the ANKYRA 2.2.1 anchoring layer, unchanged; it takes the carrier's forecasts where ANKYRA takes a
foundation model's. Its code and docstrings call this input the foundation model.

### Pseudo-origins

The layer reads the carrier's forecast issued at the origin (k = 0) and at q = o − 744k for k = 1 … 6, one 31-day step
apart, wherever the 1,344-hour context before q is completely observed after the short-gap rule
(`anchor.pseudo_origin_contexts`). The target month [q, q + 744) of a pseudo-origin is already observed at the origin, so
the layer can score the carrier against the unit's own history on months that have happened.
`carrier_forecasts(carriers, ..., origin=o, cut=C′)` computes exactly these forecasts, each from data before its own
issue time, and drops any pseudo-origin before the cut.

### Why the test set starts 186 days after the cut

Each carrier forecast at a pseudo-origin q must be a genuine forecast for the carrier: issued at or after its training
cut, so that the month it forecasts was never a training label. The test set therefore starts 6 × 744 hours = 186 days
after the cut: then q = o − 744k ≥ C′ for every k ≤ 6 and every test origin o, and every pseudo-target lies after every
training label (property C10; in stage 2 the newest label ends 24 hours before the cut). Without the gap the per-unit
weights would be estimated on in-sample carrier errors and would trust the carrier too much. The price is a carrier that
is at least half a year older than it would be in use.

### What the layer estimates from the completed pseudo-forecasts

Per unit, from the carrier's forecasts at the pseudo-origins against the realised months (details in the first
release's `docs/METHOD.md`; constants unchanged):

- **Level.** Six historical level candidates (the recent 7-day mean, the long-history mean over at most 8,760 hours, the
  mean of the window one year earlier, each with and without weather adjustment; all with a day-type offset) and the
  carrier's own window mean as a seventh candidate. Each candidate's window-mean error is computed at up to 12
  pseudo-origins (up to 6 for the carrier); each candidate's error scale is the root mean square of those errors (frozen
  configuration `level.scale = "rms"`), and the weights are inverse-square weights on that scale, shrunk towards equal
  weights: w_j = (K_eff · w̃_j + 8 · w0_j) / (K_eff + 8), with K_eff the number of completed pseudo-origins (P7–P9).
- **Daily path.** Seven historical candidate paths for the 31 daily means. A candidate's error at a pseudo-origin is the
  root mean square over the 31 days of its normalised daily-path error; its error scale is the **median** of the absolute errors over
  up to 12 pseudo-origins (frozen configuration `day.scale = "mad"`, the code's historical label for this median, not a
  centred median absolute deviation), not the root mean square used for the level. The weights are inverse-square
  weights on that scale, shrunk towards equal weights in the same form with shrinkage 2 (`history/_day.py`,
  `weights_c`).
- **Handover of the daily means.** For the weeks of days 1–7, 8–14, 15–21 and 22–31, the daily means are
  m_d = (1 − α_w) m^H_d + α_w m^C_d, where m^H is the history side (the level with the carrier candidate, plus the daily
  path) and m^C the carrier's daily means; α_w = (n λ̂_w + 1) / (n + 2), with λ̂_w the least-squares combination weight,
  clipped to [0, 1], from the n ≤ 6 completed pseudo-origin pairs (P10). No pair gives α_w = ½.
- **Within-day trust.** The carrier's day-demeaned shape w^C is mixed with the unit's analog-day shape S (up to eight
  complete pre-origin days of the same day type, within ±14 days of the same day of year, in the same daylight-saving
  state, closest in temperature to the day's climatology, rescaled to the variability before the origin):
  w = w^C + ω (S − w^C), with one ω per window, ω = min(clip(λ̂, 0, 1) · n / (n + 2), ½), estimated from the n ≤ 3
  completed pseudo-origins. A day with fewer than four analogs keeps the carrier's shape.
- **Short-gap rule.** Gaps of at most 6 hours inside one 744-hour block are interpolated in the copy of the pre-origin
  record that the estimator reads, and a pseudo-origin's target month qualifies with at least 90% observed hours.

The forecast is F(d, h) = m_d + w(d, h), projected onto F ≥ 0. The layer also reports `anchoring_record`: the carrier's
level weight and the log ratio of the best historical level candidate's pseudo-origin error to the carrier's
(descriptive; not used by the forecast).

**Arguments used in the evaluation.** `group`: the unit's category; `temp_sigma_std`: the population's value fitted on
the days before the training cut; `dst_region`: the unit's region; `observed`: the finite load hours; every other
argument at the 2.2.1 default (`gap_tolerance=True`, `single_trust=True`, `within_anchor=True`,
`micro_load_rule=True`).

## State windows and the seasonal-naive rule

The anchoring layer has two state rules, both on load before the origin:

- **off-state:** the observed load of the last 168 hours is at most 10⁻⁶ kW;
- **micro-load:** all 1,344 context hours are within 10⁻³ kW of zero.

On such a window the layer returns the carrier's forecast unchanged. That return rests on the premise that the carrier
forecasts a switched-off or near-zero meter well, which held for foundation models and is not guaranteed for a trained
carrier. ankyra-lite therefore applies its own rule on these windows (`ankyra_lite.forecast`, default
`state_rule="seasonal_naive"`): the delivered forecast is the last observed week repeated over the 744 hours (missing
hours as 0), its level readout is that forecast's mean, and its peak readout is read from it. `state_rule=None` keeps the
layer's own return. The rule was written after an earlier scoring of the test windows
([EVALUATION.md](EVALUATION.md#identity-of-the-evidence)).

## Readouts

- **Energy.** 744 × the level, read before the nonnegativity projection (`ForecastResult.energy_kwh`). The evaluation
  scores the energy of the delivered trajectory (744 × its mean), as the first release did; the two differ only where
  the projection binds.
- **Peak.** `anchor.readouts.peak_readout` on the delivered trajectory: U = max over days d of (L̂_d + A_τd), with L̂_d
  the forecast mean of day d and A_t the largest excursion above its own daily mean among the four most recent context
  days of day type t (κ = 1; P15–P17).
- **Interval.** Not part of the forecaster. The evaluation centres the anchoring layer's residual-quantile interval on
  the delivered trajectory as a description ([EVALUATION.md](EVALUATION.md#intervals)).

The fields of `ForecastResult` are listed in the [README](../README.md).

## Inference cost

One forecast runs the carrier at up to seven issue times (the origin and up to six pseudo-origins) for each of the three
seeds, 21 network evaluations of 61,613 parameters, then the anchoring layer (the historical estimator at the origin and
its pseudo-origins, the analog-day shapes), the state rule and the peak readout. Everything runs on one CPU thread;
nothing is downloaded. Measured with this repository's package and the released checkpoints on 6 Oslo test windows,
after one untimed warm-up, on one CPU thread without a GPU
([`results/cost_per_window.csv`](../results/README.md#cost)):

| Step | Time |
|---|---:|
| one forecast in all (carrier forecasts, anchoring layer with the state rule, peak readout) | 0.584 s per window |
| carrier forecasts at the origin and the pseudo-origins (7.0 issue times on average) × 3 seeds | 0.075 s per window |
| one carrier forecast (one issue time, one seed; mean) | 0.0036 s |
| anchoring layer call with the carrier forecasts supplied, with the state rule | 0.509 s per window |
| peak readout | 0.0003 s per window |
| loading the three checkpoints (once) | 0.045 s |

The anchoring layer, not the carrier, takes most of the time (0.509 of 0.584 s). A checkpoint is 0.2657 MB.

The stored evaluation forecasts were computed by the research code on a GPU; on the CPU the package reproduces them up
to floating-point differences between the two devices. On the timed windows the largest relative difference is
1.98 × 10⁻⁶ for the carrier forecasts and 1.47 × 10⁻⁶ for the delivered trajectory (the two `check:` rows of
`cost_per_window.csv`; see also `tests/test_carrier_equivalence.py`).

## Constants

The anchoring layer's constants are those of ANKYRA 2.2.1, unchanged (1,344-hour context, 744-hour horizon, 12 / 6
pseudo-origins, shrinkage 8 / 2 / 2, within-day trust over 3 pseudo-origins capped at ½, analog days ±14 days, 8 kept, 4
required, off-state 10⁻⁶ kW, micro-load 10⁻³ kW, short gaps of at most 6 hours, 90% observed pseudo-target, peak
envelope over 4 same-type days with κ = 1). The carrier's constants:

| Constant | Value | Code |
|---|---|---|
| daily, monthly and future-day tokens | 366 × 69, 24 × 16 (blocks of 744 h), 31 × 16 | `features.py` |
| scale | max(last 744-h mean, last 8,760-h mean) of the observed hours; 1 if not positive | `features.py` |
| width, attention heads | 48, 4 | `ModelConfig` |
| dilations (daily, monthly) | 1, 2, …, 128; 1, 2, 4, 8 (two shared blocks, kernel 3) | `ModelConfig` |
| hourly dictionary rank, bound of its change | 8, 0.1 | `ModelConfig` |
| copy pointer: recent days, key dimension | 56, 16 | `build_carrier` |
| eligibility (context and labels observed) | 75% | `TrainConfig.min_observed_fraction` |
| origin grids: training, validation | 24 h, 168 h | `TrainConfig` |
| validation span | 93 days (3 × 744 h) before the cut | `FixedSpan` |
| caps per population | 16,000 training, 2,000 validation windows | `TrainConfig` |
| draws per epoch, share from the target | 16,000, ½ | `TrainConfig` |
| learning rate, weight decay, batch, gradient clip | 3 × 10⁻⁴, 10⁻⁴, 64, 1.0 per group | `TrainConfig` |
| loss | MSE + 0.1 × generalised KL, log1p per window | `forecast_loss` |
| stage 1: maximum epochs, patience | 30, 5 | `TrainConfig` |
| stage 2: epochs, label boundary before the cut | 8, 24 h | `CONTINUATION_EPOCHS`, `ReturnSpan` |
| gap from the training cut to the test set | 186 days (6 × 744 h) | `evaluation/windows.py` |
| seeds | 0, 1, 2 | `TrainConfig.seeds` |

[`results/constants.csv`](../results/README.md#constants) lists every constant of ankyra-lite in one file.

## What is not claimed

- **Not a general-purpose model.** One carrier is trained per population and evaluated on that population's test
  windows only. A checkpoint applied to another building or population without retraining has not been evaluated.
- **Not a first test.** The test windows had been read before, and the copy pointer's direction, the seasonal-naive
  state rule and the recipe's second stage were chosen after reading them ([EVALUATION.md](EVALUATION.md#identity-of-the-evidence)).
- **No mechanism is proven by the recipe study.** The returned span adds same-season data for the short-record
  populations, but it also changes recency, sample diversity and the replay of older windows, and adds 8 epochs of
  training; the study's readings do not separate these.
- **Properties are not accuracy.** C1–C10 and P1–P20 hold for any weights; they make the forecaster inspectable and
  bound what its parts can do. Accuracy is measured.
- **No intervals.** The interval of the evaluation is the anchoring layer's, centred on ankyra-lite's trajectory as a
  description; ankyra-lite was not designed or tuned for it.
- **No bitwise reproducibility of training.** GPU training is not bitwise repeatable; the package reproduces single
  optimisation steps, not the study's full runs.
- **Inherited limits of the anchoring layer.** Hourly data only, a 744-hour horizon only, a complete 1,344-hour context
  at the origin, and no temperature-signature prior for the category Commercial.
