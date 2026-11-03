# Method

**Maintenance note.** The [code-audit corrections](CODE_AUDIT_CORRECTIONS_20261022.md) distinguish the original scored implementation from subsequent interface/guard maintenance. No historical score or frozen provenance hash is replaced.

This document describes ankyra-lite 0.2.0 (recipe Rw, six seeds, the continuation layer on): a small carrier network,
trained per population, whose forecasts at the origin and at six pseudo-origins go into an unchanged copy of the
ANKYRA 2.2.1 anchoring layer, followed by the seasonal-naive state rule and the origin-continuity layer, which corrects
the first hours and days of the anchored trajectory toward the last observed hours. Every number below was checked
against the code: `ankyra_lite/carrier/model.py`, `features.py`, `inference.py` and `train.py`,
`ankyra_lite/forecast.py`, `ankyra_lite/continuation.py`, `ankyra_lite/readouts.py` and the anchoring layer
`ankyra_lite/anchor/` (a byte-identical copy; [PROVENANCE.md](../ankyra_lite/anchor/PROVENANCE.md)). The release
switches (the default of the continuation layer, the recipe and the seeds of the default checkpoints) are in
`ankyra_lite/_release.py`; with the layer off, the 0.1 checkpoints (recipe R) and seeds 0–2, the package gives the
forecasts of 0.1.1 bit for bit. The exact properties of the three parts, with proofs and numerical checks, are in
[theory/PROOFS.md](../theory/PROOFS.md) (P1–P20 for the anchoring layer, C1–C14 for the carrier and its training
recipe, O1–O9 for the continuation layer).

Contents: [Information at the origin](#information-at-the-origin) · [The carrier network](#the-carrier-network)
([inputs](#inputs) · [encoder](#encoder) · [read-out](#read-out) · [energy-budget decoder](#energy-budget-decoder) ·
[copy pointer](#copy-pointer) · [parameters](#parameters)) · [Training](#training) ([windows and split](#windows-and-split) ·
[pooled training set](#pooled-training-set) · [sampler](#sampler) · [loss](#loss) · [optimisation](#optimisation) ·
[early stopping](#stage-1-early-stopping) · [span return](#stage-2-the-validation-span-returned) ·
[loss weights of recipe Rw](#stage-2-of-recipe-rw-inverse-noise-loss-weights) ·
[seeds](#seeds-and-ensemble) · [released weights](#released-weights) · [other recipes](#other-recipes-in-the-package)) ·
[How the anchoring layer uses the carrier](#how-the-anchoring-layer-uses-the-carrier)
([pseudo-origins](#pseudo-origins) · [three orthogonal blocks](#three-orthogonal-blocks) · [level](#level) ·
[centred daily path](#centred-daily-path) · [within-day shape](#within-day-shape) ·
[week-by-week handover](#week-by-week-handover) · [short-gap rule](#short-gap-rule)) ·
[State windows](#state-windows-and-the-seasonal-naive-rule) · [The continuation layer](#the-continuation-layer) ·
[Readouts](#readouts) · [Output fields](#output-fields) ·
[Inference cost](#inference-cost) · [Constants](#constants) · [What is not claimed](#what-is-not-claimed)

## Information at the origin

A forecast at origin o is a function of the following, and nothing else:

- load and temperature at hours t < o (the carrier reads at most the last 366 days for its daily tokens and the 24 blocks
  of 744 hours before them; the anchoring layer reads the whole pre-origin record);
- the day types (weekday or public holiday) of the 31 forecast days, treated as known at the origin;
- the unit's category: an input of the carrier, and the selector of the anchoring layer's fixed temperature-signature
  prior. The prior was fitted once on BDG2 windows ending before July 2016 and is never refitted; the category
  Commercial has none ([Level](#the-category-commercial-has-no-prior));
- the population's fixed temperature-anomaly scale `temp_sigma_std`, fitted on the days before the training cut
  ([DATA.md](DATA.md#the-input-format)), and the daylight-saving rule of the unit's region (analog days only);
- the carrier's weights, fitted on labels that end before the training cut, at least 186 days before the first test
  origin;
- the carrier's forecasts issued at o and at each pseudo-origin q = o − 744k (k = 1 … 6), each built from data before q
  only;
- the four constants of the continuation layer, fitted once on development windows before 0.2.0 was frozen and shared
  by every population; the layer itself reads the last 672 hours of load before the origin, the day type of the last
  observed day and those of the first seven forecast days (O3);
- the carrier's loss weights in training (recipe Rw), computed from each training window's own last 28 context days
  before its origin and from the sampler's draw probabilities (C14); they enter only through the carrier's weights,
  which are fitted on labels that end before the training cut.

The continuation layer adds no information after the origin: it is a function of the trajectory the anchoring layer
delivers, of load before the origin and of the calendar of the first forecast week, which ankyra-lite treats as known
at the origin like the rest of the forecast month's calendar.

The carrier's weights are fitted once per population and are not changed by a forecast. The anchoring layer estimates
its per-unit quantities (candidate weights, handover weights, within-day trust, temperature signature) at every origin
from the unit's own pre-origin record and its completed pseudo-origins:

- every pseudo-target ends by o;
- each level pseudo-forecast at o − 744k uses only data before o − 744k, and the carrier's forecast at o − 744k is
  built from data before that issue time (C4);
- the weekly handover and the within-day trust use the realised values of these pseudo-targets.

Two exceptions concern only the pre-origin record:

- One daily-path candidate, `cur` (the path implied by the level construction; [Centred daily path](#centred-daily-path)),
  reuses the weather weight estimated at the origin. Its pseudo-errors are therefore not strictly out of sample with
  respect to that weight, although all of them precede the origin.
- `temp_sigma_std` is fitted once per population, on the complete days before the training cut
  ([DATA.md](DATA.md#the-input-format)). The historical estimator's pseudo-origins o − 744k with k = 7 … 12 (and those
  of the interval readout) fall inside that fit period when the origin lies less than 744k hours after the cut, as on
  the first months of every test set; they then use a scale that reflects temperature recorded after them. All of that
  temperature precedes the origin. The carrier's pseudo-origins (k ≤ 6) lie at or after the cut, after the fit period.

Two fixed components come from outside the unit's own record: the carrier's weights (fitted on labels that end before
the population's training cut, borrowed windows included; [DATA.md](DATA.md#windows-borrowed-from-other-populations))
and the anchoring layer's temperature-signature prior. Neither sees the unit's future. The prior was fitted on BDG2
windows ending before July 2016. BDG2 is one of the nine populations here (a study population, not an outside source);
its test windows start on 2017-08-01, after those windows end. The prior can still reflect data recorded later than
the oldest test windows of other populations (Cambridge's test set starts in July 2013).

Checks in the code: [`tests/test_carrier_unit.py`](../tests/test_carrier_unit.py) (no read at or after the origin, the
pseudo-origin rule), the anchoring layer's own [`tests/anchor/test_no_future_information.py`](../tests/anchor/test_no_future_information.py),
[`tests/test_forecast_v2.py`](../tests/test_forecast_v2.py) (no information after the origin with the continuation
layer) and properties C4, C10, C14 and O3 in [theory](../theory/README.md). These establish input isolation in the implementation. They
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

**Monthly tokens** (24 × 16), 24 fixed blocks of 744 hours older than the daily span (not calendar months; together
with it, at most 1,110 days of history); a block with no observed hour is masked out and set to zero:

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

Every forecast day attends over the **complete days within the unit's last 56 day positions** (all 24 hours observed; zero-load days
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

One carrier is trained per target population P, with six seeds (three in version 0.1). Every constant of the recipe
is a field of `ankyra_lite.carrier.train.TrainConfig`; `recipe(name, init_checkpoint)` builds the named configurations.
The released recipe of version 0.2.0, recipe Rw, has two stages: stage 1 is `recipe("from_scratch")` (= `TrainConfig()`;
labelled `P3s` in `results/recipe_history.csv`, where it is the starting point of every arm) and stage 2 is
`recipe("span_return_weighted", <stage-1 checkpoint>)`, recipe R's stage 2 (`recipe("span_return", ...)`, labelled `R`
there, released in 0.1.0) with inverse-noise loss weights. Stage 1 alone is not the released recipe.

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
[EVALUATION.md](EVALUATION.md#why-the-recipe-returns-the-validation-span). This is recipe R's stage 2; recipe Rw, the
released recipe of 0.2.0, adds the loss weights below.

### Stage 2 of recipe Rw: inverse-noise loss weights

Recipe Rw is recipe R with one change, in stage 2 only: each training window carries a weight in the loss
(`recipe("span_return_weighted", <stage-1 checkpoint>)`, also `recipe("Rw", ...)`; `TrainConfig.noise_weighting =
NoiseWeights()`). Stage 1 is unchanged: the same from-scratch training with early stopping on the 93-day span. For
seeds 0, 1 and 2 stage 2 starts from the stage-1 checkpoints that recipe R's released weights continued; seeds 3, 4
and 5 were trained by the same stage-1 recipe ([Seeds and ensemble](#seeds-and-ensemble)).

**Why.** After log1p compression every window still has equal weight in recipe R. Windows whose last weeks are noisy
in a way no month-ahead forecast can follow (spiky household meters, irregular operation) carry large per-window
losses and steer the carrier toward fitting noise. Recipe Rw lowers their weight by the inverse square root of their
own recent noise, with a floor so that quiet windows are not raised without limit, and keeps every population's share
of the training draws. The rule was taken unchanged from an earlier study of this carrier and frozen before 0.2.0.
Its planned effect was small (about +0.25% in whole-window hourly error), below what the seed noise lets one resolve.
On the development windows (re-used evaluation) Rw against R with six seeds was +0.42% over the month, with the peak
readout resolvably worse (−0.92%) within the guard written before the read; on the test windows it was +0.31%
[+0.06, +0.56] over the month and +0.04% on the peak (`results/recipe_history.csv`).

**The weights** (`noise_weights`, computed once per training pool before training):

1. **Noise estimate.** For training window i with origin o and causal scale s_i (the carrier's scale input), σ̂²_i is
   the mean of ((y_t − y_{t−168}) / s_i)² over the hours t of the last 21 days before o whose hour one week earlier is
   also observed, read from the window's day tokens 338–365 (`noise_sigma2`: x = sinh(token) = y / s_i). With fewer
   than 168 such pairs σ̂²_i is undefined.
2. **Draw probabilities.** p_i is the probability that one draw of the sampler picks window i
   (`draw_probabilities`): ½ for the target's windows together (each unit with equal total probability) and ½ split
   equally across the borrowed populations (unit-balanced within each), exactly the sampler of [Sampler](#sampler).
3. **Floor.** f is the 25th percentile of the defined σ̂² over the pool, weighted by p.
4. **Raw weight.** ω_i = max(σ̂²_i, f)^(−½); a window with undefined σ̂² gets the p-weighted median of the defined raw
   weights.
5. **Per-population normalisation.** w_i = c_Q ω_i, with c_Q such that the p-weighted mean weight of every population
   Q is 1: each population keeps its share of the draws, and the weighting moves weight only between windows of the
   same population (C12).
6. **Loss.** For a batch B, Σ_{i∈B} w_i ℓ_i / Σ_{i∈B} w_i with ℓ_i = log(1 + MSE_i + 0.1 · KL_i), the per-window
   term of [Loss](#loss), unchanged (`weighted_log1p_loss`).

Everything else is recipe R's stage 2: a fresh AdamW (learning rate 3 × 10⁻⁴, weight decay 10⁻⁴), batch 64, 16,000
draws per epoch from the stream `default_rng(10000 + seed)`, gradient clipping at 1.0 per parameter group, exactly 8
epochs and the final weights, with no selection.

- **Causality.** The weights read only the context before each training window's origin (C14): no label hour, no
  validation or test window. The floor and the normalisers are computed on the training pool, whose labels end at least
  24 hours before the cut. The rule is global (γ = ½, the 25th percentile, 168 pairs, 21 days); nothing is set per
  unit or per population.
- **Checks.** With every weight equal to 1 the weighted loss, its gradient and two optimisation steps equal recipe R's
  bit for bit on the CPU (C11). The first batch of every weighted epoch is checked against the weights implied by its
  own day tokens (`expected_weight`, relative tolerance 10⁻⁹), which catches misaligned rows.
- **Records.** `TrainResult.weights_record` and, for the released weights, the weights record named in
  `checkpoints/MANIFEST.json`: the floor (σ̂² between 0.0087 (Cambridge) and 0.0389 (GoiEner households) over the nine
  populations, `floor_sigma2`), each population's normaliser and draw share, the share of draws without a noise
  estimate, the Kish effective-sample-size ratio under the draw probabilities (0.69–0.82, `kish_ess_ratio_draws`) and
  the largest weight over the median (1.56–6.23).

```python
stage2_rw = recipe("span_return_weighted", "runs/{target}/stage1/seed{seed}.pt")   # recipe Rw: weighted, 8 epochs
sources = {n: prepare_source(n, data[n], cut[n], stage2_rw, seed_index=i, with_validation=(n == target))
           for i, n in enumerate(names)}
final = train_ensemble(target, sources, stage2_rw, seeds=range(6), checkpoint_dir=f"runs/{target}/rw")
```

`train_ensemble` computes the weights once per pool and shares them across the seeds.

### Seeds and ensemble

Seeds 0 to 5. A seed sets Python's, NumPy's and PyTorch's generators, and through them the weight initialisation
and, in stage 1, the sampler stream `default_rng(seed)`; the caps, the pool subsample, the returned span and (recipe
Rw) the loss weights do not depend on the seed. The carrier forecast at every issue time is the plain average of the
six seeds' forecasts, accumulated in float64 in ascending seed order and divided by six (`average_seed_forecasts`,
`carrier_forecasts` with a list of carriers); pass the checkpoints in ascending seed order, as
`ankyra_lite.checkpoint_paths(population)` returns them, and one training cut `cut=` for all six.

Seeds 0, 1 and 2 are those of version 0.1 (their stage-1 runs are the ones behind the 0.1 checkpoints); seeds 3, 4 and
5 were trained for 0.2.0 by the same stage-1 recipe, on the same data (Drammen and CINELDI on the corrected CINELDI
store). Averaging more seeds lowers the carrier's retraining variance: six seeds against three were +0.07% (not
resolved) on the development windows and +0.41% [+0.15, +0.70] on the test windows, recipe R without the layer
(`results/recipe_history.csv`). The two groups of three seeds (0–2 and 3–5) are kept as separate products, so that the
seed spread can be read without anchoring each seed alone ([EVALUATION.md](EVALUATION.md#seeds)). A three-seed
ensemble remains possible: pass three checkpoints. The example below is recipe R's (version 0.1); recipe Rw's stage 2
is above.

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

The repository ships 81 carriers: the 54 of the release configuration (recipe Rw, 9 populations × 6 seeds) and,
unchanged, the 27 of version 0.1 (recipe R, 9 × 3). Each is the stage-2 final weights of a study run, copied byte for
byte by `tools/export_checkpoints.py --out checkpoints`:

```
checkpoints/<population>/rw/seed<s>.pt  recipe Rw, s in {0, …, 5}: the carriers of version 0.2.0 (the default)
checkpoints/<population>/seed<s>.pt     recipe R, s in {0, 1, 2}: the carriers of version 0.1 (byte-identical)
checkpoints/MANIFEST.json               per file: SHA-256, size, parameters, recipe and recipe key, seed group, loss
                                        weighting (rule and constants) with its weights record, training cut (hour
                                        index and date), store, source run, training record and the stage-1 run it
                                        continued, each with its SHA-256; which files form the release configuration
```

`population` is one of bdg2, cambridge, heew, eweld, goiener_confirm, households, oslo, drammen and cineldi. Each file
is `{"model": state_dict, "epoch": 8}` (about 0.27 MB) and loads strictly with `ankyra_lite.load_carrier(path)`. The
export checks every copy against the SHA-256 in its run record and, for three test windows of Oslo and three of CINELDI
(corrected store) per seed of the release configuration, reproduces the stored GPU carrier forecasts on the CPU at the
same issue times; the rule is a relative difference below 10⁻⁴, and the largest one recorded in
`checkpoints/MANIFEST.json` (`verification`) is 2.0 × 10⁻⁵ (CINELDI, seed 2). A checkpoint belongs to one population
and its training cut: pass that cut as `cut=` and use origins at least 186 days after it
([DATA.md](DATA.md#windows-borrowed-from-other-populations)); it has not been evaluated on any other population.

### Other recipes in the package

`recipe("span_return", ...)` is recipe R, released in 0.1 (its 27 checkpoints ship unchanged).
`recipe("plain_continuation", ...)` (stage 2 without the returned span) and `recipe("proportional_validation", ...)` (a
validation span of 10% of the population's record, clipped to 38–93 days, with early stopping) are the two other arms
of the study that chose recipe R; `TrainConfig()` alone is stage 1 without stage 2. They are kept for reproduction and
are not the released recipe; their readings are in [EVALUATION.md](EVALUATION.md#why-the-recipe-returns-the-validation-span).

## How the anchoring layer uses the carrier

`ankyra_lite.anchor.forecast(history, foundation={k: forecast issued at o − 744k}, group=..., temp_sigma_std=...,
dst_region=...)` is the ANKYRA 2.2.1 anchoring layer, unchanged; it takes the carrier's forecasts where ANKYRA takes a
foundation model's. Its code and docstrings call this input the foundation model, and two output names keep that word
(`foundation_within_day_kw`, the key `fm` of `level_weights`); here both denote the carrier. The subsections below
describe what the layer does with the carrier's forecasts. They follow the copied code: `ankyra_lite/anchor/core.py`
(`forecast`, the carrier level candidate, the handover, the state rules, the short-gap rule), `analog.py` (the
within-day shape), `blocks.py`, `readouts.py` and the frozen reference estimator `history/` (the historical level and
daily path).

### Pseudo-origins

The layer reads the carrier's forecast issued at the origin (k = 0) and at q = o − 744k for k = 1 … 6, one 31-day step
apart, wherever the 1,344-hour context before q is completely observed after the short-gap rule
(`anchor.pseudo_origin_contexts`). The target month [q, q + 744) of a pseudo-origin is already observed at the origin, so
the layer can score the carrier against the unit's own history on months that have happened.
`carrier_forecasts(carriers, ..., origin=o, cut=C′)` computes exactly these forecasts, each from data before its own
issue time, and drops any pseudo-origin before the cut. The carrier is given issue times, not contexts: it builds each
forecast from the record before q as supplied, so an hour that the layer's short-gap rule interpolates in its own copy
([below](#short-gap-rule)) is a missing hour for the carrier unless the caller filled it before the call.

### Why the test set starts 186 days after the cut

Each carrier forecast at a pseudo-origin q must be a genuine forecast for the carrier: issued at or after its training
cut, so that the month it forecasts was never a training label. The test set therefore starts 6 × 744 hours = 186 days
after the cut: then q = o − 744k ≥ C′ for every k ≤ 6 and every test origin o, and every pseudo-target lies after every
training label (property C10; in stage 2 the newest label ends 24 hours before the cut). Without the gap the per-unit
weights would be estimated on in-sample carrier errors and would trust the carrier too much. The price is a carrier that
is at least half a year older than it would be in use.

### Three orthogonal blocks

At origin o a forecast F(d, h) covers D = 31 origin-aligned days of H = 24 hours, T = 744 hours in all. For any
744-hour vector v define

- the level ℓ(v) = T⁻¹ Σ_{d,h} v(d, h);
- the centred daily path b_d(v) = H⁻¹ Σ_h v(d, h) − ℓ(v);
- the within-day path w(d, h)(v) = v(d, h) − H⁻¹ Σ_r v(d, r).

These are orthogonal projections of ranks 1, 30 and 713, so for an error e = F − y

  T⁻¹ Σ_{d,h} e(d, h)² = ℓ(e)² + D⁻¹ Σ_d b_d(e)² + T⁻¹ Σ_{d,h} w(d, h)(e)².

Replacing one block of a forecast changes its mean squared error by exactly the change in that block's loss. The
anchoring layer uses this identity to assemble the forecast block by block (the level and the daily path from the
history side and the carrier, the within-day block from the carrier and the unit's analog days); the evaluation uses
it to attribute the error ([`results/block_shares.csv`](../results/README.md), Figure
[14](../figures/fig14_block_attribution.png)); the readouts are defined on it. The carrier's own trajectory splits the
same way: its level, daily path and within-day shape are the blocks of its energy budget (C2). On the test windows,
weighting the two sources separately by block was not shown to add accuracy over one whole-window weight per window
([EVALUATION.md](EVALUATION.md#block-wise-weights-against-one-whole-window-weight)).

- Because 7 and 31 are coprime, the 744-hour window has no separate weekly Fourier subspace.
- Weekly effects therefore enter the daily-path and within-day blocks and are represented by day types (Monday = 0 …
  Sunday = 6, holiday = 7), in the anchoring layer as in the carrier's tokens.

The four-block refinement behind the Fourier statement, and the replacement calculus built on the identity, are P1–P6
in [theory/PROOFS.md](../theory/PROOFS.md). Code: `ankyra_lite/anchor/blocks.py` (`level`, `daily_path`, `within_day`,
`block_losses`); for a forecast, `ankyra_lite.readouts.blocks`; the properties as operators: `theory/operators.py`.

### Level

Load is normalised by the mean l0 and the standard deviation s0 of the 744 hours before the origin, with s0 floored at
max(0.01 |l0|, 10⁻³) kW. A record whose whole context stays within the 10⁻³ kW floor of zero is not normalised by its
own scale; the [micro-load rule](#state-windows-and-the-seasonal-naive-rule) hands it to the state rule.

**Six historical candidates.** Each of three means comes with and without weather adjustment:

- the recent 7-day mean;
- the long-history mean over at most 8,760 hours;
- the mean of the 744-hour window one year earlier (the window that starts 8,760 hours before the origin).

Every candidate adds a day-type offset estimated from the last eight weeks (the offsets of the target days).
Weather-adjusted candidates in addition replace the temperature response of their own window by the expected response
over the target days: the expectation over a normal temperature anomaly of standard deviation `temp_sigma_std` around
the pre-origin climatology of each target day (an annual harmonic fitted on the temperature before the origin). No
observed future weather is used.

**Temperature signature.** The response is the unit's *temperature signature*: a convex piecewise-linear function of
the daily-mean temperature, with hinges every 5 °C from −10 °C to 40 °C (temperature standardised as
(T − 15 °C) / 10 °C). It is fitted on the unit's last 365 days before the origin and shrunk towards the fixed prior
curve of the unit's category. The fit uses the days whose 24 hours are all observed, removes the mean of each day type
from the daily means first, and is a ridge regression towards the prior curve (strength 3) with nonnegative hinge
slopes, which is what makes the curve convex. With fewer than 60 such days the prior curve is used as it is.

**Carrier candidate.** One further candidate is the carrier's own window mean at the origin, origin-normalised with the
calendar offset removed. The level is therefore a weighted combination of seven candidates: six historical ones and
the carrier's.

**Weights.** Each candidate's window-mean error is computed at up to K = 12 earlier pseudo-origins o − 744k (k ≤ 6 for
the carrier candidate, from the carrier's forecasts issued at those pseudo-origins). With s_j the root mean square of at
least two such errors of candidate j,

  w̃_j = [I_j / max(s_j, 10⁻⁶)²] / Σ_r [I_r / max(s_r, 10⁻⁶)²],   w_j = (K_eff · w̃_j + K0 · w0_j) / (K_eff + K0),   K0 = 8,

where I_j = 1 if candidate j has at least two pseudo-origin errors and 0 otherwise, w0 is uniform over the supported
candidates (I_j = 1), and K_eff counts the pseudo-origins (at most 12) at which both the unadjusted recent and the
unadjusted long-history candidate have an error. Candidates with fewer errors (the annual ones, and the carrier
candidate with at most six) are shrunk with the same K_eff. A candidate that cannot be formed at the origin (for
example the annual window of a record shorter than a year) gets weight 0 and the others are renormalised.

- With complete records K_eff(m) = min{12, ⌊(m − 1,344) / 744⌋}₊ for a history of m hours (P7).
- When no candidate has two errors, the weights are equal.
- The annual candidates need 10,248 hours (about 14 months) before they can be weighted: a year of record supplies an
  annual prediction, but its two pseudo-origin errors need 8,760 + 2 × 744 hours.
- In the historical estimator, with matched support and no candidate deleted at the origin, the shrinkage keeps the
  total weather weight within [(1 − ρ)/2, (1 + ρ)/2], ρ = K_eff / (K_eff + 8). With the carrier candidate the prior
  weather mass is 3/7 once all seven candidates are supported, and the general bound of P8 applies.

The weights are reported as `level_weights` ([Output fields](#output-fields)); these support and shrinkage properties
are P7–P9 in [theory/PROOFS.md](../theory/PROOFS.md). How many complete pseudo-origins the test windows had is in
[EVALUATION.md](EVALUATION.md#history-length-and-the-anchoring-record) (`K_eff` 9–12 on every test window).

Code: `ankyra_lite/anchor/history/` (frozen reference estimator), `ankyra_lite/anchor/core.py`
(`level_with_model_candidate`).

**The temperature-anomaly scale `temp_sigma_std`.** One positive number per population, in units of 10 °C (0.25 means
2.5 °C), supplied by the caller; the package checks only that it is a finite positive number and cannot check when or
how it was fitted. How the study computed it for ankyra-lite (the first release's arithmetic on the complete days
before the training cut) is in [DATA.md](DATA.md#the-input-format).

#### The category Commercial has no prior

The frozen configuration holds a prior signature curve for Industrial, Office, Public and Residential. It holds none
that the code reads for Commercial: the fifth curve of `ankyra_lite/anchor/history/frozen_config.json` is stored under
the key `group_4`, and the estimator looks curves up by category name. A unit labelled Commercial therefore gets a
temperature signature of zero:

- its three weather-adjusted level candidates equal the unadjusted ones, and its weather-response daily path is zero;
  the rest of the forecaster, the carrier included, is unaffected;
- this is the behaviour that was evaluated: 13 BDG2 units of the study carry the category Commercial
  ([DATA.md](DATA.md#1-bdg2-2017-building-meters-usa--europe));
- the copied layer is not edited, so it is kept. For a new Commercial unit the choice is between this behaviour and
  one of the four categories that have a prior (the category is also an input of the carrier).

### Centred daily path

Seven candidate paths for the 31 daily means, with their keys in `diagnostics["day_weights"]`
([Output fields](#output-fields)):

- `cur`, the path implied by the level construction (the 8-week day-type offsets plus the weather-response path times
  the level's total weather weight);
- `c_w8`, `c_w4`, `c_w2`, the day-type offsets from the last 8, 4 and 2 weeks;
- `c_ann_type`, the day-type offsets of the annual window;
- `p_ann_cal`, that window's daily means in date order;
- `g_w`, the weather-response path.

They are combined with inverse-square weights shrunk towards equal weights, as for the level, with K0 = 2 and up to
12 pseudo-origins. The rule differs from the level's in three places (`ankyra_lite/anchor/history/_day.py`,
`weights_c`):

- a candidate's error at one pseudo-origin is the root mean square, over the 31 days, of its centred-path error, and
  its score s_j is the **median** of these values over the pseudo-origins (at least two), not their root mean square
  (the configuration calls it `"mad"`, a historical label, not a centred median absolute deviation);
- K_eff counts the pseudo-origins at which `cur` has an error;
- when no candidate has two errors, all weight goes to `cur`, not to equal weights.

The annual candidates get weight zero at an origin whose own annual window is incomplete, and the rest are
renormalised. The result sums to zero, so it redistributes energy between days without changing the level.

### Within-day shape

**Carrier shape.** w^C is the day-demeaned carrier forecast at the origin (`foundation_within_day_kw`).

**Analog-day shape.** For each forecast day d, the unit's own *analog days* are the complete pre-origin days with the
same day type, within ±14 days of the same day of year, in the same daylight-saving state (`dst_region`), and preceded
by a complete 744-hour window. Up to eight are kept: those closest in daily-mean temperature to the climatology of day
d (ties: the more recent). Candidates come from the whole pre-origin record, not only from the 1,344-hour context. With
A_d the kept analog days, x̄_j the mean of day j, s_j the standard deviation of the 744 hours before day j and s_o that
of the 744 hours before the origin (each floored at max(0.01 |mean|, 10⁻³) kW),

  S(d, h) = (s_o / |A_d|) · Σ_{j ∈ A_d} (x_j(h) − x̄_j) / s_j .

Each analog day's shape is thus rescaled from the variability before it to the variability before the origin. A day
with fewer than four analogs keeps w^C for that day. If max |S| exceeds three times the largest absolute value of the
1,344-hour context, the whole window keeps w^C (a guard against scale floors on near-constant records;
`analog_kept` = False).

**Anchoring.** One weight ω is estimated for the whole window:

  w(d, h) = w^C(d, h) + ω · (S(d, h) − w^C(d, h)),   ω = min( clip(λ̂, 0, 1) · n / (n + 2), ½ ),

  λ̂ = Σ_q ⟨S_q − w^C_q, y_q − w^C_q⟩ / Σ_q ‖S_q − w^C_q‖²,

with the inner products and norms taken over all 744 hours of each window, estimated over the unit's n ≤ 3 completed
pseudo-origin windows q = o − 744k′, k′ = 1, 2, 3 (complete context and complete target month, and a carrier forecast
at q). S_q is the analog shape built from data before q, w^C_q the carrier's within-day block issued at q and y_q the
realised within-day block. The weight is a least-squares weight on the disagreement between the two shapes, shrunk
towards the carrier (zero) and capped at one half. No completed pseudo-origin, or no disagreement, gives ω = 0 and the
carrier's within-day block. `within_trust` reports ω and `within_pseudo_pairs` reports n.

The switch `single_trust=False` estimates one weight per lead block (days 1–7, 8–14, 15–21, 22–31) instead, with the
sums of λ̂ restricted to the hours of that block; `within_anchor=False` leaves the carrier's within-day block unchanged.
The evaluation's ablation rows use both ([EVALUATION.md](EVALUATION.md#what-the-parts-add)).

- Both w^C and S have zero daily means, so the pre-projection daily means are unchanged by the anchoring, and with
  them the level, the daily path and the energy readout (744 × level). The delivered trajectory max(F, 0) can differ
  where the projection binds, and so can a readout computed from it, such as the peak readout.
- The three pseudo-origin forecasts are among the six that the layer already reads for the handover: no further
  carrier forecast is needed.
- The analog shape needs a previous year at the same dates. Records shorter than about 13 months give no analogs and
  the block stays the carrier's.

How large the trust is on the test windows, and on how many of them it acts, is in
[EVALUATION.md](EVALUATION.md#within-day-anchoring-where-it-acts) (`results/within_day_trust.csv`).

Code: `ankyra_lite/anchor/analog.py` (`AnalogShapes`, `within_trust`, `anchored_within_day`),
`ankyra_lite/anchor/core.py` (`forecast`).

**History-only configuration.** `ankyra_lite.anchor.forecast(..., foundation=None)`, which
`ankyra_lite.forecast(history, None, ...)` calls, returns the history side alone: the six-candidate level and the daily
path, with the same-day-type within-day default (the eight most recent context days of each type), no carrier candidate,
no handover and no analog-day shape. The off-state and micro-load rules return the carrier's forecast, so they do not
apply in this configuration, and no state rule is applied to it. It is a reduced configuration for ablation and for use
without a carrier, and it is the comparator of contrast (ii), ankyra-lite against the history-only configuration
([EVALUATION.md](EVALUATION.md#what-the-parts-add)). Contrast (ii) measures the incremental performance of the
carrier-enabled configuration. It does not isolate the information in the carrier's forecast, because the history-only
configuration uses the day-type default rather than the analog-day shape and omits the state rules. The evaluation
therefore also uses a *carrier-free control*: the unchanged layer called with this configuration's own forecasts, at the
origin and at the six pseudo-origins, in the carrier's place (as the `foundation` mapping), so that the control runs
the same procedure as the carrier-enabled configuration. It is a control of the evaluation, not a mode of the package
([EVALUATION.md](EVALUATION.md#contrast-ii-by-block-and-the-carrier-free-control)).

### Week-by-week handover

At the origin, let m^H_d be the history-side daily means (the seven-candidate level, which includes the carrier
candidate, plus the daily path) and m^C_d the carrier's daily means. For the weeks w = days 1–7, 8–14, 15–21 and 22–31,

  m_d = (1 − α_w) · m^H_d + α_w · m^C_d,   α_w = (n · λ̂_w + K0/2) / (n + K0),   K0 = 2,

so α_w = (n λ̂_w + 1) / (n + 2). λ̂_w is the least-squares combination weight, clipped to [0, 1]:

  λ̂_w = clip( Σ_k ⟨y_k − m^H0_k, m^C_k − m^H0_k⟩_w / Σ_k ‖m^C_k − m^H0_k‖²_w , 0, 1 ),

with the sums over the days of week w. It is estimated from the n ≤ 6 completed pseudo-origin pairs of the unit,
where:

- y_k is the realised daily means of the pseudo-target (a day of a partly observed pseudo-target counts only with at
  least 12 observed hours; [Short-gap rule](#short-gap-rule));
- m^C_k is the carrier's daily means issued at pseudo-origin k;
- m^H0_k is the **fixed division**'s daily means at that pseudo-origin: the six-candidate level plus the daily path,
  *without* the carrier candidate and without the handover, estimated from the data before it.

The weight is a least-squares weight rather than an inverse-MSE weight. Errors shared by the two sources pull
inverse-MSE weights towards ½, whereas the least-squares weight depends only on where the sources differ (P10 in
[theory/PROOFS.md](../theory/PROOFS.md)).

The weights are estimated on the fixed division and applied to m^H, which includes the carrier candidate. Using the
fixed division at the pseudo-origins avoids nested pseudo-origins. If no pair is available, α_w = ½; if the two sources
do not differ on a week of the pairs, λ̂_w = ½ for that week. The weights are reported as `lead_week_weights`, the
number of pairs as `pseudo_pairs`, and the fixed division at the origin as `fixed_division_daily_means_kw`.

On the test windows, the handover ablation does not separate the four weekly weights from one weight per window on
any of the nine populations (the point estimates favour the single weight by 0.1–0.5%); mixing at a fixed half weight
gives most of the gain, and the per-unit weights add to it on two populations resolvably
([EVALUATION.md](EVALUATION.md#what-the-handover-contributes), `results/handover_granularity.csv`). The weekly form is
kept as the copied layer computes it.

The forecast is F(d, h) = m_d + w(d, h), projected onto F ≥ 0. It reads seven carrier forecasts per window (one at the
origin, six at pseudo-origins), each the average of the six seeds (three in version 0.1).

Code: `ankyra_lite/anchor/core.py` (`lead_week_weights`, `lead_week_transition`, `forecast`).

### Short-gap rule

Every quantity above that is estimated from completed pseudo-origins (the level weights, the handover weights, the
within-day trust) needs, at each pseudo-origin o − 744k, a complete 1,344-hour context and a complete 744-hour target
month. Real meter records have short gaps, and a single missing hour inside those 2,088 hours would remove the whole
pseudo-origin. With `gap_tolerance=True` (the default) the layer tolerates two things, both confined to the pre-origin
record:

1. **Short gaps are interpolated.** A gap of at most 6 missing hours is replaced by the straight line between its two
   observed neighbours, provided the gap and both neighbours lie inside one 744-hour block counted back from the
   origin. A pseudo-origin is a block boundary, so no interpolated hour before a pseudo-origin uses a value at or after
   it. Gaps that touch the origin, the start of the record or a block boundary, and longer gaps, stay missing.
2. **A pseudo-origin's target month may be partly observed.** It qualifies when at least 90% of its 744 hours are
   observed; its window mean is the mean over the observed hours, and in the handover sums a day counts only with at
   least 12 observed hours. This applies to the seven-candidate level and to the handover pairs only.

**What the rule reaches.** Rule 1 is not confined to the pseudo-origins. After the off-state and micro-load checks the
whole pre-origin record that the layer reads is replaced by the interpolated copy, and every later step reads that
copy:

- the contexts and targets of the pseudo-origins;
- the origin's own historical candidates (for example the long-history mean and the annual window) and the fit of
  the temperature signature;
- the pool of analog days.

Rule 2 is narrower than rule 1. The weights of the daily path, the six-candidate level of the fixed division and the
within-day triples still require a completely observed target month, counted on the interpolated copy. The interval
readout (`ankyra_lite.anchor.readouts.residual_quantiles`, and `ankyra_lite.readouts.interval` built on it) is a
separate call on the record the caller passes and applies neither rule.

The context at the origin is never filled by the package. The normal carrier-inference path requires completeness
and raises an error for a gap. A precomputed mapping can instead enter the inherited off-state branch before that
validation, using the observed last-week maximum; its seasonal-naive fallback zero-fills missing last-week values.
This exception is an input-path distinction; the inherited rule does not establish that missing hours represent a shutdown. The off-state and micro-load rules read the raw context. `anchor.pseudo_origin_contexts`
returns the interpolated pseudo-origin contexts; `carrier_forecasts` uses it only to choose the pseudo-origins and
builds each carrier forecast from the record as supplied ([Pseudo-origins](#pseudo-origins)).
`ankyra_lite.fill_short_gaps` (= `anchor.fill_short_gaps`) applies rule 1 to a user's own record before the call
([DATA.md](DATA.md#how-much-history)); the frozen reference estimator is unchanged and receives a partially observed
target with its missing hours set to the observed mean. `gap_tolerance=False` switches both parts off; a record
without gaps gives the same forecast either way.

One detail of `gap_tolerance=False` with a callable `foundation` (a function from contexts to forecasts, which
`ankyra_lite.forecast` does not use; it passes a mapping): the callable is still given the interpolated pseudo-origin
contexts (`anchor.forecast` calls `pseudo_origin_contexts` with its default). On a record with short gaps it can
therefore be asked for a few pseudo-origins that the layer then does not read; the result is unchanged and the only
cost is the extra calls. With a mapping of precomputed forecasts nothing extra is computed.

Code: `ankyra_lite/anchor/core.py` (`fill_short_gaps`, `GAP_MAX_H = 6`, `TARGET_MIN_OBS = 0.9`,
`pseudo_origin_contexts`, `forecast`) and `ankyra_lite/anchor/history/api.py` (the partially observed pseudo-targets).

### Arguments used in the evaluation

`group`: the unit's category; `temp_sigma_std`: the population's value fitted on the days before the training cut;
`dst_region`: the unit's region; `observed`: the finite load hours; every other argument at its default
(`gap_tolerance=True`, `single_trust=True`, `within_anchor=True`, `micro_load_rule=True`).

## State windows and the seasonal-naive rule

The anchoring layer has two state rules, both on load before the origin:

- **off-state:** the observed load of the last 168 hours is at most 10⁻⁶ kW. The test is on the largest observed value
  of those hours: unobserved hours are ignored, and the value itself is compared, not its absolute value;
- **micro-load:** all 1,344 context hours are within 10⁻³ kW of zero (max |load| ≤ 10⁻³ kW, compared in double
  precision). The threshold reuses the floor of the normalisation scale s0 ([Level](#level)); the context must be
  completely observed. A return under the micro-load rule alone runs the estimator's history, category and
  temperature-scale checks; a return under the off-state rule does not. `micro_load_rule=False` switches the rule off.

Both thresholds are absolute values in kW, like the scale floor: loads must be supplied in kW, and rescaling a record
across a threshold changes which branch is taken. The micro-load rule is a sufficient condition, not a derived
boundary: a record slightly above the threshold is not covered, although its scale may also be at the floor, and the
rule cannot foresee a restart within the forecast month. Neither rule is a certificate that the inputs were validated.

On such a window the layer returns the carrier's forecast unchanged (without the projection onto nonnegative load), and
the weights, counts, the fixed division and the analog fields keep their empty defaults because the estimator is not
run. That return rests on the premise that the carrier forecasts a switched-off or near-zero meter well, which held for
foundation models and is not guaranteed for a trained carrier. ankyra-lite therefore applies its own rule on these
windows (`ankyra_lite.forecast`, default `state_rule="seasonal_naive"`): the delivered forecast is the last observed
week repeated over the 744 hours (missing hours as 0), its level readout is that forecast's mean, the continuation layer is bypassed, and its peak readout
is read from it. `state_rule=None` keeps the layer's own return. The rule was written after an earlier scoring of the
test windows ([EVALUATION.md](EVALUATION.md#identity-of-the-evidence)); the state windows of each population are counted
in [EVALUATION.md](EVALUATION.md#state-windows-and-the-three-near-zero-bdg2-meters).

## The continuation layer

`ankyra_lite.continuation` corrects the first hours and days of the delivered trajectory toward the last observed
hours before the origin. It runs after the anchoring layer, the state rule and the readouts
(`forecast(..., continuation=True)`, the release default), has four constants fitted once and frozen, and trains
nothing at forecast time.

**Why.** The carrier forecasts the month from daily and monthly tokens; it has no path that continues from the last
observed hours. On the development windows this shows as a clear curve: the log error of the anchored forecast at
hour 0 rose almost one for one with the log anomaly of the last observed hours against the forecast's own curve (a
slope of about 1, explaining 40–48% of the squared log error at that hour), and the slope was still about 0.6 at hour
11 and 0.5 at hour 47. The error curve has two parts, a fast one of a few hours and a slow one of about a week; from
the second week on the forecast already followed recent level changes about as the load did, so the slow part needs
no more than a week. The zero-shot foundation models, which read the last hours directly, lead on forecast day 1
([README](../README.md#where-ankyra-lite-is-weak)).

**Why after the anchoring layer.** A change inside the carrier reaches the delivered forecast only in part: on the
development windows 0.6 to 0.8 of a change in the carrier's forecast passed through the anchoring layer, whose
history side pulls the forecast back toward the unit's past. Placed after the anchoring layer, the correction is not
diluted, and the anchoring layer stays byte-identical.

**The layer, one window** (origin o; F(h), h = 0 … 743, the trajectory before the layer):

1. **Bypass.** On a state window (off-state or micro-load) the trajectory is delivered unchanged, F′ = F.
2. **Scale.** s = mean of the observed load in the 672 hours before o (floor 10⁻³ kW); ε = 0.01 s.
3. **Reference day.** d* = the first forecast day d = 0 … 6 whose workday class equals that of the last observed day;
   day types 0–4 are workdays and 5, 6 and 7 (Saturday, Sunday, holiday) are not, each day's type read at its 12th
   hour (the same calendar and hour as the anchoring layer and the peak readout). When none of the seven days has
   that class, d* = 6 (a release rule of the layer's design; the study's development code used d* = 0, which agrees
   wherever a day of the class exists, as on every development window; on the test windows the fallback was used on
   33 EWELD windows, `results/continuation_record.csv`).
4. **Two anomalies**, each clipped to [−1, 1]:
   - fast: x_f = ln[(mean y(o − 3 … o − 1) + ε) / (max(mean F(24d* + 21 … 24d* + 23), 0) + ε)], the last 3 observed
     hours against the forecast's same clock hours on the reference day; 0 when one of the 3 hours is unobserved;
   - day: x_d = ln[(mean y(o − 24 … o − 1) + ε) / (max(mean F(24d* … 24d* + 23), 0) + ε)], the last observed day
     against the forecast's reference day; 0 when one of the 24 hours is unobserved;
   - negative loads: when a span's mean plus ε is at or below 0 the logarithm is undefined, and that anomaly is set to 0
     and flagged (`nonpositive_fast`, `nonpositive_day`) instead of raising an error. This release rule was added after
     the study, whose code raised an error there; no study window and no test window reached it.
5. **Correction.** Δ(h) = κ₁ x_f e^(−h/ρ₁) + κ₂ x_d e^(−h/ρ₂) and F′(h) = F(h) e^(Δ(h)).

| Constant | Value | What it sets |
|---|---|---|
| κ₁ | 0.479 | weight of the fast anomaly at the first hour |
| ρ₁ | 12 h | decay time of the fast term |
| κ₂ | 0.330 | weight of the day anomaly at the first hour |
| ρ₂ | 168 h | decay time of the day term |

**Where the constants come from.** ρ₁ and ρ₂ were chosen from the grids {2, 3, 4, 6, 8, 12} h and {72, 120, 168} h,
and κ₁ and κ₂ fitted, in one pooled fit on the earlier development windows of six populations (the unit-equal
whole-window squared-error ratio), on the study's development-cut products of the carrier; κ was rounded to three
decimals (fitted 0.47897 and 0.33035; `continuation.FULL_PRECISION`). The same four numbers are used for every
population, for the three test populations without development windows (BDG2, GoiEner non-household, the households)
and for the external tests; they are never refitted. `continuation.DIGEST` (SHA-256 of the constants and the rule)
identifies them, and every forecast's record carries the digest of the constants it used (O9).

**How the form was chosen.** The day anomaly uses the last 24 hours, not the last 3: the 3-hour anomaly also carries
the forecast's within-day shape error at hours 21–23, and for spiky households a day mean is less noisy. A first fit
with a slow decay of 744 h lowered the peak readout on the later development windows by 1.3%, and the slow decay was
then limited to at most 168 h. The form was thus chosen on the development windows, partly after reading them. The
development check of the frozen constants, on the later development windows with six-seed carriers of recipe Rw
trained to a development cut (+0.48% whole window, +9.49% day 1, +23.6% hours 0–2; re-used evaluation,
`results/recipe_history.csv`, `Rw6_c vs Rw6`), tests whether the constants carry over to that carrier, not the form;
those windows had served to choose the form, so it is not an independent confirmation.

**What it guarantees** (theory/PROOFS.md, O1–O9; under each property's stated assumptions):

The exponential bounds are real-arithmetic statements. The maintenance implementation retains the original
`F64 * exp(delta)` order, restores input zeros including signed zero, and rejects a non-finite non-state correction
or final output with `ValueError`; it does not clip custom constants. Nonzero floating-point values can underflow
to zero. Default constants and ordinary finite inputs retain their arithmetic path.

- a nonnegative trajectory stays nonnegative and exact zeros stay zero (O1);
- each hour moves by a factor between e^(−E(h)) and e^(E(h)), E(h) = κ₁e^(−h/12) + κ₂e^(−h/168): between 0.445 and
  2.246 at the first hour, 0.698 and 1.431 at the last hour of day 1, 0.885 and 1.130 from day 8, and 0.996 and 1.004
  at the last hour (O2);
- it reads only the 672 hours before the origin and the day types at o − 12 and o + 12 + 24d, d = 0 … 6 (O3);
- state windows, and a window with one of its last 3 hours unobserved, are returned bit for bit (O4);
- with `continuation=False` the layer is off and the trajectory before it is delivered; `continuation.ZERO` gives the
  same trajectory bit for bit on every window with a finite trajectory (O5). Reproducing a version 0.1 forecast also
  needs the 0.1 checkpoints (recipe R, seeds 0–2) and the same inputs;
- energy, level and peak are read before it (O6, [Readouts](#readouts));
- a forecast whose reference day already matches the last observed hours is left unchanged, and two anomalies of the
  same sign move the whole window the same way (O7);
- it does not depend on the unit of the load while the scale stays above its floor (O8).

**Limits.** The normal carrier-inference path requires the 1,344 hours before the origin completely observed.
The precomputed-mapping off-state exception above bypasses continuation as a state window; direct continuation calls
can also receive missing-hour inputs. The history-only configuration (`forecast(history, None, ...)`) is never
continued, because it was not evaluated with the layer. The bound limits how far an hour moves, not its error: when the
load changes level right after the origin the correction can point the wrong way.

**What it did on the test windows** (`results/continuation_layer.csv`, `results/continuation_record.csv`; descriptive,
[EVALUATION.md](EVALUATION.md#the-continuation-layer)): with the layer on against off on the same carrier, +0.46%
[+0.33, +0.70] in whole-window hourly error, +11.1% on day 1 and +23.4% in hours 0–2 (1,891 units of the nine
populations pooled; with populations equally weighted +0.43%, +12.77% and +27.30%, `recipe_history.csv`). 326 state
windows were bypassed; the fast anomaly was clipped on 6.4% and the day anomaly on 4.2% of the other windows; 33
windows used the fallback d* = 6; no span was unobserved or nonpositive; the largest |Δ| was 0.809.

`ForecastResult.continuation` holds what the layer did on the window (`ContinuationRecord`):

| Field | Meaning |
|---|---|
| `bypass` | `"state window"` (F′ = F), `"missing hours"` (an unobserved hour among the last 3; x_f = x_d = 0, F′ = F), `"nonpositive load"` (both anomalies set to 0 by the negative-load rule, or the fast one and the day span unobserved; F′ = F) or None |
| `d_star`, `matched` | the reference day 0 … 6, and False when no day of the first week shared the last observed day's class (d* = 6) |
| `scale_kw`, `eps_kw` | s and ε |
| `x_fast`, `x_day` | the clipped anomalies (0 when one of their hours is unobserved, when their span mean plus ε is at or below 0, or on a state window) |
| `raw_fast`, `raw_day` | the anomalies before clipping (NaN when not computed) |
| `missing_fast`, `missing_day`, `clipped_fast`, `clipped_day` | an hour of the span was unobserved; the anomaly was clipped |
| `nonpositive_fast`, `nonpositive_day` | the span's mean plus ε was at or below 0, so its anomaly was set to 0 |
| `delta` | (744,) Δ(h) |
| `max_abs_delta` | max \|Δ(h)\| (0 on a state window) |
| `constants`, `digest` | the constants used and the digest of them with the rule |

`continuation.apply(trajectory_kw, load_kw, day_types, origin)` applies the layer to any trajectory, and
`continuation.apply_batch` to many windows at once (the evaluation path, applied to the stored anchored products).

## Readouts

**Readout rule A.** Energy, level and peak are read before the continuation layer: for fixed carrier forecasts,
inputs, anchoring and state rule, toggling the layer keeps them bit for bit (O6). The rule was fixed before the test
windows were read for this version, and follows the anchoring layer's precedent of reading energy before its
projection. Reading energy and peak from F′ instead had moved them on the development windows by −0.5% to +0.9%
(energy) and −1.3% to +0.6% (peak). Its cost is the disclosure below: the delivered hourly curve's mean is not the
energy readout.

**Energy.** `ForecastResult.energy_kwh` = 744 × `level_kw` kWh for loads in kW. Outside state windows `level_kw` is the
anchoring layer's level, the mean of the 31 daily means after the handover, read before the projection; on a state
window with the seasonal-naive rule it is the mean of the delivered seasonal naive, which needs no projection. It is
read before the continuation layer.

- The projection max(F, 0) cannot increase any hour's absolute error when the truth is nonnegative (P11).
- It raises the energy of the trajectory before the continuation layer by Σ max(−F, 0), which is known at the origin
  (`ankyra_lite.readouts.projection_energy_increase(r)`; on the raw trajectory,
  `anchor.readouts.projection_energy_increase_kwh`).
- The continuation layer then changes the delivered trajectory's energy by Σ F′ − Σ F, also known at the origin
  (`ankyra_lite.readouts.continuation_energy_change(r)`), bounded by −Σ F(h)(1 − e^(−E(h))) and Σ F(h)(e^(E(h)) − 1):
  between −7.34% and +9.31% of Σ F for a flat trajectory, and between −55.5% and +124.6% for any nonnegative
  trajectory, the extremes approached only when almost all of the energy lies in the first hours (rounded outward; O6).
- So Σ `trajectory_kw` = `energy_kwh` + `projection_energy_increase` + `continuation_energy_change`. These terms may be
  zero or cancel, so the delivered curve's sum generally, but not necessarily, differs from the energy readout.
- **Disclosure** (`results/continuation_record.csv`, row `all nine populations`; windows the layer acts on): the relative
  difference between the delivered curve's mean and the level readout had a median of 0.77%, a 99th percentile of 7.8%
  and a maximum of 310% (GoiEner households; GoiEner non-household 89.9%; 6.5–13.3% on the other populations). The
  layer's own energy change lay between −14.8% and +33.4%; a larger gap is a window where the projection binds, so that
  the trajectory before the layer already held more energy than 744 × level. On the development windows used to
  choose the layer, the relative change of the curve (Σ F′ − Σ F)/Σ F had a median of 0.6%, a 99th percentile of 7% and
  a maximum of 9%.
- No map can simultaneously keep the energy readout, return nonnegative load and guarantee no increase in error (P12).

The evaluation scores curve energy as 744 × the window mean of the trajectory before the continuation layer
(`trajectory_before_continuation_kw`) for ankyra-lite, and of each baseline's own trajectory; the level readout
744 × `level_kw` is a sensitivity. The two differ only on the windows where the projection binds
([EVALUATION.md](EVALUATION.md#overview)). The continuation layer's own effect on the delivered curve's energy is
reported separately (`results/continuation_layer.csv`, the `energy of the delivered trajectory` rows).

**Peak.** Let L̂_d be the forecast mean of day d, τ_d its day type and F̄ the forecast window mean. Let
a_j = max_h x_j(h) − x̄_j be the excursion of context day j above its own mean (x_j(h) is the load of hour h of that
day), and A_t the largest excursion among the four most recent context days of type t. If the context has no day of
type t, A_t is the largest excursion of all its days; if it has no holiday, the holiday entry takes the Sunday value.
Then

  U = max_d (L̂_d + A_{τ_d}),   P̂_κ = F̄ + κ (U − F̄)   (κ = 1).

`ForecastResult.peak_kw` is this readout of the trajectory before the continuation layer,
`trajectory_before_continuation_kw` (`ankyra_lite.forecast.peak`, which calls `anchor.readouts.peak_readout`; the type of
each origin-aligned day is read at its 12th hour). On a state window it is read from the seasonal naive. With the layer
off the two trajectories are the same array. The operator has four properties:

1. **Motivation.** For exact conditional means, max_{d,h} E[y(d, h)] ≤ E[max_{d,h} y(d, h)]. Fitted forecasters are
   median-type summaries, so this motivates the operator rather than guaranteeing it (P14).
2. **Scalar form.** Because maxima commute, U equals the hour-wise envelope max_{d,h} (L̂_d + max_j a_j(h)). Also
   U ≥ max_d L̂_d ≥ F̄ (P15).
3. **Median property.** Suppose daily levels are constant within type, excursions are drawn independently from each
   selected sample, and a type attaining the envelope has m samples and n future days with ((m − 1)/m)ⁿ < ½. Then the
   median of the future maximum equals U exactly. Four samples and three repetitions suffice (P17).
4. **Error decomposition.** With M the realised peak of the window and d* the day on which it occurs,
   U − M = ℓ(e) + [b_{d*}(F) − b_{d*}(y)] + [A_{τ_{d*}} − a_{d*}(y)] + Δ_sel with Δ_sel ≥ 0. Level and daily-path gains
   therefore carry into the peak (P16). Whether they do on the test windows is measured against the same readout on
   the carrier alone ([EVALUATION.md](EVALUATION.md#overview), `results/peak_readout.csv`).

**Interval.** Not part of the forecaster and not a field of `ForecastResult`. `ankyra_lite.readouts.interval(history,
r, group=..., temp_sigma_std=...)` returns quantile trajectories at the levels 0.05, 0.1, 0.5, 0.9 and 0.95 (rows 1
and 3 bound the 80% interval) and the number of pseudo-windows used; it makes the three calls of
`anchor.readouts` (`residual_quantiles`, `origin_scale`, `interval_bands`). The residuals come from the history-only
forecaster (historical level and daily path with the same-day-type within-day default, no carrier), not from
ankyra-lite's own forecasts. Its hourly residuals at up to 12 completed pseudo-origins are normalised by each
pseudo-window's origin scale and pooled by hour of day and workday status (day types 0–4 against 5–7). Their empirical
quantiles are multiplied by the origin's scale and added to the forecast, `trajectory_kw`, i.e. after the
continuation layer when it is on, because the interval belongs to the hourly forecast; interval tables of 0.2.0
therefore differ from those of 0.1 where the layer acts. Three details of the implementation:

- a class and hour with fewer than five residuals takes the quantiles of the other class at that hour (NaN when
  neither has five, always the case with no complete pseudo-window);
- the quantile trajectories are clipped at zero and made non-crossing across levels;
- the band is not centred on the point forecast, because the residual quantiles of the history-only forecaster need
  not straddle zero (a bound can lie on the same side of the point forecast as the median).

A pseudo-window counts only when its context and target month are completely observed in the record passed (the
short-gap rule does not apply). In the evaluation the interval is placed around ankyra-lite's delivered trajectory as
a description; it was designed for the first release's forecaster and not re-tuned for this carrier
([EVALUATION.md](EVALUATION.md#intervals)).

Proofs, counterexamples and an optional day-level projection are P11–P18 in [theory/PROOFS.md](../theory/PROOFS.md).
Code: `ankyra_lite/readouts.py`, `ankyra_lite/forecast.py` (`peak`), `ankyra_lite/anchor/readouts.py`; the
properties as operators: `theory/operators.py`.

## Output fields

`ankyra_lite.forecast` returns a `ForecastResult`. All loads are in kW. Its own fields:

| Field | Meaning |
|---|---|
| `trajectory_kw` | the 744 delivered hourly values: with the continuation layer on, F′ = F e^Δ; otherwise the trajectory before the layer (the same array) |
| `trajectory_before_continuation_kw` | the anchoring layer's trajectory (nonnegative), or the seasonal naive on a state window with the default state rule. The peak is read from it; level and energy are read at the same step, before the continuation layer (the level before the nonnegativity projection) |
| `level_kw` | the delivered level readout: the anchoring layer's level (the mean of `daily_means_kw`, before the projection onto nonnegative load), or the mean of the seasonal naive on a state window. It is read before the continuation layer and need not equal the mean of `trajectory_kw`. It is not the seven-candidate level of the [Level](#level) section, which is the history side's input to the handover |
| `energy_kwh` | the energy readout, 744 × `level_kw` |
| `peak_kw` | the peak readout of `trajectory_before_continuation_kw` ([Readouts](#readouts)) |
| `state_window` | the anchoring layer reported `off_state` or `micro_load` (always False for the history-only configuration) |
| `state_rule` | `"seasonal_naive"` when that rule replaced the layer's return, else None |
| `continuation` | the `ContinuationRecord` of the layer ([The continuation layer](#the-continuation-layer)); None when the layer is off and for the history-only configuration |
| `carrier` | {k: (744,)}: the carrier forecasts passed to the layer, issued at o − 744k ({} for the history-only configuration) |
| `anchored` | the anchoring layer's own `AnkyraForecast` |

Every other attribute is read from `anchored` (`ForecastResult.__getattr__`):

| Field | Meaning |
|---|---|
| `daily_means_kw` | the 31 daily means after the week-by-week handover |
| `within_day_kw` | the final within-day block (zero mean on each day), with the analog-day anchoring where it acts |
| `foundation_within_day_kw` | the carrier's own within-day block at the origin, unmodified (`anchor.blocks.within_day` of `carrier[0]`). On state windows `within_day_kw` equals it |
| `analog_shape_kw` | the unit's analog-day shape S (a day with fewer than four analogs carries the carrier's shape); None when it was not built |
| `within_trust` | the weight ω on the analog-day shape, listed for days 1–7, 8–14, 15–21, 22–31 (four equal values; four separate weights with `single_trust=False`) |
| `within_pseudo_pairs` | the number of completed pseudo-origin windows behind `within_trust` (at most 3) |
| `analog_kept` | False when the whole-window guard rejected the analog shape (or none was built) and the window keeps the carrier's shape |
| `lead_week_weights` | the four handover weights α_w on the carrier's daily means |
| `pseudo_pairs` | the number of completed pseudo-origin pairs behind `lead_week_weights` (at most 6) |
| `level_weights` | the weight of each level candidate (keys below) |
| `fixed_division_daily_means_kw` | the daily means of the fixed division: the six-candidate level plus the daily path, without the carrier candidate and without the handover |
| `off_state`, `micro_load` | which of the layer's two state rules fired, if any |
| `anchoring_record` | `carrier_level_weight` (= `level_weights["fm"]`), `history_vs_carrier_log_ratio` and `pseudo_origins`: the unit's own pseudo-forecast record on the level, read before the forecast and not used by it. The log ratio is the smallest pseudo-origin RMS error among the six historical level candidates over that of the carrier's window mean (NaN with fewer than two such pseudo-origins); negative means the unit's history beat the carrier at its earlier origins. Its deciles against the realised gain are in [EVALUATION.md](EVALUATION.md#history-length-and-the-anchoring-record) |

`r.anchored.trajectory_kw`, `r.anchored.level_kw` and `r.anchored.energy_kwh` are the anchoring layer's own values; they
differ from `trajectory_before_continuation_kw` and the delivered readouts only on state windows with the seasonal-naive
rule, and `trajectory_kw` differs from them wherever the continuation layer acts.

**State windows.** On an off-state or micro-load window the fields read from `anchored` describe the layer's return,
not the delivered seasonal naive: `daily_means_kw`, `within_day_kw` and `foundation_within_day_kw` are the carrier
forecast's, and the weights, counts, `fixed_division_daily_means_kw`, the analog fields and `anchoring_record` keep their
empty defaults (`lead_week_weights` (), `level_weights` {}, `pseudo_pairs` 0, `within_trust` four zeros,
`within_pseudo_pairs` 0, `analog_kept` False, `analog_shape_kw` and `fixed_division_daily_means_kw` None,
`anchoring_record` {}), because the estimator is not run. The history-only configuration likewise has no carrier
fields: `level_weights`, `lead_week_weights` and `anchoring_record` are empty, `foundation_within_day_kw` and
`analog_shape_kw` are None, and `fixed_division_daily_means_kw` equals `daily_means_kw`.

**Keys of `level_weights`.** The first letter is the window of the mean, the second says whether it is
weather-adjusted (`u` unadjusted, `w` weather-adjusted).

| Key | Level candidate |
|---|---|
| `s_u`, `s_w` | the recent 7-day mean (short window) |
| `l_u`, `l_w` | the long-history mean over at most 8,760 hours |
| `a_u`, `a_w` | the mean of the 744-hour window one year earlier (annual) |
| `fm` | the carrier's own window mean |

The weights sum to one. A candidate with fewer than two pseudo-origin errors has weight zero, unless no candidate has
two; then the weights are equal.

**Keys of the daily-path weights.** The seven [daily-path candidates](#centred-daily-path) are weighted inside the
historical estimator. `ForecastResult` does not carry their weights;
`ankyra_lite.anchor.history.estimate_from_history(history, group=..., temp_sigma_std=...).diagnostics["day_weights"]`
does, with `diagnostics["level_weights"]` for the six historical level candidates before the carrier candidate is
added (call it on the record with `ankyra_lite.fill_short_gaps` applied to reproduce what the layer reads under the
short-gap rule).

| Key | Daily-path candidate |
|---|---|
| `cur` | the path implied by the level construction: the 8-week day-type offsets plus the weather-response path times the level's total weather weight |
| `c_w8`, `c_w4`, `c_w2` | day-type offsets from the last 8, 4 and 2 weeks |
| `c_ann_type` | day-type offsets of the annual window |
| `p_ann_cal` | the annual window's daily means in date order |
| `g_w` | the weather-response path |

Code: `ankyra_lite/forecast.py` (`ForecastResult`), `ankyra_lite/anchor/core.py` (`AnkyraForecast`),
`ankyra_lite/anchor/history/api.py` (`Estimate`).

## Inference cost

One forecast runs the carrier at up to seven issue times (the origin and up to six pseudo-origins) for each of the six
seeds, 42 network evaluations of 61,613 parameters, then the anchoring layer (the historical estimator at the origin and
its pseudo-origins, the analog-day shapes), the state rule, the peak readout and the continuation layer. Everything runs on one CPU thread;
nothing is downloaded. Measured with this repository's package and the released checkpoints on 6 Oslo test windows,
after one untimed warm-up, on one CPU thread without a GPU
([`results/cost_per_window.csv`](../results/README.md#cost)):

| Step | Time |
|---|---:|
| one forecast in all (carrier forecasts, anchoring layer with the state rule, continuation layer, peak readout) | 0.655 s per window |
| carrier forecasts at the origin and the pseudo-origins (7.0 issue times on average) × 6 seeds | 0.135 s per window |
| one carrier forecast (one issue time, one seed; mean) | 0.0032 s |
| anchoring layer call with the carrier forecasts supplied, with the state rule and the continuation layer | 0.520 s per window |
| peak readout | 0.0003 s per window |
| loading the six checkpoints (once) | 0.057 s |

The anchoring layer, not the carrier, takes most of the time (0.520 of 0.655 s); six seeds take about twice the
carrier time of three (0.075 s in 0.1). The continuation layer is a few vector operations on 744 values and is timed
inside the anchoring-layer row. A checkpoint is 0.2657 MB.

The stored evaluation forecasts were computed by the research code on a GPU; on the CPU the package reproduces them up
to floating-point differences between the two devices. On the timed windows the largest relative difference is
1.89 × 10⁻⁶ for the carrier forecasts, 1.09 × 10⁻⁶ for the trajectory before the continuation layer and 9.8 × 10⁻⁷ for
the delivered trajectory (the `check:` rows of `cost_per_window.csv`; see also `tests/test_carrier_equivalence.py`).

## Constants

The anchoring layer's constants are those of the copied layer, unchanged; none was set or tuned for ankyra-lite. "Where
it was fixed" refers to the development of the anchoring layer in the first release (provenance only):

| Constant | Value | Code | Where it was fixed |
|---|---|---|---|
| context / horizon | 1,344 h / 744 h | `blocks.CONTEXT`, `HORIZON` | before any evaluation |
| pseudo-origins (historical candidates / carrier candidate and handover) | 12 / 6, spaced 744 h | `frozen_config.json` (`level.K`), `core.K_FM`, `STEP` | development data: household aggregates / the Spanish development store |
| K0 of the level, daily-path and handover weights | 8, 2, 2 (towards equal weights, equal weights, ½) | `frozen_config.json`, `core.K0_WEEK` | development data; never changed afterwards |
| handover weeks | days 1–7, 8–14, 15–21, 22–31 | `core.WEEKS` | development data |
| within-day trust: pseudo-origins, shrinkage, cap | 3, n/(n + 2) towards 0, ω ≤ ½ | `analog.K_WITHIN`, `K0_WITHIN`, `W_CAP` | in a candidate family written down before the within-day rule was chosen; selected on development populations and pre-cutoff windows |
| analog days: window, kept, required, guard | ±14 days of year, 8, 4, 3 × max \|context\| | `analog.WIN`, `KMAX`, `KMIN`, `SANITY` | with the first version of the analog-day shape; never changed |
| scale floor | max(0.01 \|mean\|, 10⁻³) kW | `history/_eo.origin_scale`, `analog.floor_of` | the historical estimator |
| off-state threshold | last 168 h at most 10⁻⁶ kW | `core.ZERO_KW` | the pre-existing zero-load threshold |
| micro-load threshold | 10⁻³ kW on the magnitude of the load, over the 1,344-hour context | `core.MICRO_KW` | the pre-existing scale floor; the rule was written after a BDG2 result of the first release had been seen (post hoc) |
| short-gap rule: longest interpolated gap; observed share of a pseudo-origin's target month | 6 h; 0.9 | `core.GAP_MAX_H`, `TARGET_MIN_OBS` | a specification written before implementation, after the first release's UNICON test had shown few completed pseudo-origins |
| peak envelope | 4 most recent same-type days, κ = 1 | `readouts.historical_excursions(k=4)`, `peak_from_daily_levels(kappa=1.0)` | the earlier peak readout of the method (P18) |
| temperature standardisation; signature hinges | (T − 15 °C) / 10 °C; every 5 °C from −10 °C to 40 °C | `frozen_config.json` | in the frozen configuration, with the signature prior |
| climatology prior: amplitude, strength | 10 °C, 60 | `history/_climate.py` | the historical estimator |

Constants in the code that [`results/constants.csv`](../results/README.md#constants) does not list:

- a day of a partly observed pseudo-target enters the handover sums with at least 12 observed hours (`core.forecast`);
- the interval readout searches 12 pseudo-origins (`readouts.K_INTERVAL`), needs at least 5 residuals in an hour-of-day
  and workday cell, and returns the levels 0.05, 0.1, 0.5, 0.9 and 0.95 (`readouts.LEVELS`);
- the same-day-type within-day default of the history-only configuration and the interval uses the 8 most recent days
  of each type (`K_RECENT` in `history/_eo.py`);
- the temperature signature needs 60 observed days in the last 365 and has shrinkage strength 3 (`signature_lambda`),
  and a weight needs two pseudo-origin errors (`min_pseudo_errors`); both are in `history/frozen_config.json`;
- the error scale of a level candidate is floored at 10⁻⁶ in the inverse-square weights (`core._weights`).

The climatology strength 60 is the ridge coefficient on the amplitude of the annual harmonic (`CLIM_PRIOR_STRENGTH`),
not a number of days; the data term it is weighed against is the sum of the squared cosine over the pre-origin days,
about half the number of days, so the prior counts as much as about 120 days of daily temperatures. The constants
cannot be varied without patching the copied files, so no sensitivity to them is measured in this repository.

The carrier's constants:

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
| stage 2: epochs, label boundary before the cut | 8, 24 h (recipes R and Rw) | `CONTINUATION_EPOCHS`, `ReturnSpan` |
| gap from the training cut to the test set | 186 days (6 × 744 h) | `evaluation/windows.py` |
| seeds | 0–5 (the release configuration); 0–2 for version 0.1 | `ankyra_lite._release.CARRIER_SEEDS` |

The loss weights of recipe Rw (`NoiseWeights`, fixed before 0.2.0; never tuned on test windows):

| Constant | Value | Code |
|---|---|---|
| noise estimate: context days read, pair lag, observed flag | day tokens 338–365 (pairs on days 345–365), 7 days, flag > ½ | `NoiseWeights.context_days`, `lag_days`, `observed_threshold` |
| minimum number of pairs | 168 | `NoiseWeights.min_pairs` |
| floor | the draw-weighted 25th percentile of σ̂² over the pool | `NoiseWeights.floor_quantile` |
| exponent | ω = max(σ̂², floor)^(−½) | `NoiseWeights.gamma` |
| normalisation | draw-weighted mean weight 1 per population | `noise_weights` |

The continuation layer (`continuation.CONSTANTS` and `continuation.RULE`, identified by `continuation.DIGEST`
= d514e3ec93a745b14950671297bf2f922fd512424a804e292377e1b0c4cc1dce):

| Constant | Value | Where it was fixed |
|---|---|---|
| κ₁, ρ₁, κ₂, ρ₂ | 0.479, 12 h, 0.330, 168 h | one pooled fit on the earlier development windows of six populations; ρ from fixed grids, κ rounded to three decimals |
| scale: hours, floor; ε | 672 h, 10⁻³ kW; 0.01 × scale | the layer's design |
| anomaly spans | the last 3 hours (fast), the last 24 hours (day) | the layer's design, on development windows |
| clip | ±1 | the layer's design |
| workday class; type hour | day type < 5; the 12th hour of each origin-aligned day | the anchoring layer's calendar convention |
| reference days; fallback | days 0–6; d* = 6 | the layer's design (the development code used 0; no development window was affected; 33 EWELD test windows used it) |
| nonpositive span mean (negative loads) | the anomaly is set to 0 and the window flagged | a release rule added after the study (the study code raised an error); not part of `RULE`; no test window reached it |

[`results/constants.csv`](../results/README.md#constants) lists every constant of ankyra-lite in one file.

## What is not claimed

- **Not a general-purpose model.** One carrier is trained per population and evaluated on that population's test
  windows only. A checkpoint applied to another building or population without retraining has not been evaluated.
- **Not a first test.** The test windows had been read before, and the copy pointer's direction, the seasonal-naive
  state rule and recipe R's second stage were chosen after reading them ([EVALUATION.md](EVALUATION.md#identity-of-the-evidence)).
  They had been read many times before 0.2.0 was read on them once, after its configuration and constants were
  frozen; that read sets direction only.
- **The continuation layer works at the start of the month.** Its gains are concentrated in the first hours and days
  (+23.4% in hours 0–2, +11.1% on day 1, +0.46% over the whole window; `results/continuation_layer.csv`, nine
  populations pooled); it is not a month-long correction, it was resolvably negative on Oslo's days 8–31, and it does
  not make ankyra-lite lead the zero-shot foundation models on forecast day 1.
- **Energy and peak are unchanged by toggling the layer on a fixed carrier.** They are read before it, by a rule fixed
  in advance; this does not make comparisons across carriers identical or prove noninferiority.
- **The constants were fitted on development windows and not on the populations without them.** BDG2, GoiEner
  non-household and the households had no development windows; the layer is applied to them unchanged.
- **No mechanism is established for the loss weights.** Their effect is of the order of the seed noise (+0.31% on the
  test windows, recipe Rw against R with six seeds), and down-weighting noisy windows also changes recency and sample
  diversity, which the study does not separate.
- **Six seeds reduce retraining variance; they do not remove it.** The seed spread is reported
  ([EVALUATION.md](EVALUATION.md#seeds)); the intervals of the main tables still resample units and months only.
- **No mechanism is proven by the recipe study.** The returned span adds same-season data for the short-record
  populations, but it also changes recency, sample diversity and the replay of older windows, and adds 8 epochs of
  training; the study's readings do not separate these.
- **Properties are not accuracy.** C1–C14, O1–O9 and P1–P20 hold under their stated input, arithmetic and support
  assumptions, not for arbitrary overflowing weights, pools or trajectories; they make
  the forecaster inspectable and bound what its parts can do. Accuracy is measured.
- **No intervals.** The interval of the evaluation is the anchoring layer's, centred on ankyra-lite's trajectory as a
  description; ankyra-lite was not designed or tuned for it.
- **No bitwise reproducibility of training.** GPU training is not bitwise repeatable; the package reproduces single
  optimisation steps, not the study's full runs.
- **Inherited limits of the anchoring layer.** Hourly data only, a 744-hour horizon only, a complete 1,344-hour context
  on the normal inference path (the mapping/off-state exception is documented above), and no temperature-signature prior for the category Commercial.
