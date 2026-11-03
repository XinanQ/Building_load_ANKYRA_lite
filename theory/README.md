# Theory: the exact properties behind ankyra-lite

[中文](README.zh-CN.md) · [Proofs](PROOFS.md) · [Operators](operators.py) · [Checks](test_operators.py) · [Carrier checks](test_carrier_properties.py) · [Weighting checks](test_weighting_properties.py) · [Continuation checks](test_continuation_properties.py) · [Back to the forecaster](../README.md)

This folder holds the mathematical properties that sit beside the forecaster. The forecaster is the package
[`ankyra_lite/`](../ankyra_lite/), and nothing in it imports this folder. The properties make the forecaster
inspectable; they are not evidence of accuracy, which is measured ([results/SUMMARY.md](../results/SUMMARY.md)).
There are three groups, one for each part of ankyra-lite:

- **P1–P20, the anchoring layer.** `ankyra_lite.anchor` is a byte-identical copy of the ANKYRA 2.2.1 package, so these
  are the properties, operators and checks of the ANKYRA 2.2.1 theory, with the imports adapted. They explain and
  bound the anchoring layer:
  - why a month-ahead forecast is split into a level, a daily path and a within-day shape;
  - how much history the weights need, and how far they can move from equal;
  - what the energy and peak readouts can and cannot guarantee;
  - why the evaluation reports several estimands side by side.
- **C1–C14, the carrier.** New in ankyra-lite. C1–C10: what the construction of the carrier network guarantees for any
  weights, trained or not, under the arithmetic and input assumptions of each claim:
  - its forecast is a nonnegative energy budget whose mean is its level;
  - it is built only from data before its issue time, without a unit identifier, and scales with the load;
  - its copy pointer starts at zero and never moves the level;
  - with a 186-day gap between the training cut and the test set, every pseudo-origin forecast that the anchoring
    layer scores is out of sample for the carrier.

  C11–C14 (new in 0.2.0): what the inverse-noise loss weighting of recipe Rw's second stage guarantees for a
  training pool that meets their stated support and positive-floor conditions (an arbitrary pool need not):
  - with equal weights it is recipe R bit for bit;
  - every population keeps its share of the draws;
  - every weight is finite, positive and bounded by the floor;
  - a weight is read only from its window's own last 28 context days.
- **O1–O9, the continuation layer.** New in 0.2.0: what the origin-continuity layer after the anchoring layer
  guarantees for any admitted anchored trajectory under each claim's assumptions (floating-point finiteness and strict
  positivity need separate representability conditions; arbitrary custom constants are not covered):
  - it multiplies every hour by a positive factor with stated bounds, so signs and exact zeros are kept;
  - it reads only the 672 hours before the origin and eight hours of the calendar;
- it leaves state windows untouched and switches off without changing the carrier-specific pipeline; reproducing the
  version 0.1 product also requires its recipe-R checkpoints of seeds 0–2 (`checkpoints/<population>/seed0-2.pt`), the same inputs and the original process;
- the energy readout (`energy_kwh`), level and peak are read before it. The delivered curve sum generally differs from the energy readout (`energy_kwh`) by the
  pre-layer curve/readout difference and the continuation change, though the difference can be zero.

Each property is supported in three ways:

| File | What it gives |
|---|---|
| [PROOFS.md](PROOFS.md) | statement, proof and the counterexamples that mark where it stops; where this repository's result files measure a consequence on data, the file |
| [operators.py](operators.py) | P1–P20 as functions that apply to any forecaster's 744-hour trajectory |
| [test_operators.py](test_operators.py) | a numerical check of P1–P20, counterexamples included (31 checks) |
| [test_carrier_properties.py](test_carrier_properties.py) | a numerical check of C1–C10 on random and perturbed carrier weights, counterexamples included (27 checks) |
| [test_weighting_properties.py](test_weighting_properties.py) | a numerical check of C11–C14 on synthetic training pools and untrained carriers, counterexamples included (15 checks) |
| [test_continuation_properties.py](test_continuation_properties.py) | a numerical check of O1–O9 on synthetic windows and, end to end, on an artificial building, counterexamples included (36 checks) |

```bash
python -m unittest discover -s theory -t .
```

The command runs the four check files (111 checks; CPU only, about 35 seconds on one thread); no checkpoint and no
study data are needed. Run it from the repository root.

![Exact properties on a test window](../figures/fig7_operators.png)

*Figure 7. Panel a replaces the level, the daily path and the within-day block of the carrier's forecast by
ankyra-lite's, one block at a time, on the Cambridge test window of Figure 6: each bar splits exactly into the three
block losses (P1, P4), and with all three blocks replaced the forecast is ankyra-lite's. Panels b–d show how much
history the weights need and how far shrinkage lets them move (P7–P9) and why energy is read before the clip
(P11, P12); panels e and f show the peak readout and its error decomposition (P15, P16). Drawn by
[`figures/make_figures.py`](../figures/README.md).*

## The twenty properties of the anchoring layer

Properties marked *standard* are known results that the design relies on; they are collected here, not claimed as new.
The others hold under the conditions in the third column. None is an accuracy guarantee.

| | Property | Holds when | What the anchoring layer uses it for | Code | Check |
|---|---|---|---|---|---|
| P1 | [Block identity](PROOFS.md#blocks) (standard) | always | assigning each block to a source | `anchor.blocks.block_losses` | `BlockGeometryTests` |
| P2 | [Four blocks; no weekly Fourier bin](PROOFS.md#blocks) | always (744-hour window) | day types instead of a weekly component | `operators.four_block_losses` | `BlockGeometryTests` |
| P3 | [Energy is the level](PROOFS.md#blocks) | always | the energy readout | `anchor.blocks.level` | `BlockGeometryTests` |
| P4 | [Exact replacement and the level-share bound](PROOFS.md#replacement) | fixed replacement estimates | scoring any division from block losses | `operators.replace_blocks` | `ReplacementTests` |
| P5 | [Two-source division: pooled optimum, unit-level asymmetry](PROOFS.md#replacement) | two sources, block-wise | why a fixed division is not enough | `operators.division_log_ratio` | `ReplacementTests` |
| P6 | [Correction accounting](PROOFS.md#replacement) (standard) | always | diagnosing a correction | `operators.correction_accounting` | `ReplacementTests` |
| P7 | [Completed error support](PROOFS.md#historical-weights) | complete records; missing hours only lower the count | how much history the weights need | `operators.completed_windows` | `SupportTests` |
| P8 | [Shrinkage bounds](PROOFS.md#historical-weights) | matched support, no deletion at the origin | how far the weights can move | `operators.mixture_mass_bounds` | `ShrinkageTests` |
| P9 | [Deletion renormalises the mixture](PROOFS.md#historical-weights) | always; a counterexample shows where P8 stops | when the bounds stop holding | `operators.retained_data_share` | `ShrinkageTests` |
| P10 | [Least squares versus inverse MSE under shared errors](PROOFS.md#historical-weights) | forecasts share an error component | the weekly handover weights | `operators.two_source_weights` | `ShrinkageTests` |
| P11 | [Clipping cannot increase any hour's error](PROOFS.md#nonnegativity-and-energy) (standard) | nonnegative load | the delivered trajectory | `operators.clip_nonnegative` | `FeasibilityTests` |
| P12 | [Energy, nonnegativity and no-harm are incompatible](PROOFS.md#nonnegativity-and-energy) | always (two-hour counterexample) | reading energy from the level | `operators.project_to_mean` | `FeasibilityTests` |
| P13 | [Day-level projection: a conditional guarantee](PROOFS.md#nonnegativity-and-energy) | nonnegative level and load; level under-forecast by at most the smallest true daily mean | an optional projection | `operators.daily_projection` | `FeasibilityTests` |
| P14 | [A mean path underestimates the expected peak](PROOFS.md#peak) (standard) | exact conditional means | a separate peak readout | — | `PeakTests` |
| P15 | [Scalar form and bounds of the peak operator](PROOFS.md#peak) | always | the peak readout | `anchor.readouts.peak_readout` | `ReadoutTests` in [tests/anchor/](../tests/anchor/test_blocks_and_readouts.py) |
| P16 | [Peak error decomposition](PROOFS.md#peak) | always; the amplitude term has no fixed sign | how level gains reach the peak | `operators.peak_error_decomposition` | `PeakTests` |
| P17 | [Finite-support median property](PROOFS.md#peak) | constant levels within type, independent draws, $((m-1)/m)^n<1/2$ | what the envelope estimates | `operators.max_lower_median` | `PeakTests` |
| P18 | [Asymmetric peak costs](PROOFS.md#peak) (standard) | always | choosing a peak method under a cost | `operators.asymmetric_peak_cost` | `PeakTests` |
| P19 | [Pooled versus unit-equal summaries](PROOFS.md#estimands) | always (identity) | why several estimands are reported | `anchor.metrics.pooled_ratio_decomposition` | `EstimandTests` |
| P20 | [Evidence bound for choosing among candidates](PROOFS.md#evidence-bound) (standard; null benchmark is an identity) | T pseudo-origin errors per candidate | statistical background for shrinking the level weights strongly (a worst-case rate, not a measurement on these data) | `operators.aggregation_rate`, `operators.exchangeable_hit_rate` | `EvidenceBoundTests` |

`operators.*` is [operators.py](operators.py) in this folder; `anchor.*` is `ankyra_lite.anchor`, the anchoring layer,
whose own functions carry P1, P3, P15 and P19. Check names are test classes in [test_operators.py](test_operators.py)
unless a folder is given.

## The fourteen properties of the carrier

All fourteen follow from the code of `ankyra_lite.carrier`. C1–C10 hold for any weights, trained or not, under the arithmetic and input
assumptions of each claim, except C9(a), which is about the initialisation; the proofs cite the code lines. Real-arithmetic statements are checked in float64,
statements about the delivered carrier forecasts on the package's own float32 path. C11–C14 (new in 0.2.0) hold
under their stated support and positive-floor conditions, which an arbitrary pool need not satisfy; they concern the loss weighting of recipe Rw's second stage (`train.noise_weights`,
`train.weighted_log1p_loss`), and the proofs cite the functions.

| | Property | Holds when | What it gives ankyra-lite | Code | Check |
|---|---|---|---|---|---|
| C1 | [Nonnegative trajectory](PROOFS.md#exact-properties-of-the-carrier-network) | any weights; strictly positive in real arithmetic only (shares can underflow to 0) | a nonnegative carrier candidate; a non-finite forecast is rejected before anchoring | `model.decode_budget`, `inference.carrier_forecasts` | `NonnegativityTests` |
| C2 | [Energy budget: the trajectory's mean is the level](PROOFS.md#exact-properties-of-the-carrier-network) | always (exact in real arithmetic) | the carrier's level, daily path and within-day shape are the P1 blocks of its trajectory | `model.decode_budget` | `EnergyBudgetTests` |
| C3 | [Day shares, hour shares and pointer attention are distributions](PROOFS.md#exact-properties-of-the-carrier-network) | always | the copied profile is a mixture of the unit's own complete days | `model.CopyPointerNetwork` | `ShareTests` |
| C4 | [No input at or after the issue time](PROOFS.md#exact-properties-of-the-carrier-network) | always; the day types of the forecast month are read | every pseudo-origin forecast is a genuine forecast from its own issue time | `features.features`, `inference.carrier_forecasts` | `CausalityTests` |
| C5 | [The encoder is causal in token order](PROOFS.md#exact-properties-of-the-carrier-network) | always | the pointer's key for a recent day summarises that day and earlier days only | `model.SharedCausalBlock` | `EncoderOrderTests` |
| C6 | [Masked tokens are ignored](PROOFS.md#exact-properties-of-the-carrier-network) | always; the masks themselves are inputs | placeholder values of missing days cannot reach the forecast | `model.StructuredLoadNetwork` | `MaskTests` |
| C7 | [Scale equivariance](PROOFS.md#exact-properties-of-the-carrier-network) | $c>0$ and positive load in the last 8,760 h; bit for bit for powers of two; a counterexample otherwise | the carrier forecast does not depend on the unit of the load | `features.features`, `model.StructuredLoadNetwork` | `ScaleTests` |
| C8 | [No unit identifier](PROOFS.md#exact-properties-of-the-carrier-network) | always; the category is an input | the same history in the same category gives the same forecast | `features.features`, `model.build_carrier` | `UnitIdentityTests` |
| C9 | [The copy pointer: zero at initialisation, level unchanged, inactive without a complete recent day](PROOFS.md#exact-properties-of-the-carrier-network) | (a) at initialisation; (b)–(d) any weights | the pointer reshapes the month without moving its energy | `model.CopyPointerNetwork` | `PointerTests` |
| C10 | [Pseudo-origin out-of-sample guarantee](PROOFS.md#exact-properties-of-the-carrier-network) | test origins at least 4,464 h (186 days) after the training cut | every completed pseudo-forecast the anchoring layer scores is out of sample for the carrier | `inference.pseudo_origins`, `train.make_windows` | `PseudoOriginTests` |
| C11 | [Unit weights are recipe R](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | bit for bit on the CPU; any constant weight in real arithmetic | recipe Rw reduces to recipe R, and the 0.1 recipes and their records are unchanged | `train.weighted_log1p_loss`, `train.train_step`, `train.recipe` | `UnitWeightTests` |
| C12 | [Each population keeps its share of the draws](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | in expectation over the sampler's draws (exact) | the weighting moves weight only within a population, never between the target and the borrowed windows | `train.draw_probabilities`, `train.noise_weights` | `SourceBalanceTests` |
| C13 | [Bounded weights and the floor](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | a positive floor (a counterexample shows a zero floor) | every pool window has a finite positive weight, at most the floor's, and a quarter of the draw mass sits at the floor | `train.noise_weights`, `train.weighted_quantile` | `WeightBoundTests` |
| C14 | [The weights read only each window's own recent context](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | always; float16 tokens round the estimate | no label, no evaluation window and no random stream enters a weight; the weights do not depend on the unit of the load | `train.noise_sigma2`, `train.pool_sigma2` | `WeightInputTests` |

`model.*`, `features.*`, `inference.*` and `train.*` are the modules of `ankyra_lite.carrier`; check names are test
classes in [test_carrier_properties.py](test_carrier_properties.py) (C1–C10) and
[test_weighting_properties.py](test_weighting_properties.py) (C11–C14).

## The nine properties of the continuation layer

All nine follow from the code of `ankyra_lite.continuation` (new in 0.2.0) and of `ankyra_lite.forecast`, and hold for
any trajectory the anchoring layer delivers. The analytic statements allow any constants κ₁, κ₂ ≥ 0, ρ₁, ρ₂ > 0 unless a
number is given; in floating point they also need representability conditions: with extreme custom constants a
non-finite correction or output on an ordinary window raises `ValueError` (since 0.2.1), and a nonzero hour can underflow
to zero.
The layer multiplies the anchored trajectory F by e^{Δ(h)}, Δ(h) = κ₁x_f e^{−h/12} + κ₂x_d e^{−h/168}, where x_f and
x_d are the clipped log ratios of the last 3 and 24 observed hours to the forecast's own reference day; the bound
E(h) = κ₁e^{−h/12} + κ₂e^{−h/168} is 0.809 at h = 0, at most 0.1215 from day 8 and 0.004 at the last hour.

| | Property | Holds when | What it gives ankyra-lite | Code | Check |
|---|---|---|---|---|---|
| O1 | [Nonnegativity, sign and zeros](PROOFS.md#exact-properties-of-the-continuation-layer) | in real arithmetic; in floating point a subnormal hour can vanish (counterexample) and extreme custom constants raise `ValueError` instead of a non-finite hour; a last-hours mean at or below −ε (negative loads) sets that anomaly to 0 | the delivered hourly forecast stays nonnegative, and exact zeros stay zero | `continuation.apply_batch` | `NonnegativityTests` |
| O2 | [Bounded correction](PROOFS.md#exact-properties-of-the-continuation-layer) | always (bitwise against the envelope) | each hour moves by a factor in [e^{−E(h)}, e^{E(h)}]: [0.445, 2.246] at the origin, [0.885, 1.130] from day 8 | `continuation.delta`, `continuation.envelope` | `BoundTests` |
| O3 | [Causality](PROOFS.md#exact-properties-of-the-continuation-layer) | always; the calendar of the first forecast week is read | the layer reads only the 672 hours before the origin and eight hours of the calendar | `continuation.window_inputs` | `CausalityTests` |
| O4 | [Bypass](PROOFS.md#exact-properties-of-the-continuation-layer) | state windows; unobserved last hours (not reachable through `forecast`); last-hours means at or below −ε (negative loads, a release rule) | the state rule's forecast is delivered unchanged; missing last hours switch the layer off; a nonpositive last-hours mean sets its anomaly to 0 (flagged), and both nonpositive switch the layer off | `continuation.apply_batch`, `forecast.forecast` | `BypassTests`, `ForecastBypassTests` |
| O5 | [Degeneracy](PROOFS.md#exact-properties-of-the-continuation-layer) | constants zero, or the layer off | with the layer off the carrier-specific version 0.1 pipeline is unchanged; reproducing the 0.1 product also requires its recipe-R checkpoints of seeds 0–2 (`checkpoints/<population>/seed0-2.pt`), inputs and original process | `continuation.ZERO`, `forecast.forecast` | `DegeneracyTests` |
| O6 | [Readouts are read before the layer](PROOFS.md#exact-properties-of-the-continuation-layer) | always (readout rule) | level/the energy readout (`energy_kwh`)/peak are unchanged by toggling the layer on a fixed carrier; delivered curve sum generally differs from the energy readout (`energy_kwh`) by the pre-layer curve/readout gap and layer change, which can be zero | `forecast.forecast`, `readouts.continuation_energy_change` | `ReadoutTests` |
| O7 | [Consistency and monotonicity](PROOFS.md#exact-properties-of-the-continuation-layer) | (a) exact float equality; (b) same-sign anomalies (opposite signs: one sign change), last-hours means above −ε | a forecast that already continues the last hours is left alone; the correction follows the direction of the anomalies | `continuation.anomalies`, `continuation.delta` | `ConsistencyTests` |
| O8 | [Scale equivariance](PROOFS.md#exact-properties-of-the-continuation-layer) | c > 0 and the scale above its 10⁻³ kW floor; bit for bit for powers of two; a counterexample otherwise | the layer does not depend on the unit of the load | `continuation.anomalies` | `ScaleTests` |
| O9 | [Reference day and identity](PROOFS.md#exact-properties-of-the-continuation-layer) | always; the fallback d* = 6 differs from the development code | the anomalies compare the same clock hours and the same workday class; the constants and rule are identified by one digest | `continuation.reference_day`, `continuation.digest` | `ReferenceDayTests` |

`continuation.*` is `ankyra_lite.continuation`, `forecast.*` and `readouts.*` the modules of `ankyra_lite`; check
names are test classes in [test_continuation_properties.py](test_continuation_properties.py).

## How the properties enter the forecaster

- **Blocks (P1–P3).** The forecast is assembled from three orthogonal blocks, so each can be weighted, scored and
  read out separately, and energy is read from the level alone.
- **Division of labour (P4–P6).** Any division between two forecasters can be scored from block losses, but a division
  that is optimal on pooled error can still lose for the typical unit. The anchoring layer therefore weights each unit
  by its own record.
- **Weights (P7–P10).** A record of $m$ hours supports $\min\lbrace12,\lfloor(m-1344)/744\rfloor\rbrace_+$ completed
  errors; below two the weights are equal. Shrinkage bounds how far they move, and the handover uses least-squares
  weights because shared errors pull inverse-MSE weights towards one half.
- **Energy and nonnegativity (P11–P13).** Clipping cannot hurt any hour, but no map keeps energy, nonnegativity and
  no-harm together, so energy is read before the clip. (On off-state and micro-load windows the anchoring layer
  returns the carrier's trajectory unchanged, without the clip; ankyra-lite's default state rule then delivers the
  weekly seasonal naive instead.)
- **Peak (P14–P18).** The maximum of a smooth forecast underestimates the peak, so the peak is read from a historical
  excursion envelope with an exact scalar form, an error decomposition and a median property under a stated working
  distribution.
- **Estimands (P19).** Pooled and unit-equal summaries can disagree in sign for an algebraic reason, so the result
  files report the unit-equal log ratio, the pooled MSE ratio, the mean per-unit rank and conventional metrics
  together.
- **Evidence bound (P20).** The minimax rate of aggregation is a worst-case result: for seven level candidates and at
  most six pseudo-origin errors its order is a sizeable fraction of the error variance. It is the statistical
  background for shrinking the weights strongly; it is not a measurement on these data and does not by itself show
  that the candidates cannot be told apart here ([PROOFS.md](PROOFS.md#evidence-bound)). On the test windows the
  candidate with the largest weight is the best one at the origin more often than chance, but in only about one
  window in five (`results/level_candidate_hit_rate.csv`). The record statistics that
  the anchoring layer reports (`anchoring_record`) are descriptive indicators; the forecaster does not claim to select
  the best candidate.
- **The carrier (C1–C10).** The carrier hands the anchoring layer a nonnegative trajectory whose mean is its level
  (C1–C3), built only from data before each issue time and ignoring placeholder values (C4–C6), in the load's own
  scale and without a unit identifier (C7, C8). Its copy pointer reshapes the month without moving the level and
  starts as the base network (C9). With the 186-day gap between the training cut and the test set, every pseudo-origin
  forecast that the anchoring layer scores against the unit's own history is out of sample for the carrier (C10),
  which is what the layer's per-unit weights assume.
- **The loss weighting of recipe Rw (C11–C14).** In its second stage recipe Rw weights each training window by the
  inverse square root of its weekly-naive noise over the last three weeks, floored at the pool's 25th percentile. With
  equal weights it is recipe R bit for bit (C11); the weights move emphasis only within a population, never between
  the target and the borrowed windows (C12); they are finite, positive and bounded (C13); and they are read from each
  window's own context before its origin, without labels or random numbers (C14).
- **The continuation layer (O1–O9).** After the anchoring layer and the state rule, the layer multiplies each hour of
  the trajectory by e^{Δ(h)}, with Δ set by how far the last 3 and 24 observed hours lie from the forecast's own
  reference day. The factor is positive and bounded, shrinking from [0.445, 2.246] at the first hour to [0.996, 1.004]
  at the last (O1, O2); it reads only the 672 hours before the origin and the calendar of the first week (O3); state
  windows bypass it, and switched off the layer leaves the carrier-specific pipeline unchanged; reproducing version
  0.1 also requires its recipe-R checkpoints of seeds 0–2 (`checkpoints/<population>/seed0-2.pt`), inputs and original process (O4, O5). The energy readout (`energy_kwh`), level and peak are read
  before it, so the delivered curve sum generally differs from the energy readout (`energy_kwh`) by the pre-layer curve/readout gap and the
  continuation change, though the difference can be zero (O6); a
  forecast that already continues the last hours is left alone, and same-sign anomalies move the whole window one way
  (O7); it does not depend on the unit of the load (O8); and it compares like with like, the same clock hours and the
  same workday class (O9).

## Consequences measured in this repository

The properties hold for any weights; where this repository's result files measure one of their consequences on the
test windows, the file is listed here and named in the statement in [PROOFS.md](PROOFS.md). All of these readings are
descriptive; the numbers and their reading are in [docs/EVALUATION.md](../docs/EVALUATION.md).

| | What is measured | File |
|---|---|---|
| P1 | the share of each block in the hourly squared error of ankyra-lite and the baselines | [`results/block_shares.csv`](../results/block_shares.csv); Figure [14](../figures/fig14_block_attribution.png) |
| P4 | the replacement of the carrier's blocks by the anchored ones: ankyra-lite against its carrier alone, block by block; the exact replacement on one window | [`results/contrast_ii_blocks.csv`](../results/contrast_ii_blocks.csv) (`anchored vs carrier alone`); Figure [7a](../figures/fig7_operators.png) |
| P5, P10 | the handover against a fixed division and fixed weights (no weight, ½, one least-squares weight per window, the weekly weights) and the estimated weights by forecast week | [`results/handover_granularity.csv`](../results/handover_granularity.csv), [`results/lead_weeks.csv`](../results/lead_weeks.csv) |
| P7 | ankyra-lite by the length of the record before the origin and by K_eff | [`results/history_length.csv`](../results/history_length.csv) |
| P11, P12 | the windows where the projection binds, and the energy contrasts with the level readout | [`results/energy_level_readout.csv`](../results/energy_level_readout.csv) |
| P15 | the peak readout against trajectory maxima, the previous window's peak and the same readout on the baselines | [`results/peak_readout.csv`](../results/peak_readout.csv) |
| P16 | the same readout on the carrier alone: whether the anchored daily means carry into the peak | [`results/peak_readout.csv`](../results/peak_readout.csv) (`same readout on the carrier alone`) |
| P19 | the unit-equal log ratio, the pooled MSE ratio, the mean per-unit rank and conventional metrics side by side | `results/benchmark_pairwise.csv`, `benchmark_mean_unit_rank.csv`, `conventional_metrics_late.csv` |
| P20 | the hit rate of the level candidate with the largest weight against the chance rate; the anchoring record against the realised gain | [`results/level_candidate_hit_rate.csv`](../results/level_candidate_hit_rate.csv), [`results/anchoring_gain_deciles.csv`](../results/anchoring_gain_deciles.csv) |
| C12, C13 | per population: the floor, the normalisers, each population's draw share, the Kish effective-sample-size ratio and the largest weight over the median of the loss weights | `weights_record` of each Rw checkpoint in [`checkpoints/MANIFEST.json`](../checkpoints/MANIFEST.json) |
| O2, O7 | the layer on against the layer off, on the same carrier, at hour 0, hours 0–2 (segment `lead hours 0-3`), day 1, days 2–7, days 8–31 and the whole window | [`results/continuation_layer.csv`](../results/continuation_layer.csv) |
| O4, O6, O9 | state windows bypassed, the shares of clipped anomalies, the windows that used the reference-day fallback, the largest \|Δ\|, and the relative difference between the delivered curve's mean and the energy readout | [`results/continuation_record.csv`](../results/continuation_record.csv) |

P1–P20 and their files are those of the ANKYRA 2.2.1 theory folder, with the imports changed from `ankyra` to
`ankyra_lite.anchor`. The data evidence of that theory (from the method's earlier development and from the ANKYRA
study, with TimesFM 2.5 as the carrier) is not a result of this repository and is left out; where this repository's
result files measure a consequence, PROOFS.md names the file. C1–C10 and `test_carrier_properties.py` are new in
ankyra-lite; C11–C14 with `test_weighting_properties.py` and O1–O9 with `test_continuation_properties.py` are new in
0.2.0.
