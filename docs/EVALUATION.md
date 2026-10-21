# Evaluation

The scored statistics are in [`results/`](../results) ([file list and columns](../results/README.md)); the figures are
drawn from those files by [`figures/make_figures.py`](../figures/README.md). Data sources and the split are in
[DATA.md](DATA.md), the forecaster in [METHOD.md](METHOD.md), and the scoring code in
[`evaluation/`](../evaluation/README.md).

The comparison mirrors that of the first ANKYRA release (the ANKYRA 2.2.1 repository): the same test windows, the same
estimands and intervals, the same scoring functions and the same baselines, with ankyra-lite as the reference
forecaster. ANKYRA itself and its versions are not among the forecasters. The study's working protocols, scripts and
saved forecasts are not part of this repository.

**Contents**

- [Identity of the evidence](#identity-of-the-evidence) (read first)
- [Populations and tiers](#populations-and-tiers)
- [Baselines](#baselines): [LSTM-T-KAN-M](#lstm-t-kan-m) ·
  [the linear-context model](#the-linear-context-model-tide-implementation) · [GBT-T](#the-corrected-gbt-t) ·
  [training budget](#training-budget-of-the-trained-baselines) · [what each side may know](#what-each-side-may-know)
- [Estimands](#estimands)
- [How each table is scored](#how-each-table-is-scored)
- [State windows and the three near-zero BDG2 meters](#state-windows-and-the-three-near-zero-bdg2-meters)
- [Intervals](#intervals)
- [Why the recipe returns the validation span](#why-the-recipe-returns-the-validation-span)
- [Version 0.2.0: the development check and the test read](#version-020-the-development-check-and-the-test-read)
- [Results](#results): [overview](#overview) · [zero-shot foundation models](#against-the-zero-shot-foundation-models) ·
  [rank tests](#rank-tests) · [conventional metrics](#conventional-metrics) ·
  [by forecast day](#by-forecast-day-and-energy-through-the-month) · [the continuation layer](#the-continuation-layer) ·
  [blocks](#error-blocks-and-scaled-errors) ·
  [intervals](#intervals-around-ankyra-lite) · [parts](#what-the-parts-add) ·
  [handover](#what-the-handover-contributes) · [whole-window weight](#block-wise-weights-against-one-whole-window-weight) ·
  [within-day trust](#within-day-anchoring-where-it-acts) · [generic partners](#generic-combination-partners) ·
  [seeds](#seeds) ·
  [other carriers](#other-carriers-in-the-same-anchoring-layer) ·
  [contrast (ii)](#contrast-ii-by-block-and-the-carrier-free-control) ·
  [history length](#history-length-and-the-anchoring-record) · [robustness and cost](#robustness-and-cost) ·
  [where ankyra-lite is weak](#where-ankyra-lite-is-weak)
- [External tests](#external-tests)
- [Limitations](#limitations)
- [Scoring-code check and reproduction](#scoring-code-check-and-reproduction)
- [References](#references)

## Identity of the evidence

**Every number of the nine-population comparison is a re-evaluation of test windows that had been read before. None is
a first test.** Version 0.2.0 (recipe Rw, six seeds, the continuation layer) was frozen on development windows and then
read once on these test windows, which had been read many times before; that read sets the direction of each change
([below](#version-020-the-development-check-and-the-test-read)). The external tests ([EXTERNAL_TESTS.md](EXTERNAL_TESTS.md))
have their own identity, stated with each number there: the 0.2.0 re-run reads, a second time, test sets that the frozen
external test of 0.1 read once (LCL, UNICON, HKUST, and a same-source out-of-time supplementary check on GoiEner
households).

- **The test windows were read before.** They are the late windows of the first release, scored there for ANKYRA and
  the 20 external baselines. The study then scored earlier versions of the small carrier on the same windows, made an
  hourly-only comparison of the carrier with the baselines, and read them once more in the study that chose the recipe
  (below).
- **Parts of the design were chosen after looking at them.**
  - The **copy pointer**: its direction (copying the unit's own recent days into the within-day shape and the daily
    means) came from an analysis of the carrier's errors on these windows.
  - The **seasonal-naive state rule** was written after an earlier scoring of these windows.
  - The **released recipe (R)**, which returns the validation span to the training set, was chosen on 19 October 2026
    after one directional reading of the span-return study on these test windows
    ([below](#why-the-recipe-returns-the-validation-span)). The study's plan and its predictions were written before
    that reading; the choice of recipe was made after it.
  - The **continuation layer**: its form was chosen on development windows, partly after reading them, and its four
    constants were fitted on the earlier development windows of six populations
    ([METHOD.md](METHOD.md#the-continuation-layer)); recipe Rw's loss weighting was taken from an earlier study of this
    carrier. The 0.2.0 configuration was frozen before the test read.
- **Development windows** existed on six populations (Cambridge, HEEW, EWELD, Oslo, Drammen, CINELDI): windows between
  a development cut 279 days before the training cut and the training cut, on which the carrier's configuration was
  chosen. The three short-record populations (BDG2 2017 and both GoiEner populations) had none, so no design choice was
  checked on them before their test windows were read.
- **Consequence.** The tables describe how the released forecaster performs on these windows. They do not estimate how
  it would perform on windows that played no part in its design. The external tests of this version on data outside
  these test windows ([External tests](#external-tests)) do not either: their populations had been read before by
  earlier forecasters, or share a source with a study population.

Further caveats that apply to every table:

- **Calendar.** 26 EWELD and 10 Cambridge test windows use a holiday calendar announced after their origin, for every
  forecaster that reads day types ([DATA.md](DATA.md#calendar-caveat-holidays-announced-after-the-origin)).
- **Pretraining overlap.** Seven BDG2 sites are in TimesFM 2.5's pretraining collection
  ([DATA.md](DATA.md#bdg2-and-the-pretraining-collection-of-timesfm-25)).
- **Linear-context model.** The baseline formerly labelled TiDE is a linear map of the load context
  ([below](#the-linear-context-model-tide-implementation)).
- **The Suzhou park** of the first release has no ankyra-lite forecast and is in no table; it contributes borrowed
  training windows only.

## Populations and tiers

| Population | Country | Test windows / units | Tier | Development windows | Source |
|---|---|---|---|---|---|
| BDG2 2017, building meters | USA / Europe | 474 / 142 | test | no | Miller et al., *Sci. Data* 2020, [doi:10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x) |
| University of Cambridge estate | UK | 730 / 108 | test | yes | Langtry & Choudhary 2024, [doi:10.5281/zenodo.10955332](https://doi.org/10.5281/zenodo.10955332) |
| HEEW, Arizona State University | USA | 658 / 138 | test | yes | Dong et al., *Sci. Data* 2025, [doi:10.1038/s41597-025-06010-8](https://doi.org/10.1038/s41597-025-06010-8) |
| EWELD industrial and commercial meters | China | 523 / 197 | test | yes | Liu et al., *Sci. Data* 2023, [doi:10.1038/s41597-023-02503-6](https://doi.org/10.1038/s41597-023-02503-6) |
| GoiEner non-household supply points | Spain | 890 / 481 | test | no | Quesada et al., *Sci. Data* 2024, [doi:10.1038/s41597-023-02846-0](https://doi.org/10.1038/s41597-023-02846-0) |
| GoiEner households | Spain | 696 / 696 | test | no | as above |
| COFACTOR-SBHUB Oslo schools | Norway | 581 / 41 | preview | yes | Lien et al., *Data in Brief* 2025, [doi:10.1016/j.dib.2025.112288](https://doi.org/10.1016/j.dib.2025.112288) |
| COFACTOR Drammen municipal buildings | Norway | 689 / 43 | preview | yes | Lien et al., *Sci. Data* 2025, [doi:10.1038/s41597-025-04708-3](https://doi.org/10.1038/s41597-025-04708-3) |
| CINELDI industrial customers | Norway | 475 / 45 | preview | yes | Sandell et al., *Data in Brief* 2023, [doi:10.1016/j.dib.2023.109121](https://doi.org/10.1016/j.dib.2023.109121) |

**Tiers.** The tier names are the first release's: `test` for its six test populations and `preview` for Oslo, Drammen
and CINELDI (its fourth preview population, the Suzhou park, is not here). In ankyra-lite every population has its own
training cut and test set and is scored in the same way; the tier says only how the tables group them. As in the first
release, the error by forecast day, the energy through the month, the error blocks and the pooled rank tests cover the
six test populations; every other table covers all nine.

Test windows are the late windows of the first release: month-start origins at or after the training cutoff of its
trained baselines, with a complete 1,344-hour context and 744-hour target and six completed pseudo-origins
([DATA.md](DATA.md#two-kinds-of-windows)). CINELDI is scored on the corrected store, with true dates.

## Baselines

ankyra-lite is compared with **21 baselines** (22 forecasters in all): the 20 external baselines of the first release
and LSTM-T-KAN-M, the multivariate LSTM-T-KAN model built from its authors' public code and their exp2 feature list
([below](#lstm-t-kan-m)). All are ordinary baselines here, in every table and every rank.

| Information | Forecasters (model id in the result files) |
|---|---|
| Given the first release's shared covariate set (11 past, 10 future, 6 static features; no observed future weather); each uses what its architecture accepts | the linear-context model (`TiDE`; trained), iTransformer-X (trained; no future covariates), GBT-T (trained gradient boosting on 21 derived features), Chronos-2-X (zero-shot; no static features), TimesFM-X (zero-shot; covariates through linear regression) |
| Load and calendar, trained | LSTM-T-KAN-M (16 channels derived from the past load and the past calendar; no temperature) |
| Load only, zero-shot foundation models | TimesFM 2.5 (`TimesFM`), Chronos-2 (`Chronos-2`) |
| Load only, trained per population | DLinear, PatchTST, iTransformer, LSTM |
| Load only, statistical | Holt–Winters (damped, additive, period 168 h), MSTL (24 h and 168 h) |
| Calendar and climatological temperature | per-unit ridge regression refitted at every origin (`RIDGE-L`) |
| Simple | previous-month (`SP-profile`), last-year (`LYR-profile`) and 4-week (`WeekProf4`) profiles; seasonal naive with a period of one day (`SN-day`) and one week (`SN-week`); a zero-shot gradient-boosting model (`GBT`) |

The shared covariate set is the first release's: load; temperature (its climatology in the horizon); the load one year
earlier (lag 8,736 h, always before the origin) with its availability mask; sines and cosines of hour, weekday and day
of year; a non-working-day flag; the category (one-hot); the log ratio of the long-history to the recent mean. It was
fixed in writing by the first release before its baselines were run, and nothing in it is observed after the origin.

**Hourly features** (past = the 1,344-hour context, future = the 744-hour horizon):

| # | Feature | Past | Future | Construction |
|---|---|:---:|:---:|---|
| 1 | load | observed | — | per-unit z-score; statistics from the context or from before the training cutoff |
| 2 | temperature | observed | climatology | horizon values are the frozen annual harmonic fitted on pre-origin temperature; observed future temperature is never used |
| 3 | load one year earlier | ✓ | ✓ | t − 8,736 h (52 weeks, weekday-aligned; always before o), same normalisation as 1, 0 where missing |
| 4 | availability of 3 | ✓ | ✓ | 1 where feature 3 is observed |
| 5–6 | hour of day | sin, cos | sin, cos | period 24 |
| 7–8 | weekday | sin, cos | sin, cos | period 7 |
| 9–10 | day of year | sin, cos | sin, cos | period 365.25 |
| 11 | holiday / non-working day | ✓ | ✓ | day type 7 (public holiday) or Saturday/Sunday |

Dimensions: 11 past, 10 future (features 2–11). **Static features** (one set per unit): the category (Industrial,
Office, Public, Residential, Commercial; one-hot) and the log ratio of the long-history mean (at most 8,760 h) to the
context mean; six dimensions.

**How each forecaster receives the set**

| Forecaster | Receives | Not accepted by the architecture |
|---|---|---|
| ankyra-lite | the raw record behind the set (load, temperature, day types, category), not the engineered features. The carrier builds its own daily, monthly and future-day tokens from it ([METHOD.md](METHOD.md#inputs)); the anchoring layer reads the raw pre-origin record and the carrier's forecasts at o and o − 744k. Feature 3 is not used as constructed: the layer's annual candidates read the window 8,760 h (365 days) before the origin, and the carrier sees the days of a year earlier only through its 366 daily tokens. The carrier reads no temperature of the forecast month; the layer forms the climatology of the horizon from the pre-origin temperature itself | — |
| Linear-context (TiDE implementation) | 11 past, 10 future, 6 static (given; not used) | — |
| iTransformer-X | 11 past as variable channels, 6 static as constant channels | future covariates |
| GBT-T | 21 features per forecast hour: three load aggregates, features 3–4, two temperature values, seven calendar values, the lead time, six static | — |
| Chronos-2-X | features 2–11 as past and known-future covariates | static features |
| TimesFM-X | features 2–11 through its linear covariate regression, the category as a categorical covariate | — |
| PatchTST | 11 past + 6 static as channels, but channel-independent: its load forecast equals the load-only one | information between channels |
| LSTM-T-KAN-M | not the shared set: 16 channels derived from the past load and the past calendar ([below](#lstm-t-kan-m)) | — |
| load-only models (TimesFM, Chronos-2, DLinear, iTransformer, LSTM, Holt–Winters, MSTL, naive and profile forecasters) | feature 1 only | — |
| per-unit ridge | calendar and climatological temperature, refitted at every origin | — |

**Deliberately excluded** (the first release's rule): observed future weather; humidity and other meteorological
variables (available for one population only); hand-made signal transforms (smoothing, wavelet, Fourier or EMD
decompositions), which add no information and are a common source of look-ahead; load-derived statistics (used only
inside GBT-T, which cannot read a sequence; LSTM-T-KAN-M's own channels follow its authors' list). The baselines' code,
with its assertions on every feature's time index, is not in this repository.

**Training of the first release's trained baselines** (unchanged; their saved forecasts are scored): one model per
population across its units; all training targets end before the cutoff, which is the first month start at or after
the population's median evaluation origin (the start of the test set here); learning rate chosen from
{1e-4, 3e-4, 1e-3} on the last 20% of training origins; at most 15 epochs with patience 3; three seeds averaged; scored
on the late windows only. Leakage checks (feature time indices, lag indices, normalisation statistics, training-target
end) are assertions in the study's code.

### LSTM-T-KAN-M

LSTM-T-KAN is the LSTM model with a temporal Kolmogorov–Arnold network head of Shen, Yang, Song and Liang (2026,
doi:10.1016/j.scsadv.2026.100065). **LSTM-T-KAN-M is its multivariate version: model and training code copied line by
line from the authors' public repository (commit `ce24938`, MIT licence), plus the authors' exp2 feature list.** In
detail:

- **Model code.** The classes `KANLinear`, `KAN` and `LSTM_T_KAN` of the repository's `src/load_forecasting/models/` were
  copied by hand, line by line (only whitespace and comments condensed), with the hyper-parameters of the authors'
  `configs/exp2.yaml`: a one-layer LSTM (hidden 64) over a 132-hour input window; the last hidden state, after dropout
  0.1, feeds a linear head 64 → 744 plus a gated KAN residual tanh(α) · KAN(tanh(h / 4)) (α initialised at 0.5; KAN
  layers 64 → 128 → 64 → 744, grid 5, cubic splines, no grid update). 709,481 parameters.
- **Training code.** The training loop follows the authors' `src/load_forecasting/training/trainer.py` of the same
  commit: AdamW (weight decay 10⁻⁵), learning-rate halving on plateau of the validation loss (patience 3, minimum
  10⁻⁵), gradient-norm clipping at 1.0, batch 64 (shuffled, last partial batch kept), mean squared error on the
  normalised target, early stopping on the validation loss with the best weights restored.
- **Feature list.** The 16 input channels of the authors' `configs/exp2.yaml`, in their order: load (`Power (kW)`);
  maximum, median and minimum of the 24 hours ending at the hour, their range (`Peak_to_Peak`) and their RMS; the load
  1, 2 and 9 hours earlier; the 95th and 5th percentiles of the 168 hours ending at the hour (`Near_peak_load`,
  `Near_base_load`) and the number of the last 24 hours above / below them (`High_load_duration`,
  `Base_load_duration`); the largest absolute level-2 Haar approximation coefficient of the last 24 hours
  (`Approx_max`); `Hour` = cos(2π h / 23) as in the authors' `run_exp2.py`; `Is_holiday` = 1 on day type 7. Every channel
  is computed from hours before the origin only (299 hours of load); M reads no temperature and no calendar of the
  forecast days.
- **Not a strict reproduction of the paper.** The repository was read only (nothing in it was executed or installed),
  and the configuration differs from the paper and from the authors' code in the ten points below and in the training
  budget of the dated amendment that follows them.

| # | The paper / the authors' code | LSTM-T-KAN-M here | Reason |
|---|---|---|---|
| 1 | horizons of 1–30 days, one model per horizon, longest 720 steps | one model, all 744 steps in one forward pass | the 744-hour horizon of this evaluation |
| 2 | one model per building (a Suzhou office) | one model per population, trained across its units | the first release's rule for its trained baselines; most units have too few windows before the cutoff |
| 3 | Min-Max scaling fitted on the first 70% | per-unit z-score with mean and standard deviation from hours before the cutoff | a model shared by many units needs per-unit scaling; as for the other trained baselines |
| 4 | 70 / 10 / 20 chronological split, stride-1 windows | the first release's cutoff, origin step 24 h, at most 16,000 training and 2,000 selection pairs | the first release's rule |
| 5 | 3-point smoothing of the series (paper) | none | the evaluation targets are not smoothed |
| 6 | up to 100 epochs, early-stopping patience 10 (authors' configuration and trainer) | at most 30 epochs, patience 8 (amendment below; the protocol first set 50 and 10) | the study lead's decision on the training budget, taken before any forecast was scored |
| 7 | learning rate 10⁻⁴ (configuration) or 10⁻³ (paper text) | chosen from {10⁻⁴, 3 × 10⁻⁴, 10⁻³} by the lowest selection loss of seed 0 | the two sources disagree; the grid contains both and is the first release's grid |
| 8 | a single run, seed 42 | the mean of seeds 0, 1 and 2 (clipped at zero) | the first release's rule |
| 9 | pre-engineered feature columns (the input file is not public); per-horizon feature selection in the paper | one fixed list of the 16 exp2 channels; the definitions above are the study's reading of the column names, because the repository holds no feature-construction code; no weather (none is in the exp2 list) | the authors' feature code is not public |
| 10 | validation windows drawn inside the validation split | the selection set is the last 20% of training origins by time; the training targets end before its first origin | the first release's rule |

**Amendment of the training budget (same day as the protocol, before any forecast was scored).** At most 30 epochs and
early-stopping patience 8, against the authors' 100 epochs and patience 10; everything else of the authors' recipe is
unchanged. Of LSTM-T-KAN-M, only the BDG2 model had already been trained under 50 epochs and patience 10; replaying the
30 / 8 rule on its recorded per-epoch selection curves left the selected learning rate and the best epochs of all three
seeds unchanged (12, 10 and 16), so its forecast was kept byte for byte. All other populations were trained under
30 / 8.

- **Data protocol of the first release's panel.** One model per population across its units, the panel's training
  cutoff and training pairs, all training and selection targets ending at or before the cutoff, scoring on the late
  windows only. Because the authors' trainer needs a validation set for its schedule, early stopping and weight
  restoration, the final seeds are trained on the training pairs only (the selection pairs take no gradient steps).
- **Budget.** At most 30 epochs with patience 8, a plateau schedule and best-weight restoration: larger than that of
  the release's other trained baselines (at most 15 epochs, patience 3).
- **Ranks.** The study's protocol for this baseline kept the multivariate version in tables of its own, without a rank;
  in this repository it is an ordinary baseline in every table and rank, as are the 20 others.
- **CINELDI.** Like the first release's trained baselines that read the calendar (GBT-T, the linear-context model,
  iTransformer-X), LSTM-T-KAN-M was retrained on the corrected CINELDI store with the same code, protocol,
  learning-rate grid and seeds. Before the retrain, the same code run on the original store reproduced the stored
  original-store forecast bit for bit. The retrain passed its checks (training cutoff 2021-04-01, the CINELDI test
  start; selected learning rate 10⁻³; best epochs 14, 12 and 11 for seeds 0, 1 and 2), and its forecast is the one
  scored in every CINELDI row; the column `flag` is empty on every row of every result file
  (`populations` → `cineldi` → `extra_baselines` in `results/compare_baselines.json`).

### The linear-context model (TiDE implementation)

The first release's TiDE implementation ends its temporal decoder in a layer normalisation over a single output, which
is constant. Its forecast is therefore a linear map of the 1,344-hour load context plus a constant, and its covariates
have no effect. It keeps the model id `TiDE` in the result files, is labelled `Linear-context (TiDE implementation)`,
and was not retrained. It remains among the five baselines given the shared covariate set because it received them.

### The corrected GBT-T

GBT-T is the first release's own gradient-boosting implementation. Its first version normalised each window by its own
context scale with a floor that exploded on flat or all-zero contexts; the scores here are of the corrected version,
which uses the unit's pre-cutoff standard deviation only where the floor was active. On EWELD its per-window scaling
still overshoots on many windows, so read GBT-T's EWELD result as a limit of that design.

### Training budget of the trained baselines

This section takes up one alternative explanation of the comparisons with trained models: that they were held back by
a small training budget. The release's trained baselines were trained with at most 15 epochs and early-stopping
patience 3. The first release retrained the four load-only trained families (DLinear, iTransformer, LSTM, PatchTST) on
each population with at most **30 epochs and patience 8**, with nothing else changed (cutoff, training pairs, selection
set, learning-rate grid, seeds, normalisation and scoring), under a setting fixed before any relaxed model was trained
or scored. Those saved forecasts are scored here against ankyra-lite on the test windows
([`results/training_budget_sensitivity.csv`](../results/README.md#training-budget-and-long-context-baselines); nothing
was trained for this repository; on CINELDI the relaxed forecasts, made on the original store, are matched to the
corrected store by origin + 1,416 hours, with equal targets checked). Every population had been read before, so this is
a re-evaluation; there is no pass line, and the main tables are not replaced.

| Model | ankyra-lite resolved better, hourly, original → relaxed budget (of 9) | Same, monthly energy | Relaxed against original, hourly: resolved better / not resolved / resolved worse / identical forecasts | Selected run used all 30 epochs |
|---|---:|---:|---:|---:|
| DLinear | 9 → 9 | 7 → 7 | 1 / 3 / 2 / 3 | 1 |
| iTransformer | 7 → 6 | 4 → 4 | 0 / 3 / 1 / 5 | 0 |
| LSTM | 9 → 9 | 9 → 7 | 7 / 1 / 0 / 1 | 4 |
| PatchTST | 7 → 7 | 3 → 2 | 0 / 4 / 0 / 5 | 1 |

- **ankyra-lite 0.2.0 against the retrained baselines.** 35 of the 36 hourly verdicts are unchanged; on HEEW the
  contrast with iTransformer becomes not resolved (+8.3% → +6.5%). No contrast is resolved in favour of a baseline under
  either budget, and no point estimate changes sign. In monthly energy 3 of the 36 verdicts change, each from resolved
  to not resolved: LSTM on BDG2 (+18.9% → +15.8%) and CINELDI (+28.1% → +11.4%), PatchTST on the households
  (+17.8% → +12.4%).
- **LSTM gains from the larger budget.** Its own hourly error falls resolvably on seven populations: Cambridge 3.4%,
  HEEW 7.6%, EWELD 7.6%, GoiEner non-household 5.0%, GoiEner households 2.0%, Drammen 9.3%, CINELDI 11.9% (these
  baseline-against-baseline contrasts do not involve ankyra-lite). ankyra-lite's margin over it shrinks accordingly and
  stays resolved, for example from 25.3% to 15.2% on CINELDI and from 29.8% to 22.6% on Drammen.
- **The other three families change little.** In 13 of their 27 cases the relaxed run selects the same learning rate
  and epoch count as the original, so the forecasts are identical. DLinear improves resolvably on Cambridge (2.6%).
- **A larger budget is not always better on the test windows.** By construction the relaxed model's error on the
  selection set is no higher than the original's, but it is resolvably worse than its original on the test windows in
  three cases: iTransformer on EWELD (14.0%), DLinear on EWELD (3.1%) and DLinear on the GoiEner households (0.5%). The
  relaxed models are reported as they are; the better of the two budgets was not picked per population.

Reading and limits. The hourly verdicts held under this one relaxation of the training budget, with one exception. The larger budget may
still be too small for LSTM: on Cambridge and GoiEner non-household its selected epoch is the 30th, and on 6 of the 36
models the selected run was not stopped early. Only the budget was varied, with one alternative; architecture, context
length and learning-rate grid were not. The covariate-informed trained baselines (the linear-context model,
iTransformer-X, GBT-T) were not retrained; LSTM-T-KAN-M already has the larger budget
([above](#lstm-t-kan-m)).

### What each side may know

- **ankyra-lite** uses only pre-origin load and temperature, the forecast month's calendar, the category and a
  temperature-anomaly scale fitted before the training cut ([METHOD.md](METHOD.md#information-at-the-origin)). Its
  carrier's newest training label ends at least 186 days and 24 hours before the first test origin.
- **The trained baselines** are fitted on targets that end before the test start: up to half a year more recent than
  ankyra-lite's carrier. The 186-day gap is what lets the anchoring layer score the carrier out of sample at six
  pseudo-origins; it is a structural disadvantage for ankyra-lite in recency, not an advantage.
- **The covariate-informed baselines** receive future temperature only as its climatological expectation, never as
  observed weather.
- **LSTM-T-KAN-M** reads the last 299 hours of load and the calendar of those hours; it has no temperature, no
  calendar of the forecast month and a shorter context than ankyra-lite.
- **The zero-shot foundation models** have no local training; their pretraining corpora are the one exposure no
  forecaster here controls.

## Estimands

The estimands, intervals and random streams are those of the first release.

- **Unit-equal log RMS ratio (primary).** For unit i, R_i is the RMS of its hourly errors over its test windows, and
  r = N⁻¹ Σ_i log(R_i^lite / R_i^baseline). It is reported as the improvement 100[1 − exp(r)]; positive favours
  ankyra-lite. A unit whose error is exactly zero for either forecaster has no finite log ratio and drops out of that
  contrast.
- **Intervals.** 95% intervals from 2,000 bootstrap replicates that resample units and target-midpoint months
  independently (unit × month). A contrast is *resolved* when its interval excludes zero; *not resolved* does not mean
  equally good. Every hourly contrast draws from a fresh generator with the study's seed; the energy contrasts of one
  population share one generator (models in sorted order); every peak contrast draws from a fresh generator.
- **Mean per-unit rank.** The 22 forecasters are ranked within each unit by RMSE (rounded to 12 decimals, ties
  averaged); the mean over units is reported, and the position is 1 + the number of forecasters with a lower mean rank.
  It stays defined where some units have exactly zero error.
- **Pooled MSE ratio.** The ratio of the two forecasters' mean squared errors over all windows; large units weigh more.
- **Why several summaries.** Pooled and unit-equal summaries can disagree in sign for an algebraic reason (P19 in
  [theory/PROOFS.md](../theory/PROOFS.md)), and absolute-error metrics favour median-type forecasts while squared-error
  metrics favour mean-type ones. No single summary is reported alone.

## How each table is scored

Every table reads ankyra-lite's anchored forecast (its delivered trajectory, its level and peak readouts and its state
flags) and the saved forecasts of the 21 baselines on the same test windows; nothing is trained or tuned for the
comparison.

| Quantity | Definition | File |
|---|---|---|
| Hourly contrasts | the primary estimand on the hourly errors over the 744 hours of the delivered trajectory (after the continuation layer); the same trajectory enters the ranks, rank tests, conventional metrics, scaled errors, error by forecast day and error blocks | `benchmark_pairwise.csv` |
| Mean per-unit rank | as above, 22 forecasters | `benchmark_mean_unit_rank.csv` |
| Rank tests | per-unit RMSE (unrounded), units as blocks: Friedman's test of equal ranks, the Nemenyi critical difference at 5%, paired Wilcoxon signed-rank tests of ankyra-lite against each baseline with Holm's correction; per population and for the six test populations pooled (units concatenated). Computed after scoring; descriptive | `rank_tests.csv` |
| Conventional metrics | RMSE and MAE (mean over units, kW); CV(RMSE), NMBE and WAPE (median over units, %); WAPE pooled over windows; units with a mean load below 10⁻⁶ kW are left out of the ratio metrics | `conventional_metrics_late.csv` |
| Scaled errors | each window's RMSE and MAE divided by the in-sample error of the weekly seasonal naive (lag 168 h) over its 1,344-hour context; median and geometric mean over units; windows with a zero scale left out. Descriptive | `scaled_errors.csv` |
| Error by forecast day | for unit u and day d, the RMSE of that day's 24 hours over the unit's windows, in % of the unit's mean load; geometric mean over one fixed set of units whose daily errors are nonzero for all 22 forecasters (so the ratio of two curves on day d is the primary estimand restricted to that day); also unit-mean RMSE and MAE and median-unit CV(RMSE) and normalised MAE. Six test populations | `lead_day_metrics.csv` |
| Energy through the month | the error of the energy of each day, and of the energy from day 1 through day d, with the same aggregation (ankyra-lite before the continuation layer); day 31 is the monthly energy error. Six test populations | `lead_day_energy.csv` |
| Continuation layer | the primary estimand of ankyra-lite with the layer against the same carrier and anchoring layer without it, by lead segment (hour 0, hours 0–2, day 1, days 2–7, days 8–31, the whole window) and the energy of the delivered trajectory; per population and pooled over the 1,891 units; and what the layer did on every window (bypasses, clipped anomalies, the reference-day fallback, the largest \|Δ\|, the delivered mean against the level readout) | `continuation_layer.csv`, `continuation_record.csv` |
| Monthly energy | the primary estimand on the monthly energy error, 744 × the error of the window mean of the trajectory (the first release's definition): for ankyra-lite the trajectory before the continuation layer (readout rule A, [METHOD.md](METHOD.md#readouts)), for every baseline its own; the median-unit absolute percentage energy error of each forecaster; BDG2 also without the three near-zero meters. A sensitivity replaces ankyra-lite's trajectory mean by its level readout (they differ only where the nonnegativity projection binds). Nine populations | `energy_error.csv`, `energy_level_readout.csv` |
| Error blocks | the hourly squared error of every window splits exactly into a level, a daily-path and a within-day part (P1); the share of each block, pooled and at the median unit, and ankyra-lite's contrast with TimesFM and with Chronos-2-X within each block. Six test populations; descriptive | `block_shares.csv` |
| Peak readout | below | `peak_readout.csv` |

**Peak readout rule.** ankyra-lite's peak readout is the anchoring layer's envelope operator on the daily means of its
trajectory before the continuation layer (plus, for each day, the largest recent excursion of the same day type; the
readout on the seasonal naive on state windows). The first release's rule compares it with **the same readout applied to each
baseline's trajectory**, so that both sides use the same operator (`same readout on X`). The file also holds the
contrast with each baseline's own trajectory maximum (`trajectory maximum of X`), with ankyra-lite's own trajectory
maximum (that of the delivered trajectory, after the layer), and with the previous window's observed peak (the maximum of the last 744 context hours). The error is the
absolute peak error; the estimand is the unit-equal log ratio on it.

**Further tables** (ankyra-lite against versions of itself, its parts, its seeds, other carriers in the same anchoring
layer, the handover arms and the lead weeks, the within-day trust, whole-window and generic combinations, the level
candidates' hit rate, the history length, the anchoring record, the training budget of the baselines, long-context
baselines, the training curves, the recipe study, perturbed contexts, the constants and the cost) are described file
by file in [results/README.md](../results/README.md). The rows of switched-off parts of the anchoring layer re-run the
unchanged anchoring layer on the saved six-seed carrier forecasts (`evaluation/lite_reanchor.py`) and apply the
continuation layer as released; the re-run of the six-seed ensemble must reproduce the evaluated product bit for bit,
before and after the layer, before any of them is used. The handover arms, the
whole-window combinations, the within-day trust and the hit rate read the layer's own fields of every window from one
more re-run (mode `fields`), which must reproduce the product and the call's weights exactly; they compare forecasts
before the continuation layer. The rows of the two three-seed groups (seeds 0–2 and 3–5) read the stored group
products, with the continuation layer applied.

## State windows and the three near-zero BDG2 meters

On a window where the anchoring layer's off-state rule (the last 168 observed hours at most 10⁻⁶ kW) or micro-load rule
(all 1,344 context hours within 10⁻³ kW of zero) fires, ankyra-lite delivers the weekly seasonal naive: the last
168 hours repeated, missing hours as 0 ([METHOD.md](METHOD.md#state-windows-and-the-seasonal-naive-rule)). The rules
read only the load context, so the same windows are state windows for every recipe: 33 of the 474 BDG2 test windows,
13 of 890 on GoiEner non-household, 35 of 696 on the households and 245 of 523 on EWELD; none on the other five
populations. The context of a test window is complete, so on these windows ankyra-lite's forecast equals that of the
`SN-week` baseline, and the continuation layer is bypassed.

**The three near-zero BDG2 meters.** Three BDG2 meters (Lamb_education_Harold, Lamb_education_Hillary, Lamb_office_Jo)
read 0.0002–0.0003 kW in every window and hold 10 of the 474 test windows. The micro-load rule fires on all of them, and
the seasonal naive repeats the meter's constant reading, so ankyra-lite's error there is exactly zero. Consequences:

- the unit-equal contrasts need a nonzero error on both sides, so these meters drop out of them: the BDG2 point
  estimates with and without them are equal, and only the intervals differ (13 of the 21 hourly contrasts are
  resolved better with them, 12 without; iTransformer is the one that is not resolved without them);
- they count in the mean per-unit rank, where ankyra-lite shares the lowest error (RMSE rounded to 12 decimals, ties
  averaged) with the forecasters that also repeat the constant reading, such as `SN-week`: its mean unit rank is 4.676
  on all 142 units and 4.683 on the 139 without them, first of 22 in both cases;
- in contrasts of the carrier alone, which has no state rule, they can dominate the BDG2 unit means, because the
  carrier's error there is not zero while some baselines' errors are near zero.

`bdg2_near_zero_sensitivity.csv` gives the BDG2 contrasts and ankyra-lite's mean rank with and without them;
`energy_error.csv` and `seed_spread.csv` carry BDG2 rows without them.

## Intervals

ankyra-lite does not produce an interval of its own. For a description, the evaluation applies the anchoring layer's
interval readout of the first release, unchanged, and places it around ankyra-lite's delivered trajectory (after the
continuation layer):

- the residuals are those of the **history-only forecaster** (historical level and daily path with the same-day-type
  within-day default, no carrier) at up to 12 completed pseudo-origins of the unit; they do not depend on the carrier;
- each residual is divided by its pseudo-window's origin scale, and the residuals are pooled by hour of day and workday
  status; their empirical quantiles (levels 0.05, 0.1, 0.5, 0.9, 0.95), multiplied by the scale at the origin, are added
  to ankyra-lite's delivered trajectory. The band is placed around the trajectory but is not symmetric about it, because
  the residual quantiles of the history-only forecaster need not straddle zero;
- they are compared with the native quantiles of TimesFM 2.5 (0.1, 0.5, 0.9) and Chronos-2 (0.05 to 0.95): coverage at
  80% and 90%, by forecast week, width, pinball losses and the 80% Winkler score, whose unit-equal contrasts use the same
  bootstrap as above.

The interval was designed for the first release's forecaster and not re-designed or re-tuned for this carrier; its
numbers are descriptive. On units that switch off (EWELD in particular) its residuals are divided by a scale at its
floor, and its width carries no information.

## Why the recipe returns the validation span

This is version 0.1's recipe study; its rows are unchanged in 0.2.0, whose recipe Rw keeps the span return and adds
loss weights in stage 2. The recipe was the last part of ankyra-lite 0.1 to be fixed. Its first stage (training from scratch with early stopping on
the fixed 93-day validation span) took 23.7–38.3% of the own windows of the three short-record populations out of
training, and with them the season that part of their test months needed
([DATA.md](DATA.md#training-validation-and-test-sets)). The study compared three continuations of the same stage-1
checkpoints (three seeds each, on the test windows of all nine populations, Drammen and CINELDI on the corrected store).
The codes in the last column are the values of `first_arm` and `second_arm` in
[`results/recipe_history.csv`](../results/README.md#recipe-history):

| Arm | What it does | Code |
|---|---|---|
| from-scratch stage alone | stage 1 only: the 93-day validation set held out, early stopping (at most 30 epochs, patience 5) | `P3s` |
| plain continuation (control) | stage 1, then 8 more epochs with a fresh optimiser, the validation set still held out, final weights kept | `F` |
| span return (**released in 0.1**; the base of recipe Rw) | as the plain continuation, plus the validation span returned to the training set at the population's own sampling density (labels ending within 24 hours of the cut not returned) | `R` |
| proportional validation set | stage 1, then a continuation with a validation span of 10% of the population's record (clipped to 38–93 days) and early stopping (the start weights count as epoch 0; at least 5, at most 30 epochs, patience 5) | `V` |

**Status.** The plan and its predictions were written before the one reading of the arms; the test windows had been
read many times before, so the reading gives directions and is not a confirmation. The plan predicted a gain on the
*switch layer* (the test windows of the three short-record populations whose target month had no same-season training
labels before the span is returned and has them after it: 150 BDG2, 406 GoiEner non-household and 415 household
windows), a smaller gain on the three populations pooled, and about zero on their other test windows (the *non-switch
layer*); its numeric ranges are in the study's plan, not in this repository. Estimand: the unit-equal log RMS ratio,
pooled over populations with each population equally weighted. Interval: the unit × month bootstrap of the three-seed
ensembles combined with the spread of the three single-seed contrasts (Welch 95%). Positive favours the first-named arm.
Every number below is a row of `results/recipe_history.csv`, copied unrounded there from the study's scoring record.

| Reading (hourly unless stated) | Span return against the plain continuation (`R vs F`) |
|---|---|
| Three short-record populations pooled (primary) | +1.17% [−0.64, +2.94], not resolved; all three seeds positive (+2.11, +1.11, +0.92); BDG2 +0.54, GoiEner non-household +2.21, households +0.74 |
| Their switch layer | **+3.11% [+0.98, +5.21], resolved**; positive on every seed (+4.19, +3.54, +2.49) and population (BDG2 +2.80, GoiEner non-household +5.18, households +1.32) |
| Their non-switch layer | −0.52% [−2.08, +1.02], not resolved |
| Nine populations pooled | +0.11% [−0.82, +1.03], not resolved |
| Six populations with development windows | −0.42% [−1.22, +0.37], not resolved |
| **Six populations with development windows, forecast days 2–7** | **−1.35% [−2.43, −0.28], resolved against the span return** (Drammen −4.35, Oslo −2.05, CINELDI −1.60, HEEW −0.65, Cambridge +0.04, EWELD +0.41) |
| **Drammen** (per-population point estimate, no interval) | **−2.90%**; monthly energy −8.76% (from the level readout and from the delivered trajectory alike); peak −4.24% |
| Three short-record populations: monthly energy, peak | energy +3.95% [−3.93, +11.23] from the delivered trajectory (+4.09% [−3.67, +11.28] from the level readout); peak +0.67% [−3.88, +5.02]; none resolved |
| Nine populations: hourly, monthly energy, peak (the plan's checks) | +0.11% [−0.82, +1.03]; +0.32% [−3.30, +3.81] (delivered trajectory); −0.61% [−2.52, +1.27]; every interval contains zero |

- **The span return is not uniformly better.** Its gain is concentrated where the training set lacked the season of
  the test month, as predicted; pooled over all nine populations it is not distinguishable from the plain
  continuation, it is worse on Drammen (the largest point loss, in hourly error, energy and peak), and over the six
  populations with development windows it is resolvably worse on forecast days 2–7. These losses are part of recipe R
  and therefore of recipe Rw.
- **The mechanism is not isolated.** Returning the span adds same-season data, but it also changes the recency of the
  training data, the diversity of the samples and the replay of older windows. The readings support the season
  explanation; they do not prove it.
- **The proportional validation set brought no gain.** Three short-record populations +0.39% [−0.89, +1.65] against
  the plain continuation, not resolved; on the four long-record populations, where it keeps the 93-day span and differs
  from the plain continuation only in epoch selection and budget, −0.35% [−1.15, +0.43] (not resolved). On the switch
  layer it is resolvably worse than the span return (−2.15% [−3.96, −0.36]), and on the non-switch layer its peak
  readout is resolvably worse than the span return's (−4.92% [−9.52, −0.52]). Its early stopping counts the start
  weights as epoch 0, and on BDG2 it kept them for every seed: its three-seed forecast there equals that of the
  from-scratch stage (`V vs P3s`, BDG2: exactly 0.0%).
- **Continuation alone.** Against the from-scratch stage alone, nine populations pooled (unit × month interval only, no
  seed layer; descriptive): plain continuation +0.97% [+0.55, +1.52], span return +1.08% [+0.36, +1.86], proportional
  validation set +0.86% [+0.56, +1.32]. Pooled over the nine populations, most of the difference between the released
  recipe and the from-scratch stage comes from training longer, not from the returned span; on the three short-record
  populations the span return against the from-scratch stage is +1.77% [−0.03, +3.36] and the plain continuation
  +0.61% [−0.03, +1.48]. On Drammen the span return is 2.30% worse than the from-scratch stage alone (per-population
  point estimate).

**Decision.** The author chose the span return on 19 October 2026, with the disclosure of the Drammen loss and of the
resolved loss on forecast days 2–7 above. It is a rule applied identically to every population (train, select the
stopping epoch, then use all pre-cut data), not a per-population choice.

Figure [18](../figures/fig18_recipe_history.png) (a–c) shows the readings of the recipe study.

## Version 0.2.0: the development check and the test read

**Candidate.** Recipe Rw (recipe R with inverse-noise loss weights in stage 2), six seeds averaged at every issue time,
the unchanged anchoring layer and state rule, and the origin-continuity layer with four frozen constants; energy,
level and peak read before the layer (readout rule A). Every row below is in `results/recipe_history.csv` (column
`reading`); configurations are named by recipe and seeds (R3 = 0.1.0: recipe R, seeds 0–2; R6, Rw3, Rw6) and `_c`
marks the continuation layer.

**Development check (re-used evaluation).** On the later development windows of the six populations that have them,
with carriers trained to a development cut, read once before the formal training: the layer on the Rw6 carrier
+0.48% [+0.37, +0.59] over the month and +9.49% on day 1; Rw6 against R6 +0.42% [+0.18, +0.67]; R6 against R3 +0.07%
(not resolved); the full candidate against R3 +0.97% [+0.62, +1.30] and +10.17% on day 1. The loss weights made the
peak readout resolvably worse there (Rw6 against R6 −0.92% [−1.62, −0.14]); the energy and peak guards written before
the read did not flag credible harm. The development windows had also served to choose the layer's form, so this check
is not an independent confirmation.

**The test read** (once; the test windows had been read many times before, so it sets directions only). Rules written
before the read could drop each change; none was triggered. Nine populations with equal weight; the 0.2.0-against-0.1
and weighting rows combine the unit × month bootstrap with the spread of the two three-seed groups, the others are unit
× month only:

| Contrast | Whole window | Hours 0–2 | Day 1 | Days 2–7 | Days 8–31 | Energy (curve before the layer) | Peak |
|---|---|---|---|---|---|---|---|
| 0.2.0 against 0.1 (`primary: 0.2.0 vs 0.1.0`) | **+1.15% [+0.60, +1.71]** | +28.46% | +13.97% | +2.67% | +0.67% | +1.03% [−1.77, +3.74] | −0.35% [−2.18, +1.45] |
| six seeds against three (`decomposition: seeds (R6 vs R3)`) | +0.41% [+0.15, +0.70] | +0.84% nr | +0.57% | +0.75% | +0.32% | +0.72% nr | −0.39% nr |
| loss weights (`decomposition: weighting (Rw6 vs R6)`) | +0.31% [+0.06, +0.56] | +0.76% nr | +0.81% | +0.39% nr | +0.30% | +0.30% nr | +0.04% nr |
| continuation layer on / off (Rw6) | +0.43% [+0.14, +0.78] | +27.30% | +12.77% | +1.55% | +0.04% nr | 0 (rule A) | 0 (rule A) |

"nr": not resolved; every other entry is resolved. By population, 0.2.0 against 0.1 over the month: BDG2 +1.71%,
GoiEner non-household +1.91%, the households +0.94% [+0.37, +1.51], Cambridge +1.42%, HEEW +1.44%, EWELD +0.19%,
Oslo −0.50%, Drammen +2.42%, CINELDI +0.81%; the three populations without development windows +1.52%
[+0.61, +2.42], the six with them +0.97% [+0.28, +1.65]. Descriptive reduced configurations against 0.1 (whole
window): R6 with the layer +0.86%, Rw3 with the layer +0.80%, Rw3 +0.33%, Rw6 +0.72%, 0.1 with the layer +0.50%. The
parts are not additive and were read on the same windows. Figure [18](../figures/fig18_recipe_history.png) (d–e).

## Results

Every number of this section is a row of a file in [`results/`](../results/README.md), written by the full
regeneration with ankyra-lite 0.2.0 (each block names its file); [`results/SUMMARY.md`](../results/SUMMARY.md) is
the readable summary of the same run. All of them re-evaluate test windows that had been read before
([above](#identity-of-the-evidence)). Improvements are 100[1 − exp(r)] of the unit-equal log RMS ratio, positive in
favour of ankyra-lite; `*` marks a contrast resolved in favour of ankyra-lite (the 95% unit × month interval lies
below zero) and `(−)` one resolved against it.

### Overview

Resolved better / resolved worse, out of the 21 baselines (`benchmark_pairwise.csv`, `energy_error.csv`,
`energy_level_readout.csv`, `peak_readout.csv`; ranks from `benchmark_mean_unit_rank.csv`; state windows from
`compare_baselines.json`):

| Population | Windows / units | State windows | Rank of 22 (mean unit rank) | Hourly | Energy | Energy (level readout) | Peak (same readout) | Peak (vs the baseline's trajectory maximum) |
|---|---|---:|---|---|---|---|---|---|
| BDG2 2017 | 474 / 142 | 33 | 1 (4.68) | 13 / 0 | 7 / 0 | 7 / 0 | 4 / 0 | 17 / 0 |
| GoiEner non-household | 890 / 481 | 13 | 1 (5.02) | 14 / 0 | 13 / 0 | 13 / 0 | 2 / 0 | 19 / 0 |
| GoiEner households | 696 / 696 | 35 | 1 (5.34) | 17 / 1 | 16 / 0 | 16 / 0 | 11 / 0 | 21 / 0 |
| Cambridge | 730 / 108 | 0 | 1 (3.04) | 21 / 0 | 20 / 0 | 20 / 0 | 12 / 0 | 21 / 0 |
| HEEW Arizona | 658 / 138 | 0 | 1 (5.11) | 14 / 0 | 7 / 0 | 7 / 0 | 7 / 0 | 16 / 0 |
| EWELD | 523 / 197 | 245 | 1 (7.09) | 13 / 0 | 6 / 0 | 6 / 0 | 5 / 0 | 19 / 0 |
| Oslo | 581 / 41 | 0 | 1 (1.78) | 20 / 0 | 19 / 0 | 19 / 0 | 17 / 0 | 17 / 0 |
| Drammen | 689 / 43 | 0 | 1 (1.67) | 20 / 0 | 13 / 0 | 13 / 0 | 6 / 1 | 19 / 0 |
| CINELDI | 475 / 45 | 0 | 1 (2.89) | 21 / 0 | 13 / 0 | 13 / 0 | 3 / 0 | 19 / 0 |

- **Rank.** ankyra-lite has the lowest mean per-unit rank of the 22 forecasters on all nine populations. The next
  forecaster is the per-unit ridge on BDG2 (7.01), GoiEner households (7.06) and CINELDI (5.69), iTransformer on
  GoiEner non-household (7.20), iTransformer-X on Cambridge (5.99), Chronos-2-X on HEEW (6.80) and Drammen (3.58),
  TimesFM on EWELD (8.16) and GBT-T on Oslo (2.63).
- **Hourly** (the delivered trajectory). The only resolved loss is on the GoiEner households, to TimesFM (−4.6%). The
  resolved gains range from +4.3% (the per-unit ridge, households) to +54.0% (Oslo). On Oslo the one contrast that is
  not resolved is GBT-T (+2.9%), on Drammen Chronos-2-X (+7.3%); on Cambridge and CINELDI all 21 are resolved.
- **Energy** (the monthly energy error of ankyra-lite's trajectory before the continuation layer and of each
  baseline's own trajectory): no resolved loss on any population. Replacing that trajectory's mean by ankyra-lite's
  level readout gives the same counts (`energy_level_readout.csv`); the two differ only where the nonnegativity
  projection binds (BDG2 1, GoiEner non-household 138, households 21, Cambridge 10, EWELD 18, Drammen 2, CINELDI 27
  windows; none on HEEW and Oslo; column `windows_projection_binds`). By the median-unit absolute percentage energy
  error ankyra-lite is first of 22 on GoiEner non-household (12.09%), the households (12.99%), Oslo (7.61%) and
  Drammen (7.72%), second on BDG2, third on Cambridge and CINELDI, fifth on HEEW and eighth on EWELD
  (`energy_error.csv`).
- **Peak.** Against the same readout on each baseline's trajectory (the first release's rule), the one resolved loss
  is to GBT-T on Drammen (−10.6%). Against each baseline's own trajectory maximum the readout is resolved better on
  16–21 of the 21 per population and worse on none, and against the maximum of ankyra-lite's own delivered trajectory
  it is resolved better on all nine populations (+21.3% on Oslo to +68.2% on the households). Against the previous
  window's observed peak it is resolved better on Oslo (+26.8%) and resolved worse on CINELDI (−18.2%); elsewhere not
  resolved, from −34.7% (GoiEner non-household) to +15.7% (Drammen).
- **Peak against the carrier alone.** The same readout applied to the carrier's own six-seed forecast at the origin
  (no anchoring layer, no state rule; comparator `same readout on the carrier alone`, not a baseline) is the
  counterpart of the first release's row against its fixed division. ankyra-lite's readout is resolved worse than it on
  Oslo (−2.5%) and Drammen (−2.9%) and not resolved on the other seven (−6.9% on HEEW to +5.6% on GoiEner
  non-household). The readout depends on a forecast only through its daily means (the excursions come from the context;
  P15, P16), so it inherits the daily-path contrasts of [contrast (ii) by block](#contrast-ii-by-block-and-the-carrier-free-control):
  there ankyra-lite's daily path (before the layer) is resolved worse than its carrier's on two populations, and the
  within-day block, where the layer gains most, does not enter the readout. On these populations the anchoring layer's
  hourly gains do not carry measurably into the peak readout.

Figures [0](../figures/fig0_advantage.png) (the four zero-shot variants), [2](../figures/fig2_test_ranks.png) (ranks
and head-to-head counts), [3](../figures/fig3_test_pairwise.png) (hourly contrasts with intervals),
[11](../figures/fig11_energy_error.png) (monthly energy), [5](../figures/fig5_peak_operator.png) (peak readout) and
[12](../figures/fig12_consistency.png) (position on every population); [figures/README.md](../figures/README.md) lists
each figure's source files.

### Against the zero-shot foundation models

Improvement in % (`benchmark_pairwise.csv`, `energy_error.csv`, and `peak_readout.csv` with the same readout):

| Population | Hourly vs TimesFM | Energy vs TimesFM | Peak vs TimesFM | Hourly vs Chronos-2-X | Energy vs Chronos-2-X | Peak vs Chronos-2-X |
|---|---|---|---|---|---|---|
| BDG2 2017 | +0.7 | +14.7 | +4.0 | −4.0 | +3.3 | −0.2 |
| GoiEner non-household | +4.7 | +20.2 * | +0.8 | −0.5 | +12.4 | +3.4 |
| GoiEner households | −4.6 (−) | +32.8 * | +10.2 * | −2.9 | +32.8 * | +11.7 * |
| Cambridge | +15.4 * | +24.6 * | +10.0 * | +10.0 * | +19.7 * | +4.5 |
| HEEW Arizona | +2.5 | −0.1 | −1.2 | +0.8 | −0.8 | +0.0 |
| EWELD | +1.5 | −2.8 | +2.9 | +4.2 | +16.8 | +0.6 |
| Oslo | +23.0 * | +34.9 * | +16.4 * | +14.6 * | +20.4 * | +7.1 |
| Drammen | +20.9 * | +29.0 * | +14.6 | +7.3 | +4.3 | −1.1 |
| CINELDI | +10.5 * | +22.4 * | +6.9 | +6.7 * | +16.7 | +7.6 |

On BDG2, GoiEner non-household, HEEW and EWELD no hourly contrast with the four zero-shot variants (TimesFM,
TimesFM-X, Chronos-2, Chronos-2-X) is resolved, and some point estimates are negative (BDG2 against Chronos-2-X −4.0%,
GoiEner non-household against Chronos-2-X −0.5%, EWELD against Chronos-2 −0.2%). On the households the energy contrasts
are resolved in favour of ankyra-lite against all four variants (+28.7% to +33.6%), and the hourly one against it for
TimesFM (−4.6%; TimesFM-X −0.9%, Chronos-2 −2.8% and Chronos-2-X −2.9% are not resolved). Over the 36 contrasts per
readout with the four variants, hourly error is resolved better in 15 and worse in 1, monthly energy better in 19 and
worse in none, the peak better in 9 and worse in none.

**With a year of context.** `long_context_baselines.csv` (outside the 22 forecasters and their ranks; descriptive)
scores zero-shot TimesFM 2.5 with an 8,736-hour context and Chronos-2 with 8,192 hours, the zero-shot runs of the first
release. Hourly, ankyra-lite is resolved better than TimesFM-8736 on Cambridge (+12.3%), Oslo (+22.5%), Drammen
(+15.3%) and CINELDI (+7.7%), not resolved on four populations (BDG2 −6.1%) and resolved worse on the GoiEner
households (−5.3%); against Chronos-2-8192 it is resolved better on the same four and worse on the households (−5.2%).
In monthly energy it is resolved better than TimesFM-8736 on five populations and than Chronos-2-8192 on four, and worse
than neither. The longer context itself makes TimesFM resolvably better than its 1,344-hour version on four populations
and Chronos-2 on one (worse on two).

### Rank tests

Per-unit RMSE, units as blocks (`rank_tests.csv`; computed after scoring, descriptive):

- **Six test populations pooled** (1,762 units): Friedman p < 10⁻³⁰⁰; Nemenyi critical difference 0.786;
  ankyra-lite's mean rank 5.234 of 22, against 7.451 for the next forecaster (the per-unit ridge) and 8.568 for the
  third (iTransformer-X), a gap larger than the critical difference. Holm-corrected Wilcoxon tests find it better than
  all 21 baselines and worse than none.
- **Per population** it is significantly better than 21 of the 21 baselines on seven populations, than 20 on Oslo
  (GBT-T is not significantly different there) and than 19 on EWELD (TimesFM and Chronos-2); no baseline is
  significantly better on any population.

Figure [15](../figures/fig15_rank_tests.png) shows the pooled mean ranks with the critical difference and the paired
tests per population.

### Conventional metrics

ankyra-lite's position among the 22 forecasters (`conventional_metrics_late.csv`; RMSE and MAE are unit means in kW,
CV(RMSE) and WAPE are taken at the median unit):

| Metric | BDG2 | GoiEner NH | GoiEner HH | Cambridge | HEEW | EWELD | Oslo | Drammen | CINELDI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RMSE (unit mean) | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 |
| MAE (unit mean) | 2 | 2 | 4 | 1 | 4 | 4 | 1 | 1 | 4 |
| CV(RMSE) (median unit) | 3 | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 1 |
| WAPE (median unit) | 2 | 4 | 5 | 1 | 4 | 5 | 1 | 1 | 2 |
| WAPE (pooled) | 2 | 2 | 5 | 1 | 5 | 6 | 1 | 1 | 4 |

A position is one plus the number of forecasters strictly ahead at the file's two decimals; ties at two decimals: on
the households the per-unit ridge has the same RMSE and, with TimesFM-X, the same MAE; on HEEW TimesFM-X has the same
MAE. The forecasters ahead of it are mostly the zero-shot foundation models (Chronos-2-X, Chronos-2, TimesFM,
TimesFM-X). GBT-T is ahead on BDG2's CV(RMSE), on HEEW's RMSE, MAE and pooled WAPE and on CINELDI's MAE and pooled
WAPE, and the daily seasonal naive on EWELD's pooled WAPE. MAE and WAPE are minimised by the median of the predictive
distribution, squared-error metrics by its mean; ankyra-lite's carrier is trained, and the anchoring weights are
estimated, on squared error.

Figure [8](../figures/fig8_loss_metrics.png) shows the four losses of every forecaster on every population.

### By forecast day and energy through the month

Six test populations, 22 forecasters (`lead_day_metrics.csv`: each day's CV(RMSE) over its 24 hours, geometric mean
over a fixed set of units, the delivered trajectory; `lead_day_energy.csv`: the error of the energy from day 1 through
day d, ankyra-lite before the continuation layer, day 31 being the monthly energy):

| | BDG2 | GoiEner NH | GoiEner HH | Cambridge | HEEW | EWELD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| hourly loss: position on day 1 | 6 | 2 | 5 | 4 | 5 | 5 |
| hourly loss: days of 31 on which it is lowest | 8 | 0 | 0 | 27 | 11 | 0 |
| hourly loss: worst position on any day | 6 | 6 | 5 | 4 | 6 | 11 |
| energy to date: days of 31 on which it is lowest | 12 | 12 | 4 | 24 | 6 | 0 |
| energy to date: position on day 31 | 1 | 1 | 3 | 1 | 1 | 6 |
| energy to date: days 8–31 below all four zero-shot variants (of 24) | 12 | 20 | 24 | 24 | 6 | 0 |

- **The first days of the month are still its weakest.** On day 1 zero-shot foundation models are ahead of it on every
  population (Chronos-2-X alone on GoiEner non-household; TimesFM, Chronos-2 and Chronos-2-X on Cambridge; all four
  variants on the other four), and on BDG2 also the daily seasonal naive; no trained baseline is ahead on day 1 (in
  version 0.1 trained baselines were ahead on five of the six). The change came from the continuation layer
  ([below](#the-continuation-layer)).
- **After the first week** its hourly loss is the lowest on all 24 days from day 8 on Cambridge, on 11 on HEEW and 8 on
  BDG2, and on none on EWELD or either GoiEner population, where zero-shot Chronos-2-X or Chronos-2 is lowest.
- **Energy.** The cumulative energy error at day 31 is the lowest of the 22 on four of the six populations; on the
  households it is third (behind the per-unit ridge and `SP-profile`), on EWELD sixth (behind PatchTST, TimesFM-X,
  Chronos-2, TimesFM and the weekly seasonal naive). From day 8 it stays below all four zero-shot variants on every day
  on the households and Cambridge.

Figures [9](../figures/fig9_loss_by_day.png) (hourly loss by day), [9c](../figures/fig9c_continuation_layer.png)
(beside it: the continuation layer by lead time), [9b](../figures/fig9b_energy_by_day.png) (energy to date) and
[10](../figures/fig10_loss_by_day_relative.png) (each baseline relative to ankyra-lite).

### The continuation layer

`continuation_layer.csv`: ankyra-lite with the layer against the same carrier, anchoring layer and state rule without
it (the seasonal naive on state windows in both), hourly improvement by lead segment, unit × month intervals (no seed
layer: the same carrier on both sides; descriptive). Hours 0–2 is the file's segment `lead hours 0-3`.

| Population | Hour 0 | Hours 0–2 | Day 1 | Days 2–7 | Days 8–31 | Whole window |
|---|---|---|---|---|---|---|
| BDG2 2017 | +39.2 * | +32.2 * | +19.2 * | +3.7 * | +0.2 | +0.8 * |
| GoiEner non-household | +32.4 * | +29.8 * | +12.0 * | +2.4 * | +0.3 * | +0.5 * |
| GoiEner households | +18.4 * | +13.5 * | +8.4 * | +1.7 * | +0.1 * | +0.4 * |
| Cambridge | +22.6 * | +17.7 * | +7.6 * | +0.4 | −0.1 | +0.2 |
| HEEW Arizona | +31.9 * | +27.5 * | +14.3 * | +3.1 * | +0.3 * | +0.9 * |
| EWELD | +37.0 * | +28.4 * | +7.5 * | +1.1 | +0.3 | +0.5 * |
| Oslo | +35.1 * | +30.0 * | +12.6 * | −1.0 | −0.5 (−) | −0.3 |
| Drammen | +42.8 * | +39.5 * | +25.1 * | +2.3 | −0.1 | +1.0 |
| CINELDI | +24.8 * | +23.9 * | +6.4 * | +0.2 | −0.1 | +0.0 |
| nine populations pooled (1,891 units) | +28.0 * | +23.4 * | +11.1 * | +1.9 * | +0.2 * | +0.5 * |

- With populations equally weighted instead of units pooled, the same contrast is +0.43% over the month, +27.30% in
  hours 0–2 and +12.77% on day 1 (`recipe_history.csv`).
- The layer helps on day 1 on every population; after the first week it is close to zero, and on Oslo it is
  resolvably negative on days 8–31 (−0.5%), with a whole-window point estimate of −0.3%.
- The monthly energy of the delivered trajectory changes by +1.9% pooled (not resolved; −2.8% to +3.3% by population,
  none resolved). The energy readout is not changed by the layer (rule A).
- `continuation_record.csv`: 326 state windows bypassed the layer (33 BDG2, 13 GoiEner non-household, 35 households,
  245 EWELD); the fast anomaly was clipped on 6.4% and the day anomaly on 4.2% of the other windows (20.0% and 10.9% on
  the households); 33 EWELD windows used the reference-day fallback d* = 6; no span was unobserved or nonpositive; the
  largest |Δ| was 0.809 (Drammen 0.782). The delivered curve's mean differed from the level readout by 0.77% at the
  median, 7.8% at the 99th percentile and 310% at most (GoiEner households; [METHOD.md](METHOD.md#readouts)).

### Error blocks and scaled errors

- `block_shares.csv` (six test populations, the delivered trajectory, descriptive): the hourly squared error splits
  exactly into level, daily path and within-day blocks. Pooled over windows, the within-day block carries 69.3% of
  ankyra-lite's error on the households, 64.0% on Cambridge and 55.2% on GoiEner non-household; the level block the
  largest share on EWELD (48.9%) and HEEW (38.8%); the daily path on BDG2 (43.2%). Against TimesFM by block,
  ankyra-lite is better in the level on BDG2 (+15.3%), GoiEner non-household (+22.8%), the households (+34.7%),
  Cambridge (+23.7%) and HEEW (+2.5%) and worse in the level on EWELD (−4.3%); on the households its hourly deficit sits
  in the daily path (−9.8%) and the within-day block (−3.1%). The block contrasts carry no interval.
- `scaled_errors.csv` (descriptive): by median-unit RMSSE (weekly seasonal-naive in-sample scale) ankyra-lite is first
  on seven populations (GoiEner non-household, the households, Cambridge, HEEW, Oslo, Drammen, CINELDI) and second on
  BDG2 and EWELD. By median-unit MASE, an absolute-error scale, it is first on Cambridge, Oslo, Drammen and CINELDI,
  fifth on BDG2, GoiEner non-household, the households and HEEW and sixth on EWELD.

Figure [14](../figures/fig14_block_attribution.png) shows the block shares and the block contrasts.

### Intervals around ankyra-lite

The anchoring layer's interval placed around ankyra-lite's delivered trajectory (descriptive; [above](#intervals);
`intervals_by_population.csv`, `intervals_winkler_contrasts.csv`, `intervals_households.json`):

| Population | 80% coverage: ankyra-lite / TimesFM / Chronos-2 | 90% coverage: ankyra-lite / Chronos-2 | Winkler 80 vs TimesFM | vs Chronos-2 |
|---|---|---|---|---|
| BDG2 2017 | 0.766 / 0.349 / 0.629 | 0.858 / 0.776 | −2.7 | −17.2 |
| GoiEner non-household | 0.792 / 0.430 / 0.720 | 0.876 / 0.851 | +6.2 | −9.1 (−) |
| GoiEner households | 0.782 / 0.345 / 0.680 | 0.874 / 0.815 | +6.8 * | −1.4 |
| Cambridge | 0.790 / 0.393 / 0.710 | 0.880 / 0.836 | +15.3 * | −1.1 |
| HEEW Arizona | 0.776 / 0.373 / 0.726 | 0.863 / 0.848 | +9.3 | −2.5 |
| EWELD | 0.844 / 0.606 / 0.804 | 0.894 / 0.883 | −3.7 | −39.6 (−) |
| Oslo | 0.847 / 0.379 / 0.678 | 0.920 / 0.813 | +21.0 * | +4.5 * |
| Drammen | 0.855 / 0.388 / 0.725 | 0.917 / 0.841 | +20.6 * | +3.2 |
| CINELDI | 0.818 / 0.438 / 0.736 | 0.901 / 0.859 | +15.5 * | −4.3 (−) |

- **Coverage of the 80% band:** 0.766 (BDG2) to 0.855 (Drammen), against 0.345 to 0.606 for TimesFM's native quantiles
  and 0.629 to 0.804 for Chronos-2's; first week 0.808–0.884, fourth week 0.718–0.829.
- **80% Winkler score.** Against TimesFM's native interval it is resolved better on five populations (+6.8% on the
  households to +21.0% on Oslo) and worse on none; against Chronos-2's it is resolved better on Oslo (+4.5%) and
  resolved worse on EWELD (−39.6%), GoiEner non-household (−9.1%) and CINELDI (−4.3%). On EWELD the band's width
  carries no information (mean 80% width 388,321 kW, from units that switch off).
- **Households.** The same interval around the carrier alone instead of ankyra-lite changes the Winkler score by
  0.10% (not resolved): the interval comes from the history-only residuals, not from the carrier.

Figures [13](../figures/fig13_intervals.png) (households) and [16](../figures/fig16_intervals_populations.png) (coverage
on every population and by forecast week).

### What the parts add

Hourly improvement of ankyra-lite 0.2.0 over the version without the part, or over the named alternative
(`component_contributions.csv`, `ablation.csv`; "identical": the two forecasts are the same on every window). The
recipe and carrier rows compare the whole 0.2.0 configuration with saved study products, so they include the loss
weights, the six seeds and the continuation layer as well:

| Part | BDG2 | GoiEner NH | GoiEner HH | Cambridge | HEEW | EWELD | Oslo | Drammen | CINELDI |
|---|---|---|---|---|---|---|---|---|---|
| anchoring layer, state rule and continuation layer together (against the carrier alone) | +3.6 * | +3.3 | −1.3 | +0.1 | +2.2 * | +0.5 | −0.5 | +0.8 | +1.3 |
| the carrier inside the anchoring layer (against the history-only configuration) | +8.0 * | +21.1 * | +11.9 * | +15.0 * | +13.8 * | +16.8 * | +20.1 * | +17.4 * | +6.2 * |
| continuation layer (same carrier and anchoring layer) | +0.8 * | +0.5 * | +0.4 * | +0.2 | +0.9 * | +0.5 * | −0.3 | +1.0 | +0.0 |
| within-day anchoring | +0.7 * | +2.4 | −0.0 | +0.3 | +0.8 * | −0.0 | +0.6 * | +0.9 * | +0.9 * |
| seasonal-naive state rule | +0.3 | −0.0 | +0.3 | identical | identical | +2.1 | identical | identical | identical |
| **0.2.0 against 0.1** (every change together) | +1.7 | +1.9 * | +0.9 * | +1.4 * | +1.4 * | +0.2 | −0.5 | +2.4 * | +0.8 |
| loss weights (recipe Rw against R, six seeds, with the layer) | +0.2 | +0.9 | +0.3 | +0.5 | +0.5 | −0.3 | −0.2 | +0.1 | +0.6 * |
| six seeds against seeds 0–2 (recipe Rw, with the layer) | +0.7 | +0.4 | +0.2 | +0.8 | −0.0 | +0.0 | −0.3 | +0.9 | +0.5 |
| six-seed ensemble against a three-seed group (mean of the two groups; point estimate) | +0.5 | +0.2 | +0.1 | +0.3 | +0.4 | +0.2 | +0.3 | +0.5 | +0.2 |
| 0.2.0 against the from-scratch stage alone (0.1 recipe study) | +2.9 * | +4.6 * | +2.3 * | +3.1 * | +2.3 * | +1.3 | +1.7 * | +0.2 | +1.6 |
| 0.2.0 against the plain continuation F (0.1 recipe study) | +2.2 * | +4.1 | +1.7 * | +2.0 * | +1.2 * | −0.2 | −0.4 | −0.4 | +1.1 |
| the carrier's level candidate, the handover and the within-day anchoring (against the fixed division) | +2.9 * | +4.3 * | +2.9 * | +7.9 * | +6.8 * | +8.4 * | +8.3 * | +7.5 * | +2.7 * |
| weekly handover weights (against a fixed half weight) | +0.9 * | −0.1 | +0.4 | +1.2 | +0.9 | +1.0 | +0.4 | +1.0 * | +0.0 |
| block-wise weights (against one whole-window weight of the same two forecasts) | −0.2 | −0.2 | +1.5 * | +0.4 | +0.4 | +0.5 | +0.4 | −0.0 | −0.2 |
| block-wise weights (against a half-and-half average of the same two forecasts) | +1.0 | +1.7 | +4.0 * | +3.9 * | +2.9 * | +3.1 * | +2.5 * | +1.5 * | +0.3 |

- **Labels.** In `component_contributions.csv` the two rows of the 0.1 recipe study keep their 0.1 names
  (`continuation with the span return (from-scratch stage -> released recipe)`, `span return (...)`); in 0.2.0 they
  compare the whole 0.2.0 configuration with the study's three-seed products of the from-scratch stage and of F, so they
  are not the span return's own contribution (that is `R vs F` in [the recipe study](#why-the-recipe-returns-the-validation-span)).
- **Anchoring the carrier.** Contrast (i), the anchoring layer against the carrier alone (before the continuation
  layer; `carriers_in_anchoring_layer.csv`), is resolved on two populations in hourly error (BDG2 +2.8%, CINELDI
  +1.3%) and on none in monthly energy; on the households it is −1.7% (not resolved). With the continuation layer the
  same contrast is resolved on BDG2 and HEEW (first row above). Contrast (ii), against the history-only configuration,
  is resolved on all nine in hourly error (+6.1% to +20.6%) and on five in monthly energy (Cambridge +20.0%, HEEW
  +14.5%, EWELD +34.8%, Oslo +20.7%, Drammen +12.7%).
- **What the two contrasts answer.** Contrast (i) asks whether the unit's own history helps the carrier; contrast (ii)
  whether the carrier-enabled configuration is better than the history-only one. Improving a carrier and the carrier
  contributing are not the same thing, so the two are reported side by side. In the linear forecast-combination
  setting, one forecast encompasses another when the optimal combination assigns zero weight to the latter. The two
  contrasts address the related question of incremental predictive value; they are not formal encompassing tests. An
  unresolved difference between two errors is not evidence of encompassing, and the shrunk weights that the anchoring
  layer uses are not test statistics for encompassing (on forecast encompassing: Chong and Hendry, 1986; Harvey,
  Leybourne and Newbold, 1998; [references](#references)). Each contrast is read by its 95% unit × month interval, as
  elsewhere on this page. Contrast (ii) measures the incremental performance of the carrier-enabled configuration; it
  does not isolate the information in the carrier's forecast, because the history-only configuration uses the
  day-type default rather than the analog-day shape and omits the state rules
  ([METHOD.md](METHOD.md#within-day-shape)). The carrier-free control
  ([below](#contrast-ii-by-block-and-the-carrier-free-control)) holds the procedure the same on both sides.
- **Small switches.** One within-day trust value per window, the short-gap rule and the micro-load rule each change the
  hourly error by at most 0.2%. The state rule matters only where state windows exist; no energy contrast of these
  switches is resolved.
- **Intervals of these rows** resample units and months only; the 0.2.0 rows with the seed layer are in
  [the 0.2.0 test read](#version-020-the-development-check-and-the-test-read).
- **Earlier carriers** (`ablation.csv`; eight populations, none on the corrected CINELDI store): against the same
  network without the copy pointer on the from-scratch recipe, ankyra-lite 0.2.0 is better by +1.8% to +5.0% (resolved
  on seven, not on Drammen, +2.2%); against the earlier structured network without the pointer, trained on weekly
  windows, by +2.0% to +11.7% (resolved on all eight). These rows include every 0.2.0 change.

Figure [4](../figures/fig4_ablation.png) shows these and the other rows of `ablation.csv`.

### What the handover contributes

The handover mixes the history side's daily means m^H (the level with the carrier candidate, plus the daily path) with
the carrier's daily means m^C week by week ([METHOD.md](METHOD.md#week-by-week-handover)). The first release's handover
ablation, rebuilt here with ankyra-lite's anchoring layer (`handover_granularity.csv`), varies only that weight and
keeps everything else as in ankyra-lite before the continuation layer: the level with the carrier candidate, the
delivered within-day block, the projection and the seasonal naive on state windows. The arms are computed from the
anchoring layer's own fields of every window (the `fields` mode of `evaluation/lite_reanchor.py`), and the weekly arm
reproduces the anchored forecast bit for bit (gate in `compare_baselines.json`). The four arms, by their labels in the
file:

- `A0`: no weight on the carrier's daily means;
- `Ah`: a fixed weight of one half;
- `AM`: one least-squares weight per window, from the same pseudo-origin pairs with the 31 days pooled, shrunk in the
  same way;
- `AW`: the four weekly weights, that is ankyra-lite before the continuation layer.

Hourly improvement of the first arm over the second (unit-equal; `*` resolved in its favour), the share of the weekly
arm's gain over `A0` that the fixed half already captures (the ratio of the two log ratios, from `compare_baselines.json`)
and the range of the mean weekly handover weight on the carrier (`lead_weeks.csv`):

| Population | Ah vs A0 | AW vs Ah | AW vs AM | Share of AW's gain captured by Ah | Mean weight on the carrier, weeks 1–4 |
|---|---:|---:|---:|---:|---|
| BDG2 2017 | +0.7 | +0.9 * | −0.1 | 0.44 | 0.48–0.56 |
| GoiEner non-household | +1.3 | −0.1 | −0.1 | 1.11 | 0.56–0.59 |
| GoiEner households | +1.8 * | +0.4 | −0.1 | 0.83 | 0.53–0.56 |
| Cambridge | +4.6 * | +1.2 | −0.2 | 0.80 | 0.63–0.71 |
| HEEW Arizona | +2.9 * | +0.9 | −0.3 | 0.76 | 0.59–0.67 |
| EWELD | +5.3 * | +1.0 | −0.5 | 0.85 | 0.54–0.62 |
| Oslo | +4.9 * | +0.4 | −0.2 | 0.92 | 0.68–0.70 |
| Drammen | +4.5 * | +1.0 * | −0.2 | 0.82 | 0.69–0.76 |
| CINELDI | +1.3 * | +0.0 | −0.1 | 0.99 | 0.57–0.60 |

- **Mixing.** Mixing in the carrier's daily means at a fixed half weight is resolvably better than no mixing on seven
  populations, and gives most of the weekly weights' gain: 0.76–0.99 of it on seven of the eight populations where the
  weekly arm is resolved better than `A0`, 0.44 on BDG2. On GoiEner non-household the weekly arm is not resolved against
  `A0` (+1.2%), so its share is not interpreted.
- **Per-unit weights.** Estimating the weight per unit adds −0.1% to +1.2% over a fixed half; the gain is resolved on
  BDG2 (+0.9%) and Drammen (+1.0%) and never resolvably negative.
- **Weekly weights.** Separate weekly weights add nothing over one weight per window: the point estimates favour the
  single weight on all nine populations (−0.1% to −0.5%), none resolved. The measured contribution is per-unit,
  error-weighted mixing, not weekly resolution; the weekly form is kept as the copied layer computes it.
- **Without the handover** the history side loses to the carrier: `A0` against the carrier alone is resolved worse on
  Cambridge (−5.9%), the households (−3.5%), Oslo (−6.2%) and Drammen (−4.9%).
- **The fixed division** (the history side's six-candidate level and daily path, without the carrier candidate and the
  handover, plus the carrier's own within-day block; ablation row `fixed division (history level and daily path, no
  handover)`) is resolved worse than ankyra-lite on all nine populations (+2.7% on CINELDI to +8.4% on EWELD in favour
  of ankyra-lite). `A0` differs from it by the carrier candidate in the level and the within-day anchoring and is
  resolved better than it on all nine (+0.7% to +3.2%).

**By forecast week** (`lead_weeks.csv`, hourly error of each week's hours; ankyra-lite with the continuation layer):
ankyra-lite against the carrier alone is resolved better on BDG2 in weeks 1, 2 and 4, on GoiEner non-household and HEEW
in week 1 and on CINELDI in week 4, resolved worse on the households in weeks 2 and 3, and not resolved in any week
elsewhere. Against TimesFM it is resolved worse in week 1 on HEEW and GoiEner non-household, in week 2 on BDG2 and in
every week on the households, and resolved better in the later weeks on Cambridge (weeks 2–4), Oslo (weeks 1–4),
Drammen (weeks 3–4) and CINELDI (weeks 3–4). The fixed division is resolved worse than the carrier alone in at least
one week on seven populations (in all four on Cambridge, the households and Oslo). The mean handover weight on the
carrier stays between 0.48 and 0.76 in every week of every population. Descriptive; the file has no interval across
seeds.

- **BDG2 against TimesFM, weeks 1 and 2.** These two point estimates (−81.2%, not resolved, and −13.1%, resolved)
  describe a few meters, not the population. In week 1 three meters (Lamb_office_Vasiliki, Lamb_assembly_Bertie,
  Lamb_office_Velma; one window each, origin 1 August 2017) read the same small non-zero value through the whole
  context and the first eight forecast days. TimesFM repeats that value almost exactly, so its error on these units is
  close to zero, while ankyra-lite, which also reads the unit's longer history, forecasts a much higher load; their unit
  log ratios are far larger than any other unit's and make up most of the week-1 unit mean. In week 2 Hog_lodging_Edgar
  and Eagle_office_Amie, whose readings are also constant over the week, play the same role. They are not the
  [three near-zero meters](#state-windows-and-the-three-near-zero-bdg2-meters), and none of these windows is a state
  window (the readings are above the micro-load threshold). This meter-level reading was checked on version 0.1's
  products; the values in `lead_weeks.csv` are unflagged.

### Block-wise weights against one whole-window weight

This section takes up one alternative explanation of the gain: that it comes from weighting the sources separately in
each block. ankyra-lite weights its two sources, the history side and the carrier, separately in the level (the carrier
candidate), the daily means (the weekly handover) and the within-day block (the trust). The first release's control
uses the same two forecasts with one weight per window: H, the history side (the fixed-division daily means plus the
unit's analog-day shape at the origin), and C, the carrier's forecast at the origin, are combined as (1 − a) H + a C
with one least-squares weight a for the whole 744-hour trajectory, estimated at the unit's completed pseudo-origins (the
daily part from the handover pairs, the within-day part from the trust triples, each divided by its number), clipped
to [0, 1] and shrunk towards ½ by n/(n + 2) like the handover weights, then projected; the seasonal naive applies on
state windows as in ankyra-lite. Rows `one whole-window combination weight of the history side and the carrier` and
`half-and-half average of the history side and the carrier` of `ablation.csv`, contrasts in
`component_contributions.csv` (ankyra-lite 0.2.0, with the continuation layer, against the two controls):

- **Hourly**, ankyra-lite is resolved better than the whole-window weight on the GoiEner households (+1.5%) and on no
  other population, and resolved worse on none; the other eight lie between −0.2% and +0.5%.
- **Monthly energy**: none of the nine contrasts is resolved (−3.9% on BDG2 to +3.0% on EWELD).
- **A fixed half-and-half average** of H and C is weaker: ankyra-lite is resolved better than it on six populations
  hourly (the households +4.0%, Cambridge +3.9%, HEEW +2.9%, EWELD +3.1%, Oslo +2.5%, Drammen +1.5%) and worse on none;
  in monthly energy none is resolved. The mean weight on the carrier is 0.56–0.73.

Read plainly: on these populations splitting the weights by block adds little measurable accuracy over one per-unit
whole-window weight between the same two forecasts, in line with the handover result above (weekly weights add nothing
over one weight per window); the gain over a fixed average comes from weighting by the unit's own pseudo-forecast
errors. The block decomposition remains the exact identity used to attribute the error, to define the readouts and to
diagnose failures (Figure [14](../figures/fig14_block_attribution.png), [METHOD.md](METHOD.md#three-orthogonal-blocks));
on this evidence it is not a source of accuracy by itself. The control is a special case of the same procedure and does
not test other ways of weighting by block. These are re-evaluations; nothing in the forecaster was changed.

### Within-day anchoring: where it acts

`within_day_trust.csv`: on the windows the anchoring layer anchors (all but the state windows), the trust ω on the
unit's analog-day shape ([METHOD.md](METHOD.md#within-day-shape)), the share of windows with ω > 0 and the share on which
the plausibility guard kept the analog shape. Every anchored test window had all three completed pseudo-origin
triples. The last column is the within-day row of [What the parts add](#what-the-parts-add).

| Population | Anchored windows | Mean ω | Median ω | ω > 0 | Analog shape kept | Within-day anchoring, hourly |
|---|---:|---:|---:|---:|---:|---|
| BDG2 2017 | 441 | 0.20 | 0.19 | 85% | 98% | +0.7 * |
| GoiEner non-household | 877 | 0.17 | 0.11 | 76% | 98% | +2.4 |
| GoiEner households | 661 | 0.05 | 0.00 | 43% | 99% | −0.0 |
| Cambridge | 730 | 0.15 | 0.13 | 82% | 99.9% | +0.3 |
| HEEW Arizona | 658 | 0.17 | 0.16 | 85% | 100% | +0.8 * |
| EWELD | 278 | 0.14 | 0.11 | 76% | 94% | −0.0 |
| Oslo | 581 | 0.13 | 0.11 | 76% | 100% | +0.6 * |
| Drammen | 689 | 0.21 | 0.21 | 88% | 100% | +0.9 * |
| CINELDI | 475 | 0.19 | 0.17 | 87% | 100% | +0.9 * |

The mean trust is largest on Drammen and BDG2 (0.20–0.21) and stays well below its cap of ½ everywhere; on the
households it is zero on most windows (median 0) and the forecast barely changes; on EWELD the guard rejects the analog
shape most often (on 6% of the anchored windows). Descriptive.

### Generic combination partners

This analysis asks the question of contrast (ii) from the history side: does the history side supply more than a
generic local forecast would, as the carrier's combination partner? `generic_combinations.csv` (the first release's
analysis with ankyra-lite's carrier): the whole-window combination of the previous section with the history side H
replaced by a generic local forecast X (the weekly seasonal naive, the 4-week profile or the last-year profile, each
computed from the load before every pseudo-origin); and the equal-weight average ½ (carrier + X), with X also the
per-unit ridge or MSTL (the saved baselines). File labels: `B2(H)` the whole-window combination with the history side,
`B2(X)` the same with X, `EW(X)` the equal-weight average. Nine populations, test windows; populations on which the
first-named is resolved better / not resolved / resolved worse:

| Contrast | Hourly | Monthly energy |
|---|---|---|
| B2(history side) vs B2(seasonal naive, week) | 1 / 8 / 0 | 0 / 9 / 0 |
| B2(history side) vs B2(four-week profile) | 0 / 9 / 0 | 0 / 9 / 0 |
| B2(history side) vs B2(last-year profile) | 1 / 8 / 0 | 0 / 9 / 0 |
| B2(history side) vs the best B2(X) | 0 / 9 / 0 | 0 / 9 / 0 |
| B2(history side) vs the best equal-weight average | 2 / 7 / 0 | 0 / 8 / 1 |
| ankyra-lite vs the best of all of them | 0 / 9 / 0 | 0 / 9 / 0 |

"Best" was picked after scoring, per population and error, which favours the generic side; the best equal-weight
partner is the per-unit ridge in 15 of the 18 cells, and the one resolved-worse energy row is Drammen, the
whole-window combination with H against the equal-weight average with the per-unit ridge (−13.4%). The history side is
resolved better as a partner than the weekly seasonal naive on Oslo and than the last-year profile on EWELD (hourly),
and never resolved worse. With this carrier, the history side was not separated from a generic local forecast as a
combination partner on most populations, so no advantage over these generic partners is established here.

### Seeds

`seed_spread.csv`: the hourly contrast against TimesFM of the anchored forecast (with the continuation layer) of each
group of three seeds (seeds 0–2, seeds 3–5) and of the six-seed ensemble; the file also holds each seed's carrier
alone:

| Population | Seeds 0–2 | Seeds 3–5 | Six seeds |
|---|---|---|---|
| BDG2 2017 | +0.0 | +0.4 | +0.7 |
| GoiEner non-household | +4.3 | +4.6 | +4.7 |
| GoiEner households | −4.9 (−) | −4.7 (−) | −4.6 (−) |
| Cambridge | +14.7 * | +15.6 * | +15.4 * |
| HEEW Arizona | +2.5 | +1.8 | +2.5 |
| EWELD | +1.5 | +1.2 | +1.5 |
| Oslo | +23.2 * | +22.3 * | +23.0 * |
| Drammen | +20.2 * | +20.8 * | +20.9 * |
| CINELDI | +10.1 * | +10.5 * | +10.5 * |

The two groups differ by at most 0.97 points; the verdict is the same for both groups and the ensemble on every
population. Besides the seed layer of the 0.2.0 rows of `recipe_history.csv`, the spread across seeds is the only
measure of retraining noise in `results/`; the intervals of the main tables do not include it.

### Other carriers in the same anchoring layer

`carriers_in_anchoring_layer.csv`: the same anchoring layer with three other trained carriers, trained on their own
population only with the same training cut as the small carrier (on state windows the layer returns their own
forecasts; none gets the continuation layer). Populations, of nine, resolved better / not resolved / resolved worse;
the small carrier's contrasts (i) and (ii) are before the continuation layer, and the last two columns compare the
delivered ankyra-lite (hourly) and its energy readout:

| Carrier | (i) anchored vs carrier alone: hourly | energy | (ii) anchored vs history only: hourly | energy | ankyra-lite vs this anchored carrier: hourly | energy |
|---|---|---|---|---|---|---|
| small carrier network (ankyra-lite) | 2 / 7 / 0 | 0 / 9 / 0 | 9 / 0 / 0 | 5 / 4 / 0 | — | — |
| LSTM | 8 / 1 / 0 | 6 / 3 / 0 | 3 / 5 / 1 | 0 / 7 / 2 | 9 / 0 / 0 | 6 / 3 / 0 |
| PatchTST | 7 / 2 / 0 | 3 / 6 / 0 | 7 / 2 / 0 | 3 / 6 / 0 | 7 / 2 / 0 | 1 / 8 / 0 |
| LSTM-T-KAN-M | 8 / 1 / 0 | 7 / 2 / 0 | 1 / 6 / 2 | 0 / 5 / 4 | 9 / 0 / 0 | 6 / 3 / 0 |

The anchoring layer improves the weaker carriers more (contrast (i)); with the small carrier the combination improves
on the history-only configuration on all nine populations in hourly error, with the LSTM and LSTM-T-KAN-M on three and
one. The resolved losses of contrast (ii) include EWELD for both (hourly −83.1% and −104.4%), where the layer returns
those carriers' forecasts on its 245 state windows, and Cambridge for LSTM-T-KAN-M (−10.3%). No anchored carrier is
resolved better than ankyra-lite on any population. As in [What the parts add](#what-the-parts-add), the two contrasts
measure incremental predictive value and are not formal encompassing tests: a carrier whose contrast (ii) is not
resolved is not shown to be encompassed by the history side, and the layer's shrunk weights on it are not test
statistics ([references](#references)).

### Contrast (ii) by block and the carrier-free control

`contrast_ii_blocks.csv`, `contrast_ii_control.csv` (descriptive): populations, of nine, on which the first-named
forecast is resolved better / not resolved / resolved worse, all test windows ("anchored": ankyra-lite before the
continuation layer):

| Contrast | Hourly | Level | Daily path | Within-day |
|---|---|---|---|---|
| anchored vs history only | 9 / 0 / 0 | 5 / 4 / 0 | 6 / 3 / 0 | 8 / 1 / 0 |
| anchored with the continuation layer vs history only | 9 / 0 / 0 | 5 / 4 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| anchored vs carrier alone | 2 / 7 / 0 | 0 / 9 / 0 | 1 / 6 / 2 | 6 / 3 / 0 |
| carrier alone vs history only | 7 / 2 / 0 | 3 / 6 / 0 | 5 / 3 / 1 | 8 / 1 / 0 |
| anchored vs carrier-free control | 9 / 0 / 0 | 5 / 4 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| carrier-free control vs history only | 8 / 1 / 0 | 0 / 9 / 0 | 1 / 8 / 0 | 8 / 1 / 0 |

Against the carrier-free control (the same procedure with the history-only forecasts in the carrier's place),
ankyra-lite is resolved better in hourly error on all nine populations (+3.8% on CINELDI to +15.6% on Oslo). The
carrier's forecast enters the level candidates, the handover and the within-day mixture at once, so the split by
block locates the difference only roughly. These are readings of incremental predictive value, block by block, not
formal encompassing tests: a block in which a contrast is not resolved is not evidence that one forecast encompasses
the other in that block ([What the parts add](#what-the-parts-add), [references](#references)).

### History length and the anchoring record

- `history_length.csv`, all nine populations pooled (descriptive; the strata are confounded with the population):

  | Hours since the unit's first observation, at the origin | Windows | vs carrier alone | vs TimesFM | vs Chronos-2-X |
  |---|---:|---|---|---|
  | < 10,248 h (no annual candidate) | 450 | −0.6 | −2.4 | −2.6 |
  | 10,248 h – 2 years | 1,715 | +1.7 | +0.4 | −2.7 |
  | 2–3 years | 829 | +2.5 | +11.8 | +7.2 |
  | ≥ 3 years | 2,722 | +0.3 | +9.1 * | +5.8 |

  Against the zero-shot foundation models the point estimates are negative below 10,248 hours, mixed up to two years
  and positive above; only the ≥ 3-year stratum against TimesFM is resolved. No test window has fewer than 9 complete
  pseudo-origin windows (`K_eff` 9–11: 451 windows; `K_eff` 12: 5,265).
- `anchoring_gain_deciles.csv` (deciles of 538–539 windows each over the nine populations: the windows with a finite
  statistic, units with a mean load below 10⁻⁶ kW left out; descriptive): ankyra-lite's monthly-level gain over its
  carrier alone is largest in the lowest decile of the carrier's level weight (+8.8%) and negative in the seventh and
  eighth (−7.2%, −3.7%); by `history_vs_carrier_log_ratio` it is largest in the two lowest deciles, where the unit's
  history beat the carrier at earlier pseudo-origins (+10.5%, +9.4%; the hourly gain is resolved there, +3.1% and
  +2.4%). Neither statistic orders the gain monotonically, and the carrier weight is the layer's own weight, so its
  association with the gain is partly mechanical.
- `level_candidate_hit_rate.csv` (descriptive; P20 in [theory/PROOFS.md](../theory/PROOFS.md#evidence-bound)): how
  often the level candidate with the largest weight, the one with the smallest pseudo-origin errors, is the candidate
  whose window mean at the origin is closest to the realised window mean. Pooled over the nine populations (5,367
  anchored windows with a unique largest weight), the hit rate is 0.217 for the seven candidates against a chance rate
  of 0.144 (difference +0.073, 95% interval +0.061 to +0.084, resampling units), and 0.233 against 0.169 for the six
  historical candidates alone. The difference excludes zero on seven populations for the seven candidates (not EWELD
  and CINELDI) and on eight for the six (not CINELDI). The carrier's window mean has the largest weight on 40–71% of the
  windows of a population and is the closest candidate on 15–29%. The pseudo-origin errors therefore carry real but
  weak information about which candidate will be best, which is the case for strongly shrunk weights that P20
  describes.

### Robustness and cost

- `robustness.csv`: 64 Drammen test windows (43 units) re-forecast on the CPU with the released checkpoints from a
  corrupted context (carrier, anchoring layer, state rule, continuation layer). Removing 5% or 20% of the context hours
  (filled by interpolation) changes the hourly error by at most 0.2%; a 24-hour zero block in the last week raises it by
  2.3% (carrier alone 2.4%); a shift of the context by +1 / −1 hour by 10.8% / 13.3% (carrier alone 10.8% / 13.9%); one
  spike at ten times the context maximum by 52.5% (carrier alone 40.7%), and the spike moves the peak readout by a
  median of +997%. Scaling the whole history by 0.5 or 2 leaves the forecast exactly scale-equivariant (largest
  deviation 0). On every corrupted context except the spike, ankyra-lite's hourly error is 2.9–3.4% below that of the
  carrier alone on the same record; with the spike it is 5.2% above. No state rule fired on any corrupted context.
- `cost_per_window.csv`: 0.655 s per forecast on one CPU thread (6 Oslo test windows), of which 0.135 s are the carrier
  forecasts at 7 issue times × 6 seeds and 0.520 s the anchoring layer with the state rule and the continuation layer
  ([METHOD.md](METHOD.md#inference-cost)).

### Where ankyra-lite is weak

- **Households, hourly.** On the GoiEner households TimesFM has the resolvably lower hourly error (−4.6%), and the
  other three zero-shot variants lower point estimates (−0.9% to −2.9%, not resolved); the anchoring layer does not
  improve the carrier there (−1.7% before the continuation layer, −1.3% with it, not resolved), and ankyra-lite ranks
  fourth by MAE and fifth by WAPE. Its monthly energy and peak errors there are resolvably lower than those of all four
  zero-shot variants.
- **The first days of the month.** On day 1 it ranks 2nd to 6th of 22 on the six test populations, behind zero-shot
  foundation models on all of them; the continuation layer narrows the gap without closing it.
- **Short records and the zero-shot foundation models.** On BDG2 and GoiEner non-household, which had no development
  windows and the shortest training records, and on HEEW and EWELD, the hourly contrasts with the four zero-shot
  variants are not resolved, with negative point estimates on BDG2 (Chronos-2-X −4.0%), GoiEner non-household
  (Chronos-2-X −0.5%) and EWELD (Chronos-2 −0.2%). The monthly energy against TimesFM is −0.1% on HEEW and −2.8% on
  EWELD (not resolved), and on EWELD the cumulative energy at day 31 ranks sixth.
- **The recipe.** Recipe R's span return, kept in recipe Rw, is 2.90% worse than the plain continuation on Drammen and
  resolvably worse on forecast days 2–7 of the six populations with development windows (−1.35%)
  ([above](#why-the-recipe-returns-the-validation-span)).
- **The continuation layer on Oslo** is resolvably negative on days 8–31 (−0.5%) ([above](#the-continuation-layer)).
- **Peak against the last observed peak.** On CINELDI the previous window's observed peak is resolvably closer to the
  realised peak (−18.2%); on GoiEner non-household and EWELD the point estimates are −34.7% and −23.7%.
- **A spike in the context** raises the hourly error by 52.5% and ruins the peak readout ([above](#robustness-and-cost)).
- **Zero-shot foundation models with a year of context.** TimesFM 2.5 with 8,736 hours of context and Chronos-2 with
  8,192 hours are resolvably better in hourly error on the GoiEner households (−5.3% and −5.2%)
  ([above](#against-the-zero-shot-foundation-models)).
- **Peak against the carrier's own readout.** On Oslo and Drammen the same peak readout on the carrier alone is
  resolvably closer to the realised peak (−2.5% and −2.9%; [peak readout](../README.md#peak-readout)).
- **The history side as a combination partner** is not separated from generic local forecasts on most populations,
  and on Drammen the best equal-weight average of the carrier with the per-unit ridge (picked after scoring) has a
  resolvably lower monthly energy error than the whole-window combination with the history side (−13.4%;
  [above](#generic-combination-partners)).

## External tests

**Version 0.2.0: re-run once (20 October 2026).** The 0.2.0 configuration was re-run on the four scored test sets of
the frozen external test of 0.1 (the same windows, comparators and estimators, read there once with the frozen
candidate, recipe R with six seeds) and scored once. For each test set the stage-1 carriers of that test (six seeds)
were continued with recipe Rw on the test set's own training pool, averaged, anchored by the unchanged layer with the
seasonal-naive state rule, corrected by the continuation layer with its frozen constants (no refit), read out by rule A
and scored with the frozen estimators; the shipped checkpoints were not used. This is a second read of each test set,
the first of this version. Full report: [docs/EXTERNAL_TESTS.md](EXTERNAL_TESTS.md); every number is a row of
[`results/external_tests.csv`](../results/external_tests.csv), copied from the score records by
`evaluation/external_tables.py`.

- **LCL, UNICON and HKUST** are previously read populations: earlier forecasters, earlier versions of the small network
  among them, had been scored there, the anchoring layer was fixed after those readings, and the frozen external test
  of 0.1 read them once. HKUST is a secondary test set with a small sample.
- **GoiEner households, out of time: a same-source out-of-time supplementary check** (new households and a later period
  of study population 6's source), not a new population.
- **Helsinki (retrospective): trained, anchored and sealed, not scored**, and not touched by the re-run.
  **UCI370: skipped.**

Primary readouts, hourly error, six seeds (\* resolved better; no pass line). They are 12 of 15 pre-specified nominal
95% intervals (3 contrasts × 5 test sets, Helsinki's sealed), with no multiplicity correction; with no true difference,
about 0.75 of the 15 would be resolved by chance. LCL is in the pretraining data of TimesFM 2.5, Chronos and Chronos-2;
whether GoiEner is in it is not verified; UNICON and HKUST are not on the published lists, which does not establish
absence. The last column gives the 0.1 frozen candidate's estimate against TimesFM 2.5 on the same windows (history).

| Test set | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule | 0.1 candidate vs TimesFM 2.5 |
|---|---|---|---|---|---|
| LCL households (London) | 710 / 710 / 5 | +2.74% [−1.16, +6.48] | +24.37% [+17.71, +30.48] * | +3.13% [+2.06, +4.19] * | +2.68% |
| UNICON (La Trobe University campuses) | 450 / 59 / 23 | −2.42% [−14.33, +8.24] | +11.83% [+0.73, +21.68] * | +10.71% [+5.51, +15.62] * | −5.54% |
| HKUST campus incomer meters (secondary) | 89 / 29 / 8 | +10.88% [−9.01, +27.14] | +26.18% [+6.34, +41.81] * | +4.63% [−2.96, +11.67] | +10.24% |
| GoiEner households, out of time (supplementary) | 1,417 / 282 / 6 | +4.25% [+1.88, +6.57] * | +26.04% [+23.82, +28.20] * | +5.55% [+3.49, +7.57] * | +4.19% * |

- Over the four scored test sets, 8 of the 12 primary intervals are resolved in ankyra-lite's favour (7 for the 0.1
  candidate; UNICON against the weekly seasonal naive is newly resolved), 4 are not resolved and none against it.
- **Against zero-shot TimesFM 2.5** only the GoiEner check is resolved (+4.25%); on LCL (+2.74%; LCL is in TimesFM 2.5's
  pretraining data), UNICON (−2.42%) and HKUST (+10.88%) the intervals include zero; not resolved is not a finding of
  no difference.
- **Against the 0.1 frozen candidate** (recipe R, six seeds, no layer) 0.2.0 is resolved better over the month on UNICON
  (+2.95%) and not resolved on the other three (+0.06% to +0.72%); on day 1 it is resolved better on LCL (+4.89%) and
  UNICON (+16.67%).
- **Forecast day 1.** Zero-shot TimesFM 2.5 still has resolvably lower day-1 error on LCL (−10.43%; it was −16.63% for
  the 0.1 configuration); on UNICON (−13.29%, before −37.30%), HKUST (−1.43%) and GoiEner (−1.99%) the day-1 contrast is
  not resolved. In forecast hours 0–2 TimesFM 2.5 is resolvably ahead on LCL (−25.38%), UNICON (−73.88%) and GoiEner
  (−18.26%). On UNICON ankyra-lite's mean unit rank is third of 14.
- **Day-1 rule, written before the read.** If the external test sets' mean day-1 log gap to TimesFM 2.5 shrank, from
  the 0.1 configuration to 0.2.0, by less than a third of the shrinkage on the development windows, the README would
  make no claim that 0.2.0 narrows the short-term gap. Outcome: claim allowed (external mean shrinkage +0.085 in log
  units, line +0.036, development shrinkage +0.107; per test set LCL +0.055, UNICON +0.192, HKUST +0.088, GoiEner
  +0.004; `external_tests.csv`, blocks `day-1 rule (design 3.3)` and `day-1 gap to TimesFM 2.5`). The claim is only
  that the day-1 gap to TimesFM 2.5 narrowed; ankyra-lite does not lead TimesFM 2.5 on day 1.

![External tests](../figures/fig17_external_tests.png)

*Figure 17. External tests of ankyra-lite 0.2.0, the 0.2.0 configuration re-run on the test sets of the frozen external
test and scored once: the three primary readouts (a–c), the same contrasts by lead time, forecast hours 0–2 included
(d–f), and descriptive contrasts between ankyra-lite's own configurations by lead time (g: 0.2.0 against the 0.1
configuration; h: the continuation layer on against off; i: the loss weighting of recipe Rw against recipe R, without
the layer). Filled: the 95% interval excludes zero; an interval end beyond the axis is drawn as an arrow. LCL, UNICON,
HKUST: previously read populations; GoiEner households out of time: same-source out-of-time supplementary check;
Helsinki: sealed, not scored.*

**Version 0.1 (19 October 2026), history.** The frozen external test of 0.1 trained recipe R carriers (six seeds,
the three-seed release configuration as a secondary readout) and scored each test set once; its numbers are in the
last column above and in [docs/EXTERNAL_TESTS.md](EXTERNAL_TESTS.md#version-01-external-test-19-october-2026). It
had 7 of 12 primary intervals resolved in ankyra-lite's favour and none against it.

The subgroups, the other readouts, the decisions awaiting the author's review and the run notes are in
[EXTERNAL_TESTS.md](EXTERNAL_TESTS.md).

## Limitations

- **Re-evaluation only on the nine populations.** No test set of the nine-population comparison is unseen by the
  design ([above](#identity-of-the-evidence)); the external re-run reads previously read populations a second time and
  one same-source out-of-time check, not unseen populations ([External tests](#external-tests)).
- **The continuation layer** was chosen on development windows, partly after reading them; its constants were fitted
  on six populations and applied unchanged to the three without development windows; its reference-day fallback
  (d* = 6, used on 33 EWELD test windows) and its negative-load rule are release rules. Energy, level and peak are read
  before it, so the delivered curve's mean differs from the energy readout (median 0.77%, at most 310%;
  [METHOD.md](METHOD.md#readouts)).
- **The loss weights' effect is small** (+0.31% over the month, of the order of the seed spread; on the development
  windows the peak readout was resolvably worse, −0.92%).
- **One carrier per population.** Transfer of a checkpoint to another building or population has not been evaluated.
- **Short records.** BDG2 2017 and both GoiEner populations have the shortest training records, no development
  windows, and test sets of 5–8 months of one year; read their results as the second half of that year.
- **Recipe.** Against the continuation without the returned span, the released recipe gains on the windows that lacked
  same-season training data (+3.11%, resolved) and loses on Drammen (−2.90%) and on forecast days 2–7 of the
  populations with development windows (−1.35%, resolved); over the nine populations its advantage over that control
  is not resolved (+0.11%) (`recipe_history.csv`).
- **Recency.** The carrier's newest training label is at least 186 days older than the trained baselines' allow; this is
  the price of out-of-sample pseudo-origins.
- **Calendar.** 36 test windows (26 EWELD, 10 Cambridge) contain a holiday announced after their origin; the forecasts
  were not remade without it.
- **Pretraining overlap.** Seven BDG2 sites are in TimesFM 2.5's pretraining collection; its effect on the BDG2
  comparisons is not measured.
- **Baselines.** The release's trained baselines have a budget of 15 epochs and patience 3, LSTM-T-KAN-M 30
  and 8. Under the first release's relaxed budget (30 epochs, patience 8) for the four load-only trained families, 35
  of 36 hourly verdicts are unchanged and 3 of 36 energy verdicts change to not resolved; only that one relaxation was
  tried, and the covariate-informed trained baselines were not retrained
  ([above](#training-budget-of-the-trained-baselines)). The linear-context model's covariates are inert.
  GBT-T overshoots on EWELD. LSTM-T-KAN-M is not a strict reproduction of its paper (ten disclosed deviations and a smaller
  training budget than the authors'; [above](#lstm-t-kan-m)); on CINELDI it is the retrain on the corrected store,
  which passed its checks.
- **Intervals** are descriptive and were not designed for this carrier; they are placed after the continuation layer.
- **Training is not bitwise repeatable** on a GPU; six seeds are averaged, and the spread of the two three-seed groups is reported
  (`seed_spread.csv`), but the intervals of the main tables resample units and months only, not retraining.
- **Multiplicity.** 21 baselines × 9 populations × several quantities give hundreds of intervals; counts of resolved
  contrasts are descriptive summaries, not a multiplicity-adjusted test.

## Scoring-code check and reproduction

- **Scoring-code check.** Before ankyra-lite is put in the reference position, `evaluation/compare_baselines.py`
  recomputes the late-window rows of ten result tables of the first release with the release's own reference
  forecaster and compares them field by field, at the published rounding, with the published files read from the first
  release repository's commit `487e0d2` (`git show`, read-only). A table that does not reproduce exactly is not written.
  The recomputed release numbers are not part of any result file here. In the regeneration all ten tables reproduced,
  with 0 mismatches in 94,131 compared fields (all exact). Record:
  [`results/REPRODUCTION_CHECK.json`](../results/README.md#reproduction-and-records).
- **Same windows.** For every population the anchored product must have exactly the study's test windows, state flags
  and targets; otherwise the script stops.
- **Product reading.** In version 0.1 the per-population contrasts of recipe R against the plain continuation and
  against the from-scratch stage, recomputed from the anchored products, equalled those of the study's own scorer on 9
  of 9 populations (`study_scorer_crosscheck`). In 0.2.0 the re-anchoring of the six-seed ensemble reproduces every
  stored product bit for bit, before and after the continuation layer (`extra` → `reanchor` → `ens` in
  `results/compare_baselines.json`).
- **Continuation check.** Every continued product is checked, with no target read, to have the windows, state flags,
  level and peak of the product before the layer, untouched state windows, and every other trajectory equal to the
  release's layer applied to the stored float32 trajectory, bit for bit (`continuation_gate`; passed on all nine
  populations); `evaluation/compare_baselines.py --check-products` runs the same check.
- **Scoring module.** `evaluation/scoring.py`, a clean re-implementation of the study's scoring on top of
  `ankyra_lite.anchor.metrics`, gave identical values to the study's scoring functions on BDG2 and Oslo
  ([evaluation/README.md](../evaluation/README.md#what-was-checked)).
- **Package against the research code.** The anchoring layer is byte-identical to ANKYRA 2.2.1 and gives bit-identical
  forecasts (`tests/test_anchor_equivalence.py`); the carrier's features, initialisation and forecasts and the training
  recipe reproduce the research code (`tests/test_carrier_equivalence.py`, `tests/test_train.py`).
- **Which forecasts are scored.** `evaluation/products.json` names, per population, the 0.2.0 configuration, the
  anchored six-seed product before and after the continuation layer, the two three-seed group products and the research
  runs behind them (paths relative to the study's research folder). `evaluation/products_0_1.json` is the mapping of
  version 0.1 (`--recipe r010`).

### Numbers that are not in `results/`

Every number of the evaluation comes from a file of `results/`, with these exceptions, which come from the study's own
records (in its research folder, not part of this repository), from the data or from public lists:

| Number | Where it appears | Source |
|---|---|---|
| 8,486 compared values of the scoring-module check (7,528 on BDG2, 958 on Oslo) | [evaluation/README.md](../evaluation/README.md#what-was-checked) | the output of that one-off check, run on 18 October 2026; it was not repeated in the regeneration |
| 26 of the 523 EWELD and 10 of the 730 Cambridge test windows with a holiday announced after the origin | [Limitations](#limitations), [DATA.md](DATA.md#calendar-caveat-holidays-announced-after-the-origin) | the study's calendar check (`paper_writing/EO_CALENDAR_AND_CINELDI_START_FINDINGS_v1_20261018.md`); the totals 523 and 730 are the test windows of `datasets.csv` |
| 13 BDG2 units with the category Commercial | [DATA.md](DATA.md#1-bdg2-2017-building-meters-usa--europe) | the category labels (`groups`) of the BDG2 data store |
| The UNICON store's meters on three of the five campuses (Bundoora, Albury-Wodonga, Bendigo); the HKUST and UNICON store temperatures equal to the study's ERA5 archive files (Open-Meteo) hour for hour | [EXTERNAL_TESTS.md](EXTERNAL_TESTS.md#unicon-la-trobe-university-campuses) | a read-only check of the study's prepared stores (`campus`, `temp`) against the archive files, run on 19 October 2026 |
| The BDG2 meters behind the week-1 and week-2 contrasts with TimesFM (three in week 1 with one window each at the origin of 1 August 2017, constant through the context and the first eight forecast days; two in week 2) | [What the handover contributes](#what-the-handover-contributes), [results/README.md](../results/README.md#handover-within-day-trust-and-combinations) | a read-only check of the saved forecasts (the anchored product and the panel's TimesFM forecasts) and the BDG2 data store, run on 19 October 2026; it reproduces the week-1 and week-2 log ratios of `lead_weeks.csv` exactly |
| 709,481 parameters of LSTM-T-KAN-M; best epochs 12, 10 and 16 of its BDG2 seeds | [LSTM-T-KAN-M](#lstm-t-kan-m) | the study's LSTM-T-KAN training record (`paper_writing/EO_LSTM_TKAN_BASELINE_RESULTS_v2_20261011.md`) |
| Seven BDG2 sites (Bear, Bull, Cockatoo, Fox, Hog, Panther, Rat) listed in the GIFT-Eval pretraining collection of TimesFM 2.5, none named in the Chronos or Chronos-2 training tables | [Limitations](#limitations), [DATA.md](DATA.md#bdg2-and-the-pretraining-collection-of-timesfm-25), the README (results and limitations) and `results/SUMMARY.md` | the study's check of the public lists on 10 October 2026 (the TimesFM 2.5 model card, the file lists of the GIFT-Eval pretraining collection and of LOTSA, the Chronos and Chronos-2 training tables), recorded in `paper_writing/EO_V47_CHANGE_LIST_v1_20261010.md`; it is a check of names in lists, not a measurement of the data |

Run the tests from the repository root with `python -m unittest discover -s tests -t .` and the theory checks with
`python -m unittest discover -s theory -t .`; the tests that need the first release's checkout, the 0.1.1 checkout or
the study's local data skip with a reason when those are absent. Version 0.2.0 has 267 tests and 109 theory checks; the
full run passed on 21 October 2026 (CPU, one thread): 267 tests with 15 skipped (2 result-file checks without a matching
record in this run, 4 comparisons with version 0.1 that passed when run separately against the 0.1.1 commit, 9
store-array checks skipped for free memory, of which 7 passed when run separately) and the 109 theory checks.

| Module | Tests |
|---|---:|
| `tests/anchor/` (the anchoring layer's own tests) | 62 |
| `tests/test_anchor_equivalence.py` | 14 |
| `tests/test_carrier_unit.py` | 14 |
| `tests/test_readouts_lite.py` (the readouts of `ankyra_lite.readouts`, the history-only call, the fields read through `ForecastResult`, the examples) | 13 |
| `tests/test_carrier_equivalence.py` (study's local data) | 7 |
| `tests/test_train.py` (part on the study's local data) | 19 |
| `tests/test_train_rw.py` (recipe Rw; part on the study's research folder) | 19 |
| `tests/test_continuation.py` (the continuation layer; part on the study's research folder) | 24 |
| `tests/test_forecast_v2.py` (the forecast API against 0.1.1; part on the 0.1.1 checkout) | 17 |
| `tests/test_store_arrays.py` (study's local data) | 9 |
| `tests/test_compare_baselines.py` (part on the first release's files, the study's data and `results/`) | 42 |
| `tests/test_evaluation.py` | 15 |
| `tests/test_external_tables.py` (the external-test table) | 12 |
| `theory/test_operators.py`, `theory/test_carrier_properties.py`, `theory/test_weighting_properties.py`, `theory/test_continuation_properties.py` | 31, 27, 15, 36 |

## References

For the two contrasts read under [What the parts add](#what-the-parts-add),
[Other carriers](#other-carriers-in-the-same-anchoring-layer) and
[Contrast (ii) by block](#contrast-ii-by-block-and-the-carrier-free-control) (forecast encompassing):

- Chong, Y. Y., Hendry, D. F. (1986). Econometric evaluation of linear macro-economic models. *Review of Economic
  Studies* 53(4), 671–690.
- Harvey, D. I., Leybourne, S. J., Newbold, P. (1998). Tests for forecast encompassing. *Journal of Business &
  Economic Statistics* 16(2), 254–259.
