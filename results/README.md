# Results data

The files in this directory are the scored statistics of **ankyra-lite 0.1.0**, the released recipe (the study's span
return: training from scratch with early stopping on the 93-day validation set, then the validation span returned to
the training set for 8 more epochs; three seeds averaged; [METHOD.md](../docs/METHOD.md#training)). They cover the test
windows of nine populations (the late windows of the first ANKYRA release) and compare ankyra-lite with 21 baselines.
Every file is written by [`evaluation/compare_baselines.py`](../evaluation/compare_baselines.py) (with
`extra_tables.py`, `lite_reanchor.py`, `recipe_tables.py`, `robustness.py` and `cost_timing.py`), from the anchored
forecasts listed in [`evaluation/products.json`](../evaluation/products.json); [`SUMMARY.md`](SUMMARY.md) is the
readable summary of the same run. How the numbers were obtained is in [EVALUATION.md](../docs/EVALUATION.md). The one
exception is `external_tests.csv`, the external tests, written by `evaluation/external_tables.py` from the external
run's score records ([below](#external-tests)).

Read first:

- **Every number of the nine-population files is a re-evaluation of test windows that had been read before**, and parts
  of the design, the recipe included, were chosen after looking at them
  ([EVALUATION.md](../docs/EVALUATION.md#identity-of-the-evidence)). Nothing was tuned on the exported statistics
  themselves. `external_tests.csv` has its own identity, given in its column `identity` for every row
  ([External tests](#external-tests)).
- **The reference is ankyra-lite.** On its rows `model` is `ankyra-lite`; `log_ratio` compares ankyra-lite with the
  row's forecaster. Column names taken over from the first release that say `ankyra` (`median_peak_ape_ankyra_pct`,
  `wilcoxon_holm_p_vs_ankyra`, `ankyra_mean_unit_rank`, `ankyra_design_status`, `ankyra_kw`, and the `ankyra_*`
  columns of `robustness.csv`) refer to ankyra-lite.
- **No number of ANKYRA itself or of its versions is in any file** (`compare_baselines.json` records under
  `forecasters_left_out` that they were left out). The 21 baselines are the 20 external baselines of the first release and LSTM-T-KAN-M; all
  are ordinary baselines in every table and rank.
- **Descriptive** marks a file computed after scoring, from the scored forecasts, that tests nothing.
- The ten tables of the main comparison keep the columns of the first release's files of the same name
  (`Building_load_ANKYRA/results`), plus a trailing column `flag`; the other files are described below with their own
  columns, as the code writes them. All files are UTF-8 with LF line endings.

Contents: [Words used in every file](#words-used-in-every-file) · [Forecasters](#forecasters) ·
[Populations and splits](#populations-and-splits) · [The main comparison](#the-main-comparison) ·
[Error by forecast day and energy](#error-by-forecast-day-and-energy) · [Readouts and intervals](#readouts-and-intervals) ·
[Design choices](#design-choices) · [Contrast (ii) by block](#contrast-ii-by-block-and-a-carrier-free-control) ·
[Handover, within-day trust and combinations](#handover-within-day-trust-and-combinations) ·
[Training budget and long-context baselines](#training-budget-and-long-context-baselines) ·
[Training curves](#training-curves) · [Recipe history](#recipe-history) · [Constants](#constants) ·
[Robustness](#robustness) · [BDG2 near-zero meters](#bdg2-near-zero-meters) · [Cost](#cost) ·
[Example window](#example-window) · [Reproduction and records](#reproduction-and-records) ·
[External tests](#external-tests) ·
[Files of the first release without a counterpart](#files-of-the-first-release-without-a-counterpart)

## Words used in every file

| Column or word | Meaning |
|---|---|
| `set` | Population, named as in `datasets.csv` (`BDG2 2017`, `Cambridge`, `HEEW Arizona`, `EWELD`, `GoiEner non-household`, `GoiEner households`, `Oslo`, `Drammen`, `CINELDI`); pooled rows name their pool (`six test populations pooled`, `all nine populations`). |
| `tier` | `test` (the first release's six test populations) or `preview` (Oslo, Drammen, CINELDI); [EVALUATION.md](../docs/EVALUATION.md#populations-and-tiers). |
| window | One forecast: the 744 hours (31 days) after an origin, made from the history before it. |
| unit | One meter, building, supply point or household. |
| `subset` | `late` on every row: the test windows (origin at or after the training cutoff of the trained baselines), where all 22 forecasters exist. |
| `windows`, `units` | Number of windows and units in the row. |
| `model`, `model_label`, `model_class` | Key, readable name and class of a forecaster ([below](#forecasters)); `model_class` is blank on ankyra-lite's rows. |
| `log_ratio` | Unit-equal log RMS ratio: the mean over units of log(RMS error of ankyra-lite / RMS error of the other forecaster), each RMS over the unit's windows. Negative favours ankyra-lite. Files that compare another pair say so in their entry (then the first-named forecaster is the numerator). |
| `um_low`, `um_high` | 95% interval of `log_ratio`: 2,000 bootstrap draws resampling units and target months independently. A contrast is *resolved* when the interval excludes zero. |
| `improvement_pct` | 100[1 − exp(`log_ratio`)]. Positive favours ankyra-lite (or the first-named forecaster). |
| `pooled_mse_ratio` | Ratio of the two mean squared errors over all windows; large units weigh more, and it can differ in sign from `log_ratio`. |
| `resolved` | `first better` (interval below zero), `second better` (above zero), `not resolved` (includes zero; not the same as equally good), `identical forecasts`, or `undefined` (no finite log ratio). |
| `unit_set` | BDG2 only: `all units`, or `without the three near-zero meters` (three meters near zero in every window; [EVALUATION.md](../docs/EVALUATION.md#state-windows-and-the-three-near-zero-bdg2-meters)). |
| `flag` | A note that applies to every row of the population, or empty. It is empty on every row of every file of this run: Drammen and CINELDI are on the corrected CINELDI store, and LSTM-T-KAN-M on CINELDI is its retrain on that store, which passed its checks (`compare_baselines.json`, `populations` → `cineldi` → `extra_baselines`). The code would set it on CINELDI only if that forecast had to be taken from the original store. |
| level, daily path, within-day | The three blocks of a forecast: its monthly mean, its daily means minus that mean, and its hours minus their daily mean (P1 in [theory](../theory/PROOFS.md)). |
| pseudo-origin | An earlier origin o − 744k (k = 1 … 6) at which the carrier's forecast is scored against the unit's own realised month. |
| carrier | The small carrier network, whose forecasts the anchoring layer anchors; "the carrier alone" is its three-seed forecast at the origin, without the anchoring layer and without the state rule. |
| state window | A window on which the anchoring layer's off-state or micro-load rule fires; ankyra-lite delivers the weekly seasonal naive there. |

### Forecasters

| `model` | `model_label` | `model_class` |
|---|---|---|
| `ankyra-lite` | ankyra-lite | blank |
| `TiDE` | Linear-context (TiDE implementation) | `same information, trained` |
| `iTransformer-X` | iTransformer-X | `same information, trained` |
| `GBT-T` | GBT-T | `same information, trained` |
| `Chronos-2-X`, `TimesFM-X` | Chronos-2-X, TimesFM-X | `same information, zero-shot foundation model` |
| `TimesFM`, `Chronos-2` | TimesFM, Chronos-2 | `load only, zero-shot foundation model` |
| `DLinear`, `PatchTST`, `iTransformer`, `LSTM` | the same | `load only, trained` |
| `LSTM-T-KAN-M` | LSTM-T-KAN (authors' public code, exp2 feature list) | `load and calendar, trained` |
| `Holt-Winters`, `MSTL` | the same | `load only, statistical` |
| `RIDGE-L` | Ridge (per unit) | `per-unit regression` |
| `GBT` | GBT (zero-shot) | `zero-shot gradient boosting` |
| `SP-profile`, `LYR-profile`, `WeekProf4` | Previous-month profile, Last-year profile, 4-week profile | `profile` |
| `SN-day`, `SN-week` | Seasonal naive (day), Seasonal naive (week) | `naive` |

- `same information`: given the first release's shared covariate set (11 past, 10 future and 6 static features); each
  model uses the part its architecture accepts. `TiDE` is the first release's TiDE implementation, whose covariate path
  is inert: its forecast is a linear map of the load context
  ([EVALUATION.md](../docs/EVALUATION.md#the-linear-context-model-tide-implementation)).
- `load only`: the load context alone; `load and calendar`: channels derived from the past load and the past calendar.
- The trained baselines of the first release were trained once per population before its training cutoff.
  LSTM-T-KAN-M is the multivariate LSTM-T-KAN: model and training code copied line by line from its authors' public
  repository (commit `ce24938`) plus the authors' exp2 feature list, trained under the same data protocol. It is not a
  strict reproduction of the paper; its ten deviations and its training budget are listed in
  [EVALUATION.md](../docs/EVALUATION.md#lstm-t-kan-m). On CINELDI it was retrained on the corrected store with the same
  code, protocol, learning-rate grid and seeds, like the first release's calendar-reading trained baselines.

## Populations and splits

**`datasets.csv`** — the nine populations, their test windows and the training, validation and test split of each
([DATA.md](../docs/DATA.md#training-validation-and-test-sets)). One row per population.

- `set`, `tier`, `country`, `population` (kind of unit): as in the first release.
- `ankyra_design_status`: kept from the first release's column; `seen` on every row here: each population's test
  windows had been read when ankyra-lite was designed.
- `units`, `windows`: all windows of the first release's panel; `units_late`, `windows_late`: its late windows, which
  are ankyra-lite's test windows.
- `development_windows`: `yes` where the study had development windows to check the carrier's design (six
  populations), `no` for BDG2 2017 and both GoiEner populations.
- `store`: `original`, or `corrected` where the carrier was trained with the corrected CINELDI store (CINELDI itself, and
  Drammen, whose pooled training set holds CINELDI windows).
- `first_training_origin`: the earliest origin of the population's own windows in its pooled training set.
- `validation_start`, `training_cut`: the start v0 of the 93-day validation span and the training cut.
- `test_start`: the first day of the test set (186 days after the cut); `first_test_origin`, `last_test_origin`,
  `test_windows`, `test_units`: the test set.
- `record_days_before_cut`: days from the first training origin to the cut.
- `validation_share_pct`: share of the population's own eligible daily-grid windows (before the caps) that lie in the
  validation span: span candidates / (training candidates + span candidates), from the study's split record.
- `test_origin_months`: calendar months with at least one test origin; `test_months_spanned`: months from the first to
  the last test origin.
- `gap_days_cut_to_test_start`: 186 on every row.
- `calendar`: `true dates` (every date in the row is a true calendar date; CINELDI on the corrected store).
- `own_training_windows`: the population's own windows in its pooled training set of stage 1 (after the caps and the
  count matching; before the span return).
- `returned_validation_windows`: the number of the population's own validation-span windows returned to its training
  set in stage 2, n = round(own pool windows / own training candidates × span candidates)
  ([METHOD.md](../docs/METHOD.md#stage-2-the-validation-span-returned)).
- `returned_label_end`: the date on which the latest label of the returned windows ends (at least 24 hours before the
  training cut).
- `continuation_epochs`: the epochs of stage 2 (8 on every row).

## The main comparison

**`benchmark_pairwise.csv`** — ankyra-lite against each of the 21 baselines, hourly error, per population. Columns:
`set`, `tier`, `subset`, `windows`, `units`, `model`, `model_label`, `model_class`, `log_ratio`, `um_low`, `um_high`,
`improvement_pct`, `pooled_mse_ratio`, `flag`.

**`benchmark_mean_unit_rank.csv`** — mean per-unit rank of the 22 forecasters. Within each unit the forecasters are
ranked by RMSE, each RMSE rounded to 12 decimals and ties averaged; `mean_unit_rank` is the mean over units, `n_models`
the number of forecasters ranked (22). Columns: `set`, `tier`, `subset`, `model`, `model_label`, `mean_unit_rank`,
`n_models`, `flag`.

**`rank_tests.csv`** — rank tests on the per-unit RMSE of the 22 forecasters, per population and for the six test
populations pooled (units concatenated). Descriptive. Columns: `set`, `tier`, `units` (the blocks of the test),
`model`, `model_label`, `mean_rank`, `friedman_p` (Friedman's test of equal ranks; `<1e-300` when below the smallest
double), `nemenyi_cd` (Nemenyi critical difference of mean ranks at 5%), `wilcoxon_holm_p_vs_ankyra` (Holm-corrected
p-value of the paired Wilcoxon signed-rank test of ankyra-lite against the forecaster; blank on ankyra-lite's row),
`flag`. Rows are sorted by `mean_rank`. Unlike `benchmark_mean_unit_rank.csv` this file ranks the unrounded RMSE values,
so the two can differ where errors are equal after rounding but not exactly (the near-zero BDG2 meters).

**`conventional_metrics_late.csv`** — conventional errors of the 22 forecasters. Columns: `set`, `tier`, `windows`,
`units`, `model`, `model_label`, `model_class`, `RMSE_kW` and `MAE_kW` (mean over units, kW), `CV_RMSE_pct`, `NMBE_pct`
and `WAPE_pct` (median over units, %), `WAPE_pooled_pct` (pooled over windows, %), `units_excluded` (units with a mean
load below 10⁻⁶ kW, left out of the ratio metrics), `flag`.

**`scaled_errors.csv`** — RMSSE and MASE of the 22 forecasters: each window's error divided by the in-sample error of
the weekly seasonal naive (lag 168 h) over its 1,344-hour context. Descriptive. Columns: `set`, `tier`, `model`,
`model_label`, `RMSSE_median_unit`, `RMSSE_geo_mean`, `MASE_median_unit`, `MASE_geo_mean` (median and geometric mean
over units), `units` (units with at least one window whose scale is not zero), `units_below_1_share`, `flag`.

## Error by forecast day and energy

**`lead_day_metrics.csv`** — hourly error by forecast day 1–31, six test populations, 22 forecasters. Descriptive.
Columns: `set`, `tier`, `windows`, `units`, `model`, `model_label`, `model_class`, `day`, `RMSE_kW` and `MAE_kW` (mean
over units), `CV_RMSE_pct` and `nMAE_pct` (median over units, % of the unit's mean load), `GM_CV_RMSE_pct` (geometric
mean, over one fixed set of units, of each unit's RMSE on that day in % of its mean load; the set holds the units with a
nonzero mean load and a nonzero error on every day for every forecaster), `n_units_gm` (size of that set), `flag`.

**`lead_day_energy.csv`** — energy error by forecast day for the same forecasts. Descriptive. Columns: `set`, `tier`,
`windows`, `units_in_U` (size of the fixed unit set, which also requires nonzero daily and cumulative energy errors),
`model`, `model_label`, `model_class`, `day`, `GM_CV_daily_energy_pct` (error of the energy of day d),
`GM_CV_cumulative_energy_pct` (error of the energy delivered from day 1 through day d; day 31 is the monthly energy
error), each an RMS over the unit's windows in % of its mean load, as a geometric mean over the set; `flag`.

**`energy_error.csv`** — monthly energy error of ankyra-lite against each baseline, nine populations. The energy error
of a window is 744 × the error of the window mean of the delivered trajectory (P3), for every forecaster. Columns:
`set`, `tier`, `unit_set` (BDG2 also `without the three near-zero meters`), `windows`, `units`, `model`, `model_label`,
`model_class`, `log_ratio`, `um_low`, `um_high`, `improvement_pct` (the unit-equal contrast on that error; zero on
ankyra-lite's own row), `median_energy_ape_pct` (each forecaster's median-unit absolute percentage energy error),
`flag`. The sensitivity with ankyra-lite's level readout in place of its trajectory mean is `energy_level_readout.csv`
(below; also `energy_level_readout` in `compare_baselines.json`).

**`energy_level_readout.csv`** — the energy contrasts of `energy_error.csv` (all units) with ankyra-lite's independent
energy readout, 744 × its level read before the nonnegativity projection (P3), in place of the mean of its delivered
trajectory; same estimator, random stream and model order. Descriptive (a sensitivity; `energy_error.csv` keeps the
first release's definition). Columns: `set`, `tier`, `windows`, `units`, `model`, `model_label`, `log_ratio`, `um_low`,
`um_high`, `improvement_pct`, `resolved`, `median_energy_ape_pct` (ankyra-lite's median-unit absolute percentage energy
error with the level readout, repeated on the rows of a population), `windows_projection_binds` (windows on which the
two readouts differ by more than 10⁻⁶ relative, i.e. where the projection binds), `flag`. Written from the saved record
(`compare_baselines.json`), so it holds exactly the numbers of `SUMMARY.md`'s column "Energy (level readout)".

**`block_shares.csv`** — where the hourly squared error sits, six test populations. Descriptive. One row per `set`,
`model`, `quantity` and `block`, the number in `value`; also `windows`, `units`, `flag`.

- `block`: `level`, `daily path`, `within-day`, or `hourly` (the total).
- For `model` = `ankyra-lite`, `TimesFM`, `Chronos-2-X`: `quantity` = `pooled_share` (the block's share of the
  forecaster's squared error pooled over windows) or `median_unit_share` (the median over units of that share).
- For `model` = `ankyra-lite vs TimesFM` and `ankyra-lite vs Chronos-2-X`: `quantity` = `improvement_pct` or
  `log_ratio` of ankyra-lite against that model within the block, or `units_in_ratio`.

## Readouts and intervals

**`peak_readout.csv`** — ankyra-lite's peak readout (the anchoring layer's envelope operator on the daily means of its
delivered trajectory) against other peak estimates of the same windows, nine populations. Columns: `set`, `tier`,
`comparator`, `log_ratio`, `um_low`, `um_high`, `improvement_pct` (the unit-equal contrast on the absolute peak error),
`median_peak_ape_ankyra_pct` and `median_peak_ape_previous_peak_pct` (median-unit absolute percentage peak error of
ankyra-lite's readout and of the previous window's peak, repeated on the rows of a population), `flag`. Values of
`comparator`:

- `trajectory maximum`: the maximum of ankyra-lite's own delivered trajectory;
- `previous-window peak`: the maximum of the last 744 context hours (last month's observed peak);
- `same readout on X`: the same operator applied to baseline X's trajectory (the first release's rule);
- `trajectory maximum of X`: baseline X's own trajectory maximum;
- `same readout on the carrier alone`: the same operator applied to the carrier's three-seed forecast at the origin,
  without the anchoring layer and without the state rule (added on 19 October 2026). It is the counterpart of the first
  release's row against the fixed division: it shows whether the anchoring layer's gains in the daily means carry into
  the peak (P16). It is not a baseline and is not counted with them (`SUMMARY.md`, Figure 5c).

**`intervals_by_population.csv`** — the anchoring layer's interval placed around ankyra-lite's delivered trajectory
(residual quantiles of the history-only forecaster at the unit's pseudo-origins; [EVALUATION.md](../docs/EVALUATION.md#intervals))
and the native quantiles of TimesFM 2.5 and Chronos-2, nine populations. Descriptive. Columns: `arm`
(`ankyra_lite`, `timesfm_native`, `chronos_native`), `set`, `tier`, `units`, `windows`; `cov80`, `cov90` (coverage, a
fraction) and `cov80_week1`–`cov80_week4`, `cov90_week1`–`cov90_week4` by forecast week (days 1–7, 8–14, 15–21,
22–31); `width80`, `width90`, `winkler80`, `winkler80_week1`–`winkler80_week4`, `pinball_0.05`–`pinball_0.95`,
`pinball_mean_common` (the mean over the quantiles every arm has: 0.1, 0.5, 0.9), all means in kW; `flag`. The 90%
columns and the 0.05 and 0.95 pinball losses are blank for TimesFM. On units that switch off the interval's width and
Winkler score carry no information.

**`intervals_winkler_contrasts.csv`** — unit-equal contrasts of the 80% Winkler score of the ankyra-lite interval
against `timesfm_native` and `chronos_native` (`comparator`). Columns: `version` (`ankyra-lite 0.1.0`), `set`,
`comparator`, `log_ratio`, `um_low`, `um_high`, `improvement_pct`, `resolved`, `flag`.

**`intervals_households.json`** — the same interval on the GoiEner households (test windows). Keys: `set`, `windows`,
`units`, `note` (the definitions), `arms` and `contrasts`. `arms` has four entries, `ours_around_ankyra_lite`,
`ours_around_carrier` (the same interval around the carrier alone), `timesfm_native` and `chronos_native`, each with
`cov80`, `width80`, `winkler80`, the pinball losses `pinball_*` and `pinball_mean_common`, and except for TimesFM
`cov90` and `width90`; `contrasts` compares the ankyra-lite interval with each other arm on `winkler80` and
`pinball_mean_common` (`log_ratio`, `UM` = the 95% interval in log units, `pct`).

## Design choices

**`ablation.csv`** — ankyra-lite against reduced and alternative versions of itself, hourly and monthly-energy error,
nine populations. Columns: `set`, `tier`, `ablation`, `log_ratio`, `um_low`, `um_high`, `improvement_pct` (ankyra-lite
over the version; positive = ankyra-lite better), `error` (`hourly` or `energy`), `windows`, `units`, `resolved`
(`first better` = ankyra-lite better), `note` (how the version was obtained), `flag`. Values of `ablation`:

- `carrier alone (no anchoring layer)`: the three-seed carrier forecast at the origin;
- `state rule off (the anchoring layer's own return on state windows)`: the carrier's forecast instead of the seasonal
  naive on state windows;
- `single seed 0 (its carrier anchored alone)`, `single seed 1 (...)`, `single seed 2 (...)`: the stored anchored
  product of each seed's carrier alone;
- `anchoring layer without the within-day anchoring`, `within-day trust per lead block instead of one value per window`,
  `anchoring layer without the short-gap rule`, `anchoring layer without the micro-load rule` (identical by construction
  where no window is a micro-load window): the unchanged anchoring layer re-run on the saved carrier forecasts with one
  switch changed;
- `history-only configuration (anchoring layer without a carrier)`: the anchoring layer's configuration without a
  carrier (saved study forecasts);
- the study's other recipe arms on the same windows and store (`recipe_variants` of `evaluation/products.json`):
  `recipe: from-scratch stage alone, no continuation (anchored)`, `recipe: 8 continuation epochs, validation set held
  out, no span return (anchored)` and `recipe: continuation with a proportional validation set and early stopping
  (anchored)`;
- earlier carriers of the study, all on the from-scratch recipe and its earlier continuations, anchored by the same
  layer: `earlier carrier: structured network without the copy pointer, weekly training windows (anchored)`,
  `carrier without the copy pointer, from-scratch recipe (anchored)`, `earlier continuation study: unweighted loss
  (anchored)`, `earlier continuation study: weighted loss (anchored)` and `earlier continuation study: weighted loss and
  warm-up (anchored)`. They exist on the original CINELDI store only, so CINELDI has no such rows; on Drammen their
  `note` says that their pooled training set used the original CINELDI store.

Versions re-run with the anchoring layer get the seasonal naive on their own state windows, as ankyra-lite does. The
hourly and energy rows against the 8-epoch continuation, and the hourly rows against the from-scratch stage alone,
reproduce the per-population values of `recipe_history.csv`.

Four rows were added on 19 October 2026, after each population's other rows, from the anchoring layer's own fields
(the `fields` mode of `evaluation/lite_reanchor.py`; [below](#handover-within-day-trust-and-combinations)); seasonal
naive on state windows in every one:

- `fixed division (history level and daily path, no handover)`: the fixed-division daily means
  (`fixed_division_daily_means_kw`: the historical level and daily path, without the carrier's level candidate) plus the
  carrier's own within-day block, projected onto nonnegative load;
- `fixed half handover weight`: ankyra-lite with the weekly handover weights replaced by ½ (arm `Ah` of
  `handover_granularity.csv`);
- `one whole-window combination weight of the history side and the carrier`: (1 − a) H + a C, projected, with H the
  history side (fixed-division daily means plus the analog-day shape S0), C the carrier at the origin and a one
  least-squares weight per window of C against H over the whole 744-hour trajectory at the unit's completed
  pseudo-origins: the daily part from the handover pairs (24 × the daily sums, divided by their number), the within-day
  part from the trust triples (divided by their number), clipped to [0, 1] and shrunk to ½ by n/(n + 2), n = handover
  pairs (the first release's control B2);
- `half-and-half average of the history side and the carrier`: H/2 + C/2, projected (the first release's B1).

**`component_contributions.csv`** — what each part of ankyra-lite adds: direct contrasts taken from `ablation.csv`,
hourly and energy. Columns: `set`, `component`, `improvement_pct`, `log_ratio`, `um_low`, `um_high`, `kind`, `error`,
`resolved`, `flag`. Positive = the version with the component (ankyra-lite) better. Components: `the whole anchoring
layer (carrier alone -> ankyra-lite)`, `seasonal-naive state rule`, `within-day anchoring`, `one within-day trust value
per window`, `short-gap rule`, `micro-load rule`, `the carrier inside the anchoring layer (history only ->
ankyra-lite)`, the two recipe steps (`span return (the validation span returned to the training set; both with 8
continuation epochs)`: against the 8-epoch continuation; `continuation with the span return (from-scratch stage ->
released recipe)`: against the from-scratch stage alone), `three-seed ensemble (against one seed, mean over the
three seeds)` (`kind` = `mean of three direct contrasts (point estimate)`, blank interval; all other rows `direct
contrast`), and, added on 19 October 2026, `block-wise weights instead of one whole-window weight` (against the
whole-window combination weight of `ablation.csv`) and `block-wise weights against a half-and-half average`.

**`seed_spread.csv`** — each seed alone against the three-seed ensemble, and each seed's and the ensemble's hourly
contrast against TimesFM and Chronos-2-X, for the carrier alone and for the anchored forecast. Columns: `set`, `tier`,
`unit_set` (BDG2 also without the three near-zero meters: the carrier alone has no state rule, so on those meters its
error is not zero while some baselines' errors are near zero), `stage` (`carrier alone`, `anchored (ankyra-lite)`),
`seed` (`seed 0`–`seed 2`, `three-seed ensemble`), `error`, `comparator` (`three-seed ensemble`, `TimesFM`,
`Chronos-2-X`), `windows`, `units`, `log_ratio`, `um_low`, `um_high`, `improvement_pct` (the first-named, i.e. the
`seed`, against the `comparator`), `resolved`, `flag`. The intervals resample units and months; the spread across the
three seeds is the only measure of retraining noise in this directory.

**`carriers_in_anchoring_layer.csv`** — ankyra-lite against the same anchoring layer with another trained carrier, and
each carrier's anchored forecast against the carrier alone and against the history-only configuration. Columns: `set`,
`tier`, `subset`, `windows`, `units`, `carrier` (`small carrier network (ankyra-lite)`, `LSTM`, `PatchTST`,
`LSTM-T-KAN-M`), `contrast` (`anchored vs carrier alone`, `anchored vs history only`, `ankyra-lite vs anchored
carrier`), `error`, `log_ratio`, `um_low`, `um_high`, `improvement_pct` (first-named against second), `resolved`,
`flag`. The other carriers were trained on their own population only, with the same training cut as the small carrier
(186 days before the test start), and anchored with the same layer (saved study forecasts); on state windows the layer
returns their own forecasts, whereas ankyra-lite delivers the seasonal naive. On CINELDI they were trained on the
original store and anchored on the corrected store.

**`history_length.csv`** — ankyra-lite against the carrier alone, TimesFM and Chronos-2-X (`comparator`) by the length
of the unit's history at the origin, test windows, per population and pooled (`all nine populations`). Descriptive.
Columns: `set`, `stratum`, `windows`, `units`, `comparator`, `log_ratio`, `um_low`, `um_high`, `improvement_pct`,
`resolved`, `flag`. Strata by hours since the unit's first observation (`< 10,248 h (no annual candidate)`,
`10,248 h - 2 years`, `2-3 years`, `>= 3 years`) and by `K_eff`, the number of complete 744-hour pseudo-origin windows the
history allows, capped at 12 (`K_eff 6-8`, `K_eff 9-11`, `K_eff 12`). A stratum with fewer than 10 windows or 3 units
has no row; `K_eff 6-8` has no window in this run, so it has no row at all. The strata are confounded with the
population.

**`anchoring_gain_deciles.csv`** — the two statistics of the anchoring layer's `anchoring_record` against the realised
gain of ankyra-lite over its carrier alone, test windows of the nine populations, deciles over the windows on which the
statistic is defined (not on state windows, whose anchoring record is empty). Descriptive:
the carrier weight is the anchoring layer's own weight, so its association with the gain is partly mechanical. Columns:
`statistic` (`carrier_level_weight`, `history_vs_carrier_log_ratio`), `decile`, `lower_edge`, `upper_edge`, `windows`,
`populations`, `level_improvement_pct`, `level_um_low`, `level_um_high` (ankyra-lite over the carrier alone on the
monthly level, with the 95% interval of the log ratio), `hourly_improvement_pct`, `hourly_um_low`, `hourly_um_high` (the
same on the hourly error). Units with a mean load below 10⁻⁶ kW are left out.

## Contrast (ii) by block, and a carrier-free control

Two files with the columns of the first release's files of the same name, with ankyra-lite as the carrier-enabled
configuration and the small carrier network as the only carrier (the first release had three carriers). Descriptive.
Columns of both: `set`, `tier`, `subset` (`late`), `analysis` (`all late windows`, or `off-state and micro-load windows
excluded`, the second only for the populations with state windows: BDG2 2017, EWELD and both GoiEner populations),
`windows`, `units`, `carrier` (`small carrier network (ankyra-lite)`, or `none` on a contrast without a carrier),
`contrast` (first vs second), `block` (`hourly`, or one of its three parts `level`, `daily path`, `within-day`, which add
up to the hourly squared error of every window, P1), `log_ratio`, `um_low`, `um_high`, `improvement_pct` (first-named
against second, on the RMS over a unit's windows of the block's error; a new generator per contrast and block),
`resolved`, `flag`. The squared level error is the squared monthly-energy error divided by 744², so a `level` row is
the contrast on monthly energy.

**`contrast_ii_blocks.csv`** — in which block the configurations differ. Values of `contrast`:

- `anchored vs history only`: ankyra-lite against the anchoring layer's history-only configuration (no carrier; saved
  study forecasts);
- `anchored vs carrier alone`: ankyra-lite against its carrier alone;
- `carrier alone vs history only`: the carrier's own forecast against the history-only configuration.

The `hourly` and `level` rows of the first two contrasts on all late windows reproduce the `hourly` and `energy` rows of
`carriers_in_anchoring_layer.csv` (checked when the file is written). The history-only configuration uses the day-type
default instead of the analog-day shape and has no state rules, so this contrast compares two complete configurations
and does not isolate what the carrier's forecast contributes.

**`contrast_ii_control.csv`** — what is left of contrast (ii) when both sides run the same procedure. The *carrier-free
control* is the unchanged anchoring layer called with the history-only configuration's own forecasts (at the origin and
at each pseudo-origin, each from the record up to that point) in the carrier's place; it is the first release's
control, saved by the study (it does not involve the carrier), read for CINELDI from the study's re-scoring on the
corrected store. Values of `contrast`:

- `carrier-free control vs history only` (`carrier` = `none`): what the procedure adds without any carrier;
- `anchored vs carrier-free control`: ankyra-lite against the control; both sides run the same procedure and differ in
  the forecast that stands in the carrier's place, and on state windows (ankyra-lite: the seasonal naive; the control:
  the history-only forecast);
- `anchored vs history only with the full analog-day shape`: against a second control, the history-only configuration
  with its day-type default replaced by the control's analog-day shape (no trust weight and no cap);
- `history only with the full analog-day shape vs history only` (`carrier` = `none`).

The rows with `carrier` = `none` do not depend on ankyra-lite; on Oslo and BDG2 they equal the first release's
committed rows. In the contrast against the control the carrier's forecast enters the level candidates, the handover
of the daily means and the within-day mixture at once; the rows by block give a rough location only.

## Handover, within-day trust and combinations

Five files added on 19 October 2026, the counterparts of the first release's analyses of the handover, the within-day
anchoring, the whole-window combination and the level candidates. They read the `fields` mode of
`evaluation/lite_reanchor.py`: the product's three-seed ensemble re-anchored once more with the unchanged layer, keeping
its own fields of every window (daily means, fixed-division daily means, within-day blocks, analog shape, weights) and
the pseudo-origin quantities rebuilt from its public functions (the six handover pairs with
`history.estimate_from_history`, the three trust triples with `analog.AnalogShapes`). Its gate, recorded under
`extra` → `populations` → population → `fields_gate` in `compare_baselines.json`: the re-run reproduces the stored
product bit for bit, and the rebuilt pairs, triples and level reproduce the call's `lead_week_weights`, `within_trust`,
`level_weights` and `daily_means_kw` exactly on every non-state window. Descriptive, on the test windows of the nine
populations; seasonal naive on state windows in every arm; estimator and intervals as in the main tables (hourly
seed SEED + 9, energy SEED + 11, a fresh generator per contrast).

**`handover_granularity.csv`** — what the week-by-week handover contributes. Four arms share the level with the
carrier's level candidate, the delivered within-day block, the projection and the state rule, and differ only in the
weight α on the carrier's daily means m^C against the history side's m^H (the level with the carrier candidate plus the
daily path): `A0` none (α = 0), `Ah` a fixed half, `AM` one weight per window from the same handover pairs with the 31
days pooled, `AW` the weekly weights (= ankyra-lite; the arm reproduces the delivered forecast bit for bit, checked).
m^H is the anchoring layer's own (recomputed from its level and checked against the handover identity
daily = (1 − α)m^H + α m^C, α ≤ 7/8). The columns of the first release's file: `set`, `tier`, `contrast` (first arm vs
second; the first-named is the numerator), `log_ratio`, `um_low`, `um_high`, `improvement_pct`, `pooled_mse_ratio`, and
`flag`. Values of `contrast`: `C1 AW vs Ah`, `C2 AW vs AM`, `C3 AM vs Ah`, `C4 Ah vs A0` (the first release's four
numbered contrasts), `AW vs A0`, `AM vs A0`, each arm against `carrier alone`, `TimesFM` and `fixed division` (the
ablation row of that name), and `C1` and `C2` by forecast week (`..., week 1` to `week 4`: days 1–7, 8–14, 15–21,
22–31). Hourly error only.

**`lead_weeks.csv`** — by forecast week, ankyra-lite and the fixed division against the carrier alone and against
TimesFM, and the estimated handover. The counterpart of the first release's `lead_weeks_first_read.csv` (there on the
three populations scored first after the handover was fixed; here on the nine). Columns: `set`, `tier`, `week` (1–4),
`days`, `windows`, `units`, then `log_ratio`, `um_low`, `um_high` of `ankyra_lite_vs_carrier`, `ankyra_lite_vs_timesfm`,
`fixed_division_vs_carrier` and `fixed_division_vs_timesfm` (the hourly error of that week's hours; negative favours the
first-named), `mean_weight_on_carrier` (the mean over non-state windows of the week's handover weight α_w on the
carrier's daily means; ½ without evidence), `flag`.

*Caveat: BDG2 against TimesFM in weeks 1 and 2.* There ankyra-lite is resolved worse (log ratios 0.654 and 0.135,
that is −92.4% and −14.4%; `flag` is empty and the values are as computed), but these two rows describe a few meters,
not the population. In week 1 three meters, `Lamb_office_Vasiliki`, `Lamb_assembly_Bertie` and `Lamb_office_Velma`
(one window each, origin 1 August 2017), read the same small non-zero value through the whole context and the first
eight forecast days, and a varying load again from day 9. TimesFM repeats that value almost exactly, so its error on
these three units is close to zero; ankyra-lite, which also reads the unit's longer history, forecasts a much higher
load. Their unit log ratios are far larger than any other unit's and make up most of the week-1 unit mean. In week 2
`Hog_lodging_Edgar` and `Eagle_office_Amie`, whose readings are also constant over the week, play the same role. These
are not the three near-zero meters of [`bdg2_near_zero_sensitivity.csv`](#bdg2-near-zero-meters), and none of these
windows is a state window (the readings are above the micro-load threshold). The meter-level facts come from a
read-only check of the saved forecasts ([docs/EVALUATION.md](../docs/EVALUATION.md#numbers-that-are-not-in-results)).

**`within_day_trust.csv`** — where the within-day anchoring acts. One row per population: `set`, `tier`, `windows`,
`state_windows`, `anchored_windows` (the others), and over the anchored windows `mean_trust` and `median_trust` (the
weight on the unit's analog-day shape, 0 to ½), `share_trust_positive`, `share_analog_kept` (the plausibility guard kept
the analog shape), `mean_triples` (completed pseudo-origin triples behind the trust, at most 3) and
`share_three_triples`, `flag`.

**`generic_combinations.csv`** — is the history side a better combination partner for the carrier than a generic local
forecast? The columns of the first release's file plus `flag`: `set`, `tier`, `subset` (`late`), `windows`, `units`,
`contrast`, `best`, `candidates`, `error` (`hourly` or `energy`), `log_ratio`, `um_low`, `um_high`, `improvement_pct`,
`resolved`, `availability_note` (empty: every candidate exists on every test window). `B2(H)` is the whole-window
combination of `ablation.csv` (history side and carrier); `B2(X)` the same procedure with H replaced by X = `SN-week`,
`WeekProf4` or `LYR-profile`, each computed from the load before every pseudo-origin (the panel's definitions; at the
origin they equal the saved baselines, checked for the first two); `EW(X)` = max(½ carrier + ½ X, 0) for X = `SN-week`,
`WeekProf4`, `LYR-profile`, `RIDGE-L`, `MSTL` (the saved baselines). Values of `contrast`: `B2(H) vs B2(SN-week)`,
`B2(H) vs B2(WeekProf4)`, `B2(H) vs B2(LYR-profile)`, `B2(H) vs best B2(X)`, `B2(H) vs best equal-weight`,
`ankyra-lite vs best generic`. "Best" was picked after scoring, per population and error (the candidate against which
the first-named has the largest log ratio, i.e. the lowest error); `best` names it and `candidates` lists the pool. This
choice favours the generic side.

**`level_candidate_hit_rate.csv`** — how often the level candidate with the largest weight (the smallest pseudo-origin
error) is the one whose window mean at the origin is closest to the realised window mean (P20). Each candidate's
window mean at the origin is rebuilt from the layer's level hypotheses (checked: their weighted sum equals the level
with the carrier candidate). Columns: `set` (also `all nine populations`), `tier`, `candidates` (`seven (six historical
and the carrier)` or `six historical`), `windows` (anchored windows with a unique largest weight), `units`,
`windows_tied_excluded` (windows whose largest weight is shared, e.g. without pseudo-origin evidence), `hit_rate`,
`chance_rate` (the mean of 1 / the number of candidates available: the rate of any rule when the pseudo-origin errors
carry no information about the target, P20), `hit_minus_chance` with its 95% interval `um_low`, `um_high` (2,000 bootstrap
draws resampling units, seed 20261019), `share_carrier_selected` and `share_carrier_best` (seven candidates only), `flag`.
Descriptive.

## Training budget and long-context baselines

Two robustness files of the comparison, added on 19 October 2026 (`evaluation/budget_tables.py`,
`evaluation/long_context_tables.py`). They read saved forecasts of the study (nothing was trained or re-forecast for
them) and score them on the test windows with the estimator and seeds of the main tables (hourly SEED + 9, energy
SEED + 11). CINELDI: those forecasts were made on the original store; the corrected store's hour h + 1,416 is the
original store's hour h and the models read no date, so windows are mapped by that shift, and the targets of the two
stores were checked equal on every mapped window.

**`training_budget_sensitivity.csv`** — were the four trained load-only baselines (`DLinear`, `iTransformer`, `LSTM`,
`PatchTST`) held back by their training budget? The first release retrained each on each population with at most 30
epochs and early-stopping patience 8 in place of 15 and 3 (cutoff, training pairs, selection set, learning-rate grid,
seeds and scoring unchanged; protocol fixed before training); those forecasts are scored here against ankyra-lite. One
row per population, model and `error` (72 rows). The columns of the first release's file, plus `error` (`hourly` or
`energy`) and `flag`:

- `original_log_ratio`, `original_um_low`, `original_um_high`, `original_improvement_pct`, `original_resolved`:
  ankyra-lite against the baseline as trained for the main tables; the hourly rows reproduce `benchmark_pairwise.csv`
  (checked when written);
- `relaxed_*`: ankyra-lite against the baseline retrained with the relaxed budget. `*_resolved` is `ankyra-lite better`,
  `baseline better`, `not resolved` or `identical forecasts`;
- `relaxed_vs_original_*`: the relaxed baseline against its original on the same windows (negative `log_ratio` = the
  relaxed model has the lower error); `relaxed_vs_original_resolved` is `relaxed better`, `original better`, `not
  resolved` or `identical forecasts` (training is deterministic, so a relaxed run that selects the same learning rate and
  epoch count repeats the original forecast);
- `lr_original`, `epochs_original`, `lr_relaxed`, `epochs_relaxed`: the selected learning rate and epochs;
  `selected_run_reached_30_epochs`: the selection run at the selected learning rate ran all 30 epochs;
  `selected_epoch_is_30`: the selected epoch is the last one allowed.

**`long_context_baselines.csv`** — zero-shot TimesFM 2.5 with an 8,736-hour (52-week) context (`TimesFM-8736`) and
Chronos-2 with 8,192 hours, the limit of its configuration (`Chronos-2-8192`), the zero-shot rows of the first
release's `long_context_carriers.csv` (its anchored rows are ANKYRA configurations and are not used). Where the record
is shorter or a gap falls inside the span, the context is the longest completely observed run of hours ending at the
origin. Outside the 22 forecasters of the main tables and their ranks. Columns: `set`, `tier`, `subset`, `windows`,
`units`, `contrast` (`ankyra-lite vs TimesFM-8736`, `ankyra-lite vs Chronos-2-8192`, `TimesFM-8736 vs TimesFM`,
`Chronos-2-8192 vs Chronos-2` (the same model with the 1,344-hour context of the main tables), `ankyra-lite vs TimesFM`
and `ankyra-lite vs Chronos-2` (these two reproduce the main tables' point estimates)), `error`, `log_ratio`, `um_low`,
`um_high`, `improvement_pct`, `resolved`, `median_context_hours` (of the long-context model of the row), `flag`.

## Training curves

**`training_curves.csv`** — the training record per population and seed, both stages of the released recipe. Columns:

- `set`, `tier`, `seed`;
- `stage`: `from scratch (early stopping)` (stage 1) or `span return (fixed epochs)` (stage 2);
- `epoch`: 1 … epochs run in stage 1; 0 (the stage-1 weights the continuation starts from) … 8 in stage 2;
- `training_loss`: the mean training loss of the epoch's draws in stage 2; blank in stage 1 (its records keep the
  validation criterion only) and at stage-2 epoch 0;
- `validation_criterion`: the stage-1 criterion (the mean over units of log(1 + unit mean of the squared error divided
  by the squared scale) on the weekly validation windows; [METHOD.md](../docs/METHOD.md#stage-1-early-stopping)). In
  stage 2 the same criterion is computed on the same windows, which by then are training data: it describes and selects
  nothing. Stage-2 epoch 0 reproduces the best stage-1 value (the start check);
- `validation_held_out`: `yes` in stage 1, `no (returned to the training set)` in stage 2;
- `best_epoch`, `epochs_run`: stage 1's selected epoch and its epochs run (at most 30, patience 5); in stage 2 blank and
  8;
- `selected`: `True` on the kept weights (stage 1's best epoch, stage 2's last epoch);
- `returned_span_draws`: in stage 2, how many of the epoch's draws from the population fell on returned span windows;
- `flag`.

## Recipe history

**`recipe_history.csv`** — the readings of the study that chose the recipe, on the test windows of the nine
populations (108 rows). Every number is copied unrounded from the study's scoring record; nothing is recomputed. The
readings, their status and their negatives are described in
[EVALUATION.md](../docs/EVALUATION.md#why-the-recipe-returns-the-validation-span). Columns:

- `comparison`, `first_arm`, `second_arm`: `R vs F`, `V vs F`, `V vs R`, and `F vs P3s`, `R vs P3s`, `V vs P3s`. R is the
  released recipe (validation span returned, 8 epochs), F the control (8 epochs, validation set held out), V the
  proportional validation set with early stopping, P3s the from-scratch stage alone; `first_arm_recipe` and
  `second_arm_recipe` describe each arm in words;
- `population_group`, `populations`: the three short-record populations (3), all nine (9), the six with development
  windows (6), the four long-record ones on which V keeps the 93-day validation set (4);
- `layer`: `all test windows`; for the three short-record populations also the `switch layer` (test windows whose
  target month had no same-season training labels before the span return and has them after it: 150 / 406 / 415 windows
  of BDG2 2017 / GoiEner non-household / GoiEner households) and the `non-switch layer` (the other test windows); the
  full description is in the column;
- `quantity`: `hourly`; `monthly (curve sum)` (the error of the delivered trajectory's monthly mean) and
  `monthly (independent readout)` (the error of the level readout); `peak` (the peak readout against the realised
  maximum); `lead day 1`, `lead days 2-7`, `lead days 8-31` (the hourly error on those forecast days). The layers carry
  hourly, independent-readout energy and peak only; the rows against P3s are hourly only;
- `improvement_pct`, `interval_low_pct`, `interval_high_pct`: the unit-equal contrast in % (positive favours the
  first-named arm) and its 95% interval; `interval` says which interval: for R, F and V against each other the Welch
  interval that combines the unit × target-month bootstrap of the three-seed ensembles with the spread of the three
  single-seed contrasts, for the rows against P3s the unit × month bootstrap of the ensembles only (no seed layer;
  descriptive);
- `label`: `resolved positive`, `resolved negative` or `not resolved`;
- `seed0_pct`, `seed1_pct`, `seed2_pct`: the single-seed contrasts (blank on the rows against P3s);
  `unit_month_low_pct`, `unit_month_high_pct`: the unit × month interval of the ensemble contrast alone; `welch_df`:
  the Welch degrees of freedom;
- `by_population_pct`: the per-population contrasts, as `name value; name value; ...`.

## Constants

**`constants.csv`** — every constant of ankyra-lite in one file (43 rows), the counterpart of the first release's
file. Columns: `group`, `constant`, `value`, `where_fixed`, `sensitivity`, `component`.

- `component` = `carrier` (23 rows): the carrier's and its training recipe's constants, read from the package at run
  time (the defaults of `ankyra_lite.carrier.train.TrainConfig`, of the span-return stage and of the network
  configuration), so that they follow the code; groups `carrier: windows`, `carrier: split`, `carrier: pooled training
  set`, `carrier: sampler`, `carrier: optimisation`, `carrier: early stopping`, `carrier: span return`, `carrier:
  seeds` and `carrier network`. `sensitivity` is filled where the recipe study varied the constant (with a pointer to
  `recipe_history.csv`), else blank.
- `component` = `anchoring layer (unchanged copy; values and where_fixed as in the first release)` (20 rows): `group`,
  `constant`, `value` and `where_fixed` copied verbatim from the first release's `constants.csv`. `sensitivity` is
  blank: the first release measured it with its own carrier, not with this one.

The list in [METHOD.md](../docs/METHOD.md#constants) is the same.

## Robustness

**`robustness.csv`** — ankyra-lite's forecasts recomputed on the CPU by this repository's package from a corrupted
context, against the forecast from the clean context, on test windows of one population (64 windows spread evenly over
its test windows; Drammen by default). The carrier forecasts at the origin and every pseudo-origin, the anchoring
layer, the state rule and the peak readout are recomputed with the released weights for every corruption; the targets
are untouched. This file, unlike the others, is new inference rather than a re-scoring of saved forecasts. The
corruptions and the random stream (`default_rng(20261003)`) are the first release's; its comparator, the foundation
model alone, is replaced by the carrier alone (the three-seed carrier forecast at the origin from the same corrupted
record), and its `timesfm_*` columns are therefore `carrier_*`. One row per `corruption`:

- `miss05`, `miss20`: 5% or 20% of the context hours removed and filled by linear interpolation; `zero24`: a 24-hour
  zero-filled block in the last week; `spike`: one hour at 10 × the context maximum, 36 hours before the origin;
  `shift_plus1`, `shift_minus1`: the context shifted by one hour; `scale0.5`, `scale2`: the whole history multiplied
  by 0.5 or 2 (the forecast is scaled back before comparing).

Columns:

- `ankyra_error_vs_clean_log_ratio`, `carrier_error_vs_clean_log_ratio`: unit-equal log RMS ratio of the hourly error
  with the corrupted context to the error with the clean one (positive = worse), for ankyra-lite and the carrier alone;
- `ankyra_windows_worse_share`, `carrier_windows_worse_share`: share of windows whose error grew;
- `ankyra_displacement_over_s0_median`, `carrier_displacement_over_s0_median`: median over windows of the RMS change of
  the forecast, divided by the unit's scale s0 (the standard deviation of the last 744 hours, at least 1% of their mean
  and at least 10⁻³ kW);
- `level_change_pct_median`, `level_change_pct_max`: median and largest absolute change of the level readout, in %;
  `peak_readout_change_pct_median`: median change of the peak readout, in %; `peak_error_vs_clean_log_ratio`: unit-equal
  log ratio of the peak-readout error (positive = worse);
- `off_state_fired`, `micro_load_fired`: windows on which a state rule fired on the corrupted context;
- `ankyra_vs_carrier_corrupted_log_ratio`: ankyra-lite against the carrier alone on the corrupted context (negative
  favours ankyra-lite);
- `equivariance_max_rel_dev`: on the two scaling rows, the largest relative deviation from exact scale equivariance;
  blank otherwise;
- `windows`, `units`, `set`.

`compare_baselines.json` (`extra`, `robustness`) also records the clean forecasts' largest relative difference from the
stored evaluation product (computed on a GPU), the window sample and the CPU time.

## BDG2 near-zero meters

**`bdg2_near_zero_sensitivity.csv`** — the BDG2 hourly contrasts against the 21 baselines with and without the three
meters that are near zero in every window (10 of the 474 test windows). On them ankyra-lite's error is exactly zero (the
seasonal naive repeats the constant reading), so they drop out of the unit-equal contrasts and count in the mean rank.
Columns: `unit_set`, `subset`, `model`, `log_ratio`, `um_low`, `um_high`, `improvement_pct`, `ankyra_mean_unit_rank`
(ankyra-lite's mean per-unit rank among the 22 on that unit set, repeated on each row).

## Cost

**`cost_per_window.csv`** — what one ankyra-lite forecast costs with this repository's package, on the CPU, one
thread. Columns: `item`, `value`, `unit`. Rows: the carrier's parameter count and checkpoint size; the device and the
number of test windows timed (after one untimed warm-up); the time per window of the whole forecast (carrier forecasts,
anchoring layer, peak readout) and its components (`component:` rows: the carrier forecasts at the origin and the
pseudo-origins for the three seeds, one carrier forecast, the anchoring layer with the state rule, the peak readout);
the time to load the three checkpoints; and two checks (`check:` rows): the largest relative difference of the carrier
forecasts from the stored evaluation forecasts (computed on a GPU) and of the delivered trajectory from the stored
product.

## Example window

**`example_window_cambridge.csv`**, **`example_window_cambridge.json`** — one Cambridge test window (University of
Cambridge estate archive, CC BY 4.0), used in Figures 6 and 7. The first release's example window lies before the
Cambridge test set; this is the test window of the same building whose origin is nearest to it (the JSON's `selection`).
The CSV has `hour` (0 is the origin; negative hours are the 1,344-hour context), `load_kw`, and from hour 0 on
`timesfm_kw` (zero-shot TimesFM 2.5), `ankyra_kw` (ankyra-lite; the column keeps the first release's name) and
`carrier_kw` (the carrier alone). The JSON has `set`, `building`, `target_month_midpoint`, `forecast_period`,
`context_day_types` and `target_day_types` (Monday 0 … Sunday 6, holiday 7), `day_type_codes`, `source`, `note` and
`selection`.

## Reproduction and records

**`REPRODUCTION_CHECK.json`** — the scoring-code check: before ankyra-lite is put in the reference position, the
late-window rows of ten result tables of the first release are recomputed with the release's own reference forecaster
and compared field by field with the published files at commit `487e0d2`. Keys: `utc`, `what`,
`reference_of_the_check`, `published`, `published_state` (the revision and commit read), `faces`, `tables` (per table:
`published_rows_in_scope`, `recomputed_rows`, `fields_compared`, `exact`, `within_last_digit`, `max_abs_diff`,
`mismatches`, `missing`, `extra`, `status`), `populations`, `tables_reproduced`, `tables_not_reproduced`. Only the match
counts are kept; the recomputed release numbers are in no file here. A table that does not reproduce is not written.

**`compare_baselines.json`** — every number behind `SUMMARY.md`, per population (`populations`: the hourly contrasts and
ranks, energy and its level-readout sensitivity, conventional metrics, rank tests, peak contrasts, the product checks,
and under `extra` the records of the files of the sections above), with the run's metadata (`product_label`,
`product_version`, `recipe`, `products`, `forecasters_left_out`, `baselines`, `tables_written`, `extra_files`,
`partial_run`, ...). Paths are written relative to named roots (`<research root>`, `<original release root>`,
`<repo>`), never as personal absolute paths.

**`compare_baselines.log`** — the log of the run that wrote these files, with the same path scrubbing.

**`compare_baselines_supplement.log`** — the log of `compare_baselines.py --supplement` (19 October 2026), which added
the files of [Handover, within-day trust and combinations](#handover-within-day-trust-and-combinations), [Training
budget and long-context baselines](#training-budget-and-long-context-baselines) and `energy_level_readout.csv`, and
the added rows of `ablation.csv`, `component_contributions.csv` and `peak_readout.csv`, to the complete run of
`compare_baselines.log`. No existing number was recomputed or changed: every other line of the three extended files is
kept byte for byte (checked when written), and `compare_baselines.json` gained keys and list entries only, except
`extra` → `absent_files`, replaced by the updated list (`supplement` records the
files and rows written; the per-population records under `extra` → `populations` gained `budget`, `long_context`,
`peak_carrier`, `fields_gate`, `handover`, `combination`, `within_trust` and `hit_rate`, and their `ablation` entries the
four added rows). `SUMMARY.md` was rebuilt from that record (`--summary-only` gives the same file).

**`SUMMARY.md`** — the readable summary of the run: overview per population, the contrasts against the main baselines,
the pooled rank tests, the sections of the extra files, the scoring-code check and the file list.

The released weights behind these products are in [`checkpoints/`](../checkpoints/MANIFEST.json), with their hashes and
source runs ([METHOD.md](../docs/METHOD.md#released-weights)).

## External tests

**`external_tests.csv`** — the external tests of this version ([EXTERNAL_TESTS.md](../docs/EXTERNAL_TESTS.md), Figure
17), written by [`evaluation/external_tables.py`](../evaluation/external_tables.py), not by `compare_baselines.py`. Every
row is copied from the score record of one test set, written once by the study's scorer after the run was frozen on 19
October 2026; nothing is recomputed. The test sets are LCL, UNICON and HKUST (first read of this candidate on a
previously read population) and GoiEner households out of time (a same-source out-of-time supplementary check); the
Helsinki retrospective set is trained and sealed, not scored, and has no row. The carriers were trained for each test
set with the released recipe; they are not the shipped checkpoints. Contrasts with ANKYRA itself or its versions are
not in the file. The records hold no per-day curves (the forecast-day groups are rows of this file), so there is no
by-day file.

| Column | Meaning |
|---|---|
| `test_set`, `test_set_id` | the test set (the record's label) and its id (`lcl`, `unicon`, `hkust`, `goiener_oot`) |
| `identity` | the identity of every number of the row, as the record states it (its word "face" written "test set"); state it with the number |
| `role` | `primary` (the three pre-specified contrasts), `secondary` or `descriptive` |
| `block` | `primary`; `other readouts` (six seeds); `release configuration` (seeds 0–2); `other saved baselines`; `mean unit rank`; the descriptive blocks `span return vs plain continuation` (recipe R against F, seeds 0–2), `continuation and span return vs from-scratch stage`, `vs history-only configuration`, `vs carrier-free control`; `subgroup` |
| `subgroup` | the subgroup of a `subgroup` row; the variant of a `mean unit rank` row |
| `first` | `ankyra-lite (6 seeds)` or `ankyra-lite (3 seeds, release configuration)` |
| `model`, `model_label` | the second forecaster: a baseline (ids and labels as in `benchmark_pairwise.csv`) or one of ankyra-lite's own configurations (`Ns` carrier-free control with the state rule, `N` carrier-free control, `H` history-only configuration, `F_ens3` plain continuation, `P3s_ens6` from-scratch stage alone) |
| `readout` | `hourly`, `monthly energy` (the energy readout), `monthly energy (trajectory sum)`, `peak`, `forecast day 1`, `forecast days 2-7`, `forecast days 8-31` |
| `windows`, `units`, `months` | the contrast's windows, units and target months (units with mean load below 10⁻⁶ kW are left out) |
| `improvement_pct` | 100 (1 − exp(−d)), d = mean over units of log(RMS error of `model` / RMS error of `first`); positive = `first` lower |
| `interval_low_pct`, `interval_high_pct`, `interval` | the 95% interval and how it was formed: Welch, unit-and-month bootstrap combined with the seed spread (M = `seeds`), or the unit-and-month bootstrap alone |
| `seeds`, `seeds_positive`, `welch_df` | M, the number of single-seed contrasts above zero, the Welch degrees of freedom |
| `label` | `resolved better`, `not resolved` (the interval includes zero; not a finding of no difference), `resolved worse`, or `not computed` |
| `unit_month_low_pct`, `unit_month_high_pct` | the unit-and-month bootstrap interval alone |
| `mean_unit_rank`, `candidate_position`, `models_ranked` | rows of `block` = `mean unit rank`: the forecaster's mean per-unit rank, ankyra-lite's position, the number of forecasters ranked |
| `pretraining_overlap` | the record's note on the comparator's pretraining data (LCL is in TimesFM 2.5, Chronos and Chronos-2), without its clause naming a forecaster that is not in this repository |
| `trained_labels_to_C` | `True` for a trained baseline whose training labels reach C, 186 days later than the carrier's C′ |
| `note` | why a value is missing (too few units or months; an interval undefined in the record) |
| `source` | the record and the field the row was copied from (`score_<id>.json: section / … / readout`) |

## Files of the first release without a counterpart

| First release file | Why there is none here |
|---|---|
| `conventional_metrics.csv` (all windows) | ankyra-lite forecasts exist on the test windows only; the earlier panel windows lie in or before its training period |
| `constants_sensitivity.csv` | the first release ran it on design windows without any test window; ankyra-lite forecasts exist on the test windows only, and the anchoring layer's constants are fixed in its byte-identical copy; the recipe constants the study varied are in `recipe_history.csv` and the `sensitivity` column of `constants.csv` |
| `contrast_ii_control_description.csv` | it describes the calls of the carrier-free control, which does not involve the carrier; the contrasts are in `contrast_ii_control.csv` |
| `ankyra_2_vs_1x_by_day.csv`, `bdg2_micro_load_windows.csv`, `reevaluation_2_*.json`, `closeout_status.json`, the version folders | records of ANKYRA's own versions and design decisions (ankyra-lite's own recipe history is `recipe_history.csv`) |
| `carrier_swap*.csv`, the anchored rows of `long_context_carriers.csv` | foundation models as ANKYRA's carrier are ANKYRA configurations; the two zero-shot long-context forecasts are scored in `long_context_baselines.csv`, and trained carriers inside this anchoring layer are in `carriers_in_anchoring_layer.csv` |
| `trained_carriers_sensitivity.csv` | it measures what the earlier training cutoff of a trained carrier costs that carrier alone; ankyra-lite's carriers, and the other carriers of `carriers_in_anchoring_layer.csv`, are trained with that earlier cutoff already |
| `lcl_*`, `hkust_*`, `helsinki_*`, `unicon_*` | the external tests of this version are reported differently: `external_tests.csv` ([above](#external-tests)) holds LCL, UNICON and HKUST (first read of this candidate on a previously read population) and a same-source out-of-time supplementary check (GoiEner households); Helsinki (retrospective) is trained and sealed, not yet scored; UCI370 was skipped |

The first release's `handover_granularity.csv`, `generic_combinations.csv` and `training_budget_sensitivity.csv` have
counterparts of the same name here, `lead_weeks_first_read.csv` has `lead_weeks.csv`, `trained_carriers.csv` has
`carriers_in_anchoring_layer.csv`, and the zero-shot rows of `long_context_carriers.csv` have `long_context_baselines.csv`.
