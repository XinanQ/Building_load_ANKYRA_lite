# ankyra-lite against the baselines

Test windows (`late` subset: origins after the training cutoff of the trained baselines) of 9 populations. The reference is **ankyra-lite** (version 0.2.0): the small carrier network
(copy-pointer structured load network, 61,613 parameters) feeding the ANKYRA 2.2.1 anchoring layer, month-ahead (744 h)
hourly forecasts. For each population the network was trained from scratch on a pooled training set (windows of several populations whose labels end before that population's validation set) with early stopping on the population's last 93 days before the training cut; then that validation span was returned to the training set and training continued for 8 epochs with each training window's loss weighted by the frozen inverse-noise rule (recipe Rw); six seeds averaged at every issue time before anchoring; the origin-continuity layer then acts on the anchored trajectory, while energy, level and peak are read before it (readout rule A).
It is compared with **21 baselines** (22 forecasters in all): the 20 external baselines of the ANKYRA release
and LSTM-T-KAN-M, the LSTM-T-KAN model with its model and training code copied line by line from the authors' public
repository (commit ce24938) and the authors' exp2 feature list (load and calendar). ANKYRA itself and its versions
are not among the forecasters.

## Read this first

- **Re-evaluation.** The test windows have been read before (the release scoring, the scoring of the small carrier
  network and an earlier hourly comparison), and parts of the recipe were chosen after looking at them (the copy
  pointer's direction, the seasonal-naive state rule). Every number here is a retrospective description, not a first test.
- **Drammen and CINELDI.** Both carriers were retrained with the corrected CINELDI store (the original store labels CINELDI's first hour 59 days too early): CINELDI on its own corrected data, Drammen with corrected CINELDI windows in its pooled training set. CINELDI is scored on the corrected store, with the baselines recomputed there: re-forecast where the start date enters, and the calendar-using trained baselines (GBT-T, TiDE, iTransformer-X and LSTM-T-KAN-M) retrained with the same code and seeds (LSTM-T-KAN-M also with the same protocol and learning-rate grid; its retrain first reproduced its stored original-store forecast bit for bit). Drammen's baselines are unchanged (its own data are not affected).
- **TiDE** has a known implementation flaw: a decoder layer normalises a one-dimensional output to a constant, so
  the model reduces to a linear map of the 1,344-hour context plus a constant, and its covariates are inert. It is
  kept under its release name, labelled `Linear-context (TiDE implementation)`.
- **LSTM-T-KAN-M** was trained per population with the model and training code copied line by line from the
  authors' public repository (commit ce24938, MIT licence) and the authors' exp2 feature list. It is not a strict
  reproduction of the paper. Deviations: one 744-step model per population instead of one model per horizon and
  building; per-unit z-scores from the hours before the cut instead of Min-Max scaling; the panel's split (training
  cut, last 20% of training origins for selection, origins every 24 h, at most 16,000 training pairs) instead of a
  70/10/20 stride-1 split; no 3-point smoothing; at most 30 epochs with early-stopping patience 8 (authors: 100 and
  10); learning rate chosen from 1e-4, 3e-4 and 1e-3 on the selection pairs; mean of seeds 0, 1 and 2 instead of one
  run with seed 42; one fixed feature list with this repository's definitions and no weather. Every deviation is
  listed in docs/EVALUATION.md. It is an ordinary baseline here, in every table and rank.
- TimesFM 2.5's pretraining collection includes seven BDG2 sites.
- On the three near-zero BDG2 meters (10 test windows) ankyra-lite's error is exactly zero: the micro-load rule
  fires and the seasonal naive repeats the meter's constant reading. The unit-equal contrasts need a nonzero error
  on both sides, so these meters drop out of them, and they count with ankyra-lite ranked first in the mean unit rank
  (`bdg2_near_zero_sensitivity.csv`).
- The Suzhou park (four series) has no ankyra-lite forecast and is not in the tables.
- **The recipe was chosen on these test windows.** The span return was picked in the study's one reading of four recipe
  arms on these windows (`recipe_history.csv`). Its gain is concentrated on the short-record populations' windows without
  same-season training labels; it is about zero elsewhere, and it has two disclosed negatives: Drammen (hourly, against
  the same continuation without the span return) and forecast days 2-7 of the six development populations (resolved).
- No model was trained or tuned for this comparison; saved forecasts were scored with the release's own scoring
  functions. The rows of switched-off parts of the anchoring layer re-run the unchanged anchoring layer on the saved
  carrier forecasts (`evaluation/lite_reanchor.py`); the single-seed rows read the study's anchored single-seed products
  (one checked bit for bit by re-anchoring); the cost and robustness tables run this repository's package on the CPU.

## How to read the numbers

- `improvement_pct` = 100[1 - exp(r)], r = the unit-equal log RMS ratio of ankyra-lite against the baseline
  (positive favours ankyra-lite). `*` = resolved better (the 95% unit x target-month bootstrap interval of r lies
  below zero), `(-)` = resolved worse (the interval lies above zero), no mark = not resolved.
- **Hourly**: hourly errors over the 744 hours. **Energy**: monthly energy error, 744 x the error of the window's
  mean load of the forecaster's trajectory (the release definition); for ankyra-lite the trajectory before the continuation
  layer (readout rule A; the same as the delivered trajectory when the layer is off).
- **Peak**: ankyra-lite's peak readout (the envelope operator on the daily means of its trajectory before the continuation layer) against
  the same readout applied to the baseline's trajectory, which is the release rule (`same readout on X`); the files
  also hold the baseline's own trajectory maximum (`trajectory maximum of X`).
- **Rank**: position among the 22 forecasters by mean per-unit rank (forecasters ranked within each unit by RMSE).
- On windows where the anchoring layer's off-state or micro-load rule fires, ankyra-lite returns the weekly
  seasonal naive forecast (the last 168 hours repeated).
- In the result files the reference forecaster is ankyra-lite: `model` = `ankyra-lite` on its rows, and column names
  that say `ankyra` (`median_peak_ape_ankyra_pct`, `wilcoxon_holm_p_vs_ankyra`, `ankyra_mean_unit_rank`,
  `ankyra_design_status`, `ankyra_kw`) refer to ankyra-lite.

## Overview

Counts are over the 21 baselines: resolved better / resolved worse.

| Population | Windows / units | Seasonal-naive windows | Rank of 22 (mean unit rank) | Hourly | Energy | Energy (level readout) | Peak (same readout) | Peak (trajectory max) | Flag |
|---|---|---:|---|---|---|---|---|---|---|
| BDG2 2017 | 474 / 142 | 33 | 1 (4.68) | 13 / 0 | 7 / 0 | 7 / 0 | 4 / 0 | 17 / 0 |  |
| GoiEner non-household | 890 / 481 | 13 | 1 (5.02) | 14 / 0 | 13 / 0 | 13 / 0 | 2 / 0 | 19 / 0 |  |
| GoiEner households | 696 / 696 | 35 | 1 (5.34) | 17 / 1 | 16 / 0 | 16 / 0 | 11 / 0 | 21 / 0 |  |
| Cambridge | 730 / 108 | 0 | 1 (3.04) | 21 / 0 | 20 / 0 | 20 / 0 | 12 / 0 | 21 / 0 |  |
| HEEW Arizona | 658 / 138 | 0 | 1 (5.11) | 14 / 0 | 7 / 0 | 7 / 0 | 7 / 0 | 16 / 0 |  |
| EWELD | 523 / 197 | 245 | 1 (7.09) | 13 / 0 | 6 / 0 | 6 / 0 | 5 / 0 | 19 / 0 |  |
| Oslo | 581 / 41 | 0 | 1 (1.78) | 20 / 0 | 19 / 0 | 19 / 0 | 17 / 0 | 17 / 0 |  |
| Drammen | 689 / 43 | 0 | 1 (1.67) | 20 / 0 | 13 / 0 | 13 / 0 | 6 / 1 | 19 / 0 |  |
| CINELDI | 475 / 45 | 0 | 1 (2.89) | 21 / 0 | 13 / 0 | 13 / 0 | 3 / 0 | 19 / 0 |  |

- **Hourly.** Resolved losses: GoiEner households: TimesFM (-4.6 (-)).
- **Energy** is the release definition (each forecaster's trajectory; ankyra-lite's before the continuation layer). The release's energy table covers the six test
  populations and the Suzhou park; the Oslo, Drammen and CINELDI rows are additional, with the same estimator.
- **Energy (level readout)** is a sensitivity: ankyra-lite's independent energy readout (744 x its level) in place
  of its trajectory's mean (before the continuation layer). The two differ only on windows where the projection onto nonnegative load
  binds: BDG2 2017 1, GoiEner non-household 138, GoiEner households 21, Cambridge 10, HEEW Arizona 0, EWELD 18, Oslo 0, Drammen 2, CINELDI 27 windows.
- **Continuation layer.** The hourly tables score the delivered trajectory (after the origin-continuity layer); the
  energy tables (`energy_error.csv`, `lead_day_energy.csv`), the level readout and the peak readout read ankyra-lite before
  the layer (readout rule A), so the delivered trajectory's mean differs from the energy readout (`continuation_record.csv`).
- **Peak (trajectory max)** counts the contrasts against each baseline's own trajectory maximum.

## By population

Improvement of ankyra-lite in % (positive = ankyra-lite better). Energy (level) = with ankyra-lite's level
readout; Peak (readout) = against the same readout on the baseline; Peak (max) = against the baseline's
trajectory maximum. The CSV files hold all 21 baselines.

### BDG2 2017 (474 windows, 142 units; rank 1 of 22, mean unit rank 4.68)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +0.7 | +14.7 | +14.7 | +4.0 | +35.8 * |
| TimesFM-X | +5.0 | +18.5 | +18.5 | +2.6 | +35.6 * |
| Chronos-2 | +5.1 | +15.2 | +15.2 | +7.8 | +37.3 * |
| Chronos-2-X | -4.0 | +3.3 | +3.3 | -0.2 | +34.2 * |
| PatchTST | +10.5 * | +14.5 | +14.5 | +4.6 | +34.1 * |
| iTransformer-X | +10.2 * | +21.1 * | +21.1 * | +9.6 | +39.3 * |
| LSTM | +16.1 * | +18.9 * | +18.9 * | +3.7 | +35.9 * |
| GBT-T | +7.1 | +11.3 | +11.3 | +1.2 | +7.8 |
| LSTM-T-KAN-M | +12.7 * | +11.6 | +11.6 | +2.7 | +41.4 * |
| RIDGE-L | +5.0 | +10.7 | +10.7 | +4.8 | +46.9 * |
| Seasonal naive (week) | +19.2 * | +21.2 | +21.2 | +11.9 | +8.9 |

Also: peak readout vs its own trajectory maximum +37.7 *; vs the previous window's peak -4.4; energy without the three near-zero meters: TimesFM +14.7, TimesFM-X +18.5, Chronos-2 +15.2, Chronos-2-X +3.3, PatchTST +14.5, iTransformer-X +21.1 *, LSTM +18.9 *, GBT-T +11.3, LSTM-T-KAN-M +11.6, RIDGE-L +10.7, SN-week +21.2.

### GoiEner non-household (890 windows, 481 units; rank 1 of 22, mean unit rank 5.02)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +4.7 | +20.2 * | +20.5 * | +0.8 | +41.7 * |
| TimesFM-X | +5.2 | +17.7 * | +18.0 * | +2.7 | +39.6 * |
| Chronos-2 | +2.5 | +17.7 | +18.0 | +6.7 | +28.2 * |
| Chronos-2-X | -0.5 | +12.4 | +12.7 | +3.4 | +35.5 * |
| PatchTST | +10.4 * | +14.4 | +14.7 | +5.0 | +45.8 * |
| iTransformer-X | +5.2 | +16.6 | +16.8 | -2.9 | +34.3 * |
| LSTM | +16.8 * | +20.1 * | +20.3 * | +1.1 | +43.0 * |
| GBT-T | +29.2 * | +44.5 * | +44.6 * | -1.1 | +60.3 * |
| LSTM-T-KAN-M | +15.2 * | +19.9 * | +20.1 * | -0.2 | +41.4 * |
| RIDGE-L | +6.9 | +8.0 | +8.3 | -11.6 | +32.5 * |
| Seasonal naive (week) | +21.5 * | +15.8 | +16.0 | -8.6 | -23.6 |

Also: peak readout vs its own trajectory maximum +39.7 *; vs the previous window's peak -34.7.

### GoiEner households (696 windows, 696 units; rank 1 of 22, mean unit rank 5.34)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | -4.6 (-) | +32.8 * | +32.9 * | +10.2 * | +65.8 * |
| TimesFM-X | -0.9 | +28.7 * | +28.9 * | +10.0 * | +66.4 * |
| Chronos-2 | -2.8 | +33.6 * | +33.7 * | +11.0 * | +66.3 * |
| Chronos-2-X | -2.9 | +32.8 * | +32.9 * | +11.7 * | +65.8 * |
| PatchTST | +5.7 * | +17.8 * | +18.0 * | +6.7 * | +67.5 * |
| iTransformer-X | +6.8 * | +16.4 | +16.6 | +7.1 * | +66.9 * |
| LSTM | +8.5 * | +16.7 * | +17.0 * | +3.4 | +68.3 * |
| GBT-T | +11.2 * | +21.2 * | +21.5 * | +5.8 | +57.8 * |
| LSTM-T-KAN-M | +8.6 * | +19.0 * | +19.2 * | +2.5 | +68.6 * |
| RIDGE-L | +4.3 * | +2.0 | +2.3 | +5.0 | +65.6 * |
| Seasonal naive (week) | +20.7 * | +6.8 | +7.1 | +6.4 | +30.3 * |

Also: peak readout vs its own trajectory maximum +68.2 *; vs the previous window's peak +2.3.

### Cambridge (730 windows, 108 units; rank 1 of 22, mean unit rank 3.04)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +15.4 * | +24.6 * | +24.6 * | +10.0 * | +41.4 * |
| TimesFM-X | +16.8 * | +26.5 * | +26.6 * | +9.3 | +40.8 * |
| Chronos-2 | +16.8 * | +30.7 * | +30.8 * | +9.8 * | +42.7 * |
| Chronos-2-X | +10.0 * | +19.7 * | +19.8 * | +4.5 | +40.2 * |
| PatchTST | +14.5 * | +20.3 * | +20.4 * | +4.4 | +36.4 * |
| iTransformer-X | +9.0 * | +8.0 | +8.1 | +3.1 | +35.8 * |
| LSTM | +36.2 * | +47.1 * | +47.1 * | +19.8 * | +45.6 * |
| GBT-T | +12.7 * | +22.7 * | +22.8 * | +9.2 | +30.2 * |
| LSTM-T-KAN-M | +45.2 * | +60.4 * | +60.5 * | +26.9 * | +46.7 * |
| RIDGE-L | +13.6 * | +23.2 * | +23.2 * | +8.9 * | +46.0 * |
| Seasonal naive (week) | +27.8 * | +31.9 * | +31.9 * | +10.5 | +26.7 * |

Also: peak readout vs its own trajectory maximum +32.6 *; vs the previous window's peak +5.2.

### HEEW Arizona (658 windows, 138 units; rank 1 of 22, mean unit rank 5.11)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +2.5 | -0.1 | -0.1 | -1.2 | +23.6 |
| TimesFM-X | +4.7 | +3.5 | +3.5 | +2.5 | +33.8 * |
| Chronos-2 | +5.4 | +6.2 | +6.2 | -2.0 | +29.8 * |
| Chronos-2-X | +0.8 | -0.8 | -0.8 | +0.0 | +27.3 * |
| PatchTST | +6.3 | +4.8 | +4.8 | -1.0 | +33.2 * |
| iTransformer-X | +5.0 | -0.7 | -0.7 | -0.2 | +33.6 * |
| LSTM | +32.0 * | +34.0 * | +34.0 * | +18.4 * | +49.8 * |
| GBT-T | +5.4 * | +3.1 | +3.1 | -2.6 | +14.1 |
| LSTM-T-KAN-M | +30.5 * | +38.2 * | +38.2 * | +27.0 * | +45.1 * |
| RIDGE-L | +4.2 | +1.6 | +1.6 | +3.6 | +39.9 * |
| Seasonal naive (week) | +16.9 * | +7.2 | +7.2 | +2.8 | +10.2 |

Also: peak readout vs its own trajectory maximum +32.6 *; vs the previous window's peak +9.8.

### EWELD (523 windows, 197 units; rank 1 of 22, mean unit rank 7.09)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +1.5 | -2.8 | -2.8 | +2.9 | +31.9 * |
| TimesFM-X | +2.1 | -2.2 | -2.2 | -2.7 | +35.2 * |
| Chronos-2 | -0.2 | -1.7 | -1.7 | +0.7 | +41.1 * |
| Chronos-2-X | +4.2 | +16.8 | +16.8 | +0.6 | +38.8 * |
| PatchTST | +5.1 | -5.4 | -5.5 | +5.0 | +36.6 * |
| iTransformer-X | +3.4 | +5.0 | +5.0 | -0.4 | +34.3 * |
| LSTM | +21.4 * | +37.3 * | +37.3 * | +25.8 * | +60.4 * |
| GBT-T | +29.0 * | +24.5 | +24.5 | +30.3 * | +70.2 * |
| LSTM-T-KAN-M | +18.6 * | +36.8 * | +36.8 * | +24.3 * | +56.8 * |
| RIDGE-L | +3.5 | +5.0 | +5.0 | +4.0 | +46.1 * |
| Seasonal naive (week) | +15.2 * | -0.2 | -0.2 | -3.6 | -34.8 |

Also: peak readout vs its own trajectory maximum +43.4 *; vs the previous window's peak -23.7.

### Oslo (581 windows, 41 units; rank 1 of 22, mean unit rank 1.78)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +23.0 * | +34.9 * | +34.9 * | +16.4 * | +38.5 * |
| TimesFM-X | +25.0 * | +38.6 * | +38.6 * | +16.8 * | +37.4 * |
| Chronos-2 | +23.5 * | +37.0 * | +37.0 * | +15.4 * | +41.6 * |
| Chronos-2-X | +14.6 * | +20.4 * | +20.4 * | +7.1 | +31.8 * |
| PatchTST | +15.6 * | +23.6 * | +23.6 * | +9.9 * | +18.4 |
| iTransformer-X | +10.2 * | +10.2 | +10.2 | +0.7 | +15.4 |
| LSTM | +42.3 * | +53.8 * | +53.8 * | +30.5 * | +54.3 * |
| GBT-T | +2.9 | +5.9 | +5.9 | -4.2 | -17.5 |
| LSTM-T-KAN-M | +40.0 * | +53.3 * | +53.3 * | +29.9 * | +49.3 * |
| RIDGE-L | +23.7 * | +32.8 * | +32.8 * | +16.4 * | +45.6 * |
| Seasonal naive (week) | +36.4 * | +44.6 * | +44.6 * | +18.5 * | +38.8 * |

Also: peak readout vs its own trajectory maximum +21.3 *; vs the previous window's peak +26.8 *.

### Drammen (689 windows, 43 units; rank 1 of 22, mean unit rank 1.67)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +20.9 * | +29.0 * | +29.0 * | +14.6 | +40.5 * |
| TimesFM-X | +16.6 * | +22.9 * | +22.9 * | +12.0 | +37.3 * |
| Chronos-2 | +17.9 * | +27.4 * | +27.4 * | +12.3 | +39.6 * |
| Chronos-2-X | +7.3 | +4.3 | +4.3 | -1.1 | +31.2 * |
| PatchTST | +12.9 * | +8.7 | +8.7 | -4.6 | +19.8 * |
| iTransformer-X | +11.8 * | +8.8 | +8.8 | -4.0 | +23.0 * |
| LSTM | +29.8 * | +23.0 | +23.0 | +18.4 * | +46.2 * |
| GBT-T | +11.3 * | +13.8 * | +13.8 * | -10.6 (-) | +9.5 |
| LSTM-T-KAN-M | +26.1 * | +23.6 * | +23.6 * | +16.4 * | +46.4 * |
| RIDGE-L | +9.7 * | -5.0 | -5.0 | +4.3 | +41.7 * |
| Seasonal naive (week) | +29.0 * | +30.0 * | +30.0 * | +12.9 | +24.7 * |

Also: peak readout vs its own trajectory maximum +25.9 *; vs the previous window's peak +15.7.

### CINELDI (475 windows, 45 units; rank 1 of 22, mean unit rank 2.89)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +10.5 * | +22.4 * | +22.4 * | +6.9 | +52.7 * |
| TimesFM-X | +8.9 * | +20.3 * | +20.3 * | +5.6 | +49.2 * |
| Chronos-2 | +8.5 * | +19.8 | +19.8 | +7.8 | +48.8 * |
| Chronos-2-X | +6.7 * | +16.7 | +16.7 | +7.6 | +49.4 * |
| PatchTST | +6.8 * | +8.6 | +8.6 | +2.8 | +45.0 * |
| iTransformer-X | +7.4 * | +13.2 | +13.2 | +2.7 | +47.1 * |
| LSTM | +25.3 * | +28.1 * | +28.1 * | +11.1 | +59.2 * |
| GBT-T | +13.1 * | +25.3 * | +25.3 * | +6.8 | +40.6 * |
| LSTM-T-KAN-M | +18.7 * | +14.4 | +14.4 | +4.9 | +52.3 * |
| RIDGE-L | +5.8 * | +9.8 | +9.8 | +7.0 | +53.7 * |
| Seasonal naive (week) | +24.6 * | +23.7 * | +23.8 * | +8.0 | +27.6 |

Also: peak readout vs its own trajectory maximum +46.5 *; vs the previous window's peak -18.2 (-).

## Rank tests, six test populations pooled

1762 units, Friedman p < 1e-300, Nemenyi critical difference 0.786; ankyra-lite's mean rank
5.234 (position 1 of 22). Holm-corrected Wilcoxon: significantly better than 21 of 21 baselines, significantly worse than 0. Descriptive.

## Data splits

| Population | Development windows | First training origin | Validation set from | Training cut | Test set (first ... last origin) | Test windows / units | Origin months | Days of training origins before the cut | Validation share | Returned validation windows |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| BDG2 2017 | no | 2016-02-26 | 2016-10-26 | 2017-01-27 | 2017-08-01 ... 2017-12-01 | 474 / 142 | 5 | 336 | 23.7% | 5042 |
| GoiEner non-household | no | 2018-02-26 | 2018-07-26 | 2018-10-27 | 2019-05-01 ... 2019-12-01 | 890 / 481 | 8 | 243 | 38.3% | 5381 |
| GoiEner households | no | 2018-02-26 | 2018-09-25 | 2018-12-27 | 2019-07-01 ... 2019-12-01 | 696 / 696 | 6 | 304 | 28.2% | 6292 |
| Cambridge | yes | 2000-02-26 | 2012-09-25 | 2012-12-27 | 2013-07-01 ... 2022-12-01 | 730 / 108 | 107 | 4688 | 1.9% | 312 |
| HEEW Arizona | yes | 2014-02-26 | 2018-02-25 | 2018-05-29 | 2018-12-01 ... 2022-12-01 | 658 / 138 | 39 | 1553 | 4.5% | 749 |
| EWELD | yes | 2016-07-15 | 2018-10-26 | 2019-01-27 | 2019-08-01 ... 2022-07-01 | 523 / 197 | 36 | 926 | 11.0% | 1969 |
| Oslo | yes | 2012-11-12 | 2018-12-26 | 2019-03-29 | 2019-10-01 ... 2024-06-01 | 581 / 41 | 52 | 2328 | 3.6% | 304 |
| Drammen | yes | 2018-02-26 | 2019-11-27 | 2020-02-28 | 2020-09-01 ... 2022-02-01 | 689 / 43 | 18 | 732 | 9.2% | 396 |
| CINELDI | yes | 2019-04-12 | 2020-06-26 | 2020-09-27 | 2021-04-01 ... 2022-02-01 | 475 / 45 | 11 | 534 | 13.7% | 389 |

True calendar dates (`datasets.csv`, column `calendar`). The test set starts 186 days after the training cut, so that
the carrier's forecasts at all six pseudo-origins are issued after the cut. The validation set is held out in the
from-scratch stage only; the span return puts it back (`returned_validation_windows`: the population's own validation
windows returned at its own sampling density, labels ending at least 24 h before the cut).

## Training (`training_curves.csv`)

| Population | From-scratch stage: best epoch (seeds 0-5) | Epochs run | Span return: epochs | Returned windows | Start criterion reproduced |
|---|---|---|---|---|---|
| BDG2 2017 | 12, 9, 27, 16, 24, 29 | 17, 14, 30, 21, 29, 30 | 8, 8, 8, 8, 8, 8 | 5042, 5042, 5042, 5042, 5042, 5042 | True |
| GoiEner non-household | 15, 3, 16, 5, 19, 9 | 20, 8, 21, 10, 24, 14 | 8, 8, 8, 8, 8, 8 | 5381, 5381, 5381, 5381, 5381, 5381 | True |
| GoiEner households | 15, 16, 5, 6, 10, 7 | 20, 21, 10, 11, 15, 12 | 8, 8, 8, 8, 8, 8 | 6292, 6292, 6292, 6292, 6292, 6292 | True |
| Cambridge | 16, 12, 16, 29, 21, 29 | 21, 17, 21, 30, 26, 30 | 8, 8, 8, 8, 8, 8 | 312, 312, 312, 312, 312, 312 | True |
| HEEW Arizona | 27, 20, 23, 27, 27, 30 | 30, 25, 28, 30, 30, 30 | 8, 8, 8, 8, 8, 8 | 749, 749, 749, 749, 749, 749 | True |
| EWELD | 10, 9, 7, 6, 14, 10 | 15, 14, 12, 11, 19, 15 | 8, 8, 8, 8, 8, 8 | 1969, 1969, 1969, 1969, 1969, 1969 | True |
| Oslo | 14, 9, 30, 30, 15, 12 | 19, 14, 30, 30, 20, 17 | 8, 8, 8, 8, 8, 8 | 304, 304, 304, 304, 304, 304 | True |
| Drammen | 21, 4, 23, 3, 2, 13 | 26, 9, 28, 8, 7, 18 | 8, 8, 8, 8, 8, 8 | 396, 396, 396, 396, 396, 396 | True |
| CINELDI | 14, 6, 6, 6, 5, 13 | 19, 11, 11, 11, 10, 18 | 8, 8, 8, 8, 8, 8 | 389, 389, 389, 389, 389, 389 | True |

The from-scratch stage records the validation criterion of every epoch, not the training loss (`training_loss` is
blank there). The span-return epochs record both; their validation criterion is computed on windows that are back in
the training set, so it is a description, not a held-out score. Epoch 0 of the span return is the selected from-scratch
weights (its criterion reproduces the best from-scratch value).

## Recipe history (`recipe_history.csv`)

The released recipe was chosen in the study's one reading of four recipe arms on these test windows: the from-scratch
stage alone, and three continuations of it for 8 epochs from the same weights, with the validation set held out (plain
continuation), with the validation span returned to the training set (span return, the released recipe) and with a
proportional validation set and early stopping (the study's P3s, F, R and V arms; the codes are the `first_arm` and
`second_arm` values of the file). Positive favours the first-named arm; Welch 95% intervals include the seed layer.

| Readout | Span return against the plain continuation |
|---|---|
| three short-record populations, hourly | +1.17% (-0.64 to +2.94; not resolved) |
| switch layer (BDG2 2017 150, GoiEner non-household 406, GoiEner households 415 windows), hourly | +3.11% (+0.98 to +5.21; resolved positive) |
| non-switch layer, hourly | -0.52% (-2.08 to +1.02; not resolved) |
| nine populations, hourly | +0.11% (-0.82 to +1.03; not resolved) |
| six development populations, hourly | -0.42% (-1.22 to +0.37; not resolved) |
| six development populations, forecast days 2-7 | -1.35% (-2.43 to -0.28; resolved negative) |
| three short-record populations, monthly energy (delivered trajectory) | +3.95% (-3.93 to +11.23; not resolved) |
| three short-record populations, peak | +0.67% (-3.88 to +5.02; not resolved) |

- The switch layer is the short-record populations' test windows whose target month had no same-season training labels
  before the span return and has them after it. The gain sits there, as the prediction written before the reading put
  it; elsewhere it is about zero, and over the nine populations it is not resolved.
- **Negatives, disclosed:** Drammen -2.90% hourly (span return against the plain continuation), and forecast days
  2-7 of the six development populations -1.35% (resolved negative).
- The proportional validation set with early stopping gained nothing (four long-record populations, against the plain
  continuation: -0.35% (-1.15 to +0.43; not resolved)); on the switch layer it trails the span return
  (-2.15% (-3.96 to -0.36; resolved negative)).
- Against the un-continued from-scratch stage (unit x month intervals only, nine populations): span return
  +1.08%, plain continuation +0.97%, proportional validation +0.86%.
- These test windows had been read before; the reading gives directions, not a confirmation.

### Version 0.2.0 (the 0.2.0 study's development check and one test read)

Configuration after the test read: `Rw6_c` (weighting True, six seeds True, continuation layer True; after selection: False). Positive favours the first-named configuration; the 0.2.0 against
0.1.0 and the weighting rows have the seed layer of the two three-seed groups.

| Readout (nine populations) | Result |
|---|---|
| 0.2.0 vs 0.1.0, hourly | +1.15% (+0.60 to +1.71; resolved positive) |
| 0.2.0 vs 0.1.0, lead day 1 | +13.97% (+11.87 to +16.02; resolved positive) |
| layer on / off, hourly | +0.43% (+0.14 to +0.78; resolved positive) |
| weighting Rw6 vs R6, hourly | +0.31% (+0.06 to +0.56; resolved positive) |
| seeds R6 vs R3, hourly | +0.41% (+0.15 to +0.70; resolved positive) |

- The test windows had been read many times before; this reading sets directions only.

## Ablation (`ablation.csv`, `component_contributions.csv`)

ankyra-lite against versions of itself, hourly error, improvement of ankyra-lite in % (`*` resolved better,
`(-)` resolved worse). `-` = not available in this run.

| Version | BDG2 2017 | GoiEner non-household | GoiEner households | Cambridge | HEEW Arizona | EWELD | Oslo | Drammen | CINELDI |
|---|---|---|---|---|---|---|---|---|---|
| carrier alone (no anchoring layer) | +3.6 * | +3.3 | -1.3 | +0.1 | +2.2 * | +0.5 | -0.5 | +0.8 | +1.3 |
| state rule off (the anchoring layer's own return on state windows) | +0.3 | -0.0 | +0.3 | 0 (identical) | 0 (identical) | +2.1 | 0 (identical) | 0 (identical) | 0 (identical) |
| without the continuation layer (the same carrier and anchoring layer) | +0.8 * | +0.5 * | +0.4 * | +0.2 | +0.9 * | +0.5 * | -0.3 | +1.0 | +0.0 |
| three-seed group 0 (seeds 0-2, anchored alone) | +0.7 | +0.4 | +0.2 | +0.8 | -0.0 | +0.0 | -0.3 | +0.9 | +0.5 |
| three-seed group 1 (seeds 3-5, anchored alone) | +0.3 | +0.1 | +0.0 | -0.3 | +0.8 | +0.4 | +0.9 | +0.1 | -0.0 |
| anchoring layer without the within-day anchoring | +0.7 * | +2.4 | -0.0 | +0.3 | +0.8 * | -0.0 | +0.6 * | +0.9 * | +0.9 * |
| within-day trust per lead block instead of one value per window | +0.1 | +0.1 | +0.1 | +0.1 | +0.1 * | +0.0 | -0.0 | +0.1 | -0.0 |
| anchoring layer without the short-gap rule | +0.2 | -0.0 | +0.1 | +0.0 | -0.1 | +0.1 | 0 (identical) | -0.0 | 0 (identical) |
| anchoring layer without the micro-load rule | +0.1 | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) |
| history-only configuration (anchoring layer without a carrier) | +8.0 * | +21.1 * | +11.9 * | +15.0 * | +13.8 * | +16.8 * | +20.1 * | +17.4 * | +6.2 * |
| 0.1.0 configuration: recipe R (span return), seeds 0-2, no continuation layer (anchored) | +1.7 | +1.9 * | +0.9 * | +1.4 * | +1.4 * | +0.2 | -0.5 | +2.4 * | +0.8 |
| configuration: recipe R (span return), seeds 0-2, with the continuation layer (anchored) | +0.9 | +1.3 * | +0.6 * | +1.2 * | +0.6 | -0.3 | -0.5 | +1.3 * | +0.8 * |
| configuration: recipe R (span return), six seeds, without the continuation layer (anchored) | +0.9 | +1.4 * | +0.7 * | +0.8 * | +1.3 * | +0.2 | -0.4 | +1.1 | +0.6 |
| configuration: recipe R (span return), six seeds, with the continuation layer (anchored) | +0.2 | +0.9 | +0.3 | +0.5 | +0.5 | -0.3 | -0.2 | +0.1 | +0.6 * |
| configuration: recipe Rw (span return with inverse-noise loss weighting), seeds 0-2, without the continuation layer (anchored) | +1.4 * | +1.0 | +0.6 * | +1.0 * | +0.8 | +0.6 | -0.3 | +1.9 | +0.6 |
| configuration: recipe Rw (span return with inverse-noise loss weighting), seeds 0-2, with the continuation layer (anchored) | +0.7 | +0.4 | +0.2 | +0.8 | -0.0 | +0.0 | -0.3 | +0.9 | +0.5 |
| recipe: from-scratch stage alone, no continuation (anchored) | +2.9 * | +4.6 * | +2.3 * | +3.1 * | +2.3 * | +1.3 | +1.7 * | +0.2 | +1.6 |
| recipe: 8 continuation epochs, validation set held out, no span return (anchored) | +2.2 * | +4.1 | +1.7 * | +2.0 * | +1.2 * | -0.2 | -0.4 | -0.4 | +1.1 |
| recipe: continuation with a proportional validation set and early stopping (anchored) | +2.9 * | +3.0 | +1.1 * | +1.7 * | +1.2 | +1.2 * | -0.1 | -0.0 | +1.4 |
| earlier carrier: structured network without the copy pointer, weekly training windows (anchored) | +4.6 * | +8.7 * | +2.0 * | +11.7 * | +5.7 * | +2.9 * | +5.5 * | +4.3 * | - |
| carrier without the copy pointer, from-scratch recipe (anchored) | +2.9 * | +4.1 * | +2.4 * | +5.0 * | +3.7 * | +1.8 * | +2.2 * | +2.2 | - |
| earlier continuation study: unweighted loss (anchored) | +2.7 * | +3.5 | +1.6 * | +1.8 * | +2.3 * | +1.1 * | -0.1 | +1.6 | - |
| earlier continuation study: weighted loss (anchored) | +1.7 * | +2.8 | +1.0 * | +1.1 | +1.0 | +1.9 * | +0.7 | +1.8 | - |
| earlier continuation study: weighted loss and warm-up (anchored) | +1.8 * | +3.1 | +1.0 * | +1.6 * | +0.1 | +2.0 * | +1.6 * | +1.8 | - |

The rows of switched-off parts of the anchoring layer need re-anchoring (`evaluation/lite_reanchor.py`); the single
seeds are the stored single-seed products; the history-only configuration, the recipe arms and the earlier carriers are
saved study forecasts. The earlier carriers (without the copy pointer, the earlier continuation study, the structured
network) were trained with the from-scratch recipe or its earlier continuations, not with the span return.

## Seeds (`seed_spread.csv`)

Hourly improvement against TimesFM: the carrier alone per seed and the six-seed ensemble, and anchored per three-seed group
(the stored group products) and the ensemble:

| Population | Stage | Members (in order) | Ensemble |
|---|---|---|---|
| BDG2 2017 | carrier alone | seed 0: -69.2 (-) / seed 1: -75.3 (-) / seed 2: -77.8 (-) / seed 3: -74.7 (-) / seed 4: -74.6 (-) / seed 5: -77.3 (-) | -68.5 (-) |
| BDG2 2017 | anchored (ankyra-lite) | group 0: +0.0 / group 1: +0.4 | +0.7 |
| BDG2 2017 (without the three near-zero meters) | carrier alone | seed 0: -4.6 / seed 1: -9.0 (-) / seed 2: -6.4 / seed 3: -7.4 / seed 4: -8.3 / seed 5: -8.2 | -3.0 |
| BDG2 2017 (without the three near-zero meters) | anchored (ankyra-lite) | group 0: +0.0 / group 1: +0.4 | +0.7 |
| GoiEner non-household | carrier alone | seed 0: -1.9 / seed 1: -5.0 / seed 2: -5.6 / seed 3: +0.3 / seed 4: -4.0 / seed 5: -5.6 (-) | -1.4 |
| GoiEner non-household | anchored (ankyra-lite) | group 0: +4.3 / group 1: +4.6 | +4.7 |
| GoiEner households | carrier alone | seed 0: -5.7 (-) / seed 1: -5.2 (-) / seed 2: -5.9 (-) / seed 3: -4.4 (-) / seed 4: -5.7 (-) / seed 5: -5.1 (-) | -4.8 (-) |
| GoiEner households | anchored (ankyra-lite) | group 0: -4.9 (-) / group 1: -4.7 (-) | -4.6 (-) |
| Cambridge | carrier alone | seed 0: +12.2 * / seed 1: +13.2 * / seed 2: +11.5 * / seed 3: +14.4 * / seed 4: +13.6 * / seed 5: +12.1 * | +15.3 * |
| Cambridge | anchored (ankyra-lite) | group 0: +14.7 * / group 1: +15.6 * | +15.4 * |
| HEEW Arizona | carrier alone | seed 0: -2.7 / seed 1: -1.7 / seed 2: -8.2 (-) / seed 3: -3.2 / seed 4: -5.6 (-) / seed 5: -3.9 | +0.3 |
| HEEW Arizona | anchored (ankyra-lite) | group 0: +2.5 / group 1: +1.7 | +2.5 |
| EWELD | carrier alone | seed 0: -17.6 / seed 1: -15.5 / seed 2: -14.7 / seed 3: -16.6 / seed 4: -11.7 / seed 5: -11.5 | -13.6 |
| EWELD | anchored (ankyra-lite) | group 0: +1.5 / group 1: +1.2 | +1.5 |
| Oslo | carrier alone | seed 0: +22.7 * / seed 1: +19.2 * / seed 2: +22.1 * / seed 3: +21.5 * / seed 4: +19.3 * / seed 5: +21.7 * | +23.4 * |
| Oslo | anchored (ankyra-lite) | group 0: +23.2 * / group 1: +22.3 * | +23.0 * |
| Drammen | carrier alone | seed 0: +17.6 * / seed 1: +20.2 * / seed 2: +12.9 * / seed 3: +18.8 * / seed 4: +18.2 * / seed 5: +16.9 * | +20.2 * |
| Drammen | anchored (ankyra-lite) | group 0: +20.2 * / group 1: +20.8 * | +20.9 * |
| CINELDI | carrier alone | seed 0: +5.7 / seed 1: +5.3 / seed 2: +7.4 / seed 3: +7.5 / seed 4: +7.4 / seed 5: +7.1 | +9.3 * |
| CINELDI | anchored (ankyra-lite) | group 0: +10.1 * / group 1: +10.5 * | +10.5 * |

The carrier alone has no state rule: on the three near-zero BDG2 meters its error is not zero while some baselines' errors
are near zero, which dominates the unit-equal mean (hence the BDG2 row without those meters).

## Prediction intervals (`intervals_by_population.csv`, `intervals_winkler_contrasts.csv`, `intervals_households.json`)

The release interval (residual quantiles of the history-only estimator at the unit's pseudo-origins, independent of the
carrier) centred on ankyra-lite, against the native quantiles of TimesFM and Chronos-2; test windows. Descriptive: the
interval was not designed for this carrier.

| Population | 80% coverage (ankyra-lite / TimesFM / Chronos-2) | 90% coverage (ankyra-lite / Chronos-2) | Winkler 80 vs TimesFM | vs Chronos-2 |
|---|---|---|---|---|
| BDG2 2017 | 0.766 / 0.349 / 0.629 | 0.858 / 0.776 | -2.7 | -17.2 |
| GoiEner non-household | 0.792 / 0.430 / 0.720 | 0.876 / 0.851 | +6.2 | -9.1 (-) |
| GoiEner households | 0.782 / 0.345 / 0.680 | 0.874 / 0.815 | +6.8 * | -1.4 |
| Cambridge | 0.790 / 0.393 / 0.710 | 0.880 / 0.836 | +15.3 * | -1.1 |
| HEEW Arizona | 0.776 / 0.373 / 0.726 | 0.863 / 0.848 | +9.3 | -2.5 |
| EWELD | 0.844 / 0.606 / 0.804 | 0.894 / 0.883 | -3.7 | -39.6 (-) |
| Oslo | 0.847 / 0.379 / 0.678 | 0.920 / 0.813 | +21.0 * | +4.5 * |
| Drammen | 0.855 / 0.388 / 0.725 | 0.917 / 0.841 | +20.6 * | +3.2 |
| CINELDI | 0.818 / 0.438 / 0.736 | 0.901 / 0.859 | +15.5 * | -4.3 (-) |

## Trained carriers inside the anchoring layer (`carriers_in_anchoring_layer.csv`)

Hourly improvement in %: ankyra-lite against the same anchoring layer with each other trained carrier (`ankyra-lite vs
anchored carrier`), and each carrier's anchored forecast against the carrier alone (contrast (i)).

| Population | vs anchored LSTM | vs anchored PatchTST | vs anchored LSTM-T-KAN-M | (i) small net | (i) LSTM | (i) PatchTST | (i) LSTM-T-KAN-M |
|---|---|---|---|---|---|---|---|
| BDG2 2017 | +5.2 * | +3.6 | +7.0 * | +2.8 * | +6.9 * | +5.1 | +9.5 * |
| GoiEner non-household | +12.7 * | +5.6 * | +15.4 * | +2.8 | +12.5 * | +7.1 * | +13.1 * |
| GoiEner households | +8.0 * | +3.8 * | +9.1 * | -1.7 | +1.8 | +0.3 | +2.7 |
| Cambridge | +16.7 * | +9.2 * | +22.8 * | -0.1 | +20.3 * | +5.4 * | +32.2 * |
| HEEW Arizona | +14.0 * | +1.7 | +15.6 * | +1.4 | +20.6 * | +4.6 * | +22.2 * |
| EWELD | +15.2 * | +3.0 | +19.7 * | -0.0 | +6.4 * | +2.2 * | +7.1 * |
| Oslo | +19.9 * | +11.0 * | +22.9 * | -0.2 | +23.2 * | +7.4 * | +27.7 * |
| Drammen | +12.2 * | +5.9 * | +11.9 * | -0.2 | +14.7 * | +7.8 * | +14.5 * |
| CINELDI | +6.9 * | +2.8 * | +8.5 * | +1.3 * | +12.7 * | +6.1 * | +15.8 * |

The other carriers were trained on their own population only, with the same training cut as the small network.
On state windows the anchoring layer returns those carriers' own forecasts, whereas ankyra-lite delivers the seasonal naive.

## Contrast (ii) by error block (`contrast_ii_blocks.csv`, `contrast_ii_control.csv`)

Populations (of 9) where the first-named forecast is resolved better / not resolved / resolved worse, all test windows:

| Contrast | Hourly | Level | Daily path | Within-day |
|---|---|---|---|---|
| anchored vs history only | 9 / 0 / 0 | 5 / 4 / 0 | 6 / 3 / 0 | 8 / 1 / 0 |
| anchored vs carrier alone | 2 / 7 / 0 | 0 / 9 / 0 | 1 / 6 / 2 | 6 / 3 / 0 |
| carrier alone vs history only | 7 / 2 / 0 | 3 / 6 / 0 | 5 / 3 / 1 | 8 / 1 / 0 |
| anchored vs carrier-free control | 9 / 0 / 0 | 5 / 4 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| carrier-free control vs history only | 8 / 1 / 0 | 0 / 9 / 0 | 1 / 8 / 0 | 8 / 1 / 0 |

`anchored` is ankyra-lite. The history-only configuration and the carrier-free control (the anchoring layer called
with the history-only forecasts in the carrier's place) are saved forecasts of the same anchoring layer without a
carrier. On state windows ankyra-lite delivers the seasonal naive; the rows `off-state and micro-load windows excluded` leave those
windows out.

## History length at the origin (`history_length.csv`), all nine populations

| Stratum | Windows | vs carrier alone | vs TimesFM | vs Chronos-2-X |
|---|---:|---|---|---|
| < 10,248 h (no annual candidate) | 450 | -0.6 | -2.4 | -2.6 |
| 10,248 h - 2 years | 1715 | +1.7 | +0.4 | -2.7 |
| 2-3 years | 829 | +2.5 | +11.8 | +7.2 |
| >= 3 years | 2722 | +0.3 | +9.1 * | +5.8 |
| K_eff 6-8 | 0 | - | - | - |
| K_eff 9-11 | 451 | -0.6 | -2.3 | -2.6 |
| K_eff 12 | 5265 | +1.4 * | +3.4 | +0.4 |

## BDG2 near-zero meters (`bdg2_near_zero_sensitivity.csv`)

Mean unit rank of ankyra-lite: 4.676 on all units (142 units), 4.683 without the three near-zero meters (139 units; rank 1 of 22).

## Anchoring record (`anchoring_gain_deciles.csv`)

The two statistics of `anchoring_record` against the realised gain of ankyra-lite over its carrier alone (descriptive; the carrier weight is the anchoring layer's own weight). The deciles hold 5383 windows of 9 populations: of the 5602 pooled windows (units with a mean load above 1e-6 kW), those with a finite statistic.

| Statistic | Level gain by decile (%) | Hourly gain by decile (%) |
|---|---|---|
| `carrier_level_weight` | +9 +5 +1 +6 +5 +5 -7 -4 +3 +3 | +3 +1 +1 +1 +2 +1 +1 +0 +1 +1 |
| `history_vs_carrier_log_ratio` | +11 +9 +1 +4 +2 -1 -0 -8 +1 -1 | +3 +2 +1 +1 +1 +0 -0 +1 +1 +1 |

## Cost (`cost_per_window.csv`)

Measured on 6 Oslo test windows, CPU, one thread: carrier forecasts at 7 issue times x 6 seeds 0.135 s/window, anchoring layer 0.520 s/window, total 0.655 s/window. The carrier forecasts match the stored evaluation forecasts to a relative difference of 1.9e-06.

## Robustness to corrupted contexts (`robustness.csv`)

64 Drammen test windows (43 units), only the context corrupted, re-forecast on the CPU with the
released checkpoints (carrier forecasts at every issue time, anchoring layer, state rule, continuation layer). Change of the hourly RMS error
against the clean context (positive = worse); the comparator is the carrier alone from the same corrupted record.

| Corruption of the context | ankyra-lite: hourly error vs clean | carrier alone | level change, median / max | peak readout change, median | ankyra-lite vs carrier alone (corrupted) |
|---|---:|---:|---:|---:|---:|
| 5% of hours missing, filled by linear interpolation | +0.0% | -0.1% | +0.02% / 0.6% | -0.01% | -2.9% |
| 20% of hours missing, filled | +0.2% | +0.1% | +0.12% / 1.7% | -0.23% | -2.9% |
| a 24-hour zero-filled block in the last week | +2.3% | +2.4% | -3.46% / 7.2% | -1.45% | -3.0% |
| one spike at 10x the context maximum, 36 hours before the origin | +52.5% | +40.7% | +4.05% / 10.4% | +996.79% | +5.2% |
| context shifted by +1 hour | +10.8% | +10.8% | +0.46% / 6.2% | +0.26% | -2.9% |
| context shifted by -1 hour | +13.3% | +13.9% | -0.11% / 3.7% | -0.06% | -3.4% |
| whole history scaled by 0.5 | +0.0% | +0.0% | +0.00% / 0.0% | +0.00% | -2.9% |
| whole history scaled by 2 | +0.0% | +0.0% | +0.00% / 0.0% | +0.00% | -2.9% |

Largest deviation from exact scale equivariance: 0.00e+00. State rules fired: miss05 0, miss20 0, zero24 0, spike 0, shift_plus1 0, shift_minus1 0, scale0.5 0, scale2 0. The clean forecasts match the stored evaluation product to a relative difference of 1.2e-06. CPU time 5.4 min (0.56 s per window and corruption).

## Constants (`constants.csv`)

36 carrier rows (training recipe, span return, network; read from the package defaults) and 20 anchoring-layer rows (group, constant, value and where_fixed as in the first release; the layer
is a byte-identical copy). The `sensitivity` column holds the study's recipe readouts where a recipe constant was varied.

## Continuation layer (`continuation_layer.csv`, `continuation_record.csv`)

The origin-continuity layer on against off on the same carrier and anchoring layer (ankyra-lite after against before the
layer, seasonal naive on state windows in both), hourly error by lead time, improvement in % (`*` resolved better, `(-)`
resolved worse; unit x month interval, no seed layer: the same carrier on both sides):

| Population | lead hour 0 | lead hours 0-3 | forecast day 1 | forecast days 2-7 | forecast days 8-31 | whole window |
|---|---|---|---|---|---|---|
| BDG2 2017 | +39.2 * | +32.2 * | +19.2 * | +3.7 * | +0.2 | +0.8 * |
| GoiEner non-household | +32.4 * | +29.8 * | +12.0 * | +2.4 * | +0.3 * | +0.5 * |
| GoiEner households | +18.4 * | +13.5 * | +8.4 * | +1.7 * | +0.1 * | +0.4 * |
| Cambridge | +22.6 * | +17.7 * | +7.6 * | +0.4 | -0.1 | +0.2 |
| HEEW Arizona | +31.9 * | +27.5 * | +14.3 * | +3.1 * | +0.3 * | +0.9 * |
| EWELD | +37.0 * | +28.4 * | +7.5 * | +1.1 | +0.3 | +0.5 * |
| Oslo | +35.1 * | +30.0 * | +12.6 * | -1.0 | -0.5 (-) | -0.3 |
| Drammen | +42.8 * | +39.5 * | +25.1 * | +2.3 | -0.1 | +1.0 |
| CINELDI | +24.8 * | +23.9 * | +6.4 * | +0.2 | -0.1 | +0.0 |
| all nine populations | +28.0 * | +23.4 * | +11.1 * | +1.9 * | +0.2 * | +0.5 * |

Readout rule A: energy, level and peak are read before the layer, so the delivered trajectory's mean differs from the
energy readout (744 x level). Relative difference on the windows the layer acts on, and what the layer did:

| Population | Delivered mean vs energy readout, p50 / p99 / max | State windows bypassed | Clipped spans (fast / day) | No-match reference day | max abs Delta |
|---|---|---:|---|---:|---:|
| BDG2 2017 | 0.46% / 7.93% / 8.71% | 33 | 6.1% / 6.6% | 0 | 0.809 |
| GoiEner non-household | 1.21% / 9.00% / 89.89% | 13 | 11.6% / 7.9% | 0 | 0.809 |
| GoiEner households | 1.54% / 9.93% / 309.77% | 35 | 20.0% / 10.9% | 0 | 0.809 |
| Cambridge | 0.39% / 4.46% / 13.32% | 0 | 1.1% / 0.4% | 0 | 0.809 |
| HEEW Arizona | 0.33% / 3.30% / 6.58% | 0 | 0.5% / 0.2% | 0 | 0.809 |
| EWELD | 1.04% / 7.71% / 9.49% | 245 | 12.9% / 6.5% | 33 | 0.809 |
| Oslo | 0.88% / 5.29% / 7.57% | 0 | 0.3% / 0.3% | 0 | 0.809 |
| Drammen | 1.13% / 4.88% / 6.45% | 0 | 0.3% / 0.3% | 0 | 0.782 |
| CINELDI | 0.94% / 7.10% / 9.43% | 0 | 7.2% / 6.1% | 0 | 0.809 |
| all nine populations | 0.77% / 7.81% / 309.77% | 326 | 6.4% / 4.2% | 33 | 0.809 |

## What the handover contributes (`handover_granularity.csv`, `lead_weeks.csv`)

Four arms share the level with the carrier candidate, the delivered within-day block, the projection and the state rule; they
differ only in the weight on the carrier's daily means: none (A0), a fixed half (Ah), one weight per window from the handover
pairs (AM), the weekly weights (AW = ankyra-lite). Hourly improvement of the first-named arm in % (`*` resolved better, `(-)` resolved
worse); `share` = the part of AW's gain over A0 that the fixed half weight already gives (log-ratio share).

| Population | AW vs Ah | AW vs AM | Ah vs A0 | AW vs A0 | share (Ah) | ankyra-lite vs fixed division | mean weight on the carrier, weeks 1-4 |
|---|---|---|---|---|---|---|---|
| BDG2 2017 | +0.9 * | -0.1 | +0.7 | +1.7 * | 44% | +2.9 * | 0.56 / 0.48 / 0.54 / 0.53 |
| GoiEner non-household | -0.1 | -0.1 | +1.3 | +1.2 | 111% | +4.3 * | 0.59 / 0.57 / 0.56 / 0.56 |
| GoiEner households | +0.4 | -0.1 | +1.8 * | +2.1 * | 83% | +2.9 * | 0.56 / 0.53 / 0.55 / 0.56 |
| Cambridge | +1.2 | -0.2 | +4.6 * | +5.7 * | 80% | +7.9 * | 0.71 / 0.64 / 0.63 / 0.64 |
| HEEW Arizona | +0.9 | -0.3 | +2.9 * | +3.8 * | 76% | +6.8 * | 0.67 / 0.61 / 0.59 / 0.62 |
| EWELD | +1.0 | -0.5 | +5.3 * | +6.2 * | 85% | +8.4 * | 0.62 / 0.59 / 0.55 / 0.54 |
| Oslo | +0.4 | -0.2 | +4.9 * | +5.3 * | 92% | +8.3 * | 0.69 / 0.70 / 0.68 / 0.70 |
| Drammen | +1.0 * | -0.2 | +4.5 * | +5.5 * | 82% | +7.5 * | 0.69 / 0.76 / 0.74 / 0.72 |
| CINELDI | +0.0 | -0.1 | +1.3 * | +1.3 * | 99% | +2.7 * | 0.57 / 0.57 / 0.57 / 0.60 |

By forecast week (`lead_weeks.csv`), hourly improvement of ankyra-lite and of the fixed division over the carrier alone:

| Population | ankyra-lite vs carrier, weeks 1-4 | fixed division vs carrier, weeks 1-4 |
|---|---|---|
| BDG2 2017 | +7.3 * +4.4 * +2.7 +2.1 * | +3.1 +1.9 -3.4 (-) -1.1 |
| GoiEner non-household | +5.3 * +3.7 +3.3 +3.4 | +0.7 -1.0 -2.1 -1.6 |
| GoiEner households | -0.1 -2.9 (-) -4.5 (-) -2.5 | -6.6 (-) -9.4 (-) -10.8 (-) -7.1 (-) |
| Cambridge | +0.8 +0.7 +0.4 -0.3 | -8.0 (-) -7.6 (-) -8.4 (-) -10.0 (-) |
| HEEW Arizona | +4.7 * +1.8 +2.3 +1.9 | -4.7 -5.6 -4.8 -5.6 (-) |
| EWELD | +1.5 -3.0 -1.4 -0.5 | -6.9 -15.7 (-) -12.6 (-) -10.2 |
| Oslo | -0.5 -1.1 -0.8 +0.2 | -7.9 (-) -10.7 (-) -13.0 (-) -9.2 (-) |
| Drammen | +3.2 -0.4 +0.4 +0.4 | -3.3 -11.6 (-) -7.6 (-) -7.3 (-) |
| CINELDI | +1.4 +1.3 +1.5 +1.4 * | -0.9 -2.4 -1.7 -1.5 |

The arms are rebuilt from the anchoring layer's own fields (lite_reanchor.py, mode `fields`); AW reproduces the delivered
forecast bit for bit, and the history-side daily means before the handover are those of the layer's own level.

## Within-day anchoring: where it acts (`within_day_trust.csv`)

| Population | Anchored windows | Mean trust on the analog shape | Windows with trust > 0 | Analog shape kept |
|---|---:|---:|---:|---:|
| BDG2 2017 | 441 | 0.203 | 85% | 98% |
| GoiEner non-household | 877 | 0.169 | 76% | 98% |
| GoiEner households | 661 | 0.054 | 43% | 99% |
| Cambridge | 730 | 0.154 | 82% | 100% |
| HEEW Arizona | 658 | 0.174 | 85% | 100% |
| EWELD | 278 | 0.139 | 76% | 94% |
| Oslo | 581 | 0.131 | 76% | 100% |
| Drammen | 689 | 0.213 | 88% | 100% |
| CINELDI | 475 | 0.192 | 87% | 100% |

## Block-wise weights against one whole-window weight (`ablation.csv`, `component_contributions.csv`, `generic_combinations.csv`)

ankyra-lite against one whole-window combination weight of the same two forecasts (the history side: fixed-division daily
means plus the analog-day shape; the carrier at the origin; one least-squares weight per window from the unit's pseudo-origins,
shrunk to 1/2) and against their half-and-half average; improvement of ankyra-lite in %:

| Population | vs whole-window weight (hourly / energy) | vs half-and-half (hourly / energy) | mean weight on the carrier |
|---|---|---|---:|
| BDG2 2017 | -0.2 / -3.9 | +1.0 / +2.8 | 0.56 |
| GoiEner non-household | -0.2 / +0.9 | +1.7 / +3.3 | 0.61 |
| GoiEner households | +1.5 * / -0.1 | +4.0 * / -0.3 | 0.65 |
| Cambridge | +0.4 / +1.1 | +3.9 * / +6.4 | 0.70 |
| HEEW Arizona | +0.4 / +0.4 | +2.9 * / +7.6 | 0.68 |
| EWELD | +0.5 / +3.0 | +3.1 * / +11.4 | 0.63 |
| Oslo | +0.4 / +1.8 | +2.5 * / +4.9 | 0.73 |
| Drammen | -0.0 / +0.3 | +1.5 * / +2.7 | 0.72 |
| CINELDI | -0.2 / -1.2 | +0.3 / -0.3 | 0.63 |

Generic partners (`generic_combinations.csv`), populations (of 9) first-named resolved better / not resolved / resolved worse:

| Contrast | Hourly | Energy |
|---|---|---|
| B2(H) vs B2(SN-week) | 1 / 8 / 0 | 0 / 9 / 0 |
| B2(H) vs B2(WeekProf4) | 0 / 9 / 0 | 0 / 9 / 0 |
| B2(H) vs B2(LYR-profile) | 1 / 8 / 0 | 0 / 9 / 0 |
| B2(H) vs best B2(X) | 0 / 9 / 0 | 0 / 9 / 0 |
| B2(H) vs best equal-weight | 2 / 7 / 0 | 0 / 8 / 1 |
| ankyra-lite vs best generic | 0 / 9 / 0 | 0 / 9 / 0 |

B2(H) is the whole-window weight with the history side, B2(X) the same with a generic forecaster X in its place (X computed
from the load at every pseudo-origin), EW(X) the equal-weight average of the carrier and X; `best` was picked after scoring.

## Level candidates: hit rate of the pseudo-selected candidate (`level_candidate_hit_rate.csv`)

| Population | Seven candidates: hit rate / chance | Six historical: hit rate / chance |
|---|---|---|
| BDG2 2017 | 0.210 / 0.145 | 0.251 / 0.170 |
| GoiEner non-household | 0.206 / 0.143 | 0.225 / 0.167 |
| GoiEner households | 0.197 / 0.143 | 0.232 / 0.167 |
| Cambridge | 0.221 / 0.146 | 0.229 / 0.172 |
| HEEW Arizona | 0.217 / 0.147 | 0.280 / 0.173 |
| EWELD | 0.188 / 0.144 | 0.236 / 0.169 |
| Oslo | 0.257 / 0.143 | 0.217 / 0.167 |
| Drammen | 0.255 / 0.143 | 0.231 / 0.167 |
| CINELDI | 0.175 / 0.143 | 0.194 / 0.167 |
| all nine populations (seven (six historical and the carrier)) | 0.217 / 0.144 (difference +0.073, 95% unit bootstrap +0.061 to +0.084) | |
| all nine populations (six historical) | 0.233 / 0.169 (difference +0.064, 95% unit bootstrap +0.053 to +0.075) | |

Descriptive (P20): the candidate with the largest weight (smallest pseudo-origin error) against the candidate whose window
mean at the origin is closest to the realised mean; windows with tied weights (no evidence) are left out.

Peak readout of ankyra-lite against the same readout on the carrier alone (`peak_readout.csv`, comparator `same readout on the carrier alone`): BDG2 2017 -1.6; GoiEner non-household +5.6; GoiEner households +3.5; Cambridge -1.7; HEEW Arizona -6.9; EWELD +4.2; Oslo -2.5 (-); Drammen -2.9 (-); CINELDI +0.0.

## Training budget of the trained baselines (`training_budget_sensitivity.csv`)

ankyra-lite against DLinear, iTransformer, LSTM and PatchTST as trained for the main tables (at most 15 epochs, patience 3)
and retrained with at most 30 epochs and patience 8 (the first release's sensitivity run; cutoff, pairs, selection set,
learning-rate grid and seeds unchanged). Improvement of ankyra-lite in % (`*` resolved better, `(-)` resolved worse):

| Population | DLinear original -> relaxed | iTransformer original -> relaxed | LSTM original -> relaxed | PatchTST original -> relaxed |
|---|---|---|---|---|
| BDG2 2017 | +8.5 * -> +8.5 * | +9.8 * -> +9.8 * | +16.1 * -> +15.3 * | +10.5 * -> +10.5 * |
| Cambridge | +22.8 * -> +20.7 * | +14.0 * -> +14.0 * | +36.2 * -> +34.0 * | +14.5 * -> +13.7 * |
| HEEW Arizona | +18.7 * -> +18.7 * | +8.3 * -> +6.5 | +32.0 * -> +26.4 * | +6.3 -> +6.7 |
| EWELD | +11.5 * -> +12.6 * | +4.7 -> +6.3 | +21.4 * -> +18.3 * | +5.1 -> +5.1 |
| GoiEner non-household | +12.5 * -> +12.5 * | +2.1 -> +2.1 | +16.8 * -> +12.4 * | +10.4 * -> +10.4 * |
| GoiEner households | +9.7 * -> +10.0 * | +7.0 * -> +5.1 * | +8.5 * -> +7.1 * | +5.7 * -> +5.8 * |
| Oslo | +26.9 * -> +27.5 * | +15.9 * -> +15.9 * | +42.3 * -> +42.3 * | +15.6 * -> +15.6 * |
| Drammen | +13.9 * -> +14.5 * | +11.7 * -> +11.7 * | +29.8 * -> +22.6 * | +12.9 * -> +11.8 * |
| CINELDI | +9.5 * -> +9.1 * | +6.6 * -> +7.3 * | +25.3 * -> +15.2 * | +6.8 * -> +6.8 * |

Hourly: 35 of 36 verdicts unchanged, 0 point estimates change sign; energy: 33 of 36 unchanged, 0 sign changes. Relaxed against original (hourly): 8 better, 11 not resolved, 3 worse, 14 identical forecasts. Selected run at the 30-epoch cap: 6; selected epoch = 30: 2. ankyra-lite's rank among the 22 forecasters with the four relaxed versions substituted is unchanged on every population.

## Foundation models with a year of context (`long_context_baselines.csv`)

Zero-shot TimesFM 2.5 with an 8,736-hour context and Chronos-2 with an 8,192-hour context, scored as two further baselines
outside the main ranks. Improvement in % (`*` resolved better for the first-named, `(-)` resolved worse):

| Population | ankyra-lite vs TimesFM-8736 (hourly / energy) | ankyra-lite vs Chronos-2-8192 | TimesFM-8736 vs TimesFM | Chronos-2-8192 vs Chronos-2 |
|---|---|---|---|---|
| BDG2 2017 | -6.1 / -3.8 | +6.8 / +18.4 | -35.0 / -17.6 | -56.3 (-) / -58.2 |
| Cambridge | +12.3 * / +20.6 * | +16.6 * / +29.4 * | +3.5 * / +5.0 | +0.3 / +1.8 |
| HEEW Arizona | -2.2 / -7.9 | +4.8 / +4.3 | +4.6 / +7.2 | +0.6 / +1.9 |
| EWELD | -0.8 / -8.5 | +0.1 / +2.9 | +0.5 / +3.4 | -373.0 (-) / -397.6 (-) |
| GoiEner non-household | +0.2 / +13.5 | -0.9 / +15.5 | +4.6 * / +8.0 | +2.8 / +2.3 |
| GoiEner households | -5.3 (-) / +28.7 * | -5.2 (-) / +31.9 * | +3.4 / +9.0 * | +4.4 * / +5.2 |
| Oslo | +22.5 * / +34.9 * | +24.8 * / +37.6 * | +0.6 / -0.1 | -1.7 / -0.9 |
| Drammen | +15.3 * / +19.3 * | +16.3 * / +21.2 | +6.6 * / +12.0 * | +1.8 / +7.9 |
| CINELDI | +7.7 * / +17.6 * | +9.0 * / +20.5 * | +3.1 * / +5.8 | -0.5 / -0.9 |

## Files of the first release without a counterpart here

| File | Why there is none |
|---|---|
| `conventional_metrics.csv` | the release's file covers all windows; ankyra-lite forecasts exist on the test windows only (the earlier windows lie in or before its training period); `conventional_metrics_late.csv` holds the test windows |
| `constants_sensitivity.csv` | the release ran it on design windows without any test window; ankyra-lite forecasts exist on the test windows only, and the anchoring layer's constants are fixed in its byte-identical copy; the recipe constants the study varied are in `recipe_history.csv` and the `sensitivity` column of `constants.csv` |
| `contrast_ii_control_description.csv` | describes the calls of the carrier-free control, which does not involve the carrier (the same control as in the first release; `contrast_ii_control.csv` holds the contrasts) |
| `ankyra_2_vs_1x_by_day.csv, bdg2_micro_load_windows.csv, reevaluation_2_*.json, closeout_status.json` | records of ANKYRA's own versions and design decisions; ankyra-lite's own recipe history is `recipe_history.csv` |
| `carrier_swap*.csv, long_context_carriers.csv (the anchored rows)` | foundation models as ANKYRA's carrier are ANKYRA configurations; the two zero-shot long-context forecasts of `long_context_carriers.csv` are scored against ankyra-lite in `long_context_baselines.csv`, and trained carriers inside this anchoring layer are in `carriers_in_anchoring_layer.csv` |
| `trained_carriers_sensitivity.csv` | it measures what the earlier training cutoff of a trained carrier costs that carrier alone; ankyra-lite's carriers, and the other carriers of `carriers_in_anchoring_layer.csv`, are trained with that earlier cutoff already |
| `lcl_*, hkust_*, helsinki_*, unicon_*` | the external tests of this version are reported differently: carriers trained with the released recipe for each test set, LCL, UNICON and HKUST scored once (first read of this candidate on a previously read population) and a same-source out-of-time supplementary check (GoiEner households), all in `external_tests.csv` and docs/EXTERNAL_TESTS.md; Helsinki (retrospective) is trained and sealed, not yet scored; UCI370 was skipped |
| `ankyra_1x/, ankyra_2_0_0/, ankyra_2_0_1/, ankyra_2_1/ (version archives)` | ankyra-lite's version history is `recipe_history.csv` (the from-scratch stage and the recipe arms of the study) |

Notes of this run: CINELDI: the earlier carriers (from-scratch recipe and its earlier continuations) exist on the original CINELDI store only; not compared with a corrected-store product; CINELDI: the other trained carriers (LSTM, PatchTST, LSTM-T-KAN-M) were trained on the original CINELDI store and anchored on the corrected store (only their anchoring was re-run there); LSTM and PatchTST read load only, LSTM-T-KAN-M as a carrier also reads its calendar channels.

## Scoring-code check

Before the reference was replaced, the scoring code was checked: the late-window rows of ten release tables were
recomputed with the release's own reference forecaster and compared with the published files of the first release
repository (the peak table on all windows, as published). A table counts as reproduced when every compared field
agrees to the published rounding; only reproduced tables are written. The recomputed release numbers are not part
of any result file of this repository.

| Table | Published rows | Fields compared | Exact | Last digit | Mismatches | Result |
|---|---:|---:|---:|---:|---:|---|
| `benchmark_pairwise` | 200 | 2000 | 2000 | 0 | 0 | REPRODUCED |
| `benchmark_mean_unit_rank` | 210 | 840 | 840 | 0 | 0 | REPRODUCED |
| `conventional_metrics_late` | 250 | 3000 | 3000 | 0 | 0 | REPRODUCED |
| `energy_error` | 168 | 1680 | 1680 | 0 | 0 | REPRODUCED |
| `lead_day_metrics` | 4557 | 50127 | 50127 | 0 | 0 | REPRODUCED |
| `lead_day_energy` | 4557 | 31899 | 31899 | 0 | 0 | REPRODUCED |
| `block_shares` | 336 | 1008 | 1008 | 0 | 0 | REPRODUCED |
| `rank_tests` | 231 | 1617 | 1617 | 0 | 0 | REPRODUCED |
| `scaled_errors` | 210 | 1680 | 1680 | 0 | 0 | REPRODUCED |
| `peak_readout` | 40 | 280 | 280 | 0 | 0 | REPRODUCED |

Hourly cross-check against the earlier hourly-only comparison (20 baselines): not applicable: the earlier hourly-only comparison scored the from-scratch stage alone, not this product; the product reading is checked against the study's own scorer instead (`study_scorer_crosscheck`).

## Files

All in `results/`. The files of the main comparison have the columns of the release files of the same name plus a
trailing `flag` column; the other files are described with their columns in `results/README.md`.

| File | Content |
|---|---|
| `benchmark_pairwise.csv` | hourly contrasts against each of the 21 baselines (late windows) |
| `benchmark_mean_unit_rank.csv` | mean per-unit rank of the 22 forecasters |
| `conventional_metrics_late.csv` | RMSE, MAE, CV(RMSE), NMBE, WAPE of the 22 forecasters |
| `energy_error.csv` | monthly energy contrasts and median energy APE (BDG2 also without the three near-zero meters) |
| `lead_day_metrics.csv` | hourly error by forecast day 1-31 (six test populations) |
| `lead_day_energy.csv` | daily and cumulative energy error by forecast day (six test populations) |
| `block_shares.csv` | where the hourly squared error sits (level, daily path, within-day) for ankyra-lite, TimesFM and Chronos-2-X |
| `rank_tests.csv` | Friedman, Nemenyi and Holm-corrected Wilcoxon tests on per-unit RMSE (descriptive) |
| `scaled_errors.csv` | RMSSE and MASE with the weekly seasonal-naive in-sample scale (descriptive) |
| `peak_readout.csv` | peak readout contrasts (late windows): own trajectory maximum, previous-window peak, and for each baseline the same readout and its trajectory maximum |
| `recipe_history.csv` | the study's one reading of the recipe arms (from-scratch stage, F, R, V): ankyra-lite's version history; and the 0.2.0 study's development check and test read |
| `constants.csv` | every constant: the carrier's (training recipe, span return, network) and the anchoring layer's (as in the first release) |
| `datasets.csv` | populations, the training, validation and test split, and the returned validation span, and the carrier seeds |
| `robustness.csv` | ankyra-lite and the carrier alone on corrupted contexts (CPU re-forecast with the released checkpoints) |
| `contrast_ii_blocks.csv` | ankyra-lite against the history-only configuration and the carrier alone, by error block |
| `contrast_ii_control.csv` | ankyra-lite against the carrier-free control (the same procedure without a carrier), by error block |
| `example_window_cambridge.csv` | one Cambridge test window: load, TimesFM, ankyra-lite (`ankyra_kw`) and the carrier alone and ankyra-lite before the continuation layer |
| `cost_per_window.csv` | parameters, device and CPU seconds per window (one thread) |
| `intervals_by_population.csv` | the release interval centred on ankyra-lite and the native quantiles of TimesFM and Chronos-2 (test windows) |
| `intervals_winkler_contrasts.csv` | unit-equal Winkler contrasts of the ankyra-lite interval against the native quantiles |
| `ablation.csv` | ankyra-lite against reduced and alternative versions of itself (hourly and energy) |
| `component_contributions.csv` | what each part of ankyra-lite adds (direct contrasts) |
| `seed_spread.csv` | each seed alone against the three-seed ensemble, and each seed's contrast against TimesFM and Chronos-2-X |
| `carriers_in_anchoring_layer.csv` | ankyra-lite against other trained carriers inside the same anchoring layer |
| `history_length.csv` | ankyra-lite against the carrier alone, TimesFM and Chronos-2-X by the history length at the origin |
| `bdg2_near_zero_sensitivity.csv` | the BDG2 hourly contrasts with and without the three near-zero meters |
| `anchoring_gain_deciles.csv` | the anchoring record against the realised gain of ankyra-lite over its carrier alone (descriptive) |
| `training_curves.csv` | both training stages per population and seed: validation criterion (and training loss) by epoch, best epoch |
| `continuation_layer.csv` | the origin-continuity layer on against off on the same carrier, by lead time (hourly) and the delivered energy |
| `continuation_record.csv` | what the layer did on the test windows, and the delivered mean against the energy readout (readout rule A) |
| `example_window_cambridge.json` | the building, period, day types and source of the example window |
| `intervals_households.json` | the interval on GoiEner households: four arms and contrasts |
| `handover_granularity.csv` | the handover arms (none, a fixed half weight, one weight per window, the weekly weights) against each other, the carrier alone, TimesFM and the fixed division |
| `lead_weeks.csv` | ankyra-lite and the fixed division against the carrier alone and TimesFM by forecast week, and the mean handover weight on the carrier |
| `within_day_trust.csv` | where the within-day anchoring acts: trust on the analog-day shape, windows with a positive trust, analog shape kept |
| `generic_combinations.csv` | one whole-window combination weight of the carrier with the history side against the same with generic forecasters, and equal-weight averages |
| `level_candidate_hit_rate.csv` | how often the level candidate with the smallest pseudo-origin error is the best one at the origin, against chance |
| `energy_level_readout.csv` | the energy contrasts with ankyra-lite's level readout (744 x level) in place of its delivered trajectory's mean |
| `training_budget_sensitivity.csv` | ankyra-lite against DLinear, iTransformer, LSTM and PatchTST trained with the original budget (15 epochs, patience 3) and with the relaxed budget (30 epochs, patience 8), hourly and energy, and each relaxed model against its original |
| `long_context_baselines.csv` | ankyra-lite against TimesFM 2.5 with an 8,736-hour and Chronos-2 with an 8,192-hour context (zero-shot), and each against itself with the 1,344-hour context; outside the main ranks |
| `REPRODUCTION_CHECK.json` | the scoring-code check: every table's comparison with the published release files |
| `compare_baselines.json` | all numbers behind this summary, per population, with the product checks |
| `compare_baselines.log` | the log of the run that wrote these files |

Produced by `evaluation/compare_baselines.py` (products: `evaluation/products.json`).
