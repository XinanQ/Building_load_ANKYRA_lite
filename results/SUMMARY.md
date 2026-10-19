# ankyra-lite against the baselines

Test windows (`late` subset: origins after the training cutoff of the trained baselines) of 9 populations. The reference is **ankyra-lite** (version 0.1.0): the small carrier network
(copy-pointer structured load network, 61,613 parameters) feeding the ANKYRA 2.2.1 anchoring layer, month-ahead (744 h)
hourly forecasts. For each population the network was trained from scratch on a pooled training set (windows of several populations whose labels end before that population's validation set) with early stopping on the population's last 93 days before the training cut; then that validation span was returned to the training set and training continued for 8 epochs (the study's R recipe, the released recipe); three seeds averaged before anchoring.
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
  mean load of the delivered trajectory (the release definition, applied to every forecaster).
- **Peak**: ankyra-lite's peak readout (the envelope operator on the daily means of its delivered trajectory) against
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
| BDG2 2017 | 474 / 142 | 33 | 1 (5.35) | 12 / 0 | 4 / 0 | 4 / 0 | 4 / 0 | 17 / 0 |  |
| GoiEner non-household | 890 / 481 | 13 | 1 (5.42) | 14 / 0 | 14 / 0 | 14 / 0 | 1 / 0 | 18 / 0 |  |
| GoiEner households | 696 / 696 | 35 | 1 (5.49) | 17 / 2 | 17 / 0 | 17 / 0 | 11 / 0 | 21 / 0 |  |
| Cambridge | 730 / 108 | 0 | 1 (3.23) | 21 / 0 | 20 / 0 | 20 / 0 | 10 / 0 | 21 / 0 |  |
| HEEW Arizona | 658 / 138 | 0 | 1 (5.56) | 12 / 0 | 5 / 0 | 5 / 0 | 5 / 0 | 16 / 0 |  |
| EWELD | 523 / 197 | 245 | 1 (7.02) | 13 / 0 | 6 / 0 | 6 / 0 | 6 / 0 | 19 / 0 |  |
| Oslo | 581 / 41 | 0 | 1 (1.76) | 20 / 0 | 19 / 0 | 19 / 0 | 17 / 0 | 17 / 0 |  |
| Drammen | 689 / 43 | 0 | 1 (2.14) | 20 / 0 | 9 / 0 | 9 / 0 | 6 / 1 | 19 / 0 |  |
| CINELDI | 475 / 45 | 0 | 1 (3.36) | 20 / 0 | 11 / 0 | 11 / 0 | 3 / 0 | 19 / 0 |  |

- **Hourly.** Resolved losses: GoiEner households: TimesFM (-5.6 (-)); GoiEner households: Chronos-2-X (-3.8 (-)).
- **Energy** is the release definition (delivered trajectory). The release's energy table covers the six test
  populations and the Suzhou park; the Oslo, Drammen and CINELDI rows are additional, with the same estimator.
- **Energy (level readout)** is a sensitivity: ankyra-lite's independent energy readout (744 x its level) in place
  of the delivered trajectory's mean. The two differ only on windows where the projection onto nonnegative load
  binds: BDG2 2017 1, GoiEner non-household 142, GoiEner households 29, Cambridge 9, HEEW Arizona 0, EWELD 21, Oslo 0, Drammen 2, CINELDI 33 windows.
- **Peak (trajectory max)** counts the contrasts against each baseline's own trajectory maximum.

## By population

Improvement of ankyra-lite in % (positive = ankyra-lite better). Energy (level) = with ankyra-lite's level
readout; Peak (readout) = against the same readout on the baseline; Peak (max) = against the baseline's
trajectory maximum. The CSV files hold all 21 baselines.

### BDG2 2017 (474 windows, 142 units; rank 1 of 22, mean unit rank 5.35)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | -1.0 | +12.0 | +12.0 | +5.3 | +36.7 * |
| TimesFM-X | +3.4 | +15.9 | +15.9 | +4.0 | +36.5 * |
| Chronos-2 | +3.4 | +12.4 | +12.4 | +9.1 | +38.2 * |
| Chronos-2-X | -5.8 | +0.1 | +0.1 | +1.2 | +35.1 * |
| PatchTST | +8.9 * | +11.8 | +11.8 | +5.9 | +35.1 * |
| iTransformer-X | +8.6 * | +18.5 | +18.5 | +10.9 | +40.2 * |
| LSTM | +14.6 * | +16.3 | +16.3 | +5.1 | +36.8 * |
| GBT-T | +5.5 | +8.4 | +8.4 | +2.7 | +9.1 |
| LSTM-T-KAN-M | +11.1 * | +8.8 | +8.8 | +4.1 | +42.3 * |
| RIDGE-L | +3.3 | +7.8 | +7.8 | +6.1 | +47.7 * |
| Seasonal naive (week) | +18.0 * | +18.8 | +18.8 | +13.2 | +10.4 |

Also: peak readout vs its own trajectory maximum +34.0 *; vs the previous window's peak -2.8; energy without the three near-zero meters: TimesFM +12.0, TimesFM-X +15.9, Chronos-2 +12.4, Chronos-2-X +0.1, PatchTST +11.8, iTransformer-X +18.5, LSTM +16.3, GBT-T +8.4, LSTM-T-KAN-M +8.8, RIDGE-L +7.8, SN-week +18.8.

### GoiEner non-household (890 windows, 481 units; rank 1 of 22, mean unit rank 5.42)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +2.8 | +19.7 * | +20.0 * | +0.1 | +41.3 * |
| TimesFM-X | +3.3 | +17.2 * | +17.5 * | +2.0 | +39.2 * |
| Chronos-2 | +0.6 | +17.2 | +17.5 | +6.1 | +27.7 * |
| Chronos-2-X | -2.4 | +11.9 | +12.1 | +2.7 | +35.0 * |
| PatchTST | +8.7 * | +13.9 | +14.2 | +4.3 | +45.4 * |
| iTransformer-X | +3.3 | +16.1 * | +16.3 * | -3.6 | +33.8 * |
| LSTM | +15.1 * | +19.6 * | +19.9 * | +0.4 | +42.6 * |
| GBT-T | +27.9 * | +44.1 * | +44.3 * | -1.8 | +60.1 * |
| LSTM-T-KAN-M | +13.6 * | +19.4 * | +19.6 * | -0.9 | +41.0 * |
| RIDGE-L | +5.1 | +7.4 | +7.7 | -12.4 | +32.0 * |
| Seasonal naive (week) | +20.0 * | +15.2 | +15.5 | -9.4 | -24.5 |

Also: peak readout vs its own trajectory maximum +43.7 *; vs the previous window's peak -35.8.

### GoiEner households (696 windows, 696 units; rank 1 of 22, mean unit rank 5.49)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | -5.6 (-) | +36.4 * | +36.5 * | +11.0 * | +66.1 * |
| TimesFM-X | -1.9 | +32.6 * | +32.7 * | +10.9 * | +66.7 * |
| Chronos-2 | -3.8 | +37.1 * | +37.2 * | +11.8 * | +66.6 * |
| Chronos-2-X | -3.8 (-) | +36.4 * | +36.5 * | +12.5 * | +66.1 * |
| PatchTST | +4.8 * | +22.2 * | +22.4 * | +7.6 * | +67.8 * |
| iTransformer-X | +5.9 * | +20.8 * | +21.0 * | +8.0 * | +67.2 * |
| LSTM | +7.7 * | +21.2 * | +21.3 * | +4.2 | +68.6 * |
| GBT-T | +10.4 * | +25.5 * | +25.6 * | +6.7 | +58.2 * |
| LSTM-T-KAN-M | +7.7 * | +23.3 * | +23.5 * | +3.4 | +68.9 * |
| RIDGE-L | +3.4 * | +7.2 | +7.4 | +5.8 | +65.9 * |
| Seasonal naive (week) | +19.9 * | +11.8 | +12.0 | +7.3 | +30.9 * |

Also: peak readout vs its own trajectory maximum +67.9 *; vs the previous window's peak +3.2.

### Cambridge (730 windows, 108 units; rank 1 of 22, mean unit rank 3.23)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +14.2 * | +23.1 * | +23.2 * | +10.4 | +41.7 * |
| TimesFM-X | +15.6 * | +25.1 * | +25.1 * | +9.7 | +41.0 * |
| Chronos-2 | +15.6 * | +29.4 * | +29.4 * | +10.1 | +42.9 * |
| Chronos-2-X | +8.7 * | +18.2 * | +18.2 * | +4.9 | +40.4 * |
| PatchTST | +13.3 * | +18.8 * | +18.9 * | +4.8 | +36.6 * |
| iTransformer-X | +7.7 * | +6.2 | +6.3 | +3.5 | +36.1 * |
| LSTM | +35.3 * | +46.1 * | +46.1 * | +20.1 * | +45.9 * |
| GBT-T | +11.4 * | +21.3 * | +21.3 * | +9.5 | +30.5 * |
| LSTM-T-KAN-M | +44.4 * | +59.7 * | +59.7 * | +27.2 * | +46.9 * |
| RIDGE-L | +12.3 * | +21.7 * | +21.7 * | +9.2 * | +46.2 * |
| Seasonal naive (week) | +26.7 * | +30.6 * | +30.6 * | +10.8 | +27.0 * |

Also: peak readout vs its own trajectory maximum +34.1 *; vs the previous window's peak +5.6.

### HEEW Arizona (658 windows, 138 units; rank 1 of 22, mean unit rank 5.56)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +1.1 | -1.6 | -1.6 | -1.7 | +23.1 |
| TimesFM-X | +3.3 | +2.1 | +2.1 | +2.0 | +33.5 * |
| Chronos-2 | +4.0 | +4.7 | +4.7 | -2.6 | +29.4 * |
| Chronos-2-X | -0.7 | -2.4 | -2.4 | -0.5 | +26.9 * |
| PatchTST | +4.9 | +3.3 | +3.3 | -1.5 | +32.9 * |
| iTransformer-X | +3.7 | -2.2 | -2.2 | -0.8 | +33.2 * |
| LSTM | +31.0 * | +33.0 * | +33.0 * | +18.0 * | +49.5 * |
| GBT-T | +4.0 | +1.6 | +1.6 | -3.2 | +13.6 |
| LSTM-T-KAN-M | +29.5 * | +37.3 * | +37.3 * | +26.6 * | +44.8 * |
| RIDGE-L | +2.8 | +0.1 | +0.1 | +3.0 | +39.6 * |
| Seasonal naive (week) | +15.7 * | +5.8 | +5.8 | +2.2 | +9.7 |

Also: peak readout vs its own trajectory maximum +33.1 *; vs the previous window's peak +9.3.

### EWELD (523 windows, 197 units; rank 1 of 22, mean unit rank 7.02)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +1.4 | -5.3 | -5.4 | +5.7 | +33.9 * |
| TimesFM-X | +2.0 | -4.7 | -4.8 | +0.2 | +37.1 * |
| Chronos-2 | -0.4 | -4.2 | -4.3 | +3.6 | +42.8 * |
| Chronos-2-X | +4.0 | +14.7 | +14.7 | +3.5 | +40.6 * |
| PatchTST | +5.0 | -8.0 | -8.1 | +7.8 | +38.5 * |
| iTransformer-X | +3.2 | +2.7 | +2.6 | +2.6 | +36.2 * |
| LSTM | +21.2 * | +35.8 * | +35.7 * | +27.9 * | +61.6 * |
| GBT-T | +28.9 * | +22.7 | +22.6 | +32.3 * | +71.0 * |
| LSTM-T-KAN-M | +18.5 * | +35.3 * | +35.2 * | +26.5 * | +58.1 * |
| RIDGE-L | +3.3 | +2.7 | +2.6 | +6.8 | +47.6 * |
| Seasonal naive (week) | +15.0 * | -2.7 | -2.7 | -0.7 | -30.9 |

Also: peak readout vs its own trajectory maximum +48.4 *; vs the previous window's peak -20.2.

### Oslo (581 windows, 41 units; rank 1 of 22, mean unit rank 1.76)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +23.4 * | +35.4 * | +35.4 * | +17.1 * | +39.1 * |
| TimesFM-X | +25.4 * | +39.2 * | +39.2 * | +17.5 * | +37.9 * |
| Chronos-2 | +23.9 * | +37.6 * | +37.6 * | +16.1 * | +42.1 * |
| Chronos-2-X | +15.0 * | +21.0 * | +21.0 * | +7.9 | +32.3 * |
| PatchTST | +16.1 * | +24.2 * | +24.2 * | +10.7 * | +19.1 |
| iTransformer-X | +10.6 * | +10.9 | +10.9 | +1.5 | +16.1 |
| LSTM | +42.6 * | +54.2 * | +54.2 * | +31.1 * | +54.7 * |
| GBT-T | +3.4 | +6.7 | +6.7 | -3.3 | -16.5 |
| LSTM-T-KAN-M | +40.3 * | +53.7 * | +53.7 * | +30.5 * | +49.8 * |
| RIDGE-L | +24.1 * | +33.3 * | +33.3 * | +17.1 * | +46.1 * |
| Seasonal naive (week) | +36.7 * | +45.0 * | +45.0 * | +19.2 * | +39.3 * |

Also: peak readout vs its own trajectory maximum +18.1; vs the previous window's peak +27.4 *.

### Drammen (689 windows, 43 units; rank 1 of 22, mean unit rank 2.14)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +18.9 * | +25.7 * | +25.7 * | +14.3 | +40.3 * |
| TimesFM-X | +14.5 * | +19.3 | +19.3 | +11.7 | +37.1 * |
| Chronos-2 | +15.8 * | +24.0 | +24.0 | +12.1 | +39.4 * |
| Chronos-2-X | +5.0 | -0.1 | -0.1 | -1.4 | +31.0 * |
| PatchTST | +10.8 * | +4.5 | +4.5 | -4.9 | +19.5 * |
| iTransformer-X | +9.6 * | +4.7 | +4.7 | -4.3 | +22.8 * |
| LSTM | +28.1 * | +19.4 | +19.4 | +18.2 * | +46.0 * |
| GBT-T | +9.1 * | +9.8 | +9.8 | -10.9 (-) | +9.2 |
| LSTM-T-KAN-M | +24.3 * | +20.1 | +20.1 | +16.1 * | +46.3 * |
| RIDGE-L | +7.4 * | -9.8 | -9.8 | +4.0 | +41.5 * |
| Seasonal naive (week) | +27.2 * | +26.8 * | +26.8 * | +12.7 | +24.5 * |

Also: peak readout vs its own trajectory maximum +30.8 *; vs the previous window's peak +15.4.

### CINELDI (475 windows, 45 units; rank 1 of 22, mean unit rank 3.36)

| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |
|---|---|---|---|---|---|
| TimesFM | +9.8 * | +21.2 * | +21.2 * | +5.2 | +51.8 * |
| TimesFM-X | +8.2 * | +19.0 | +19.1 | +3.8 | +48.3 * |
| Chronos-2 | +7.8 * | +18.6 | +18.6 | +6.1 | +47.8 * |
| Chronos-2-X | +6.0 | +15.4 | +15.5 | +5.8 | +48.4 * |
| PatchTST | +6.1 * | +7.2 | +7.2 | +1.0 | +43.9 * |
| iTransformer-X | +6.7 * | +11.9 | +11.9 | +0.9 | +46.2 * |
| LSTM | +24.7 * | +27.0 * | +27.0 * | +9.4 | +58.5 * |
| GBT-T | +12.4 * | +24.2 * | +24.2 * | +5.0 | +39.4 * |
| LSTM-T-KAN-M | +18.1 * | +13.1 | +13.1 | +3.1 | +51.4 * |
| RIDGE-L | +5.0 * | +8.4 | +8.4 | +5.3 | +52.8 * |
| Seasonal naive (week) | +24.0 * | +22.6 * | +22.6 * | +6.2 | +26.2 |

Also: peak readout vs its own trajectory maximum +48.8 *; vs the previous window's peak -20.4 (-).

## Rank tests, six test populations pooled

1762 units, Friedman p < 1e-300, Nemenyi critical difference 0.786; ankyra-lite's mean rank
5.496 (position 1 of 22). Holm-corrected Wilcoxon: significantly better than 21 of 21 baselines, significantly worse than 0. Descriptive.

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

| Population | From-scratch stage: best epoch (seeds 0, 1, 2) | Epochs run | Span return: epochs | Returned windows | Start criterion reproduced |
|---|---|---|---|---|---|
| BDG2 2017 | 12, 9, 27 | 17, 14, 30 | 8, 8, 8 | 5042, 5042, 5042 | True |
| GoiEner non-household | 15, 3, 16 | 20, 8, 21 | 8, 8, 8 | 5381, 5381, 5381 | True |
| GoiEner households | 15, 16, 5 | 20, 21, 10 | 8, 8, 8 | 6292, 6292, 6292 | True |
| Cambridge | 16, 12, 16 | 21, 17, 21 | 8, 8, 8 | 312, 312, 312 | True |
| HEEW Arizona | 27, 20, 23 | 30, 25, 28 | 8, 8, 8 | 749, 749, 749 | True |
| EWELD | 10, 9, 7 | 15, 14, 12 | 8, 8, 8 | 1969, 1969, 1969 | True |
| Oslo | 14, 9, 30 | 19, 14, 30 | 8, 8, 8 | 304, 304, 304 | True |
| Drammen | 21, 4, 23 | 26, 9, 28 | 8, 8, 8 | 396, 396, 396 | True |
| CINELDI | 14, 6, 6 | 19, 11, 11 | 8, 8, 8 | 389, 389, 389 | True |

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

The product reading reproduces the study's own scorer on 9 of 9 populations: the per-population contrasts against the plain continuation and against the from-scratch stage equal those of the score
record (largest difference 0.0e+00 percentage points).

## Ablation (`ablation.csv`, `component_contributions.csv`)

ankyra-lite against versions of itself, hourly error, improvement of ankyra-lite in % (`*` resolved better,
`(-)` resolved worse). `-` = not available in this run.

| Version | BDG2 2017 | GoiEner non-household | GoiEner households | Cambridge | HEEW Arizona | EWELD | Oslo | Drammen | CINELDI |
|---|---|---|---|---|---|---|---|---|---|
| carrier alone (no anchoring layer) | +3.2 * | +3.5 | -1.0 | +0.3 | +1.4 | +1.2 | +0.0 | +0.4 | +2.6 * |
| state rule off (the anchoring layer's own return on state windows) | +0.2 | -0.0 | +0.3 | 0 (identical) | 0 (identical) | +2.6 | 0 (identical) | 0 (identical) | 0 (identical) |
| single seed 0 (its carrier anchored alone) | +0.6 | -0.5 | +0.7 * | +0.9 | +0.5 | +0.4 | +0.6 | +1.7 * | +0.9 |
| single seed 1 (its carrier anchored alone) | +2.5 * | +1.2 | -0.0 | +1.5 * | +0.7 | +0.5 | +3.7 * | -1.5 | +1.2 * |
| single seed 2 (its carrier anchored alone) | -0.6 | +1.6 | +0.4 | +2.2 * | +4.5 * | +0.8 | +1.2 | +3.9 * | +0.4 |
| anchoring layer without the within-day anchoring | +0.9 * | +3.2 * | -0.1 | +0.5 * | +0.9 * | -0.0 | +0.5 * | +1.0 * | +1.2 * |
| within-day trust per lead block instead of one value per window | +0.0 | +0.2 | +0.1 | +0.1 | +0.1 * | -0.0 | +0.0 | +0.1 | +0.0 |
| anchoring layer without the short-gap rule | +0.2 | -0.0 | +0.2 | +0.0 | -0.1 | +0.1 | 0 (identical) | -0.0 | 0 (identical) |
| anchoring layer without the micro-load rule | +0.1 | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) | 0 (identical) |
| history-only configuration (anchoring layer without a carrier) | +6.4 * | +19.5 * | +11.1 * | +13.8 * | +12.6 * | +16.6 * | +20.5 * | +15.3 * | +5.4 * |
| recipe: from-scratch stage alone, no continuation (anchored) | +1.2 | +2.7 | +1.4 * | +1.7 * | +0.9 * | +1.1 | +2.2 * | -2.3 | +0.8 |
| recipe: 8 continuation epochs, validation set held out, no span return (anchored) | +0.5 | +2.2 | +0.7 | +0.5 | -0.2 | -0.3 | +0.1 | -2.9 (-) | +0.3 |
| recipe: continuation with a proportional validation set and early stopping (anchored) | +1.2 | +1.1 | +0.1 | +0.2 | -0.2 | +1.1 | +0.4 | -2.5 (-) | +0.6 |
| earlier carrier: structured network without the copy pointer, weekly training windows (anchored) | +3.0 * | +7.0 * | +1.1 | +10.4 * | +4.3 * | +2.7 * | +5.9 * | +1.9 | - |
| carrier without the copy pointer, from-scratch recipe (anchored) | +1.2 | +2.2 | +1.5 | +3.7 * | +2.3 * | +1.6 | +2.6 * | -0.3 | - |
| earlier continuation study: unweighted loss (anchored) | +1.0 | +1.6 | +0.6 * | +0.4 | +0.9 | +0.9 | +0.4 | -0.9 | - |
| earlier continuation study: weighted loss (anchored) | -0.1 | +0.9 | +0.0 | -0.3 | -0.4 | +1.7 | +1.2 | -0.6 | - |
| earlier continuation study: weighted loss and warm-up (anchored) | +0.1 | +1.2 | +0.1 | +0.2 | -1.3 (-) | +1.9 | +2.0 | -0.6 | - |

The rows of switched-off parts of the anchoring layer need re-anchoring (`evaluation/lite_reanchor.py`); the single
seeds are the stored single-seed products; the history-only configuration, the recipe arms and the earlier carriers are
saved study forecasts. The earlier carriers (without the copy pointer, the earlier continuation study, the structured
network) were trained with the from-scratch recipe or its earlier continuations, not with the span return.

## Seeds (`seed_spread.csv`)

Hourly improvement against TimesFM, per seed and for the three-seed ensemble (carrier alone, and anchored):

| Population | Stage | Seed 0 | Seed 1 | Seed 2 | Ensemble |
|---|---|---|---|---|---|
| BDG2 2017 | carrier alone | -71.3 (-) | -76.8 (-) | -71.6 (-) | -69.8 (-) |
| BDG2 2017 | anchored (ankyra-lite) | -1.6 | -3.6 | -0.4 | -1.0 |
| BDG2 2017 (without the three near-zero meters) | carrier alone | -5.6 | -9.3 (-) | -4.9 | -4.4 |
| BDG2 2017 (without the three near-zero meters) | anchored (ankyra-lite) | -1.6 | -3.6 | -0.4 | -1.0 |
| GoiEner non-household | carrier alone | -2.9 | -5.6 | -8.1 | -3.7 |
| GoiEner non-household | anchored (ankyra-lite) | +3.3 | +1.7 | +1.2 | +2.8 |
| GoiEner households | carrier alone | -8.6 (-) | -6.3 (-) | -7.1 (-) | -7.0 (-) |
| GoiEner households | anchored (ankyra-lite) | -6.3 (-) | -5.6 (-) | -6.0 (-) | -5.6 (-) |
| Cambridge | carrier alone | +12.7 * | +12.1 * | +10.6 * | +13.9 * |
| Cambridge | anchored (ankyra-lite) | +13.4 * | +12.8 * | +12.3 * | +14.2 * |
| HEEW Arizona | carrier alone | -2.7 | -3.2 | -8.7 (-) | -0.3 |
| HEEW Arizona | anchored (ankyra-lite) | +0.6 | +0.4 | -3.5 | +1.1 |
| EWELD | carrier alone | -21.2 (-) | -17.6 | -14.8 | -17.0 |
| EWELD | anchored (ankyra-lite) | +0.9 | +0.8 | +0.5 | +1.4 |
| Oslo | carrier alone | +22.4 * | +19.5 * | +21.6 * | +23.4 * |
| Oslo | anchored (ankyra-lite) | +23.0 * | +20.5 * | +22.4 * | +23.4 * |
| Drammen | carrier alone | +16.5 * | +19.9 * | +13.3 * | +18.6 * |
| Drammen | anchored (ankyra-lite) | +17.5 * | +20.2 * | +15.7 * | +18.9 * |
| CINELDI | carrier alone | +4.8 | +4.2 | +6.4 | +7.4 * |
| CINELDI | anchored (ankyra-lite) | +9.0 * | +8.7 * | +9.4 * | +9.8 * |

The carrier alone has no state rule: on the three near-zero BDG2 meters its error is not zero while some baselines' errors
are near zero, which dominates the unit-equal mean (hence the BDG2 row without those meters).

## Prediction intervals (`intervals_by_population.csv`, `intervals_winkler_contrasts.csv`, `intervals_households.json`)

The release interval (residual quantiles of the history-only estimator at the unit's pseudo-origins, independent of the
carrier) centred on ankyra-lite, against the native quantiles of TimesFM and Chronos-2; test windows. Descriptive: the
interval was not designed for this carrier.

| Population | 80% coverage (ankyra-lite / TimesFM / Chronos-2) | 90% coverage (ankyra-lite / Chronos-2) | Winkler 80 vs TimesFM | vs Chronos-2 |
|---|---|---|---|---|
| BDG2 2017 | 0.758 / 0.349 / 0.629 | 0.850 / 0.776 | -3.9 | -18.6 |
| GoiEner non-household | 0.783 / 0.430 / 0.720 | 0.868 / 0.851 | +5.5 | -9.9 (-) |
| GoiEner households | 0.777 / 0.345 / 0.680 | 0.870 / 0.815 | +6.3 * | -1.9 |
| Cambridge | 0.791 / 0.393 / 0.710 | 0.881 / 0.836 | +15.2 * | -1.3 |
| HEEW Arizona | 0.779 / 0.373 / 0.726 | 0.865 / 0.848 | +9.4 * | -2.4 |
| EWELD | 0.841 / 0.606 / 0.804 | 0.893 / 0.883 | -4.0 | -40.0 (-) |
| Oslo | 0.851 / 0.379 / 0.678 | 0.921 / 0.813 | +21.2 * | +4.7 * |
| Drammen | 0.848 / 0.388 / 0.725 | 0.915 / 0.841 | +20.4 * | +2.9 |
| CINELDI | 0.812 / 0.438 / 0.736 | 0.896 / 0.859 | +15.2 * | -4.7 (-) |

## Trained carriers inside the anchoring layer (`carriers_in_anchoring_layer.csv`)

Hourly improvement in %: ankyra-lite against the same anchoring layer with each other trained carrier (`ankyra-lite vs
anchored carrier`), and each carrier's anchored forecast against the carrier alone (contrast (i)).

| Population | vs anchored LSTM | vs anchored PatchTST | vs anchored LSTM-T-KAN-M | (i) small net | (i) LSTM | (i) PatchTST | (i) LSTM-T-KAN-M |
|---|---|---|---|---|---|---|---|
| BDG2 2017 | +4.3 * | +2.7 | +6.1 * | +3.2 * | +6.9 * | +5.1 | +9.5 * |
| GoiEner non-household | +11.5 * | +4.3 | +14.2 * | +3.5 | +12.5 * | +7.1 * | +13.1 * |
| GoiEner households | +7.5 * | +3.3 * | +8.6 * | -1.0 | +1.8 | +0.3 | +2.7 |
| Cambridge | +15.7 * | +8.1 * | +21.8 * | +0.3 | +20.3 * | +5.4 * | +32.2 * |
| HEEW Arizona | +13.5 * | +1.1 | +15.1 * | +1.4 | +20.6 * | +4.6 * | +22.2 * |
| EWELD | +15.4 * | +3.3 | +20.0 * | +1.2 | +6.4 * | +2.2 * | +7.1 * |
| Oslo | +20.0 * | +11.2 * | +23.0 * | +0.0 | +23.2 * | +7.4 * | +27.7 * |
| Drammen | +10.9 * | +4.6 * | +10.7 * | +0.4 | +14.7 * | +7.8 * | +14.5 * |
| CINELDI | +6.2 * | +2.0 * | +7.7 * | +2.6 * | +12.7 * | +6.1 * | +15.8 * |

The other carriers were trained on their own population only, with the same training cut as the small network.
On state windows the anchoring layer returns those carriers' own forecasts, whereas ankyra-lite delivers the seasonal naive.

## Contrast (ii) by error block (`contrast_ii_blocks.csv`, `contrast_ii_control.csv`)

Populations (of 9) where the first-named forecast is resolved better / not resolved / resolved worse, all test windows:

| Contrast | Hourly | Level | Daily path | Within-day |
|---|---|---|---|---|
| anchored vs history only | 9 / 0 / 0 | 4 / 5 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| anchored vs carrier alone | 2 / 7 / 0 | 1 / 8 / 0 | 1 / 6 / 2 | 7 / 2 / 0 |
| carrier alone vs history only | 6 / 3 / 0 | 2 / 7 / 0 | 5 / 3 / 1 | 7 / 2 / 0 |
| anchored vs carrier-free control | 9 / 0 / 0 | 4 / 5 / 0 | 5 / 4 / 0 | 7 / 2 / 0 |
| carrier-free control vs history only | 8 / 1 / 0 | 0 / 9 / 0 | 1 / 8 / 0 | 8 / 1 / 0 |

`anchored` is ankyra-lite. The history-only configuration and the carrier-free control (the anchoring layer called
with the history-only forecasts in the carrier's place) are saved forecasts of the same anchoring layer without a
carrier. On state windows ankyra-lite delivers the seasonal naive; the rows `off-state and micro-load windows excluded` leave those
windows out.

## History length at the origin (`history_length.csv`), all nine populations

| Stratum | Windows | vs carrier alone | vs TimesFM | vs Chronos-2-X |
|---|---:|---|---|---|
| < 10,248 h (no annual candidate) | 450 | -0.6 | -3.7 | -3.9 |
| 10,248 h - 2 years | 1715 | +1.8 * | -1.3 | -4.4 |
| 2-3 years | 829 | +2.6 | +10.8 | +6.1 |
| >= 3 years | 2722 | +0.3 | +8.3 * | +5.0 |
| K_eff 6-8 | 0 | - | - | - |
| K_eff 9-11 | 451 | -0.5 | -3.6 | -3.9 |
| K_eff 12 | 5265 | +1.6 * | +2.0 | -1.0 |

## BDG2 near-zero meters (`bdg2_near_zero_sensitivity.csv`)

Mean unit rank of ankyra-lite: 5.345 on all units (142 units), 5.367 without the three near-zero meters (139 units; rank 1 of 22).

## Anchoring record (`anchoring_gain_deciles.csv`)

The two statistics of `anchoring_record` against the realised gain of ankyra-lite over its carrier alone (descriptive; the carrier weight is the anchoring layer's own weight). The deciles hold 5383 windows of 9 populations: of the 5602 pooled windows (units with a mean load above 1e-6 kW), those with a finite statistic.

| Statistic | Level gain by decile (%) | Hourly gain by decile (%) |
|---|---|---|
| `carrier_level_weight` | +7 +4 -2 +3 +6 +1 +2 +1 -1 -1 | +2 +2 +0 +1 +1 +1 +1 +2 +1 +0 |
| `history_vs_carrier_log_ratio` | +10 +8 +3 -0 -1 -0 -6 -1 +5 -4 | +3 +3 +1 +0 +0 +1 -1 +0 +2 +1 |

## Cost (`cost_per_window.csv`)

Measured on 6 Oslo test windows, CPU, one thread: carrier forecasts at 7 issue times x 3 seeds 0.075 s/window, anchoring layer 0.509 s/window, total 0.584 s/window. The carrier forecasts match the stored evaluation forecasts to a relative difference of 2.0e-06.

## Robustness to corrupted contexts (`robustness.csv`)

64 Drammen test windows (43 units), only the context corrupted, re-forecast on the CPU with the
released checkpoints (carrier forecasts at every issue time, anchoring layer, state rule). Change of the hourly RMS error
against the clean context (positive = worse); the comparator is the carrier alone from the same corrupted record.

| Corruption of the context | ankyra-lite: hourly error vs clean | carrier alone | level change, median / max | peak readout change, median | ankyra-lite vs carrier alone (corrupted) |
|---|---:|---:|---:|---:|---:|
| 5% of hours missing, filled by linear interpolation | -0.0% | -0.1% | +0.03% / 0.6% | +0.01% | -2.0% |
| 20% of hours missing, filled | +0.1% | -0.1% | +0.14% / 2.3% | -0.24% | -2.0% |
| a 24-hour zero-filled block in the last week | +3.2% | +3.0% | -3.44% / 7.0% | -1.25% | -2.0% |
| one spike at 10x the context maximum, 36 hours before the origin | +41.1% | +23.6% | +3.67% / 12.5% | +998.39% | +11.8% |
| context shifted by +1 hour | +9.4% | +9.2% | +0.32% / 5.3% | +0.24% | -1.9% |
| context shifted by -1 hour | +11.7% | +12.7% | +0.22% / 5.1% | +0.05% | -3.1% |
| whole history scaled by 0.5 | +0.0% | +0.0% | +0.00% / 0.0% | +0.00% | -2.2% |
| whole history scaled by 2 | +0.0% | +0.0% | +0.00% / 0.0% | +0.00% | -2.2% |

Largest deviation from exact scale equivariance: 0.00e+00. State rules fired: miss05 0, miss20 0, zero24 0, spike 0, shift_plus1 0, shift_minus1 0, scale0.5 0, scale2 0. The clean forecasts match the stored evaluation product to a relative difference of 2.3e-06. CPU time 4.9 min (0.51 s per window and corruption).

## Constants (`constants.csv`)

23 carrier rows (training recipe, span return, network; read from the package defaults) and 20 anchoring-layer rows (group, constant, value and where_fixed as in the first release; the layer
is a byte-identical copy). The `sensitivity` column holds the study's recipe readouts where a recipe constant was varied.

## Files of the first release without a counterpart here

| File | Why there is none |
|---|---|
| `conventional_metrics.csv` | the release's file covers all windows; ankyra-lite forecasts exist on the test windows only (the earlier windows lie in or before its training period); `conventional_metrics_late.csv` holds the test windows |
| `handover_granularity.csv` | needs other handover rules inside the anchoring layer, which must stay byte-identical to ANKYRA 2.2.1 |
| `constants_sensitivity.csv` | the release ran it on design windows without any test window; ankyra-lite forecasts exist on the test windows only; the recipe constants the study varied are in `recipe_history.csv` and the `sensitivity` column of `constants.csv` |
| `contrast_ii_control_description.csv` | describes the calls of the carrier-free control, which does not involve the carrier (the same control as in the first release; `contrast_ii_control.csv` holds the contrasts) |
| `lead_weeks_first_read.csv, ankyra_2_vs_1x_by_day.csv, bdg2_micro_load_windows.csv, carrier_swap*.csv, long_context_carriers.csv, generic_combinations.csv, trained_carriers_sensitivity.csv, training_budget_sensitivity.csv` | records of ANKYRA's own versions, carriers or baselines; the comparable ankyra-lite content is in `ablation.csv`, `carriers_in_anchoring_layer.csv`, `seed_spread.csv` and `recipe_history.csv` |
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

Product check against the study's own scorer of the recipe arms: 9 of 9 populations reproduce its per-population contrasts (largest difference 0.0e+00 percentage points).

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
| `recipe_history.csv` | the study's one reading of the recipe arms (from-scratch stage, F, R, V): ankyra-lite's version history |
| `constants.csv` | every constant: the carrier's (training recipe, span return, network) and the anchoring layer's (as in the first release) |
| `datasets.csv` | populations, the training, validation and test split, and the returned validation span |
| `robustness.csv` | ankyra-lite and the carrier alone on corrupted contexts (CPU re-forecast with the released checkpoints) |
| `contrast_ii_blocks.csv` | ankyra-lite against the history-only configuration and the carrier alone, by error block |
| `contrast_ii_control.csv` | ankyra-lite against the carrier-free control (the same procedure without a carrier), by error block |
| `example_window_cambridge.csv` | one Cambridge test window: load, TimesFM, ankyra-lite (`ankyra_kw`) and the carrier alone |
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
| `example_window_cambridge.json` | the building, period, day types and source of the example window |
| `intervals_households.json` | the interval on GoiEner households: four arms and contrasts |
| `REPRODUCTION_CHECK.json` | the scoring-code check: every table's comparison with the published release files |
| `compare_baselines.json` | all numbers behind this summary, per population, with the product checks |
| `compare_baselines.log` | the log of the run that wrote these files |

Produced by `evaluation/compare_baselines.py` (products: `evaluation/products.json`).
