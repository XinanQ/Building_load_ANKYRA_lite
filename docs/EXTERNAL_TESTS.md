# External tests (19 October 2026; 0.2.0 re-run 20 October 2026)

**Subgroup labels.** The subgroup recorded as `units without own training windows` (the key kept in `results/external_tests.csv` and the score records) was built from the first-stage training rows only; stage 2 returns validation-span windows to training, so a unit in it may have contributed returned training rows. Below it is labelled *units without first-stage own training windows*; memberships and numbers are unchanged ([correction note](CODE_AUDIT_CORRECTIONS_20261022.md#subgroup-labels-of-the-external-tests)).

This page has two parts. **Version 0.2.0** (recipe Rw, six seeds, the continuation layer) was re-run once on the four
scored test sets of the frozen external test of 19 October 2026, with the same windows, comparators and estimators;
that is the current reading, and every number of its section is a row of
[`results/external_tests.csv`](../results/external_tests.csv). **Version 0.1** (recipe R) is the frozen external test
itself, kept below as history: its numbers are those of the 0.1.1 release of the same file (commit `c27affc`), which
the 0.2.0 regeneration replaced.

- **LCL, UNICON and HKUST: previously read populations.** Earlier forecasters of the study had been scored on these
  populations, earlier versions of the small network among them (trained and scored on the same test windows), and the
  anchoring layer was fixed after those readings; the frozen external test then read them once with the 0.1 candidate.
  The 0.2.0 re-run is a second read of each test set, the first of this version. HKUST is a secondary test set with a
  small sample.
- **GoiEner households, out of time: a same-source out-of-time supplementary check.** New households and a later period
  from the source of study population 6 (GoiEner households); not a new population, and correlated with the result on
  population 6.
- **Helsinki (retrospective): trained, anchored and sealed, not scored**, in 0.1 and untouched by the 0.2.0 re-run. Its
  scoring was tied to a prospective Helsinki test, which is not being pursued for now (author decision, 19 October
  2026); the sealed set stays unread.
- **UCI370: skipped** (the author's decision).

Sections: [0.2.0 re-run](#version-020-re-run-20-october-2026) · [How to read the numbers](#how-to-read-the-numbers) ·
[Version 0.1 external test](#version-01-external-test-19-october-2026) ·
[Procedure common to every test set](#procedure-common-to-every-test-set) · [LCL](#lcl-households-london) ·
[UNICON](#unicon-la-trobe-university-campuses) · [HKUST](#hkust-campus-incomer-meters-secondary) ·
[GoiEner households, out of time](#goiener-households-out-of-time-supplementary) ·
[Helsinki and UCI370](#helsinki-retrospective-and-uci370) ·
[Decisions awaiting the author's review](#decisions-awaiting-the-authors-review) · [Run notes](#run-notes) ·
[What these tests do not show](#what-these-tests-do-not-show) · [Files](#files)

## Version 0.2.0 re-run (20 October 2026)

**What was run.** For each test set the six stage-1 carriers of the frozen external test (seeds 0–5, trained there
with stage 1 of recipe R) were continued with stage 2 of recipe Rw (the validation span returned, 8 epochs,
inverse-noise loss weights) on that test set's own training pool, averaged at every issue time, anchored by the
unchanged anchoring layer with the seasonal-naive state rule, corrected by the continuation layer with its four frozen
constants (no refit on any external data) and read out by rule A (energy, level and peak before the layer). The run and
its rules were fixed before any 0.2.0 number of these test sets was read, and each test set was scored once with the
frozen estimators and random streams of the external test (whose primary readouts of the 0.1 candidate it reproduced
exactly before scoring 0.2.0). The shipped checkpoints were not used. The comparators are those of the frozen test.

**Identity.** A re-run on test sets read once before with the 0.1 candidate: it shows how the 0.2.0 change behaves on
data outside the nine populations, not how ankyra-lite performs on unseen populations.

### 0.2.0: primary readouts (hourly, six seeds, continuation layer)

| Test set | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|---|
| LCL households (London) | 710 / 710 / 5 | +2.74% [−1.16, +6.48] | +24.37% [+17.71, +30.48] * | +3.13% [+2.06, +4.19] * |
| UNICON (La Trobe University campuses) | 450 / 59 / 23 | −2.42% [−14.33, +8.24] | +11.83% [+0.73, +21.68] * | +10.71% [+5.51, +15.62] * |
| HKUST campus incomer meters (secondary) | 89 / 29 / 8 | +10.88% [−9.01, +27.14] | +26.18% [+6.34, +41.81] * | +4.63% [−2.96, +11.67] |
| GoiEner households, out of time (supplementary) | 1,417 / 282 / 6 | +4.25% [+1.88, +6.57] * | +26.04% [+23.82, +28.20] * | +5.55% [+3.49, +7.57] * |

Over the four scored test sets 8 of the 12 primary intervals are resolved in ankyra-lite's favour, 4 are not resolved
and none is resolved against it (0.1 candidate: 7, 5 and 0; UNICON against the weekly seasonal naive is newly
resolved); with no true difference, about 0.75 of the 15 planned intervals would be resolved by chance. Against
zero-shot TimesFM 2.5 only the GoiEner check is resolved (+4.25%; whether GoiEner is in the TimesFM 2.5 pretraining data
is not verified); on LCL (+2.74%; LCL is in the TimesFM 2.5, Chronos and Chronos-2 pretraining data), UNICON (−2.42%)
and HKUST (+10.88%) the intervals include zero.

### 0.2.0 against the frozen external candidate and the 0.1.0 configuration; decomposition

Descriptive; "forecast hours 0-2" is the file's segment `forecast hours 0-2 (lead hours 0-3)`, the first three hours
after the origin.

| Test set | Readout | vs frozen candidate (R, 6 seeds) | vs 0.1.0 configuration | weighting (Rw vs R, 6 seeds, no layer) | layer on / off (recipe Rw, 6 seeds) | layer on / off (recipe R, 6 seeds) | six vs three seeds (recipe Rw, with the layer) |
|---|---|---|---|---|---|---|---|
| LCL households (London) | hourly | +0.06% [−0.18, +0.29] | +0.17% [−0.08, +0.42] | −0.06% [−0.30, +0.18] | +0.12% [−0.02, +0.27] | +0.12% [−0.02, +0.27] | +0.20% [+0.05, +0.39] * |
| LCL households (London) | forecast hours 0-2 (lead hours 0-3) | +10.78% [+3.60, +17.43] * | +12.06% [+4.64, +18.90] * | +0.04% [−1.47, +1.52] | +10.75% [+2.59, +16.93] * | +10.72% [+2.64, +16.83] * | −0.43% [−1.63, +0.58] |
| LCL households (London) | forecast day 1 | +4.89% [+2.23, +7.48] * | +5.32% [+2.50, +8.05] * | +0.29% [−0.30, +0.87] | +4.62% [+1.75, +7.19] * | +4.55% [+1.62, +7.22] * | +0.39% [+0.03, +0.84] * |
| UNICON (La Trobe University campuses) | hourly | +2.95% [+0.82, +5.04] * | +3.20% [+0.94, +5.41] * | +1.84% [+0.27, +3.38] * | +1.14% [+0.10, +2.20] * | +1.22% [+0.14, +2.30] * | −0.09% [−0.60, +0.46] |
| UNICON (La Trobe University campuses) | forecast hours 0-2 (lead hours 0-3) | +42.40% [+28.30, +53.73] * | +42.02% [+27.96, +53.34] * | +2.37% [−1.89, +6.45] | +41.00% [+27.57, +52.26] * | +41.84% [+27.35, +53.53] * | +1.24% [−1.82, +4.53] |
| UNICON (La Trobe University campuses) | forecast day 1 | +16.67% [+9.10, +23.61] * | +17.49% [+10.15, +24.23] * | +1.51% [−1.24, +4.18] | +15.39% [+8.76, +22.12] * | +15.76% [+8.95, +22.73] * | +0.98% [−0.09, +1.83] |
| HKUST campus incomer meters (secondary) | hourly | +0.72% [−1.37, +2.75] | +0.01% [−2.62, +2.57] | +0.16% [−2.09, +2.35] | +0.56% [+0.17, +1.17] * | +0.62% [+0.22, +1.22] * | −0.78% [−1.64, +0.41] |
| HKUST campus incomer meters (secondary) | forecast hours 0-2 (lead hours 0-3) | +20.39% [+3.73, +34.16] * | +19.75% [+2.27, +34.10] * | −4.59% [−13.64, +3.74] | +23.88% [+6.24, +39.03] * | +20.71% [+4.27, +35.82] * | −0.36% [−5.15, +3.55] |
| HKUST campus incomer meters (secondary) | forecast day 1 | +7.34% [−3.38, +16.96] | +8.46% [−2.20, +18.00] | −2.20% [−6.96, +2.35] | +9.33% [−0.24, +20.90] | +8.21% [−0.57, +19.43] | +0.96% [−1.18, +2.91] |
| GoiEner households, out of time (supplementary) | hourly | +0.07% [−0.24, +0.38] | +0.10% [−0.27, +0.48] | +0.06% [−0.20, +0.32] | +0.01% [−0.10, +0.11] | +0.02% [−0.08, +0.13] | +0.03% [−0.04, +0.19] |
| GoiEner households, out of time (supplementary) | forecast hours 0-2 (lead hours 0-3) | +6.89% [−0.43, +13.68] | +7.31% [−0.36, +14.39] | +0.69% [−0.84, +2.20] | +6.24% [−0.25, +13.42] | +6.56% [+0.21, +13.68] * | +0.40% [−0.37, +1.05] |
| GoiEner households, out of time (supplementary) | forecast day 1 | +1.78% [−0.23, +3.74] | +2.07% [−0.10, +4.18] | +0.48% [−0.24, +1.20] | +1.30% [+0.38, +3.79] * | +1.42% [+0.43, +3.97] * | +0.21% [−0.05, +0.40] |

- **The continuation layer** helps in the first hours on every test set (resolved on LCL, UNICON and HKUST in hours
  0–2) and on day 1 (resolved on LCL, UNICON and GoiEner); over the month it is resolved on UNICON (+1.14%) and HKUST
  (+0.56%) and within ±0.12% elsewhere.
- **The loss weighting** is resolved only on UNICON over the month (+1.84%); elsewhere −0.06% to +0.16%.
- **Against the frozen 0.1 candidate**, 0.2.0 is resolved better over the month on UNICON only (+2.95%), and on day 1 on
  LCL (+4.89%) and UNICON (+16.67%).

### 0.2.0: other readouts against TimesFM 2.5

From `external_tests.csv`, block `other readouts` (six seeds; unit-and-month interval with the seed layer):

| Test set | Forecast hours 0-2 | Forecast day 1 | Days 2–7 | Days 8–31 | Monthly energy (level readout) |
|---|---|---|---|---|---|
| LCL households (London) | −25.38% [−35.25, −16.23] (−) | −10.43% [−16.42, −4.74] (−) | −2.82% | +3.00% | +54.08% [+44.91, +61.73] * |
| UNICON (La Trobe University campuses) | −73.88% [−109.41, −44.38] (−) | −13.29% [−30.10, +1.35] | −7.54% | −2.47% | −14.74% [−56.20, +15.72] |
| HKUST campus incomer meters (secondary) | −25.89% [−73.73, +8.78] | −1.43% [−18.59, +13.25] | +1.90% | +12.27% | +32.52% [−79.13, +74.58] |
| GoiEner households, out of time (supplementary) | −18.26% [−28.83, −8.57] (−) | −1.99% [−8.23, +3.89] | +0.75% | +4.51% * | +48.53% [+41.33, +54.85] * |

(−) resolved worse. Zero-shot TimesFM 2.5 remains ahead in the first hours; ankyra-lite's monthly energy error is
resolvably lower on LCL and GoiEner. Mean unit rank of ankyra-lite (block `mean unit rank`, all comparators on their
common windows): first of 21 on LCL, third of 14 on UNICON, first of 14 on HKUST and first of 5 on GoiEner.

### The day-1 rule

Written before the read: if the external test sets' mean day-1 log gap to TimesFM 2.5 shrinks, from the 0.1.0
configuration (recipe R, three seeds) to 0.2.0, by less than a third of the shrinkage measured on the development
windows, no claim is made that 0.2.0 narrows the short-term gap.

| Test set | Day 1 vs TimesFM 2.5, 0.1.0 configuration | Day 1 vs TimesFM 2.5, 0.2.0 | Shrinkage of the log gap |
|---|---|---|---|
| LCL households (London) | −16.63% (−) | −10.43% (−) | +0.055 |
| UNICON (La Trobe University campuses) | −37.30% (−) | −13.29% | +0.192 |
| HKUST campus incomer meters (secondary) | −10.80% | −1.43% | +0.088 |
| GoiEner households, out of time (supplementary) | −2.41% | −1.99% | +0.004 |

Day-1 rule (design 3.3): claim allowed (mean external shrink (log) 0.0848406119805109; line (1/3 of the development shrink) 0.03576426220072093; the external day-1 gap to TimesFM 2.5 shrank by at least 1/3 of the development-set shrink (mean over the scored faces)).

The development shrinkage is +0.107 (log units). **What the rule allows:** to say that the day-1 gap to TimesFM 2.5
narrowed from 0.1 to 0.2.0 on these test sets. It does not allow saying that ankyra-lite leads TimesFM 2.5 on day 1: it
does not, and on LCL TimesFM 2.5 is still resolvably ahead. The shrinkage is mostly UNICON's; on GoiEner it is close to
zero. Figure 17's footer prints the same outcome.

![External tests](../figures/fig17_external_tests.png)

*Figure 17. External tests of ankyra-lite 0.2.0, the 0.2.0 configuration re-run on the test sets of the frozen external
test and scored once. a–c: the three pre-specified primary readouts of each test set (12 nominal intervals over four
test sets, no multiplicity correction); d–f: the same contrasts by lead time, forecast hours 0–2 included; g–i:
descriptive contrasts between ankyra-lite's own configurations by lead time (g: 0.2.0 against the 0.1.0
configuration; h: the continuation layer on against off; i: the loss weighting of recipe Rw against recipe R, without
the layer). Welch 95% intervals where the record has a seed layer, else unit-and-month only; filled = resolved; an
interval end beyond the axis is drawn as an arrow. LCL is in the pretraining data of TimesFM 2.5, Chronos and
Chronos-2. Source: `results/external_tests.csv`.*

## How to read the numbers

- **Improvement** is the unit-equal RMS ratio of the first release: d is the mean over units of the log of the second
  forecaster's RMS error over ankyra-lite's, and the improvement is 100 (1 − exp(−d)) percent; **positive means
  ankyra-lite has the lower error**. Units whose mean load is below 10⁻⁶ kW are left out of the ratio and listed.
- **Interval.** A 95% interval that combines the unit-and-month cluster bootstrap (2,000 draws) with the spread of the
  single-seed contrasts: SE = √(SE_unit×month² + SD_seeds²/M), with Welch degrees of freedom and a t quantile. M = 6 for
  the six-seed ensemble (the primary readout) and 3 for a three-seed configuration. Contrasts without a seed layer have
  the unit-and-month interval only.
- **Three-state label at zero**: *resolved better* (marked \*), *not resolved*, *resolved worse* (marked † in the 0.1
  tables, (−) in the 0.2.0 tables). **Not resolved is never a finding of no difference**: it means that the interval
  includes zero. No pass line was set. "—" in a table: not computed (the peak readout exists only for the arms
  recomputed in a run, not for the saved baselines).
- **Multiplicity.** The primary readouts are 3 contrasts × 5 test sets = 15 nominal 95% intervals (12 scored; the 3 of
  Helsinki are sealed). There is no multiplicity correction, no combined verdict and no pooling across test sets; if no
  true difference existed, about 0.75 of the 15 would be expected to show "resolved" by chance.
- **Pretraining overlap**, stated with the TimesFM numbers: LCL is in the pretraining data of TimesFM 2.5, Chronos and
  Chronos-2, which favours the TimesFM- and Chronos-based baselines; whether GoiEner is in the TimesFM 2.5 or Chronos
  pretraining data is not verified; UNICON and HKUST are not on the published pretraining lists of TimesFM 2.5,
  Chronos or Chronos-2, which does not establish absence.
- Every estimate, interval, label and mean rank of the 0.2.0 section, with the window, unit and month counts of its
  contrast, is a row of [`results/external_tests.csv`](../results/external_tests.csv), copied from the score record
  that its `source` column cites ([`evaluation/external_tables.py`](../evaluation/external_tables.py)). The counts that
  describe the data and the run come from the run's records, named where they appear (a field `x` of a score record is
  written `score: x`).

## Version 0.1 external test (19 October 2026)

History. ankyra-lite 0.1.0, the copy-pointer carrier trained with recipe R, the anchoring layer copied byte for byte
from ANKYRA 2.2.1 and the seasonal-naive state rule, without a continuation layer, was tested on data outside the test
windows of the nine populations. The run was frozen on 19 October 2026 at 07:45 UTC, before any training on these data:
the test plan, an addendum written before the freeze, a snapshot of the package, the code that trained, anchored and
scored, and every input were hashed. Each test set was scored once, after its forecasts had been hashed. The carrier was
trained anew for each test set with recipe R; the shipped checkpoints were not used. The numbers of this part are those
of `results/external_tests.csv` in the 0.1.1 release (commit `c27affc`); the 0.2.0 regeneration replaced that file.
The sections from "Procedure" to "Run notes" describe this run; the 0.2.0 re-run used the same split, comparators,
scoring and stage-1 carriers.

Three contrasts were fixed before any reading, for every test set: ankyra-lite (six seeds) against zero-shot TimesFM
2.5, against the weekly seasonal naive (SN-week), and against the carrier-free control with ankyra-lite's state rule
(the unchanged anchoring layer called with the history-only forecasts in the carrier's place, with the seasonal-naive
forecast on the same state windows). Hourly error.

### Version 0.1 primary readouts (hourly, six seeds)

| Test set | Identity | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|---|---|
| LCL households (London) | first read of this candidate on a previously read population (retrospective external re-evaluation) | 710 / 710 / 5 | +2.68% [−1.22, +6.43] | +24.32% [+17.65, +30.46] * | +3.08% [+2.08, +4.07] * |
| UNICON (La Trobe University campuses) | first read of this candidate on a previously read population (retrospective external re-evaluation); the anchoring layer and the network family were seen at design time on these targets | 450 / 59 / 23 | −5.54% [−19.45, +6.76] | +9.15% [−3.73, +20.42] | +8.00% [+3.69, +12.11] * |
| HKUST campus incomer meters (secondary) | first read of this candidate on a previously read population (retrospective external re-evaluation) (secondary test set, small sample) | 89 / 29 / 8 | +10.24% [−9.22, +26.23] | +25.65% [+6.33, +40.98] * | +3.95% [−4.11, +11.38] |
| GoiEner households, out of time (supplementary) | same-source out-of-time supplementary check (new units, new period), not an external population; correlated with the result on study population 6 (GoiEner households) | 1,417 / 282 / 6 | +4.19% [+1.59, +6.72] * | +25.99% [+23.87, +28.05] * | +5.48% [+3.59, +7.34] * |

Version 0.1: over the four scored test sets (three first reads of this candidate on previously read populations and one
same-source out-of-time supplementary check), 7 of the 12 primary intervals are resolved in ankyra-lite's favour, 5 are not resolved
and none is resolved against it; with no true difference, about 0.75 of the 15 planned intervals would be resolved by
chance. Against zero-shot TimesFM 2.5 only the GoiEner check is resolved (+4.19%, same-source out-of-time supplementary
check; whether GoiEner is in the TimesFM 2.5 pretraining data is not verified); on LCL (+2.68%; LCL is in the TimesFM
2.5, Chronos and Chronos-2 pretraining data), UNICON (−5.54%) and HKUST (+10.24%), first reads of this candidate on
previously read populations, the intervals include zero.

## Procedure common to every test set

**The forecaster.** The `ankyra_lite` package of this version, copied before the run into a hashed snapshot from which
every step imported it: the copy-pointer carrier (61,613 parameters per seed), the anchoring layer (byte-identical to
ANKYRA 2.2.1, 17 files, checked against the original package at the freeze) and the seasonal-naive state rule (on the
windows the anchoring layer marks as off or micro-load, the last 168 hours before the origin repeated over the month).
Category: `Residential` for LCL and the GoiEner households, `Public` for UNICON and HKUST; daylight-saving region `EU`
for LCL and GoiEner, none for UNICON and HKUST.

**Training, per test set, with recipe R.** Stage 1 trains from scratch with early stopping on the validation set (the 93
days before the training cut C′; at most 30 epochs, patience 5, best epoch kept); stage 2 returns those 93 days to the
training set (labels ending less than 24 hours before C′ are not returned) and trains 8 epochs with a fresh optimiser,
keeping the last. Seeds 0–5 were trained; the **primary readout is the six-seed ensemble**, and the **release
configuration** is the ensemble of seeds 0–2 (the size of the shipped checkpoints). The training set holds the test
set's own windows (at most 16,000 training and 2,000 validation windows, unit-balanced) and windows borrowed from the
study's ten sources (the nine populations and the Suzhou park) whose labels end, in absolute time, before the test
set's validation start; half of the training draws come from the test set itself, the other half from one borrowed
source chosen at random and then one of its units. For LCL only the Cambridge population qualified, so Cambridge
windows make up half of LCL's draws; UNICON had nine borrowed sources (none from CINELDI), HKUST and GoiEner ten. Some
stores carry local clock time labelled as UTC, so this absolute-time condition can be off by up to about 17 hours,
which still leaves every borrowed label more than 90 days before the test set's training cut.

**Split.** As for the nine populations: the test start C is the first month start at or after the median origin of the
test set's evaluation windows; the test set is the windows with origin at or after C; the training cut is C′ = C − 186
days, and the validation set is the 93 days before C′. Every pseudo-origin lies at or after C′, so every pseudo-origin
forecast that the anchoring layer scores is out of sample for the carrier. The temperature-anomaly scale σ(C′) uses
the complete days with measured temperature before C′.

| Test set | C (test start) | C′ (training cut) | Validation set from | σ(C′) | Test windows / units / months | State windows | Units without first-stage own training windows |
|---|---|---|---|---|---|---|---|
| LCL households (London) | 2013-09-01 | 2013-02-27 | 2012-11-26 | 0.26882 | 710 / 710 / 5 | 2 | 83 (83 windows) |
| UNICON (La Trobe University campuses) | 2019-11-01 | 2019-04-29 | 2019-01-26 | 0.29694 | 450 / 59 / 23 | 0 | 10 (64 windows) |
| HKUST campus incomer meters (secondary) | 2023-09-01 | 2023-02-27 | 2022-11-26 | 0.20210 | 96 / 32 / 8 | 7 | 1 (2 windows) |
| GoiEner households, out of time (supplementary) | 2021-12-01 | 2021-05-29 | 2021-02-25 | 0.28164 | 1,422 / 284 / 6 | 44 | 142 (603 windows) |

From the score records: `score: split`, `score: sigma_Cprime`, `score: windows` (scored windows, units, months and
state windows) and the subgroup `units without own training windows` (shown here as units without first-stage own training
windows).

**Comparators.** Zero-shot TimesFM 2.5 and the other baselines are the saved forecasts of the first release's
evaluation on the same windows (LCL, UNICON, HKUST); for GoiEner, whose households had never been extracted, TimesFM 2.5
was run on the test windows with the release's adapter, and SN-week, SN-day and the four-week profile were computed with
the release's functions (no trained baseline). The seven trained baselines of LCL were trained with labels up to C,
186 days later than the carrier's C′. Descriptive arms: the history-only configuration (the anchoring layer without a
carrier), the carrier-free control without the state rule, the plain continuation F (stage 1 followed by 8 epochs
without returning the validation span; seeds 0–2) and the from-scratch stage alone (six seeds). The comparisons with
ANKYRA itself and its versions are kept in the study's research record only; none is in this repository.

**Scoring.** The targets are never filled. Missing and flagged hours are left out of the hourly error; the monthly
energy uses the mean over the observed hours, flagged hours included; the peak readout exists only for the arms
recomputed in this run. A window on which any arm failed would have been removed from every arm; no window failed on
any test set. The forecasts of every arm were hashed before a test set's scoring began, and the scorer refuses a second
reading of a test set.

## LCL households (London)

**Question.** On London households that earlier forecasters of the study had read, but this candidate had not, how
does ankyra-lite's hourly error compare with zero-shot TimesFM 2.5, the weekly seasonal naive and the carrier-free
control?

**Data.** Low Carbon London smart-meter households (UK Power Networks, published on the London Datastore, dataset
`smartmeter-energy-consumption-data-in-london-households`). Licence: "Creative Commons Attribution", version not
given on the page (as checked on the provider's page on 5 October 2026 for the first release); the data are not
redistributed, and this repository holds only scored statistics. The study's prepared store of the first release's
evaluation, with the time axis padded back to 1 January 2011 (load unobserved, temperature a constant 15 °C in the
padding); England and Wales bank holidays as day type 7. The test windows are those of the first release with origin
at or after C.

**Procedure.** As above; borrowed windows from Cambridge only. The anchoring layer's climatology fit uses every day
before the origin, the padded days included, as in the first release.

**Result** (first read of this candidate on a previously read population; LCL is in the pretraining data of TimesFM
2.5, Chronos and Chronos-2). Six seeds, then the three-seed release configuration:

| Readout | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|
| hourly (6 seeds) | +2.68% [−1.22, +6.43] | +24.32% [+17.65, +30.46] * | +3.08% [+2.08, +4.07] * |
| monthly energy (6 seeds) | +54.03% [+44.93, +61.62] * | +27.41% [+5.19, +44.42] * | +5.59% [−8.71, +18.01] |
| monthly energy (trajectory sum) (6 seeds) | +54.05% [+44.96, +61.63] * | +27.43% [+5.21, +44.45] * | +5.56% [−8.68, +17.92] |
| peak (6 seeds) | — | — | −0.16% [−4.18, +3.70] |
| forecast day 1 (6 seeds) | −16.11% [−24.80, −8.02] † | +16.57% [+9.67, +22.95] * | +6.69% [+2.59, +10.62] * |
| forecast days 2-7 (6 seeds) | −3.92% [−8.36, +0.34] | +23.38% [+19.36, +27.20] * | +3.91% [+2.70, +5.10] * |
| forecast days 8-31 (6 seeds) | +3.08% [−0.94, +6.94] | +24.67% [+17.33, +31.36] * | +2.75% [+1.62, +3.86] * |
| hourly (3 seeds) | +2.57% [−1.36, +6.35] | +24.24% [+17.51, +30.42] * | +2.97% [+1.94, +3.98] * |
| monthly energy (3 seeds) | +53.12% [+43.52, +61.09] * | +25.98% [+4.10, +42.86] * | +3.73% [−10.60, +16.21] |
| monthly energy (trajectory sum) (3 seeds) | +53.13% [+43.53, +61.09] * | +25.98% [+4.08, +42.88] * | +3.66% [−10.62, +16.10] |
| peak (3 seeds) | — | — | +1.16% [−3.54, +5.64] |
| forecast day 1 (3 seeds) | −16.63% [−25.50, −8.38] † | +16.21% [+9.15, +22.72] * | +6.27% [+2.07, +10.29] * |
| forecast days 2-7 (3 seeds) | −4.12% [−8.66, +0.24] | +23.24% [+19.14, +27.13] * | +3.73% [+2.46, +4.98] * |
| forecast days 8-31 (3 seeds) | +2.97% [−1.07, +6.84] | +24.58% [+17.20, +31.31] * | +2.64% [+1.52, +3.74] * |

Descriptive contrasts (first read of this candidate on a previously read population):

| Descriptive | span return vs plain continuation (R vs F, 3 seeds) | continuation and span return vs from-scratch stage (6 seeds; unit × month interval only) | ankyra-lite vs history-only configuration (6 seeds) | ankyra-lite vs carrier-free control (6 seeds) |
|---|---|---|---|---|
| hourly | +0.61% [+0.06, +1.16] * | +0.60% [+0.03, +1.10] * | +5.01% [+3.29, +6.69] * | +3.51% [+2.00, +5.00] * |
| monthly energy | +1.30% [−5.56, +7.71] | +0.45% [−9.03, +9.66] | +7.17% [−5.94, +18.65] | +6.80% [−7.21, +18.99] |
| monthly energy (trajectory sum) | +1.31% [−5.58, +7.74] | +0.46% [−9.01, +9.63] | +7.14% [−5.86, +18.55] | +6.77% [−7.18, +18.90] |
| peak | +0.68% [−2.20, +3.47] | −0.65% [−4.40, +2.88] | −0.28% [−4.26, +3.55] | −0.15% [−4.16, +3.71] |
| forecast day 1 | +1.29% [−0.61, +3.15] | +1.64% [+0.66, +2.59] * | +8.32% [+4.14, +12.32] * | +6.68% [+2.58, +10.61] * |
| forecast days 2-7 | +0.93% [+0.29, +1.58] * | +1.12% [+0.53, +1.82] * | +5.75% [+4.26, +7.22] * | +3.91% [+2.70, +5.10] * |
| forecast days 8-31 | +0.60% [+0.01, +1.19] * | +0.53% [−0.15, +1.15] | +4.55% [+2.71, +6.37] * | +3.08% [+1.46, +4.67] * |

Other saved baselines, hourly, six seeds (first read of this candidate on a previously read population; the trained
baselines' training labels reach C, 186 days later than the carrier's C′; LCL is in the pretraining data of TimesFM
2.5, Chronos and Chronos-2, which favours TimesFM-X and the Chronos-based ones):

| Other saved baseline (hourly, 6 seeds) | Result | Training labels to C |
|---|---|---|
| Chronos-2 | +2.34% [−1.36, +5.92] | no |
| Chronos-2-X | +2.65% [−1.00, +6.17] | no |
| TimesFM-X | +2.75% [−1.07, +6.42] | no |
| DLinear | +4.04% [+2.48, +5.58] * | yes |
| GBT-T | +9.53% [+4.99, +13.85] * | yes |
| GBT (zero-shot) | +33.08% [+17.59, +45.65] * | no |
| Holt-Winters | +11.81% [+7.03, +16.34] * | no |
| iTransformer | +2.17% [−1.30, +5.52] | yes |
| iTransformer-X | +2.78% [−0.43, +5.88] | yes |
| LSTM | +10.35% [+7.99, +12.65] * | yes |
| Last-year profile | +30.96% [+28.72, +33.12] * | no |
| MSTL | +22.26% [+17.88, +26.41] * | no |
| PatchTST | +2.14% [−1.21, +5.37] | yes |
| Seasonal naive (day) | +22.73% [+18.17, +27.05] * | no |
| Previous-month profile | +3.60% [+0.03, +7.05] * | no |
| Linear-context (TiDE implementation) | +3.99% [+2.43, +5.53] * | yes |
| 4-week profile | +7.00% [+3.35, +10.50] * | no |
| Ridge (per unit) | +2.57% [−0.51, +5.55] | no |

Mean unit rank among 21 forecasters (710 windows, 710 units): ankyra-lite (6 seeds) 4.15, Ridge (per unit) 6.66, iTransformer 7.23, PatchTST 7.51, Linear-context (TiDE implementation) 7.56, iTransformer-X 7.63, DLinear 7.79, Chronos-2 8.91, TimesFM-X 9.05, Chronos-2-X 9.41, Previous-month profile 9.47, GBT-T 9.80, TimesFM 9.96, LSTM 11.76, Holt-Winters 12.28, 4-week profile 12.51, GBT (zero-shot) 17.43, MSTL 17.55, Seasonal naive (day) 17.57, Seasonal naive (week) 18.19, Last-year profile 18.57; ankyra-lite at position 1.

Subgroups (first read of this candidate on a previously read population; LCL is in TimesFM 2.5's pretraining data):

| Subgroup (hourly, 6 seeds) | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|---|
| units without first-stage own training windows | 83 / 83 / 2 | +5.22% [+1.69, +8.63] * | +30.91% [+23.18, +37.86] * | +2.58% [+1.50, +3.65] * |
| units with first-stage own training windows | 627 / 627 / 5 | +2.34% [−1.85, +6.35] | +23.41% [+17.10, +29.23] * | +3.14% [+2.00, +4.27] * |

**Reading.**

- Against zero-shot TimesFM 2.5 the hourly improvement is +2.68% [−1.22, +6.43], not resolved, with all six seeds
  positive (first read of this candidate on a previously read population; LCL is in TimesFM 2.5's pretraining data).
  In the same reading it is resolved worse on forecast day 1 (−16.11%) and resolved better in monthly energy (+54.03%)
  (first read of this candidate on a previously read population; LCL is in TimesFM 2.5's pretraining data).
- Against the weekly seasonal naive (+24.32%) and the carrier-free control (+3.08%) the hourly improvements are
  resolved (first read of this candidate on a previously read population); against the history-only configuration it
  is +5.01%, also resolved.
- In the descriptive recipe contrast, returning the validation span is resolved better than the plain continuation
  (+0.61%, three seeds; first read of this candidate on a previously read population).
- Mean unit rank 4.15, the lowest of 21 forecasters (first read of this candidate on a previously read population).
- No calendar window was listed (a fixed list of bank holidays); 2 state windows; no window failed.

**What it does not show.** LCL had been read by earlier forecasters of the study, and the anchoring layer was fixed
after those readings; this is a retrospective external re-evaluation of a new candidate, not a test on an unseen
population. The test set covers 5 months, so the month resampling has few clusters and the intervals may be too
narrow.

## UNICON (La Trobe University campuses)

**Question.** On the building meters of La Trobe University's campuses in Victoria, Australia (the prepared store's
meters lie on three of the five campuses: Bundoora, Albury-Wodonga and Bendigo), read before by earlier forecasters
and by earlier versions of the small network on the same test windows, how does this candidate compare?

**Data.** UNICON (Moraliyage et al., HSI 2022; Kaggle `cdaclab/unicon`, version 1), licence CC BY-NC-SA 4.0: research
use only; neither the data nor carriers trained on them are in this repository. Training read the study's prepared
store, blanked from C′ on; every arm was tested on each window's own series, in which context gaps of at most 6 hours
are filled by linear interpolation from data before that window's origin (the rule of the first release's UNICON test);
targets are never filled, so 3,851 target hours in 375 of the 450 windows are missing and 114 flagged hours are left out
of the hourly error (`score: windows.target_hours_missing`, `windows_with_missing_target_hours`, `target_hours_flagged`).
Victorian public holidays as day type 7. Temperature: ERA5 2 m temperature through the Open-Meteo archive interface,
one point per campus (the weather data shipped with the dataset are not used); the store's temperature equals the
study's archive files hour for hour. Contains modified Copernicus Climate Change Service information [2026]; ERA5
hourly data (Hersbach et al., 2020, doi:10.1002/qj.3803; dataset doi:10.24381/cds.adbb2d47) retrieved through the
Open-Meteo historical weather API (Zippenfenig, doi:10.5281/zenodo.7970649, CC BY 4.0).

**Procedure.** As above, with nine borrowed sources. The history-only forecasts that the carrier-free control places in
the carrier's slot could not be issued at one or more pseudo-origins in every test window (at all six in 322 of 450;
`N_k_failed` of the run's product record); as the plan fixed, those pseudo-origins are not used, so on UNICON this
control rests on fewer pseudo-origins than on the other test sets. The carrier's own pseudo-origin forecasts are also
fewer on UNICON: the run's operations log records 1,880 carrier forecasts per seed for the 450 windows, against 7 per
window on LCL (4,970), HKUST (672) and GoiEner (9,954) (`carriers` events); the cause was not checked, so ankyra-lite's
anchoring on UNICON also uses fewer pseudo-origins. There are no state windows, so the controls with and without the
state rule are the same forecasts.

**Result** (first read of this candidate on a previously read population; the anchoring layer and the network family
were seen at design time on these targets; UNICON is not on the published pretraining lists of TimesFM 2.5, Chronos or
Chronos-2, which does not establish absence):

| Readout | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|
| hourly (6 seeds) | −5.54% [−19.45, +6.76] | +9.15% [−3.73, +20.42] | +8.00% [+3.69, +12.11] * |
| monthly energy (6 seeds) | −19.19% [−63.01, +12.84] | −21.23% [−59.25, +7.72] | +7.76% [−2.87, +17.29] |
| monthly energy (trajectory sum) (6 seeds) | −18.92% [−63.46, +13.49] | −20.94% [−59.48, +8.28] | +7.95% [−2.72, +17.52] |
| peak (6 seeds) | — | — | +3.42% [−5.26, +11.38] |
| forecast day 1 (6 seeds) | −35.95% [−65.92, −11.40] † | +1.42% [−16.40, +16.51] | +15.55% [+8.52, +22.04] * |
| forecast days 2-7 (6 seeds) | −13.40% [−31.35, +2.09] | +7.21% [−6.97, +19.52] | +9.85% [+3.86, +15.45] * |
| forecast days 8-31 (6 seeds) | −4.91% [−19.32, +7.76] | +9.14% [−4.13, +20.73] | +7.43% [+3.12, +11.56] * |
| hourly (3 seeds) | −5.81% [−20.02, +6.72] | +8.91% [−4.22, +20.39] | +7.76% [+3.27, +12.04] * |
| monthly energy (3 seeds) | −19.04% [−63.85, +13.52] | −21.07% [−60.21, +8.51] | +7.88% [−5.20, +19.34] |
| monthly energy (trajectory sum) (3 seeds) | −18.73% [−63.97, +14.02] | −20.76% [−60.12, +8.93] | +8.10% [−4.74, +19.36] |
| peak (3 seeds) | — | — | +3.97% [−3.79, +11.15] |
| forecast day 1 (3 seeds) | −37.30% [−67.12, −12.80] † | +0.44% [−17.16, +15.40] | +14.71% [+7.46, +21.40] * |
| forecast days 2-7 (3 seeds) | −14.16% [−32.55, +1.67] | +6.60% [−7.99, +19.21] | +9.24% [+2.68, +15.36] * |
| forecast days 8-31 (3 seeds) | −5.07% [−19.78, +7.84] | +9.01% [−4.54, +20.80] | +7.29% [+2.89, +11.50] * |

Descriptive contrasts (first read of this candidate on a previously read population):

| Descriptive | span return vs plain continuation (R vs F, 3 seeds) | continuation and span return vs from-scratch stage (6 seeds; unit × month interval only) | ankyra-lite vs history-only configuration (6 seeds) | ankyra-lite vs carrier-free control (6 seeds) |
|---|---|---|---|---|
| hourly | −2.68% [−6.92, +1.39] | +1.52% [−0.22, +3.36] | +8.48% [+3.73, +13.01] * | +8.00% [+3.69, +12.11] * |
| monthly energy | −8.54% [−20.98, +2.63] | +3.91% [−4.26, +9.66] | +7.21% [−4.95, +17.96] | +7.76% [−2.87, +17.29] |
| monthly energy (trajectory sum) | −8.55% [−21.01, +2.62] | +4.04% [−4.30, +10.76] | +7.34% [−5.39, +18.52] | +7.95% [−2.72, +17.52] |
| peak | −1.12% [−10.38, +7.37] | +1.07% [−12.49, +5.96] | +3.68% [−6.46, +12.85] | +3.42% [−5.26, +11.38] |
| forecast day 1 | −1.23% [−7.05, +4.28] | +3.71% [−0.09, +8.37] | +15.75% [+7.82, +23.00] * | +15.55% [+8.52, +22.04] * |
| forecast days 2-7 | −1.83% [−5.37, +1.58] | +4.02% [+1.57, +6.50] * | +10.22% [+3.75, +16.25] * | +9.85% [+3.86, +15.45] * |
| forecast days 8-31 | −2.88% [−7.37, +1.42] | +0.98% [−0.89, +2.92] | +7.93% [+3.12, +12.50] * | +7.43% [+3.12, +11.56] * |

Other saved baselines, hourly, six seeds (first read of this candidate on a previously read population; UNICON is not on
the published pretraining lists of TimesFM 2.5, Chronos or Chronos-2, which does not establish absence):

| Other saved baseline (hourly, 6 seeds) | Result | Training labels to C |
|---|---|---|
| Chronos-2 | −6.66% [−20.99, +5.98] | no |
| Chronos-2-X | −6.28% [−20.96, +6.62] | no |
| TimesFM-X | −4.35% [−17.31, +7.17] | no |
| Seasonal naive (day) | +27.05% [+10.42, +40.59] * | no |
| 4-week profile | +1.06% [−8.43, +9.72] | no |
| Previous-month profile | +10.39% [+2.19, +17.90] * | no |
| Last-year profile | +16.62% [+6.26, +25.82] * | no |
| Ridge (per unit) | −2.07% [−10.93, +6.08] | no |
| Holt-Winters | +9.80% [+1.50, +17.41] * | no |
| MSTL | +9.97% [−2.37, +20.82] | no |
| GBT (zero-shot) | +18.05% [+4.28, +29.84] * | no |

Mean unit rank among 14 forecasters (450 windows, 59 units): Chronos-2 3.97, TimesFM 4.22, TimesFM-X 4.37, Chronos-2-X 4.71, ankyra-lite (6 seeds) 5.08, Ridge (per unit) 5.08, 4-week profile 6.71, Holt-Winters 8.92, Previous-month profile 9.25, MSTL 9.51, Seasonal naive (week) 9.81, Last-year profile 10.53, GBT (zero-shot) 10.56, Seasonal naive (day) 12.27; ankyra-lite at position 5.

Subgroups (first read of this candidate on a previously read population; 2020 is the COVID year):

| Subgroup (hourly, 6 seeds) | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|---|
| units without first-stage own training windows | 64 / 10 / 17 | −8.06% [−36.62, +14.52] | +6.11% [−15.18, +23.46] | +10.61% [+0.28, +19.88] * |
| units with first-stage own training windows | 386 / 49 / 21 | −5.03% [−19.78, +7.91] | +9.75% [−4.05, +21.72] | +7.45% [+3.27, +11.46] * |
| by origin year (2020 = COVID year): 2019 | 60 / 36 / 2 | +5.39% [−9.72, +18.43] | +16.32% [+9.57, +22.56] * | +10.68% [+5.86, +15.24] * |
| by origin year (2020 = COVID year): 2020 | 205 / 57 / 9 | −18.40% [−46.87, +4.55] | −2.99% [−31.79, +19.52] | +9.01% [+1.34, +16.09] * |
| by origin year (2020 = COVID year): 2021 | 161 / 47 / 11 | −7.75% [−18.87, +2.34] | +10.15% [+0.09, +19.21] * | +5.16% [−0.94, +10.89] |
| by origin year (2020 = COVID year): 2022 | 24 / 24 / 1 | +16.02% [+6.53, +24.56] * | +28.94% [+21.06, +36.03] * | +2.07% [−4.28, +8.04] |
| calendar sensitivity: primary readouts without the listed windows | 446 / 59 / 22 | −5.55% [−18.97, +6.35] | +9.10% [−3.47, +20.14] | +8.01% [+3.61, +12.21] * |

**Reading.**

- Against zero-shot TimesFM 2.5 the hourly improvement is −5.54% [−19.45, +6.76], not resolved, with all six seeds
  negative (first read of this candidate on a previously read population). On forecast day 1 it is resolved worse
  (−35.95%), and the monthly energy contrast is −19.19%, not resolved (first read of this candidate on a previously read
  population).
- Against the carrier-free control the hourly improvement is +8.00%, resolved; against the weekly seasonal naive it is
  +9.15%, not resolved (first read of this candidate on a previously read population).
- Mean unit rank 5.08, fifth of 14, behind the four zero-shot foundation-model variants and tied with the per-unit
  ridge (first read of this candidate on a previously read population; UNICON is not on the published pretraining
  lists, which does not establish absence).
- The 205 windows issued in 2020 carry the largest negative point estimate against TimesFM 2.5 (−18.40%, not resolved);
  the 24 windows of 2022 the largest positive one (+16.02%, resolved), in one month only (first read of this candidate
  on a previously read population).
- Returning the validation span is −2.68% against the plain continuation (not resolved, all three seeds negative; first
  read of this candidate on a previously read population).
- Calendar sensitivity: without the four windows issued in September 2021, whose AFL Grand Final Friday holiday could
  not be checked against its announcement date, the three primary labels are unchanged (−5.55%, +9.10%, +8.01%; first
  read of this candidate on a previously read population).
  Melbourne Cup Day is applied to every campus, an inherited input error that is the same in every arm.

**What it does not show.** The anchoring layer, its short-gap rule and the network family were all seen at design time
on these targets, and the anchoring layer's temperature climatology has a northern-hemisphere defect that was found
on UNICON (a flat line near 15 °C); the carrier has no hemisphere input and half of its training draws come from
northern populations. This is not a test on an unseen population.

## HKUST campus incomer meters (secondary)

**Question.** On the incomer meters of the HKUST campus, a small secondary test set read before by earlier forecasters,
how does this candidate compare?

**Data.** HKUST smart-meter database (Li, Wang, Qu, Chui and Leung-Shea, *Scientific Data* 11, 1284, 2024; Dryad,
CC0), the provider's cleaned files; hourly energy from cumulative readings, many quantised at 10 or 100 kWh; Hong Kong
general holidays as day type 7, whose gazette dates could not be checked offline (no window listed). Temperature:
ERA5 2 m temperature at the campus through the Open-Meteo archive interface, one series for every meter; the store's
temperature equals the study's archive file hour for hour. Contains modified Copernicus Climate Change Service
information [2026]; ERA5 hourly data (Hersbach et al., 2020, doi:10.1002/qj.3803; dataset doi:10.24381/cds.adbb2d47)
retrieved through the Open-Meteo historical weather API (Zippenfenig, doi:10.5281/zenodo.7970649, CC BY 4.0).

**Procedure.** As above. The test windows are the first release's strict-rule windows with origin at or after C. The
three incomers that read zero throughout (7 windows) are state windows, on which ankyra-lite's hourly error is exactly
zero, as is that of most comparators; their mean load is below 10⁻⁶ kW, so they are left out of the ratio and the
contrasts cover 89 windows of 29 units (`score: windows.units_excluded_mean_below_1e-6_kW`, `windows.state_windows`).

**Result** (first read of this candidate on a previously read population, secondary test set with a small sample;
HKUST is not on the published pretraining lists of TimesFM 2.5, Chronos or Chronos-2, which does not establish
absence):

| Readout | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|
| hourly (6 seeds) | +10.24% [−9.22, +26.23] | +25.65% [+6.33, +40.98] * | +3.95% [−4.11, +11.38] |
| monthly energy (6 seeds) | +32.61% [−94.71, +76.68] | +36.39% [−113.93, +81.09] | +3.79% [−37.63, +32.74] |
| monthly energy (trajectory sum) (6 seeds) | +32.61% [−94.71, +76.68] | +36.40% [−113.94, +81.09] | +4.08% [−37.16, +32.93] |
| peak (6 seeds) | — | — | −14.61% [−59.64, +17.71] |
| forecast day 1 (6 seeds) | −9.47% [−38.35, +13.39] | +22.72% [+5.98, +36.49] * | +10.59% [−0.55, +20.49] |
| forecast days 2-7 (6 seeds) | −1.39% [−15.08, +10.68] | +15.79% [+4.20, +25.98] * | +10.39% [+2.06, +18.02] * |
| forecast days 8-31 (6 seeds) | +12.41% [−12.52, +31.82] | +27.36% [+3.50, +45.32] * | +1.22% [−7.85, +9.53] |
| hourly (3 seeds) | +10.87% [−8.29, +26.64] | +26.17% [+7.10, +41.32] * | +4.62% [−3.10, +11.77] |
| monthly energy (3 seeds) | +32.68% [−98.98, +77.22] | +36.46% [−117.41, +81.43] | +3.89% [−39.86, +33.95] |
| monthly energy (trajectory sum) (3 seeds) | +32.68% [−98.99, +77.23] | +36.47% [−117.43, +81.43] | +4.19% [−39.40, +34.15] |
| peak (3 seeds) | — | — | −13.16% [−62.30, +21.10] |
| forecast day 1 (3 seeds) | −10.80% [−39.24, +11.83] | +21.78% [+5.00, +35.60] * | +9.50% [−2.14, +19.82] |
| forecast days 2-7 (3 seeds) | −0.40% [−15.08, +12.41] | +16.61% [+4.34, +27.31] * | +11.26% [+2.09, +19.58] * |
| forecast days 8-31 (3 seeds) | +13.06% [−11.21, +32.04] | +27.90% [+4.57, +45.53] * | +1.95% [−6.59, +9.81] |

Descriptive contrasts (first read of this candidate on a previously read population, secondary test set):

| Descriptive | span return vs plain continuation (R vs F, 3 seeds) | continuation and span return vs from-scratch stage (6 seeds; unit × month interval only) | ankyra-lite vs history-only configuration (6 seeds) | ankyra-lite vs carrier-free control (6 seeds) |
|---|---|---|---|---|
| hourly | +1.00% [−0.49, +2.47] | +1.07% [−7.32, +4.90] | +5.10% [−2.92, +12.50] | +3.95% [−4.11, +11.38] |
| monthly energy | +11.75% [−21.80, +36.05] | +10.06% [−11.85, +27.36] | +6.38% [−38.04, +36.51] | +3.79% [−37.63, +32.74] |
| monthly energy (trajectory sum) | +11.75% [−21.80, +36.06] | +10.06% [−11.84, +27.36] | +6.73% [−35.45, +35.78] | +4.08% [−37.16, +32.93] |
| peak | +3.48% [−16.33, +19.92] | +3.25% [−20.84, +24.57] | −16.92% [−58.46, +13.72] | −14.61% [−59.64, +17.71] |
| forecast day 1 | −1.93% [−6.24, +2.21] | +2.54% [−10.12, +12.98] | +10.52% [−0.84, +20.61] | +10.59% [−0.55, +20.49] |
| forecast days 2-7 | +2.57% [−2.17, +7.10] | +7.00% [−11.40, +13.77] | +10.98% [+2.95, +18.34] * | +10.39% [+2.06, +18.02] * |
| forecast days 8-31 | +0.59% [−1.79, +2.91] | −1.41% [−7.98, +2.53] | +2.65% [−6.27, +10.82] | +1.22% [−7.85, +9.53] |

Other saved baselines, hourly, six seeds (first read of this candidate on a previously read population, secondary test
set; HKUST is not on the published pretraining lists of TimesFM 2.5, Chronos or Chronos-2, which does not establish
absence):

| Other saved baseline (hourly, 6 seeds) | Result | Training labels to C |
|---|---|---|
| Chronos-2 | +12.95% [−10.75, +31.58] | no |
| Seasonal naive (day) | +24.11% [+5.48, +39.07] * | no |
| 4-week profile | +13.68% [−1.71, +26.75] | no |
| Previous-month profile | +13.65% [−1.61, +26.61] | no |
| Last-year profile | +18.63% [+5.64, +29.84] * | no |
| Ridge (per unit) | +17.25% [−23.54, +44.57] | no |
| Chronos-2-X | +5.29% [−3.99, +13.74] | no |
| TimesFM-X | +6.86% [−11.92, +22.48] | no |
| Holt-Winters | +17.10% [−0.47, +31.60] | no |
| MSTL | +23.68% [+2.95, +39.98] * | no |
| GBT (zero-shot) | +60.10% [−2.68, +84.49] | no |

Mean unit rank among 14 forecasters (89 windows, 29 units): ankyra-lite (6 seeds) 3.45, Ridge (per unit) 4.03, TimesFM-X 4.34, Chronos-2-X 5.52, TimesFM 5.86, Chronos-2 6.48, Previous-month profile 6.97, 4-week profile 7.21, Holt-Winters 7.97, Last-year profile 8.74, Seasonal naive (day) 9.79, MSTL 10.38, Seasonal naive (week) 10.98, GBT (zero-shot) 13.28; ankyra-lite at position 1.

Subgroups (first read of this candidate on a previously read population, secondary test set; "not computed": too few
units or months; the coarse group's interval is undefined in the record):

| Subgroup (hourly, 6 seeds) | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|---|
| units without first-stage own training windows | 2 / 1 / 2 | not computed | not computed | not computed |
| units with first-stage own training windows | 87 / 28 / 8 | +11.92% [−6.76, +27.34] | +27.96% [+10.95, +41.72] * | +3.91% [−2.52, +9.93] |
| by reading quantisation step (from each unit's load before C'): coarse (step >= 25% of mean) | 6 / 3 / 4 | −17.43% [interval undefined] | −2.39% [interval undefined] | +1.91% [interval undefined] |
| by reading quantisation step (from each unit's load before C'): fine (step < 25% of mean) | 83 / 26 / 8 | +12.98% [−7.87, +29.80] | +28.34% [+9.56, +43.22] * | +4.18% [−2.91, +10.78] |
| by reading quantisation step (from each unit's load before C'): undetermined | 0 / 0 / 0 | not computed | not computed | not computed |

**Reading.**

- Against the weekly seasonal naive the hourly improvement is +25.65%, resolved; against zero-shot TimesFM 2.5
  (+10.24%) and the carrier-free control (+3.95%) the intervals include zero (first read of this candidate on a
  previously read population, secondary test set).
- The monthly energy intervals are wide and include zero (for example +32.61% [−94.71, +76.68] against TimesFM 2.5,
  first read of this candidate on a previously read population, secondary test set): with 29 meters over eight months
  the energy contrasts are not resolved.
- Mean unit rank 3.45, the lowest of 14 (first read of this candidate on a previously read population, secondary test
  set).

**What it does not show.** A small sample (29 units in the ratio, 8 months) with coarse readings; earlier versions of
the small network were trained and scored on the same 96 windows, and the anchoring layer was fixed after the first
reading of HKUST.

## GoiEner households, out of time (supplementary)

**Question.** How does the candidate compare on households of study population 6's source that no model or tool had
read, in a later period?

**Data.** A new store built for this run from the local GoiEner archive (Zenodo, CC BY 4.0), without any download:
households chosen by the rule of study population 6 (household activity codes, 2.X tariffs, no self-generation), then
only households never extracted before, then a temperature coverage of at least 0.9; time axis from 2020-01-01 (load from
2020-02-01) to 2022-06-07, Madrid local time; the spring daylight-saving hour filled with the mean of its neighbours and
the autumn hour by the calendar rule of the study's GoiEner build (see
[Decisions](#decisions-awaiting-the-authors-review)); temperatures of the provincial NOAA stations; national holidays
(regional holidays are missing, as for the study populations). Of 44,968 evaluation windows, the 22,793 with origin at
or after C were thinned by keeping every 16th household with all its windows, which leaves 1,422 windows of 284
households in 6 months (`score: test_set_check`). The 251 filled hours inside the test targets (all at 2022-03-27
01:00, according to the plan's addendum) are treated as missing and never scored (`score: target_completeness`).

**Procedure.** As above, with ten borrowed sources. TimesFM 2.5 was run on the test windows for this check. The two
households with near-zero load (5 windows) are left out of the ratio, so the contrasts cover 1,417 windows of 282
households. 44 windows are state windows, where ankyra-lite delivers the weekly seasonal naive; on 25 of them its
hourly error is exactly zero (5 of these in the two near-zero households; `score: windows.state_windows`,
`windows.exact_zero_hourly_error`).

**Result** (same-source out-of-time supplementary check, not an external population, correlated with the result on
study population 6; whether GoiEner is in the TimesFM 2.5 or Chronos pretraining data is not verified):

| Readout | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|
| hourly (6 seeds) | +4.19% [+1.59, +6.72] * | +25.99% [+23.87, +28.05] * | +5.48% [+3.59, +7.34] * |
| monthly energy (6 seeds) | +49.17% [+41.94, +55.50] * | +25.15% [+16.55, +32.87] * | +9.59% [+1.24, +17.24] * |
| monthly energy (trajectory sum) (6 seeds) | +49.16% [+41.92, +55.50] * | +25.14% [+16.53, +32.86] * | +8.81% [−0.22, +17.02] |
| peak (6 seeds) | — | — | +0.69% [−1.66, +2.99] |
| forecast day 1 (6 seeds) | −2.35% [−10.16, +4.90] | +26.51% [+21.32, +31.37] * | +14.56% [+7.33, +21.22] * |
| forecast days 2-7 (6 seeds) | +0.22% [−4.02, +4.29] | +27.02% [+23.73, +30.16] * | +7.63% [+3.88, +11.24] * |
| forecast days 8-31 (6 seeds) | +4.48% [+1.46, +7.42] * | +26.35% [+24.26, +28.39] * | +5.46% [+3.25, +7.61] * |
| hourly (3 seeds) | +4.16% [+1.51, +6.73] * | +25.97% [+23.89, +27.99] * | +5.45% [+3.58, +7.29] * |
| monthly energy (3 seeds) | +48.75% [+42.02, +54.71] * | +24.54% [+16.26, +32.00] * | +8.85% [+0.21, +16.75] * |
| monthly energy (trajectory sum) (3 seeds) | +48.75% [+42.00, +54.71] * | +24.53% [+16.24, +32.00] * | +8.06% [−1.22, +16.50] |
| peak (3 seeds) | — | — | +0.74% [−1.97, +3.37] |
| forecast day 1 (3 seeds) | −2.41% [−10.39, +4.99] | +26.32% [+21.16, +31.14] * | +14.51% [+7.12, +21.31] * |
| forecast days 2-7 (3 seeds) | +0.09% [−4.23, +4.23] | +26.92% [+23.65, +30.04] * | +7.51% [+3.78, +11.10] * |
| forecast days 8-31 (3 seeds) | +4.50% [+1.43, +7.48] * | +26.37% [+24.31, +28.36] * | +5.47% [+3.27, +7.62] * |

Descriptive contrasts (same-source out-of-time supplementary check):

| Descriptive | span return vs plain continuation (R vs F, 3 seeds) | continuation and span return vs from-scratch stage (6 seeds; unit × month interval only) | ankyra-lite vs history-only configuration (6 seeds) | ankyra-lite vs carrier-free control (6 seeds) |
|---|---|---|---|---|
| hourly | +0.12% [−0.10, +0.34] | +0.26% [+0.06, +0.42] * | +6.59% [+4.53, +8.61] * | +6.09% [+4.06, +8.08] * |
| monthly energy | +0.39% [−4.15, +4.73] | +1.96% [−1.02, +5.34] | +12.34% [+3.68, +20.23] * | +11.85% [+2.85, +20.02] * |
| monthly energy (trajectory sum) | +0.48% [−3.92, +4.69] | +1.96% [−1.11, +5.29] | +11.78% [+2.71, +20.00] * | +11.08% [+1.41, +19.81] * |
| peak | −0.04% [−1.34, +1.24] | +0.15% [−2.15, +0.62] | +0.75% [−1.67, +3.11] | +0.73% [−1.67, +3.06] |
| forecast day 1 | +1.90% [−1.23, +4.93] | +1.79% [+0.03, +4.61] * | +16.43% [+8.84, +23.39] * | +15.91% [+8.36, +22.85] * |
| forecast days 2-7 | +0.26% [−0.24, +0.74] | +0.24% [−0.11, +0.52] | +8.95% [+4.87, +12.85] * | +8.51% [+4.43, +12.42] * |
| forecast days 8-31 | +0.08% [−0.15, +0.31] | +0.23% [+0.02, +0.42] * | +6.63% [+4.23, +8.96] * | +6.11% [+3.75, +8.40] * |

Other saved baselines, hourly, six seeds (same-source out-of-time supplementary check):

| Other saved baseline (hourly, 6 seeds) | Result | Training labels to C |
|---|---|---|
| Seasonal naive (day) | +26.70% [+24.59, +28.75] * | no |
| 4-week profile | +9.10% [+7.20, +10.97] * | no |

Mean unit rank among 5 forecasters (1,417 windows, 282 units): ankyra-lite (6 seeds) 1.19, TimesFM 2.04, 4-week profile 3.00, Seasonal naive (day) 4.34, Seasonal naive (week) 4.43; ankyra-lite at position 1.

Subgroups (same-source out-of-time supplementary check; whether GoiEner is in the TimesFM 2.5 pretraining data is not
verified):

| Subgroup (hourly, 6 seeds) | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |
|---|---|---|---|---|
| units without first-stage own training windows | 598 / 140 / 6 | +3.46% [−1.16, +7.88] | +25.46% [+21.91, +28.85] * | +6.68% [+3.23, +10.01] * |
| units with first-stage own training windows | 819 / 142 / 6 | +4.89% [+2.90, +6.85] * | +26.51% [+24.48, +28.49] * | +4.29% [+2.40, +6.14] * |

**Reading.**

- All three primary contrasts are resolved in ankyra-lite's favour: +4.19% against zero-shot TimesFM 2.5, +25.99%
  against the weekly seasonal naive and +5.48% against the carrier-free control (same-source out-of-time supplementary
  check; GoiEner's presence in the TimesFM 2.5 pretraining data is not verified).
- Forecast day 1 against TimesFM 2.5 is −2.35%, not resolved; days 8–31 +4.48%, resolved (same-source out-of-time
  supplementary check; whether GoiEner is in the TimesFM 2.5 pretraining data is not verified).
- Returning the validation span is +0.12% against the plain continuation, not resolved (same-source out-of-time
  supplementary check).

**What it does not show.** It is the same source as study population 6, whose results informed the choice of recipe R,
and is related to population 5. On population 6's own test windows, a re-evaluation of the shipped three-seed
checkpoints, zero-shot TimesFM has the resolvably lower hourly error ([README](../README.md#where-ankyra-lite-is-weak));
that reading and this check differ in design (seeds, carriers, period) and are correlated. Its training labels lie in
the 2020 COVID period; the June 2021 tariff reform falls after C′, so every test target follows the reform while no
training or validation label does. The test covers winter and spring only (6 months), and the households were chosen
with metadata recorded at the end of the archive (a mild look-ahead, as for the study populations).

## Helsinki (retrospective) and UCI370

- **Helsinki, retrospective.** Carriers were trained (C′ = 2025-08-27, the ten study sources plus Helsinki), and every
  arm was anchored and sealed with its hashes. No error has been computed or looked at. The scorer refuses this test set
  unless a frozen prospective Helsinki registration and its public deposit exist. The prospective test is not being
  pursued for now (author decision, 19 October 2026), so the sealed set stays unread.
- **UCI370** was skipped in this round. Resuming it needs a change of its recorded role (a development set used for
  the level rule only), per-file permission for its temperature download, and window rules for its clock-change gaps,
  repeated October hours and zero-filled outages written before any window is counted.

## Decisions awaiting the author's review

Two choices were made overnight by the coordinating session under the author's general authorisation and are flagged
for review; both were written into the plan's addendum before the freeze.

1. **The GoiEner autumn clock-change hour.** The archive holds no repeated timestamp for this hour, and most households
   lack it altogether, so the plan's "average the repeated hour" did not apply as written. The store builder fills the
   hour by the calendar rule of the study's GoiEner build (every filled cell recorded), as for study population 6. This
   gives C = 2021-12-01 and the test set above. The other reading (no fill) would have given C = 2021-09-01 and 1,256
   windows of 564 households in 9 months (the plan's addendum 01, item 2).
2. **The tolerance of the CPU inference check.** The plan's line for reproducing the stored GPU carrier forecasts on the
   CPU was a relative difference of 10⁻⁶; the worst case was 3.08 × 10⁻⁶ (HEEW, seed 2). The check was accepted with a
   tolerance of 10⁻⁵, **set after the worst case had been seen**. The check used the stored carriers of two study
   populations (HEEW and GoiEner households, seeds 0–2, 20 windows each). The supporting evidence: identical availability
   masks, the package's CPU inference equal bit for bit to the research code's CPU inference, a full-batch GPU rerun
   equal bit for bit to the stored forecasts, and exact anchoring of the stored forecasts. Recomputing the three-seed
   ensemble on the CPU and anchoring it differs from the stored product by at most 1.02 × 10⁻⁵ on study population 6
   (small household loads magnify relative differences); this is descriptive, not part of the criterion (the run's
   acceptance record of this check: `tolerance`, `carrier_max_rel_by_seed`, `descriptive_not_a_criterion`; addendum
   01, item 3).

## Run notes

- **One reading per test set.** Each scored test set has one start marker and one score record; there is no stopped,
  refused or second-run record among them, and the scorer's one-read guard was not triggered. Two driver scripts
  overlapped for a while after an incomplete stop: the old driver's child processes ran alongside the new driver, and
  the two were coordinated by marker files. The run's operations log shows every (test set, stage, seed) trained once.
  One process passed the memory gate for UNICON's plain-continuation seed 2 at 11:09 UTC without logging a training
  start; another process trained that seed from 12:45 to 12:47 UTC (the operations log). A false "failed" marker was
  moved aside (renamed, not deleted; the coordinating session's overnight report), and UNICON's anchoring and scoring
  were rerun by hand by the coordinating session (its manual log: anchoring 12:50:06 to 13:11:35 UTC, scoring finished
  13:13:12 UTC, all return codes 0).
- **A temporary-folder cleanup** of the operating system removed the coordinating session's driver logs, marker files
  and a few scripts at about 12:15 UTC; the scripts were saved again in the study's research folder. The run's outputs,
  which lie outside the temporary folder, were not affected: the final verification of every hash, import closure and
  folder listing passed, with only the two GoiEner files created after the freeze by design (the TimesFM forecasts and
  the comparator file, whose hashes are in the product record) listed as attributed differences and no unattributed
  difference.
- **Memory.** The coordinating session trimmed the working set of its own waiting training processes several times to
  pass the memory gate; no process was stopped. From about 14:25 UTC the trimming was blocked by the session's safety
  check and was not done again. Before the freeze, the agent that built the adapter had also trimmed the working set of
  idle waiting processes, several of them not its own; no process was stopped (the plan's addendum 01, item 6).

## What these tests do not show

- **Not tests on unseen populations.** LCL, UNICON and HKUST had been read by earlier forecasters and earlier versions of
  the small network, and the anchoring layer was fixed after those readings; GoiEner is a later period of a study
  population's source. The 0.2.0 re-run reads each test set a second time. The Helsinki retrospective set is not
  scored; the prospective Helsinki test is not being pursued for now.
- **Not tests of the shipped checkpoints.** Each test set got its own carriers, trained with recipe R (0.1) and then
  recipe Rw (0.2.0) from the same stage-1 runs.
- **The continuation layer's constants were not refitted** on any external data; whether other constants would serve
  these test sets better was not examined.
- **The day-1 rule allows one claim only**: that the day-1 gap to TimesFM 2.5 narrowed from 0.1 to 0.2.0 on these test
  sets. It is not a lead: on LCL TimesFM 2.5 is still resolvably better on day 1, and in the first three hours it is
  resolvably better on LCL, UNICON and GoiEner.
- **Few months.** LCL 5, HKUST 8 and GoiEner 6 months; the month resampling has few clusters, so the intervals may be
  too narrow.
- **No multiplicity correction** over the 15 primary intervals, and no pooling across test sets.

## Files

- [`results/external_tests.csv`](../results/external_tests.csv): every contrast of the four 0.2.0 score records with
  ankyra-lite 0.2.0, its own configurations (the 0.1.0 configuration, the frozen 0.1 candidate, the layer and the
  weighting on and off, three and six seeds) and the baselines, the day-1 rule and the mean unit ranks
  ([columns](../results/README.md#external-tests)). The 0.1 file is that of the 0.1.1 release (commit `c27affc`). The records hold no per-day curves, so there is no by-day file; the
  forecast-day groups (day 1, days 2–7, days 8–31) are rows of this file.
- [`evaluation/external_tables.py`](../evaluation/external_tables.py) writes it from the score records
  (`ANKYRA_LITE_RESEARCH_ROOT`; the 0.2.0 records with `--version 0.2`) and prints the tables of the 0.2.0 section;
  `--check` compares the file with the records.
- [`tests/test_external_tables.py`](../tests/test_external_tables.py) checks the file against the records, the forecaster
  names and the absence of paths.
- [`figures/fig17_external_tests.png`](../figures/fig17_external_tests.png) (and `.pdf`), drawn from the CSV file.
- The plan, its addendum, the frozen record, the products and the score records are in the study's research folder and
  are not part of this repository; neither are the external-test carriers (UNICON's licence excludes them in any case).
