# Later tests and stronger comparators (22 October – 6 November 2026)

This page collects the evidence on ankyra-lite obtained after the comparison of [EVALUATION.md](EVALUATION.md) and the
external re-run of [EXTERNAL_TESTS.md](EXTERNAL_TESTS.md), for the paper in preparation.

| Section | Data | Identity |
|---|---|---|
| [openMeter](#openmeter-germany-a-pre-specified-test-on-a-new-source) | German households, public buildings and businesses; test months 2024-02 to 2025-07 | a new source in an earlier period; the test was fixed in writing before any of its load values were read and scored once (3 November 2026); the only confirmatory test |
| [HEAPO and Cambridge 2023](#heapo-and-cambridge-2023) | heat-pump households in the canton of Zurich; the 2023 months of the Cambridge estate | new data; a new period of a study population; each scored once (22 October 2026); stronger comparators added later on the same windows (descriptive) |
| [Nine populations](#nine-populations-a-fixed-trained-learner-and-the-network-trained-to-the-test-start) | the nine populations of the comparison | test windows read during development; descriptive |
| [Fine-tuned Chronos-2](#nine-populations-chronos-2-fine-tuned-on-the-same-training-pool) | the nine populations of the comparison | a comparison fixed in writing before it was read, read once (6 November 2026) on test windows read during development; not a confirmation |

- **Versions.** HEAPO and Cambridge 2023 ran with ankyra-lite 0.2.0 and openMeter with 0.2.1 (exported read-only from
  its commit); the package of 0.2.2 differs from 0.2.1 only in its version string, and 0.2.1 differs from 0.2.0 in
  behaviour only in filling missing day types of old windows and in rejecting non-finite values in the continuation layer.
- **Where the numbers come from.** The tables are copied, by a script of the study, from the paper's supplement, whose
  numbers were formatted once from the score records; labels such as "S11.1", "Section 3.2" or "Table 5" refer to the
  paper. The openMeter readings of Table O2 are also in
  [`results/openmeter_test.csv`](../results/openmeter_test.csv) at full precision.
- **Names.** L: ankyra-lite as delivered (the network, the anchoring layer with the state rule, and the continuation
  layer); in Table N2 only, L is ankyra-lite before the continuation layer and Lc after it. N: the network alone (six
  seeds, no anchoring layer). G: a fixed gradient-boosted learner fitted per population on the network's training pool
  ([below](#a-fixed-gradient-boosted-learner-g)). $N_C$: the network retrained with labels up to the test start $C$ and
  used alone. FTC, C2T, C2Y and B8: [below](#nine-populations-chronos-2-fine-tuned-on-the-same-training-pool). Readings that involve the paper's TimesFM configuration of the same anchoring layer are reported in the paper,
  not here, as in the rest of this repository. One-year TimesFM 2.5: up to 8,736 h of context. Positive
  percentages: the first-named forecaster has the lower error.

## openMeter, Germany: a pre-specified test on a new source

**Result.** ankyra-lite's monthly energy error was 31.0% lower than one-year TimesFM 2.5's (seed layer A [+24.8, +36.6], seed layer B
[+24.9, +36.6], units × months [+25.2, +37.0]; 7,894 windows of 1,045 units over 14 origin months): confirmed under the registered rule.

- On the 89% of windows with complete targets the reading was 31.9% [+25.2, +38.0]; 11% of the
  windows had incomplete targets.
- Households (34.8%) and businesses (19.4%) were confirmed; public buildings (13.7%, seed layer A [−7.6, +30.8]) were not. The
  three categories cover different months (Table O1).
- The lead was in energy. Hourly error was 3.1% lower (seed layer A [+1.3, +4.8]; units × months [−0.1, +3.6]); day 1 against the
  foundation-model settings (+2.1% to +3.3%) and the peak readout against last month's observed peak
  (+1.5%, [−1.7, +4.6]) were not resolved; G came closest in energy (8.4%, [+4.9, +11.8]).
- Fine-tuned Chronos-2, N-HiTS and TFT were not run on this source.
- The result covers this source, this period, the frozen configuration and the registered metric. It is not a
  zero-shot result, and it does not show which part of the configuration produces the lead.

**Identity.** The test was fixed in writing before any of its load values were read, and scored once. The protocol (28 October 2026) was
written before any openMeter metadata had been seen; amendment 01 (30 October; secondary readings and reporting
duties) was hashed before the first metadata call. The metadata snapshot (31 October 2026, 20:20:50 UTC) decided the
branch mechanically: for test starts from November 2025, after both foundation models' release, only one sensor
qualified, and its metadata are erroneous, so branch A failed and branch B (a new source, an earlier period) set
$C$ = 1 February 2024, the latest month start before November 2025 with at least 200 qualifying units (1,062: 824
households, 190 public buildings, 48 businesses). The test units were fixed by these metadata. Contrary to the
protocol's order, the time series and weather files were downloaded (with per-file approval) before the freeze; before
the freeze only record counts, first and last timestamps of each page, file hashes and the presence of an hourly
temperature were read, and no load value. Amendment 02 (build rules for five cases seen in those counts and timestamps)
was approved with the freeze on 1 November 2026, before any load value was read. Scoring ran once on 3 November 2026
(15:37 UTC) under an exclusive guard, after a dry run on synthetic targets with the same pre-scoring checks.

**Data processing.** Meter readings were differenced (a negative difference is missing); quarter-hours were summed to
hours (any missing quarter makes the hour missing); exact duplicate records were dropped and conflicting ones made
missing; readings off the 15-min or hourly grid and negative or non-finite interval values were missing; the repeated
autumn clock hour was placed by record order; all series were moved to a fixed UTC+1 grid. No value was removed for
its size. Holidays are each unit's state holidays as published at the origin (the calendar check ran before any load
was read). Temperature is the platform's weather series for the 1,034 test units with at least 90% hourly coverage; 27 of the
other 28 use the hourly median of covered locations in the same two-digit post-code area, and one, a business without
any eligible window, has no temperature source. Only
temperature before the origin is used (climatology and recent anomaly). Post codes and locations are used for these
assignments only and are not released.

**Windows.** As for HEAPO: history for six pseudo-origins, a complete context
$[o-1{,}344, o)$, at least 75% of the 744 target hours observed; missing target hours are not scored. Of the 1,062
listed units, 1,048 had at least one eligible window (7,915 windows over the 18 listed origins; eligible windows exist
for 14 of them). Three units with near-zero realised load (one business, two households; 21 windows) were excluded by
the paper's near-zero rule, leaving 7,894 windows of 1,045 units. The state rule delivered 90 windows (households
37, public buildings 53, businesses 0).

**Table O1.** Windows by origin month and category: eligible / scored (after the near-zero exclusion) / complete
targets. — no eligible window. No origin from September to December 2024 has an eligible window.

| Origin month | Households: eligible / scored / complete | Businesses: eligible / scored / complete | Public buildings: eligible / scored / complete | Scored, all |
|---|---|---|---|---|
| 2024-02 | 811 / 809 / 809 | 47 / 46 / 46 | 166 / 166 / 166 | 1,021 |
| 2024-03 | 824 / 822 / 822 | 47 / 46 / 46 | 166 / 166 / 165 | 1,034 |
| 2024-04 | 824 / 822 / 822 | 47 / 46 / 46 | 164 / 164 / 163 | 1,032 |
| 2024-05 | 824 / 822 / 822 | 47 / 46 / 46 | 171 / 171 / 171 | 1,039 |
| 2024-06 | 824 / 822 / 822 | 47 / 46 / 46 | — | 868 |
| 2024-07 | 824 / 822 / 822 | 47 / 46 / 46 | — | 868 |
| 2024-08 | 813 / 811 / 0 | 47 / 46 / 0 | — | 857 |
| 2025-01 | — | — | 165 / 165 / 164 | 165 |
| 2025-02 | — | — | 169 / 169 / 169 | 169 |
| 2025-03 | — | — | 170 / 170 / 169 | 170 |
| 2025-04 | — | — | 169 / 169 / 167 | 169 |
| 2025-05 | — | — | 167 / 167 / 166 | 167 |
| 2025-06 | — | — | 166 / 166 / 166 | 166 |
| 2025-07 | — | — | 169 / 169 / 169 | 169 |
| All | 5,744 / 5,730 / 4,919 | 329 / 322 / 276 | 1,842 / 1,842 / 1,835 | 7,894 |

**Training and comparators.** ankyra-lite 0.2.1 (commit 14db352; exported read-only from git) was retrained exactly as
for HEAPO: the ten borrowed research populations, each cut at its own $C'$, plus openMeter as one population (the
1,062 test units and 199 training-only units, data before $C'$ = 30 July 2023; validation on openMeter's own weekly
windows in $[v_0, C')$, $v_0$ = 28 April 2023); categories mapped to Residential, Public and Commercial; six seeds, two
stages, one process per seed and stage. HEAPO had been run with 0.2.0, which differs only in filling missing old
day types and in rejecting non-finite values in the continuation layer. G, the fixed gradient-boosted learner described below, uses the same code and hyperparameters on the same pool. TimesFM 2.5 (Hugging Face google/timesfm-2.5-200m-pytorch, revision 1d952420; repository created
2 September 2025) and Chronos-2 (amazon/chronos-2, revision 29ec3766; repository created 30 October 2025) ran zero-shot
from local checkpoints whose hashes were recorded at the freeze; batch size 1. One-year TimesFM 2.5 read a median
context of 8,736 h (80.6% of windows at the 8,736-h maximum; minimum
1,353 h). Fine-tuned Chronos-2, N-HiTS and TFT, registered as conditional secondary comparators,
were not run: the freeze decided not to train them within the test's deadline.

**Estimands and the verdict.** The registered readout rule takes the hourly curve after the continuation layer and
energy and peak before it. On a window, ankyra-lite's energy error is its level minus the observed-hour mean load; a
curve-only forecaster's is the mean of its curve over the observed hours minus the same mean. Per unit, $R_i$ is the
RMS over its windows; contrasts are the equal-unit mean log ratio of the evaluation over windows where every arm is
finite. Intervals: units × months crossed bootstrap (2,000 draws; seeds 20261007 for energy, 20261005 for hourly and day-1
and 20261009 for peak readings); seed layer A adds the spread of the six single-seed contrasts (Welch–Satterthwaite t); seed layer B adds the
spread of two three-seed groups (normal). Confirmed: all three lower bounds above zero; direction consistent but not
confirmed: positive point estimate with at least one interval including zero; not supported: point estimate at or below
zero. The six single-seed log ratios of the primary contrast were 0.339, 0.350, 0.366, 0.318, 0.350, 0.356; the two groups, 0.368, 0.364.

**Table O2.** The registered readings, % lower error for ankyra-lite [95% interval]; the three that involve the paper's TimesFM configuration of the anchoring layer are reported in the paper. Secondary readings are
unadjusted and outside the verdict. Peak contrasts against curve maxima compare a readout with the maximum of a
smooth curve, which is biased low; they are not evidence of a better peak forecast.

| Reading (ankyra-lite against) | % lower error | Seed layer A (Welch) | Seed layer B (two groups) | Units × months | Windows / units / months | Verdict |
|---|---|---|---|---|---|---|
| **Primary:** monthly energy, one-year TimesFM 2.5 | +31.0 | [+24.8, +36.6] | [+24.9, +36.6] | [+25.2, +37.0] | 7,894 / 1,045 / 14 | confirmed under the registered rule |
| Monthly energy, the same, complete targets only | +31.9 | [+25.2, +38.0] | [+25.3, +38.0] | [+25.2, +37.9] | 7,030 / 1,045 / 13 |  |
| Monthly energy, one-year TimesFM 2.5 mean output | +18.3 | [+11.9, +24.3] | [+12.0, +24.2] | [+11.2, +23.7] | 7,894 / 1,045 / 14 |  |
| Monthly energy, eight-week TimesFM 2.5 | +33.3 | [+27.7, +38.5] | [+27.8, +38.4] | [+27.9, +38.7] | 7,894 / 1,045 / 14 |  |
| Monthly energy, Chronos-2 with covariates | +38.5 | [+33.8, +42.9] | [+33.8, +42.8] | [+33.8, +42.8] | 7,894 / 1,045 / 14 |  |
| Monthly energy, G (fixed gradient boosting) | +8.4 | [+4.9, +11.8] | [+5.1, +11.7] | [+5.3, +11.7] | 7,894 / 1,045 / 14 |  |
| Hourly, one-year TimesFM 2.5 | +3.1 | [+1.3, +4.8] | [+1.3, +4.8] | [−0.1, +3.6] | 7,894 / 1,045 / 14 |  |
| Day 1, one-year TimesFM 2.5 | +2.4 | [−5.7, +9.8] | [−5.7, +9.8] | [−9.7, +6.1] | 7,894 / 1,045 / 14 |  |
| Day 1, one-year TimesFM 2.5 mean output | +2.1 | [−4.2, +8.1] | [−4.2, +8.1] | [−5.7, +6.8] | 7,894 / 1,045 / 14 |  |
| Day 1, eight-week TimesFM 2.5 | +3.3 | [−7.0, +12.5] | [−7.0, +12.5] | [−13.6, +7.0] | 7,894 / 1,045 / 14 |  |
| Day 1, Chronos-2 with covariates | +2.6 | [−6.3, +10.7] | [−6.3, +10.7] | [−11.3, +6.4] | 7,894 / 1,045 / 14 |  |
| Day 1, G | +6.1 | [+2.4, +9.6] | [+2.4, +9.6] | [+5.5, +12.3] | 7,894 / 1,045 / 14 |  |
| Day 1, weekly seasonal naive | +32.4 | [+26.6, +37.7] | [+26.6, +37.7] | [+26.8, +37.9] | 7,894 / 1,045 / 14 |  |
| Peak readout, last month's observed peak | +1.5 | [−1.7, +4.6] | [−1.7, +4.6] | [−2.8, +3.5] | 7,894 / 1,045 / 14 |  |
| Peak readout, one-year TimesFM 2.5 curve maximum | +66.4 | [+62.5, +69.9] | [+62.5, +69.9] | [+65.9, +72.3] | 7,894 / 1,045 / 14 |  |
| Peak readout, one-year TimesFM 2.5 mean-output maximum | +65.2 | [+61.2, +68.9] | [+61.2, +68.9] | [+64.6, +71.4] | 7,894 / 1,045 / 14 |  |
| Peak readout, eight-week TimesFM 2.5 curve maximum | +66.2 | [+62.5, +69.5] | [+62.5, +69.5] | [+65.6, +71.9] | 7,894 / 1,045 / 14 |  |
| Peak readout, Chronos-2 with covariates curve maximum | +64.9 | [+61.2, +68.3] | [+61.2, +68.3] | [+64.3, +70.8] | 7,894 / 1,045 / 14 |  |
| Peak readout, G curve maximum | +66.3 | [+61.9, +70.2] | [+61.9, +70.2] | [+65.8, +72.9] | 7,894 / 1,045 / 14 |  |
| Peak readout, weekly seasonal naive maximum | +30.4 | [+23.5, +36.7] | [+23.5, +36.7] | [+21.7, +35.2] | 7,894 / 1,045 / 14 |  |
| Households: monthly energy, one-year TimesFM 2.5 | +34.8 | [+28.3, +40.6] | [+28.4, +40.5] | [+28.6, +40.6] | 5,730 / 822 / 7 | confirmed under the registered rule |
| Households: the same, mean output | +20.5 | [+13.1, +27.2] | [+13.2, +27.1] | [+12.5, +26.4] | 5,730 / 822 / 7 |  |
| Households: monthly energy, eight-week TimesFM 2.5 | +35.3 | — | — | [+30.2, +41.0] | 5,730 / 822 / 7 |  |
| Households: monthly energy, Chronos-2 with covariates | +41.8 | — | — | [+36.9, +46.9] | 5,730 / 822 / 7 |  |
| Households: monthly energy, G | +7.9 | — | — | [+5.3, +11.1] | 5,730 / 822 / 7 |  |
| Businesses: monthly energy, one-year TimesFM 2.5 | +19.4 | [+2.8, +33.3] | [+2.7, +33.3] | [+2.0, +32.5] | 322 / 46 / 7 | confirmed under the registered rule |
| Businesses: the same, mean output | +10.5 | [−6.4, +24.7] | [−6.4, +24.7] | [−9.2, +22.6] | 322 / 46 / 7 |  |
| Businesses: monthly energy, eight-week TimesFM 2.5 | +28.5 | — | — | [+9.5, +39.4] | 322 / 46 / 7 |  |
| Businesses: monthly energy, Chronos-2 with covariates | +29.3 | — | — | [+8.4, +41.7] | 322 / 46 / 7 |  |
| Businesses: monthly energy, G | +13.3 | — | — | [+0.8, +24.1] | 322 / 46 / 7 |  |
| Public buildings: monthly energy, one-year TimesFM 2.5 | +13.7 | [−7.6, +30.8] | [−7.6, +30.8] | [−7.0, +28.7] | 1,842 / 177 / 11 | direction consistent, not confirmed |
| Public buildings: the same, mean output | +9.7 | [−10.3, +26.1] | [−10.3, +26.1] | [−10.3, +23.9] | 1,842 / 177 / 11 |  |
| Public buildings: monthly energy, eight-week TimesFM 2.5 | +24.7 | — | — | [+7.5, +37.8] | 1,842 / 177 / 11 |  |
| Public buildings: monthly energy, Chronos-2 with covariates | +23.6 | — | — | [+10.6, +34.4] | 1,842 / 177 / 11 |  |
| Public buildings: monthly energy, G | +9.9 | — | — | [−4.8, +21.8] | 1,842 / 177 / 11 |  |

Month-t reading: +21.4% (t interval over 14 target months [+5.8, +34.4]; months weighted equally; not part of the verdict). Common bias removed in hindsight (descriptive): +27.0% after removal [+23.1, +32.6], against +31.0% before; share of the log difference removed 15.0% [−3.4, 25.3]; 25 category × month cells, re-estimated within every resample.

**Verification.** The pre-scoring check list (21 items, all passed) covered the frozen-file and tool hashes, the
product files against the test view, the absence of negative targets, the comparator coverage and the state windows.
Fifty helper modules imported at run time by the training, G and comparator jobs were added to the tool record after
execution, with hashes equal to those the jobs had recorded; nothing was retrained (author's decision, 3 November
2026). An independent script, written from the scoring definitions without calling the scoring code, reproduced 54
values (the primary point estimate, all seed and group contrasts, both seed-layer intervals given the scorer's units × months standard error, the complete-target,
mean-output, energy, hourly, category and month-t point estimates, and the window, unit and month counts), all within
$10^{-15}$; its own units × months bootstrap gave a standard error of 0.0423 against the scorer's 0.0430. The common-bias,
peak and day-1 readings were not reproduced independently.

**Pretraining exposure.** The target months precede both models' release, so exposure cannot be excluded. The
published lists were read on 3 November 2026 in the manner of the earlier check of these lists (October 2026). At the frozen revision, the TimesFM 2.5
model card names the GIFT-Eval pretraining collection, Wikimedia pageviews, Google Trends and synthetic data. The
collection's file listing (152 data sets) and the papers describing GIFT-Eval (Aksu et al., 2024) and LOTSA (Woo et
al., 2024) were searched for the source's name and for Germany, and every energy set in the collection was traced to its
stated source and institution; Table 6 of the Chronos-2 paper (Ansari et al., 2025; its 23 real pretraining data sets)
and Table 3 of the Chronos paper (Ansari et al., 2024) were read the same way. openMeter was not found. The only German item, an electricity-price task in the
benchmark on which Chronos-2 is evaluated, is not training data. One single aggregated series in the collection (ELF)
could not be traced to a location; it is not a set of meters. Absence is not thereby established: series can enter a
corpus under another name, and the models' other training data (Wikimedia pageviews and Google Trends for TimesFM 2.5,
synthetic data for both) are not listed series by series. The source's public GitHub files were last uploaded on
19 October 2023 (page read on 28 October 2026), before the test targets begin; this does not establish absence either.

**Deviations and disclosures.** (1) Downloads preceded the freeze (see Identity). (2) Amendment 02 was written after
the downloads, from counts and timestamps only. (3) Fifty run-time helper files entered the tool record after execution
(Verification). (4) Forty-one integer post codes had lost their leading zero; the build restored them before training
(author's decision of 2 November 2026), which moved the temperature fill of 13 units (11 test, 2 training-only) from
the state to the post-code-area tier. (5) The conditional comparators were not run.

**Licence.** Contains information from openMeter, which is made available here under the Open Database License
(ODbL 1.0). The meter and weather series, the post codes and the prepared store are not redistributed; the tables
above are derived results.

## HEAPO and Cambridge 2023

**Identity.** HEAPO (Brudermueller et al., 2025) is new data: heat-pump households in the canton of Zurich, never
touched by the study. Cambridge 2023 is the 2023 months of the Cambridge estate: a new period of a study population
whose development windows helped fit the continuation layer and whose training set contains the old Cambridge test
period. For each set the network was retrained from scratch on the set's own pool with the released recipe and six
seeds; the continuation layer's constants were not refitted, and three contrasts (TimesFM 2.5, the weekly seasonal
naive and the carrier-free control) were fixed in advance. Each set was scored once, under rules written before any
target was read. HEAPO: test start 2023-08-01, 4,326 windows of 1,301 households over
4 origin months (the provider removed the autumn clock-change days, so no November or December origin is
eligible). Cambridge 2023: test start 2023-01-01, 753 windows of 78 buildings over
12 origin months. HEAPO's energy and peak are read on observed hours only (a level proxy and an observed
peak). Neither set was found by name in the published pretraining lists of TimesFM 2.5, Chronos or Chronos-2; absence
is not established.

**Table H1.** Primary contrasts, whole-window hourly error, both seed readings (HEAPO and Cambridge 2023).

| Set | Against | Improvement % | Reading A (Welch, six seeds) | Reading B (two three-seed groups) | Verdict |
|---|---|---|---|---|---|
| HEAPO | TimesFM 2.5 (zero-shot) | +7.35 | +2.30 to +12.15 | +2.32 to +12.13 | \* |
| HEAPO | weekly seasonal naive | +22.41 | +20.56 to +24.23 | +20.59 to +24.19 | \* |
| HEAPO | carrier-free control | +4.75 | +3.27 to +6.22 | +3.34 to +6.15 | \* |
| Cambridge 2023 | TimesFM 2.5 (zero-shot) | +13.86 | +5.69 to +21.33 | +5.58 to +21.42 | \* |
| Cambridge 2023 | weekly seasonal naive | +28.16 | +18.97 to +36.30 | +18.90 to +36.36 | \* |
| Cambridge 2023 | carrier-free control | +8.53 | +4.56 to +12.33 | +4.34 to +12.53 | \* |

The two readings agreed on all six verdicts. With the continuation layer on both sides, ankyra-lite was +4.23% (HEAPO) and +7.33%
(Cambridge 2023) better than the control, both resolved. Mean unit rank by hourly RMSE among seven
forecasters (1 best): HEAPO ankyra-lite 1.39, control 2.93, four-week profile 3.31, TimesFM 2.5 3.75, Chronos-2 3.93, daily
seasonal naive 6.26, weekly seasonal naive 6.44; Cambridge 2023 ankyra-lite 1.63, control 2.90, TimesFM 2.5 3.50, Chronos-2
3.78, four-week profile 3.88, weekly seasonal naive 5.76, daily seasonal naive 6.55.

**Table H2.** Other readouts of the primary contrasts (descriptive; same reading; unit × month intervals). HEAPO: observed-hours level proxy and observed peak as defined in S11.1; Cambridge 2023: complete-window energy and peak.

| Readout | HEAPO vs TimesFM 2.5 | HEAPO vs weekly seasonal naive | HEAPO vs control | Cambridge 2023 vs TimesFM 2.5 | Cambridge 2023 vs weekly seasonal naive | Cambridge 2023 vs control |
|---|---|---|---|---|---|---|
| Window energy / HEAPO level proxy (separate readout) | +43.31\* | +24.83 | +21.71\* | +25.80\* | +34.11\* | +10.07 |
| Window energy / HEAPO level proxy (curve) | +43.34\* | +24.87 | +21.74\* | +25.81\* | +34.12\* | +10.04 |
| Peak readout (HEAPO: observed peak) | +70.41\* | +33.84\* | +2.24 | +46.87\* | +39.94\* | +6.93 |
| Hours 0–2 | −20.82 | +20.07 | +17.74\* | −17.86 | +36.66\* | +37.05\* |
| Day 1 | −4.01 | +26.99\* | +15.23\* | +1.09 | +34.68\* | +24.20\* |
| Days 2–7 | +2.89 | +26.46\* | +5.73\* | +5.03 | +24.85\* | +13.31\* |
| Days 8–31 | +8.14\* | +21.81\* | +4.39\* | +15.12\* | +28.46\* | +7.54\* |

The peak readout is fair only against the carrier-free control, which has the same readout: HEAPO +2.24 and Cambridge
2023 +6.93, neither resolved. Against TimesFM 2.5, Chronos-2 and the seasonal naive the comparator's peak is its curve
maximum, which flatters ankyra-lite (+70.41 and +46.87 against TimesFM 2.5); these columns are not evidence.
In the first hours and on day 1 the point estimates against TimesFM 2.5 are negative or near zero on both sets, not
resolved; no short-term lead over TimesFM 2.5 is claimed.

**Stronger comparators, added later (descriptive).** After the scored-once reading, rules fixed in writing added, on
the same windows, one-year TimesFM 2.5 (median and mean outputs), TimesFM 2.5's eight-week mean output, Chronos-2 with
covariates and last year's week-aligned profile: inference only, no training or tuning, one scoring per set. This is a supplementary analysis of windows already scored, not a first reading.

**Marks of Tables H3–H5.** Cells: % lower error [95% interval from six single-seed pairs, Welch, reading A]. \* or (−): resolved under both seed-layer readings (A and B, two groups of three seeds), the pre-specified rule. ‡: resolved under the seed layer, but the interval resampling units and target months includes zero. §: not resolved under the seed layer, but the units × months interval excludes zero. °: reading A excludes zero but reading B does not, so the contrast is not resolved. HEAPO has only four target months, so its units × months intervals are rough.

**Table H3. HEAPO (4,326 windows, 1,301 units, 4 months): ankyra-lite as delivered against each comparator.**

| Comparator | Hourly | Observed-hours level proxy | Day 1 | Days 2–7 | Days 8–31 | Observed peak (descriptive) |
|---|---|---|---|---|---|---|
| TimesFM 2.5, eight weeks (median) | +7.4\* [+2.3, +12.1] | +43.3\* [+7.6, +65.2] | −4.0§ [−11.0, +2.6] | +2.9 [−3.3, +8.7] | +8.1\* [+2.7, +13.3] | +70.4\* [+65.5, +74.6] |
| TimesFM 2.5, eight weeks (mean output) | +6.5\*‡ [+1.6, +11.2] | +37.2§ [−3.9, +62.0] | −4.9§ [−11.7, +1.5] | +1.6 [−4.7, +7.5] | +7.4\* [+2.0, +12.6] | +69.3\* [+64.2, +73.7] |
| TimesFM 2.5, one year (median) | +6.4\*‡ [+1.9, +10.8] | +40.7\* [+5.8, +62.7] | −4.5§ [−11.4, +2.0] | +2.1 [−4.0, +7.9] | +7.2\*‡ [+2.3, +11.8] | +70.0\* [+64.9, +74.4] |
| TimesFM 2.5, one year (mean output) | +5.5\*‡ [+1.5, +9.4] | +32.7 [−5.0, +56.8] | −4.9 (−) [−9.6, −0.5] | +1.2 [−3.5, +5.7] | +6.4\*‡ [+1.9, +10.7] | +68.8\* [+63.4, +73.4] |
| Chronos-2 with covariates | +4.5 [−2.2, +10.7] | +33.2 [−35.6, +67.1] | −4.7§ [−12.3, +2.4] | +1.6 [−4.0, +6.9] | +5.0§ [−2.4, +11.7] | +65.4\* [+61.0, +69.3] |
| Last year's profile | +25.5\* [+20.8, +30.0] | +13.2 [−35.0, +44.1] | +31.1\* [+27.8, +34.2] | +28.1\* [+24.2, +31.7] | +25.1\* [+20.0, +29.8] | +29.2\* [+21.5, +36.1] |
| Network alone | −0.0 [−0.5, +0.4] | −6.6 [−20.4, +5.5] | +0.7 [−2.1, +3.4] | −0.2 [−1.4, +0.9] | −0.0 [−0.4, +0.4] | +67.1\* [+61.8, +71.7] |

**Table H4. Cambridge 2023 (753 windows, 78 units, 12 months): ankyra-lite as delivered against each comparator.**

| Comparator | Hourly | Energy | Day 1 | Days 2–7 | Days 8–31 | Peak (descriptive) |
|---|---|---|---|---|---|---|
| TimesFM 2.5, eight weeks (median) | +13.9\* [+5.7, +21.3] | +25.8\* [+5.9, +41.5] | +1.1 [−13.5, +13.8] | +5.0 [−1.0, +10.7] | +15.1\* [+6.2, +23.2] | +46.9\* [+34.9, +56.6] |
| TimesFM 2.5, eight weeks (mean output) | +13.8\* [+5.6, +21.2] | +25.2\* [+5.6, +40.6] | +1.2 [−13.5, +14.0] | +4.7 [−1.3, +10.3] | +15.0\* [+6.1, +23.1] | +46.4\* [+34.4, +56.2] |
| TimesFM 2.5, one year (median) | +10.5\* [+1.9, +18.3] | +21.3\*‡ [+0.6, +37.7] | −1.4 [−16.9, +12.0] | +3.5 [−3.5, +10.0] | +11.5\* [+2.0, +20.0] | +45.8\* [+33.8, +55.6] |
| TimesFM 2.5, one year (mean output) | +10.4\* [+1.9, +18.2] | +20.5\*‡ [+0.9, +36.2] | −1.1 [−16.6, +12.4] | +3.1 [−3.8, +9.5] | +11.4\* [+2.0, +19.9] | +45.0\* [+32.8, +54.9] |
| Chronos-2 with covariates | +7.9\* [+0.7, +14.7] | +16.7 [−2.1, +32.0] | −3.8 [−12.2, +3.9] | +6.5 [−1.0, +13.4] | +8.3\* [+0.6, +15.3] | +42.8\* [+32.5, +51.5] |
| Last year's profile | +33.5\* [+26.2, +40.1] | +36.5\* [+22.5, +47.9] | +51.4\* [+42.3, +59.1] | +42.8\* [+36.5, +48.5] | +31.1\* [+23.2, +38.2] | +30.9\* [+6.5, +49.0] |
| Network alone | +0.3 [−1.5, +2.0] | −2.5 [−10.3, +4.8] | +8.0\* [+3.0, +12.8] | +2.2 [−0.5, +4.9] | −0.1 [−2.3, +2.0] | +32.4\* [+21.3, +41.9] |

**Table H5. HEAPO and Cambridge 2023 (descriptive; not pooled; not part of the pre-specified reading).** HEAPO uses the observed-hours level proxy of S11.1; these are not complete-window energy comparisons. L = ankyra-lite as delivered (hourly after the continuation layer, energy from the energy readout). N and L rows: seed layer reading A (six single-seed pairs, Welch); \* or (−) only when reading B (two groups of three seeds) agrees. G rows: units × target months.

| Contrast | HEAPO hourly | HEAPO level proxy | Cambridge 2023 hourly | Cambridge 2023 energy |
|---|---|---|---|---|
| N vs G | +4.0 [−1.8, +9.4] | +10.9 [−36.6, +41.9] | +6.4 [−0.0, +12.5] | +4.0 [−13.9, +19.1] |
| L vs G | +3.9 [−1.8, +9.3] | +5.0 [−48.2, +39.2] | +6.7\* [+1.1, +11.9] | +1.6 [−12.6, +14.1] |
| G vs TimesFM 2.5 eight weeks | +3.6 [−13.1, +6.5] | +40.3 [−26.8, +64.8] | +7.7 [−4.1, +15.0] | +24.6\* [+0.4, +36.6] |
| G vs one-year TimesFM 2.5 | +2.6 [−14.5, +5.2] | +37.6 [−46.2, +61.3] | +4.1 [−8.5, +11.1] | +20.0 [−7.4, +32.1] |


## Nine populations: a fixed trained learner and the network trained to the test start

Both analyses re-use the nine populations' test windows, read during development; they are descriptive.

### A fixed gradient-boosted learner (G)

**What was found.** A gradient-boosted learner, fixed before reading and not tuned, was fitted per population on the network's candidate training pool up to $C'$: one model each for the level, the day shares and the hour shares, composed so that its curve averages exactly to its level. It read summary features over the same history range, with temperature and calendar information (Supplement S18.6). The pool and range were shared, but feature representations, optimisation and model selection were not matched; this is a fixed-learner comparison, not an architectural ablation. The network alone had 8.9% lower energy error (+2.8% to +14.5%) and 7.8% lower hourly error (+6.2% to +9.3%), resolved in all four pools; on single populations the energy lead ranged from 4.7% to 13.6%, none resolved. The learner was 12.6% better than eight-week TimesFM 2.5 in energy (+5.8% to +18.9%), not resolvably better than the two stronger zero-shot settings, and reached 0.59 (0.32 to 0.82) of the network's log energy advantage over eight-week TimesFM 2.5 (a share of the log advantage, not of a mechanism, conditional on the fitted models). The pre-specified 5% margin lies inside the network's energy interval, so neither the learner's non-inferiority nor a lead of at least 5% was established. A fixed, untuned learner thus did not remove the network's lead; tuned or sequence-based alternatives were not run, and on HEAPO and Cambridge 2023 most differences were not resolved (Supplement S18.6).

**The learner.** Fixed before reading and not tuned. Per population, three `HistGradientBoostingRegressor` models (squared loss, learning rate 0.05, at most 15 leaves, at least 50 samples per leaf, L2 1.0, at most 2,000 rounds, early stopping after 50 rounds on the population's validation windows) predict the level (window mean ÷ the network's causal scale $s$), the day shares and the within-day hour shares by day class. They are fitted on the network's candidate stage-1 and stage-2 training pools up to $C'$ (level: all windows; day and hour models: at most 25,000 windows per population and stage, drawn uniformly with a fixed seed), with the network's sampling probabilities (stage 2: times the inverse-noise weights) as relative weights. Features use only pre-origin data and the known forecast calendar: mean load over the last 168, 744 and 8,760 h, the same window one and two years earlier, temperature climatology minus recent temperature, working-day shares, month phase and category; day and hour models add day type, position, annual phase and last year's matching day or hour ratios. The forecast is $\hat y_{d,h}=s\,r^+p_d\,q_{c(d),h}$ with nonnegative, renormalised shares, so its level is exactly $s\,r^+$ and its energy depends only on the level model. Its history range does not exceed the network's (366 days of hourly data and 24 744-h blocks), but it is not a strict subset of the network's inputs. Sharing the candidate pool and history range does not match representations, optimisation or model selection; this is not an architectural ablation. Fits are deterministic; there is no seed layer on the learner's side.

**Table N1. Network alone (N) against the learner (G), and G against zero-shot forecasters, nine populations, % [95% interval].** N rows: seed layer; G rows: units × target months only. \* resolved in favour of the first-named, (−) against. The row "N vs TimesFM 2.5 eight weeks" is this learner's scorer's own reading; its intervals for the three smaller pools differ from main-text Table 3 (built differently) by up to 1.1 points, with the same estimates and verdicts.

| Contrast, block | Nine | Benchmark subset | Three non-development | Six development |
|---|---|---|---|---|
| N vs G, energy (level) | +8.9\* [+2.8, +14.5] | +7.8\* [+0.8, +14.2] | +10.3\* [+0.6, +19.1] | +8.1\* [+0.6, +15.0] |
| N vs G, hourly | +7.8\* [+6.2, +9.3] | +7.2\* [+5.4, +9.0] | +8.3\* [+5.7, +10.8] | +7.5\* [+5.5, +9.4] |
| N vs G, daily path | +12.4\* [+10.3, +14.5] | +12.7\* [+10.4, +14.9] | +13.7\* [+10.2, +17.0] | +11.7\* [+9.1, +14.3] |
| N vs G, within-day | +4.5\* [+3.4, +5.6] | +4.3\* [+2.7, +5.8] | +5.5\* [+3.2, +7.8] | +4.0\* [+2.7, +5.2] |
| G vs TimesFM 2.5 eight weeks, energy | +12.6\* [+5.8, +18.9] | +8.8 [−0.4, +16.5] | +11.6 [−4.6, +19.4] | +13.1\* [+6.1, +21.1] |
| G vs one-year TimesFM 2.5, energy | +5.9 [−1.0, +11.6] | +1.2 [−8.5, +9.1] | +1.2 [−13.4, +11.3] | +8.2\* [+1.2, +15.8] |
| G vs Chronos-2 with covariates, energy | +5.6 [−3.1, +11.7] | +6.9 [−4.8, +14.1] | +4.6 [−12.0, +13.7] | +6.1 [−4.0, +14.4] |
| N vs TimesFM 2.5 eight weeks, energy | +20.4\* [+13.4, +26.8] | +15.9\* [+7.5, +23.5] | +20.7\* [+10.2, +30.1] | +20.2\* [+11.1, +28.3] |
| Share: r(G vs eight-week TimesFM 2.5) ÷ r(N vs eight-week TimesFM 2.5), energy | 0.59 [0.32, 0.82] | 0.53 [−0.06, 0.98] | 0.53 [−0.42, 0.90] | 0.62 [0.34, 0.93] |


The share is a ratio of mean log RMSE ratios (Section 3.4), computed within each paired resample of units and target months; it is a share of the log energy-error advantage, not of the error and not of a mechanism. Its interval has no seed layer, so it is conditional on the fitted models. It is read only where N vs eight-week TimesFM 2.5 is resolved, which holds in all four pools.

**Pre-specified reading (nine populations, N vs G energy).** The rule fixed before reading compared the interval with 0 and with a 5% margin (error ratio 1/0.95). The interval [+2.8, +14.5] excludes 0 and contains +5%, so the pre-specified sentence is: "the network is clearly better; this pre-fixed simple learner reached about 0.59 [0.32, 0.82] of its advantage". Neither the learner's non-inferiority within 5% nor a network lead of at least 5% was established. The rule excludes reading the result as equivalence, as an attribution to information or per-population training, or as an attribution of the remaining difference to the network's architecture.

### The network trained to the test start

**Question.** Under a pre-specified update policy, how do two deployment options compare on the same windows: the network trained once to $C'$ and anchored at every origin (ankyra-lite; L before the continuation layer, Lc after it), or the same recipe trained once to $C$ with weights then fixed and used alone ($N_C$)? The options differ in training cut, selection period, training pool and weights at once, so the contrasts do not isolate the layer.

**Table N2. Nine populations, % [95% interval, seed layer on both sides].** Positive: the first-named has lower error. \* resolved in favour of the first-named, (−) against.

| Readout | L vs $N_C$ (primary) | Lc vs $N_C$ (configuration as delivered) | $N_C$ vs N (186 more days of labels, new selection period, retraining noise) |
|---|---|---|---|
| Hourly | +0.2 [−1.4, +1.8] | +0.6 [−0.9, +2.2] | +0.3 [−1.1, +1.8] |
| Energy | −0.2 [−4.9, +4.3] | −0.2 [−4.9, +4.3] (energy is read before the continuation layer) | +0.8 [−3.6, +4.9] |
| Day 1 | −5.1 (−) [−8.5, −1.9] | +8.2\* [+5.1, +11.3] | +2.0 [−1.2, +5.1] |
| Days 2–7 | −0.6 [−2.4, +1.2] | +1.0 [−0.9, +2.8] | +0.8 [−0.9, +2.4] |
| Days 8–31 | +0.1 [−1.6, +1.7] | +0.1 [−1.6, +1.7] | +0.4 [−1.1, +1.9] |

$N_C$ alone, energy: vs eight-week TimesFM 2.5 +21.0\* [+13.8, +27.5]; vs one-year TimesFM 2.5 +14.9\* [+7.6, +21.6].

## Nine populations: Chronos-2 fine-tuned on the same training pool

**Identity.** A comparison fixed in writing before it was read (comparators, recipes, estimators, intervals, random
streams and the reading rule), read once on 6 November 2026 on the test windows of the nine populations (5,716
windows). These windows were read during development, so this is a pre-specified comparison on re-used windows, not a
confirmation. The tables are formatted from the read's result file, not copied from the paper. Positive percentages: the
first-named forecaster has the lower error.

**Forecasters.** N: the network alone (the six seeds of the 0.2.0 carrier, without the anchoring layer; energy is read
before the continuation layer). B8: the same network trained on eight weeks of history. FTC: Chronos-2 fine-tuned per
population on the network's training pool, three seeds, learning rate 1e-5, with a context of 1,344 hours plus the same
period one year earlier and no category inputs; its forecast is the probability-weighted mean of its quantiles. C2T:
zero-shot Chronos-2 given FTC's inputs. C2Y: zero-shot Chronos-2 given one year of context. The comparison was run on the
network alone, not on ankyra-lite as delivered; on these populations the anchoring layer did not resolvably change the
carrier's monthly energy on any population (contrast (i), 0 / 9 / 0, [README](../README.md#at-a-glance)). The anchoring
layer applied to FTC is a foundation-model configuration of the layer and, like the TimesFM one, is reported in the
paper, not here.

**Estimator.** As in the paper (Section 3.4): unit-equal log RMSE ratios, pooled with equal population weights; 95%
intervals from 2,000 paired resamples of units and target months. For the network against FTC the point is the mean of
the two three-seed network groups' ratios, and the interval adds both sides' seed variance (Welch: SE² = SE² of units and
months + s² of the network's six seeds / 6 + s² of FTC's three seeds / 3); FTC against C2T adds FTC's seed variance, and
the network against C2Y the network's. Units whose loss is zero in every vector of a row (3 BDG2 units whose
windows are all state windows) are left out of that row's seed layer, as the ratio already leaves them out.

**Table F1. Nine populations, % [95% interval].** \* resolved in favour of the first-named, (−) against.

| Contrast | Energy | Hourly |
|---|---|---|
| N vs FTC | −12.2 (−) [−21.3, −3.8] | −4.3 [−9.4, +0.4] |
| B8 vs FTC | −13.7 (−) [−22.4, −5.7] | −4.5 [−9.3, +0.2] |
| FTC vs C2T | +24.4\* [+18.2, +30.1] | +11.4\* [+7.1, +15.4] |
| N vs C2Y | +5.4 [−2.4, +12.6] | +1.8 [−0.8, +4.3] |
| B8 vs N | −2.1 [−8.7, +4.2] | −0.3 [−2.1, +1.5] |
| N vs TimesFM 2.5, eight weeks | +20.4\* [+13.4, +26.8] | +8.1\* [+4.9, +11.1] |
| N vs one-year TimesFM 2.5 | +14.2\* [+7.3, +20.6] | +4.7\* [+2.3, +7.1] |
| N vs Chronos-2-X | +14.0\* [+6.2, +21.1] | +3.3\* [+0.2, +6.3] |
| N vs G | +8.9\* [+2.8, +14.5] | +7.8\* [+6.2, +9.3] |

**Table F2. N against FTC by pool and population, % [95% interval].**

| Pool or population | Energy | Hourly |
|---|---|---|
| Nine populations | −12.2 (−) [−21.3, −3.8] | −4.3 [−9.4, +0.4] |
| Benchmark subset (first six) | −11.6 (−) [−23.3, −1.1] | −3.1 [−10.4, +3.8] |
| Three non-development | −20.8 (−) [−37.3, −6.3] | −11.4 (−) [−18.1, −5.1] |
| Six development | −8.2 [−19.6, +2.2] | −1.0 [−7.9, +5.5] |
| BDG2 | −16.7 [−45.9, +6.6] | −16.0 [−36.9, +1.7] |
| Cambridge | −15.4 [−36.1, +2.1] | −8.3 (−) [−13.5, −3.4] |
| HEEW | +9.3 [−29.2, +36.4] | +19.1 [−12.5, +41.8] |
| EWELD | −4.9 [−34.8, +18.3] | +1.2 [−13.3, +13.8] |
| GoiEner non-household | −44.0 (−) [−90.6, −8.8] | −17.3 (−) [−25.3, −9.8] |
| GoiEner households | −4.9 [−23.8, +11.1] | −1.7 [−5.8, +2.2] |
| Oslo | −5.7 [−25.7, +11.2] | −7.8 (−) [−14.4, −1.5] |
| Drammen | −6.4 [−36.9, +17.4] | −3.0 [−12.3, +5.5] |
| CINELDI | −29.8 (−) [−55.1, −8.5] | −10.2 (−) [−15.9, −4.8] |

**Reading.** By the decision rule fixed before the read, a trained comparator resolved better in energy gives the branch
"a comparator is resolved better": Chronos-2 fine-tuned on the same training pool had 10.9% lower monthly energy error
than the network (3.7% to 17.6%; the network's row is −12.2 (−) [−21.3, −3.8]). The hourly difference was not resolved
(4.2% lower for FTC). The network's energy error stayed below that of eight-week and one-year TimesFM 2.5,
Chronos-2-X and G (8.9% to 20.4%, resolved) and below one-year Chronos-2's by 5.4% (not
resolved). Fine-tuning lowered Chronos-2's energy error by 24.4% against its zero-shot twin. FTC's lead was largest
on the three populations without development windows (17.2%, 6.0% to 27.1%) and not resolved on the six
development populations (7.5%); of the single populations it was resolved on GoiEner non-household and CINELDI only. In
hourly error FTC was resolved better on the three non-development populations pooled (10.3% lower) and on
Cambridge, GoiEner non-household, Oslo and CINELDI; these pool and single-population readings are secondary and not
adjusted for multiplicity.

**How it was read.** The forecasts and comparator runs were accepted first (training records, forecasts, a CPU
regeneration of the network's members and the anchoring products), and the comparison was then read once through a
guarded process bound to the author's approval of the exact input package. Before any new row, three published rows of
the paper were reproduced exactly from their stored losses, and the rebuilt network losses matched the stored ones on all
nine populations; every row was recomputed independently from the saved resamples afterwards. Two earlier attempts
stopped at pre-specified checks before any result was computed (a summation-order difference in a consistency check of
the rebuilt losses; zero-loss units in the seed layer); each was corrected with the author's approval, and their records
are kept with the study's research records.

**What this does not show.** One fine-tuned foundation model was compared, and inputs and training were not matched
(FTC's 1,344-hour context plus the previous year and no category inputs; six network seeds against three FTC seeds).
N-HiTS was pre-specified as a second trained comparator but did not pass its sanity check under the shared recipe (on
HEEW validation windows it was worse than a weekly seasonal-naive forecast and a per-unit ridge regression) and was not
run on the nine populations; TFT and DeepAR were not run. Compute was not compared on matched terms: FTC was fine-tuned
and run on a GPU, ankyra-lite runs on one CPU thread (0.66 s per forecast on six Oslo windows), and no matched
end-to-end timing was made. The differences are not attributed to architecture, pretraining, adaptation or inputs.

## What is known about the source of the energy lead

Matched eight-week training left the long record's gain unresolved (hourly 0.3%, −1.6% to +2.1%; energy 2.0%, −4.4% to +8.0%), so the frozen network's loss when truncated at inference shows sensitivity to truncation, not the value of the long record. No energy benefit of the decoder's explicit monthly normalisation was established (+0.4%, −2.4% to +3.1%, against a geometric daily-level parameterisation that keeps the level–shape decomposition), which does not test the decoder as a whole. An earlier, level-only version of the fixed learner, fitted on the development populations, also stayed behind the network (by 11.8%, +4.3% to +18.8%; Supplement S15). The energy lead is stable, but its source is not located. The common-bias and seasonal analyses above are consistent with a contribution of seasonal information and, on households, of the median point forecast, but they do not isolate either.

## The calendar known at the origin

The day types of the forecast month are treated as known at the origin
([Limitations](../README.md#limitations)). On 36 test windows of the comparison (26 EWELD windows issued on
1 January 2020, whose calendar contains the extension of the 2020 Spring Festival holiday announced on 27 January 2020,
and 10 Cambridge windows issued on 1 September 2022, whose calendar contains the bank holiday of 19 September 2022
announced on 10 September 2022) the forecasts used a holiday announced after the origin. For the paper the affected
forecasts were recomputed with the calendar known at the origin, without retraining, and re-scored (the linear-context
model, whose fitted model had not been stored, kept its forecasts). No number of the paper's title, highlights,
abstract or conclusion changed; ten numbers of its main text moved by 0.1 percentage points at printed precision, none
changing resolution, and none of its eleven main comparisons changed sign, resolution or rank; in its supplement about
460 printed numbers moved, and seven single-population entries or counts changed resolution. Leaving the 36 windows out
moved point estimates by at most 0.2 points, with no change of interval status. The result files of this repository
keep the earlier computation.

## What these results do not show

- **One confirmed reading.** The openMeter verdict covers monthly energy against one-year TimesFM 2.5 on this source,
  its earlier period, the frozen configuration and the registered metric. It does not extend to the hourly curve, the
  first days, the peak, other comparators or other sources, and it is not a zero-shot result.
- **Pretraining exposure.** openMeter, HEAPO and Cambridge 2023 were not found by name in the published pretraining
  lists of TimesFM 2.5, Chronos and Chronos-2; this does not exclude exposure.
- **Few origins.** HEAPO has 4 origin months and Cambridge 2023 has 12; openMeter's categories cover different months
  (households and businesses 2024-02 to 2024-08, public buildings 2024-02 to 2024-05 and 2025-01 to 2025-07).
- **Short-term accuracy.** On none of the three sets is a lead over TimesFM 2.5 in the first hours or on day 1 claimed.
- **The source of the energy lead** is not located ([above](#what-is-known-about-the-source-of-the-energy-lead)).
- **Trained comparators.** Chronos-2 fine-tuned on the same training pool is the only fine-tuned foundation model
  compared, on the nine populations only ([above](#nine-populations-chronos-2-fine-tuned-on-the-same-training-pool)); N-HiTS did not pass its pre-specified sanity check,
  and TFT and DeepAR were not run.

## Data and licences

- **openMeter** (openmeter.de; series retrieved through the platform's API): contains information from openMeter, which
  is made available here under the Open Database License (ODbL 1.0). The meter and weather series, the post codes used
  to assign temperature and the prepared store are not redistributed.
- **HEAPO** (Brudermueller et al., 2025; Zenodo record 15056919): CC BY 4.0; not redistributed.
- **Cambridge 2023**: the University of Cambridge estate archive (Langtry and Choudhary, 2024), CC BY 4.0; not
  redistributed.

## Files

- [`results/openmeter_test.csv`](../results/openmeter_test.csv): the openMeter readings of Table O2, at full precision
  (columns in [results/README.md](../results/README.md)).
- [`results/fairness_test.csv`](../results/fairness_test.csv): every scored row of the fine-tuned comparison above, at
  full precision, except the anchoring layer applied to FTC (columns in [results/README.md](../results/README.md)).
- The protocol, its two amendments, the freeze record and the score record of the openMeter test are kept with the
  study's research records; the code that built its store, ran its comparators and scored it is not part of this
  repository.
