# Data: sources, preparation and splits

This page says where the nine populations of the ankyra-lite evaluation come from, how the study prepared them, and
how each population's record is divided into a training set, a validation set and a test set. It is written for a
reader who wants to rebuild the inputs from the public sources or to check that no test target reached the training.

The nine populations are those of the first ANKYRA release (the ANKYRA 2.2.1 repository): its six test populations and
three of its four preview populations. The data stores are the same stores, with one exception: CINELDI is read from
the corrected store described [below](#9-cineldi-industrial-customers-norway). The first release's
`docs/DATA.md` (repository [XinanQ/Building_load_ANKYRA](https://github.com/XinanQ/Building_load_ANKYRA)) is the full
record of the sources; this page repeats what ankyra-lite needs and adds what is specific to it: the training windows of the carrier
network, the split, the windows borrowed from other populations, and the corrected CINELDI store.

Contents: [What is and is not in this repository](#what-is-and-is-not-in-this-repository) ·
[Input format](#the-input-format) · [How much history](#how-much-history) · [Two kinds of windows](#two-kinds-of-windows) ·
[Training, validation and test sets](#training-validation-and-test-sets) ·
[Borrowed windows](#windows-borrowed-from-other-populations) ·
[Calendar caveat](#calendar-caveat-holidays-announced-after-the-origin) ·
[Pretraining overlap](#bdg2-and-the-pretraining-collection-of-timesfm-25) · [The populations](#the-populations) ·
[What cannot be rebuilt](#what-cannot-be-rebuilt-from-this-page)

## What is and is not in this repository

- **In the repository:** the forecaster ([`ankyra_lite/`](../ankyra_lite): the carrier network, its training recipe and
  the copy of the anchoring layer), the 27 released carrier checkpoints (9 populations × 3 seeds, with
  `checkpoints/MANIFEST.json`; [METHOD.md](METHOD.md#released-weights)), the scoring code
  ([`evaluation/`](../evaluation/README.md)), the scored results ([`results/`](../results/README.md)), the figures, and
  one Cambridge example window (University of Cambridge estate archive, CC BY 4.0, attributed in
  `results/example_window_cambridge.json`).
- **Not in the repository:** raw data, the prepared data stores and the saved forecasts of the baselines. None of them
  is redistributed. Follow each provider's access route and terms. The checkpoints contain weights fitted on these data,
  not the data.
- **Foundation models.** ankyra-lite uses none. The zero-shot foundation models appear only as baselines; the study ran
  local snapshots with verified hashes: TimesFM 2.5 200M (Apache-2.0), revision
  `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`, and Chronos-2 (`amazon/chronos-2`, Apache-2.0), revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c`.
- **Archive identifiers, sizes and hashes** below are those the study recorded when it obtained the files. Where it
  recorded none, the entry says "not recorded here; see the provider". Nothing on this page is a guess.

## The input format

Every population was brought to one hourly format. Two parts of ankyra-lite read it: the carrier network
(`ankyra_lite.carrier.features.features`) and the anchoring layer (`ankyra_lite.anchor.History` plus three arguments of
`forecast`).

| Input | Content | Carrier network | Anchoring layer |
|---|---|---|---|
| `load_kw` | hourly mean load in kW from the record anchor; NaN where missing | reads only hours before each issue time; a negative value counts as missing; a day with no observed hour is masked out | the last 1,344 hours before the origin must be completely observed (the package never fills them); earlier gaps are NaN, and gaps of at most 6 hours inside one 744-hour block are interpolated in the copy its estimator reads |
| `observed` | optional Boolean mask of the load | — (finite and nonnegative = observed) | `False` hours are ignored; the study passed `observed = finite load` |
| `temperature_c` | hourly outdoor temperature in °C | reads only hours before each issue time, through daily means and their coverage; missing hours are allowed | finite everywhere before the origin; no future temperature is needed |
| `day_types` | one integer per hour for the history and the 744 forecast hours: Monday = 0 … Sunday = 6, public holiday = 7 | read through the forecast month of each issue time (the calendar is treated as known at the origin); outside the array the calendar weekday is used | the same |
| `start_timestamp` | 1 January 00:00 of the first year, with a `+00:00` offset; index 0 of every array | sets the annual phase of every token and the record's absolute time (used to pool populations) | fixes the phase of the annual temperature harmonic and the daylight-saving state used to match analog days |
| `group` | Industrial, Office, Public, Residential or Commercial | one-hot category input | selects the fixed temperature-signature prior; Commercial has none and gets a zero temperature signature |
| `temp_sigma_std` | one positive scalar per population (below) | — | the temperature anomaly over which the weather-adjusted candidates take their expectation |
| `dst_region` | "EU", "US" or "none" | — | used only to match analog days |

**How the study computed `temp_sigma_std` for ankyra-lite.** The arithmetic is that of the first release: standardise
the temperature as (T − 15 °C) / 10 °C, take daily means, subtract a centred 31-day moving mean, and take the standard
deviation of what remains, pooled over the units, with 15 days dropped at each end where the moving mean is
incomplete. The fit period is different: ankyra-lite fits it on **the complete days before the population's training
cut** (all units whose every day is defined; if there is none, the units with at least 95% of their days defined, with
the missing days interpolated). The training cut lies 186 days before the first test origin and at or before every
pseudo-origin of every test window, so the fit period overlaps no test target and no pseudo-target. One value per
population is used for every window and every recipe. The values are recorded in the study's split records and are
not listed here.

Points that matter when rebuilding:

- **Loads in kW.** The anchoring layer's off-state (10⁻⁶ kW) and micro-load (10⁻³ kW) thresholds are absolute. kWh per
  hour equals the mean kW of that hour; Wh per hour is divided by 1,000; four 15-minute kWh values are summed. The
  carrier itself is scale-equivariant (it divides every load by a causal scale; property C7 in
  [theory](../theory/README.md)).
- **No hidden time handling.** Neither part resamples, interpolates the context at the origin or converts daylight
  saving time. The grid must be a regular hourly grid. Several populations are placed on the provider's fixed-offset
  local grid and passed with a `+00:00` anchor; day types then follow the local calendar.
- **The 1 January anchor.** A record that starts later in the year is padded back to 1 January with unobserved load
  (NaN), not shifted. The original CINELDI store missed this padding ([below](#9-cineldi-industrial-customers-norway)).
- **Day types** are the weekday of the grid's calendar date, replaced by 7 on a public holiday of the population's
  country.

## How much history

The two parts need different amounts of history. The carrier needs no complete hours: it masks unobserved hours and
days (C6) and reads at most the last 366 days for its daily tokens, the 24 blocks of 744 hours before them for its
monthly tokens (26,640 hours, about three years, in all) and the last 8,760 hours for its scale; older hours are not
read. The anchoring layer reads the whole pre-origin record and needs **complete hours**: a context and a
pseudo-target count only when completely observed (90% observed for the seven-candidate level and the handover
pairs), and every missing hour can only lower the counts below (P7). With a complete record of m hours before the
origin:

| Complete hours before the origin | What the code can then do | Code |
|---:|---|---|
| 1,344 (56 days) | the forecast runs: the 1,344-hour context at the origin must be completely observed, or `anchor.InputError` is raised (`ankyra_lite.forecast` raises `ValueError` earlier, when it computes the carrier forecasts). With no completed pseudo-origin the level weights are equal, the handover weights are ½ and the within-day trust is 0 | `anchor.forecast`, `history.api._validate_history` |
| 2,088 (1,344 + 744) | the first pseudo-origin completes: the handover weights leave ½ (α_w = (λ̂_w + 1)/3 with one pair) | `anchor.lead_week_weights` |
| 2,832 (1,344 + 2 × 744; 118 days) | two pseudo-origin errors per candidate: the level weights leave equal weights (data share K_eff/(K_eff + 8) = 0.2), and so do the daily-path weights | P7; `history/_day.py` |
| 5,808 (1,344 + 6 × 744) | all six carrier pseudo-origins: the carrier candidate and the handover have their full evidence | `core.K_FM` |
| about 13 months | analog days at the same dates one year earlier: the within-day anchoring can act (its trust needs analog days at a completed pseudo-origin as well) | `anchor.analog.AnalogShapes` |
| 10,248 (8,760 + 2 × 744; about 14 months) | the annual level candidates get two pseudo-origin errors and can be weighted | P7 |
| 10,272 (1,344 + 12 × 744) | all twelve pseudo-origins of the historical candidates: K_eff = 12, data share 0.6 | P7 |

These are necessary support lengths, not accuracy guarantees. Every test window of the nine populations had at least
9 complete pseudo-origin windows ([EVALUATION.md](EVALUATION.md#history-length-and-the-anchoring-record)).

**The released checkpoints and the 186-day rule.** A checkpoint belongs to one population and its training cut. Pass
that cut as `cut=` to `ankyra_lite.forecast` (or to `carrier_forecasts`): a pseudo-origin before the cut is then not
used, and an origin before the cut raises `ValueError`. Only an origin at least 186 days (6 × 744 hours) after the cut
keeps all six carrier pseudo-origins; nearer the cut the carrier candidate and the handover rest on fewer pairs. For a
unit whose record is not in the carrier's training data, `cut=None` drops nothing
([METHOD.md](METHOD.md#why-the-test-set-starts-186-days-after-the-cut)).

**A record that starts in the middle of a year.** Pad the load back to 1 January 00:00 of the first year with NaN (or
`observed = False`), give a finite temperature for the padded hours too (archive temperature where it exists: the
layer's climatology is fitted on every day before the origin, the padded days included, and a temperature that does
not vary over the year before the origin raises `SignatureDegeneracy`), write the day types of the padded days, and
pass `start_timestamp` as that 1 January 00:00 with a `+00:00` offset. No date is converted: keep the record on one
fixed-offset grid. [`examples/own_unit.py`](../examples/own_unit.py) does this for an artificial meter that starts on
18 March 2019, with holidays and gaps, and shows the forecast with the shipped checkpoints and `cut=None`, the
history-only configuration, the interval readout and the anchoring record.

**Gaps in a user's own record.** `ankyra_lite.fill_short_gaps` (= `ankyra_lite.anchor.fill_short_gaps`) returns a
copy of the pre-origin load in which every gap of at most 6 hours is linearly interpolated between its two observed
neighbours, provided the gap and both neighbours lie inside one 744-hour block counted back from the origin (the end of
the array). It leaves gaps that touch the origin, the start of the record or a block boundary, and gaps longer than
6 hours, as NaN. The anchoring layer applies the same rule to the copy its own estimator reads, but it requires the
context at the origin to be complete first and never fills it, and the carrier reads the record as supplied (an
interpolated hour is a missing hour for it unless the caller filled it). Filling the record before the call therefore
also hands the interpolated hours to the carrier ([METHOD.md](METHOD.md#short-gap-rule)). The study's test windows
have complete contexts and targets and did not need it.

## Two kinds of windows

A window is a pair (unit, origin o): the 1,344 hours before o are its context and the 744 hours from o are its target
(its labels, in training).

**Test windows** are the late windows of the first release's evaluation panel, unchanged, so that every number here is
on the same windows as the first release's tables:

- origins at the first hour of a calendar month; a completely observed 1,344-hour context and 744-hour target;
- the load complete from six monthly steps before the context up to the origin (at least 5,808 hours), so that the
  anchoring layer can use six pseudo-origins;
- at most 1,500 windows per population by the first release's fixed thinning rule (month-start origins only, then every
  k-th window per unit);
- **late**: the origin lies at or after the training cutoff of the first release's trained baselines (the first month
  start at or after the population's median evaluation origin). Only there does every baseline have a
  forecast. These cutoffs are the test starts of the table [below](#training-validation-and-test-sets).

Hours that a provider imputed, replaced or zero-filled are missing where the provider documents it or where the
population's stated rule detects it; a masked hour makes a window ineligible, and the preparation never fills it.

**Training windows** of the carrier are built from each population's own store, with load and temperature at and after
the population's training cut set to missing before any feature is computed:

- origins on a daily grid (every 24 hours from hour 1,344), with o + 744 ≤ the training cut;
- a window is eligible when at least 75% of its context hours and at least 75% of its target hours are observed
  (finite and nonnegative); the carrier's features mask the rest;
- all units of the population's prepared store are used, and a store can hold more units than the test panel: BDG2
  1,244 meters (178 in the panel), EWELD the 386 meters of all three cities (274 in the panel), GoiEner non-household
  873 supply points (486), GoiEner households 6,226 (1,109), Cambridge 120 (119), HEEW 147 (142), Oslo 48 (45),
  Drammen and CINELDI 45 each (45);
- the windows of the validation set (early stopping) lie on a weekly grid (every 168 hours from hour 1,344), with the
  same eligibility.

## Training, validation and test sets

The rule is the same for every population:

- **Test set:** the population's late windows (above). Its start C is the trained baselines' training cutoff.
- **Training cut** C′ = C − 4,464 hours (6 × 744 hours = 186 days). Nothing at or after C′ is used to fit the carrier's
  weights.
- **Validation span:** the last 93 days before the cut (3 × 744 hours), from v0 = max(2,088, C′ − 2,232) to C′.
- **Training set:** the population's own windows whose labels end at or before v0, plus windows borrowed from other
  populations whose labels end before the same instant in absolute time ([below](#windows-borrowed-from-other-populations)).
- **Gap:** the 186 days between C′ and C make every carrier forecast at the six pseudo-origins o − 744k (k = 1 … 6) of
  every test window an out-of-sample forecast issued at or after the cut (property C10 in
  [theory](../theory/README.md)); the reason is given in [METHOD.md](METHOD.md#why-the-test-set-starts-186-days-after-the-cut).

**How the released recipe uses the validation span.** The recipe has two stages ([METHOD.md](METHOD.md#training)).
In the first, the carrier is trained from scratch on the training set and the validation span is held out: its
windows decide when to stop. In the second, the validation span is **returned** to the population's own training
windows (labels ending within 24 hours of the cut are not returned) and training continues for a fixed 8 epochs, with
no further selection. The final weights have therefore seen labels up to 24 hours before the cut; the borrowed windows
still end before v0. In time order:

```
 first training origin ...... training set ...... | validation span (93 days) | cut | 186-day gap | test set ...
 stage 1:  trained on ............................  | early stopping only       |
 stage 2:  trained on ................................................ returned, up to 24 h before the cut
```

The newest label used to update the weights ends at least 187 days (186 days and 24 hours) before the first test
origin; in the first stage alone it was at least 279 days (93 + 186). The forecast inputs always use the load up to the
origin.

**The split of each population.** Dates are true calendar dates (CINELDI on the corrected store). "Days of training
origins before the cut" counts from the first training origin to the cut; "validation share" is the share of the
population's own eligible daily-grid windows (before the caps) that fall in the validation span; "months spanned" is
the number of calendar months from the first to the last test origin. Every value of the table is a column of
[`results/datasets.csv`](../results/README.md#populations-and-splits).

| Population | First training origin | Validation span from | Training cut | Test set (first … last origin) | Test windows / units | Days of training origins before the cut | Validation share | Months spanned | Development windows |
|---|---|---|---|---|---:|---:|---:|---:|---|
| Cambridge | 2000-02-26 | 2012-09-25 | 2012-12-27 | 2013-07-01 … 2022-12-01 | 730 / 108 | 4,688 | 1.9% | 114 | yes |
| HEEW Arizona | 2014-02-26 | 2018-02-25 | 2018-05-29 | 2018-12-01 … 2022-12-01 | 658 / 138 | 1,553 | 4.5% | 49 | yes |
| EWELD | 2016-07-15 | 2018-10-26 | 2019-01-27 | 2019-08-01 … 2022-07-01 | 523 / 197 | 926 | 11.0% | 36 | yes |
| Oslo | 2012-11-12 | 2018-12-26 | 2019-03-29 | 2019-10-01 … 2024-06-01 | 581 / 41 | 2,328 | 3.6% | 57 | yes |
| Drammen | 2018-02-26 | 2019-11-27 | 2020-02-28 | 2020-09-01 … 2022-02-01 | 689 / 43 | 732 | 9.2% | 18 | yes |
| CINELDI | 2019-04-12 | 2020-06-26 | 2020-09-27 | 2021-04-01 … 2022-02-01 | 475 / 45 | 534 | 13.7% | 11 | yes |
| BDG2 2017 | 2016-02-26 | 2016-10-26 | 2017-01-27 | 2017-08-01 … 2017-12-01 | 474 / 142 | 336 | 23.7% | 5 | no |
| GoiEner non-household | 2018-02-26 | 2018-07-26 | 2018-10-27 | 2019-05-01 … 2019-12-01 | 890 / 481 | 243 | 38.3% | 8 | no |
| GoiEner households | 2018-02-26 | 2018-09-25 | 2018-12-27 | 2019-07-01 … 2019-12-01 | 696 / 696 | 304 | 28.2% | 6 | no |

Source: [`results/datasets.csv`](../results/README.md#populations-and-splits) (columns `first_training_origin`,
`validation_start`, `training_cut`, `first_test_origin`, `last_test_origin`, `test_windows`, `test_units`,
`record_days_before_cut`, `validation_share_pct`, `test_months_spanned`, `development_windows`), which also gives the
population's own windows in its pooled training set and the number of validation windows returned in stage 2.
"Development windows": the study checked the carrier's design choices on windows between a development cut 279 days
before the training cut and the training cut; the three short-record populations had too little record for such
windows, so no design choice was checked on them before the test windows were read.

Figure [1b](../figures/fig1b_data_splits.png) draws the split of every population, in calendar time and aligned on
the training cut.

**The same rule has different consequences.** This is the point of the split to keep in mind when reading the results.

- **Short records.** BDG2 2017 and the two GoiEner populations have 243–336 days of training origins before the cut.
  The fixed 93-day validation span takes 23.7–38.3% of their own windows, against 1.9–13.7% on the other six.
- **Short test sets.** Their test sets cover 5–8 months of a single year: BDG2 August to December 2017, GoiEner
  non-household May to December 2019, the households July to December 2019. Read their results as "the second half of
  that year", not as a full annual cycle. Drammen spans 18 months, CINELDI 11, the other four 36 to 114 months (EWELD
  36, HEEW 49, Oslo 57, Cambridge 114).
- **Seasons.** On these three populations the validation span is the season that the training set otherwise lacks
  (BDG2: late October 2016 to late January 2017; GoiEner non-household: late July to late October 2018; households:
  late September to late December 2018), and part of their test months have no same-season month in the training set
  before v0. Of their 2,060 test windows, 971 (150 BDG2, 406 GoiEner non-household, 415 households) have a target month
  without same-season training labels before the span is returned and with them after it; the other 1,089 (324, 484
  and 281) do not change in this respect (the "switch" and "non-switch" layers of
  [`results/recipe_history.csv`](../results/README.md#recipe-history)). The second stage of the recipe was adopted for
  this reason ([EVALUATION.md](EVALUATION.md#why-the-recipe-returns-the-validation-span)).
- **Weighting.** Each population counts equally in pooled readings, whatever the size of its test set (474–890
  windows); every GoiEner household has exactly one test window.

## Windows borrowed from other populations

The carrier of a target population P is trained on a pooled training set:

- **Sources.** Ten study sources contribute windows: the nine populations and the Suzhou industrial park (four category
  aggregates of the first release; [below](#10-suzhou-industrial-park-china-training-data-only)), which has no
  ankyra-lite forecast and is not in any table.
- **Time limit.** With T = P's validation start v0 in absolute time (hours since 2000-01-01 00:00 UTC, from each
  record's anchor), every source Q contributes its daily-grid windows whose labels end at or before T. For P itself this
  is exactly its training set.
- **Each source's windows come from its own training period.** A source's windows are built from its own record with
  load and temperature blanked at its own training cut, so a borrowed window lies in that source's training set or
  validation span, never in its test period, and ends before P's validation start.
- **Size.** Every source's windows are first capped at most 16,000 training and 2,000 validation windows per
  population, on each origin grid. The number of windows a source contributes equals the number of its capped
  weekly-grid windows that pass the same time limit; when more of its capped daily-grid rows pass it, a unit-balanced
  subsample **of those capped daily-grid rows** is drawn (not of all its eligible daily windows;
  [METHOD.md](METHOD.md#pooled-training-set)). The daily grid adds origin-day coverage, not windows.
- **The second stage** keeps the borrowed windows and their time limit; only P's own returned span is added.

Which sources each pool contains follows from the dates:

| Target | Borrowed sources |
|---|---|
| Cambridge | none (no window of another source ends before its validation start, 2012-09-25) |
| BDG2 2017 | Cambridge, HEEW, EWELD, Oslo, Suzhou park |
| HEEW Arizona | BDG2, Cambridge, EWELD, Oslo, Suzhou park |
| EWELD | BDG2, Cambridge, HEEW, both GoiEner populations, Oslo, Drammen, Suzhou park |
| GoiEner non-household | BDG2, Cambridge, HEEW, EWELD, GoiEner households, Oslo, Drammen, Suzhou park |
| GoiEner households | BDG2, Cambridge, HEEW, EWELD, GoiEner non-household, Oslo, Drammen, Suzhou park |
| Oslo | BDG2, Cambridge, HEEW, EWELD, both GoiEner populations, Drammen, Suzhou park |
| Drammen | the other eight populations (CINELDI from the corrected store) and the Suzhou park |
| CINELDI | the other eight populations and the Suzhou park |

Source: the study's training records of the released recipe. The sampler draws half of every epoch from P and half
from the borrowed sources ([METHOD.md](METHOD.md#sampler)).

A checkpoint belongs to one population and its training cut. Applying it to another population's windows is not
covered by the evaluation, and an origin less than 186 days after the cut would let pseudo-origins fall into the
training period.

## Calendar caveat: holidays announced after the origin

The day types of the forecast month are treated as known at the origin, by the carrier (its future-day tokens) and by
the anchoring layer, as by every baseline that reads the calendar. For two populations the holiday calendar of some
test windows was announced after their origin:

- **EWELD:** 26 of the 523 test windows (origin 2020-01-01) contain the extension of the 2020 Spring Festival holiday,
  announced by the State Council on 27 January 2020 (国办发明电〔2020〕1号).
- **Cambridge:** 10 of the 730 test windows (origin 2022-09-01) contain the bank holiday of the State Funeral on
  19 September 2022, announced on 10 September 2022.

Rule of the check: a window is affected when a calendar entry of its 31 target days rests on a notice published on or
after the origin date. Calendars computed by a fixed rule were closed by the rule: HEEW (US federal holidays), Oslo,
Drammen and CINELDI (Norwegian statutory holidays), both GoiEner populations (only Spanish national holidays generated
by rule) and the BDG2 sites. The forecasts were not remade with the calendar as known at the origin; the result files
are computed on all test windows, these included. The two counts come from the study's calendar check, not from a file
of `results/` ([EVALUATION.md](EVALUATION.md#numbers-that-are-not-in-results)).

## BDG2 and the pretraining collection of TimesFM 2.5

Seven BDG2 sites (Bear, Bull, Cockatoo, Fox, Hog, Panther, Rat) are listed in the GIFT-Eval pretraining collection that
the TimesFM 2.5 model card names as training data (and in LOTSA); they are not named in the Chronos or Chronos-2
training tables (checked 10 October 2026). Unlisted does not mean unseen. This can favour TimesFM and TimesFM-X in the
BDG2 comparisons; how much has not been measured. ankyra-lite has no pretrained weights: its carrier is trained only on
the study's own populations, within the time limits above.

## The populations

Licences were checked on the providers' pages on 5 October 2026. "Tier" is the first release's tier
([EVALUATION.md](EVALUATION.md#populations-and-tiers)); "test windows" are the late windows.

### 1. BDG2 2017 (building meters, USA / Europe)

- **Source.** Miller et al., *Sci. Data* 2020, [doi:10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x);
  Zenodo 10.5281/zenodo.3887306, CC BY 4.0. Archive version, file size and hash: not recorded here; see the provider.
- **Record.** Hourly electricity meters, 2016–2017, anchor 2016-01-01. Two building pools prepared earlier in the study,
  disjoint from each other, were joined.
- **Load column and unit conversion, temperature source, time zone handling, holiday calendar, category mapping,
  daylight-saving region:** not recorded here; see the provider.
- **Tier and windows.** Test; 474 test windows of 142 units (934 panel windows of 178 units in all).
- **Caveats.** Seven sites are in TimesFM 2.5's pretraining collection ([above](#bdg2-and-the-pretraining-collection-of-timesfm-25)).
  Nine meters at one site read 0.0002–0.0005 kW for months; three of them (Lamb_education_Harold,
  Lamb_education_Hillary, Lamb_office_Jo) are near zero in every window and hold 10 of the 474 test windows
  ([EVALUATION.md](EVALUATION.md#state-windows-and-the-three-near-zero-bdg2-meters)). 13 units carry the category
  Commercial (the store's category labels), which has no temperature-signature prior in the anchoring layer. The signature prior was fitted once on
  BDG2 windows ending before July 2016.

### 2. University of Cambridge estate (UK)

- **Source.** Langtry & Choudhary 2024, [doi:10.5281/zenodo.10955332](https://doi.org/10.5281/zenodo.10955332), version
  2.0, CC BY 4.0. File size and hash: not recorded here.
- **Load.** The processed per-building electricity files, column `equipment load [kWh]`, hourly; kWh per hour used as
  kW; years up to 2022.
- **Masking.** The provider writes missing readings as zeros; runs of three or more consecutive zero hours are treated
  as missing (a heuristic). The provider clips values at ten times the mean.
- **Temperature.** The Met Office file shipped in the archive (Bedford, `air_temperature [degC]`, 2000–2022), one series
  for all buildings; gaps linearly interpolated.
- **Grid, holidays, category, daylight saving.** UTC, anchor 2000-01-01; England and Wales bank holidays 2000–2023
  (Easter rule, fixed dates, documented substitutions and one-off holidays); Public; EU.
- **Tier and windows.** Test; 730 test windows of 108 units (1,456 of 119 in all). The one example window in
  `results/` comes from this archive.
- **Caveat.** 10 test windows contain a bank holiday announced after their origin ([above](#calendar-caveat-holidays-announced-after-the-origin)).

### 3. HEEW, Arizona State University (USA)

- **Source.** Dong et al., *Sci. Data* 2025, [doi:10.1038/s41597-025-06010-8](https://doi.org/10.1038/s41597-025-06010-8);
  figshare 10.6084/m9.figshare.28425647, version 1, CC BY 4.0, file `HEEW-20250912.zip`, MD5
  `b3ad8aa2ba8c40d74f001d483963cea9`. 147 buildings, hourly, 2014–2022.
- **Load.** Column `Electricity` of the raw per-building files, kWh per hour as kW, gross of PV.
- **Masking.** An hour is missing if the raw value is empty or below −0.01, if the cleaned value is −1 or not a number,
  or if raw and cleaned differ by more than 10⁻³; then runs of three or more zero hours, as for Cambridge.
- **Temperature.** Column `Temperature` of the archive's cleaned weather file, converted from °F to °C where its median
  exceeds 45; gaps interpolated; one series.
- **Grid, holidays, category, daylight saving.** The provider's local clock at fixed UTC−7, anchor 2014-01-01 labelled
  `+00:00`; US federal holidays with observed-day rules (Juneteenth from 2021); Public; none.
- **Tier and windows.** Test; 658 test windows of 138 units (1,282 of 142 in all).

### 4. EWELD (industrial and commercial meters, China)

- **Source.** Liu et al., *Sci. Data* 2023, [doi:10.1038/s41597-023-02503-6](https://doi.org/10.1038/s41597-023-02503-6);
  figshare 10.6084/m9.figshare.21893808, version 3, CC BY 4.0; 258,090,540 bytes, MD5
  `eea1455aff6518aae1fbfc397b178111`. 386 meters in three anonymised cities; the population is the second city
  (350 meters).
- **Load.** 15-minute kWh; hourly kW is the sum of the four quarter-hour values; an hour with a missing quarter is
  missing.
- **Masking.** Runs of at least eight identical positive quarter-hour values (0.4% of values) are masked as fill
  traces. Runs of zeros (13% of values) are kept: a shutdown and a filled gap cannot be told apart.
- **Temperature.** The archive's weather file of the meter's city (15-minute, °F), converted to °C, averaged to hours,
  gaps interpolated.
- **Grid, holidays, category, daylight saving.** Hourly, assumed UTC+8 (checked against the pooled weekday profile),
  anchor 2016-01-01; Chinese statutory holidays 2016–2022 with make-up workdays, transcribed from the State Council
  notices; by ISIC section: A, C, D, E, F, H → Industrial; G, J, K, L, M, N → Office; I, O, P, Q, S → Public; none.
- **Tier and windows.** Test; 523 test windows of 197 units (1,023 of 274 in all). The store holds the meters of all
  three cities, and the carrier's training windows come from all 386.
- **Caveats.** Many contexts are entirely zero, so the anchoring layer's state rules fire on many test windows
  ([EVALUATION.md](EVALUATION.md#state-windows-and-the-three-near-zero-bdg2-meters)). 26 test windows contain a
  holiday extension announced after their origin ([above](#calendar-caveat-holidays-announced-after-the-origin)).

### 5. GoiEner non-household supply points (Spain)

- **Source.** Quesada et al., *Sci. Data* 2024, [doi:10.1038/s41597-023-02846-0](https://doi.org/10.1038/s41597-023-02846-0);
  Zenodo 10.5281/zenodo.7362094 and 10.5281/zenodo.7859413, CC BY 4.0. Archive version, size and hash: not recorded
  here; see the provider.
- **Units.** 486 category-labelled supply points (Industrial 159, Office 80, Public 247), not certified individual
  buildings, disjoint from the GoiEner supply points on which ANKYRA's anchoring layer was developed.
- **Grid, daylight saving.** Anchor 2018-01-01; EU. The provider's treatment of the clock-change hours was kept.
- **Load column and unit conversion, temperature source, holiday calendar (only Spanish national holidays generated
  by rule are in the store), the rule that assigns the categories:** not recorded here; see the provider.
- **Tier and windows.** Test; 890 test windows of 481 units (1,237 of 486 in all). Short record: 243 days of training
  origins before the cut ([above](#training-validation-and-test-sets)).

### 6. GoiEner households (Spain)

- **Source.** As for population 5.
- **Units.** A store of 6,226 households on a time axis from 2018-01-01 to 2020-01-31, thinned for the panel to
  month-start origins and every sixth window per household, then to the panel's 1,109 households. The carrier's
  training windows come from all households of the store.
- **Minimum history.** The panel counted the 10,248 hours of history required at the origin from the start of the time
  axis, not from the household's first reading.
- **Grid, category, daylight saving.** Anchor 2018-01-01; Residential; EU. Load column, unit conversion, temperature
  source and holiday calendar: not recorded here; see the provider.
- **Tier and windows.** Test; 696 test windows of 696 units (one each; 1,258 of 1,109 in all). Short record: 304 days of
  training origins before the cut.

### 7. COFACTOR-SBHUB Oslo schools (Norway)

- **Source.** Lien et al., *Data in Brief* 2025, [doi:10.1016/j.dib.2025.112288](https://doi.org/10.1016/j.dib.2025.112288);
  data.sintef.no, [doi:10.60609/czgf-5e46](https://doi.org/10.60609/czgf-5e46), CC BY 4.0; 111,225,480 bytes (matching
  the portal). Hash: not recorded here.
- **Load and temperature.** 48 school files; column `ElImp` (Wh per hour) divided by 1,000; column `Tout`, with the
  hourly mean of the other buildings before a building's first row.
- **Grid, holidays, category, daylight saving.** The provider's fixed UTC+1 grid, anchor 2012-01-01 (records start
  between October 2012 and January 2021); Norwegian statutory holidays 2012–2025 (Easter rule); Public; EU.
- **Tier and windows.** Preview; 581 test windows of 41 units (1,147 of 45 in all).

### 8. COFACTOR Drammen municipal buildings (Norway)

- **Source.** Lien et al., *Sci. Data* 2025, [doi:10.1038/s41597-025-04708-3](https://doi.org/10.1038/s41597-025-04708-3);
  Zenodo 10.5281/zenodo.14752397, version 3, CC BY 4.0; 22,417,463 bytes, MD5 `13640361f6fcb2bf428dfad3cfa60888`;
  45 building files.
- **Load and temperature.** Column `ElImp` (Wh per hour) / 1,000, gaps kept missing; column `Tout` (a reanalysis
  temperature), gaps interpolated within each building.
- **Grid, holidays, category, daylight saving.** The provider's fixed UTC+1 grid, timestamps labelling the start of the
  hour, anchor 2018-01-01 (rows with another offset, off the hour or duplicated are dropped); Norwegian statutory
  holidays 2018–2022; 20 kindergartens, 16 schools and 7 nursing homes → Public, 2 offices → Office; EU.
- **Tier and windows.** Preview; 689 test windows of 43 units (1,375 of 45 in all).
- **Corrected-store retraining.** Drammen's own data are unchanged, but its pooled training set contains CINELDI
  windows. On the original CINELDI store those windows were placed 59 days too early in absolute time, so part of the
  CINELDI windows in a Drammen pool built from that store ended after Drammen's validation start (none after its
  training cut). The released Drammen carrier was trained from scratch with the corrected store, whose CINELDI windows
  all end before Drammen's validation start (`store` = `corrected` in `results/datasets.csv` and in
  `checkpoints/MANIFEST.json`).

### 9. CINELDI industrial customers (Norway)

- **Source.** Sandell et al., *Data in Brief* 2023, [doi:10.1016/j.dib.2023.109121](https://doi.org/10.1016/j.dib.2023.109121);
  Zenodo 10.5281/zenodo.10361330, version 2.3, CC BY 4.0. File size and hash: not recorded here.
- **Load.** 45 customer files with rows `Bus_ID;Timestamp;Load_kWh`, hourly, 1 March 2019 to 17 March 2022; kWh per
  hour as kW.
- **Grid.** The files use the local wall clock; the stamps are placed on a fixed UTC+1 grid, duplicates (autumn)
  averaged, the missing spring hour left missing.
- **Temperature.** None in the record. The city-mean outdoor temperature of the Oslo school data (population 7), about
  90 km away, is used as a proxy.
- **Holidays, category, daylight saving.** Norwegian statutory holidays; every customer Industrial (after the title of
  the record; the record has no categories); EU.
- **Tier and windows.** Preview; 475 test windows of 45 units (929 of 45 in all).
- **The corrected store.** The store as first built wrote `start_iso` = 2019-01-01 but placed the first observed hour
  (1 March 2019) at index 0, so every date derived from the anchor was 59 days early. The corrected store, built on
  18 October 2026 by the first release's correction, pads the 1,416 hours from 1 January to 28 February 2019 as
  unobserved load, so that index 0 is truly 2019-01-01; the loads, the day types and the window selection are
  unchanged, and every origin index is the original one plus 1,416 hours. The anchor fixes the annual phase of the
  carrier's tokens, the anchoring layer's temperature climatology and its daylight-saving state, and the absolute time
  of the pooled training set. ankyra-lite's CINELDI carrier was trained from scratch with the corrected store, anchored
  on it and scored on it; the dates on this page are true dates. The baselines on CINELDI are those re-scored on the
  corrected store by the first release's correction (re-forecast where the start date enters; the trained baselines
  that read the calendar retrained with the same code and seeds). LSTM-T-KAN-M, which the first release does not
  have, was retrained on the corrected store in the same way, with the same code, protocol, learning-rate grid and
  seeds; before that, the retraining code reproduced its stored original-store forecast bit for bit. The retrain passed
  its checks (training cutoff 2021-04-01, the CINELDI test start; selected learning rate 10⁻³; best epochs 14, 12 and 11
  for seeds 0, 1 and 2) and is the forecast scored in every CINELDI row, so the column `flag` is empty on every row
  (`populations` → `cineldi` → `extra_baselines` in `results/compare_baselines.json`).
- **Caveats.** The temperature is a proxy and the category is assumed.

### 10. Suzhou industrial park (China): training data only

- **Source.** Zhou et al., *Sci. Data* 2023, [doi:10.1038/s41597-023-02786-9](https://doi.org/10.1038/s41597-023-02786-9);
  OSF 10.17605/OSF.IO/AGK8S, CC BY 4.0. Four category aggregates, 2016–2019, anchor 2016-01-01; daylight saving none.
  Load column, aggregation, temperature source and holiday calendar: not recorded here; see the provider.
- **Role.** A source of borrowed training windows ([above](#windows-borrowed-from-other-populations)), within its own
  training period. ankyra-lite has no forecast for it, and it is in no table of this repository.

## What cannot be rebuilt from this page

- **BDG2, both GoiEner populations and the Suzhou park.** The archive version, the load column, the unit conversion, the
  temperature source and the holiday calendar are not recorded here; for GoiEner and the park, neither is the rule
  that forms the categories.
- **Unit and window lists.** The units and origins of each test panel are not published. The rules above reproduce the
  selection only with the same archive version and unit order.
- **The temperature-anomaly scale.** The recipe is given above; its value per population is not listed.
- **The carrier's training rows.** They follow from the stores, the cuts and the recipe's seeds
  (`ankyra_lite.carrier.train.TrainConfig`); the package reproduces the study's rows row for row on Oslo and BDG2
  (`tests/test_train.py`, needs the study's local data).
- **Provider-side processing.** A complete, finite prepared series does not show that every value is a direct physical
  observation. Gap filling by the provider, later revisions of the weather archives and the pretraining corpora of the
  baseline foundation models lie outside what this page documents.
- **Prepared stores and forecasts.** The prepared arrays, the baselines' forecasts and their code are not distributed.
  The scored statistics are in [`results/`](../results/README.md); the carrier checkpoints are in `checkpoints/`
  ([METHOD.md](METHOD.md#released-weights)).
