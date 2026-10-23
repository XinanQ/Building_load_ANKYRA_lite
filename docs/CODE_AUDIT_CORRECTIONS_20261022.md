# Code-audit corrections and maintenance provenance (version 0.2.1)

22 October 2026. Version 0.2.1 is a maintenance release after a code audit of 0.2.0. It changes code paths that the
released evaluation never took and the documentation of methods, readouts and limits; it does not change a forecast
result. No training or scoring was repeated.

## Provenance

- **Every result in this repository was produced by version 0.2.0** (commit 6bcd706, tag v0.2.0): `results/`,
  `figures/` and the 81 checkpoints in `checkpoints/` are byte-identical in 0.2.1. The scored runs keep the code
  identities they were recorded with; a later code hash is not evidence for an earlier scored run.
- **Default outputs are unchanged.** With the released Rw six-seed checkpoints, the 0.2.1 package reproduces the 0.2.0
  outputs bit for bit on six synthetic regression cases (an ordinary window, a shutdown, a micro-load, a window where
  the nonnegative projection binds, rejection of a missing hour at the normal entry, and the low-level missing-hour
  bypass): 47 arrays and every result field. This is a regression check of the package on synthetic histories, not a
  re-run of the real evaluation panels.

## What changed in the code

| Change | Before (0.2.0) | Now (0.2.1) |
|---|---|---|
| Continuation switch in `evaluation/cost_timing.py` and `evaluation/robustness.py` | switching the layer off passed an empty argument, so `forecast()` used its default (on). The released runs had the layer on, so no released row is affected | the switch is passed explicitly as `continuation=True/False` |
| Monthly features with a missing calendar (`ankyra_lite/carrier/features.py`) | the workday/weekend statistics of the older 744-hour blocks used no weekday fallback for missing day types | missing day types fall back to the weekday on the same absolute hour axis, as the daily and future tokens always did; given holiday labels are kept. Complete calendars give the same features as before. The public `forecast()` path requires a complete integer calendar and is unaffected |
| Wilcoxon direction (`evaluation/scoring.py`, two call sites in `evaluation/compare_baselines.py`) | the direction of a significant pairwise difference was read from the mean ranks across all models | it is read from the pair's own signed-rank sums; the two-sided p-values, the Holm correction, the mean ranks and Friedman/Nemenyi are unchanged. Re-checked on the released tables: all 186 Holm-significant pairs keep their direction |
| Continuation arithmetic for extreme custom constants (`ankyra_lite/continuation.py`) | very large constants could silently give inf or NaN, including zero times infinity | the original order (`F64 * exp(delta)`) is kept, exact input zeros including negative zero are restored, state windows are bypassed bit for bit, and a non-finite correction or output on an ordinary window raises `ValueError`. Default constants and ordinary finite inputs keep their arithmetic path |
| Re-anchoring cache (`evaluation/lite_reanchor.py`) | entries were keyed by the product label and the window limit, so a cache could be reused across implementations or configurations | each entry lives in a namespace bound to the implementation, configuration, seeds, input contents and product; payload hashes are checked; an exclusive claim file blocks a concurrent first write. A run that fails after claiming leaves the claim in place on purpose; to recompute, rename the `.claim` file (no deletion needed). Old cache folders are never read |

Regression tests for these changes are in `tests/test_audit_fixes.py`.

## Method and readout definitions (documentation corrections)

- The carrier can read **366 daily tokens plus 24 older fixed 744-hour blocks**, at most **1,110 days**, with masks
  for absent history. The **1,344-hour** context is an eligibility and anchoring requirement, not the whole feature
  span. The older blocks are not calendar months.
- The copy pointer uses complete days **within the latest 56 day positions**. Its attention is normalised, but its
  zero-initialised linear gates are signed; the complete correction is not a bounded convex mixture.
- Returned-span training uses a density-matched sample of daily-grid windows whose labels end no later than
  **C′ − 24 h**. It does not return every validation row or use every pre-cut window. Rw uses exponent **−1/2**, and its
  quantile, neutral raw-weight fallback and source normalisation use the sampler's draw probabilities.
- Complete availability gives at most **42 member predictions** from seven issue times and six members, with batching
  supported. Unavailable pseudo-origins reduce that count. The **0.655 s** measurement applies to six Oslo windows
  under its recorded single-thread CPU conditions.
- Observation thresholds differ by component: the seven-candidate level and the handover can use **90 %** observed
  pseudo-targets after the inherited short-gap rule; handover daily means need at least **12 h**; the fixed historical
  estimator and within-day trust need complete targets.

Let F be the trajectory before the continuation layer (the projected anchoring output, or the seasonal naive on state
windows) and F′ the delivered trajectory. The independent energy is **744L**, the main release energy comparison uses
**ΣF**, and the peak readout uses **Π(F)**. Both the projection and the continuation layer can separate the delivered
integral from 744L; switching the layer off removes only its own contribution. The **21.3 %–68.2 %** peak comparison
is Π(F) against max F′, not a pure operator effect on a fixed curve. The disclosed cross-carrier post-continuation
energy rows keep their delivered-curve definition and their numbers.

Reproducing **0.1.1** requires the same inputs, the original recipe-R three-seed checkpoints and the continuation layer
off. Switching the layer off for the Rw six-seed checkpoints gives that carrier's pre-continuation result.

## Subgroup labels of the external tests

The subgroup recorded as `units without own training windows` (the key in `results/external_tests.csv` and the score
records) was built from the first-stage training rows only (`split == "train"`). Stage 2 returns validation-span
windows to training, so a unit in this subgroup may have contributed returned training rows. The documents now call it
*units without first-stage own training windows* (and the complement *units with first-stage own training windows*);
memberships and numbers are unchanged. It does not establish transfer to units excluded from every fitting stage.

## Interface and arithmetic boundaries

The normal carrier-inference path requires a complete final context. A precomputed-forecast mapping can enter the
inherited off-state branch before complete-context validation, using the observed last-week maximum; the
seasonal-naive fallback then zero-fills missing last-week hours. This inherited rule does not establish that missing
hours represent a shutdown. The anchoring implementation is unchanged.

The continuation envelope is nonincreasing in real arithmetic and strictly decreasing when at least one coefficient is
positive; ZERO is constant. Opposite-sign anomalies with distinct decay rates allow at most one crossing, which need not
occur within the forecast horizon. The correction acts across all **744 h**. Nonzero values can underflow to zero, so
real-arithmetic positivity does not guarantee strict floating-point positivity for arbitrary constants.

## Information availability and audit limits

No added future-load read was found in the inspected paths. Load and temperature masking, pseudo-origin cut checks and
the audited cross-source absolute-time boundaries passed inspection. This does **not** certify that every input was
known at issue time: the historical panel contains the disclosed post-origin holiday announcements on **26 EWELD** and
**10 Cambridge** test windows. Calendar labels are retained, and there is no generic announcement-time filter.

The audit did not rebuild every raw source file, retrain checkpoints, rescore real targets or re-derive every earlier
subgroup. Frozen evidence and later maintenance are kept separate.
