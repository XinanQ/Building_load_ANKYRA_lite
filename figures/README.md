# Figures

All figures are drawn by [`make_figures.py`](make_figures.py) from the result files in [`../results`](../results) (and, for
Figure 7, from the anchoring layer `ankyra_lite.anchor` and the exact properties in [`../theory/`](../theory/)). Each
figure is written twice: `figN_name.pdf` (vector) and `figN_name.png` (300 dpi). The style follows the figures of the
ANKYRA release; the reference forecaster is ankyra-lite (red), and the comparison set is whatever the result files hold:
the 21 baselines (the 20 external baselines of the release and LSTM-T-KAN-M, drawn as ordinary baselines), 22
forecasters in all. ANKYRA itself and its versions are never drawn. A result file that holds a forecaster outside these
22 is stale: the script drops the rows of any LSTM-T-KAN model other than LSTM-T-KAN-M (the pattern `RETIRED` in
`make_figures.py`) with a warning (with `--strict`, the figure fails).

The forecasts of ankyra-lite are those of version 0.2.0: the carrier network is trained with recipe Rw (from scratch
per population with early stopping on the 93-day validation set, then the validation span returned to the training set
and 8 more epochs with each training window's loss weighted by the inverse-noise rule; seeds 0–5 averaged; the 54
checkpoints are in [`../checkpoints`](../checkpoints)`/<population>/rw/`), anchored by the unchanged anchoring layer,
the seasonal-naive state rule on state windows, then the origin-continuity layer. Hourly errors score the delivered
trajectory after that layer; energy, level and peak are read before it (readout rule A). Every figure of ankyra-lite's
forecasts says so in its caption (Figure 6 in its footer); Figure 9c, beside Figure 9 (kept unchanged in form), shows
what the continuation layer adds by lead time, and Figure 18 shows the recipe study behind recipe R (the negative
results included) and the 0.2.0 study's test read. `make_figures.py` recognises 0.2.0 result files by
`continuation_layer.csv`; without it (the 0.1.x result files) every figure is drawn as in 0.1.1 and Figure 9c is
skipped.

```bash
pip install matplotlib                     # needed for the figures only
python -B figures/make_figures.py --strict # all figures, as published
python -B figures/make_figures.py --only fig2,fig9b --strict
```

The published files were rendered with `--strict` from the regenerated result files of version 0.2.0 (22 figures, 44
files), with Python 3.11, Matplotlib 3.11.0 and NumPy 2.4.4, on 21 October 2026. Every figure was opened and read after
rendering; the captions or layouts of Figures 1b, 7, 9b, 9c, 10, 11 and 18 were corrected and re-rendered in that
check. Only Figure 7 imports PyTorch (through the anchoring layer); every other figure
reads CSV and JSON files only.

| Option | Effect |
|---|---|
| `--only fig2,fig9b` | draw a subset (file stems or their `figN` prefixes) |
| `--results DIR`, `--out DIR` | read the result files from / write the figures to another folder |
| `--standins DIR` (or `ANKYRA_LITE_FIGURE_STANDINS`) | read result files missing from `results/` from `DIR`; those figures carry a "stand-in data, not results" watermark |
| `--strict` | exit with status 1 if a figure is skipped or drawn from stand-in data, if a `model_class` of the result files has no colour (see [Colours](#colours)), or if a result file holds rows of an LSTM-T-KAN model other than LSTM-T-KAN-M (`RETIRED`; outside the 22: a stale file). |

A missing input skips only its own figure (`figN: skipped, input not found: ...`); so does an input that lacks what the
figure needs (`figN: skipped, ...`, e.g. `rank_tests.csv` without its pooled row after a run on fewer than the six
release test populations). The other figures are still drawn. A figure that fails for any other reason is reported as
`FAILED` with its traceback, and the script then exits with status 1. Nothing is downloaded; the script reads only local
files and, for Figure 7, never writes into the original ANKYRA checkout.

## Figures

| File | What it shows | Source |
|---|---|---|
| `fig0_advantage` | The README's first evaluation figure: ankyra-lite against the four zero-shot foundation-model variants (TimesFM, TimesFM-X, Chronos-2, Chronos-2-X) on every population, by the release's three primary readouts: a, hourly error (unit-equal RMS ratio over the 744 hours of the delivered trajectory); b, monthly energy error (all units; ankyra-lite read before the continuation layer); c, the monthly peak readout against the same readout applied to each variant's trajectory (the release's rule). Improvement with 95% unit-and-month intervals; filled = resolved better, red = resolved worse, open = not resolved; above each panel, the count of its contrasts resolved better, not resolved and resolved worse. Each variant has its own marker and its class colour (see [Colours](#colours)); the caption says where the other baselines are (figures 2, 3, 5 and 11). | `benchmark_pairwise.csv` (`subset` = `late`), `energy_error.csv` (`unit_set` = `all units`), `peak_readout.csv` (`comparator` = `same readout on <model>`) |
| `fig1_architecture` | Architecture of ankyra-lite, unrolled in time (drawn by [`architecture.py`](architecture.py)). a, the issue times on one unit's hourly record: the carrier network is issued at the origin o and at up to six pseudo-origins o − 744k (Δ = 744 h), each time from the data before that issue time; each pseudo-origin forecast F^(k) covers a month already observed at o and is scored on it (e_k, drawn as the gap between daily means), and the anchoring layer turns F^(0) into the anchored month, which the continuation layer then corrects in its first hours and days (0.2.0; the carrier is drawn with recipe Rw and six seeds). b, the carrier network (CopyPointerNetwork, width 48): the token stacks X^day (366 × 69), X^mon (24 × 16) and X^fut (31 × 16) with the scale s, the two shared gated causal blocks with their dilations, the memory H (390 × 48), the 4-head attention read-out (31 day queries from the calendar, one learned global query v; outputs z_d and v), the copy pointer over the unit's last 56 complete days (query z_d, 16-d keys from H, day-type bonus) and the energy-budget decoder drawn as the product 744ℓ × p_d × q_d,h = F_d,h. c, the anchoring layer (a copy of ANKYRA 2.2.1; nothing trained): the level weights (six historical candidates and the carrier's ℓ, ∝ 1/MSE, shrunk to equal, K0 = 8), the handover weights α_w for days 1–7 / 8–14 / 15–21 / 22–31 (shrunk to ½, K0 = 2), the within-day trust ω ≤ ½, each with the pseudo-origin scores it reads, and the output (trajectory, energy 744 m̄, peak, state windows; energy and peak read before the continuation layer) with the measured time per forecast. The series and every glyph are synthetic and schematic. | drawn from the code (`ankyra_lite/carrier/model.py`, `features.py`, `inference.py`, `train.py`, `ankyra_lite/anchor/core.py`, `analog.py`, `readouts.py`, `ankyra_lite/forecast.py`); the parameter count, the time per forecast and the thread count from `cost_per_window.csv` (rows `carrier network parameters`, `ankyra-lite forecast ...` in s/window, `device: ...`). Without that file the output box says only that the forecast runs on the CPU, and the parameter count is the released configuration's (61,613). |
| `fig1b_data_splits` | Training set, validation set, training cut, 186-day gap and test set of each population, in calendar time (a) and aligned on the training cut (b), with the test windows / units and the validation share; the caption states the released recipe (early stopping on the validation set, then the validation span returned for 8 more epochs with the inverse-noise loss weights of recipe Rw; six seeds). | `datasets.csv` (`set`, `first_training_origin`, `validation_start`, `training_cut`, `test_start`, `last_test_origin`, `test_windows`, `test_units`, `validation_share_pct`, `calendar`) |
| `fig2_test_ranks` | a: mean per-unit rank of every forecaster on each population's test windows; b: head-to-head count (resolved better / not resolved / resolved worse) over the populations. | `benchmark_mean_unit_rank.csv`, `benchmark_pairwise.csv` |
| `fig3_test_pairwise` | Unit-equal hourly improvement of ankyra-lite with 95% unit-and-month intervals, one panel per population, against the baselines given the same information as ankyra-lite (load, temperature, calendar), LSTM-T-KAN-M (load and calendar) and a selection of load-only baselines (TimesFM, Chronos-2, Holt–Winters, MSTL); dashed lines separate the three information tiers, and the caption lists the models of each and says where the other baselines are (figures 2 and 8). Ridge (per unit) is not drawn here, as in the release's figure. The axis is widened (up to −60 … +80%) to hold every point estimate; interval ends beyond it are arrows. | `benchmark_pairwise.csv` |
| `fig4_ablation` | ankyra-lite against reduced and alternative versions of itself (hourly error): a, parts of the anchoring layer, the state rule and the continuation layer switched off, the carrier alone, the history-only configuration; b, one three-seed group (seeds 0–2 or 3–5) instead of the six seeds; c, other carriers, recipes and configurations inside the same anchoring layer (the recipe variants are named F and V as in Figure 18: F, 8 more epochs without returning the validation span; V, a proportional validation set with early stopping; the 0.2.0 configurations by recipe R or Rw, seeds 0–2 or six, with or without the layer, 0.1.0 = R, seeds 0–2, no layer). Every variant of a panel has its own marker and colour (see [Colours](#colours)). Second row, the handover: d, ankyra-lite against the fixed division (history level and daily path, no handover), a fixed half handover weight, one handover weight per window, one whole-window combination weight of the history side and the carrier, and their half-and-half average; e, ankyra-lite (weeks 1-4) and the fixed division (diamonds) against the carrier alone by forecast week; f, the mean handover weight on the carrier's daily means by week (½ = the prior). The counterpart of the release's handover-and-ablation figure (its panels b-d); the row is drawn when `handover_granularity.csv` and `lead_weeks.csv` exist. | `ablation.csv` (rows with `error` = `hourly`), `handover_granularity.csv`, `lead_weeks.csv` |
| `fig5_peak_operator` | The peak readout (read from the trajectory before the continuation layer) against the maximum of ankyra-lite's own delivered trajectory, after the layer (a) and against last month's observed peak (b); c, head-to-head against the same readout applied to each baseline's trajectory (the row `same readout on the carrier alone` is not a baseline and is not counted there). | `peak_readout.csv` |
| `fig6_example_window` | One Cambridge test window: the observed load, ankyra-lite before (F, `ankyra_before_layer_kw`) and after (F′, the delivered trajectory, `ankyra_kw`) the continuation layer, the carrier alone and TimesFM 2.5, with a panel on the first three days; daily means and the error by forecast week. The footer names the recipe, the data source and how the window was chosen. | `example_window_cambridge.csv` (`ankyra_kw` is ankyra-lite; the column keeps the release's name), `example_window_cambridge.json` |
| `fig7_operators` | Exact properties of the anchoring layer, numbered as in [`../theory/README.md`](../theory/README.md) with proofs in [`../theory/PROOFS.md`](../theory/PROOFS.md): block replacement on the window of Figure 6, from the carrier alone to ankyra-lite's anchored trajectory F before the continuation layer (a, P4); historical support (b, P7); shrinkage bounds (c, P8, P9); clipping versus energy-preserving projection (d, P11, P12); the peak readout and its four-term error decomposition (e, f; P15, P16). The functions are those of [`../theory/operators.py`](../theory/operators.py), checked by [`../theory/test_operators.py`](../theory/test_operators.py). | `example_window_cambridge.csv/.json`, `ankyra_lite.anchor`, [`../theory/`](../theory/) |
| `fig8_loss_metrics` | RMSE, MAE, CV(RMSE) and WAPE of every forecaster on every population, coloured by the loss relative to ankyra-lite. | `conventional_metrics_late.csv`, `benchmark_mean_unit_rank.csv` (row order) |
| `fig9_loss_by_day` | Hourly CV(RMSE) by forecast day 1–31, geometric mean over a fixed set of units (six release test populations); ankyra-lite's delivered trajectory, after the continuation layer. Kept unchanged in form; Figure 9c sits beside it. | `lead_day_metrics.csv` |
| `fig9c_continuation_layer` | Beside Figure 9 (0.2.0 only): the origin-continuity layer on against off on the same carrier and anchoring layer (the seasonal naive on state windows in both, where the layer is bypassed), by lead time. a, the units of all populations pooled: hour 0, hours 0–2, day 1, days 2–7, days 8–31, the whole window, and the monthly energy of the delivered trajectory, with 95% unit-and-month intervals and the values written out (bold: resolved); b, a heat map by population with the same segments (filled cells and bold numbers: the interval excludes zero; the colour scale is compressed). The caption says that the energy readout is read before the layer (readout rule A); the footer gives the windows and state windows of each population and, from `continuation_record.csv`, the windows whose reference day fell back to forecast day 7 (no same-class day in days 1–7), the count of spans with a mean at or below zero (x = 0, flagged; a release rule) and the gap between the delivered mean and the energy readout (median, 99th percentile, maximum). | `continuation_layer.csv` (`set`, `segment`, `error`, `log_ratio`, `um_low`, `um_high`, `improvement_pct`, `windows`, `units`, `state_windows`); footer: `continuation_record.csv` (row `all nine populations`) |
| `fig9b_energy_by_day` | Error of the energy through each forecast day, the same aggregation, with the number of days on which ankyra-lite is lowest; ankyra-lite's energy to date is read from its trajectory before the continuation layer (readout rule A). | `lead_day_energy.csv` |
| `fig10_loss_by_day_relative` | Each named baseline's daily loss relative to ankyra-lite's (ankyra-lite = 0, the red line; the legend says so, and the line carries no in-panel label so that no curve is hidden). | `lead_day_metrics.csv` |
| `fig11_energy_error` | a: improvement in the monthly energy error over each baseline, per population (BDG2 also without the three near-zero meters); b: head-to-head count. ankyra-lite's energy is its readout before the continuation layer. | `energy_error.csv`, `benchmark_mean_unit_rank.csv` (row order) |
| `fig12_consistency` | Position of each forecaster among all forecasters on each population (mean per-unit rank), coloured by model class. | `benchmark_mean_unit_rank.csv`, `benchmark_pairwise.csv` (classes) |
| `fig13_intervals` | Prediction intervals on GoiEner households: coverage of the 80% and 90% bands and the 80% Winkler score of the anchoring layer's interval around ankyra-lite and of the native TimesFM and Chronos-2 quantiles, with the Winkler contrasts. | `intervals_by_population.csv` (arm containing `lite`, `timesfm_native`, `chronos_native`), `intervals_winkler_contrasts.csv` |
| `fig14_block_attribution` | a: median-unit share of the level, daily-path and within-day blocks in the hourly squared error of ankyra-lite, TimesFM and Chronos-2-X; b, c: ankyra-lite's improvement over TimesFM and over Chronos-2-X by block. | `block_shares.csv` |
| `fig15_rank_tests` | a: mean ranks pooled over the six release test populations with the Nemenyi critical difference and Holm-corrected Wilcoxon tests; b: the paired tests per population, with the names of the baselines significantly better than ankyra-lite next to the bar (two at most; with more, the full list is in the caption). A p-value written as a bound (`<1e-300`, below the double-precision range) is read as that bound and printed as such. Skipped (with a message) when `rank_tests.csv` has no pooled row. | `rank_tests.csv` |
| `fig16_intervals_populations` | Coverage of the nominal 80% interval on every population and by forecast week, ankyra-lite's interval against the native quantiles. | `intervals_by_population.csv` |
| `fig17_external_tests` | The external tests ([EXTERNAL_TESTS.md](../docs/EXTERNAL_TESTS.md)), the counterpart of the release's Figure 17. With the 0.2.0 rows of `external_tests.csv`: the 0.2.0 configuration re-run on the test sets of the frozen external test of 19 October 2026 (the same windows, comparators and estimators) and scored once; carriers trained anew for each test set with recipe Rw from the same stage-1 runs, six seeds averaged, the unchanged anchoring layer, the seasonal-naive state rule and the continuation layer. Rows: LCL, UNICON and HKUST (previously read populations), GoiEner households out of time (same-source out-of-time supplementary check) and, in a–c only, Helsinki (trained and sealed, not scored; an empty row). a–c: the three primary readouts (hourly error against zero-shot TimesFM 2.5, the weekly seasonal naive and the carrier-free control with the state rule); d–f: the same contrasts by lead time (forecast hours 0–2, day 1, days 2–7, days 8–31); g–i: descriptive contrasts between ankyra-lite's own configurations by lead time (g: 0.2.0 against the 0.1.0 configuration; h: the continuation layer on against off; i: the loss weighting of recipe Rw against recipe R, without the layer). Welch 95% intervals where the record has a seed layer, else unit-and-month only; filled = resolved. The caption states the identities, the nominal intervals without multiplicity correction and the pretraining overlap (LCL is in TimesFM 2.5's, Chronos's and Chronos-2's pretraining data); the footer gives the windows, units and months of each test set and the outcome of the day-1 rule (`claim allowed`: the day-1 gap to TimesFM 2.5 narrowed by at least a third of the development shrinkage; it is not a claim that ankyra-lite leads TimesFM 2.5 on day 1). With the 0.1 rows (version 0.1's file) the figure is that of 0.1.1. | `external_tests.csv` (`test_set_id`, `block`, `model`, `readout`, `improvement_pct`, `interval_low_pct`, `interval_high_pct`, `label`, `windows`, `units`, `months`), written by `evaluation/external_tables.py` from the score records |
| `fig18_recipe_history` | a–c: the training-recipe study behind recipe R, the base of recipe Rw (test windows read before; a description); d–e (0.2.0 only): the 0.2.0 study's one test read, d by readout (0.2.0 against 0.1 and the single changes: the layer on the same Rw6 carrier, the loss weighting Rw6 vs R6, six vs three seeds R6 vs R3; energy and peak read before the layer, so the layer contrast is zero by construction on the energy rows and the peak) and e 0.2.0 against 0.1 by population group (whole window, hours 0–2, day 1). All recipes start from the same from-scratch checkpoints (three seeds). F and R then train exactly 8 more epochs: F on the unchanged training set (control), R (released) with the validation span returned; V instead re-splits with a proportional validation set (10% of the own record, 38–93 days) and stops early on it (5 to 30 epochs; the start weights count as epoch 0). a: hourly R vs F, V vs F and V vs R on the three short-record populations, their switch and non-switch layers (test windows whose season the return newly covers, and the rest), the nine populations, the six development and the four long-record populations, with Welch 95% intervals (unit-and-month bootstrap combined with the seed spread); b: R vs F by readout (energy, peak, forecast day 1, days 2–7, days 8–31); c: R vs F per population. The footer lists every contrast resolved against R and the lowest population estimate. | `recipe_history.csv` as written by `evaluation/recipe_tables.py` (`comparison`, `populations`, `layer`, `quantity`, `improvement_pct`, `interval_low_pct`, `interval_high_pct`, `label`; panel c from `by_population_pct` of the nine-population rows, point estimates only). The rows against the un-continued from-scratch stage (unit-and-month intervals only) are in the file but not drawn. A plain layout (`group`, `contrast`, `readout`, `pct`, `lo`, `hi`, `label`) is also read. |

## Colours

Each baseline is coloured by the `model_class` the result files give it; the legends list only the classes drawn.

| `model_class` | Legend entry | Colour |
|---|---|---|
| `same information, trained` | Same information · trained | `#253494` |
| `same information, zero-shot foundation model` | Same information · zero-shot foundation model | `#1D91C0` |
| `load and calendar, trained` | Load and calendar · trained (LSTM-T-KAN-M) | `#D9B400` |
| `load only, zero-shot foundation model` | Load only · zero-shot foundation model | `#807DBA` |
| `load only, statistical` | Load only · statistical | `#41AB5D` |
| `load only, trained` | Load only · trained | `#EC7014` |
| `per-unit regression` | Per-unit regression (calendar, temperature) | `#8C6D31` |
| `profile`, `naive`, `zero-shot gradient boosting` | Profiles, naive, zero-shot GBT | `#9E9E9E` (`naive` `#BDBDBD`) |

A class string not in this table is drawn with a guessed colour and a printed warning; with `--strict` it is an error.
The gold of the new class was chosen to stay apart from the existing class colours in normal vision and under simulated
protanopia, deuteranopia and tritanopia (OKLab distance at least 0.088 to every other class colour and to ankyra-lite's red).

Figure 0 keeps the class colours of the four zero-shot variants (`#807DBA` load only, `#1D91C0` the shared covariate
set) and gives each its own marker, `FM_VARIANTS` in `make_figures.py` (TimesFM circle, TimesFM-X diamond, Chronos-2
square, Chronos-2-X triangle), in a fixed order within each population, so the variants stay apart without relying on
colour; a contrast resolved against ankyra-lite is filled in ankyra-lite's red, as in Figure 3.

Figure 4 does not colour by class: each variant of a panel gets the next of ten styles, `ABL_STYLE` in `make_figures.py`
(circle `#1A1A1A`, diamond `#E6AB02`, square `#7B5EA7`, triangle up `#D55E00`, triangle down `#56B4E9`, filled plus
`#332288`, filled cross `#882255`, star `#1B9E77`, hexagon `#CC79A7`, pentagon `#6A3D9A`). No two styles share a marker or a colour. The
smallest OKLab distance over normal vision and the three simulated colour-vision deficiencies is 0.104 between any two of
the first eight colours (the largest panel has eight variants), 0.071 among all ten, and 0.088 to ankyra-lite's red. A
panel with more than ten variants reuses markers and colours in new pairs, never the same pair twice, and the script
prints a warning.

The second row of Figure 4 has its own styles: panel d `FIG4_MECH_STYLE` (circle `#4D4D4D`, square `#C0762A`, diamond
`#7B5EA7`, triangle up `#1B7F79`, triangle down `#56B4E9`; one marker and colour per contrast); panels e and f encode
the forecast week by one hue from light to dark (`WEEK_RAMP_LITE` for ankyra-lite, `WEEK_RAMP_GREY` for the fixed
division, `WEEK_RAMP_CARRIER` for the handover weight), an ordinal encoding with the week also given in the legend.

## Notes

- Populations whose result rows carry a `flag` are marked † and the flag text is given in the caption or the footer
  (the flags are described in [`../results/README.md`](../results/README.md)).
- "The six release test populations" are BDG2 2017, Cambridge, HEEW Arizona, EWELD and the two GoiEner populations
  (the release's test tier); Oslo, Drammen and CINELDI were the release's preview tier. Every population has its own
  test set in this repository; the day-by-day tables, the block shares and the pooled rank tests cover the six only,
  as in the release.
- Display names: `TiDE` is drawn as **Linear-context**, as in the first release's figures (its result-file label there and
  here is `Linear-context (TiDE implementation)`; the model id `TiDE` is kept): as implemented in the release, its
  covariate path is inert and its forecast is a linear map of the load context. Every figure that draws it says so in
  its caption. Result-file labels longer than 24 characters are drawn as the model id, so LSTM-T-KAN appears as
  `LSTM-T-KAN-M`.
- LSTM-T-KAN-M: the model and training code are copied line by line from the authors' public repository (commit
  `ce24938`), with the authors' exp2 feature list (load and calendar features). It is not a strict reproduction of the
  paper; its ten deviations (among them one model for all 744 hours and one model per population) and the dated
  amendment of the training budget (at most 30 epochs, patience 8) are listed in
  [`../docs/EVALUATION.md`](../docs/EVALUATION.md). On CINELDI it was retrained on the corrected data store with the
  same code and seeds, as were the release's other calendar-using baselines. Figures 2 and 3 say this in their captions
  (the LSTM-T-KAN-M note).
- Caption notes: every figure of ankyra-lite's forecasts names its configuration (0.2.0: recipe Rw, six seeds averaged,
  the continuation layer, readout rule A; Figure 17: carriers trained for each test set with Rw, six seeds); the figures of the
  whole comparison set give its size (22 forecasters: ankyra-lite and the 21 baselines), and Figures 0 and 3, which
  draw a selection, say where the other baselines are. No figure draws ANKYRA itself or its versions, and none draws a
  forecaster outside the 22. ANKYRA is named only as the source of the anchoring layer (Figures 1 and 7, "ANKYRA 2.2.1") and of the
  release's rules and example window.
- External tests: Figure 17 draws `external_tests.csv` only; no comparison with ANKYRA itself or its versions is in
  that file. The Helsinki retrospective set is trained and sealed, not scored, and appears as an empty row.
- Axes of interval plots are widened to hold every point estimate (within fixed caps); an interval end beyond the axis
  is an arrow, and a point beyond a cap is drawn at the edge with its value.
- The figures are descriptions of a re-evaluation: the test windows had been read before, and 0.2.0 was read on them
  once more. Figure 17 reads test sets that the frozen external test of 0.1 had read once (previously read populations
  and a same-source out-of-time supplementary check).
