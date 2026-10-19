# Figures

All figures are drawn by [`make_figures.py`](make_figures.py) from the result files in [`../results`](../results) (and, for
Figure 7, from the anchoring layer `ankyra_lite.anchor` and the exact properties in [`../theory/`](../theory/)). Each
figure is written twice: `figN_name.pdf` (vector) and `figN_name.png` (300 dpi). The style follows the figures of the
ANKYRA release; the reference forecaster is ankyra-lite (red), and the comparison set is whatever the result files hold:
the 21 baselines (the 20 external baselines of the release and LSTM-T-KAN-M, drawn as ordinary baselines), 22
forecasters in all. ANKYRA itself and its versions are never drawn. A result file that holds a forecaster outside these
22 is stale: the script drops the rows of any LSTM-T-KAN model other than LSTM-T-KAN-M (the pattern `RETIRED` in
`make_figures.py`) with a warning (with `--strict`, the figure fails).

The forecasts of ankyra-lite come from the released training recipe, recipe R: the carrier network is trained from
scratch per population with early stopping on the 93-day validation set, then the validation span is returned to the
training set and training continues for 8 more epochs (seeds 0, 1 and 2 averaged; the 27 checkpoints are in
[`../checkpoints`](../checkpoints)). Every figure of ankyra-lite's forecasts says so in its caption (Figure 6 in its
footer), and Figure 18 shows the study behind that choice, the negative results included.

```bash
pip install matplotlib                     # needed for the figures only
python -B figures/make_figures.py --strict # all figures, as published
python -B figures/make_figures.py --only fig2,fig9b --strict
```

The published files were rendered with `--strict` from the regenerated result files of this version (20 figures, 40
files; Figure 17, the external tests, and Figures 4 and 18 were rendered last, on 19 October 2026), with Python 3.11,
Matplotlib 3.11.0 and NumPy 2.4.4. Only Figure 7 imports
PyTorch (through the anchoring layer); every other figure reads CSV and JSON files only.

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
| `fig1_architecture` | Structure of ankyra-lite: the inputs at the origin (366 × 69 daily tokens, 24 × 16 monthly tokens, 31 × 16 future-day tokens, the scale), the carrier network (ordered dilated gated-convolution encoder, attention read-out, energy-budget decoder with the rank-8 hourly dictionary, copy pointer over the last 56 days; trained from scratch per population, then 8 more epochs with the validation span returned), its forecasts at the origin and six pseudo-origins, the anchoring layer (level weights, handover weights for days 1–7 / 8–14 / 15–21 / 22–31, within-day trust, state windows) and the outputs (trajectory, energy, peak), with the measured time per forecast. | drawn from the code (`ankyra_lite/carrier/model.py`, `features.py`, `train.py`, `ankyra_lite/anchor/core.py`); the parameter count, the time per forecast and the thread count from `cost_per_window.csv` (rows `carrier network parameters`, `ankyra-lite forecast ...` in s/window, `device: ...`). Without that file the output box says only that the forecast runs on the CPU, and the parameter count is the released configuration's (61,613). |
| `fig1b_data_splits` | Training set, validation set, training cut, 186-day gap and test set of each population, in calendar time (a) and aligned on the training cut (b), with the test windows / units and the validation share; the caption states the released recipe (early stopping on the validation set, then the validation span returned for 8 more epochs). | `datasets.csv` (`set`, `first_training_origin`, `validation_start`, `training_cut`, `test_start`, `last_test_origin`, `test_windows`, `test_units`, `validation_share_pct`, `calendar`) |
| `fig2_test_ranks` | a: mean per-unit rank of every forecaster on each population's test windows; b: head-to-head count (resolved better / not resolved / resolved worse) over the populations. | `benchmark_mean_unit_rank.csv`, `benchmark_pairwise.csv` |
| `fig3_test_pairwise` | Unit-equal hourly improvement of ankyra-lite with 95% unit-and-month intervals, one panel per population, against the baselines given the same information as ankyra-lite (load, temperature, calendar), LSTM-T-KAN-M (load and calendar) and a selection of load-only baselines (TimesFM, Chronos-2, Holt–Winters, MSTL); dashed lines separate the three information tiers, and the caption lists the models of each and says where the other baselines are (figures 2 and 8). Ridge (per unit) is not drawn here, as in the release's figure. The axis is widened (up to −60 … +80%) to hold every point estimate; interval ends beyond it are arrows. | `benchmark_pairwise.csv` |
| `fig4_ablation` | ankyra-lite against reduced and alternative versions of itself (hourly error): a, parts of the anchoring layer and the state rule switched off, the carrier alone, the history-only configuration; b, one seed instead of three; c, other carriers and recipes inside the same anchoring layer (the recipe variants are named F and V as in Figure 18: F, 8 more epochs without returning the validation span; V, a proportional validation set with early stopping). Every variant of a panel has its own marker and colour (see [Colours](#colours)). The counterpart of the release's handover-and-ablation figure. | `ablation.csv` (rows with `error` = `hourly`) |
| `fig5_peak_operator` | The peak readout against its own trajectory maximum (a) and against last month's observed peak (b); c, head-to-head against the same readout applied to each baseline's trajectory. | `peak_readout.csv` |
| `fig6_example_window` | One Cambridge test window: the observed load, ankyra-lite, the carrier alone and TimesFM 2.5; daily means and the error by forecast week. The footer names the recipe, the data source and how the window was chosen. | `example_window_cambridge.csv` (`ankyra_kw` is ankyra-lite; the column keeps the release's name), `example_window_cambridge.json` |
| `fig7_operators` | Exact properties of the anchoring layer, numbered as in [`../theory/README.md`](../theory/README.md) with proofs in [`../theory/PROOFS.md`](../theory/PROOFS.md): block replacement on the window of Figure 6, starting from the carrier alone (a, P4); historical support (b, P7); shrinkage bounds (c, P8, P9); clipping versus energy-preserving projection (d, P11, P12); the peak readout and its four-term error decomposition (e, f; P15, P16). The functions are those of [`../theory/operators.py`](../theory/operators.py), checked by [`../theory/test_operators.py`](../theory/test_operators.py). | `example_window_cambridge.csv/.json`, `ankyra_lite.anchor`, [`../theory/`](../theory/) |
| `fig8_loss_metrics` | RMSE, MAE, CV(RMSE) and WAPE of every forecaster on every population, coloured by the loss relative to ankyra-lite. | `conventional_metrics_late.csv`, `benchmark_mean_unit_rank.csv` (row order) |
| `fig9_loss_by_day` | Hourly CV(RMSE) by forecast day 1–31, geometric mean over a fixed set of units (six release test populations). | `lead_day_metrics.csv` |
| `fig9b_energy_by_day` | Error of the energy delivered through each forecast day, the same aggregation, with the number of days on which ankyra-lite is lowest. | `lead_day_energy.csv` |
| `fig10_loss_by_day_relative` | Each named baseline's daily loss relative to ankyra-lite's (ankyra-lite = 0, the red line; the legend says so, and the line carries no in-panel label so that no curve is hidden). | `lead_day_metrics.csv` |
| `fig11_energy_error` | a: improvement in the monthly energy error over each baseline, per population (BDG2 also without the three near-zero meters); b: head-to-head count. | `energy_error.csv`, `benchmark_mean_unit_rank.csv` (row order) |
| `fig12_consistency` | Position of each forecaster among all forecasters on each population (mean per-unit rank), coloured by model class. | `benchmark_mean_unit_rank.csv`, `benchmark_pairwise.csv` (classes) |
| `fig13_intervals` | Prediction intervals on GoiEner households: coverage of the 80% and 90% bands and the 80% Winkler score of the anchoring layer's interval around ankyra-lite and of the native TimesFM and Chronos-2 quantiles, with the Winkler contrasts. | `intervals_by_population.csv` (arm containing `lite`, `timesfm_native`, `chronos_native`), `intervals_winkler_contrasts.csv` |
| `fig14_block_attribution` | a: median-unit share of the level, daily-path and within-day blocks in the hourly squared error of ankyra-lite, TimesFM and Chronos-2-X; b, c: ankyra-lite's improvement over TimesFM and over Chronos-2-X by block. | `block_shares.csv` |
| `fig15_rank_tests` | a: mean ranks pooled over the six release test populations with the Nemenyi critical difference and Holm-corrected Wilcoxon tests; b: the paired tests per population, with the names of the baselines significantly better than ankyra-lite next to the bar (two at most; with more, the full list is in the caption). A p-value written as a bound (`<1e-300`, below the double-precision range) is read as that bound and printed as such. Skipped (with a message) when `rank_tests.csv` has no pooled row. | `rank_tests.csv` |
| `fig16_intervals_populations` | Coverage of the nominal 80% interval on every population and by forecast week, ankyra-lite's interval against the native quantiles. | `intervals_by_population.csv` |
| `fig17_external_tests` | The external tests of this version ([EXTERNAL_TESTS.md](../docs/EXTERNAL_TESTS.md)), the counterpart of the release's Figure 17 (checks with the forecaster frozen on external data). Carriers trained for each test set with recipe R; each test set scored once. Rows: LCL, UNICON and HKUST (first read of this candidate on a previously read population), GoiEner households out of time (same-source out-of-time supplementary check) and, in a–c only, Helsinki (trained and sealed, not scored; an empty row). a–c: the three primary readouts (hourly error against zero-shot TimesFM 2.5, the weekly seasonal naive and the carrier-free control with the state rule), six seeds (circles) and the three-seed release configuration (grey diamonds); d–f: the same contrasts on forecast day 1, days 2–7 and days 8–31; g–i: descriptive contrasts with ankyra-lite's own configurations (the history-only configuration; R against the plain continuation F, three seeds; the continuation and span return against the from-scratch stage alone, unit-and-month interval only). Welch 95% intervals combining the unit-and-month bootstrap with the seed spread; filled = resolved. The caption states the identities, the 15 nominal intervals without multiplicity correction and the pretraining overlap (LCL is in TimesFM 2.5's, Chronos's and Chronos-2's pretraining data); the footer gives the windows, units and months of each test set. | `external_tests.csv` (`test_set_id`, `block`, `model`, `readout`, `improvement_pct`, `interval_low_pct`, `interval_high_pct`, `label`, `windows`, `units`, `months`), written by `evaluation/external_tables.py` from the score records |
| `fig18_recipe_history` | The training-recipe study behind the released recipe (test windows read before; a description). All recipes start from the same from-scratch checkpoints (three seeds). F and R then train exactly 8 more epochs: F on the unchanged training set (control), R (released) with the validation span returned; V instead re-splits with a proportional validation set (10% of the own record, 38–93 days) and stops early on it (5 to 30 epochs; the start weights count as epoch 0). a: hourly R vs F, V vs F and V vs R on the three short-record populations, their switch and non-switch layers (test windows whose season the return newly covers, and the rest), the nine populations, the six development and the four long-record populations, with Welch 95% intervals (unit-and-month bootstrap combined with the seed spread); b: R vs F by readout (energy, peak, forecast day 1, days 2–7, days 8–31); c: R vs F per population. The footer lists every contrast resolved against R and the lowest population estimate. | `recipe_history.csv` as written by `evaluation/recipe_tables.py` (`comparison`, `populations`, `layer`, `quantity`, `improvement_pct`, `interval_low_pct`, `interval_high_pct`, `label`; panel c from `by_population_pct` of the nine-population rows, point estimates only). The rows against the un-continued from-scratch stage (unit-and-month intervals only) are in the file but not drawn. A plain layout (`group`, `contrast`, `readout`, `pct`, `lo`, `hi`, `label`) is also read. |

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

Figure 4 does not colour by class: each variant of a panel gets the next of ten styles, `ABL_STYLE` in `make_figures.py`
(circle `#1A1A1A`, diamond `#E6AB02`, square `#7B5EA7`, triangle up `#D55E00`, triangle down `#56B4E9`, filled plus
`#332288`, filled cross `#882255`, star `#1B9E77`, hexagon `#CC79A7`, pentagon `#6A3D9A`). No two styles share a marker or a colour. The
smallest OKLab distance over normal vision and the three simulated colour-vision deficiencies is 0.104 between any two of
the first eight colours (the largest panel has eight variants), 0.071 among all ten, and 0.088 to ankyra-lite's red. A
panel with more than ten variants reuses markers and colours in new pairs, never the same pair twice, and the script
prints a warning.

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
- Caption notes: every figure of ankyra-lite's forecasts names its recipe (R, three seeds averaged; Figure 17: carriers
  trained for each test set with R, six seeds and the three-seed release configuration); the figures of the
  whole comparison set give its size (22 forecasters: ankyra-lite and the 21 baselines), and Figure 3, which draws a
  selection, says where the other baselines are. No figure draws ANKYRA itself or its versions, and none draws a
  forecaster outside the 22. ANKYRA is named only as the source of the anchoring layer (Figures 1 and 7, "ANKYRA 2.2.1") and of the
  release's rules and example window.
- External tests: Figure 17 draws `external_tests.csv` only; no comparison with ANKYRA itself or its versions is in
  that file. The Helsinki retrospective set is trained and sealed, not scored, and appears as an empty row.
- Axes of interval plots are widened to hold every point estimate (within fixed caps); an interval end beyond the axis
  is an arrow, and a point beyond a cap is drawn at the edge with its value.
- The figures are descriptions of a re-evaluation: the test windows had been read before. Figure 17 is the exception:
  first reads of this candidate on previously read populations and a same-source out-of-time supplementary check.
