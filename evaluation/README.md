# Evaluation code

This folder holds the scoring behind the comparison, written as a small module on top of
`ankyra_lite.anchor.metrics` (the anchoring layer's copy of `ankyra.metrics`), and the scripts that applied it to the
study's saved forecasts. Given the forecasts of several models on the same windows and the realised load, the
module produces the tables in [`results/`](../results/). It is not imported by the forecaster.

| File | Content |
|---|---|
| `scoring.py` | `score_population` (pairwise contrasts with bootstrap intervals, mean per-unit ranks and conventional metrics, on all windows and on the late windows), `rank_tests` (Friedman, Nemenyi, Holm-corrected Wilcoxon), `scaled_errors` (RMSSE, MASE), `energy_error` and `energy_ape`, `by_forecast_day`, `energy_by_forecast_day`, `block_shares` (level, daily path, within-day), `peak_readout`, `previous_window_peak`, `peak_ape` and `peak_contrasts`. |
| `windows.py` | Which origins of a unit's record are usable (optionally only those whose six pseudo-origins lie after the carrier's training cut), month-start origins, the context, target and month label of each window, and the day types the peak readout reads. |
| `baselines.py` | The three untrained reference forecasters that need only the load context: seasonal naive (day, week) and the four-week profile. |
| `run_example.py` | The whole flow on twelve artificial buildings: windows, ankyra-lite and the three baselines, scoring. No data and no downloads. |
| `compare_baselines.py`, `products.json` | The comparison on the study's saved forecasts: ankyra-lite against the 21 baselines on nine populations, written to `results/`. `products.json` (a mapping of `tools/products_for_configuration.py` for recipe Rw, six seeds and the layer on) names the release configuration of 0.2.0 (recipe Rw, six seeds, the continuation layer with its constants and digest) and, per population, the anchored six-seed product before the layer (`npz`: level, peak and the energy tables read it, readout rule A) and after it (`continued_npz`: the hourly tables), the two three-seed group products, the research runs behind them, the from-scratch stage they continued and the study's other recipe arms and 0.2.0 configurations on the same windows. Every continued product is checked, with no target read, to be the release layer applied to the product before it, bit for bit (`--check-products` runs the same check). Needs the study's local data ([below](#what-is-not-here)). |
| `products_0_1.json` | The mapping of version 0.1 (recipe R, three seeds, no layer), byte-identical to the `products.json` of 0.1.1; `--recipe r010` in `compare_baselines.py` and `external_tables.py` reproduces the tables of 0.1. `products_v020_candidate.json` is the default output file of `tools/products_for_configuration.py`; the mapping the release scores is `products.json`. |
| `products_p3s.json` | The same mapping for the from-scratch stage alone (the study's P3s recipe), selected with `--recipe p3s` in `compare_baselines.py`, `lite_reanchor.py`, `robustness.py` and `tools/export_checkpoints.py`. It is not the released recipe; it has its own product label, so its re-anchoring cache is separate, and it lists the superseded original-store Drammen and CINELDI products under `superseded`. |
| `extra_tables.py`, `lite_reanchor.py`, `cost_timing.py` | Called by `compare_baselines.py`: the result files beyond the ten tables of the main comparison (with 0.2.0, also `continuation_layer.csv` and `continuation_record.csv`), the re-anchoring of the saved carrier forecasts with the unchanged anchoring layer (the six-seed ensemble with its anchoring record, which must reproduce the stored product bit for bit before and after the continuation layer, and the parts switched off), and the cost of one forecast. They need the study's local data too. |
| `recipe_tables.py` | Called by `compare_baselines.py`: `recipe_history.csv` (the study's readings of the recipe arms of 0.1, and the 0.2.0 study's development check and test read, copied unrounded from the scoring records) and `constants.csv` (the carrier's constants read from the package, recipe Rw's loss weights and the continuation layer's constants, and the anchoring layer's rows of the first release's file). Reads no data store. |
| `external_tables.py` | `results/external_tests.csv`: the external tests ([docs/EXTERNAL_TESTS.md](../docs/EXTERNAL_TESTS.md)), read from the study's one-time scoring records (`ANKYRA_LITE_RESEARCH_ROOT`, sibling `HKAD_V3` by default) with paths scrubbed; `--version 0.2` (the default follows `products.json`) reads the 0.2.0 re-run, `--version 0.1` the frozen external test of 0.1; `--check` compares the file with a fresh build. Reads no data store. |
| `robustness.py` | Called by `compare_baselines.py`, or alone: `robustness.csv`, ankyra-lite (with the continuation layer) and the carrier alone re-forecast on the CPU (one thread) from corrupted contexts with the released weights, the first release's eight corruptions and random stream. Default: 64 Drammen test windows. Needs the study's local data. |
| `budget_tables.py` | Called by `compare_baselines.py` (alone with `--budget`): `training_budget_sensitivity.csv`, ankyra-lite against DLinear, iTransformer, LSTM and PatchTST trained with the original budget (15 epochs, patience 3) and retrained with 30 epochs and patience 8 (the first release's sensitivity run; its saved forecasts are read, nothing is trained). |
| `long_context_tables.py` | Called by `compare_baselines.py`: `long_context_baselines.csv`, ankyra-lite against zero-shot TimesFM 2.5 with an 8,736-hour and Chronos-2 with an 8,192-hour context (saved study forecasts), outside the main ranks. |

The files added on 19 October 2026 (handover arms, lead weeks, within-day trust, one whole-window weight, generic
partners, level-candidate hit rate) read the `fields` mode of `lite_reanchor.py`: the product's six-seed ensemble
re-anchored once more (before the continuation layer), keeping the anchoring layer's own fields of every window and the pseudo-origin quantities
rebuilt from its public functions (the handover pairs with `history.estimate_from_history`, the trust triples with
`analog.AnalogShapes`) and two private helpers of the copy (`_masked`, `_level_parts`), read only. Its gate: the
trajectory, level and state flags equal the stored product bit for bit, and the rebuilt quantities reproduce the call's
`lead_week_weights`, `within_trust`, `level_weights` and `daily_means_kw` exactly. `compare_baselines.py --supplement`
computes these files, the budget and long-context files, `energy_level_readout.csv` and the carrier-alone peak row for an
existing complete run and merges them into it (new files; the added rows of `ablation.csv`, `component_contributions.csv`
and `peak_readout.csv` inserted after each population's rows, every other line kept byte for byte; the record and
`SUMMARY.md` extended). In 0.2.0 these files are part of the main run.

```bash
python -m evaluation.run_example --untrained              # untrained stand-in carrier: shows the data flow only
python -m evaluation.run_example --population cambridge   # checkpoints/cambridge/rw/seed0-5.pt (the default population)
```

The example forecasts each artificial building at its last three usable month-start origins: the carrier network at
the origin and the six pseudo-origins, the anchoring layer, the seasonal-naive state rule and the continuation layer;
its energy and peak are read before the layer (readout rule A). The carrier's own forecast
at the origin is scored too, as the reference's reduced version (`family`: a pairwise contrast, left out of the
ranking). With `--untrained` (or without checkpoints) the carrier is an untrained network of the published shape
(`build_carrier()`, seed 0). By default it loads the released checkpoints of one population
(`ankyra_lite.checkpoint_paths(population)`: `checkpoints/<population>/rw/seed0-5.pt`); these belong to that population and its training cut, not to artificial
buildings, so in both cases the numbers mean nothing: the example shows the data flow and the output format. It takes
about half a minute on one CPU thread.

## Running the comparison

From the repository root, CPU only (`compare_baselines.py` sets one thread and hides the GPU):

```bash
python -B evaluation/compare_baselines.py --reanchor --workers 2          # every file of results/
python -B evaluation/compare_baselines.py --faces cineldi --out <dir> --reanchor --no-robustness --no-cost
python -B evaluation/compare_baselines.py --recipe p3s --out <dir>        # the from-scratch stage, not the release
python -B evaluation/compare_baselines.py --recipe r010 --out <dir>       # the tables of version 0.1 (products_0_1.json)
python -B evaluation/compare_baselines.py --check-products                # list the products, check the continued ones
python -B tools/products_for_configuration.py --recipe Rw --seeds 6 --continuation on --out <file>   # a products mapping
python -B evaluation/compare_baselines.py --summary-only                  # rebuild SUMMARY.md from the JSON record
python -B evaluation/robustness.py --face oslo --windows 64 --out <dir>   # robustness.csv alone
python -B evaluation/lite_reanchor.py --faces oslo --modes s0 --limit 8   # the single-seed gate alone
python -B evaluation/lite_reanchor.py --faces bdg2,oslo --modes fields   # the fields cache of the handover and combination tables
python -B evaluation/compare_baselines.py --supplement                    # the files added on 19 October 2026, merged into results/
python -B evaluation/compare_baselines.py --budget                        # training_budget_sensitivity.csv alone
python -B evaluation/external_tables.py --check                           # external_tests.csv against the score records
python -B tools/export_checkpoints.py --out <dir>                         # the 81 checkpoints and MANIFEST.json
```

- `--reanchor` runs `lite_reanchor.py` first (needed for a complete run: the ensemble supplies the anchoring record);
  `--workers` sets its processes. `ANKYRA_LITE_EVAL_CACHE` (default: the system temporary directory) keeps the
  re-anchored forecasts, keyed by the product label, so a second run reuses them.
- `--faces` restricts the populations (the scoring-code check also reads the Suzhou park); `--out` writes to another
  directory; `--no-extra` writes only the ten tables of the main comparison; `--no-cost` and `--no-robustness` skip
  `cost_per_window.csv` and `robustness.csv`; `--robustness-face` and `--robustness-windows` change the robustness
  sample; `--only-check` runs only the scoring-code check.
- Every stage that loads a data store first waits until at least `ANKYRA_LITE_MIN_FREE_GB` (default 5) GB of memory is
  available.
- `tools/export_checkpoints.py` copies the stage-2 runs' final weights byte for byte to
  `<out>/<population>/rw/seed<s>.pt` (recipe Rw, seeds 0–5) and, unless `--no-0-1`, `<out>/<population>/seed<s>.pt`
  (version 0.1, recipe R, seeds 0–2), checks each copy against the SHA-256 of its run record, loads it strictly with
  `ankyra_lite.load_carrier`, writes `<out>/MANIFEST.json` (no absolute paths) and, for a few test windows of Oslo and
  CINELDI (`--verify`, `--verify-windows`), reproduces the stored GPU carrier forecasts on the CPU (relative difference
  below 10⁻⁴, same issue times). The released files were written with `--out checkpoints`.

## Input format

One row per forecast window, the same rows in every array:

| Array | Shape | Meaning |
|---|---|---|
| `forecasts[name]` | (n, 744) | hourly forecast in kW. A model that exists only on part of the windows has NaN rows elsewhere. |
| `y` | (n, 744) | realised load in kW |
| `unit` | (n,) | unit label; the estimand weights units equally |
| `month` | (n,) | label of the month holding the window's midpoint; the bootstrap resamples units and months |
| `context` | (n, 1344) | load before the origin (scaled errors and peak readout only) |
| `context_types`, `target_types` | (n, 56), (n, 31) | day types of the context and target days, read at each day's 12th hour (`windows.day_types`; peak readout only) |

`score_population(forecasts, y, unit, month, reference="ankyra-lite", family=(...))` returns a `full` block (all
windows, the models that cover them all) and a `late` block (the windows every model covers, all models). `family`
names the reference's own reduced versions; they get a pairwise contrast but are left out of the ranking.

The peak readout of `scoring.peak_readout` is the anchoring layer's readout (daily means of a trajectory plus the
largest recent excursion of the same day type), applied to any forecaster's trajectory. `peak_contrasts` compares
ankyra-lite's peak readout with other peak estimates of the same windows: its own trajectory maximum, the previous
window's observed peak, and for each baseline the same readout on the baseline's trajectory and the baseline's
trajectory maximum.

**Random streams.** Every interval uses 2,000 replicates. As in the study's scoring, every hourly contrast draws from a
fresh generator seeded `scoring.SEED`, the energy contrasts of one population share one generator seeded
`scoring.ENERGY_SEED` (models in sorted order), and every peak contrast draws from a fresh generator seeded
`scoring.PEAK_SEED`. With these defaults the intervals reproduce the study's.

## What was checked

On two populations (BDG2 2017 and Oslo; the late windows) every table this module computes was recomputed from the
same saved forecasts both with this module and with the study's own scoring functions (as `compare_baselines.py`
calls them), and the two were compared at the rounding of the result files: the pairwise contrasts and their
intervals, the mean per-unit ranks, the conventional metrics, the energy contrasts (also with the level readout in
place of the trajectory mean), the loss and the energy error by forecast day, the error blocks, the rank tests, the
scaled errors and the peak contrasts. All 8,486 compared values (7,528 on BDG2, 958 on Oslo, where the release has no
forecast-day or block tables; counts from the output of this one-off check, which is not a file of `results/`) were
identical; the intervals reproduce because the seeds and the order of the draws are the study's. The check was run on 18 October 2026, before the released recipe and the final set of baselines were
fixed, on the from-scratch-stage products and a forecaster set that differs from the final one. It checks the scoring
code, which does not depend on the product or the forecaster set; it was not repeated on the released products. The
other seven populations were not checked this way. Two checks of every complete run cover the released products
instead: the scoring-code check against the first release's published tables
([`results/REPRODUCTION_CHECK.json`](../results/README.md#reproduction-and-records)) and the product check against the
study's own scorer of the recipe arms (`study_scorer_crosscheck`, version 0.1's run: 9 of 9 populations, largest
difference 0.0 percentage points). In 0.2.0 the continued products are checked against the release's continuation
layer bit for bit (`continuation_gate` in `results/compare_baselines.json`), and the re-anchored six-seed ensemble
reproduces every stored product before and after the layer.

The module's own tests are in [`tests/test_evaluation.py`](../tests/test_evaluation.py) (15 tests). They run on
artificial data, together with the forecaster's tests: `python -m unittest discover -s tests -t .`. The comparison
script, the checkpoint export and the result files are checked by
[`tests/test_compare_baselines.py`](../tests/test_compare_baselines.py) (42 tests; those that need the first release's
files, the study's research folder or a complete run in `results/` skip without them), the store reader by
[`tests/test_store_arrays.py`](../tests/test_store_arrays.py) (9 tests, study data), and the external-test table by
[`tests/test_external_tables.py`](../tests/test_external_tables.py) (12 tests). The whole suite has 267 tests in
version 0.2.0 ([EVALUATION.md](../docs/EVALUATION.md#scoring-code-check-and-reproduction)).

The terms used by the module (unit-equal log RMS ratio, unit-and-month bootstrap interval, late windows, mean
per-unit rank, resolved contrast) are those of the ANKYRA release; the repository's [README](../README.md) explains
how to read them.

## What is not here

- **The forecasts themselves.** Raw data and saved forecast arrays are not in the repository. The 81 carrier
  checkpoints are (`checkpoints/`, with `MANIFEST.json`: the 54 of recipe Rw and the 27 of version 0.1); the anchored
  products scored in `results/` were computed by the study's research code from the same weights on a GPU, and the
  continued products by the release's continuation layer applied to them.
- **The other baselines.** 18 of the 21 baselines (the zero-shot foundation models and their covariate-informed
  variants, the seven trained models of the first release, LSTM-T-KAN-M, Holt–Winters, MSTL, the per-unit ridge, the
  zero-shot gradient-boosting model and the last-year and previous-month profiles) are described in the repository's
  documentation ([EVALUATION.md](../docs/EVALUATION.md#baselines)); their training and inference code is not included.
- **The study's data.** `compare_baselines.py` reads the study's research folder (environment variable
  `ANKYRA_LITE_RESEARCH_ROOT`, default: the folder `HKAD_V3` next to this repository) and the first release repository
  (`ANKYRA_ORIGINAL_ROOT`, default: `Building_load_ANKYRA` next to this repository), both read-only (an audit hook
  refuses any write there), so it cannot be run from this repository alone. `scoring.py` is a clean re-implementation
  of the study's scoring, checked against it as described above.
