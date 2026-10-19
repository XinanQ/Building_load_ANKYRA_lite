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
| `compare_baselines.py`, `products.json` | The comparison on the study's saved forecasts: ankyra-lite against the 21 baselines on nine populations, written to `results/`. `products.json` names, per population, the anchored three-seed product of the released recipe and the research runs behind it (the span-return runs with their weights file, the from-scratch stage they continued, each seed's anchored product, and the study's other recipe arms on the same windows). Needs the study's local data ([below](#what-is-not-here)). |
| `products_p3s.json` | The same mapping for the from-scratch stage alone (the study's P3s recipe), selected with `--recipe p3s` in `compare_baselines.py`, `lite_reanchor.py`, `robustness.py` and `tools/export_checkpoints.py`. It is not the released recipe; it has its own product label, so its re-anchoring cache is separate, and it lists the superseded original-store Drammen and CINELDI products under `superseded`. |
| `extra_tables.py`, `lite_reanchor.py`, `cost_timing.py` | Called by `compare_baselines.py`: the result files beyond the ten tables of the main comparison, the re-anchoring of the saved carrier forecasts with the unchanged anchoring layer (the three-seed ensemble with its anchoring record, which must reproduce the stored product bit for bit, the parts switched off, and an 8-window gate that re-anchors seed 0 and must reproduce its stored product), and the cost of one forecast. They need the study's local data too. |
| `recipe_tables.py` | Called by `compare_baselines.py`: `recipe_history.csv` (the study's readings of the recipe arms, copied unrounded from its scoring record) and `constants.csv` (the carrier's constants read from the package, and the anchoring layer's rows of the first release's file). Reads no data store. |
| `external_tables.py` | `results/external_tests.csv`: the external tests of the released candidate ([docs/EXTERNAL_TESTS.md](../docs/EXTERNAL_TESTS.md)), read from the study's one-time scoring records (`ANKYRA_LITE_RESEARCH_ROOT`, sibling `HKAD_V3` by default) with paths scrubbed; `--check` compares the file with a fresh build. Reads no data store. |
| `robustness.py` | Called by `compare_baselines.py`, or alone: `robustness.csv`, ankyra-lite and the carrier alone re-forecast on the CPU (one thread) from corrupted contexts with the released weights, the first release's eight corruptions and random stream. Default: 64 Drammen test windows. Needs the study's local data. |

```bash
python -m evaluation.run_example --untrained              # untrained stand-in carrier: shows the data flow only
python -m evaluation.run_example --population cambridge   # checkpoints/cambridge/seed*.pt (the default population)
```

The example forecasts each artificial building at its last three usable month-start origins: the carrier network at
the origin and the six pseudo-origins, the anchoring layer, the seasonal-naive state rule. The carrier's own forecast
at the origin is scored too, as the reference's reduced version (`family`: a pairwise contrast, left out of the
ranking). With `--untrained` (or without checkpoints) the carrier is an untrained network of the published shape
(`build_carrier()`, seed 0). By default it loads the released checkpoints of one population
(`checkpoints/<population>/seed*.pt`); these belong to that population and its training cut, not to artificial
buildings, so in both cases the numbers mean nothing: the example shows the data flow and the output format. It takes
about half a minute on one CPU thread.

## Running the comparison

From the repository root, CPU only (`compare_baselines.py` sets one thread and hides the GPU):

```bash
python -B evaluation/compare_baselines.py --reanchor --workers 2          # every file of results/
python -B evaluation/compare_baselines.py --faces cineldi --out <dir> --reanchor --no-robustness --no-cost
python -B evaluation/compare_baselines.py --recipe p3s --out <dir>        # the from-scratch stage, not the release
python -B evaluation/compare_baselines.py --check-products                # list the products and whether their files exist
python -B evaluation/compare_baselines.py --summary-only                  # rebuild SUMMARY.md from the JSON record
python -B evaluation/robustness.py --face oslo --windows 64 --out <dir>   # robustness.csv alone
python -B evaluation/lite_reanchor.py --faces oslo --modes s0 --limit 8   # the single-seed gate alone
python -B tools/export_checkpoints.py --out <dir>                         # the 27 checkpoints and MANIFEST.json
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
- `tools/export_checkpoints.py` copies the span-return runs' final weights byte for byte to
  `<out>/<population>/seed<s>.pt`, checks each copy against the SHA-256 of its run record, loads it strictly with
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
study's own scorer of the recipe arms (`study_scorer_crosscheck` in `results/compare_baselines.json`: 9 of 9
populations, largest difference 0.0 percentage points).

The module's own tests are in [`tests/test_evaluation.py`](../tests/test_evaluation.py) (15 tests). They run on
artificial data, together with the forecaster's tests: `python -m unittest discover -s tests -t .`. The comparison
script, the checkpoint export and the result files are checked by
[`tests/test_compare_baselines.py`](../tests/test_compare_baselines.py) (33 tests; those that need the first release's
files, the study's research folder or a complete run in `results/` skip without them), and the external-test table by
[`tests/test_external_tables.py`](../tests/test_external_tables.py) (12 tests). On 19 October 2026 the whole suite
ran 176 tests on the author's machine: 175 passed, 1 skipped
([EVALUATION.md](../docs/EVALUATION.md#scoring-code-check-and-reproduction)).

The terms used by the module (unit-equal log RMS ratio, unit-and-month bootstrap interval, late windows, mean
per-unit rank, resolved contrast) are those of the ANKYRA release; the repository's [README](../README.md) explains
how to read them.

## What is not here

- **The forecasts themselves.** Raw data and saved forecast arrays are not in the repository. The 27 carrier
  checkpoints of the released recipe are (`checkpoints/`, with `MANIFEST.json`); the anchored products scored in
  `results/` were computed by the study's research code from the same weights on a GPU.
- **The other baselines.** 18 of the 21 baselines (the zero-shot foundation models and their covariate-informed
  variants, the seven trained models of the first release, LSTM-T-KAN-M, Holt–Winters, MSTL, the per-unit ridge, the
  zero-shot gradient-boosting model and the last-year and previous-month profiles) are described in the repository's
  documentation ([EVALUATION.md](../docs/EVALUATION.md#baselines)); their training and inference code is not included.
- **The study's data.** `compare_baselines.py` reads the study's research folder (environment variable
  `ANKYRA_LITE_RESEARCH_ROOT`, default: the folder `HKAD_V3` next to this repository) and the first release repository
  (`ANKYRA_ORIGINAL_ROOT`, default: `Building_load_ANKYRA` next to this repository), both read-only (an audit hook
  refuses any write there), so it cannot be run from this repository alone. `scoring.py` is a clean re-implementation
  of the study's scoring, checked against it as described above.
