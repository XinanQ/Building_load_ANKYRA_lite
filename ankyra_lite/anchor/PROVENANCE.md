# Provenance of `ankyra_lite.anchor`

`ankyra_lite.anchor` is the anchoring layer of ANKYRA 2.2.1, copied without any change from the ANKYRA repository.
It takes the carrier network's month-ahead forecasts through the `foundation` argument of `forecast` (a mapping
`{k: (744,) forecast issued at o - 744k}` or a callable `(N, 1344) contexts -> (N, 744) forecasts`), exactly as the
original takes a foundation model's forecasts.

## Source

| Item | Value |
|---|---|
| Repository (local checkout) | the author's working copy of the ANKYRA repository; the tests look for it in the folder `Building_load_ANKYRA` next to this repository, or where the environment variable `ANKYRA_ORIGINAL_ROOT` points |
| Repository (remote) | `https://github.com/XinanQ/Building_load_ANKYRA` |
| Commit | `487e0d2` (`487e0d2d77098a102e8f634bb1bbce85e01e8b59`, 2026-10-13, "Documentation made exact against the code (no code, number or version changed)") |
| `git describe` | `v2.2.1-5-g487e0d2`; the `ankyra/` tree at this commit is identical to the tag `v2.2.1` (`git diff v2.2.1 487e0d2 -- ankyra` is empty) |
| Package | `ankyra`, version `2.2.1` (`ankyra.__version__`; model 2.2, whose forecasts on finite inputs are those of 2.2.0) |
| Frozen reference estimator (`history/`) | interface version `1.1.0` (`ankyra.history.__version__`) |
| Licence | MIT, Copyright (c) 2026 Xinan Qin (full text below) |
| Copied | 2026-10-18, byte for byte from the source working copy (`shutil.copyfile`, no line-ending conversion) |

At copy time `git status --porcelain -- ankyra` in the source checkout was empty, and the `git hash-object` of every
source file equalled its blob in the commit's tree (column "git blob" below), so the copied content is that of commit
`487e0d2`.

**Line endings.** In the source working copy (a Windows checkout), 9 of the 17 files have CRLF line endings:
`analog.py`, `history/_climate.py`, `history/_day.py`, `history/_eo.py`, `history/_level.py`,
`history/_signature.py`, `history/_span.py`, `history/frozen_config.json` and `history/provenance.json`. The other 8
(`__init__.py`, `blocks.py`, `core.py`, `history/__init__.py`, `history/api.py`, `metrics.py`, `readouts.py`,
`synthetic.py`) have LF line endings. The copy keeps each file's line endings as they were. The blobs stored in git at
`487e0d2` have LF line endings for all 17 (git converts CRLF to LF when it hashes or stores these text files, which is
why `git hash-object` of each working-copy file equals the committed blob). In the table below the column "git blob"
identifies the LF content, which is the same in every checkout, while "Bytes" and both SHA-256 columns are those of the
files as copied: CRLF for the 9 files above, LF for the other 8. This repository's `.gitattributes`
(`* text=auto eol=lf`) stores and checks out text files with LF. In a fresh checkout the 8 LF files therefore match
the table exactly, and each of the 9 CRLF files is one byte shorter per line and has another SHA-256; converting its
line endings to CRLF reproduces the recorded size and hash. `tests/test_anchor_equivalence.py` accepts each file as on
disk, with LF or with CRLF line endings, and requires its git blob to equal the recorded one.

## Files

SHA-256 before (source file) and after (this copy), both with each file's line endings as copied (see above). They
are equal for every file because no file was edited.

| File | Bytes | git blob at 487e0d2 | SHA-256 before (source) | SHA-256 after (this copy) |
|---|---|---|---|---|
| `__init__.py` | 1492 | `005743c162829d28a50b341ea741023c97f102f1` | `91aaae8580d1f92fb379c479670e26a1cdabb5e9b0b93ed546ca15ad5ae465a5` | `91aaae8580d1f92fb379c479670e26a1cdabb5e9b0b93ed546ca15ad5ae465a5` |
| `analog.py` | 10213 | `7bb151a1918c42727e0200d4c25bce3e798990b4` | `faae948fa54d36cc8c355104daf40b500656072e301e6ba64376cd84faa3075d` | `faae948fa54d36cc8c355104daf40b500656072e301e6ba64376cd84faa3075d` |
| `blocks.py` | 2249 | `a5f916ef50a9e1e283438aa131633652c37e78cb` | `cff12b09ab5fcf0076ac16dfd359906b1e6cfbe6b59db28b88272d66e5c58aa7` | `cff12b09ab5fcf0076ac16dfd359906b1e6cfbe6b59db28b88272d66e5c58aa7` |
| `core.py` | 35391 | `3f422551353319a90986cf2724aa5e55228919cd` | `9f4141c3e24c47e7e30a3a03a9f791c937ae670dc65d655a86b3e0230d075be2` | `9f4141c3e24c47e7e30a3a03a9f791c937ae670dc65d655a86b3e0230d075be2` |
| `history/__init__.py` | 765 | `9b9ff48b9a6837ef0ac44f4680462090cf968db0` | `30e47e37e49aada1447a2237051716c3bffe1e9e236c47d12a0a181cb3e133c4` | `30e47e37e49aada1447a2237051716c3bffe1e9e236c47d12a0a181cb3e133c4` |
| `history/_climate.py` | 1312 | `7e1a6b99b7f6de9f3018e5cf4e3bc2c69fe4966e` | `bf3c2404da1e37d71a9b347caf946e401c6f90a4a4ec11f45e4ee47dfb701f89` | `bf3c2404da1e37d71a9b347caf946e401c6f90a4a4ec11f45e4ee47dfb701f89` |
| `history/_day.py` | 4078 | `d27231a85501003d785e4b4bfde0f9b340db741b` | `c279268e5315641ee7495485036eebddaaaa28fe2f9cdbd6ffe50171c1b38c98` | `c279268e5315641ee7495485036eebddaaaa28fe2f9cdbd6ffe50171c1b38c98` |
| `history/_eo.py` | 6208 | `c2bf23c10a478219394fdc2c47c555ff8b3eb8e8` | `4c7907b523da99dacbcbb83a18b08ee877addd1ad92a901d741ac109e8949100` | `4c7907b523da99dacbcbb83a18b08ee877addd1ad92a901d741ac109e8949100` |
| `history/_level.py` | 11468 | `2bee5dd5ae085c19c638bb223c8c60d2234dd17f` | `83092b5c7fee81ab9c79a46d07262a99e9dd72ee0f0fbeda71184a1eacee76da` | `83092b5c7fee81ab9c79a46d07262a99e9dd72ee0f0fbeda71184a1eacee76da` |
| `history/_signature.py` | 4212 | `3a4f55e3b9f2e2bbe90ebdbec29502aef3a54641` | `4263bf2d86c3f6839ddbf894fef2ed4def277a62b7d973c579cf9ae02f8c8658` | `4263bf2d86c3f6839ddbf894fef2ed4def277a62b7d973c579cf9ae02f8c8658` |
| `history/_span.py` | 4235 | `1e2c6b2690c0b7b2c3c45c356a5f1470e43cee81` | `b649669d9c595fc77fb4005d40c247314e48c8bf012a2eaada632b7188df4a52` | `b649669d9c595fc77fb4005d40c247314e48c8bf012a2eaada632b7188df4a52` |
| `history/api.py` | 20661 | `b1ee99613a4cbf0026c70c1a39420d669f20133d` | `f65df8fe2b57db6b3727d6cf687d788c2439b93608fdd7e7bb8e8123d0383c57` | `f65df8fe2b57db6b3727d6cf687d788c2439b93608fdd7e7bb8e8123d0383c57` |
| `history/frozen_config.json` | 3109 | `365107b5f49fd0c856a40a8a041743e4f9f96cfd` | `43bb3c054c9ba6bda436c85854b17dddee7d933e5194918151127696ff878caa` | `43bb3c054c9ba6bda436c85854b17dddee7d933e5194918151127696ff878caa` |
| `history/provenance.json` | 11342 | `56395b18179e34f6b62389e3ca74b0457d447df3` | `09056a155a1640f98428f20134bf27d8c84e936f8629ee1e278bf1d85d87b16b` | `09056a155a1640f98428f20134bf27d8c84e936f8629ee1e278bf1d85d87b16b` |
| `metrics.py` | 5264 | `6cfd05d7413ff973d5abaad89cc700b1215aeb4c` | `8d5470290b03a05331abcd74b0e8cb970e21939f6016e74467fb78459f629382` | `8d5470290b03a05331abcd74b0e8cb970e21939f6016e74467fb78459f629382` |
| `readouts.py` | 10229 | `99252a9f297df288b8e05b7c4147e4d5f22d5a51` | `48c2ea1474ee0d48ad011b032682378336b1d21bc1e9007eb077d80e0729244a` | `48c2ea1474ee0d48ad011b032682378336b1d21bc1e9007eb077d80e0729244a` |
| `synthetic.py` | 1551 | `a9c9d4fab638dcfc7266a3c8966d6c00651ed490` | `5afe56dd0c85282d5faae914cbce86ea0f03458f85c516e7d37672945529e451` | `5afe56dd0c85282d5faae914cbce86ea0f03458f85c516e7d37672945529e451` |

`tests/test_anchor_equivalence.py` (class `ProvenanceTests`) checks this table against the files on disk.

Not copied:

- `timesfm_adapter.py` (git blob `f8d4a9229d0145e3ffac64f6c5873b25932c3329`, SHA-256
  `81f9a6c7f91b2b5418fd3fa4e8e5e254004c9eef0cd2daf8965e143d553b66eb`): the TimesFM 2.5 loader. ankyra-lite has no
  foundation model. No copied module imports it (in the original only `tests/test_core.py` does), so no import had to
  be removed.
- `__pycache__/` directories (compiled bytecode).

## Edits

**None.** No line of any copied file was changed; every SHA-256 above is the same before and after.

Self-reference audit: what was checked, and why no change was needed.

1. **Imports.** Every intra-package import is relative (`from .history import ...`, `from . import analog as _analog`,
   `from . import _eo` in `history/`, and the deferred `from .readouts import within_day_default` in `core.py`
   line 495), so all of them resolve inside `ankyra_lite.anchor`. No module contains `import ankyra`,
   `importlib.import_module("ankyra...")`, a lookup of the package by name, or a warnings filter keyed on the module name.
2. **Package data.** `history/api.py` lines 38 and 349 read `frozen_config.json` through
   `importlib.resources.files(__package__)`. Here `__package__` is `ankyra_lite.anchor.history`, so the copy's own
   file is read. Its bytes are unchanged, so the `config_sha256` diagnostic still reports
   `43bb3c054c9ba6bda436c85854b17dddee7d933e5194918151127696ff878caa`.
3. **Versions.** No version assert exists. `__init__.py` line 15 (`__version__ = "2.2.1"`) is kept: it is the version
   of the copied layer. `history/__init__.py` line 7 (`__version__ = '1.1.0'`) is the estimator interface version.
4. **Docstrings and comments that name the original package** are kept as they are. Read `ankyra.X` as
   `ankyra_lite.anchor.X`:
   - `__init__.py` line 3 (``ankyra.analog``);
   - `core.py` lines 3 (``ankyra.blocks``), 16 (``ankyra.analog``), 46 and 145 (``ankyra.history``), 334
     (``ankyra.History``), 351 (``ankyra.timesfm_adapter``, which is not shipped here; pass the carrier's forecasts
     instead), 375 (``ankyra.InputError``) and 379 (``ankyra.SignatureDegeneracy``);
   - `readouts.py` line 138 (``ankyra.forecast``);
   - `history/__init__.py` line 7 (comment naming ``ankyra.__version__``).
5. **Paths to files of the original repository** are kept as they are; they point into that repository, not into
   ankyra-lite: `core.py` lines 186 (`results/anchoring_gain_deciles.csv`) and 347 (`docs/DATA.md`), `metrics.py`
   line 7 (`theory/PROOFS.md`), and `history/api.py` line 364 (`results/REPRODUCTION_CHECK.json`, inside the
   `scientific_status` diagnostic). That last one is a string the estimator returns at run time. Editing it would
   change the estimator's output, and the bitwise equivalence test compares that output.
6. **`history/provenance.json`** records how the underscore modules of `history/` were extracted from the study code
   (source file names, line ranges and hashes). It is copied unchanged and the code never reads it.

## Verification

- `tests/test_anchor_equivalence.py` imports the original package from the source checkout (path overridable with
  the environment variable `ANKYRA_ORIGINAL_ROOT`; the tests that need it are skipped, with the reason, when it is
  unavailable). It checks:
  - the copy against the checkout, file by file (the same git blob; a difference in line endings only is reported, any
    other difference fails);
  - the table above against the files on disk;
  - bitwise equality of every `AnkyraForecast` field and of the energy readout on 19 cases. The cases cover five
    categories; records of 10,248 to 20,000 hours; random foundation dictionaries at the origin and the
    pseudo-origins, including partial key sets; a callable foundation; the history-only configuration; short gaps;
    an observed mask with placeholder values; the off-state and micro-load returns; and every version switch;
  - bitwise equality of the peak, energy and interval readouts, the reference estimator with its diagnostics, the
    block operators, the analog shapes and within-day trust, and the evaluation estimands;
  - identical exception types and messages on twelve kinds of invalid input.
- `tests/anchor/` holds seven of the original unit tests: `test_core`, `test_no_future_information`,
  `test_gap_tolerance`, `test_analog`, `test_blocks_and_readouts`, `test_history_reference` and `test_micro_load`.
  The only change is the package name: imports point to `ankyra_lite.anchor`, and the tests call it `anchor`
  (formerly `ankyra`). `test_core` also drops `TimesfmAdapterTests` and the imports that only that class used
  (`sys`, `types`, `unittest.mock`, `timesfm_adapter`). The original `tests/test_evaluation.py` is not ported because
  it imports the original repository's `evaluation` package. SHA-256 of the source test files at `487e0d2`:

  | Source test file | SHA-256 |
  |---|---|
  | `tests/test_core.py` | `1415dc586e566c6f0e5c3a3d189eed7950e7b9657e13b5b4f50c650b122d5332` |
  | `tests/test_no_future_information.py` | `215595080c5b6534ce482add88de883cb7eaeffbb9493bd1fef999a4fa57cc69` |
  | `tests/test_gap_tolerance.py` | `131d0f233c2b0033edce1f3585fae7ef8cf044c183d101907040a3fd953ab812` |
  | `tests/test_analog.py` | `e4f4e7d55d3a920dd069353b50ef07d09ce8aadbce3cadc76ac5d3d77a5dfeec` |
  | `tests/test_blocks_and_readouts.py` | `d3ca8883237f6c3477d53a84e9e5ecaa4f8edeeb607c473216084887d2f65d15` |
  | `tests/test_history_reference.py` | `65988aca82858c026b7152bd47841d02f0cbe895ac9fd67dc7094449906a1683` |
  | `tests/test_micro_load.py` | `746a02e3afa25876d53499c41d13bbefb265a034d7cfdae584f39197faa68adb` |

- Run from the repository root: `python -m unittest discover -s tests -t .` (pytest also works if installed).

## Licence (MIT, from the source repository's `LICENSE`)

```
MIT License

Copyright (c) 2026 Xinan Qin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
