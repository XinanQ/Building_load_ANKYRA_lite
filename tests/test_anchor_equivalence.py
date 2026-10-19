"""Bitwise equivalence of the copied anchoring layer (ankyra_lite.anchor) and the original ANKYRA 2.2.1 package.

``ankyra_lite.anchor`` is a byte-for-byte copy of the ``ankyra`` package of the ANKYRA repository at commit 487e0d2
(version 2.2.1) without ``timesfm_adapter.py``; ``ankyra_lite/anchor/PROVENANCE.md`` lists the files and their hashes.
These tests import the original package from its repository checkout and check that

* the copied files are byte-identical to the originals and match the hashes recorded in PROVENANCE.md;
* for a set of artificial histories and random foundation forecasts {k: (744,) array} at the origin (k = 0) and the
  pseudo-origins o - 744k, every field of the two returned forecasts (trajectory, level, daily means, within-day
  blocks, level and lead-week weights, within-day trust, flags, anchoring record) and the energy readout agree bit for
  bit, with every version switch and with the callable and history-only forms of the ``foundation`` argument;
* the peak, energy and interval readouts, the reference estimator, the block operators, the within-day trust and the
  evaluation estimands agree bit for bit;
* both packages reject the same invalid inputs with the same exception type and message.

The original checkout is read from the environment variable ``ANKYRA_ORIGINAL_ROOT`` when it is set, else from the
folder ``Building_load_ANKYRA`` next to this repository.  When it is not available, every test that needs it is skipped
with a message saying why; ``ProvenanceTests`` need no original and always run.

Line endings.  Git may check text files out with LF or CRLF line endings (``.gitattributes``, ``core.autocrlf``, an
editor), and the author's working copy of the original holds some files with CRLF.  A file's content is therefore
identified by its git blob, i.e. the SHA-1 of its LF-normalised bytes, which is the same in every checkout.  The
SHA-256 columns of PROVENANCE.md were taken on the author's working copy; a checkout whose files differ from it only in
line endings matches them after converting LF to CRLF.  Where the two packages' raw bytes differ only in line endings,
the tests say so; any other difference fails.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import os
import re
import struct
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                                   # direct runs (python tests/test_anchor_equivalence.py)
    sys.path.insert(0, str(REPO))

from ankyra_lite import anchor as lite                          # noqa: E402
from ankyra_lite.anchor import analog as lite_analog, blocks as lite_blocks, core as lite_core  # noqa: E402
from ankyra_lite.anchor import metrics as lite_metrics, readouts as lite_readouts, synthetic as lite_synthetic  # noqa: E402
from ankyra_lite.anchor.history import api as lite_api          # noqa: E402

DEFAULT_ORIGINAL_ROOT = REPO.parent / "Building_load_ANKYRA"     # sibling checkout of the ANKYRA repository
ORIGINAL_ROOT = Path(os.environ.get("ANKYRA_ORIGINAL_ROOT") or DEFAULT_ORIGINAL_ROOT)
EXPECTED_VERSION = "2.2.1"
ANCHOR_DIR = REPO / "ankyra_lite" / "anchor"
NOT_COPIED = {"timesfm_adapter.py"}                             # the lite package has no foundation model
NOT_FROM_SOURCE = {"PROVENANCE.md"}                             # written for the copy
SUBMODULES = ("analog", "blocks", "core", "metrics", "readouts", "synthetic", "history", "history.api")
HORIZON, CONTEXT, STEP = 744, 1344, 744


# ----------------------------------------------------------------------------- the original package
def _load_original():
    """Import the original ``ankyra`` from ORIGINAL_ROOT.  Returns (package, "") or (None, reason it is unavailable)."""
    pkg_dir = ORIGINAL_ROOT / "ankyra"
    if not (pkg_dir / "__init__.py").is_file():
        return None, (f"original ANKYRA package not found at {pkg_dir}; set ANKYRA_ORIGINAL_ROOT to a checkout of the "
                      f"ANKYRA repository at commit 487e0d2 (version {EXPECTED_VERSION})")
    if "ankyra" not in sys.modules:
        sys.path.insert(0, str(ORIGINAL_ROOT))
        try:
            importlib.import_module("ankyra")
        except Exception as exc:                                # e.g. a missing dependency of the original
            return None, f"original ANKYRA package at {pkg_dir} could not be imported: {exc!r}"
        finally:
            # The checkout also holds top-level packages named ``tests`` and ``evaluation``; removing the path again keeps
            # them from shadowing this repository's packages.  Submodules are found through the package's own __path__.
            while str(ORIGINAL_ROOT) in sys.path:
                sys.path.remove(str(ORIGINAL_ROOT))
    pkg = sys.modules["ankyra"]
    if Path(pkg.__file__).resolve().parent != pkg_dir.resolve():
        return None, f"a different 'ankyra' is already imported ({pkg.__file__}), not the checkout at {pkg_dir}"
    if getattr(pkg, "__version__", None) != EXPECTED_VERSION:
        return None, f"original ANKYRA package at {pkg_dir} is version {getattr(pkg, '__version__', None)!r}, expected {EXPECTED_VERSION}"
    try:
        for name in SUBMODULES:
            importlib.import_module("ankyra." + name)
    except Exception as exc:
        return None, f"a submodule of the original ANKYRA package could not be imported: {exc!r}"
    return pkg, ""


_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True                                  # the source checkout is read-only: no __pycache__ there
try:
    ORIGINAL, SKIP_REASON = _load_original()
finally:
    sys.dont_write_bytecode = _write_bytecode


def _orig(name):
    """Submodule ``name`` of the original package (e.g. "core", "history.api")."""
    return sys.modules["ankyra." + name]


# ----------------------------------------------------------------------------- bitwise comparison
def _float_bits(x: float) -> bytes:
    return struct.pack("<d", x)


def assert_bitwise_equal(test: unittest.TestCase, a, b, where: str = "value") -> None:
    """Fail unless ``a`` and ``b`` have the same structure, the same types and the same bits (NaN included)."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        test.assertTrue(dataclasses.is_dataclass(b) and not isinstance(b, type), f"{where}: not both dataclass instances")
        test.assertEqual(type(a).__name__, type(b).__name__, f"{where}: class")
        names = [f.name for f in dataclasses.fields(a)]
        test.assertEqual(names, [f.name for f in dataclasses.fields(b)], f"{where}: fields")
        for name in names:
            assert_bitwise_equal(test, getattr(a, name), getattr(b, name), f"{where}.{name}")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        test.assertIsInstance(a, np.ndarray, where); test.assertIsInstance(b, np.ndarray, where)
        test.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), f"{where}: dtype and shape")
        if a.tobytes() != b.tobytes():
            with np.errstate(all="ignore"):
                d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) if a.dtype.kind in "fiub" else None
            test.fail(f"{where}: arrays differ in at least one bit" + ("" if d is None else f" (max |diff| {np.nanmax(d)!r})"))
        return
    test.assertIs(type(a), type(b), f"{where}: type {type(a).__name__} vs {type(b).__name__}")
    if isinstance(a, dict):
        test.assertEqual(list(a), list(b), f"{where}: keys (in order)")
        for k in a:
            assert_bitwise_equal(test, a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), f"{where}: length")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_bitwise_equal(test, x, y, f"{where}[{i}]")
    elif isinstance(a, float):
        test.assertEqual(_float_bits(a), _float_bits(b), f"{where}: {a!r} vs {b!r}")
    elif isinstance(a, np.generic):
        test.assertEqual(a.tobytes(), b.tobytes(), f"{where}: {a!r} vs {b!r}")
    elif a is None or isinstance(a, (bool, int, str, bytes)):
        test.assertEqual(a, b, where)
    else:
        test.fail(f"{where}: no bitwise comparison defined for type {type(a).__name__}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lf(data: bytes) -> bytes:
    """The bytes with CRLF line endings converted to LF (git's normalisation of a text file)."""
    return data.replace(b"\r\n", b"\n")


def _git_blob(data: bytes) -> str:
    """The git blob id of a text file's content: SHA-1 of ``blob <size>\\0`` + the LF-normalised bytes."""
    lf = _lf(data)
    return hashlib.sha1(b"blob %d\0" % len(lf) + lf).hexdigest()


def _same_up_to_line_endings(a: bytes, b: bytes) -> bool:
    return a == b or _lf(a) == _lf(b)


def _frozen_config_bytes_equal() -> bool:
    """Whether the two packages' frozen_config.json have the same raw bytes (their SHA-256 is part of the estimator's
    diagnostics, so a checkout with other line endings reports another value for the same content)."""
    rel = Path("history") / "frozen_config.json"
    return (ANCHOR_DIR / rel).read_bytes() == (ORIGINAL_ROOT / "ankyra" / rel).read_bytes()


def _package_files(root: Path, skip=()):
    """Relative paths (posix) of the files of a package tree, without __pycache__."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.relative_to(root).parts and p.relative_to(root).as_posix() not in skip}


# ----------------------------------------------------------------------------- inputs
def _synthetic_arrays(hours, **kw):
    """Load, temperature, day types and start of the original package's artificial building (own copies)."""
    h = _orig("synthetic").synthetic_history(hours, **kw)
    return (np.array(h.load_kw, dtype=np.float64), np.array(h.temperature_c, dtype=np.float64),
            np.array(h.day_types), h.start_timestamp)


def _histories(load, temp, types, start, observed=None):
    """The same record as an original ``History`` and as a lite ``History``, each with its own arrays."""
    def make(cls):
        return cls(load_kw=load.copy(), temperature_c=temp.copy(), day_types=types.copy(), start_timestamp=start,
                   observed=None if observed is None else observed.copy())
    return make(ORIGINAL.History), make(lite.History)


def _random_foundation(load, rng, kind, keys=None):
    """Random foundation forecasts {k: (744,)} for k in the pseudo-origin set of ``load`` (the contexts the forecaster reads).

    kind "perturbed": the context's last week repeated, scaled by U(0.8, 1.2), plus noise of 0.2 x the context's std;
    kind "white":     positive gamma noise around the context's mean;
    kind "signed":    normal noise with three times the context's std around a tenth of its mean (the projection binds)."""
    ctx = _orig("core").pseudo_origin_contexts(load)
    out = {}
    for k in sorted(ctx):
        if keys is not None and k not in keys:
            continue
        c = ctx[k]; mu, sd = float(c.mean()), float(c.std())
        if kind == "perturbed":
            v = np.tile(c[-168:], 5)[:HORIZON] * rng.uniform(0.8, 1.2) + rng.normal(0.0, 0.2 * sd, HORIZON)
        elif kind == "white":
            v = rng.gamma(2.0, mu / 2.0, HORIZON)
        elif kind == "signed":
            v = rng.normal(0.1 * mu, 3.0 * sd, HORIZON)
        else:
            raise ValueError(kind)
        out[k] = v
    return out


def _callable_foundation(contexts):
    """A deterministic stand-in forecaster (N, 1344) -> (N, 744)."""
    c = np.asarray(contexts, dtype=np.float64)
    t = np.arange(HORIZON)
    return (np.tile(c[:, -168:], (1, 5))[:, :HORIZON] * (1.0 + 0.05 * np.sin(t / 37.0))
            + 0.1 * c[:, -HORIZON:].std(axis=1, keepdims=True) * np.cos(t / 11.0))


def _gap_record(hours=16000):
    """Short gaps in three pseudo-origin contexts and a 92%-observed pseudo target (the 2.2 short-gap rule applies)."""
    load, temp, types, start = _synthetic_arrays(hours); o = len(load)
    for k in (3, 4, 5):
        q = o - k * STEP; load[q - 500:q - 497] = np.nan
    q = o - 6 * STEP; load[q + 100:q + 160] = np.nan
    return load, temp, types, start, None


def _masked_record(hours=16800, placeholder=500.0):
    """Hours flagged unobserved carry a finite placeholder value (the forecaster must mask them)."""
    load, temp, types, start = _synthetic_arrays(hours); o = len(load)
    obs = np.ones(o, bool)
    obs[o - 3 * STEP - 200:o - 3 * STEP - 100] = False
    obs[o - 365 * 24 + 5 * 24:o - 365 * 24 + 7 * 24] = False
    obs[3000:3400] = False
    load[~obs] = placeholder
    return load, temp, types, start, obs


def _micro_record(hours=16000):
    load, temp, types, start = _synthetic_arrays(hours); load[-CONTEXT:] = 2e-4
    return load, temp, types, start, None


def _off_record(hours=16000):
    load, temp, types, start = _synthetic_arrays(hours, off_last_hours=200)
    return load, temp, types, start, None


def _plain(hours):
    return lambda: (*_synthetic_arrays(hours), None)


# name, record builder, forecast keyword arguments, foundation ("perturbed" / "white" / "signed" / "callable" / None), keys
CASES = [
    ("office_16000_perturbed", _plain(16000), dict(group="Office", temp_sigma_std=0.25), "perturbed", None),
    ("office_16000_white", _plain(16000), dict(group="Office", temp_sigma_std=0.25), "white", None),
    ("office_16800_eu_signed", _plain(16800), dict(group="Office", temp_sigma_std=0.25, dst_region="EU"), "signed", None),
    ("residential_20000_us", _plain(20000), dict(group="Residential", temp_sigma_std=0.4, dst_region="US"), "perturbed", None),
    ("industrial_12000_eu", _plain(12000), dict(group="Industrial", temp_sigma_std=0.15, dst_region="EU"), "white", None),
    ("commercial_10248", _plain(10248), dict(group="Commercial", temp_sigma_std=0.3), "perturbed", None),
    ("public_gaps", _gap_record, dict(group="Public", temp_sigma_std=0.25, dst_region="EU"), "perturbed", None),
    ("office_observed_mask", _masked_record, dict(group="Office", temp_sigma_std=0.25, dst_region="EU"), "perturbed", None),
    ("office_partial_keys", _plain(16000), dict(group="Office", temp_sigma_std=0.25), "perturbed", {0, 2, 3, 5}),
    ("off_state", _off_record, dict(group="Office", temp_sigma_std=0.25), "perturbed", None),
    ("micro_load", _micro_record, dict(group="Office", temp_sigma_std=0.25), "perturbed", None),
    ("micro_load_rule_off", _micro_record, dict(group="Office", temp_sigma_std=0.25, micro_load_rule=False), "perturbed", None),
    ("switch_within_anchor_off", _plain(16800), dict(group="Office", temp_sigma_std=0.25, dst_region="EU", within_anchor=False), "perturbed", None),
    ("switch_per_block_trust", _plain(16800), dict(group="Office", temp_sigma_std=0.25, dst_region="EU", single_trust=False), "perturbed", None),
    ("switch_2_1_on_gaps", _gap_record, dict(group="Public", temp_sigma_std=0.25, dst_region="EU", gap_tolerance=False), "perturbed", None),
    ("switch_2_0_1", _plain(16800), dict(group="Office", temp_sigma_std=0.25, dst_region="EU", gap_tolerance=False, single_trust=False), "signed", None),
    ("switch_1_x", _plain(16800), dict(group="Office", temp_sigma_std=0.25, gap_tolerance=False, within_anchor=False, micro_load_rule=False), "perturbed", None),
    ("callable_foundation", _plain(16800), dict(group="Office", temp_sigma_std=0.25, dst_region="EU"), "callable", None),
    ("history_only", _plain(16000), dict(group="Office", temp_sigma_std=0.25), None, None),
]


_PAIRS: dict = {}


def _run_pair(case_index):
    """Forecasts of the original and the copied package on one case: (original, lite, inputs); computed once per case."""
    if case_index not in _PAIRS:
        _PAIRS[case_index] = _compute_pair(case_index)
    return _PAIRS[case_index]


def _compute_pair(case_index):
    name, build, kw, kind, keys = CASES[case_index]
    load, temp, types, start, obs = build()
    h_orig, h_lite = _histories(load, temp, types, start, obs)
    rng = np.random.default_rng(1000 + case_index)
    seen = {"orig": [], "lite": []}
    if kind is None:
        fdn_orig = fdn_lite = None
    elif kind == "callable":
        def spy(tag):
            def run(contexts):
                seen[tag].append(np.array(contexts, dtype=np.float64))
                return _callable_foundation(contexts)
            return run
        fdn_orig, fdn_lite = spy("orig"), spy("lite")
    else:
        base = load if obs is None else np.where(obs, load, np.nan)
        T = _random_foundation(base, rng, kind, keys)
        fdn_orig, fdn_lite = {k: v.copy() for k, v in T.items()}, {k: v.copy() for k, v in T.items()}
    f_orig = ORIGINAL.forecast(h_orig, foundation=fdn_orig, **kw)
    f_lite = lite.forecast(h_lite, foundation=fdn_lite, **kw)
    return f_orig, f_lite, dict(load=load, temp=temp, types=types, start=start, obs=obs, kw=kw, seen=seen)


class _NeedsOriginal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ORIGINAL is None:
            raise unittest.SkipTest(SKIP_REASON)


# ----------------------------------------------------------------------------- tests
class CopyIntegrityTests(_NeedsOriginal):
    def test_copied_files_are_identical_to_the_original(self):
        """Every file has the original's content (its git blob); the raw bytes are compared too and a difference in
        line endings only is reported, not failed (see the module docstring)."""
        src = ORIGINAL_ROOT / "ankyra"
        expected = _package_files(src, skip=NOT_COPIED)
        copied = _package_files(ANCHOR_DIR, skip=NOT_FROM_SOURCE)
        self.assertEqual(copied, expected, "the copy holds exactly the original's files, minus timesfm_adapter.py")
        line_endings_only = []
        for rel in sorted(copied):
            a, b = (ANCHOR_DIR / rel).read_bytes(), (src / rel).read_bytes()
            self.assertTrue(_same_up_to_line_endings(a, b), f"{rel}: content differs from the original")
            self.assertEqual(_git_blob(a), _git_blob(b), rel)
            if a != b:
                line_endings_only.append(rel)
        if line_endings_only:
            print(f"\n    {len(copied) - len(line_endings_only)} of {len(copied)} files byte-identical; differing in line endings "
                  f"only (checkout setting): {', '.join(line_endings_only)}", file=sys.stderr)

    def test_two_separate_packages_of_the_same_version(self):
        self.assertEqual(lite.__version__, ORIGINAL.__version__)
        self.assertEqual(lite.__version__, EXPECTED_VERSION)
        for name in SUBMODULES:
            m_lite = sys.modules["ankyra_lite.anchor." + name]; m_orig = _orig(name)
            self.assertIsNot(m_lite, m_orig, name)
            self.assertTrue(Path(m_lite.__file__).resolve().is_relative_to(ANCHOR_DIR.resolve()), m_lite.__file__)
            self.assertTrue(Path(m_orig.__file__).resolve().is_relative_to((ORIGINAL_ROOT / "ankyra").resolve()), m_orig.__file__)
        self.assertIsNot(lite.forecast, ORIGINAL.forecast)

    def test_frozen_configuration_is_the_same(self):
        assert_bitwise_equal(self, lite_api.frozen_config(), _orig("history.api").frozen_config(), "frozen_config")


class ProvenanceTests(unittest.TestCase):
    """PROVENANCE.md records every copied file; the copy still has the recorded content.  Needs no original."""

    ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{40})`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$")

    def test_recorded_hashes_match_the_copy(self):
        """Each file's git blob equals the blob recorded for commit 487e0d2, in any checkout; its size and SHA-256 equal
        the recorded ones as they are on disk, or after converting the file's line endings to those of the recorded
        working copy."""
        text = (ANCHOR_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
        rows = {m.group(1): m.groups()[1:] for m in map(self.ROW.match, text.splitlines()) if m}
        self.assertEqual(set(rows), _package_files(ANCHOR_DIR, skip=NOT_FROM_SOURCE))
        for rel, (size, blob, before, after) in sorted(rows.items()):
            data = (ANCHOR_DIR / rel).read_bytes()
            self.assertEqual(_git_blob(data), blob, f"{rel}: content differs from commit 487e0d2")
            forms = {data, _lf(data), _lf(data).replace(b"\n", b"\r\n")}        # as on disk, LF, CRLF
            self.assertIn((int(size), after), {(len(f), hashlib.sha256(f).hexdigest()) for f in forms}, rel)
            self.assertEqual(before, after, f"{rel}: the copy is unedited")

    def test_no_foundation_model_adapter_is_shipped(self):
        self.assertIsNone(importlib.util.find_spec("ankyra_lite.anchor.timesfm_adapter"))
        self.assertFalse((ANCHOR_DIR / "timesfm_adapter.py").exists())


class ForecastEquivalenceTests(_NeedsOriginal):
    def test_every_case_agrees_bit_for_bit(self):
        for i, (name, *_rest) in enumerate(CASES):
            with self.subTest(case=name):
                f_orig, f_lite, inputs = _run_pair(i)
                assert_bitwise_equal(self, f_orig, f_lite, f"{name}: forecast")
                assert_bitwise_equal(self, f_orig.energy_kwh, f_lite.energy_kwh, f"{name}: energy_kwh")
                if inputs["seen"]["orig"] or inputs["seen"]["lite"]:      # callable form: the same contexts were handed out
                    assert_bitwise_equal(self, inputs["seen"]["orig"], inputs["seen"]["lite"], f"{name}: contexts")

    def test_cases_cover_the_branches(self):
        """The case list reaches the off-state and micro-load returns, the projection, the gap rule and the analog guard."""
        flags = {}
        for i, (name, *_rest) in enumerate(CASES):
            f_orig, _, inputs = _run_pair(i)
            raw = np.repeat(f_orig.daily_means_kw, 24) + f_orig.within_day_kw
            flags[name] = dict(off=f_orig.off_state, micro=f_orig.micro_load, projected=bool((raw < 0).any()),
                               pairs=f_orig.pseudo_pairs, kept=f_orig.analog_kept)
        self.assertTrue(flags["off_state"]["off"]); self.assertTrue(flags["micro_load"]["micro"])
        self.assertFalse(flags["micro_load_rule_off"]["micro"])
        self.assertTrue(any(v["projected"] for v in flags.values()), "no case where the nonnegativity projection binds")
        self.assertGreater(flags["public_gaps"]["pairs"], flags["switch_2_1_on_gaps"]["pairs"])   # the short-gap rule recovers pairs
        self.assertTrue(flags["office_16000_perturbed"]["kept"])
        self.assertLess(flags["office_partial_keys"]["pairs"], 6)

    def test_pseudo_origin_contexts_and_gap_filling_agree(self):
        for build in (_gap_record, _masked_record, _plain(16000)):
            load, *_ = build()
            for gt in (True, False):
                assert_bitwise_equal(self, lite_core.pseudo_origin_contexts(load, gap_tolerance=gt),
                                     _orig("core").pseudo_origin_contexts(load, gap_tolerance=gt), "pseudo_origin_contexts")
            assert_bitwise_equal(self, lite_core.fill_short_gaps(load), _orig("core").fill_short_gaps(load), "fill_short_gaps")

    def test_synthetic_helpers_agree(self):
        for kw in (dict(hours=16000), dict(hours=12000, off_last_hours=200), dict(hours=16000, constant_temperature=True)):
            assert_bitwise_equal(self, lite_synthetic.synthetic_history(**kw), _orig("synthetic").synthetic_history(**kw), f"synthetic {kw}")
        ctx = np.random.default_rng(3).gamma(2.0, 5.0, (4, CONTEXT))
        assert_bitwise_equal(self, lite_synthetic.seasonal_naive(ctx), _orig("synthetic").seasonal_naive(ctx), "seasonal_naive")


class ReadoutEquivalenceTests(_NeedsOriginal):
    """Peak, energy and interval readouts computed by each package from its own forecast."""

    def test_peak_energy_and_interval_readouts_agree(self):
        R_orig, R_lite = _orig("readouts"), lite_readouts
        for name in ("office_16800_eu_signed", "residential_20000_us", "public_gaps", "office_observed_mask", "history_only"):
            i = [c[0] for c in CASES].index(name)
            with self.subTest(case=name):
                f_orig, f_lite, x = _run_pair(i)
                load = x["load"] if x["obs"] is None else np.where(x["obs"], x["load"], np.nan)
                types, o = x["types"], len(x["load"])
                ctx = load[o - CONTEXT:o][None]
                ct = types[o - CONTEXT + 12 + 24 * np.arange(56)][None]; tt = types[o + 12 + 24 * np.arange(31)][None]
                for kappa in (1.0, 0.5):
                    assert_bitwise_equal(self, R_orig.peak_readout(f_orig.trajectory_kw, ctx, ct, tt, kappa),
                                         R_lite.peak_readout(f_lite.trajectory_kw, ctx, ct, tt, kappa), f"{name}: peak, kappa {kappa}")
                assert_bitwise_equal(self, R_orig.historical_excursions(ctx, ct, tt), R_lite.historical_excursions(ctx, ct, tt), f"{name}: excursions")
                raw_o = np.repeat(f_orig.daily_means_kw, 24) + f_orig.within_day_kw; raw_l = np.repeat(f_lite.daily_means_kw, 24) + f_lite.within_day_kw
                assert_bitwise_equal(self, R_orig.energy_kwh(f_orig.trajectory_kw), R_lite.energy_kwh(f_lite.trajectory_kw), f"{name}: delivered energy")
                assert_bitwise_equal(self, R_orig.projection_energy_increase_kwh(raw_o), R_lite.projection_energy_increase_kwh(raw_l), f"{name}: projection energy")
                assert_bitwise_equal(self, R_orig.within_day_default(ctx, ct, tt), R_lite.within_day_default(ctx, ct, tt), f"{name}: within-day default")
                g, s = x["kw"]["group"], x["kw"]["temp_sigma_std"]
                Q_o, n_o = R_orig.residual_quantiles(load, x["temp"], types, x["start"], g, s, o)
                Q_l, n_l = R_lite.residual_quantiles(load, x["temp"], types, x["start"], g, s, o)
                assert_bitwise_equal(self, (Q_o, n_o), (Q_l, n_l), f"{name}: residual quantiles")
                s0_o, s0_l = R_orig.origin_scale(ctx[0]), R_lite.origin_scale(ctx[0])
                assert_bitwise_equal(self, s0_o, s0_l, f"{name}: origin scale")
                hod = (o + np.arange(HORIZON)) % 24; work = (types[o:o + HORIZON] < 5).astype(int)
                if np.isfinite(Q_o).all():
                    assert_bitwise_equal(self, R_orig.interval_bands(f_orig.trajectory_kw, s0_o, Q_o, hod, work),
                                         R_lite.interval_bands(f_lite.trajectory_kw, s0_l, Q_l, hod, work), f"{name}: interval bands")


class ComponentEquivalenceTests(_NeedsOriginal):
    def test_reference_estimator_agrees(self):
        E_orig, E_lite = _orig("history.api").estimate_from_history, lite_api.estimate_from_history
        for hours, group, sigma, policy, kw in ((16000, "Office", 0.25, "strict", {}), (16000, "Commercial", 0.25, "strict", {}),
                                                 (10248, "Residential", 0.4, "strict", {}),
                                                 (16000, "Office", 0.25, "source_prior", dict(constant_temperature=True))):
            with self.subTest(hours=hours, group=group, policy=policy):
                h_orig, h_lite = _histories(*_synthetic_arrays(hours, **kw))
                e_orig = E_orig(h_orig, group=group, temp_sigma_std=sigma, boundary_policy=policy)
                e_lite = E_lite(h_lite, group=group, temp_sigma_std=sigma, boundary_policy=policy)
                assert_bitwise_equal(self, e_orig.level_kw, e_lite.level_kw, "level_kw")
                assert_bitwise_equal(self, e_orig.daily_path_kw, e_lite.daily_path_kw, "daily_path_kw")
                d_orig, d_lite = dict(e_orig.diagnostics), dict(e_lite.diagnostics)
                if not _frozen_config_bytes_equal():                    # same content, other line endings: see the docstring
                    for d, root in ((d_orig, ORIGINAL_ROOT / "ankyra"), (d_lite, ANCHOR_DIR)):
                        self.assertEqual(d.pop("config_sha256"), _sha256(root / "history" / "frozen_config.json"))
                assert_bitwise_equal(self, d_orig, d_lite, "diagnostics")
        y, p = np.random.default_rng(4).gamma(2.0, 5.0, (2, HORIZON))
        assert_bitwise_equal(self, _orig("history.api").score_against_observations(p, y), lite_api.score_against_observations(p, y), "score")

    def test_blocks_and_within_day_trust_agree(self):
        B_orig, A_orig = _orig("blocks"), _orig("analog")
        rng = np.random.default_rng(5)
        F, y = rng.normal(10, 3, (6, HORIZON)), rng.normal(10, 3, (6, HORIZON))
        assert_bitwise_equal(self, B_orig.block_losses(F, y), lite_blocks.block_losses(F, y), "block_losses")
        for fn in ("level", "daily_means", "daily_path", "within_day"):
            assert_bitwise_equal(self, getattr(B_orig, fn)(F[0]), getattr(lite_blocks, fn)(F[0]), fn)
        triples = [(rng.normal(size=HORIZON), rng.normal(size=HORIZON), rng.normal(size=HORIZON)) for _ in range(3)]
        for single in (True, False):
            assert_bitwise_equal(self, A_orig.within_trust(triples, single=single), lite_analog.within_trust(triples, single=single), f"within_trust {single}")
        w = [0.5, 0.25, 0.0, 0.4]
        assert_bitwise_equal(self, A_orig.anchored_within_day(F[0], F[1], w), lite_analog.anchored_within_day(F[0], F[1], w), "anchored_within_day")
        load, temp, types, start = _synthetic_arrays(16800); o = len(load)
        S_orig = A_orig.AnalogShapes(load, temp, types, start, "EU", origin=o)
        S_lite = lite_analog.AnalogShapes(load, temp, types, start, "EU", origin=o)
        fb = B_orig.within_day(_orig("synthetic").seasonal_naive(load[-CONTEXT:][None])[0])
        for q in (o, o - STEP, o - 3 * STEP):
            assert_bitwise_equal(self, S_orig.shape_with_sanity(q, fb), S_lite.shape_with_sanity(q, fb), f"analog shape at {q}")

    def test_evaluation_estimands_agree(self):
        M_orig = _orig("metrics")
        rng = np.random.default_rng(6)
        unit = np.repeat([f"u{i}" for i in range(8)], 3); month = np.tile(["2020-01", "2020-02", "2020-03"], 8)
        a, b = rng.gamma(2.0, 1.0, 24), rng.gamma(2.0, 1.2, 24)
        assert_bitwise_equal(self, M_orig.log_ratio(a, b, unit), lite_metrics.log_ratio(a, b, unit), "log_ratio")
        assert_bitwise_equal(self, M_orig.um_interval(a, b, unit, month, np.random.default_rng(0), 200),
                             lite_metrics.um_interval(a, b, unit, month, np.random.default_rng(0), 200), "um_interval")
        assert_bitwise_equal(self, M_orig.contrast(a, b, unit, month, np.random.default_rng(1)),
                             lite_metrics.contrast(a, b, unit, month, np.random.default_rng(1)), "contrast")
        assert_bitwise_equal(self, M_orig.pooled_ratio_decomposition(a, b, unit), lite_metrics.pooled_ratio_decomposition(a, b, unit), "pooled")
        assert_bitwise_equal(self, M_orig.mean_unit_rank({"A": a, "B": b}, unit), lite_metrics.mean_unit_rank({"A": a, "B": b}, unit), "rank")
        y = 10 + rng.random((24, HORIZON)); pred = y + 0.3 * rng.standard_normal(y.shape)
        assert_bitwise_equal(self, M_orig.conventional(pred, y, unit), lite_metrics.conventional(pred, y, unit), "conventional")


class ErrorEquivalenceTests(_NeedsOriginal):
    """Both packages refuse the same inputs with the same exception type (by name and hierarchy) and message."""

    def _both(self, make):
        out = []
        for pkg in (ORIGINAL, lite):
            try:
                make(pkg)
            except Exception as exc:                            # noqa: BLE001 - the comparison is the point
                out.append(exc)
            else:
                out.append(None)
        return out

    def test_invalid_inputs_raise_the_same_errors(self):
        load, temp, types, start = _synthetic_arrays(16000)
        base = _orig("core").pseudo_origin_contexts(load)
        T = {k: np.tile(c[-168:], 5)[:HORIZON] for k, c in base.items()}

        def hist(pkg, load=load, temp=temp, types=types, observed=None):
            return pkg.History(load_kw=load.copy(), temperature_c=temp.copy(), day_types=types.copy(), start_timestamp=start,
                               observed=observed)

        gap = load.copy(); gap[-10] = np.nan
        nan4 = {**T, 4: T[4].copy()}; nan4[4][100] = np.nan
        micro = load.copy(); micro[-CONTEXT:] = 2e-4
        c_load, c_temp, c_types, _ = _synthetic_arrays(16000, constant_temperature=True)
        cases = {
            "lower-case group": lambda p: p.forecast(hist(p), group="office", temp_sigma_std=0.25, foundation=T),
            "record shorter than the context": lambda p: p.forecast(hist(p, load[:1000], temp[:1000], types[:1744]), group="Office", temp_sigma_std=0.25, foundation=_callable_foundation),
            "gap in the context, callable": lambda p: p.forecast(hist(p, gap), group="Office", temp_sigma_std=0.25, foundation=_callable_foundation),
            "unknown dst_region": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=0.25, foundation=T, dst_region="Mars"),
            "foundation of the wrong kind": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=0.25, foundation="timesfm"),
            "origin forecast missing": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=0.25, foundation={1: np.zeros(HORIZON)}),
            "origin forecast not finite": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=0.25, foundation={**T, 0: np.full(HORIZON, np.nan)}),
            "pseudo forecast one hour short": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=0.25, foundation={**T, 2: T[2][:743]}),
            "pseudo forecast not finite": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=0.25, foundation=nan4),
            "negative temp_sigma_std": lambda p: p.forecast(hist(p), group="Office", temp_sigma_std=-1.0, foundation=T),
            "mask of the wrong length on a micro-load context": lambda p: p.forecast(hist(p, micro, observed=np.ones(100, bool)), group="Office", temp_sigma_std=0.25, foundation={0: np.full(HORIZON, 2e-4)}),
            "constant temperature": lambda p: p.forecast(hist(p, c_load, c_temp, c_types), group="Office", temp_sigma_std=0.25, foundation=_callable_foundation),
        }
        for name, make in cases.items():
            with self.subTest(case=name):
                e_orig, e_lite = self._both(make)
                self.assertIsNotNone(e_orig, f"{name}: the original accepted the input")
                self.assertIsNotNone(e_lite, f"{name}: the copy accepted the input")
                self.assertEqual([c.__name__ for c in type(e_orig).__mro__], [c.__name__ for c in type(e_lite).__mro__], name)
                self.assertEqual(str(e_orig), str(e_lite), name)


if __name__ == "__main__":
    unittest.main()
