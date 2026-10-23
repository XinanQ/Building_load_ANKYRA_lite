"""Checks of the baseline comparison (evaluation/compare_baselines.py and its helpers), of the checkpoint export
(tools/export_checkpoints.py) and of the files the comparison wrote.

The comparator, the read-only guard, the path scrubbing, the products mappings (the released span-return recipe and its
from-scratch stage), the p-value format, the re-anchoring modes, the corruptions of the robustness test, the error
blocks and the recipe history are tested directly; the result files are checked for the release's columns, the
scoring-code check, the forecaster set (no ANKYRA version; 21 baselines including LSTM-T-KAN-M as the only LSTM-T-KAN
model; 22 forecasters), the flags (CINELDI: LSTM-T-KAN-M from its corrected-store retrain), the TiDE label, the new files and the
absence of absolute paths.  The files and rows added on 19 October 2026 (``--supplement``: training budget, handover arms,
lead weeks, within-day trust, the fixed division, the whole-window weight, the carrier-alone peak row) are checked for their
columns, labels and flags, and for the gates of the fields cache and of the handover rebuild (class ``AddedFileTests``; each
test skips while its file is absent).  Nothing is recomputed (run the script first).

Configuration.  The products-mapping, export and manifest checks read the configuration from evaluation/products.json
itself (label, version, recipe R or Rw, seeds, checkpoint folder, continuation layer), so they hold for the 0.1.x
mapping (recipe R, seeds 0-2, 27 checkpoint files) and for the 0.2.0 mapping (recipe Rw, seeds 0-5 in
<population>/rw/, plus the 27 files of the 0.1.x configuration).  ``ManifestTests`` checks checkpoints/MANIFEST.json
against that configuration and against ``ankyra_lite._release.checkpoint_paths``.

Skips.  Tests that need the first release's published files skip when neither ANKYRA_ORIGINAL_ROOT/results nor
``git show <release commit>`` can provide them; tests that need the study's research folder skip when it is absent;
the result-file tests skip when the result directory holds no complete run.  The result directory is results/ unless
ANKYRA_LITE_RESULTS names another one (e.g. the output of a development run with --out); a run on fewer than nine
populations is checked on the populations it covers.

Run from the repository root:  python -m unittest tests.test_compare_baselines -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import csv
import importlib.util
import io
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

OTHER_KAN = re.compile(r"LSTM-T-KAN-(?!M\b)")                         # an LSTM-T-KAN model other than LSTM-T-KAN-M (not a baseline)

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("ANKYRA_LITE_RESULTS") or REPO / "results")
sys.path.insert(0, str(REPO / "evaluation"))
_spec = importlib.util.spec_from_file_location("compare_baselines", REPO / "evaluation" / "compare_baselines.py")
CB = importlib.util.module_from_spec(_spec)
sys.modules["compare_baselines"] = CB
_spec.loader.exec_module(CB)                                   # installs the read-only guard; imports no study tool

NAMES = {"bdg2": "BDG2 2017", "cambridge": "Cambridge", "heew": "HEEW Arizona", "eweld": "EWELD", "goiener_confirm": "GoiEner non-household",
         "households": "GoiEner households", "oslo": "Oslo", "drammen": "Drammen", "cineldi": "CINELDI"}
TIDE = "Linear-context (TiDE implementation)"
RECIPE_NAME = {"R": "span return (R)", "Rw": "span return with inverse-noise loss weighting (Rw)"}


def configuration(raw):
    """The configuration a products mapping names, read from the mapping itself: version (from the label's version
    field), recipe key (R | Rw), seeds, checkpoint folder and whether the continuation layer is on.  A 0.1.x mapping
    (no ``configuration`` field) is recipe R, seeds 0-2, no layer, checkpoints in <population>/."""
    parts = raw.get("configuration_parts") or {}
    v020 = bool(raw.get("configuration"))
    return {"v020": v020, "version": str(raw.get("version", "")), "label": raw.get("label"),
            "recipe": parts.get("recipe") or "R", "seeds": [int(s) for s in raw.get("seeds", (0, 1, 2))],
            "folder": str(raw.get("checkpoint_folder") or ""), "layer": bool((raw.get("continuation") or {}).get("enabled"))}


def released_configuration():
    return configuration(json.loads(CB.PRODUCTS.read_text(encoding="utf-8")))


def _release_available():
    """The first release's published files: the committed state (git show) or, failing that, the working copy."""
    try:
        return bool(CB.published_text("benchmark_pairwise.csv"))
    except (OSError, UnicodeDecodeError):
        return False


def _complete_run():
    try:
        rec = json.loads((RESULTS / "compare_baselines.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (RESULTS / "REPRODUCTION_CHECK.json").exists() and bool(rec.get("faces_run")) and "populations" in rec


HAVE_RELEASE = _release_available()
NO_RELEASE = "the first release's results are not available (neither ANKYRA_ORIGINAL_ROOT/results nor git show of the release commit)"
HAVE_RESEARCH = (CB.AO / "smallnet_sr_formal_v1").exists()
NO_RESEARCH = "the study's research folder (ANKYRA_LITE_RESEARCH_ROOT) is not available"
HAVE_RESULTS = _complete_run()


def rows(name):
    with open(RESULTS / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def header(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return next(csv.reader(f))


def record():
    return json.loads((RESULTS / "compare_baselines.json").read_text(encoding="utf-8"))


def sets_run():
    return [NAMES[f] for f in record()["faces_run"]]


def tables_written():
    """The tables of the main comparison this run wrote (all ten in a complete run; a development run on populations
    without published release rows of a table cannot check, and so does not write, that table)."""
    return [t for t in CB.TABLES if t in record()["tables_written"]]


def published(name):
    """A results/ file of the first release as the check reads it (committed state; see compare_baselines.published_text)."""
    return list(csv.reader(io.StringIO(CB.published_text(name))))


def published_rows(name):
    return list(csv.DictReader(io.StringIO(CB.published_text(name))))


@unittest.skipUnless(HAVE_RELEASE, NO_RELEASE)
class ComparatorTests(unittest.TestCase):
    def test_published_rows_reproduce_themselves_and_a_change_is_caught(self):
        pub = [r for r in published_rows("benchmark_pairwise.csv") if r["set"] == "Cambridge" and r["subset"] == "late"]
        scope = lambda r: r["set"] == "Cambridge" and r["subset"] == "late"           # noqa: E731
        self.assertEqual(CB.compare("benchmark_pairwise", [dict(r) for r in pub], scope)["status"], "REPRODUCED")
        bad = [dict(r) for r in pub]
        bad[0]["log_ratio"] = str(float(bad[0]["log_ratio"]) + 2e-6)                 # two units in the last decimal
        res = CB.compare("benchmark_pairwise", bad, scope)
        self.assertEqual(res["status"], "MISMATCH")
        self.assertEqual(len(res["mismatches"]), 1)
        near = [dict(r) for r in pub]
        near[0]["log_ratio"] = str(float(near[0]["log_ratio"]) + 1e-6)                # one unit: published rounding
        self.assertEqual(CB.compare("benchmark_pairwise", near, scope)["status"], "REPRODUCED")
        self.assertEqual(CB.compare("benchmark_pairwise", [dict(r) for r in pub[1:]], scope)["status"], "MISMATCH")   # a missing row


class GuardTests(unittest.TestCase):
    def test_writes_under_the_read_only_trees_are_refused(self):
        target = CB.HKAD / "__guard_test_nonexistent_dir__" / "x.txt"                  # a missing directory: nothing can be created
        with self.assertRaises(PermissionError):
            open(target, "w")
        with self.assertRaises(PermissionError):
            os.mkdir(CB.REL / "__guard_test_nonexistent_dir__" / "sub")             # parent missing: nothing can be created
        self.assertFalse(CB._in_read_only(RESULTS / "x.csv"))


class HelperTests(unittest.TestCase):
    def test_scrub_removes_the_roots_and_home_paths(self):
        text = f"{CB.HKAD}\\analysis_outputs and {str(CB.REL).replace(os.sep, '/')}/results and C:\\Users\\someone\\x and {CB.REPO}"
        out = CB.scrub(text)
        self.assertNotIn(str(CB.HKAD), out)
        self.assertNotIn("Users", out)
        self.assertIn("<research root>", out)
        self.assertIn("<original release root>", out)
        self.assertEqual(CB.show(CB.AO / "x" / "y.npz"), "<research root>/analysis_outputs/x/y.npz")

    def test_p_values_below_the_smallest_double(self):
        self.assertEqual(CB.fmt_p(0.0), "<1e-300")
        self.assertEqual(CB.fmt_p(0.0, release=True), "0")
        self.assertEqual(CB.fmt_p(0.01234), "0.0123")
        self.assertEqual(CB._p_text(0.0), "p < 1e-300")

    def test_reference_key_is_renamed_in_records(self):
        r = CB._rename_ref({"mean_rank": {"ANKYRA": 1.5, "TimesFM": 3.0}, "x": ["ANKYRA", "LSTM"]}, "ankyra-lite")
        self.assertEqual(r, {"mean_rank": {"ankyra-lite": 1.5, "TimesFM": 3.0}, "x": ["ankyra-lite", "LSTM"]})

    def test_released_products_mapping_is_the_span_return(self):
        raw = json.loads(CB.PRODUCTS.read_text(encoding="utf-8"))
        cfg = configuration(raw)
        self.assertIn(cfg["version"], ("0.1.0", "0.2.0"))
        self.assertEqual(raw["label"], f"ankyra-lite {cfg['version']}")                 # the label the mapping itself names
        self.assertEqual(raw["recipe"], RECIPE_NAME[cfg["recipe"]])
        self.assertEqual(raw["checkpoint"], "final.pt")
        self.assertEqual(raw["replacements"], {})
        self.assertEqual(sorted(raw["faces"]), sorted(CB.NINE))
        seeds, key = cfg["seeds"], cfg["recipe"]
        if cfg["v020"]:
            self.assertEqual((key, len(seeds), cfg["layer"]), ("Rw", 6, True))                # the adopted configuration of 0.2.0
            self.assertEqual(raw["checkpoint_folder"], "rw")
            self.assertEqual(raw["seed_groups"], [[0, 1, 2], [3, 4, 5]])
            self.assertEqual(raw["continuation"]["constants"], {"kappa1": 0.479, "rho1": 12.0, "kappa2": 0.33, "rho2": 168.0})
            self.assertIn("readout rule A", raw["continuation"]["readout_rule"])
        else:
            self.assertEqual((key, seeds, cfg["layer"]), ("R", [0, 1, 2], False))
        for f, e in raw["faces"].items():
            self.assertEqual(e["store"], "corrected" if f in ("drammen", "cineldi") else "original", f)
            self.assertTrue(e["npz"].endswith(f"/arms_{key}_ens{len(seeds) if cfg['v020'] else ''}.npz"), f)
            self.assertEqual([Path(x).name for x in e["run_dirs"]], [f"{key}_s{s}" for s in seeds], f)
            self.assertEqual([Path(x).name for x in e.get("seed_npz", [])], [] if cfg["v020"] else [f"arms_R_s{s}.npz" for s in seeds], f)
            self.assertEqual([Path(x).name for x in e["start_run_dirs"]], [f"P3s_s{s}" for s in seeds], f)
            self.assertLessEqual({"F", "P3s", "V"}, set(e["recipe_variants"]), f)
            if not cfg["v020"]:
                self.assertEqual(sorted(e["recipe_variants"]), ["F", "P3s", "V"], f)
            else:
                self.assertEqual(len(e["group_npz"]), len(raw["seed_groups"]), f)
                self.assertEqual(bool(e.get("continued_npz")), cfg["layer"], f)
                self.assertTrue(e.get("weights_record"), f)                                   # recipe Rw: the loss weights' record
            paths = ([e["npz"], e["prep"], e["span_plan"]] + e["run_dirs"] + e.get("seed_npz", []) + e["start_run_dirs"]
                     + list(e["recipe_variants"].values()) + e.get("group_npz", []) + e.get("group_continued_npz", [])
                     + [x for x in (e.get("continued_npz"), e.get("weights_record")) if x])
            for path in paths:
                self.assertFalse(re.match(r"^[A-Za-z]:|^/|^\\\\", path), path)          # relative to the research root
            if f in ("drammen", "cineldi"):                                           # the corrected-store retrain
                self.assertTrue(any("/dc/" in x for x in [e["npz"]] + e["start_run_dirs"][:3]), f)
        self.assertTrue(raw["faces"]["cineldi"]["prep"].startswith("analysis_outputs/smallnet_sr_formal_v1/dc/v1rec/"))
        p = CB.load_products(CB.PRODUCTS)
        self.assertEqual(p["faces"]["oslo"]["checkpoint"], "final.pt")
        self.assertEqual(p["faces"]["oslo"]["start_checkpoint"], "best.pt")
        self.assertEqual(len(p["faces"]["oslo"]["seed_paths"]), 0 if cfg["v020"] else 3)
        self.assertEqual(len(p["faces"]["oslo"]["run_paths"]), len(seeds))
        self.assertEqual(p["faces"]["oslo"]["layer"], cfg["layer"])
        fixed = CB.kan_fixed_record() is not None                                # LSTM-T-KAN-M retrained on the corrected store
        self.assertEqual(CB.face_flag("cineldi", p["faces"]["cineldi"]), "" if fixed else CB.CORRECTED_CINELDI_FLAG)
        self.assertEqual(CB.kan_dir("cineldi", p["faces"]["cineldi"]), (CB.KAN_FIX_DIR / "cineldi", True) if fixed else (CB.KAN_DIR / "cineldi", False))
        self.assertEqual(CB.kan_dir("oslo", p["faces"]["oslo"]), (CB.KAN_DIR / "oslo", False))
        self.assertIn("LSTM-T-KAN-M", CB.CORRECTED_CINELDI_FLAG)
        self.assertEqual(CB.face_flag("drammen", p["faces"]["drammen"]), "")
        self.assertEqual(CB.face_flag("oslo", p["faces"]["oslo"]), "")
        self.assertEqual(CB.face_flag("cineldi", {"store": "original"}), CB.ORIGINAL_STORE_FLAG)
        self.assertTrue(CB.on_corrected_cineldi("cineldi", p["faces"]["cineldi"]))
        self.assertFalse(CB.on_corrected_cineldi("drammen", p["faces"]["drammen"]))

    def test_from_scratch_mapping_is_kept_and_selectable(self):
        raw = json.loads(CB.RECIPE_PRODUCTS["p3s"].read_text(encoding="utf-8"))
        self.assertNotEqual(raw["label"], json.loads(CB.PRODUCTS.read_text(encoding="utf-8"))["label"])   # its own re-anchoring cache
        self.assertEqual(sorted(raw["faces"]), sorted(CB.NINE))
        self.assertEqual(raw["checkpoint"], "best.pt")
        self.assertEqual(sorted(raw["superseded"]), ["cineldi", "drammen"])
        self.assertTrue(all(e["store"] == "original" for e in raw["superseded"].values()))
        for f, e in raw["faces"].items():
            self.assertEqual(len(e["run_dirs"]), 3)
            self.assertEqual(bool(e.get("crosscheck")), f not in ("drammen", "cineldi"), f)
            self.assertEqual(e["store"], "corrected" if f in ("drammen", "cineldi") else "original", f)
        p = CB.load_products(CB.RECIPE_PRODUCTS["p3s"])
        self.assertEqual(p["faces"]["oslo"]["seed_paths"], [])
        self.assertEqual(p["faces"]["oslo"]["checkpoint"], "best.pt")
        self.assertEqual(sorted(p["superseded"]), ["cineldi", "drammen"])

    def test_lstm_t_kan_m_is_the_only_lstm_t_kan_baseline(self):
        import extra_tables as XT
        self.assertEqual(CB.KAN, ("LSTM-T-KAN-M",))
        self.assertEqual(CB.N_BASELINES, 21)
        self.assertEqual(sorted(CB.KAN_LABEL), ["LSTM-T-KAN-M"])
        self.assertEqual(sorted(CB.KAN_CLASS), ["LSTM-T-KAN-M"])
        self.assertEqual([m for m in CB.SUMMARY_BASELINES if m.startswith("LSTM-T-KAN")], list(CB.KAN))
        self.assertEqual(XT.DEEP_CARRIERS, ("LSTM", "PatchTST", "LSTM-T-KAN-M"))
        for f in list((REPO / "evaluation").glob("*.py")) + list((REPO / "evaluation").glob("*.json")):
            self.assertIsNone(OTHER_KAN.search(f.read_text(encoding="utf-8")), f.name)
        self.assertIn("LSTM-T-KAN", CB.KAN_NOT_BASELINES)       # no LSTM-T-KAN model is read from a panel folder (M: its own folder)

    @unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
    def test_corrected_store_lstm_t_kan_m_retrain(self):
        k = CB.kan_fixed_record()
        if k is None:
            self.skipTest("no corrected-store LSTM-T-KAN-M retrain (CINELDI keeps the original-store forecast and its flag)")
        self.assertTrue(k["gate"]["pass"] and k["gate"]["bitwise"])                   # the original store was reproduced bit for bit first
        self.assertTrue(k["record"]["checks"]["pass"])
        self.assertTrue(k["record"]["checks"]["cutoff_same_instant"])
        a = np.load(CB.KAN_FIX_DIR / "cineldi" / "pred_LSTM-T-KAN-M.npz", allow_pickle=False)
        b = np.load(CB.KAN_DIR / "cineldi" / "pred_LSTM-T-KAN-M.npz", allow_pickle=False)
        self.assertTrue(np.array_equal(a["building"], b["building"]) and np.array_equal(a["origin"], b["origin"]))   # original origin numbers
        self.assertTrue(np.array_equal(a["origin_fixed_store"], a["origin"] + 1416))
        self.assertFalse(np.array_equal(a["pred"], b["pred"]))

    def test_tide_label_is_the_first_release_label(self):
        self.assertEqual(CB.TIDE_LABEL, TIDE)
        wc = CB.REL_RES / "benchmark_pairwise.csv"
        if not wc.exists():
            self.skipTest("the first release's working copy is not available")
        with open(wc, encoding="utf-8") as fh:
            labels = {r["model_label"] for r in csv.DictReader(fh) if r["model"] == "TiDE"}
        self.assertEqual(labels, {TIDE})

    def test_no_internal_root_names_in_the_scripts(self):
        for f in list((REPO / "evaluation").glob("*.py")) + list((REPO / "tools").glob("*.py")) + list((REPO / "evaluation").glob("*.json")):
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"[A-Za-z]:[\\/]+Users", f.name)
            self.assertNotRegex(text, r"Building_load_ANKYRA_lite|ankyra-lite[\\/]", f.name)          # the repository folder's own name

    def test_reanchor_modes_use_the_stored_single_seed_products(self):
        import lite_reanchor as LR
        p = CB.load_products()
        self.assertEqual(LR.default_modes(p["faces"]["oslo"]), ["ens", "no_within", "lead_trust", "no_gap", "no_micro"])
        self.assertEqual(LR.default_modes(CB.load_products(CB.RECIPE_PRODUCTS["p3s"])["faces"]["oslo"]), list(LR.MODES))
        identity = LR.cache_identity(label="x", face="oslo", mode="s0", limit=None, spec=LR.MODES["s0"], layer=False,
                                     seeds=[0], seed_groups=[[0]], constants_digest=None,
                                     input_manifest=[{"path": "synthetic", "sha256": "0" * 64}], implementation_digest="test")
        self.assertNotEqual(LR.cache_file("x", "oslo", "s0", None, identity), LR.cache_file("x", "oslo", "s0", 8, identity))
        with self.assertRaises(ValueError):                                     # every cache access needs a full identity
            LR.cache_file("x", "oslo", "s0")

    def test_corruptions_of_the_robustness_test(self):
        import robustness as RB
        rng = np.random.default_rng(RB.SEED)
        load = np.linspace(1.0, 2.0, 3000)
        o = 2500
        for kind in RB.KINDS:
            L, c = RB.corrupt(load, o, kind, rng)
            self.assertEqual(len(L), o, kind)
            self.assertTrue(np.isfinite(L[o - RB.CONTEXT:]).all(), kind)                 # gaps are filled: the context stays complete
            if kind.startswith("scale"):
                self.assertTrue(np.allclose(L, load[:o] * c), kind)
            else:
                self.assertEqual(c, 1.0, kind)
                self.assertTrue(np.array_equal(L[:o - RB.CONTEXT], load[:o - RB.CONTEXT]), kind)   # only the context is corrupted
        L, _ = RB.corrupt(load, o, "zero24", rng)
        self.assertTrue((L[o - 120:o - 96] == 0).all())
        L, _ = RB.corrupt(load, o, "spike", rng)
        self.assertAlmostEqual(L[o - 36], 10 * load[o - RB.CONTEXT:o].max())
        L, _ = RB.corrupt(load, o, "shift_plus1", rng)
        self.assertTrue(np.array_equal(L[o - RB.CONTEXT + 1:o], load[o - RB.CONTEXT:o - 1]))
        self.assertEqual(RB.SEED, 20261003)
        self.assertEqual(RB.KINDS[1:], ["miss05", "miss20", "zero24", "spike", "shift_plus1", "shift_minus1", "scale0.5", "scale2"])

    def test_error_blocks_add_up(self):
        import extra_tables as XT
        e = np.random.default_rng(0).normal(size=(5, 744))
        b = XT.Collector.blocks(e)
        self.assertTrue(np.allclose(b["level"] + b["daily path"] + b["within-day"], (e ** 2).mean(1)))
        self.assertEqual(XT.BLOCKS, ("hourly", "level", "daily path", "within-day"))

    @unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
    def test_recipe_history_copies_the_study_record(self):
        import recipe_tables as RT
        rws, rec = RT.recipe_history()
        S = json.loads(CB.SR_SCORE.read_text(encoding="utf-8"))
        r = next(x for x in rws if x["comparison"] == "R vs F" and x["populations"] == 3 and x["layer"] == "all test windows" and x["quantity"] == "hourly")
        v = S["three short-record populations"]["R vs F"]["hourly"]
        self.assertEqual(r["improvement_pct"], v["pct"])
        self.assertEqual([r["interval_low_pct"], r["interval_high_pct"]], v["welch_95_pct"])
        self.assertEqual(r["label"], v["label"])
        sw = [x for x in rws if x["layer"].startswith("switch layer") and x["comparison"] == "R vs F" and x["quantity"] == "hourly"]
        self.assertEqual(len(sw), 1)
        self.assertEqual(sw[0]["label"], S["switch layer | three short-record populations"]["R vs F"]["hourly"]["label"])
        self.assertEqual({x["comparison"] for x in rws}, {"R vs F", "V vs F", "V vs R", "F vs P3s", "R vs P3s", "V vs P3s"})
        self.assertEqual(len(rws), 4 * 3 * 7 + 2 * 3 * 3 + 6)

    @unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
    def test_products_files_exist(self):
        for name in ("R", "p3s"):
            chk = CB.check_products(CB.load_products(CB.RECIPE_PRODUCTS[name]))
            for k, v in chk.items():
                if not k.startswith("faces:"):
                    continue
                self.assertTrue(v["exists"] and v["run_dirs_exist"] and v["checkpoints_exist"], (name, k))
                if name == "R":
                    self.assertTrue(v["start_runs_exist"] and all(v["variants_exist"].values()), (name, k))
                    if released_configuration()["v020"]:                     # 0.2.0: the three-seed groups, the continued product
                        self.assertTrue(v["groups_exist"] and v["weights_record_exists"], (name, k))
                        self.assertEqual(v.get("continued_exists", False), released_configuration()["layer"], (name, k))
                    else:                                                     # 0.1.x: the stored single-seed products
                        self.assertTrue(v["seed_products_exist"], (name, k))


@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class ExportTests(unittest.TestCase):
    """tools/export_checkpoints.py into a temporary directory (one population, no data store read)."""

    def test_export_one_population(self):
        spec = importlib.util.spec_from_file_location("export_checkpoints", REPO / "tools" / "export_checkpoints.py")
        EX = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(EX)
        from ankyra_lite import load_carrier
        cfg = released_configuration()
        prods = CB.load_products(CB.PRODUCTS)
        runs = json.loads(CB.PRODUCTS.read_text(encoding="utf-8"))["faces"]["oslo"]["run_dirs"]
        out = Path(tempfile.mkdtemp(prefix="ankyra_lite_export_test_"))
        try:
            files = EX.export(out, prods, faces=("oslo",), log=lambda *a: None)
            self.assertEqual([f["file"] for f in files], ["/".join(x for x in ("oslo", cfg["folder"], f"seed{s}.pt") if x) for s in cfg["seeds"]])
            self.assertEqual((EX.folder_of(prods), EX.seeds_of(prods)), (cfg["folder"], cfg["seeds"]))
            for f, rd in zip(files, runs):
                self.assertEqual(EX.sha256(out / f["file"]), f["sha256"])
                self.assertEqual(f["parameters"], 61613)
                self.assertEqual(f["training_cut_date"], "2019-03-29")
                self.assertEqual(f["source_run"], rd)                                 # the configuration's run of that seed
                self.assertFalse(re.search(r"[A-Za-z]:[\\/]", json.dumps(f)), f)
                load_carrier(out / f["file"])                                    # strict
        finally:
            shutil.rmtree(out, ignore_errors=True)


MANIFEST = REPO / "checkpoints" / "MANIFEST.json"


@unittest.skipUnless(MANIFEST.exists(), "no checkpoints/MANIFEST.json in this checkout")
class ManifestTests(unittest.TestCase):
    """checkpoints/MANIFEST.json against the configuration evaluation/products.json names (no data store, no results/):
    0.1.x, 9 populations x seeds 0-2 in <population>/ (27 files); 0.2.0, 9 x the release seeds in
    <population>/<checkpoint_folder>/ plus, unless exported with --no-0-1, the 27 files of the 0.1.x configuration."""

    def test_manifest_counts_follow_the_configuration(self):
        cfg = released_configuration()
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(man["package"], cfg["label"])                                 # exported from this mapping
        self.assertEqual(man["count"], len(man["files"]))
        names = [f["file"] for f in man["files"]]
        self.assertEqual(len(names), len(set(names)))
        release = sorted("/".join(x for x in (p, cfg["folder"], f"seed{s}.pt") if x) for p in CB.NINE for s in cfg["seeds"])
        legacy = sorted(f"{p}/seed{s}.pt" for p in CB.NINE for s in (0, 1, 2))
        if cfg["v020"]:
            confs = man["configurations"]
            self.assertEqual(man["default_configuration"], "release")
            self.assertEqual(confs["release"]["seeds"], cfg["seeds"])
            self.assertEqual(confs["release"]["recipe_key"], cfg["recipe"])
            self.assertEqual(confs["release"]["continuation_layer"], cfg["layer"])
            want = set(release) | (set(legacy) if "0.1.x" in confs else set())
            self.assertEqual(sorted(names), sorted(want))
            self.assertEqual(sorted(f["file"] for f in man["files"] if f.get("in_release_configuration")), release)
        else:
            self.assertEqual(sorted(names), legacy)
            self.assertEqual(len(names), 27)
        for f in man["files"]:
            self.assertEqual(f["parameters"], 61613, f["file"])
            self.assertFalse(re.search(r"[A-Za-z]:[\\/]", json.dumps(f)), f["file"])
            p = REPO / "checkpoints" / f["file"]
            if p.exists():                                                             # weights may be absent in a light clone
                self.assertEqual(p.stat().st_size, f["bytes"], f["file"])

    def test_release_switches_name_the_manifest_files(self):
        from ankyra_lite import _release
        files = {f["file"] for f in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}
        root = REPO / "checkpoints"
        for p in CB.NINE:
            for path in _release.checkpoint_paths(p, root):                             # the package's default checkpoints
                self.assertIn(path.relative_to(root).as_posix(), files, (p, _release.CARRIER_RECIPE, _release.CARRIER_SEEDS))


@unittest.skipUnless(HAVE_RESULTS, "no complete run in the result directory (run evaluation/compare_baselines.py first)")
class ResultFileTests(unittest.TestCase):
    def test_scoring_code_check_passed_on_every_table(self):
        rec = json.loads((RESULTS / "REPRODUCTION_CHECK.json").read_text(encoding="utf-8"))
        self.assertIn("scoring-code check", rec["what"])
        self.assertTrue(set(tables_written()) <= set(rec["tables_reproduced"]))          # only reproduced tables are written
        if not record().get("partial_run"):
            self.assertEqual(sorted(rec["tables_reproduced"]), sorted(CB.TABLES))
            self.assertEqual(rec["tables_not_reproduced"], [])
        for t, c in rec["tables"].items():
            if c["status"].startswith("NOT CHECKED") and record().get("partial_run"):
                self.assertEqual(c["published_rows_in_scope"], 0, t)                  # no published row for these populations
                continue
            self.assertEqual(c["status"], "REPRODUCED", t)
            self.assertGreater(c["published_rows_in_scope"], 0, t)

    @unittest.skipUnless(HAVE_RELEASE, NO_RELEASE)
    def test_release_columns_plus_flag(self):
        for t in tables_written():
            self.assertEqual(header(f"{t}.csv"), published(f"{t}.csv")[0] + ["flag"], t)

    def test_no_ankyra_version_and_22_forecasters_with_lstm_t_kan_m(self):
        family = set(CB.A_FAMILY) | {"ANKYRA", "ANKYRA 1.x"}
        for t in tables_written():
            for r in rows(t):
                if "model" in r:
                    self.assertNotIn(r["model"], family, t)
                    self.assertFalse(r["model"].startswith("ANKYRA"), (t, r["model"]))
        rk, bp = rows("benchmark_mean_unit_rank"), rows("benchmark_pairwise")
        for s in sets_run():
            mine = [r for r in rk if r["set"] == s]
            self.assertEqual(len(mine), CB.N_BASELINES + 1, s)
            self.assertTrue(all(r["n_models"] == str(CB.N_BASELINES + 1) and r["subset"] == "late" for r in mine), s)
            self.assertIn("ankyra-lite", {r["model"] for r in mine})
            models = {r["model"] for r in bp if r["set"] == s}
            self.assertEqual(len(models), CB.N_BASELINES, s)
            for k in CB.KAN:
                self.assertIn(k, models, s)
        for f in list(RESULTS.glob("*.csv")) + [RESULTS / "SUMMARY.md"]:       # LSTM-T-KAN-M is the only LSTM-T-KAN model
            self.assertIsNone(OTHER_KAN.search(f.read_text(encoding="utf-8")), f.name)
        for t in [x for x in ("conventional_metrics_late", "energy_error", "rank_tests", "scaled_errors") if x in tables_written()]:
            got = {(r["set"], r["model"]) for r in rows(t)}
            for s in sets_run():
                for k in CB.KAN:
                    self.assertIn((s, k), got, (t, s, k))
        pk = {(r["set"], r["comparator"]) for r in rows("peak_readout")}
        for s in sets_run():
            for k in CB.KAN:
                self.assertIn((s, f"same readout on {k}"), pk)
        for t in [x for x in ("lead_day_metrics", "lead_day_energy") if x in tables_written()]:
            got = {(r["set"], r["model"]) for r in rows(t)}
            for s in set(sets_run()) & {NAMES[f] for f in CB.TEST6}:
                for k in CB.KAN:
                    self.assertIn((s, k), got, (t, s, k))

    def test_tide_label_in_every_table(self):
        for f in RESULTS.glob("*.csv"):
            with open(f, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("model") == "TiDE" and "model_label" in r:
                        self.assertEqual(r["model_label"], TIDE, f.name)

    def test_flags(self):
        rec = record()
        want = {p["name"]: (p["flag"] or "") for p in rec["populations"].values()}
        for t in tables_written():
            for r in rows(t):
                self.assertEqual(r["flag"], want.get(r["set"], ""), (t, r["set"]))
        for f, p in rec["populations"].items():
            store = rec["products"][f]["store"]
            if f in ("drammen", "cineldi") and store == "original":
                self.assertEqual(p["flag"], CB.ORIGINAL_STORE_FLAG)
            if f == "cineldi" and store == "corrected":
                kan = p["extra_baselines"]["LSTM-T-KAN-M"]
                fixed = kan["store"].startswith("corrected CINELDI store")
                self.assertEqual(p["flag"] or "", "" if fixed else CB.CORRECTED_CINELDI_FLAG)   # flagged only when kept from the original store
                self.assertIn("lstm_tkan_baseline_v2_cineldi_t0fix" if fixed else "lstm_tkan_baseline_v2/", kan["file"])
                if fixed:
                    self.assertTrue(kan["retrain"]["gate_pass"] and kan["retrain"]["checks_pass"])
                self.assertIn("LSTM-T-KAN-M", {r["model"] for r in rows("benchmark_pairwise") if r["set"] == "CINELDI"})
                self.assertIn("LSTM-T-KAN-M", (RESULTS / "SUMMARY.md").read_text(encoding="utf-8"))
                marked = re.search(r"LSTM-T-KAN-M \+(?![\d.])", (RESULTS / "SUMMARY.md").read_text(encoding="utf-8"))   # the cell mark, not a signed number
                self.assertEqual(bool(marked), not fixed)
            if f == "drammen" and store == "corrected":
                self.assertEqual(p["flag"] or "", "")

    def test_hourly_numbers_match_the_earlier_hourly_comparison(self):
        if not CB.HOURLY_XCHECK.exists():
            self.skipTest("earlier hourly comparison not found")
        x = json.loads(CB.HOURLY_XCHECK.read_text(encoding="utf-8"))["faces"]
        rec = record()
        bp = {(r["set"], r["model"]): r for r in rows("benchmark_pairwise")}
        checked = 0
        for face, v in x.items():
            p = rec["populations"].get(face)
            if p is None or not rec["products"][face].get("crosscheck"):
                continue
            s = NAMES[face]
            for m, c in v["vs"].items():
                r = bp[(s, m)]
                self.assertAlmostEqual(float(r["improvement_pct"]), round(c["pct"], 3), places=9, msg=(s, m))
                self.assertAlmostEqual(float(r["um_low"]), round(c["UM"][0], 6), places=9, msg=(s, m))
                self.assertAlmostEqual(float(r["um_high"]), round(c["UM"][1], 6), places=9, msg=(s, m))
            xc = p["hourly_crosscheck"]
            self.assertAlmostEqual(xc["mean_unit_rank_among_21"], v["mean_unit_rank"], places=9, msg=s)
            self.assertEqual(xc["rank_among_21"], v["rank"], s)
            self.assertEqual(xc["max_abs_diff"], 0.0, s)
            checked += 1
        if not checked:
            self.skipTest("no population of this run is the product of the earlier hourly comparison (from-scratch stage)")

    def test_product_reading_reproduces_the_study_scorer(self):
        rec = record()
        if not str(rec.get("recipe", "")).startswith("span return"):
            self.skipTest("not the span-return product")
        got = [p.get("extra", {}).get("study_scorer_crosscheck") for p in rec["populations"].values()]
        got = [g for g in got if g]
        if not got:
            self.skipTest("no study scorer record available in this run")
        for g in got:
            self.assertTrue(g["pass"], g)

    def test_reference_rows(self):
        for r in (rows("energy_error") if "energy_error" in tables_written() else []):
            if r["model"] == "ankyra-lite":
                self.assertEqual(float(r["log_ratio"]), 0.0)
        pk = rows("peak_readout")
        for s in sets_run():
            mine = {r["comparator"] for r in pk if r["set"] == s}
            for c in ("trajectory maximum", "previous-window peak", "same readout on TimesFM", "trajectory maximum of TimesFM"):
                self.assertIn(c, mine, s)

    def test_records_use_the_reference_name_and_no_p_equals_zero(self):
        rec = record()
        text = (RESULTS / "compare_baselines.json").read_text(encoding="utf-8")
        self.assertNotIn('"ANKYRA":', text)
        if "rank_tests_pooled" in rec:
            self.assertIn("ankyra-lite", rec["rank_tests_pooled"]["mean_rank"])
        self.assertNotIn("p = 0,", (RESULTS / "SUMMARY.md").read_text(encoding="utf-8"))
        self.assertNotIn("p = 0 ", (RESULTS / "SUMMARY.md").read_text(encoding="utf-8"))
        for r in rows("rank_tests"):
            self.assertNotEqual(r["friedman_p"], "0", r["set"])

    def test_no_absolute_or_personal_path_in_any_result_file(self):
        bad = re.compile(r"[A-Za-z]:[\\/]|\\\\Users|/Users/|/home/")
        for f in RESULTS.iterdir():
            if f.is_file() and f.suffix in (".csv", ".json", ".md", ".log", ".txt"):
                text = f.read_text(encoding="utf-8")
                m = bad.search(text)
                self.assertIsNone(m, (f.name, m and text[max(0, m.start() - 40):m.end() + 40]))

    def test_no_comparison_with_ankyra_in_the_tables(self):
        for f in RESULTS.glob("*.csv"):
            with open(f, encoding="utf-8") as fh:
                table = list(csv.DictReader(fh))
            for r in table:
                for k in ("model", "comparator", "ablation", "carrier", "contrast", "arm", "component", "first_arm", "second_arm"):
                    v = r.get(k) or ""
                    self.assertNotRegex(v, r"^ANKYRA|F1-off|^F0$|^F1$|ANKYRA 2|ANKYRA-", (f.name, k, v))

    @unittest.skipUnless(HAVE_RELEASE, NO_RELEASE)
    def test_extra_files_mirror_the_release_columns(self):
        rec = record()
        for name in rec.get("extra_written", []):
            self.assertTrue((RESULTS / name).exists(), name)
        mirror = {"datasets.csv": 9, "ablation.csv": 7, "component_contributions.csv": 6, "history_length.csv": 9,
                  "bdg2_near_zero_sensitivity.csv": 8, "example_window_cambridge.csv": 4, "cost_per_window.csv": 3,
                  "anchoring_gain_deciles.csv": 12, "intervals_winkler_contrasts.csv": 7, "constants.csv": 5}
        for name, n in mirror.items():
            if (RESULTS / name).exists():
                self.assertEqual(header(name)[:n], published(name)[0][:n], name)
        for name in ("contrast_ii_blocks.csv", "contrast_ii_control.csv"):
            if (RESULTS / name).exists():
                self.assertEqual(header(name), published(name)[0] + ["flag"], name)
        if (RESULTS / "intervals_by_population.csv").exists():
            self.assertEqual(header("intervals_by_population.csv")[:-1], published("intervals_by_population.csv")[0])
            arms = {r["arm"] for r in rows("intervals_by_population")}
            self.assertEqual(arms, {"ankyra_lite", "timesfm_native", "chronos_native"})
        if (RESULTS / "carriers_in_anchoring_layer.csv").exists():
            self.assertEqual(header("carriers_in_anchoring_layer.csv")[:-1], published("trained_carriers.csv")[0])
            got = {(r["set"], r["carrier"]) for r in rows("carriers_in_anchoring_layer")}
            for s in sets_run():
                for c in ("small carrier network (ankyra-lite)", "LSTM", "PatchTST", "LSTM-T-KAN-M"):
                    self.assertIn((s, c), got, (s, c))
            self.assertEqual({c for _, c in got if c.startswith("LSTM-T-KAN")}, set(CB.KAN))
        if (RESULTS / "robustness.csv").exists():
            pub = published("robustness.csv")[0]
            mine = header("robustness.csv")
            self.assertEqual([c.replace("carrier", "timesfm") for c in mine if c.replace("carrier", "timesfm") in pub], [c for c in pub])
        if (RESULTS / "constants.csv").exists():
            pub = {(r["group"], r["constant"]): r for r in published_rows("constants.csv")}
            mine = [r for r in rows("constants") if r["component"].startswith("anchoring layer")]
            self.assertEqual(len(mine), len(pub))
            for r in mine:
                p = pub[(r["group"], r["constant"])]
                self.assertEqual((r["value"], r["where_fixed"]), (p["value"], p["where_fixed"]), r["constant"])

    def test_ablation_has_the_carrier_alone_the_state_rule_and_the_recipe_arms(self):
        if not (RESULTS / "ablation.csv").exists():
            self.skipTest("no ablation.csv")
        ab = {(r["set"], r["ablation"], r["error"]) for r in rows("ablation")}
        labs = ["carrier alone (no anchoring layer)", "state rule off (the anchoring layer's own return on state windows)"]
        rec = record()
        if str(rec.get("recipe", "")).startswith("span return"):
            labs += ["recipe: 8 continuation epochs, validation set held out, no span return (anchored)"]
            if rec.get("configuration"):                 # 0.2.0: six seeds, the two stored three-seed groups, the layer
                labs += ["three-seed group 0 (seeds 0-2, anchored alone)", "three-seed group 1 (seeds 3-5, anchored alone)",
                         "without the continuation layer (the same carrier and anchoring layer)",
                         "0.1.0 configuration: recipe R (span return), seeds 0-2, no continuation layer (anchored)"]
            else:
                labs += ["single seed 0 (its carrier anchored alone)"]
        for s in sets_run():
            for lab in labs:
                for err in ("hourly", "energy"):
                    self.assertIn((s, lab, err), ab, (s, lab, err))

    def test_training_curves_have_both_stages(self):
        if not (RESULTS / "training_curves.csv").exists():
            self.skipTest("no training_curves.csv")
        tc = rows("training_curves")
        stages = {(r["set"], r["stage"]) for r in tc}
        rec = record()
        cont = str(rec.get("recipe", "")).startswith("span return")
        stage2 = ("span return with inverse-noise loss weighting (fixed epochs)" if "inverse-noise" in str(rec.get("recipe", ""))
                  else "span return (fixed epochs)")
        n_seeds = len(rec.get("seeds") or (0, 1, 2))
        for s in sets_run():
            self.assertIn((s, "from scratch (early stopping)"), stages, s)
            if cont:
                self.assertIn((s, stage2), stages, s)
                sel = [r for r in tc if r["set"] == s and r["stage"] == stage2 and r["selected"] == "True"]
                self.assertEqual(sorted(int(r["epoch"]) for r in sel), [8] * n_seeds, s)

    def test_new_files(self):
        rec = record()
        if (RESULTS / "recipe_history.csv").exists():
            rh = rows("recipe_history")
            self.assertTrue(any(r["comparison"] == "R vs F" and r["layer"].startswith("switch layer") for r in rh))
        if (RESULTS / "contrast_ii_blocks.csv").exists():
            got = {(r["set"], r["contrast"], r["block"]) for r in rows("contrast_ii_blocks") if r["analysis"] == "all late windows"}
            for s in sets_run():
                for c in ("anchored vs history only", "anchored vs carrier alone", "carrier alone vs history only"):
                    for b in ("hourly", "level", "daily path", "within-day"):
                        self.assertIn((s, c, b), got, (s, c, b))
            for p in rec["populations"].values():
                ci = p.get("extra", {}).get("contrast_ii")
                if ci:
                    self.assertTrue(ci["reproduces_carriers_in_anchoring_layer"], p["name"])
        if (RESULTS / "robustness.csv").exists():
            rb = rows("robustness")
            self.assertEqual([r["corruption"] for r in rb], ["miss05", "miss20", "zero24", "spike", "shift_plus1", "shift_minus1", "scale0.5", "scale2"])

    def test_complete_run_covers_nine_populations(self):
        rec = record()
        if rec.get("partial_run"):
            self.skipTest(f"development run on {rec['faces_run']}")
        self.assertEqual(sorted(rec["faces_run"]), sorted(CB.NINE))
        for name in ("datasets.csv", "ablation.csv", "seed_spread.csv", "intervals_by_population.csv", "intervals_households.json",
                     "carriers_in_anchoring_layer.csv", "history_length.csv", "bdg2_near_zero_sensitivity.csv", "training_curves.csv",
                     "example_window_cambridge.csv", "example_window_cambridge.json", "cost_per_window.csv", "anchoring_gain_deciles.csv",
                     "component_contributions.csv", "intervals_winkler_contrasts.csv", "recipe_history.csv", "constants.csv",
                     "contrast_ii_blocks.csv", "contrast_ii_control.csv", "robustness.csv"):
            self.assertTrue((RESULTS / name).exists(), name)


FORBIDDEN = re.compile(r"^ANKYRA|F1-off|^F0$|^F1$|ANKYRA 2|ANKYRA-")          # a comparison with ANKYRA or one of its versions
BUDGET_FAMILIES = ("DLinear", "iTransformer", "LSTM", "PatchTST")
HANDOVER_CONTRASTS = ("C1 AW vs Ah", "C2 AW vs AM", "C3 AM vs Ah", "C4 Ah vs A0", "AW vs A0", "AM vs A0")
ADDED_ABLATIONS = ("fixed division (history level and daily path, no handover)", "fixed half handover weight",
                   "one whole-window combination weight of the history side and the carrier",
                   "half-and-half average of the history side and the carrier")
ADDED_COMPONENTS = ("block-wise weights instead of one whole-window weight", "block-wise weights against a half-and-half average")
CARRIER_PEAK = "same readout on the carrier alone"
ADDED_FILES = ("training_budget_sensitivity.csv", "handover_granularity.csv", "lead_weeks.csv", "within_day_trust.csv",
               "generic_combinations.csv", "level_candidate_hit_rate.csv", "long_context_baselines.csv", "energy_level_readout.csv")


def added_record(face):
    """The record of the files added on 19 October 2026 for one population (compare_baselines.json, extra.populations)."""
    return ((record().get("extra") or {}).get("populations") or {}).get(face) or {}


def face_of(name):
    return {v: k for k, v in NAMES.items()}[name]


@unittest.skipUnless(HAVE_RESULTS, "no complete run in the result directory (run evaluation/compare_baselines.py first)")
class AddedFileTests(unittest.TestCase):
    """The files and rows added on 19 October 2026 (``compare_baselines.py --supplement``): columns, labels, flags and the
    gates of the fields cache and of the handover rebuild.  Each test skips while its file has not been written."""

    def need(self, name):
        if not (RESULTS / name).exists():
            self.skipTest(f"no {name} in the result directory (run evaluation/compare_baselines.py --supplement)")

    def test_labels_are_the_evaluation_codes(self):
        import extra_tables as XT
        self.assertEqual(XT.ADDED_ABLATIONS, ADDED_ABLATIONS)
        self.assertEqual(tuple(XT.COMPONENT[k] for k in XT.ADDED_ABLATIONS if k in XT.COMPONENT), ADDED_COMPONENTS)
        self.assertEqual(XT.CARRIER_PEAK, CARRIER_PEAK)
        self.assertEqual(tuple(c for c, _, _ in XT.HANDOVER_CONTRASTS), HANDOVER_CONTRASTS)
        for lab in ADDED_ABLATIONS + ADDED_COMPONENTS + (CARRIER_PEAK,) + HANDOVER_CONTRASTS:
            self.assertIsNone(FORBIDDEN.search(lab), lab)

    def test_training_budget_sensitivity(self):
        self.need("training_budget_sensitivity.csv")
        import budget_tables as BT
        head = header("training_budget_sensitivity.csv")
        self.assertEqual(head, BT.FIELDS)
        self.assertEqual(head[-1], "flag")
        if HAVE_RELEASE:                                                             # the first release's columns, plus error and flag
            self.assertEqual([c for c in head if c not in ("error", "flag")], published("training_budget_sensitivity.csv")[0])
        rs = rows("training_budget_sensitivity")
        self.assertEqual(tuple(BT.FAMILIES), BUDGET_FAMILIES)
        for s in sets_run():
            got = sorted((r["model"], r["error"]) for r in rs if r["set"] == s)
            self.assertEqual(got, sorted((m, e) for m in BUDGET_FAMILIES for e in ("hourly", "energy")), s)
        lite = {"ankyra-lite better", "baseline better", "not resolved", "identical forecasts"}
        for r in rs:
            for k in ("original", "relaxed"):
                self.assertIn(r[f"{k}_resolved"], lite, r)
                lo, hi = float(r[f"{k}_um_low"]), float(r[f"{k}_um_high"])
                want = "ankyra-lite better" if hi < 0 else ("baseline better" if lo > 0 else "not resolved")
                if r[f"{k}_resolved"] != "identical forecasts":
                    self.assertEqual(r[f"{k}_resolved"], want, (r["set"], r["model"], r["error"], k))
            self.assertIn(r["relaxed_vs_original_resolved"], {"relaxed better", "original better", "not resolved", "identical forecasts"})
            self.assertIn(r["subset"], {"late"})
        bp = {(r["set"], r["model"]): r for r in rows("benchmark_pairwise")}
        for r in [x for x in rs if x["error"] == "hourly"]:                          # the original budget is the main comparison
            ref = bp[(r["set"], r["model"])]
            for a, b in (("original_log_ratio", "log_ratio"), ("original_um_low", "um_low"), ("original_um_high", "um_high")):
                self.assertAlmostEqual(float(r[a]), float(ref[b]), places=6, msg=(r["set"], r["model"], a))

    def test_handover_granularity_and_the_gates(self):
        self.need("handover_granularity.csv")
        head = header("handover_granularity.csv")
        self.assertEqual(head, ["set", "tier", "contrast", "log_ratio", "um_low", "um_high", "improvement_pct", "pooled_mse_ratio", "flag"])
        if HAVE_RELEASE:
            self.assertEqual(head, published("handover_granularity.csv")[0] + ["flag"])
        rs = rows("handover_granularity")
        others = [f"{a} vs {o}" for a in ("A0", "Ah", "AM", "AW") for o in ("carrier alone", "TimesFM", "fixed division")]
        weekly = [f"{c}, week {w}" for c in HANDOVER_CONTRASTS[:2] for w in range(1, 5)]
        for s in sets_run():
            got = [r["contrast"] for r in rs if r["set"] == s]
            self.assertEqual(sorted(got), sorted(list(HANDOVER_CONTRASTS) + others + weekly), s)
        for r in rs:
            lr = float(r["log_ratio"])
            self.assertAlmostEqual(float(r["improvement_pct"]), 100 * (1 - np.exp(lr)), delta=2e-3, msg=(r["set"], r["contrast"]))
            self.assertGreater(float(r["pooled_mse_ratio"]), 0.0)
        for s in sets_run():                                                         # the gates of the population's record
            x = added_record(face_of(s))
            g = x.get("fields_gate") or {}
            self.assertTrue(g.get("pass") and g.get("fields_pass"), (s, g))            # the product bit for bit, the rebuilt quantities
            self.assertTrue(g.get("traj_equal") and g.get("level_equal") and g.get("state_flags_equal"), (s, g))
            f = g.get("fields") or {}
            self.assertEqual(f.get("rebuilt_mismatches"), 0, s)                       # AW from the rebuilt pairs == lead_week_weights, ...
            self.assertTrue(f.get("fields_on_every_non_state_window"), s)
            self.assertEqual(f.get("rebuilt_reproduce_the_call"), f.get("non_state_windows"), s)
            h = (x.get("handover") or {}).get("gate") or {}
            self.assertTrue(h.get("aw_equals_the_product") and h.get("daily_means_reproduced"), (s, h))
            self.assertLess(h.get("m_h_identity_max_rel_diff", 1.0), 1e-9, s)

    def test_lead_weeks(self):
        self.need("lead_weeks.csv")
        head = header("lead_weeks.csv")
        pairs = [f"{a}_vs_{b}" for a in ("ankyra_lite", "fixed_division") for b in ("carrier", "timesfm")]
        self.assertEqual(head, ["set", "tier", "week", "days", "windows", "units"] + [f"{p}_{k}" for p in pairs for k in ("log_ratio", "um_low", "um_high")]
                         + ["mean_weight_on_carrier", "flag"])
        rs = rows("lead_weeks")
        for s in sets_run():
            mine = [r for r in rs if r["set"] == s]
            self.assertEqual([(r["week"], r["days"]) for r in mine], [("1", "1-7"), ("2", "8-14"), ("3", "15-21"), ("4", "22-31")], s)
            alpha = (added_record(face_of(s)).get("handover") or {}).get("alpha_week_mean")
            for r in mine:
                a = float(r["mean_weight_on_carrier"])
                self.assertTrue(0.125 <= a <= 0.875, (s, r["week"], a))              # (n lambda + 1) / (n + 2) with n <= 6 pairs, or 1/2
                if alpha:
                    self.assertAlmostEqual(a, alpha[int(r["week"]) - 1], places=4, msg=(s, r["week"]))
                for p in pairs:
                    self.assertTrue(np.isfinite(float(r[f"{p}_log_ratio"])), (s, r["week"], p))

    def test_within_day_trust(self):
        self.need("within_day_trust.csv")
        self.assertEqual(header("within_day_trust.csv"), ["set", "tier", "windows", "state_windows", "anchored_windows", "mean_trust", "median_trust",
                                                          "share_trust_positive", "share_analog_kept", "mean_triples", "share_three_triples", "flag"])
        rs = {r["set"]: r for r in rows("within_day_trust")}
        rec = record()
        for s in sets_run():
            r = rs[s]
            p = rec["populations"][face_of(s)]
            self.assertEqual(int(r["windows"]), int(p["windows"]), s)
            self.assertEqual(int(r["windows"]), int(r["state_windows"]) + int(r["anchored_windows"]), s)
            self.assertEqual(int(r["state_windows"]), int(p["product"]["state_windows_seasonal_naive"]), s)
            self.assertTrue(0.0 <= float(r["median_trust"]) <= 0.5 and 0.0 <= float(r["mean_trust"]) <= 0.5, s)   # the trust is capped at one half
            for k in ("share_trust_positive", "share_analog_kept", "share_three_triples"):
                self.assertTrue(0.0 <= float(r[k]) <= 1.0, (s, k))
            self.assertTrue(0.0 <= float(r["mean_triples"]) <= 3.0, s)

    def test_added_ablation_component_and_peak_rows(self):
        if not (RESULTS / "handover_granularity.csv").exists():
            self.skipTest("the added rows are written with handover_granularity.csv (run evaluation/compare_baselines.py --supplement)")
        ab = {(r["set"], r["ablation"], r["error"]) for r in rows("ablation")}
        cc = {(r["set"], r["component"], r["error"]) for r in rows("component_contributions")}
        pk = {(r["set"], r["comparator"]) for r in rows("peak_readout")}
        for s in sets_run():
            for err in ("hourly", "energy"):
                for lab in ADDED_ABLATIONS:
                    self.assertIn((s, lab, err), ab, (s, lab, err))
                for lab in ADDED_COMPONENTS:
                    self.assertIn((s, lab, err), cc, (s, lab, err))
            self.assertIn((s, CARRIER_PEAK), pk, s)
        for name, key in (("ablation", "ablation"), ("component_contributions", "component"), ("peak_readout", "comparator")):
            seen = [(r["set"], r[key]) + ((r["error"],) if "error" in r else ()) for r in rows(name)]
            self.assertEqual(len(seen), len(set(seen)), name)                        # each added row once

    def test_no_forbidden_label_and_the_flags_in_the_added_files(self):
        written = [n for n in ADDED_FILES if (RESULTS / n).exists()]
        if not written:
            self.skipTest("none of the added files has been written")
        want = {p["name"]: (p["flag"] or "") for p in record()["populations"].values()}
        labels = ("set", "model", "model_label", "comparator", "ablation", "carrier", "contrast", "arm", "component", "first_arm",
                  "second_arm", "best", "candidates")
        for name in written:
            with open(RESULTS / name, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    for k in labels:
                        self.assertIsNone(FORBIDDEN.search(r.get(k) or ""), (name, k, r.get(k)))
                    self.assertEqual(r["flag"], want.get(r["set"], ""), (name, r["set"]))
        for name in ("ablation", "component_contributions", "peak_readout"):
            for r in rows(name):
                if r.get("ablation") in ADDED_ABLATIONS or r.get("component") in ADDED_COMPONENTS or r.get("comparator") == CARRIER_PEAK:
                    self.assertEqual(r["flag"], want.get(r["set"], ""), (name, r["set"]))


if __name__ == "__main__":
    unittest.main()
