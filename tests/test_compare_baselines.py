"""Checks of the baseline comparison (evaluation/compare_baselines.py and its helpers), of the checkpoint export
(tools/export_checkpoints.py) and of the files the comparison wrote.

The comparator, the read-only guard, the path scrubbing, the products mappings (the released span-return recipe and its
from-scratch stage), the p-value format, the re-anchoring modes, the corruptions of the robustness test, the error
blocks and the recipe history are tested directly; the result files are checked for the release's columns, the
scoring-code check, the forecaster set (no ANKYRA version; 21 baselines including LSTM-T-KAN-M as the only LSTM-T-KAN
model; 22 forecasters), the flags (CINELDI: LSTM-T-KAN-M from its corrected-store retrain), the TiDE label, the new files and the
absence of absolute paths.  Nothing is recomputed (run the script first).

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
        self.assertEqual(raw["label"], "ankyra-lite 0.1.0")
        self.assertEqual(raw["recipe"], "span return (R)")
        self.assertEqual(raw["checkpoint"], "final.pt")
        self.assertEqual(raw["replacements"], {})
        self.assertEqual(sorted(raw["faces"]), sorted(CB.NINE))
        for f, e in raw["faces"].items():
            self.assertEqual(e["store"], "corrected" if f in ("drammen", "cineldi") else "original", f)
            self.assertTrue(e["npz"].endswith("/arms_R_ens.npz"), f)
            self.assertEqual([Path(x).name for x in e["run_dirs"]], ["R_s0", "R_s1", "R_s2"], f)
            self.assertEqual([Path(x).name for x in e["seed_npz"]], ["arms_R_s0.npz", "arms_R_s1.npz", "arms_R_s2.npz"], f)
            self.assertEqual([Path(x).name for x in e["start_run_dirs"]], ["P3s_s0", "P3s_s1", "P3s_s2"], f)
            self.assertEqual(sorted(e["recipe_variants"]), ["F", "P3s", "V"], f)
            for path in [e["npz"], e["prep"], e["span_plan"]] + e["run_dirs"] + e["seed_npz"] + e["start_run_dirs"] + list(e["recipe_variants"].values()):
                self.assertFalse(re.match(r"^[A-Za-z]:|^/|^\\\\", path), path)          # relative to the research root
            if f in ("drammen", "cineldi"):
                self.assertIn("/dc/", e["npz"], f)                                    # the corrected-store retrain
        self.assertTrue(raw["faces"]["cineldi"]["prep"].startswith("analysis_outputs/smallnet_sr_formal_v1/dc/v1rec/"))
        p = CB.load_products()
        self.assertEqual(p["faces"]["oslo"]["checkpoint"], "final.pt")
        self.assertEqual(p["faces"]["oslo"]["start_checkpoint"], "best.pt")
        self.assertEqual(len(p["faces"]["oslo"]["seed_paths"]), 3)
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
        self.assertNotEqual(LR.cache_file("x", "oslo", "s0"), LR.cache_file("x", "oslo", "s0", 8))

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
                    self.assertTrue(v["seed_products_exist"] and v["start_runs_exist"] and all(v["variants_exist"].values()), (name, k))


@unittest.skipUnless(HAVE_RESEARCH, NO_RESEARCH)
class ExportTests(unittest.TestCase):
    """tools/export_checkpoints.py into a temporary directory (one population, no data store read)."""

    def test_export_one_population(self):
        spec = importlib.util.spec_from_file_location("export_checkpoints", REPO / "tools" / "export_checkpoints.py")
        EX = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(EX)
        from ankyra_lite import load_carrier
        out = Path(tempfile.mkdtemp(prefix="ankyra_lite_export_test_"))
        try:
            files = EX.export(out, CB.load_products(), faces=("oslo",), log=lambda *a: None)
            self.assertEqual([f["file"] for f in files], ["oslo/seed0.pt", "oslo/seed1.pt", "oslo/seed2.pt"])
            for f in files:
                self.assertEqual(EX.sha256(out / f["file"]), f["sha256"])
                self.assertEqual(f["parameters"], 61613)
                self.assertEqual(f["training_cut_date"], "2019-03-29")
                self.assertTrue(f["source_run"].startswith("analysis_outputs/smallnet_sr_formal_v1/B/oslo/R_s"))
                self.assertFalse(re.search(r"[A-Za-z]:[\\/]", json.dumps(f)), f)
                load_carrier(out / f["file"])                                    # strict
        finally:
            shutil.rmtree(out, ignore_errors=True)


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
        if str(record().get("recipe", "")).startswith("span return"):
            labs += ["single seed 0 (its carrier anchored alone)", "recipe: 8 continuation epochs, validation set held out, no span return (anchored)"]
        for s in sets_run():
            for lab in labs:
                for err in ("hourly", "energy"):
                    self.assertIn((s, lab, err), ab, (s, lab, err))

    def test_training_curves_have_both_stages(self):
        if not (RESULTS / "training_curves.csv").exists():
            self.skipTest("no training_curves.csv")
        tc = rows("training_curves")
        stages = {(r["set"], r["stage"]) for r in tc}
        cont = str(record().get("recipe", "")).startswith("span return")
        for s in sets_run():
            self.assertIn((s, "from scratch (early stopping)"), stages, s)
            if cont:
                self.assertIn((s, "span return (fixed epochs)"), stages, s)
                sel = [r for r in tc if r["set"] == s and r["stage"] == "span return (fixed epochs)" and r["selected"] == "True"]
                self.assertEqual(sorted(int(r["epoch"]) for r in sel), [8, 8, 8], s)

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


if __name__ == "__main__":
    unittest.main()
