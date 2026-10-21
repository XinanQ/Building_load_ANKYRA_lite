"""Checks of the external-test table (evaluation/external_tables.py, results/external_tests.csv, docs/EXTERNAL_TESTS.md).

Without the study's research folder: the file's columns, the forecaster names it may hold, no absolute path in the
file, the script or the page, and no comparison with ANKYRA itself or its versions. With the research folder
(ANKYRA_LITE_RESEARCH_ROOT, default: the sibling folder HKAD_V3; skipped when its score records are absent): the file
equals a fresh build from the score records, every number is the record field that its ``source`` column cites, the
primary table of docs/EXTERNAL_TESTS.md matches the file, the sealed Helsinki set is never read, and no contrast of
the scorer's other output files (the comparisons kept in the study's research record only) appears in the file.

Run from the repository root:  python -m unittest tests.test_external_tables -v
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import csv
import importlib.util
import json
import math
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("external_tables", REPO / "evaluation" / "external_tables.py")
ET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ET)

CSV_PATH = REPO / "results" / ET.OUT_NAME
DOC = REPO / "docs" / "EXTERNAL_TESTS.md"
# the table follows evaluation/products.json: the 0.2.0 re-run (score records under lite_upgrade_v2/external/score, first
# forecasters FIRST_V2, own configurations OWN_ARMS_V2) or the frozen 0.1 external test
V2 = ET.version_of_products() == "0.2"
FIRST = ET.FIRST_V2 if V2 else ET.FIRST
OWN = ET.OWN_ARMS_V2 if V2 else ET.OWN_ARMS
LABELS = ET.LABEL_V2 if V2 else ET.LABEL
PRIMARY_FIRST = "Rw_ens6_c" if V2 else "R_ens6"
BUILD = ET.build_v2 if V2 else ET.build
LOAD = ET.load_v2 if V2 else ET.load
SCORE_DIR = ET.score_dir_v2() if V2 else ET.score_dir()
HAVE_RECORDS = ET.available_v2() if V2 else ET.available()
NO_RECORDS = f"the score records of the external tests are not available (set ANKYRA_LITE_RESEARCH_ROOT; looked in {SCORE_DIR})"
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\Users|/Users/|/home/")
# designations of ANKYRA's own forecasters (the anchoring layer copied from 2.2.1 may be named; the forecasters may not)
LEFT_OUT = re.compile(r"ANKYRA[ -]?(?:1\.|2\.0|2\.1\b|2\.2(?!\.1)|X\b)|\bB[12]\b|^F1-off$|^F[01]$")


def csv_rows():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _leaves(node, path=()):
    """(path, dict) of every contrast of a record (a dict with a point estimate and its two forecasters)."""
    if isinstance(node, dict):
        if "pct" in node and "a" in node and "b" in node:
            yield path, node
            return
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, path + (i,))


def _resolve(record, parts):
    """ET.resolve on a citation split at " / ", where a record key may itself hold " / " (e.g. "layer on / off on
    Rw6 (...)"): the shortest run of parts that names a key at each level."""
    if not parts:
        return record
    for n in range(1, len(parts) + 1):
        try:
            node = ET.resolve(record, [" / ".join(parts[:n])])
        except (KeyError, TypeError):
            continue
        try:
            return _resolve(node, parts[n:])
        except (KeyError, TypeError):
            continue
    raise KeyError(" / ".join(parts))


def _same(a: str, b) -> bool:
    if b is None or (isinstance(b, float) and not math.isfinite(b)):
        return a == ""
    if isinstance(b, bool):
        return a == str(b)
    if isinstance(b, (int, float)):
        return a != "" and float(a) == float(b)
    return a == str(b)


class TestFileWithoutRecords(unittest.TestCase):
    def setUp(self):
        if not CSV_PATH.exists():
            self.skipTest(f"{CSV_PATH.name} has not been written")

    def test_columns_and_test_sets(self):
        rows = csv_rows()
        with open(CSV_PATH, encoding="utf-8") as f:
            self.assertEqual(next(csv.reader(f)), list(ET.COLUMNS))
        self.assertEqual([f for f in dict.fromkeys(r["test_set_id"] for r in rows) if f], list(ET.FACES))
        self.assertTrue(all(r["test_set_id"] or r["block"] == "day-1 rule (design 3.3)" for r in rows))
        self.assertTrue(rows)

    def test_only_ankyra_lite_its_own_configurations_and_the_baselines(self):
        allowed_first = set(FIRST.values()) | {""}
        allowed_model = set(ET.BASELINES) | set(OWN) | {""}
        for r in csv_rows():
            self.assertIn(r["first"], allowed_first, r["source"])
            self.assertIn(r["model"], allowed_model, r["source"])
            self.assertEqual(r["model_label"], LABELS.get(r["model"], ""), r["source"])
            self.assertTrue(r["first"] or r["model"], r["source"])
            for k in ("first", "model", "model_label", "source", "block", "subgroup", "pretraining_overlap", "identity", "note"):
                self.assertIsNone(LEFT_OUT.search(r[k]), (k, r[k]))
            if r["model"] == "TiDE":
                self.assertEqual(r["model_label"], "Linear-context (TiDE implementation)")

    def test_each_primary_contrast_once_per_test_set(self):
        rows = csv_rows()
        for f in ET.FACES:
            got = [r["model"] for r in rows if r["test_set_id"] == f and r["block"] == "primary"]
            self.assertEqual(sorted(got), sorted(ET.PRIMARY), f)
            for r in rows:
                if r["test_set_id"] == f and r["block"] == "primary":
                    self.assertEqual((r["first"], r["readout"], r["seeds"]), (FIRST[PRIMARY_FIRST], "hourly", "6"))
                    self.assertIn(r["label"], ("resolved better", "not resolved", "resolved worse"))

    def test_helsinki_is_not_scored_here(self):
        self.assertNotIn("helsinki", " ".join(r["test_set_id"] for r in csv_rows()).lower())
        for face in ET.SEALED:
            with self.assertRaises(ET.Unexpected):
                ET.load(face)
            with self.assertRaises(ET.Unexpected):
                ET.load_v2(face)

    def test_no_absolute_path_in_the_file_the_script_or_the_page(self):
        for p in (CSV_PATH, REPO / "evaluation" / "external_tables.py", DOC, REPO / "figures" / "README.md"):
            if p.exists():
                text = p.read_text(encoding="utf-8")
                m = ABS_PATH.search(text)
                self.assertIsNone(m, (p.name, m and text[max(0, m.start() - 40):m.end() + 40]))

    def test_the_page_names_no_forecaster_of_ankyra(self):
        if not DOC.exists():
            self.skipTest("docs/EXTERNAL_TESTS.md has not been written")
        text = DOC.read_text(encoding="utf-8")
        for line in text.splitlines():
            self.assertIsNone(re.search(r"ANKYRA[ -]?(?:1\.|2\.0|2\.1\b|2\.2(?!\.1)|X\b)|\bB[12]\b", line), line)

    def test_clean_and_cite(self):
        self.assertEqual(ET.clean("x (favours the TimesFM-based comparators, including Q)"), "x (favours the TimesFM-based comparators)")
        drive = "C" + ":"                                        # not written as one string: the file is scanned for paths
        self.assertEqual(ET.clean("see " + drive + "\\Users\\someone\\file.json"), "see <path>")
        self.assertEqual(ET.clean("see " + drive + "/Users/someone/file.json; next"), "see <path>; next")
        self.assertEqual(ET.clean("first read (secondary face, small sample)"), "first read (secondary test set, small sample)")
        self.assertEqual(ET.cite("mean unit rank (whatever is said here)"), "mean unit rank")
        self.assertEqual(ET.resolve({"mean unit rank (x)": {"a": 1}}, ["mean unit rank", "a"]), 1)
        with self.assertRaises(ET.Unexpected):
            ET._model("some other forecaster")


@unittest.skipUnless(HAVE_RECORDS, NO_RECORDS)
class TestFileAgainstRecords(unittest.TestCase):
    def test_file_equals_a_fresh_build(self):
        self.assertTrue(CSV_PATH.exists(), "run python -B evaluation/external_tables.py")
        self.assertEqual(CSV_PATH.read_text(encoding="utf-8"), ET.to_csv(BUILD()))

    def test_every_number_is_the_cited_field(self):
        recs = {f: LOAD(f) for f in ET.FACES}
        rule, rule_file = ET.day1_rule() if V2 else (None, "")
        for r in csv_rows():
            name, path = r["source"].split(": ", 1)
            if V2 and r["block"] == "day-1 rule (design 3.3)":
                self.assertEqual(name, rule_file)
                allowed = rule.get(path)
                self.assertEqual(r["label"], "not evaluated" if allowed is None else "claim allowed" if allowed else
                                 "no short-term gap reduction claim")
                continue
            self.assertEqual(name, f"score_{r['test_set_id']}.json" + (" (0.2.0)" if V2 else ""))
            node = _resolve(recs[r["test_set_id"]], path.split(" / "))
            if r["block"] == "mean unit rank":
                self.assertTrue(_same(r["mean_unit_rank"], node), r["source"])
                continue
            if V2 and r["block"] == "day-1 gap to TimesFM 2.5":            # the day-1 rule's rows of each test set
                iv = node.get("welch_95_pct") or [None, None]
                self.assertTrue(_same(r["improvement_pct"], node.get("pct")) and _same(r["windows"], node.get("windows")), r["source"])
                self.assertTrue(_same(r["interval_low_pct"], iv[0]) and _same(r["interval_high_pct"], iv[1]), r["source"])
                self.assertEqual(r["label"], node.get("label", "not computed"), r["source"])
                continue
            self.assertEqual(node["b"], r["model"], r["source"])
            self.assertEqual(FIRST[node["a"]], r["first"], r["source"])
            for col, key in (("improvement_pct", "pct"), ("windows", "windows"), ("units", "units"), ("months", "months"),
                             ("seeds_positive", "seeds_positive"), ("welch_df", "welch_df")):
                self.assertTrue(_same(r[col], node.get(key)), (r["source"], col))
            iv = node.get("welch_95_pct") or node.get("unit_month_interval_pct") or [None, None]
            self.assertTrue(_same(r["interval_low_pct"], iv[0]) and _same(r["interval_high_pct"], iv[1]), r["source"])
            self.assertEqual(r["label"], node.get("label", "not computed"), r["source"])

    def test_no_contrast_of_the_other_score_files_appears(self):
        """The scorer writes the comparisons kept in the study's research record only to separate files next to each
        record; none of their (estimate, interval) triples may appear in the file."""
        mine = {(r["improvement_pct"], r["interval_low_pct"], r["interval_high_pct"]) for r in csv_rows() if r["improvement_pct"]}
        mine_pct = {r["improvement_pct"] for r in csv_rows() if r["improvement_pct"]}
        others = [p for f in ET.FACES for p in SCORE_DIR.glob(f"score_{f}_*.json")]
        for p in others:
            rec = json.loads(p.read_text(encoding="utf-8"))
            for path, x in _leaves(rec):
                iv = x.get("welch_95_pct") or x.get("unit_month_interval_pct") or [None, None]
                triple = (ET.num(x["pct"]), ET.num(iv[0]), ET.num(iv[1]))
                self.assertNotIn(triple, mine, (p.name, path))
                self.assertNotIn(ET.num(x["pct"]), mine_pct, (p.name, path))

    def test_the_page_shows_the_primary_readouts_of_the_file(self):
        if not DOC.exists():
            self.skipTest("docs/EXTERNAL_TESTS.md has not been written")
        text = DOC.read_text(encoding="utf-8")
        for r in csv_rows():
            if r["block"] == "primary":
                self.assertIn(ET.cell(r), text, r["source"])
        md = ET.markdown_v2(BUILD()) if V2 else ET.markdown(BUILD())
        self.assertIn(md.split("\n\n### ")[0].strip(), text)

    def test_counts_used_on_the_page(self):
        for f in ET.FACES:
            x = ET.facts(f, ET.load(f))                       # the frozen records (the 0.2.0 re-run reads the same windows)
            self.assertEqual(x["failed"], 0, f)
            self.assertEqual(sum(x["month_counts"].values()), x["windows"], f)


if __name__ == "__main__":
    unittest.main()
