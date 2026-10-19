"""The external tests of ankyra-lite as a table: ``results/external_tests.csv`` and the tables of
``docs/EXTERNAL_TESTS.md``, read from the score records of the study's frozen external-test run.

    python -B evaluation/external_tables.py              # writes results/external_tests.csv
    python -B evaluation/external_tables.py --check      # compares results/external_tests.csv with the records (exit 1 if they differ)
    python -B evaluation/external_tables.py --markdown   # prints the tables of docs/EXTERNAL_TESTS.md

Source.  One score record per test set, ``<research root>/analysis_outputs/lite_external_v1/score/score_<set>.json``,
each written once by the study's scorer after the run was frozen (19 October 2026, 07:45 UTC). The research root is
ANKYRA_LITE_RESEARCH_ROOT, by default the sibling folder ``HKAD_V3`` of this repository. Nothing is recomputed: every
number of the CSV file is a field of one of these records, and the column ``source`` names the record and the field.
The script reads only these four files; it never opens the scorer's other output files, never reads data or forecasts,
and writes only into ``results/`` of this repository (never into the research root).

Test sets.  ``lcl``, ``unicon`` and ``hkust`` (first read of this candidate on a previously read population) and
``goiener_oot`` (same-source out-of-time supplementary check). The Helsinki retrospective set was trained, anchored and
sealed in the same run and is not scored yet; it is not read here, and the script refuses it.

What is written.  Every contrast of the records whose two forecasters are ankyra-lite (six seeds, or the three-seed
release configuration), the baselines of the comparison (allowed ids: ``BASELINES``) or ankyra-lite's own reduced
configurations (``OWN_ARMS``): the primary readouts, the other readouts, the release configuration, the other saved
baselines, the descriptive contrasts and the subgroups, plus the mean unit ranks. A forecaster id outside these lists
stops the script (``Unexpected``), so a record that held any other forecaster would not be copied. The records' text
fields are copied only where listed (identity, readout, pretraining note), with the paths and the names of left-out
forecasters removed (``clean``); no other free text is copied.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse                                        # noqa: E402
import csv                                             # noqa: E402
import io                                              # noqa: E402
import json                                            # noqa: E402
import math                                            # noqa: E402
import os                                              # noqa: E402
import re                                              # noqa: E402
from pathlib import Path                               # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
OUT_NAME = "external_tests.csv"


def research_root() -> Path:
    """ANKYRA_LITE_RESEARCH_ROOT, else the sibling folder HKAD_V3 of the repository."""
    v = os.environ.get("ANKYRA_LITE_RESEARCH_ROOT")
    return Path(v).resolve() if v else (REPO.parent / "HKAD_V3").resolve()


def score_dir(root: Path | None = None) -> Path:
    return (root or research_root()) / "analysis_outputs" / "lite_external_v1" / "score"


FACES = ("lcl", "unicon", "hkust", "goiener_oot")          # scored once each; the order of every table
SEALED = ("helsinki_retro",)                               # trained, anchored and sealed; not scored (never read here)

FIRST = {"R_ens6": "ankyra-lite (6 seeds)", "R_ens3": "ankyra-lite (3 seeds, release configuration)"}
# ankyra-lite's own reduced configurations, as named in README.md (Terms used on this page) and EVALUATION.md
OWN_ARMS = {"Ns": "carrier-free control with the state rule",
            "N": "carrier-free control",
            "H": "history-only configuration",
            "F_ens3": "plain continuation F (3 seeds)",
            "P3s_ens6": "from-scratch stage alone (6 seeds)"}
# the baselines of the comparison: model id -> model_label of results/benchmark_pairwise.csv
BASELINES = {"TimesFM": "TimesFM", "TimesFM-X": "TimesFM-X", "Chronos-2": "Chronos-2", "Chronos-2-X": "Chronos-2-X",
             "RIDGE-L": "Ridge (per unit)", "GBT": "GBT (zero-shot)", "GBT-T": "GBT-T", "Holt-Winters": "Holt-Winters",
             "MSTL": "MSTL", "SN-day": "Seasonal naive (day)", "SN-week": "Seasonal naive (week)",
             "SP-profile": "Previous-month profile", "LYR-profile": "Last-year profile", "WeekProf4": "4-week profile",
             "DLinear": "DLinear", "iTransformer": "iTransformer", "iTransformer-X": "iTransformer-X", "LSTM": "LSTM",
             "PatchTST": "PatchTST", "TiDE": "Linear-context (TiDE implementation)"}
LABEL = {**BASELINES, **OWN_ARMS, **FIRST}

PRIMARY = ("TimesFM", "SN-week", "Ns")                     # the three pre-specified primary contrasts of every test set
READOUT = {"hourly": "hourly", "monthly (independent readout)": "monthly energy",
           "monthly (curve sum)": "monthly energy (trajectory sum)", "peak": "peak",
           "lead day 1": "forecast day 1", "lead days 2-7": "forecast days 2-7", "lead days 8-31": "forecast days 8-31"}
READOUT_ORDER = tuple(READOUT.values())
SECONDARY = {"candidate (6 seeds), other readouts": "other readouts",
             "release configuration (R_ens3, seeds 0-2)": "release configuration",
             "other saved comparators (hourly; intersection windows, n reported)": "other saved baselines"}
DESCRIPTIVE = {"R_ens3 vs F_ens3": "span return vs plain continuation",
               "R_ens6 vs P3s_ens6": "continuation and span return vs from-scratch stage",
               "candidate (6 seeds) vs H": "vs history-only configuration",
               "candidate (6 seeds) vs N": "vs carrier-free control"}
DESCRIPTIVE_HEAD = {"span return vs plain continuation": "span return vs plain continuation (R vs F, 3 seeds)",
                    "continuation and span return vs from-scratch stage":
                        "continuation and span return vs from-scratch stage (6 seeds; unit × month interval only)",
                    "vs history-only configuration": "ankyra-lite vs history-only configuration (6 seeds)",
                    "vs carrier-free control": "ankyra-lite vs carrier-free control (6 seeds)"}
RANK_KEY = "mean unit rank"                                # a section key of the records that starts with these words

COLUMNS = ("test_set", "test_set_id", "identity", "role", "block", "subgroup", "first", "model", "model_label", "readout",
           "windows", "units", "months", "improvement_pct", "interval_low_pct", "interval_high_pct", "interval",
           "seeds", "seeds_positive", "welch_df", "label", "unit_month_low_pct", "unit_month_high_pct",
           "mean_unit_rank", "candidate_position", "models_ranked", "pretraining_overlap", "trained_labels_to_C",
           "note", "source")


class Unexpected(Exception):
    """A record holds a forecaster, section or readout that this script does not know (nothing is written)."""


# ----------------------------------------------------------------------------- text from the records
_PATH = re.compile(r"[A-Za-z]:[\\/][^'\"\s;,)]*|\\\\[^'\"\s;,)]+|(?<![\w.])/(?:Users|home)/[^'\"\s;,)]*")
_INCLUDING = re.compile(r",\s*including [^)]*")          # "(favours ..., including <a forecaster left out here>)": clause dropped


def clean(text) -> str:
    """A text field of a record as it may appear here: no absolute path and no name of a left-out forecaster."""
    if text is None:
        return ""
    t = _INCLUDING.sub("", str(text))
    t = _PATH.sub("<path>", t)
    t = re.sub(r"(?<![A-Za-z_])face(?![A-Za-z_])", "test set", t)                    # the records' internal word for a test set
    return re.sub(r"\s{2,}", " ", t).strip()


def cite(key: str) -> str:
    """A record key as cited in the ``source`` column. The key of the mean-rank section is cited by its first words
    (its parenthesis names the forecasters left out of the ranks)."""
    if key.startswith(RANK_KEY + " ("):
        return RANK_KEY
    return key


def resolve(record: dict, cited: list[str]):
    """The record field that a ``source`` citation names (the inverse of ``cite``)."""
    node = record
    for k in cited:
        if k in node:
            node = node[k]
            continue
        hits = [key for key in node if key.startswith(k + " (")]
        if len(hits) != 1:
            raise KeyError(" / ".join(cited))
        node = node[hits[0]]
    return node


def num(v) -> str:
    """A number as written to the CSV file: the record's float in full (repr), '' for a missing or undefined value."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "" if not math.isfinite(v) else repr(v)
    return str(v)


# ----------------------------------------------------------------------------- the records
def load(face: str, root: Path | None = None) -> dict:
    if face in SEALED:
        raise Unexpected(f"{face} is sealed and not scored; it is never read here")
    if face not in FACES:
        raise Unexpected(f"unknown test set {face!r}")
    p = score_dir(root) / f"score_{face}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("face") != face or rec.get("status") != "complete":
        raise Unexpected(f"{p.name}: face {rec.get('face')!r}, status {rec.get('status')!r}")
    return rec


def available(root: Path | None = None) -> bool:
    d = score_dir(root)
    return all((d / f"score_{f}.json").exists() for f in FACES)


def _contrasts(node, path):
    """(path, leaf) of every contrast of a record section (a dict with the two forecasters and a readout)."""
    if isinstance(node, dict):
        if "a" in node and "b" in node and "readout" in node:
            yield path, node
            return
        for k, v in node.items():
            yield from _contrasts(v, path + [k])


def _where(path):
    """(role, block, subgroup) of a contrast from its place in the record."""
    role = path[0]
    if role == "primary":
        return role, "primary", ""
    if role == "secondary":
        block = SECONDARY.get(path[1])
        if block is None:
            raise Unexpected(f"secondary section {path[1]!r}")
        return role, block, ""
    if role == "descriptive":
        if path[1] == "subgroups":
            i = path.index("primary_hourly")
            return role, "subgroup", ": ".join(path[2:i])
        block = next((v for k, v in DESCRIPTIVE.items() if path[1].startswith(k + " (")), None)
        if block is None:
            raise Unexpected(f"descriptive section {path[1]!r}")
        return role, block, ""
    raise Unexpected(f"section {role!r}")


def _model(b: str) -> str:
    if b in BASELINES or b in OWN_ARMS:
        return b
    raise Unexpected(f"forecaster {b!r} is not a baseline of the comparison or one of ankyra-lite's own configurations")


def _comparator_info(rec, b):
    c = rec.get("comparators", {}).get(b, {})
    trained = next((v for k, v in c.items() if k.startswith("trained_labels_to_C")), None)
    return clean(c.get("pretraining_overlap")), trained


def rows_of(face: str, rec: dict) -> list[dict]:
    out = []
    ident = clean(rec["identity"])
    for path, x in _contrasts({k: rec[k] for k in ("primary", "secondary", "descriptive")}, []):
        role, block, sub = _where(path)
        a, b = x["a"], _model(x["b"])
        if a not in FIRST:
            raise Unexpected(f"first forecaster {a!r}")
        rd = READOUT.get(x["readout"])
        if rd is None:
            raise Unexpected(f"readout {x['readout']!r}")
        over, trained = _comparator_info(rec, b)
        if "pretraining_overlap" in x:
            over = clean(x["pretraining_overlap"])
        if "trained_labels_to_C" in x:
            trained = x["trained_labels_to_C"]
        welch = x.get("welch_95_pct")
        um = x.get("unit_month_interval_pct") or [None, None]
        note = []
        if "status" in x:
            note.append(clean(x["status"]))
        if "pct" in x and (welch or um) and not all(isinstance(v, float) and math.isfinite(v) for v in (welch or um)):
            note.append("interval undefined in the record (too few clusters)")
        if welch:
            lo, hi, kind = welch[0], welch[1], f"Welch 95%: unit x month bootstrap and the spread over seeds (M = {x['M']})"
        elif "pct" in x:
            lo, hi, kind = um[0], um[1], "unit x month bootstrap 95% (no seed layer)"
        else:
            lo, hi, kind = None, None, ""
        out.append({"test_set": rec["label"], "test_set_id": face, "identity": ident, "role": role, "block": block, "subgroup": sub,
                    "first": FIRST[a], "model": b, "model_label": LABEL[b], "readout": rd,
                    "windows": num(x.get("windows")), "units": num(x.get("units")), "months": num(x.get("months")),
                    "improvement_pct": num(x.get("pct")), "interval_low_pct": num(lo), "interval_high_pct": num(hi),
                    "interval": kind, "seeds": num(x.get("M")) if welch else "", "seeds_positive": num(x.get("seeds_positive")),
                    "welch_df": num(x.get("welch_df")), "label": x.get("label", "not computed"),
                    "unit_month_low_pct": num(um[0]), "unit_month_high_pct": num(um[1]),
                    "mean_unit_rank": "", "candidate_position": "", "models_ranked": "",
                    "pretraining_overlap": over, "trained_labels_to_C": num(trained), "note": "; ".join(note),
                    "source": f"score_{face}.json: " + " / ".join(cite(k) for k in path)})
    rank_sec = next(k for k in rec["secondary"] if k.startswith(RANK_KEY))
    for variant, v in rec["secondary"][rank_sec].items():
        for m, r in v["mean_unit_rank"].items():
            first = m in FIRST
            if not first:
                _model(m)
            out.append({"test_set": rec["label"], "test_set_id": face, "identity": ident, "role": "secondary", "block": "mean unit rank",
                        "subgroup": variant, "first": FIRST[m] if first else "", "model": "" if first else m,
                        "model_label": "" if first else LABEL[m], "readout": "hourly",
                        "windows": num(v["windows"]), "units": num(v["units"]), "months": "",
                        **{k: "" for k in ("improvement_pct", "interval_low_pct", "interval_high_pct", "interval", "seeds",
                                           "seeds_positive", "welch_df", "label", "unit_month_low_pct", "unit_month_high_pct")},
                        "mean_unit_rank": num(r), "candidate_position": num(v["candidate_position"]) if first else "",
                        "models_ranked": num(v["models"]), "pretraining_overlap": "", "trained_labels_to_C": "", "note": "",
                        "source": f"score_{face}.json: secondary / {cite(rank_sec)} / {variant} / mean_unit_rank / {m}"})
    return out


def build(root: Path | None = None) -> list[dict]:
    rows = []
    for f in FACES:
        rows += rows_of(f, load(f, root))
    return rows


def to_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def facts(face: str, rec: dict) -> dict:
    """Counts of a record used in the text of docs/EXTERNAL_TESTS.md (no unit identifier is copied)."""
    w, sp = rec["windows"], rec["split"]
    zero_arms = sorted({e["arm"] for e in w.get("exact_zero_hourly_error", []) if e["arm"] in FIRST or e["arm"] in OWN_ARMS
                        or e["arm"] in BASELINES or re.fullmatch(r"(R|F)_s\d", e["arm"] or "")})
    return {"windows": w["scored"], "units": w["units"], "months": w["months"], "month_counts": w["month_counts"],
            "state_windows": len(w["state_windows"]), "units_excluded": len(w["units_excluded_mean_below_1e-6_kW"]),
            "windows_of_excluded_units": w["windows_of_excluded_units"],
            "exact_zero_windows": len({(e["building"], e["origin"]) for e in w.get("exact_zero_hourly_error", [])}),
            "exact_zero_arms": zero_arms, "target_hours_missing": w["target_hours_missing"],
            "target_hours_flagged": w["target_hours_flagged"], "windows_with_missing_target_hours": w["windows_with_missing_target_hours"],
            "C": sp["C_label"], "Cprime": sp["Cprime_label"], "v0": sp["v0_label"], "sigma": rec["sigma_Cprime"],
            "failed": rec["failures"]["failed_count"], "scored_utc": rec["utc"]}


# ----------------------------------------------------------------------------- the tables of docs/EXTERNAL_TESTS.md
def _f(v, d=2):
    s = f"{float(v):+.{d}f}"
    return s.replace("-", "−")


def cell(r, d=2) -> str:
    """'+2.68% [−1.22, +6.43]' with the label's mark (* resolved better, † resolved worse)."""
    if not r["improvement_pct"]:
        return "not computed"
    iv = f" [{_f(r['interval_low_pct'], d)}, {_f(r['interval_high_pct'], d)}]" if r["interval_low_pct"] else " [interval undefined]"
    mark = {"resolved better": " *", "resolved worse": " †"}.get(r["label"], "")
    return f"{_f(r['improvement_pct'], d)}%{iv}{mark}"


def _n(v) -> str:
    """A count with a thousands separator ('1,417')."""
    return f"{int(v):,}" if str(v).strip() else ""


def _pick(rows, **kw):
    hit = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return hit[0] if len(hit) == 1 else None


def markdown(rows) -> str:
    L = []
    L += ["### Primary readouts (hourly, six seeds)", "",
          "| Test set | Identity | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |",
          "|---|---|---|---|---|---|"]
    for f in FACES:
        ps = [_pick(rows, test_set_id=f, block="primary", model=m) for m in PRIMARY]
        L.append(f"| {ps[0]['test_set']} | {ps[0]['identity']} | {_n(ps[0]['windows'])} / {_n(ps[0]['units'])} / {_n(ps[0]['months'])} | "
                 + " | ".join(cell(p) for p in ps) + " |")
    L += [""]
    for f in FACES:
        name = next(r["test_set"] for r in rows if r["test_set_id"] == f)
        L += [f"### {name}", "", "| Readout | " + " | ".join(("vs TimesFM 2.5", "vs seasonal naive (week)",
                                                               "vs carrier-free control with the state rule")) + " |", "|---|---|---|---|"]
        for rd in READOUT_ORDER:
            cs = []
            for m in PRIMARY:
                blk = "primary" if rd == "hourly" else "other readouts"
                r = _pick(rows, test_set_id=f, block=blk, model=m, readout=rd)
                cs.append(cell(r) if r else "—")
            if any(c != "—" for c in cs):
                L.append(f"| {rd} (6 seeds) | " + " | ".join(cs) + " |")
        for rd in READOUT_ORDER:
            cs = [_pick(rows, test_set_id=f, block="release configuration", model=m, readout=rd) for m in PRIMARY]
            if any(cs):
                L.append(f"| {rd} (3 seeds) | " + " | ".join(cell(c) if c else "—" for c in cs) + " |")
        L += ["", "| Descriptive | " + " | ".join(DESCRIPTIVE_HEAD[b] for b in DESCRIPTIVE.values()) + " |",
              "|---|" + "---|" * len(DESCRIPTIVE)]
        for rd in READOUT_ORDER:
            cs = [_pick(rows, test_set_id=f, block=blk, readout=rd) for blk in DESCRIPTIVE.values()]
            if any(cs):
                L.append(f"| {rd} | " + " | ".join(cell(c) if c else "—" for c in cs) + " |")
        L += ["", "| Other saved baseline (hourly, 6 seeds) | Result | Training labels to C |", "|---|---|---|"]
        for r in [r for r in rows if r["test_set_id"] == f and r["block"] == "other saved baselines"]:
            L.append(f"| {r['model_label']} | {cell(r)} | {'yes' if r['trained_labels_to_C'] == 'True' else 'no'} |")
        ranks = [r for r in rows if r["test_set_id"] == f and r["block"] == "mean unit rank" and r["subgroup"].startswith("all comparators")]
        if ranks:
            me = next(r for r in ranks if r["first"])
            L += ["", f"Mean unit rank among {me['models_ranked']} forecasters ({_n(me['windows'])} windows, {_n(me['units'])} units): "
                  + ", ".join(f"{r['first'] or r['model_label']} {float(r['mean_unit_rank']):.2f}" for r in ranks)
                  + f"; ankyra-lite at position {me['candidate_position']}."]
        subs = [r for r in rows if r["test_set_id"] == f and r["block"] == "subgroup"]
        if subs:
            L += ["", "| Subgroup (hourly, 6 seeds) | Windows / units / months | vs TimesFM 2.5 | vs seasonal naive (week) | vs carrier-free control with the state rule |",
                  "|---|---|---|---|---|"]
            for sg in dict.fromkeys(r["subgroup"] for r in subs):
                cs = [_pick(rows, test_set_id=f, block="subgroup", subgroup=sg, model=m) for m in PRIMARY]
                n = cs[0]
                L.append(f"| {sg} | {_n(n['windows'])} / {_n(n['units'])} / {_n(n['months'])} | " + " | ".join(cell(c) if c else "—" for c in cs) + " |")
        L += [""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare results/external_tests.csv with the records")
    ap.add_argument("--markdown", action="store_true", help="print the tables of docs/EXTERNAL_TESTS.md")
    ap.add_argument("--out", default=None, help="output folder (default: results/)")
    a = ap.parse_args(argv)
    root = research_root()
    if not available(root):
        print("the score records of the external tests are not available (set ANKYRA_LITE_RESEARCH_ROOT)")
        return 2
    rows = build(root)
    text = to_csv(rows)
    out = Path(a.out).resolve() if a.out else RESULTS
    try:
        out.relative_to(root)
        raise SystemExit("the output folder lies in the research root, which is read-only here")
    except ValueError:
        pass
    if a.markdown:
        sys.stdout.flush()
        sys.stdout.buffer.write((markdown(rows) + "\n").encode("utf-8"))      # UTF-8 whatever the console's code page
        sys.stdout.buffer.flush()
        return 0
    target = out / OUT_NAME
    if a.check:
        same = target.exists() and target.read_text(encoding="utf-8") == text
        print(f"{OUT_NAME}: {'equal to the records' if same else 'DIFFERS from the records'} ({len(rows)} rows)")
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))
    print(f"{OUT_NAME}: {len(rows)} rows from {len(FACES)} score records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
