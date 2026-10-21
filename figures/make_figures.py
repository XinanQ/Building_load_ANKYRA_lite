"""Figures of the ankyra-lite repository, generated from ../results only.

    pip install matplotlib                    # Matplotlib (>= 3.8) is needed for the figures only
    python figures/make_figures.py            # writes figures/*.pdf (vector) and figures/*.png (300 dpi)

Options
    --only fig2,fig9b     draw a subset (file stems, or their "figN" prefixes)
    --results DIR         read the result files from DIR instead of ../results
    --out DIR             write the figures to DIR instead of this folder
    --standins DIR        read result files that are missing from the results folder from DIR instead (also the
                          environment variable ANKYRA_LITE_FIGURE_STANDINS). A figure drawn from such a file carries a
                          "stand-in data" watermark; the published figures never use this option.
    --strict              exit with status 1 if any figure is skipped or drawn from stand-in data, if a model_class of
                          the result files has no colour in CLASSES (without --strict: a warning and a guessed colour), or
                          if a result file holds rows of an LSTM-T-KAN model other than LSTM-T-KAN-M (the pattern
                          RETIRED; outside the 22: a stale file; without --strict: a warning, rows not drawn)

Each figure reads only its own result files. A missing file, or a file without what the figure needs (rank_tests.csv
without its pooled row), skips that figure with a message; the others are still drawn. Figure 7 also imports the
anchoring layer (ankyra_lite.anchor) and theory/operators.py (this repository's copy; if it is absent, the copy in the
original ANKYRA checkout, ANKYRA_ORIGINAL_ROOT, default ../Building_load_ANKYRA, is read with ``ankyra`` resolved to
``ankyra_lite.anchor``); its panels carry the property numbers of theory/README.md. Figure 17 is the counterpart of the
ANKYRA release's checks on external data: the external tests of ankyra-lite (external_tests.csv, written by
evaluation/external_tables.py from the score records; LCL, UNICON and HKUST, first read of this candidate on a previously
read population, and a GoiEner same-source out-of-time supplementary check; Helsinki trained and sealed, not scored).
Figure 18 shows the training-recipe study behind the released recipe (recipe_history.csv). Figure 0, the README's first
evaluation figure, draws ankyra-lite against the four zero-shot foundation-model variants on the release's three primary
readouts (benchmark_pairwise.csv, energy_error.csv, peak_readout.csv with the same-readout rule).

Release 0.2.0. Result files of ankyra-lite 0.2.0 (recipe Rw, six seeds, the origin-continuity layer) are recognised by
continuation_layer.csv (evaluation/extra_tables.py). With it, figure 1 draws the carrier's training (recipe Rw, six seeds)
and the continuation layer after the anchoring layer; figure 6 draws ankyra-lite before (F, ankyra_before_layer_kw) and
after (F', the delivered trajectory) the layer, with a panel on the first three days; figure 9c, beside figure 9 (which
is kept unchanged), draws the layer on against off by lead time (hour 0, hours 0-2, day 1, days 2-7, days 8-31, the whole
window, and the delivered trajectory's energy), pooled and by population; figure 17 reads the 0.2.0 rows of
external_tests.csv (forecast hours 0-2 in d-f; g-i: 0.2.0 against the 0.1.0 configuration, the layer on against off, the
loss weighting); figure 18 adds the 0.2.0 study's test read (d-e) from the rows of recipe_history.csv with a ``reading``;
the captions name the 0.2.0 recipe and readout rule A (energy, level and peak read before the layer). Without
continuation_layer.csv (0.1.x results) every figure is drawn as before and figure 9c is skipped.

The forecasters are whatever the result files hold: ankyra-lite (the reference, highlighted) and the baselines (the 20
external baselines of the ANKYRA release and LSTM-T-KAN-M), 22 forecasters in all. Rows of ANKYRA itself or of its
versions are never drawn. A result file that holds a forecaster outside these 22 is stale: rows of an LSTM-T-KAN
model other than LSTM-T-KAN-M (the pattern RETIRED) are dropped with a warning, and with --strict the figure fails. Figure 1 is drawn by
figures/architecture.py (a schematic on a synthetic series, no results) and reads only the parameter count and the
time per forecast from cost_per_window.csv (without that file it is drawn without a number).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                       # never write caches next to imported files (read-only checkouts)

import argparse                                       # noqa: E402
import csv                                            # noqa: E402
import importlib.util                                 # noqa: E402
import json                                           # noqa: E402
import math                                           # noqa: E402
import os                                             # noqa: E402
import re                                             # noqa: E402
import textwrap                                       # noqa: E402
import traceback                                      # noqa: E402
from pathlib import Path                              # noqa: E402

import numpy as np                                    # noqa: E402
import matplotlib                                     # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb      # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Patch   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = HERE
STANDINS: Path | None = None

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "mathtext.fallback": "stixsans",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "legend.frameon": False, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})

REF = "ankyra-lite"                       # the reference forecaster in every result file
LITE = "#B2182B"                          # ankyra-lite
CARRIER = "#1B7F79"                       # the carrier network
HIST = "#C0762A"                          # the unit's own history
FM = "#5E4FA2"                            # a zero-shot foundation model
CLASSES = [  # (legend entry, colour, the model_class strings of the result files that it covers)
    ("Same information · trained", "#253494", ("same information, trained",)),
    ("Same information · zero-shot foundation model", "#1D91C0", ("same information, zero-shot foundation model",)),
    ("Load and calendar · trained", "#D9B400", ("load and calendar, trained",)),
    ("Load only · zero-shot foundation model", "#807DBA", ("load only, zero-shot foundation model",)),
    ("Load only · statistical", "#41AB5D", ("load only, statistical",)),
    ("Load only · trained", "#EC7014", ("load only, trained",)),
    ("Per-unit regression (calendar, temperature)", "#8C6D31", ("per-unit regression",)),
    ("Profiles, naive, zero-shot GBT", "#9E9E9E", ("profile", "naive", "zero-shot gradient boosting")),
]
CLASS_COLOR = {c: col for _, col, cs in CLASSES for c in cs}
CLASS_COLOR["naive"] = "#BDBDBD"                  # drawn lighter than the profiles, as in the ANKYRA release
CLASS_LEGEND = [(t, col) for t, col, _ in CLASSES]
TEST = ["BDG2 2017", "Cambridge", "HEEW Arizona", "EWELD", "GoiEner non-household", "GoiEner households"]
NORWAY = ["Oslo", "Drammen", "CINELDI"]   # the release's preview tier; every population has its own test set here
ALL = TEST + NORWAY
SHORT = {"BDG2 2017": "BDG2", "Cambridge": "Cambridge", "HEEW Arizona": "HEEW", "EWELD": "EWELD",
         "GoiEner non-household": "GoiEner NH", "GoiEner households": "GoiEner HH", "Oslo": "Oslo", "Drammen": "Drammen",
         "CINELDI": "CINELDI"}
TWO_LINE = {"GoiEner non-household": "GoiEner\nNH", "GoiEner households": "GoiEner\nHH"}
# Display names. TiDE is drawn as "Linear-context", as in the first release's figures (its CSV label there is
# "Linear-context (TiDE implementation)": the covariate path of that implementation is inert, so its forecast is a linear
# map of the load context); the model id TiDE stays in the result files.
NICE = {"Holt-Winters": "Holt–Winters", "TiDE": "Linear-context", "Linear-context (TiDE implementation)": "Linear-context",
        "TiDE (as implemented: linear context model)": "Linear-context"}
LINEAR_CONTEXT_NOTE = "Linear-context: the TiDE implementation of the release, whose forecast is a linear map of the load context."
LSTM_TKAN_NOTE = ("LSTM-T-KAN-M: model and training code copied line by line from the authors' public repository (commit ce24938), "
                  "with the authors' exp2 feature list (load and calendar); not a strict reproduction of the paper (each deviation is "
                  "listed in docs/EVALUATION.md). On CINELDI it was retrained on the corrected data store with the same code and seeds.")
RECIPE_NOTE = ("ankyra-lite: the released recipe R (carrier trained from scratch with early stopping on the 93-day validation set, then "
               "8 more epochs with the validation span returned; three seeds averaged; figure 1b).")
RECIPE_NOTE_020 = ("ankyra-lite 0.2.0: carrier recipe Rw (trained from scratch with early stopping on the 93-day validation set, then 8 more "
                   "epochs with the validation span returned and each training window's loss weighted by the inverse-noise rule; six seeds "
                   "averaged; figure 1b), the anchoring layer, the state rule and the origin-continuity layer; hourly errors score the "
                   "delivered trajectory after that layer, energy, level and peak are read before it (readout rule A).")
CONT_FILE = "continuation_layer.csv"              # written by evaluation/extra_tables.py for release 0.2.0 only


def release020():
    """True when the result files are those of ankyra-lite 0.2.0 (results/continuation_layer.csv exists, or a stand-in of
    it). Checked without marking a stand-in, so a figure that does not draw the file is not watermarked."""
    return (RES / CONT_FILE).exists() or (STANDINS is not None and (STANDINS / CONT_FILE).exists())


def recipe_note():
    return RECIPE_NOTE_020 if release020() else RECIPE_NOTE


def notes(models=()):
    """The caption notes of every figure of ankyra-lite's forecasts: the recipe, and the TiDE label if TiDE is drawn."""
    return recipe_note() + (" " + LINEAR_CONTEXT_NOTE if "TiDE" in models else "") + " "


LABEL_MAX = 24                                    # longer result-file labels are drawn as the model id (e.g. LSTM-T-KAN-M)
EXCLUDED = {"ANKYRA", "A", "F0", "F1", "F1-off"}          # ANKYRA and its versions: never drawn
RETIRED = re.compile(r"LSTM-T-KAN-(?!M\b)")       # an LSTM-T-KAN model other than LSTM-T-KAN-M: outside the 22, stale, never drawn
STRICT = False                                    # --strict: an unknown model class is an error, not a grey fallback
GREY = "#595959"


# ============================================================================ plumbing
class Missing(Exception):
    """An input file of the figure is absent."""


class Deferred(Exception):
    """The figure has no counterpart yet."""


class Skipped(Exception):
    """The input files exist but do not hold what the figure needs (e.g. rank_tests.csv without its pooled row)."""


class UnknownClass(Exception):
    """A model_class string of the result files has no colour (an error in --strict mode)."""


class StaleResults(Exception):
    """A result file still holds rows of a model that is not in the comparison set (an error in --strict mode)."""


CURRENT = {"stem": "", "standin": set()}
WARNED: set = set()


def excluded(model):
    return model in EXCLUDED or model.startswith("ANKYRA")


def retired(r):
    """True if the row names a model that is not in the comparison set (as model, carrier, comparator or variant)."""
    return any(RETIRED.search(r.get(k) or "") for k in ("model", "carrier", "comparator", "ablation"))


def src(name):
    p = RES / name
    if p.exists():
        return p
    if STANDINS is not None and (STANDINS / name).exists():
        CURRENT["standin"].add(name)
        return STANDINS / name
    raise Missing(name)


def rows(name):
    """The rows of a result file without ANKYRA's rows. Rows of a model outside the comparison set (RETIRED) mean that the
    file is stale: they are dropped with a warning, and with --strict the figure fails."""
    with open(src(name), encoding="utf-8") as f:
        out = [r for r in csv.DictReader(f) if not excluded(r.get("model", ""))]
    stale = [r for r in out if retired(r)]
    if stale:
        msg = (f"{name} still holds {len(stale)} rows of an LSTM-T-KAN model other than LSTM-T-KAN-M, which is not in the "
               "comparison set (stale result file: regenerate the tables)")
        if STRICT:
            raise StaleResults(msg)
        if ("stale", name) not in WARNED:
            WARNED.add(("stale", name))
            print(f"  warning: {msg}; those rows are not drawn")
        out = [r for r in out if not retired(r)]
    return out


def pct(r):
    return 100.0 * (1.0 - math.exp(r))


def save(fig, stem, tight=True):
    """Write <stem>.pdf and <stem>.png (300 dpi). tight=False keeps the canvas as drawn (Figure 1 fills its 7.2-inch
    canvas exactly, so a tight box with padding would only widen it)."""
    if CURRENT["standin"]:
        fig.text(0.5, 0.5, "STAND-IN DATA\nnot results", transform=fig.transFigure, ha="center", va="center", rotation=24,
                 fontsize=34, color=LITE, alpha=0.15, fontweight="bold", zorder=1000, linespacing=1.1)
    box = {"bbox_inches": "tight", "pad_inches": 0.03} if tight else {}
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None}, **box)
    fig.savefig(OUT / f"{stem}.png", dpi=300, **box)
    plt.close(fig)


def label(m, lab=None):
    """Display name: NICE first, then the result file's model_label unless it is longer than LABEL_MAX characters (the
    LSTM-T-KAN-M label, for example), in which case the model id is drawn."""
    if m in NICE:
        return NICE[m]
    s = (lab or {}).get(m) or m
    if s in NICE:
        return NICE[s]
    return m if len(s) > LABEL_MAX else s


def class_color(cls, model=""):
    """Colour of a model class. A class without a colour is an error in --strict mode; otherwise a warning is printed
    once and the colour is guessed from the words of the class."""
    if model == REF:
        return LITE
    if cls in CLASS_COLOR:
        return CLASS_COLOR[cls]
    msg = f"model class {cls!r} (model {model or '?'}) has no colour in CLASSES"
    if STRICT:
        raise UnknownClass(msg)
    if cls not in WARNED:
        WARNED.add(cls)
        print(f"  warning: {msg}; drawn with a guessed colour")
    c = cls.lower()
    for key, col in (("same information", "#253494" if "trained" in c else "#1D91C0"), ("calendar", "#D9B400"),
                     ("zero-shot foundation", "#807DBA"), ("trained", "#EC7014"), ("statistical", "#41AB5D"),
                     ("regression", "#8C6D31")):
        if key in c:
            return col
    return "#9E9E9E"


def class_handles(cls_of, models, marker="s", size=4.3):
    """Legend handles of the classes of the drawn models only (every class string checked by class_color)."""
    used = {cls_of.get(m, "") for m in models if m != REF}
    col_of = {c: class_color(c, next((m for m in models if cls_of.get(m, "") == c), "")) for c in sorted(used)}
    out = [Line2D([], [], marker=marker, ls="", color=col, markersize=size, label=t_) for t_, col, cs in CLASSES if used & set(cs)]
    out += [Line2D([], [], marker=marker, ls="", color=col_of[c], markersize=size, label=c or "class not given") for c in sorted(used)
            if c not in CLASS_COLOR]
    return out


def row_major(handles, ncol):
    """Handles reordered so that a legend with ncol columns (Matplotlib fills it column by column, the first columns
    taking the extra entries) reads row by row in the given order."""
    n, ncol = len(handles), max(1, min(ncol, len(handles) or 1))
    nrow = math.ceil(n / ncol) if n else 0
    return [handles[r * ncol + c] for c in range(ncol) for r in range(nrow) if r * ncol + c < n]


def lims(points, base, cap, step=5.0, margin=3.0):
    """Axis limits of a forest panel: the base range, widened (in steps) to hold every point estimate, never beyond cap.
    Interval ends outside the limits are drawn as arrows; a point outside cap is drawn at the edge with its value."""
    v = [x for x in points if x is not None and np.isfinite(x)]
    lo, hi = min([base[0]] + [x - margin for x in v]), max([base[1]] + [x + margin for x in v])
    return max(cap[0], step * math.floor(lo / step)), min(cap[1], step * math.ceil(hi / step))


def ci(ax, y, p, a, b, xl, color, filled, edge=None, face=None, s=15, lw=1.1, marker="o", fs=5.1, z=3):
    """One point estimate p with its interval [a, b] on a horizontal axis limited to xl (filled: the interval excludes zero)."""
    lo_x, hi_x = xl
    edge = edge or color
    face = face or (color if filled else "white")
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
        ax.plot([max(a, lo_x), min(b, hi_x)], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=z - 1)
        d = 0.035 * (hi_x - lo_x)
        for e, beyond, dd in ((lo_x, a < lo_x, d), (hi_x, b > hi_x, -d)):
            if beyond:
                ax.annotate("", xy=(e, y), xytext=(e + dd, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, mutation_scale=5),
                            zorder=z - 1)
    pad = 0.012 * (hi_x - lo_x)
    pc = min(max(p, lo_x + pad), hi_x - pad)
    ax.scatter([pc], [y], s=s, marker=marker, color=face, edgecolor=edge, lw=0.9, zorder=z, clip_on=False)
    if p > hi_x or p < lo_x:
        ax.text(pc + (-2 * pad if p > hi_x else 2 * pad), y + 0.42, f"{p:+.0f}%", fontsize=fs, color=edge, ha="right" if p > hi_x else "left",
                va="center", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), zorder=z + 1)


def flag_note(*tables):
    """{set: flag text} over the given row lists, and the footnote line for the flagged populations."""
    flags = {}
    for t in tables:
        for r in t:
            if r.get("flag"):
                flags[r["set"]] = r["flag"]
    texts = sorted(set(flags.values()))
    note = ("† " + "; ".join(t[0].upper() + t[1:] for t in texts) + ".") if texts else ""
    return flags, note


def pop(s, flags=None, short=True):
    base = SHORT.get(s, s) if short else s
    return base + (" †" if flags and s in flags else "")


def num(n):
    """A small count in words, as in running text."""
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    return words[n] if 0 <= n < len(words) else f"{n:,}"


def head(s, flags=None):
    """Column heading of a heatmap: two lines where needed, the flag mark on its own line."""
    base = TWO_LINE.get(s, SHORT.get(s, s))
    if flags and s in flags:
        return base + (" †" if "\n" in base else "\n†")
    return base


CAP_FS = 6.0


def wrap(text, fs=CAP_FS, width=None):
    """Caption lines that fit the 7.2-inch width (paragraphs separated by newlines)."""
    width = width or int(round(150 * 6.0 / fs))
    out = []
    for para in text.split("\n"):
        out += textwrap.wrap(para, width, break_on_hyphens=False) or [""]
    return out


def cap_height(lines, fs=CAP_FS):
    """Height in inches of a block of caption lines (top padding included)."""
    return 0.04 + len(lines) * fs * 1.36 / 72


def put_caption(fig, lines, H, fs=CAP_FS, top=0.04, color=GREY, x=0.01):
    for j, line in enumerate(lines):
        fig.text(x, 1 - (top + j * fs * 1.36 / 72) / H, line, fontsize=fs, color=color, va="top")


def put_footer(fig, lines, H, fs=5.6, color="#7F7F7F", x=0.01):
    for j, line in enumerate(lines[::-1]):
        fig.text(x, (0.02 + j * fs * 1.36 / 72) / H, line, fontsize=fs, color=color, va="bottom")


def present(sets, table):
    have = {r["set"] for r in table}
    return [s for s in sets if s in have]


def mean_ranks(rk, sets):
    models = sorted({r["model"] for r in rk})
    val = {(r["model"], r["set"]): float(r["mean_unit_rank"]) for r in rk}
    return {m: float(np.mean([val[(m, s)] for s in sets if (m, s) in val])) for m in models}


def ordered_by_rank(rk, sets):
    mr = mean_ranks(rk, sets)
    return sorted(sorted(mr), key=lambda m: mr[m]), mr


# ============================================================================ Figure 1: architecture
N_PARAMETERS = 61_613                     # parameter_count of the released carrier configuration (used if cost_per_window.csv is absent)


def _cost():
    """(parameter count, the output box's cost line) from cost_per_window.csv: the timed ankyra-lite forecast (carrier
    forecasts, anchoring layer, peak readout) and the thread count. Without the file the line has no number."""
    try:
        cr = rows("cost_per_window.csv")
    except Missing:
        print("  note: cost_per_window.csv not found; fig1 is drawn without the time per forecast")
        return N_PARAMETERS, "runs on the CPU;\nno GPU needed"
    item = lambda r: (r.get("item") or "").strip().lower()                       # noqa: E731
    par = next((_num(r.get("value")) for r in cr if item(r).startswith("carrier network parameters")), None)
    sec = next((_num(r.get("value")) for r in cr if item(r).startswith("ankyra-lite forecast") and (r.get("unit") or "") == "s/window"), None)
    thr = next((_num(r.get("value")) for r in cr if item(r).startswith("device")), None)
    n_par = int(par) if par else N_PARAMETERS
    where = "the CPU" if thr is None else ("one CPU thread" if int(thr) == 1 else f"{num(int(thr))} CPU threads")
    if sec is None:
        print("  note: cost_per_window.csv has no 'ankyra-lite forecast' row in s/window; fig1 is drawn without the time per forecast")
        return n_par, f"runs on\n{where};\nno GPU needed"
    t_ = f"{1000 * sec:.0f} ms" if sec < 0.1 else (f"{sec:.2g} s" if sec < 10 else f"{sec:.0f} s")
    return n_par, f"about {t_} per\nforecast on\n{where}"


def _architecture():
    """figures/architecture.py (the drawing of Figure 1), loaded by its path so the script runs from any folder."""
    spec = importlib.util.spec_from_file_location("_ankyra_lite_figure_architecture", HERE / "architecture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig_architecture():
    """Figure 1, drawn by figures/architecture.py: (a) the issue times unrolled on a synthetic record, (b) the carrier
    network, (c) the anchoring layer; with 0.2.0 results (continuation_layer.csv present) also the carrier's training
    (recipe Rw, six seeds) and the origin-continuity layer after the anchoring layer. Only the parameter count and the
    time per forecast are read (cost_per_window.csv)."""
    arch = _architecture()
    n_par, cost_line = _cost()
    with plt.rc_context(arch.RC):
        fig = arch.build(n_parameters=n_par, cost_text=cost_line, release="0.2" if release020() else "0.1")
        for problem in arch.layout_problems(fig):
            print("  warning: fig1 layout: " + problem.encode("ascii", "backslashreplace").decode("ascii"))
        save(fig, "fig1_architecture", tight=False)


# ============================================================================ Figure 1b: data splits
def _date(s):
    return np.datetime64(s.strip()[:10], "D")


def fig_data_splits():
    ds = rows("datasets.csv")
    name_col = "set" if "set" in ds[0] else "population"
    first_col = next((c for c in ("first_training_origin", "first_train_origin") if c in ds[0]), None)
    need = ["validation_start", "training_cut", "test_start", "last_test_origin"]
    for c in [name_col] + need:
        if c not in ds[0]:
            raise Missing(f"datasets.csv column {c}")
    for r in ds:
        r["_name"] = r[name_col]
        r["_d"] = {c: _date(r[c]) for c in need}
        r["_d"]["first"] = _date(r[first_col]) if first_col and r.get(first_col) else None
        r["_rec"] = (int((r["_d"]["training_cut"] - r["_d"]["first"]) / np.timedelta64(1, "D")) if r["_d"]["first"] is not None
                     else int(float(r.get("record_days_before_cut") or 0)))
    ds = sorted(ds, key=lambda r: -r["_rec"])
    notes = {}
    for r in ds:
        cal = r.get("calendar", "")
        notes[r["_name"]] = r.get("note") or r.get("flag") or (f"dates are {cal}" if "(" in cal else "")
    flagged = {p for p, n_ in notes.items() if n_}
    n = len(ds)
    COL = {"training set": "#4A6FA5", "validation set": "#E8A33D", "gap (186 days)": "#D9D9D9", "test set": LITE}
    foot = wrap("The training set also borrows windows of other populations that end before the validation start (not drawn). "
                "Validation share: the share of the population's own windows held out for early stopping."
                + ("\n† " + "; ".join(f"{p}: {notes[p]}" for p in sorted(flagged)) if flagged else ""), fs=5.6, width=160)
    cap = wrap("Per population, in time order: training set | validation set (the last 93 days before the training cut) | 186-day gap | "
               "test set (from the test start to the last test origin + 31 days). Released recipe: training from scratch with early stopping "
               "on the validation set, then the validation span is returned to the training set for 8 more epochs (labels ending within 24 h "
               "of the cut excluded)" + ("; in 0.2.0 (recipe Rw) the loss of these 8 epochs is weighted per training window by the "
                                         "inverse-noise rule, and six seeds are trained and averaged" if release020() else "")
               + "; nothing at or after the training cut is used for training.")
    H = 1.15 + 0.30 * n + 0.1 * len(foot)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.36], wspace=0.06, left=0.13, right=0.995,
                          top=1 - (cap_height(cap) + 0.42) / H, bottom=(0.42 + 0.1 * len(foot)) / H)
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1], sharey=ax); tx = fig.add_subplot(gs[2], sharey=ax)
    y = np.arange(n)[::-1]
    month = np.timedelta64(31, "D")

    def yr(d):
        return 1970 + (d - np.datetime64("1970-01-01")) / np.timedelta64(1, "D") / 365.2425

    lo_d, hi_d = -560, 1240
    for yy, r in zip(y, ds):
        d = r["_d"]
        segs = [("validation set", d["validation_start"], d["training_cut"]), ("gap (186 days)", d["training_cut"], d["test_start"]),
                ("test set", d["test_start"], d["last_test_origin"] + month)]
        if d["first"] is not None:
            segs.insert(0, ("training set", d["first"], d["validation_start"]))
        else:
            bx.text(-100, yy, "first training origin not recorded ", va="center", ha="right", fontsize=5.0, color=GREY, zorder=3)
        for name, a, b in segs:
            ax.barh(yy, yr(b) - yr(a), left=yr(a), height=0.56, color=COL[name], lw=0, zorder=2)
            rel_a = (a - d["training_cut"]) / np.timedelta64(1, "D"); rel_b = (b - d["training_cut"]) / np.timedelta64(1, "D")
            ca, cb = max(rel_a, lo_d), min(rel_b, hi_d)
            if cb > ca:
                bx.barh(yy, cb - ca, left=ca, height=0.56, color=COL[name], lw=0, zorder=2,
                        hatch="////" if name.startswith("gap") else None, edgecolor="white" if name.startswith("gap") else None)
            if name == "training set":
                if rel_a < lo_d:
                    bx.text(lo_d + 10, yy, f"from −{abs(rel_a):,.0f} d", va="center", ha="left", fontsize=5.0, color="white", zorder=3)
                elif ca - lo_d < 160:
                    bx.text(ca + 10, yy, f"−{abs(rel_a):,.0f} d", va="center", ha="left", fontsize=5.0, color="white", zorder=3)
                else:
                    bx.text(ca - 10, yy, f"−{abs(rel_a):,.0f} d", va="center", ha="right", fontsize=5.0, color=GREY, zorder=3)
            if name == "test set":
                if rel_b > hi_d:
                    bx.text(hi_d - 10, yy, f"to +{rel_b:,.0f} d", va="center", ha="right", fontsize=5.0, color="white", zorder=3)
                else:
                    bx.text(cb + 10, yy, f"+{rel_b:,.0f} d", va="center", ha="left", fontsize=5.0, color=GREY, zorder=3)
        w_, u_ = r.get("test_windows", ""), r.get("test_units", "")
        share = r.get("validation_share_pct", "")
        line = f"{int(float(w_)):,} / {int(float(u_)):,}" if w_ and u_ else ""
        tx.text(0.02, yy, line, va="center", ha="left", fontsize=5.9, color="#1A1A1A")
        if share:
            tx.text(0.62, yy, f"{float(share):.1f}%", va="center", ha="left", fontsize=5.9, color="#1A1A1A")
    top_row = ds[0]["_d"]
    v_len = (top_row["training_cut"] - top_row["validation_start"]) / np.timedelta64(1, "D")
    g_len = (top_row["test_start"] - top_row["training_cut"]) / np.timedelta64(1, "D")
    bx.text(-8, y[0] + 0.36, f"{v_len:.0f} d", ha="right", va="bottom", fontsize=5.0, color="#B07A1E")
    bx.text(8, y[0] + 0.36, f"{g_len:.0f} d", ha="left", va="bottom", fontsize=5.0, color=GREY)
    ax.set_yticks(y); ax.set_yticklabels([r["_name"].replace(" 2017", "") + (" †" if r["_name"] in flagged else "") for r in ds], fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    starts = [r["_d"]["first"] or r["_d"]["validation_start"] for r in ds]
    x0 = math.floor(min(yr(s_) for s_ in starts)); x1 = math.ceil(max(yr(r["_d"]["last_test_origin"] + month) for r in ds))
    ax.set_xlim(x0, x1); ax.set_xticks(range(x0 - x0 % 5 + (5 if x0 % 5 else 0), x1 + 1, 5)); ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
    ax.set_xlabel("Calendar year", fontsize=6.6); ax.set_title("a   Calendar time", fontsize=7.2)
    bx.set_xlim(lo_d, hi_d); bx.axvline(0, color="#1A1A1A", lw=0.7, zorder=3)
    bx.set_xticks([-365, 0, 186, 365, 730, 1095]); bx.set_xticklabels(["−365", "0", "186", "365", "730", "1095"], fontsize=6.0)
    bx.tick_params(axis="y", left=False, labelleft=False); bx.spines["left"].set_visible(False)
    bx.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
    bx.set_xlabel("Days from the training cut", fontsize=6.6); bx.set_title("b   Aligned on the training cut", fontsize=7.2)
    tx.axis("off"); tx.set_xlim(0, 1)
    tx.text(0.02, n - 0.25, "Test windows\n/ units", fontsize=5.8, color=GREY, va="bottom", ha="left")
    if any(r.get("validation_share_pct") for r in ds):
        tx.text(0.62, n - 0.25, "Validation\nshare", fontsize=5.8, color=GREY, va="bottom", ha="left")
    ax.set_ylim(-0.6, n - 0.25)
    ax.legend(handles=[Patch(fc=c, label=k) for k, c in COL.items()], loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=4,
              fontsize=6.0, handlelength=1.0, handleheight=0.8, columnspacing=1.0, handletextpad=0.35)
    put_caption(fig, cap, H)
    put_footer(fig, foot, H)
    save(fig, "fig1b_data_splits")


# ============================================================================ Figure 2: test-set ranks and head-to-head
def fig_test_ranks():
    rk = [r for r in rows("benchmark_mean_unit_rank.csv") if r["subset"] == "late"]
    pw = [r for r in rows("benchmark_pairwise.csv") if r["subset"] == "late"]
    flags, note = flag_note(rk)
    sets = present(ALL, rk)
    cls = {r["model"]: r["model_class"] for r in pw}
    lab = {r["model"]: r["model_label"] for r in rk}
    models, mr = ordered_by_rank(rk, sets)
    val = {(r["model"], r["set"]): float(r["mean_unit_rank"]) for r in rk}
    M = np.array([[val.get((m, s), np.nan) for s in sets] for m in models])
    n, k = len(models), len(sets)
    nmod = int(max(float(r["n_models"]) for r in rk)) if "n_models" in rk[0] else n
    cap = wrap(f"{n} forecasters ({n - 1} baselines and ankyra-lite) on the same test windows of {num(k)} populations (origins after each "
               "population's training cutoff of the trained baselines, where every forecaster has a forecast). A re-evaluation: these windows "
               "had been read before. Mean: over the populations shown. NH/HH: non-/households. b: 95% unit-and-month bootstrap interval of "
               "the log RMS ratio. " + notes(models)
               + (LSTM_TKAN_NOTE + " " if "LSTM-T-KAN-M" in models else "") + note)
    H = 1.45 + 0.135 * n + cap_height(cap)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.05, left=0.175, right=0.985, top=1 - (cap_height(cap) + 0.62) / H,
                          bottom=0.78 / H)
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1], sharey=ax)
    cmap = LinearSegmentedColormap.from_list("rank", ["#08306B", "#2171B5", "#6BAED6", "#C6DBEF", "#F7FBFF"])
    data = np.column_stack([M, np.nanmean(M, 1)])
    ax.imshow(data, cmap=cmap, vmin=1, vmax=nmod, aspect="auto", interpolation="nearest")
    for i in range(n):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.6, color="white" if v < 0.4 * nmod else "#1A1A1A",
                    fontweight="bold" if models[i] == REF or j == data.shape[1] - 1 else "normal")
    if any(s in NORWAY for s in sets):
        ax.axvline(len([s for s in sets if s in TEST]) - 0.5, color="white", lw=1.2)
    ax.axvline(k - 0.5, color="white", lw=2.4)
    ax.set_xticks(range(k + 1)); ax.set_xticklabels([head(s, flags) for s in sets] + ["Mean"], fontsize=5.3, linespacing=1.05)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(range(n)); ax.set_yticklabels([label(m, lab) for m in models], fontsize=6.3); ax.tick_params(axis="y", length=0, pad=9)
    for t_, m in zip(ax.get_yticklabels(), models):
        if m == REF:
            t_.set_color(LITE); t_.set_fontweight("bold")
    for i, m in enumerate(models):
        ax.scatter([-0.78], [i], s=11, color=class_color(cls.get(m, ""), m), clip_on=False, marker="s", zorder=5)
    ia = models.index(REF)
    ax.add_patch(Rectangle((-0.5, ia - 0.5), data.shape[1], 1, fill=False, ec=LITE, lw=1.5, zorder=6))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"a   Mean per-unit rank on the {num(k)} test sets (lower is better)", fontsize=7.6, pad=22)
    win = np.zeros(n); tie = np.zeros(n); loss = np.zeros(n)
    for i, m in enumerate(models):
        for r in pw:
            if r["model"] == m and r["set"] in sets:
                lo, hi = float(r["um_low"]), float(r["um_high"])
                win[i] += hi < 0; loss[i] += lo > 0; tie[i] += not (hi < 0 or lo > 0)
    yy = np.arange(n)
    bx.barh(yy, win, color="#2166AC", height=0.62)
    bx.barh(yy, tie, left=win, color="#D9D9D9", height=0.62)
    bx.barh(yy, loss, left=win + tie, color=LITE, height=0.62)
    for i in range(n):
        if models[i] == REF:
            bx.text(0.12, i, "reference", va="center", fontsize=6.0, color=LITE, fontstyle="italic")
        elif win[i] > 0:
            bx.text(win[i] - 0.12, i, f"{int(win[i])}", va="center", ha="right", fontsize=5.7, color="white", fontweight="bold")
    bx.set_xlim(0, k); bx.set_xticks(range(k + 1)); bx.set_xlabel("Number of test sets", fontsize=6.8, labelpad=2)
    bx.tick_params(axis="y", left=False, labelleft=False); bx.spines["left"].set_visible(False)
    bx.set_title("b   Head-to-head on the test sets", fontsize=7.6, pad=22)
    bx.legend(handles=[Patch(fc="#2166AC", label="ankyra-lite better"), Patch(fc="#D9D9D9", label="not resolved"),
                       Patch(fc=LITE, label="ankyra-lite worse")], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
              fontsize=5.8, handlelength=1.0, handleheight=0.8, columnspacing=0.7, handletextpad=0.3)
    fig.legend(handles=row_major(class_handles(cls, models), 4), loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, fontsize=5.9,
               handletextpad=0.25, columnspacing=1.0)
    put_caption(fig, cap, H)
    save(fig, "fig2_test_ranks")


# ============================================================================ Figure 3: pairwise improvements with intervals
def info_tier(cls):
    """0: the same information as ankyra-lite (load, temperature, calendar); 1: load and calendar; 2: load only."""
    c = cls.lower()
    return 0 if c.startswith("same information") else (1 if "calendar" in c else 2)


TIER_TEXT = ("the same information as ankyra-lite: load, temperature, calendar", "load and calendar, no temperature", "load only")


def fig_test_forest():
    pw = [r for r in rows("benchmark_pairwise.csv") if r["subset"] == "late"]
    flags, note = flag_note(pw)
    sets = present(ALL, pw)
    cls = {r["model"]: r["model_class"] for r in pw}
    lab = {r["model"]: r["model_label"] for r in pw}
    have = set(cls) - {REF}
    pref = ["TiDE", "iTransformer-X", "GBT-T", "Chronos-2-X", "TimesFM-X", "LSTM-T-KAN-M", "TimesFM", "Chronos-2", "Holt-Winters", "MSTL"]
    shown = [m for m in pref if m in have]
    shown += sorted(m for m in have if m not in shown and info_tier(cls[m]) < 2)          # every baseline with calendar input
    tiers = [[m for m in shown if info_tier(cls[m]) == t] for t in range(3)]
    models = [m for t in tiers for m in t]
    nonempty = [t for t in range(3) if tiers[t]]
    vals = [float(r["improvement_pct"]) for r in pw if r["model"] in models and r["set"] in sets]
    lo_x, hi_x = lims(vals, (-30, 45), (-60, 80), margin=5)
    nr = math.ceil(len(sets) / 3)
    handles = class_handles(cls, models, marker="o", size=4)
    leg_extra = 0.15 * (math.ceil(len(handles) / 4) - 1)
    pos = ("top", "middle", "bottom") if len(nonempty) == 3 else ("top", "bottom")
    parts = []
    for j, t in enumerate(nonempty):
        names = ", ".join(label(m, lab) for m in tiers[t])
        parts.append(f"{pos[j] if len(nonempty) > 1 else 'all'}: {names} ({TIER_TEXT[t]})")
    sep = ("Dashed lines: the information given to the baselines changes; " if len(nonempty) > 1 else "Baselines shown, ")
    cap = wrap("Unit-equal improvement 100[1 − exp(r)] in hourly RMS, 95% unit-and-month bootstrap intervals; filled = interval excludes "
               "zero, red = ankyra-lite worse; arrows: interval beyond the axis. " + sep + "; ".join(parts) + ". "
               + (f"The other {num(len(have) - len(models))} baselines are in figures 2 and 8 ({len(have) + 1} forecasters in all). "
                  if len(have) > len(models) else "") + notes(models).rstrip()
               + (" " + LSTM_TKAN_NOTE if "LSTM-T-KAN-M" in models else ""))
    foot = wrap("BDG2: on the three near-zero meters ankyra-lite's error is exactly zero (state rule), so they drop out of the unit-equal "
                "contrasts. " + note, fs=5.5, width=165)
    H = 0.6 + (0.14 * len(models) + 0.42) * nr + cap_height(cap) + 0.12 * len(foot) + leg_extra
    fig, axs = plt.subplots(nr, 3, figsize=(7.2, H), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.135, right=0.99, top=1 - (cap_height(cap) + 0.2) / H, bottom=(0.62 + leg_extra + 0.12 * len(foot)) / H,
                        hspace=0.34, wspace=0.08)
    y = np.arange(len(models))[::-1].astype(float)
    cuts, k = [], 0
    for t in nonempty[:-1]:
        k += len(tiers[t]); cuts.append(len(models) - k - 0.5)
    for ax, s in zip(axs.ravel(), sets):
        ax.axvspan(0, hi_x, color="#F3F6FA", zorder=0)
        ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
        for c_ in cuts:
            ax.axhline(c_, color="#BDBDBD", lw=0.5, ls=(0, (2, 2)))
        for yy, m in zip(y, models):
            r = next((rr for rr in pw if rr["set"] == s and rr["model"] == m), None)
            if r is None:
                continue
            p, a, b = float(r["improvement_pct"]), pct(float(r["um_high"])), pct(float(r["um_low"]))
            c = class_color(cls[m], m)
            sig_better, sig_worse = float(r["um_high"]) < 0, float(r["um_low"]) > 0
            ci(ax, yy, p, a, b, (lo_x, hi_x), c, sig_better or sig_worse, edge=LITE if sig_worse else c, face=LITE if sig_worse else None)
        ax.set_title(pop(s, flags, short=False).replace(" 2017", ""), fontsize=7.2)
        ax.set_xlim(lo_x, hi_x); ax.set_ylim(-0.7, len(models) - 0.3)
        ax.set_yticks(y); ax.set_yticklabels([label(m, lab) for m in models], fontsize=6.0); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E5E5E5", lw=0.4, zorder=0)
    for ax in axs.ravel()[len(sets):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("ankyra-lite improvement over the model (%)", fontsize=6.6)
    put_caption(fig, cap, H)
    fig.legend(handles=row_major(handles, 4), loc="lower center", ncol=min(len(handles), 4), fontsize=5.8,
               bbox_to_anchor=(0.55, (0.06 + 0.12 * len(foot)) / H), handletextpad=0.2, columnspacing=1.0)
    put_footer(fig, foot, H, fs=5.5)
    save(fig, "fig3_test_pairwise")


# ============================================================================ Figure 0: three readouts against the zero-shot foundation models
# The four zero-shot foundation-model variants, top to bottom within each population: each has its own marker, and its
# colour is that of its model class (load only / the release's shared covariate set), so marker, colour and position
# all tell them apart (colour alone is not relied on under a colour-vision deficiency).
FM_VARIANTS = [("TimesFM", "o"), ("TimesFM-X", "D"), ("Chronos-2", "s"), ("Chronos-2-X", "^")]
FM_MARK_SCALE = {"D": 0.75, "^": 1.15, "s": 0.9}
BETTER = "#2166AC"                                   # resolved in ankyra-lite's favour (as in the head-to-head panels)


def fig_advantage():
    """Figure 0: ankyra-lite against the four zero-shot foundation-model variants on the release's three primary readouts
    (hourly error, monthly energy error, the peak readout under the same-readout rule), every population."""
    fm = [m for m, _ in FM_VARIANTS]
    pw = [r for r in rows("benchmark_pairwise.csv") if r["subset"] == "late"]
    hourly = {(r["set"], r["model"]): r for r in pw if r["model"] in fm}
    energy = {(r["set"], r["model"]): r for r in rows("energy_error.csv") if r["unit_set"] == "all units" and r["model"] in fm}
    peak = {(r["set"], r["comparator"][16:]): r for r in rows("peak_readout.csv")
            if r["comparator"].startswith("same readout on ") and r["comparator"][16:] in fm}
    variants = [(m, mk) for m, mk in FM_VARIANTS if any(k[1] == m for t in (hourly, energy, peak) for k in t)]
    if not variants:
        raise Skipped("the result files hold none of the four zero-shot foundation-model variants")
    cls = {r["model"]: r["model_class"] for r in pw}
    lab = {r["model"]: r["model_label"] for r in pw}
    panels = [(hourly, "a   Hourly error", "benchmark_pairwise.csv"), (energy, "b   Monthly energy error", "energy_error.csv"),
              (peak, "c   Monthly peak, same readout", "peak_readout.csv")]
    sets = [s for s in ALL if any((s, m) in t for t, *_ in panels for m, _ in variants)]
    flags, note = flag_note(*(list(t.values()) for t, *_ in panels))
    vals = []
    for t, *_ in panels:
        for r in t.values():
            if r["set"] in sets:
                vals += [float(r["improvement_pct"]), pct(float(r["um_high"])), pct(float(r["um_low"]))]
    xl = lims(vals, (-20, 30), (-60, 80), margin=1)   # interval ends included, so no interval runs into the frame
    others = len(set(cls) - {REF}) - len(variants)
    info = {m: ("shared covariate set" if info_tier(cls.get(m, "")) == 0 else "load only") for m, _ in variants}
    v2 = release020()
    readouts = ("a: hourly error, the unit-equal RMS ratio over the 744 hours (ankyra-lite's delivered trajectory, after the continuation "
                "layer); b: monthly energy error, 744 × the error of the window mean (ankyra-lite: its trajectory before the continuation "
                "layer, readout rule A; each variant: its delivered trajectory); c: monthly peak, ankyra-lite's peak readout (read before the "
                "continuation layer) against the same readout applied to each variant's trajectory (the release's rule). The layer's own "
                "effect by lead time is in figure 9c. ")
    cap = wrap("ankyra-lite against the zero-shot foundation-model variants on the test windows of every population, by the release's primary "
               "estimators. " + (readouts if v2 else "a: hourly error, the unit-equal RMS ratio over the 744 hours; b: monthly energy error, 744 × the error of the window "
               "mean of each delivered trajectory; c: monthly peak, ankyra-lite's peak readout against the same readout applied to each "
               "variant's trajectory (the release's rule). ") + "Improvement 100[1 − exp(r)] with 95% unit-and-month bootstrap intervals; "
               "filled = resolved in ankyra-lite's favour, red = resolved against it, open = not resolved; shaded: ankyra-lite better. Above each "
               "panel: the count of its contrasts resolved better, not resolved and resolved worse. TimesFM is TimesFM 2.5; the -X variants are given "
               "the release's shared covariate set (no observed future weather), the others read the load only. "
               + (f"The other {num(others)} baselines are in figures 2, 3, 5 and 11 ({others + len(variants) + 1} forecasters in all). " if others > 0 else "")
               + notes(fm).rstrip())
    foot = wrap("A unit whose error is exactly zero drops out of a unit-equal contrast (on BDG2, the three near-zero meters, where the state "
                "rule makes ankyra-lite's hourly error zero). GoiEner NH / HH: non-household / households. " + note, fs=5.5, width=165)
    per_pop = 0.42
    bottom = 0.76 + 0.12 * len(foot)                  # footer, the two legend rows, the tick labels and the axis label
    H = cap_height(cap) + 0.38 + per_pop * len(sets) + bottom
    fig, axs = plt.subplots(1, len(panels), figsize=(7.2, H), sharey=True, squeeze=False)
    axs = axs[0]
    fig.subplots_adjust(left=0.105, right=0.99, top=1 - (cap_height(cap) + 0.38) / H, bottom=bottom / H, wspace=0.09)
    nv = len(variants)
    offs = np.linspace(0.3, -0.3, nv) if nv > 1 else np.array([0.0])
    yc = np.arange(len(sets))[::-1].astype(float)
    ticks = [v for v in range(-100, 101, 25) if xl[0] <= v <= xl[1]]
    for ax, (table, title, _) in zip(axs, panels):
        ax.axvspan(0, xl[1], color="#F3F6FA", zorder=0)
        ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
        for c_ in yc[:-1] - 0.5:
            ax.axhline(c_, color="#E5E5E5", lw=0.5, zorder=0)
        n = better = worse = 0
        for y0, s in zip(yc, sets):
            for (m, mk), dy in zip(variants, offs):
                r = table.get((s, m))
                if r is None:
                    continue
                p, a, b = float(r["improvement_pct"]), pct(float(r["um_high"])), pct(float(r["um_low"]))
                sig_better, sig_worse = float(r["um_high"]) < 0, float(r["um_low"]) > 0
                n += 1; better += sig_better; worse += sig_worse
                c = class_color(cls.get(m, ""), m)
                ci(ax, y0 + dy, p, a, b, xl, c, sig_better or sig_worse, edge=LITE if sig_worse else c, face=LITE if sig_worse else None,
                   s=13 * FM_MARK_SCALE.get(mk, 1.0), lw=0.9, marker=mk)
        ax.set_xlim(*xl); ax.set_ylim(yc[-1] - 0.55, yc[0] + 0.55)
        ax.set_xticks(ticks)
        ax.grid(axis="x", color="#E5E5E5", lw=0.4, zorder=0)
        ax.set_xlabel("ankyra-lite improvement over the model (%)", fontsize=6.4)
        ax.set_title(title, fontsize=7.4, pad=12)
        t0 = ax.text(0.0, 1.012, f"{n} contrasts:  ", transform=ax.transAxes, fontsize=5.9, color=GREY, va="bottom", ha="left")
        t1 = ax.annotate(f"better {better}", xy=(1, 0), xycoords=t0, fontsize=5.9, color=BETTER, fontweight="bold", va="bottom", ha="left")
        t2 = ax.annotate(f"  ·  not resolved {n - better - worse}  ·  ", xy=(1, 0), xycoords=t1, fontsize=5.9, color=GREY, va="bottom", ha="left")
        ax.annotate(f"worse {worse}", xy=(1, 0), xycoords=t2, fontsize=5.9, color=LITE if worse else GREY, fontweight="bold", va="bottom", ha="left")
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(yc); axs[0].set_yticklabels([pop(s, flags) for s in sets], fontsize=6.6)
    handles = [Line2D([], [], marker=mk, ls="", color=class_color(cls.get(m, ""), m),
                      markersize=4.2 * FM_MARK_SCALE.get(mk, 1.0) ** 0.5, label=f"{label(m, lab)} ({info[m]})") for m, mk in variants]
    marks = [Line2D([], [], marker="o", ls="", markerfacecolor=GREY, markeredgecolor=GREY, markersize=4.0, label="filled: resolved better"),
             Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, markersize=4.0, label="open: not resolved"),
             Line2D([], [], marker="o", ls="", markerfacecolor=LITE, markeredgecolor=LITE, markersize=4.0, label="red: resolved worse")]
    ncol = max(len(handles), len(marks))
    fig.legend(handles=row_major(handles + marks, ncol), loc="lower center", ncol=ncol, fontsize=6.0,
               bbox_to_anchor=(0.5, (0.05 + 0.12 * len(foot)) / H), handletextpad=0.25, columnspacing=1.6, labelspacing=0.45)
    put_caption(fig, cap, H)
    put_footer(fig, foot, H, fs=5.5)
    save(fig, "fig0_advantage")


# ============================================================================ Figure 4: ablation
# One (marker, colour) per variant of a panel, in order; every marker and every colour differs. OKLab distance (the
# smallest over normal vision and simulated protanopia, deuteranopia, tritanopia): at least 0.104 between any two of the
# first eight, at least 0.071 among all ten, at least 0.088 from ankyra-lite's red. A panel with more variants reuses a
# marker with another colour (never the same pair twice) and prints a warning.
ABL_STYLE = [("o", "#1A1A1A"), ("D", "#E6AB02"), ("s", "#7B5EA7"), ("^", "#D55E00"), ("v", "#56B4E9"), ("P", "#332288"),
             ("X", "#882255"), ("*", "#1B9E77"), ("h", "#CC79A7"), ("p", "#6A3D9A")]
# Recipe variants named as in figure 18 (recipe_history.csv): F = the control, V = the proportional validation set.
ABL_NICE = {"recipe: 8 continuation epochs, validation set held out, no span return (anchored)":
            "recipe F: 8 more epochs, validation span not returned (anchored)",
            "recipe: continuation with a proportional validation set and early stopping (anchored)":
            "recipe V: continuation re-split with a proportional validation set, early stopping (at most 30 epochs) (anchored)",
            "recipe: from-scratch stage alone, no continuation (anchored)":
            "recipe: from-scratch stage alone, without the 8 more epochs (anchored)"}
# 0.2.0 configuration rows of ablation.csv (evaluation/extra_tables.py config_label), drawn with a short label
ABL_CONFIG = re.compile(r"(0\.1\.0 )?configuration: (recipe R \(span return\)|recipe Rw \(span return with inverse-noise loss weighting\)), "
                        r"(six seeds|seeds 0-2), (with|without|no) (?:the )?continuation layer \(anchored\)")
ABL_MARK_SCALE = {"*": 1.7, "h": 1.15, "p": 1.15}              # the star reads smaller than the other markers at one size


# Rows of ablation.csv drawn in panel d of figure 4 (with the handover arms of handover_granularity.csv), not in panel a.
FIG4_MECH = (("ablation", "fixed division (history level and daily path, no handover)", "vs the fixed division (history level and daily path, no handover)"),
             ("ablation", "fixed half handover weight", "vs a fixed half handover weight"),
             ("handover", "C2 AW vs AM", "vs one handover weight per window"),
             ("ablation", "one whole-window combination weight of the history side and the carrier", "vs one whole-window weight of the history side and the carrier"),
             ("ablation", "half-and-half average of the history side and the carrier", "vs a half-and-half average of the two"))
FIG4_MECH_STYLE = [("o", "#4D4D4D"), ("s", "#C0762A"), ("D", "#7B5EA7"), ("^", "#1B7F79"), ("v", "#56B4E9")]
WEEK_RAMP_LITE = ["#F4A582", "#E0614F", "#B2182B", "#67001F"]          # forecast weeks 1-4, light to dark (one hue)
WEEK_RAMP_GREY = ["#D0D0D0", "#A6A6A6", "#7A7A7A", "#4A4A4A"]
WEEK_RAMP_CARRIER = ["#A6DCD7", "#5FB8B0", "#1B7F79", "#0B4A46"]


def _fig4_mechanism(fig, gs, sets, flags, y, top_axes):
    """Row 2 of figure 4: d the handover arms and the whole-window combination, e by forecast week against the carrier
    alone, f the mean handover weight on the carrier.  From ablation.csv, handover_granularity.csv and lead_weeks.csv."""
    ab = [r for r in rows("ablation.csv") if r.get("error", "hourly") in ("", "hourly")]
    hg, lw = rows("handover_granularity.csv"), rows("lead_weeks.csv")
    get = {"ablation": lambda s_, k: next((r for r in ab if r["set"] == s_ and r["ablation"] == k), None),
           "handover": lambda s_, k: next((r for r in hg if r["set"] == s_ and r["contrast"] == k), None)}
    A2 = 0.3 * len(sets)
    nt = len([s_ for s_ in sets if s_ in TEST])
    axes = []

    def frame(ax, title, xlabel):
        if nt and nt < len(sets):
            ax.axhspan(y[nt - 1] - 0.5, y[0] + 0.5, color="#F3F6FA", zorder=0)
        ax.set_title(title, fontsize=7.2)
        ax.set_xlabel(xlabel, fontsize=6.5)
        ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
    # d: handover arms and the whole-window combination
    ax = fig.add_subplot(gs[0], sharey=top_axes); axes.append(ax)
    pts = [float(r["improvement_pct"]) for kind, key, _ in FIG4_MECH for s_ in sets for r in [get[kind](s_, key)] if r]
    xl = lims(pts, (-4.0, 4.0), (-15.0, 30.0), step=2.0, margin=1.0)
    ax.axvline(0, color="#7F7F7F", lw=0.6)
    offs = np.linspace(0.32, -0.32, len(FIG4_MECH))
    for (kind, key, _), (mk, col), dy in zip(FIG4_MECH, FIG4_MECH_STYLE, offs):
        for yy, s_ in zip(y, sets):
            r = get[kind](s_, key)
            if r is None or r["log_ratio"] in ("", None):
                continue
            sig = float(r["um_high"]) < 0 or float(r["um_low"]) > 0
            ci(ax, yy + dy, float(r["improvement_pct"]), pct(float(r["um_high"])), pct(float(r["um_low"])), xl, col, sig, s=10, lw=0.8, marker=mk, fs=4.8)
    ax.set_xlim(*xl)
    frame(ax, "d   Handover and one whole-window weight", "ankyra-lite improvement (%)")
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", color=col, markersize=3.4, label="\n".join(textwrap.wrap(lab, 40)))
                       for (_, _, lab), (mk, col) in zip(FIG4_MECH, FIG4_MECH_STYLE)],
              loc="upper left", bbox_to_anchor=(-0.02, -0.34 / A2), fontsize=5.4, ncol=1, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
    # e: by forecast week against the carrier alone
    bx = fig.add_subplot(gs[1], sharey=ax); axes.append(bx)
    wk = lambda s_, w: next((r for r in lw if r["set"] == s_ and int(r["week"]) == w), None)       # noqa: E731
    pts = [pct(float(r[f"{a}_vs_carrier_log_ratio"])) for s_ in sets for w in range(1, 5) for r in [wk(s_, w)] if r for a in ("ankyra_lite", "fixed_division")]
    xl = lims(pts, (-6.0, 6.0), (-40.0, 30.0), step=5.0, margin=2.0)
    bx.axvline(0, color="#7F7F7F", lw=0.6)
    for yy, s_ in zip(y, sets):
        for w, dy in zip(range(1, 5), (0.3, 0.1, -0.1, -0.3)):
            r = wk(s_, w)
            if r is None:
                continue
            lr, lo, hi = (float(r[f"ankyra_lite_vs_carrier_{k}"]) for k in ("log_ratio", "um_low", "um_high"))
            ci(bx, yy + dy, pct(lr), pct(hi), pct(lo), xl, WEEK_RAMP_LITE[w - 1], hi < 0 or lo > 0, s=9, lw=0.7, fs=4.8)
            fl, flo, fhi = (float(r[f"fixed_division_vs_carrier_{k}"]) for k in ("log_ratio", "um_low", "um_high"))
            ci(bx, yy + dy, pct(fl), None, None, xl, WEEK_RAMP_GREY[w - 1], fhi < 0 or flo > 0, s=8, marker="D", fs=4.8, z=2)
    bx.set_xlim(*xl)
    frame(bx, "e   By week, vs the carrier alone", "Improvement over the carrier alone (%)")
    bx.tick_params(axis="y", left=False, labelleft=False)
    bx.legend(handles=[Line2D([], [], marker="o", ls="", color=c_, markersize=3.2, label=f"ankyra-lite, week {w}") for w, c_ in enumerate(WEEK_RAMP_LITE, 1)]
              + [Line2D([], [], marker="D", ls="", color=WEEK_RAMP_GREY[2], markersize=3.0, label="fixed division,\nweeks 1-4")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.34 / A2), fontsize=5.4, ncol=2, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3,
              columnspacing=0.8)
    # f: the estimated handover weight
    cx = fig.add_subplot(gs[2], sharey=ax); axes.append(cx)
    cx.axvline(0.5, color="#BDBDBD", lw=0.6, ls=(0, (2, 2)))
    for yy, s_ in zip(y, sets):
        for w, dy in zip(range(1, 5), (0.3, 0.1, -0.1, -0.3)):
            r = wk(s_, w)
            if r is None or r.get("mean_weight_on_carrier") in ("", None):
                continue
            cx.scatter([float(r["mean_weight_on_carrier"])], [yy + dy], s=9, color=WEEK_RAMP_CARRIER[w - 1], edgecolor=WEEK_RAMP_CARRIER[w - 1], lw=0.6, zorder=3)
    cx.set_xlim(0.0, 1.0)
    cx.text(0.51, y[-1] - 0.55, "prior ½", fontsize=5.2, color="#9E9E9E", ha="left", va="center")
    frame(cx, "f   Handover weight", "Mean weight on the carrier's daily means")
    cx.tick_params(axis="y", left=False, labelleft=False)
    cx.legend(handles=[Line2D([], [], marker="o", ls="", color=c_, markersize=3.2, label=f"week {w}") for w, c_ in enumerate(WEEK_RAMP_CARRIER, 1)],
              loc="upper left", bbox_to_anchor=(-0.02, -0.34 / A2), fontsize=5.4, ncol=2, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
    ax.set_yticks(y); ax.set_yticklabels([pop(s_, flags) for s_ in sets], fontsize=6.4); ax.tick_params(axis="y", length=0)
    return axes


def abl_styles(names):
    """{variant: (marker, colour)}: ABL_STYLE in order, then marker/colour pairs not used before."""
    out = list(ABL_STYLE[:len(names)])
    if len(names) > len(ABL_STYLE):
        print(f"  warning: fig4 panel with {len(names)} variants and {len(ABL_STYLE)} styles: markers and colours are reused in new pairs")
        marks, cols = [m for m, _ in ABL_STYLE], [c for _, c in ABL_STYLE]
        extra = [(marks[i], cols[(i + k) % len(cols)]) for k in range(1, len(cols)) for i in range(len(marks))]
        out += extra[:len(names) - len(out)]
        if len(out) < len(names):
            raise Skipped(f"fig4 panel with {len(names)} variants: more than {len(marks) * len(cols)} distinct styles needed")
    return dict(zip(names, out))


def fig_ablation():
    ab = [r for r in rows("ablation.csv") if r.get("error", "hourly") in ("", "hourly")]
    flags, note = flag_note(ab)
    names = []
    for r in ab:
        if r["ablation"] not in names:
            names.append(r["ablation"])
    v2 = release020()
    # one seed's carrier (0.1.x) or one three-seed group of the six seeds (0.2.0, evaluation/extra_tables.py), anchored
    seeds = [a for a in names if re.search(r"\bseed \d", a.lower()) or a.lower().startswith("three-seed group")]
    alts = [a for a in names if a not in seeds and a.strip().endswith("(anchored)")]          # other carriers, recipes, configurations
    mech_rows = {key for kind, key, _ in FIG4_MECH if kind == "ablation"}
    main = [a for a in names if a not in seeds and a not in alts and a not in mech_rows]     # parts of ankyra-lite switched off
    groups_v2 = any(a.lower().startswith("three-seed group") for a in seeds)
    groups = [(g, t) for g, t in ((main, "a   Anchoring, state rule, continuation" if v2 else "a   Anchoring layer and state rule"),
                                  (seeds, "b   Three seeds instead of six" if groups_v2 else "b   One seed instead of three"),
                                  (alts, "c   Other carriers, recipes, configurations" if v2 else "c   Other carriers and recipes")) if g]
    sets = present(ALL, ab)
    try:                                                                                       # second row: handover and lead weeks
        mech = bool(rows("handover_granularity.csv")) and bool(rows("lead_weeks.csv")) and any(r["ablation"] in mech_rows for r in ab)
    except Missing:
        mech = False
        print("  note: handover_granularity.csv or lead_weeks.csv not found; fig4 is drawn without its second row")

    def short(a):
        s_ = ABL_NICE.get(a.strip(), a.strip())
        m_ = ABL_CONFIG.fullmatch(s_)
        if m_:                                    # 0.2.0 configuration rows (evaluation/extra_tables.py config_label)
            return (("0.1.0: " if m_.group(1) else "") + ("Rw" if "Rw" in m_.group(2) else "R") + ", "
                    + ("6 seeds" if m_.group(3) == "six seeds" else "seeds 0–2") + (", with layer" if m_.group(4) == "with" else ", no layer"))
        if s_.endswith("(anchored)"):
            s_ = s_[:-len("(anchored)")].strip()
        elif " (" in s_:
            s_ = s_.split(" (")[0]
        return "\n".join(textwrap.wrap(s_, 36, break_on_hyphens=False))

    cap = wrap(("ankyra-lite 0.2.0 (the carrier of recipe Rw with six seeds averaged, anchored, weekly seasonal naive on state windows, then "
                "the origin-continuity layer) " if v2 else
                "ankyra-lite (the carrier of the released recipe R with three seeds averaged, anchored, weekly seasonal naive on state windows) ")
               + "against reduced and alternative versions of itself, hourly error on the test windows. Unit-equal improvement with 95% "
               "unit-and-month intervals; filled = interval excludes zero; positive = ankyra-lite is better. Shaded: the six release test "
               "populations." + (" c: recipes F and V are those of figure 18; every other carrier and recipe is anchored by the same layer."
                                 if alts else "")
               + (" Configurations: recipe R or Rw, seeds 0–2 or six seeds, with or without the continuation layer (0.1.0: recipe R, "
                  "seeds 0–2, no layer)." if any(ABL_CONFIG.fullmatch(a.strip()) for a in alts) else "")
               + (" d: ankyra-lite against the handover arms (the fixed division without any handover, a fixed half weight on the carrier's daily "
                  "means, one weight per window instead of the weekly weights) and against one whole-window combination weight and the "
                  "half-and-half average of the history side (fixed-division daily means plus the analog-day shape) and the carrier. e: "
                  "ankyra-lite (red, weeks 1-4 light to dark, with intervals) and the fixed division (grey diamonds, point estimates; filled = "
                  "resolved) against the carrier alone by forecast week. f: the mean handover weight on the carrier's daily means by week "
                  "(½ = the prior without evidence)." if mech else ""))
    leg_lines = max(sum(short(a).count("\n") + 1 for a in g) for g, _ in groups)
    most = max(len(g) for g, _ in groups)
    per_pop = 0.5 if most >= 8 else (0.42 if most >= 5 else 0.3)
    leg_h = 0.1 * leg_lines + 0.1
    leg2_h = 0.62 if mech else 0.0
    row2 = (0.36 + 0.3 * len(sets) + 0.42 + leg2_h) if mech else 0.0                          # title gap, axes, x label, legends
    H = 0.75 + per_pop * len(sets) + cap_height(cap) + leg_h + (0.12 if note else 0) + row2
    fig = plt.figure(figsize=(7.2, H))
    ratios = [1.25 if g is main else (0.8 if g is seeds else 1.0) for g, _ in groups]
    gs = fig.add_gridspec(1, len(groups), width_ratios=ratios, wspace=0.08, left=0.105, right=0.985,
                          top=1 - (cap_height(cap) + 0.25) / H, bottom=(0.42 + leg_h + (0.12 if note else 0) + row2) / H)
    y = np.arange(len(sets))[::-1]
    style = {}
    for g, _ in groups:
        style.update(abl_styles(g))
    axes = []
    for gi, (grp, title) in enumerate(groups):
        ax = fig.add_subplot(gs[gi], sharey=axes[0] if axes else None); axes.append(ax)
        nt = len([s for s in sets if s in TEST])
        if nt and nt < len(sets):
            ax.axhspan(y[nt - 1] - 0.5, y[0] + 0.5, color="#F3F6FA", zorder=0)
        ax.axvline(0, color="#7F7F7F", lw=0.6)
        offs = np.linspace(0.33, -0.33, len(grp)) if len(grp) > 1 else [0.0]
        for a, dy in zip(grp, offs):
            mk, col = style[a]
            for yy, s in zip(y, sets):
                r = next((rr for rr in ab if rr["set"] == s and rr["ablation"] == a), None)
                if r is None or r["log_ratio"] in ("", None):
                    continue
                p, lo, hi = float(r["improvement_pct"]), pct(float(r["um_high"])), pct(float(r["um_low"]))
                sig = float(r["um_high"]) < 0 or float(r["um_low"]) > 0
                ax.plot([lo, hi], [yy + dy, yy + dy], color=col, lw=0.8)
                ax.scatter([p], [yy + dy], marker=mk, s=(10 if len(grp) >= 5 else 14) * ABL_MARK_SCALE.get(mk, 1.0),
                           color=col if sig else "white", edgecolor=col, lw=0.7, zorder=3)
        ax.set_title(title, fontsize=7.2)
        ax.set_xlabel("ankyra-lite improvement (%)", fontsize=6.5)
        ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
        ax.legend(handles=[Line2D([], [], marker=style[a][0], ls="", color=style[a][1], markersize=3.6 * ABL_MARK_SCALE.get(style[a][0], 1.0) ** 0.5,
                                  label=short(a)) for a in grp],
                  loc="upper left", bbox_to_anchor=(-0.02, -0.34 / (per_pop * len(sets))), fontsize=5.4, ncol=1, handletextpad=0.2,
                  borderaxespad=0.1, labelspacing=0.3)
        if gi:
            ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([pop(s, flags) for s in sets], fontsize=6.4); axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(y[-1] - 0.7, y[0] + 0.7)
    if mech:
        low = (0.42 + leg2_h + (0.12 if note else 0)) / H
        gs2 = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.8], wspace=0.08, left=0.105, right=0.985,
                               top=low + 0.3 * len(sets) / H, bottom=low)
        m_axes = _fig4_mechanism(fig, gs2, sets, flags, y, None)
        for a_ in m_axes:
            a_.set_ylim(y[-1] - 0.7, y[0] + 0.7)
    put_caption(fig, cap, H)
    if note:
        put_footer(fig, [note], H)
    save(fig, "fig4_ablation")


# ============================================================================ Figure 5: peak readout
CARRIER_PEAK = "same readout on the carrier alone"           # a row of peak_readout.csv that is not a baseline (not counted in c)


def fig_peak():
    pk = rows("peak_readout.csv")
    flags, note = flag_note(pk)
    sets = present(ALL, pk)
    cap = [r"Peak readout  $U=\mathrm{max}_d\,(\hat L_d+A_{\tau_d})$: ankyra-lite's daily means plus the largest recent excursion of the "
           "same day type (the anchoring layer's readout)."]
    nb_ = max((len([r for r in pk if r["set"] == s and r["comparator"].startswith("same readout on ") and not excluded(r["comparator"][16:])
                    and r["comparator"] != CARRIER_PEAK]) for s in sets), default=0)
    cap += wrap("Unit-equal improvement in absolute peak error with 95% unit-and-month intervals. a, b: blue better, red worse, open or grey "
                "not resolved. a: against the maximum of ankyra-lite's own delivered trajectory (after the continuation layer); the "
                "readout U is read before the layer (readout rule A). b: against the previous month's observed peak. c: against the same readout applied to each baseline's trajectory (the release rule)"
                + (f", all {nb_} baselines ({nb_ + 1} forecasters)" if nb_ else "") + ". Bold: the six release test populations (shaded). "
                + notes() + note)
    H = 2.25 + cap_height(cap) + 0.05
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.2, H), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0, 0.95]})
    fig.subplots_adjust(left=0.105, right=0.99, top=1 - (cap_height(cap) + 0.42) / H, bottom=0.45 / H, wspace=0.1)
    y = np.arange(len(sets))[::-1]
    nt = len([s for s in sets if s in TEST])
    xl_b = lims([float(r["improvement_pct"]) for r in pk if r["set"] in sets and r["comparator"] == "previous-window peak"], (-60, 40), (-100, 80),
                step=10)
    for a_, comp, title in ((ax, "trajectory maximum", "a   vs its own trajectory maximum"),
                            (bx, "previous-window peak", "b   vs last month's observed peak")):
        if nt < len(sets):
            a_.axhspan(y[nt - 1] - 0.5, y[0] + 0.5, color="#F3F6FA", zorder=0)
        a_.axvline(0, color="#7F7F7F", lw=0.6)
        for yy, s in zip(y, sets):
            r = next((rr for rr in pk if rr["set"] == s and rr["comparator"] == comp), None)
            if r is None:
                continue
            p, lo, hi = float(r["improvement_pct"]), pct(float(r["um_high"])), pct(float(r["um_low"]))
            better, worse = float(r["um_high"]) < 0, float(r["um_low"]) > 0
            col = "#2166AC" if better else (LITE if worse else "#8C8C8C")
            if comp == "trajectory maximum":
                a_.barh(yy, p, color=col, height=0.6, alpha=0.9)
                a_.plot([lo, hi], [yy, yy], color="#262626", lw=0.7)
            else:
                ci(a_, yy, p, lo, hi, xl_b, col, better or worse, lw=1.0)
        a_.set_title(title, fontsize=7.2)
        a_.set_xlabel("Improvement in monthly peak error (%)", fontsize=6.5)
    bx.set_xlim(*xl_b)
    # c: the same readout applied to each baseline's trajectory
    nb = 0
    for yy, s in zip(y, sets):
        R = [r for r in pk if r["set"] == s and r["comparator"].startswith("same readout on ") and not excluded(r["comparator"][16:])
             and r["comparator"] != CARRIER_PEAK]                                     # baselines only (not the carrier alone)
        nb = max(nb, len(R))
        better = sum(float(r["um_high"]) < 0 for r in R); worse = sum(float(r["um_low"]) > 0 for r in R)
        cx.barh(yy, better, color="#2166AC", height=0.6); cx.barh(yy, len(R) - better - worse, left=better, color="#D9D9D9", height=0.6)
        cx.barh(yy, worse, left=len(R) - worse, color=LITE, height=0.6)
        if better:
            cx.text(better - 0.3, yy, f"{better}", va="center", ha="right", fontsize=5.8, color="white", fontweight="bold")
        if worse:
            cx.text(len(R) + 0.3, yy, f"{worse}", va="center", ha="left", fontsize=5.8, color=LITE, fontweight="bold")
    cx.set_xlim(0, nb + 2); cx.set_xticks(range(0, nb + 1, 5)); cx.set_xlabel(f"Number of the {nb} baselines", fontsize=6.5)
    cx.set_title("c   vs each baseline, same readout", fontsize=7.2)
    cx.legend(handles=[Patch(fc="#2166AC", label="better"), Patch(fc="#D9D9D9", label="not resolved"), Patch(fc=LITE, label="worse")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=3, fontsize=5.7, handlelength=1.0, handleheight=0.8, columnspacing=0.6,
              handletextpad=0.3)
    ax.set_yticks(y); ax.set_yticklabels([pop(s, flags) for s in sets], fontsize=6.3)
    for a_ in (ax, bx, cx):
        a_.tick_params(axis="y", length=0)
    for t_, s in zip(ax.get_yticklabels(), sets):
        if s in TEST:
            t_.set_fontweight("bold")
    put_caption(fig, cap, H)
    save(fig, "fig5_peak_operator")


# ============================================================================ Figure 6: one test window
FC_NAME = {"timesfm_kw": "TimesFM 2.5", "chronos2_kw": "Chronos-2", "chronos_2_kw": "Chronos-2"}
PRE_COL = "ankyra_before_layer_kw"                 # 0.2.0 (evaluation/extra_tables.py): ankyra-lite before the continuation layer
LITE_PRE = "#E08A8F"                               # ankyra-lite before the continuation layer (a light tint of LITE)


def _example():
    ex = rows("example_window_cambridge.csv")
    try:
        meta = json.loads(src("example_window_cambridge.json").read_text(encoding="utf-8"))
    except Missing:
        meta = {}
    h = np.array([int(float(r["hour"])) for r in ex]); y = np.array([float(r["load_kw"]) if r["load_kw"] else np.nan for r in ex])
    cols = [c for c in ex[0] if c not in ("hour", "load_kw")]
    ref = next((c for c in cols if "lite" in c.lower()), "ankyra_kw" if "ankyra_kw" in cols else None)   # ankyra_kw: the release's name
    if ref is None:
        raise Missing("example_window_cambridge.csv: an ankyra-lite column")
    car = next((c for c in cols if "carrier" in c.lower()), None)
    pre = PRE_COL if PRE_COL in cols else None                    # 0.2.0: ankyra-lite before the continuation layer (F)
    others = [c for c in cols if c not in (ref, car, pre)]
    F = {c: np.array([float(r[c]) if r[c] else np.nan for r in ex]) for c in cols}
    styles = [(ref, "ankyra-lite $F'$ (delivered)" if pre else "ankyra-lite", LITE, 0.85, "-")]
    if pre:
        styles.append((pre, "ankyra-lite $F$ (before the continuation layer)", LITE_PRE, 0.8, (0, (2.4, 1.3))))
    if car:
        styles.append((car, "carrier alone (not anchored)", CARRIER, 0.75, "-"))
    palette = ["#807DBA", "#9E9E9E", "#8C6D31"]
    for i, c in enumerate(others):
        styles.append((c, FC_NAME.get(c, c.replace("_kw", "")), palette[i % len(palette)], 0.7, "-"))
    return h, y, F, styles, meta, car, pre


def fig_example():
    """Figure 6: one Cambridge test window. With 0.2.0 results (the column ankyra_before_layer_kw) ankyra-lite is drawn
    before (F) and after (F', delivered) the origin-continuity layer, and a panel zooms into the first three days."""
    h, y, F, styles, meta, _, pre = _example()
    keep = h >= -336
    h, y = h[keep], y[keep]
    F = {k: v[keep] for k, v in F.items()}
    fut = h >= 0
    ref = styles[0][0]
    src_note = meta.get("source", "University of Cambridge estate building energy archive, CC BY 4.0")
    lead = ("The carrier's own forecast at the origin; ankyra-lite after anchoring (F) and after the origin-continuity layer (F', the delivered "
            "trajectory; c: the first three days). " if pre else "The carrier's own forecast at the origin and ankyra-lite after anchoring. ")
    foot = (lead + recipe_note() + " Data: " + src_note.split(" (")[0] + (" (CC BY 4.0)." if "CC BY" in src_note else "."))
    if meta.get("selection"):
        foot += " Window: " + meta["selection"][0].lower() + meta["selection"][1:] + "."
    foot = wrap(foot, fs=5.6, width=165)
    fh = 0.06 + len(foot) * 5.6 * 1.36 / 72                     # the footer block, kept clear of panel b's axis label
    H = 3.4 + fh                                               # panels as tall as before the footer grew
    fig = plt.figure(figsize=(7.2, H))
    if pre:
        gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1.25, 0.95, 0.95], hspace=0.62, wspace=0.3, left=0.075,
                              right=0.99, top=1 - 0.504 / H, bottom=(0.40 + fh) / H)
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.6, 1], hspace=0.62, wspace=0.22, left=0.075, right=0.99,
                              top=1 - 0.504 / H, bottom=(0.40 + fh) / H)
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(h[0], 0, color="#F2F2F2", zorder=0)
    ax.plot(h[~fut], y[~fut], color="#7F7F7F", lw=0.55)
    ax.plot(h[fut], y[fut], color="#1A1A1A", lw=0.55, label="observed")
    for c, name, col, lw, ls in styles[::-1]:
        ax.plot(h[fut], F[c][fut], color=col, lw=lw, ls=ls, label=name, zorder=3 if c == ref else 2)
    ax.axvline(0, color="#4D4D4D", lw=0.6, ls=(0, (2, 2)))
    top = np.nanmax(np.concatenate([y] + [F[c][fut] for c, *_ in styles]))
    ax.set_ylim(top=top * 1.08)
    ax.text(-6, top * 1.05, "context", ha="right", va="top", fontsize=6.0, color="#7F7F7F")
    ax.text(6, top * 1.05, "forecast origin", ha="left", va="top", fontsize=6.0, color="#4D4D4D")
    ax.set_xlim(h[0], h[-1]); ax.set_xlabel("Hours from the forecast origin", fontsize=6.6); ax.set_ylabel("kW", fontsize=6.6)
    hd, lb = ax.get_legend_handles_labels()
    order = [lb.index("observed")] + [lb.index(name) for _, name, *_ in styles]
    ax.legend([hd[i] for i in order], [lb[i] for i in order], loc="lower right", ncol=len(order), fontsize=6.0,
              bbox_to_anchor=(1.0, 0.99), handlelength=1.6, borderaxespad=0.1, columnspacing=1.0 if pre else 1.2)
    bld = str(meta.get("building", "")).split("_")[-1]
    title = f"a   {meta.get('set', 'Cambridge')} test set" + (f" · building {bld}" if bld else "") + \
        (f" · forecast for {meta['forecast_period']}" if meta.get("forecast_period") else "")
    ax.set_title(title, fontsize=7.4, pad=15)
    d = np.arange(1, 32)
    dm = lambda v: v[fut][:744].reshape(31, 24).mean(1)                              # noqa: E731
    bx = fig.add_subplot(gs[1, 0])
    bx.step(d, dm(y), where="mid", color="#1A1A1A", lw=0.9, label="observed")
    for c, name, col, _, ls in styles[::-1]:
        bx.step(d, dm(F[c]), where="mid", color=col, lw=0.9, ls=ls, label=name)
    bx.set_xlim(0.5, 31.5); bx.set_xlabel("Forecast day", fontsize=6.6); bx.set_ylabel("Daily mean (kW)", fontsize=6.6)
    bx.set_title("b   Daily means: level and daily path", fontsize=7.4)
    if pre:
        _example_first_days(fig.add_subplot(gs[1, 1]), h, y, F, ref, pre)
    cx = fig.add_subplot(gs[1, 2 if pre else 1])
    wk = [(0, 7), (7, 14), (14, 21), (21, 31)]
    x = np.arange(4); nsty = len(styles); w = 0.8 / nsty
    for j, (c, name, col, _, ls) in enumerate(styles[::-1]):
        r_ = [np.sqrt(np.nanmean((F[c][fut][a * 24:b * 24] - y[fut][a * 24:b * 24]) ** 2)) for a, b in wk]
        cx.bar(x - 0.4 + w * (j + 0.5), r_, w * 0.92, color=col, label=name, hatch="//////" if c == pre else None,
               edgecolor="white" if c == pre else None, lw=0)
    cx.set_xticks(x); cx.set_xticklabels(["days 1–7", "8–14", "15–21", "22–31"], fontsize=6.2); cx.set_ylabel("Hourly RMSE (kW)", fontsize=6.6)
    cx.set_title(("d" if pre else "c") + "   Error by forecast week", fontsize=7.4)
    if not pre:                                                    # with the continuation layer, panel a's legend names every bar colour
        hd, lb = cx.get_legend_handles_labels()
        cx.legend(hd[::-1], lb[::-1], fontsize=5.6, loc="upper left", handlelength=1.2, ncol=1, labelspacing=0.25)
        cx.set_ylim(top=cx.get_ylim()[1] * (1.15 + 0.12 * nsty))
    put_footer(fig, foot, H, fs=5.6, x=0.075)
    save(fig, "fig6_example_window")


def _example_first_days(ax, h, y, F, ref, pre, before=48, after=72):
    """Figure 6c (0.2.0): the last two observed days and the first three forecast days, ankyra-lite before (F) and after
    (F') the continuation layer; the hourly RMSE of forecast day 1 for both."""
    win = (h >= -before) & (h < after)
    hw, yw = h[win], y[win]
    past, fut = hw < 0, hw >= 0
    ax.axvspan(-before, 0, color="#F2F2F2", zorder=0)
    ax.plot(hw[past], yw[past], color="#7F7F7F", lw=0.7)
    ax.plot(hw[fut], yw[fut], color="#1A1A1A", lw=0.7)
    f0, f1 = F[pre][win][fut], F[ref][win][fut]
    ax.fill_between(hw[fut], f0, f1, color=LITE, alpha=0.13, lw=0, zorder=1)
    ax.plot(hw[fut], f0, color=LITE_PRE, lw=0.9, ls=(0, (2.4, 1.3)), zorder=3)
    ax.plot(hw[fut], f1, color=LITE, lw=1.1, zorder=4)
    ax.axvline(0, color="#4D4D4D", lw=0.6, ls=(0, (2, 2)))
    for x_ in (24, 48):
        ax.axvline(x_, color="#E6E6E6", lw=0.5, zorder=0)
    day1 = (h >= 0) & (h < 24)
    r0 = np.sqrt(np.nanmean((F[pre][day1] - y[day1]) ** 2))
    r1 = np.sqrt(np.nanmean((F[ref][day1] - y[day1]) ** 2))
    ax.text(0.985, 0.04, f"day-1 RMSE: $F$ {r0:.1f} kW, $F'$ {r1:.1f} kW", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
            color=LITE, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(-before, after); ax.set_xticks([-48, -24, 0, 24, 48, 72])
    lo = np.nanmin(np.concatenate([yw, f0, f1])); hi = np.nanmax(np.concatenate([yw, f0, f1]))
    ax.set_ylim(max(0.0, lo - 0.25 * (hi - lo)), hi + 0.08 * (hi - lo))
    ax.set_xlabel("Hours from the forecast origin", fontsize=6.6); ax.set_ylabel("kW", fontsize=6.6)
    ax.set_title("c   First three days: $F$ and $F'$", fontsize=7.4)


# ============================================================================ Figure 7: exact properties of the anchoring layer
def _operators():
    """theory/operators.py of this repository, or of the original ANKYRA checkout, with ``ankyra`` resolved to the
    byte-identical copy ``ankyra_lite.anchor`` while it is imported (nothing is written to either checkout)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import ankyra_lite.anchor as anchor
    from ankyra_lite.anchor import blocks
    orig = Path(os.environ.get("ANKYRA_ORIGINAL_ROOT", str(ROOT.parent / "Building_load_ANKYRA")))
    path = next((p for p in (ROOT / "theory" / "operators.py", orig / "theory" / "operators.py") if p.exists()), None)
    if path is None:
        raise Missing("theory/operators.py")
    saved = {k: sys.modules.get(k) for k in ("ankyra", "ankyra.blocks")}
    sys.modules["ankyra"], sys.modules["ankyra.blocks"] = anchor, blocks
    try:
        spec = importlib.util.spec_from_file_location("_ankyra_lite_figure_operators", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, path


def fig_operators():
    h, y_all, F, styles, meta, car, pre = _example()
    op, op_path = _operators()
    import torch
    from ankyra_lite.anchor import blocks, readouts
    from ankyra_lite.anchor.history import estimate_from_history
    from ankyra_lite.anchor.synthetic import synthetic_history
    torch.set_num_threads(1)
    if "context_day_types" not in meta or "target_day_types" not in meta:
        raise Missing("example_window_cambridge.json (context_day_types, target_day_types)")
    fut = h >= 0
    y = y_all[fut][:744]; ctx = y_all[~fut][-1344:]
    A = F[pre or styles[0][0]][fut][:744]           # the anchoring layer's output (0.2.0: before the continuation layer)
    base_col = car or next((s_[0] for s_ in styles[1:] if s_[0] != pre), None)
    if base_col is None:
        raise Missing("example_window_cambridge.csv: a carrier column")
    T = F[base_col][fut][:744]
    base_name = "carrier alone" if car else next(s_[1] for s_ in styles[1:] if s_[0] == base_col)
    ct = np.array(meta["context_day_types"])[None]; tt = np.array(meta["target_day_types"])[None]
    exc = readouts.historical_excursions(ctx[None], ct, tt)[0]

    fig = plt.figure(figsize=(7.2, 4.85))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.55, left=0.085, right=0.985, top=0.87 if pre else 0.9, bottom=0.095)   # 0.2.0: a third caption line

    # a  exact accounting of block replacement on the window (P4)
    ax = fig.add_subplot(gs[0, 0])
    lvA, bpA = blocks.level(A), blocks.daily_path(A)
    arms = [(base_name, T), ("+ anchored level", op.replace_blocks(T, level=lvA)),
            ("+ anchored daily path", op.replace_blocks(T, level=lvA, daily_path=bpA)), ("+ anchored within-day\n(= ankyra-lite $F$)" if pre else "+ anchored within-day\n(= ankyra-lite)", A)]
    cols = ["#C0762A", "#E8B67A", "#807DBA"]; names = ["level", "daily path", "within-day"]
    tot0 = max(sum(blocks.block_losses(Fv, y)) for _, Fv in arms)
    for i, (lab, Fv) in enumerate(arms):
        parts = blocks.block_losses(Fv, y); left = 0.0
        for j, v in enumerate(parts):
            ax.barh(i, v, left=left, color=cols[j], height=0.62, label=names[j] if i == 0 else None, lw=0); left += v
        ax.text(left + tot0 * 0.02, i, f"{left:.0f}", va="center", fontsize=6.0)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([a_[0] for a_ in arms], fontsize=5.9); ax.invert_yaxis()
    ax.set_xlim(0, tot0 * 2.0); ax.set_xlabel("Hourly MSE on the window (kW²)", fontsize=6.5)   # room for the legend right of the totals
    ax.legend(fontsize=5.5, loc="lower right", handlelength=0.9, borderaxespad=0.1)
    ax.set_title("a   Replacement: block losses add", fontsize=7.3)

    # b  historical support (P7)
    bx = fig.add_subplot(gs[0, 1])
    m = np.arange(0, 24 * 760, 6)
    share = np.array([op.data_share(op.effective_pseudo_origins(x)) for x in m])
    bx.step(m / 730.0, share, where="post", color="#4A6FA5", lw=1.1)
    for k in (2, 4, 8, 12):
        mm = 1344 + 744 * k
        bx.plot([mm / 730.0], [op.data_share(k)], "o", ms=2.4, color="#4A6FA5")
        bx.text(mm / 730.0 + 0.35, op.data_share(k) - 0.045, f"K={k}", fontsize=5.5, color="#4A6FA5")
    ya = op.annual_support_hours(2) / 730.0
    bx.axvspan(0, ya, color="#F3F3F3", lw=0, zorder=0)
    bx.axvline(ya, color=GREY, lw=0.6, ls=(0, (2, 2)))
    bx.text(ya + 0.4, 0.13, "annual candidate\nweighted from\n10,248 h (14 mo)", fontsize=5.4, color=GREY, va="bottom")
    bx.text(0.4, 0.66, "equal weights\nbelow 2 errors", fontsize=5.4, color=GREY, va="top")
    bx.set_xlim(0, 24); bx.set_ylim(0, 0.7)
    bx.set_xlabel("History before the origin (months)", fontsize=6.5)
    bx.set_ylabel(r"Data share  $K_{\mathrm{eff}}/(K_{\mathrm{eff}}+8)$", fontsize=6.5)
    bx.set_title("b   Support: how much history", fontsize=7.3)

    # c  shrinkage bounds of the weather weight (P8, P9)
    cx = fig.add_subplot(gs[0, 2])
    ks = np.arange(2, 13); lo = [op.weather_weight_bounds(x)[0] for x in ks]; hi = [op.weather_weight_bounds(x)[1] for x in ks]
    cx.fill_between(ks, lo, hi, color="#EEF3FA", lw=0, label="bound (P8)")
    cx.plot(ks, lo, color="#9DB4D6", lw=0.8); cx.plot(ks, hi, color="#9DB4D6", lw=0.8)
    pts = []
    for hours in (2900, 3700, 4392, 5200, 6000, 6576, 7400, 8200, 9000, 9800, 10600):
        dg = estimate_from_history(synthetic_history(hours), group="Office", temp_sigma_std=0.25).diagnostics
        pts.append((dg["level_k_eff"], dg["weather_weight"]))
    cx.scatter([p_[0] for p_ in pts], [p_[1] for p_ in pts], s=11, color=LITE, zorder=3, lw=0, label="reference estimator")
    cx.scatter([2], [2 / 3], s=16, facecolors="white", edgecolors="#1A1A1A", lw=0.8, zorder=4, label="after deletion (P9)")
    cx.annotate("2/3 > 0.6", (2, 2 / 3), (3.3, 0.9), fontsize=5.5, color="#1A1A1A", arrowprops=dict(arrowstyle="-", lw=0.5, color="#1A1A1A"))
    cx.axhline(0.5, color="#BDBDBD", lw=0.6, ls=(0, (2, 2)))
    cx.set_xlabel("Matched error support k", fontsize=6.5); cx.set_ylabel("Total weather weight", fontsize=6.5)
    cx.set_ylim(0, 1); cx.set_xlim(1.5, 12.5); cx.set_xticks([2, 4, 6, 8, 10, 12])
    cx.legend(fontsize=5.4, loc="lower left", handlelength=1.0, borderaxespad=0.1)
    cx.set_title("c   Shrinkage bounds the weights", fontsize=7.3)

    # d  nonnegativity: clipping versus any energy-preserving projection (P11, P12)
    dx = fig.add_subplot(gs[1, 0])
    p0, yt = np.array([-1.0, 3.0]), np.array([0.0, 4.0])
    dx.add_patch(Rectangle((0, 0), 6, 6.1, color="#F4F4F4", lw=0, zorder=0))
    dx.text(5.85, 0.15, "nonnegative\nload", fontsize=5.4, color=GREY, ha="right", va="bottom")
    tline = np.linspace(-1.8, 2.8, 50)
    dx.plot(tline, 2 - tline, color="#7F7F7F", lw=0.7, ls=(0, (3, 2)))
    dx.plot([0, 2], [2, 0], color=FM, lw=2.0, solid_capstyle="butt")
    th = np.linspace(0, 2 * np.pi, 200); r0 = np.sqrt(2)
    dx.plot(yt[0] + r0 * np.cos(th), yt[1] + r0 * np.sin(th), color="#1A1A1A", lw=0.6)
    dx.plot(*p0, "o", color="#1A1A1A", ms=3.2); dx.text(-1.15, 2.78, "forecast", fontsize=5.7, ha="right", va="top")
    dx.plot(*yt, "*", color="#1A1A1A", ms=6); dx.text(0.16, 4.18, "truth", fontsize=5.7)
    dx.plot(0, 3, "o", color=LITE, ms=3.4)
    dx.annotate("clip: error 1", (0.06, 3.0), (1.75, 3.0), fontsize=5.7, color=LITE, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=LITE))
    dx.plot(0, 2, "o", color=FM, ms=3.2)
    dx.annotate("best nonnegative point\nwith the same energy:\nerror 4", (0.06, 2.0), (1.75, 2.05), fontsize=5.7, color=FM,
                va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=FM))
    dx.text(1.75, 6.05, "the same-energy line is\ntangent to the no-worse\ndisc at the forecast", fontsize=5.4, color=GREY, va="top")
    dx.set_xlim(-2.0, 6.0); dx.set_ylim(-0.3, 6.1); dx.set_aspect("equal")
    dx.axhline(0, color="#BDBDBD", lw=0.5); dx.axvline(0, color="#BDBDBD", lw=0.5)
    dx.set_xlabel("hour 1 (kW)", fontsize=6.5); dx.set_ylabel("hour 2 (kW)", fontsize=6.5)
    dx.set_title("d   Energy: clip, never re-balance", fontsize=7.3)

    # e  the peak readout on the window (P15)
    ex_ = fig.add_subplot(gs[1, 1])
    L = A.reshape(31, 24).mean(1); U = readouts.peak_readout(A, ctx[None], ct, tt)[0]
    d = np.arange(1, 32)
    ex_.bar(d, L, color="#F3C4C8", width=0.78, lw=0, label=r"daily mean $\hat L_d$")
    ex_.bar(d, exc, bottom=L, color=LITE, width=0.78, lw=0, alpha=0.85, label=r"+ excursion $A_{\tau_d}$")
    ex_.axhline(y.max(), color="#1A1A1A", lw=0.9, ls=(0, (4, 2)), label=f"observed peak {y.max():.0f}")
    ex_.axhline(U, color=LITE, lw=0.9, label=f"peak readout U = {U:.0f}")
    ex_.axhline(A.max(), color="#9E9E9E", lw=0.8, ls=(0, (1, 1.5)), label=f"trajectory max {A.max():.0f}")
    top = max(y.max(), U, A.max(), (L + exc).max())
    ex_.set_xlim(0.3, 31.7); ex_.set_ylim(0, top * 1.55)
    ex_.set_xlabel("Forecast day", fontsize=6.5); ex_.set_ylabel("kW", fontsize=6.5)
    hnd, lab = ex_.get_legend_handles_labels()
    order = [3, 4, 2, 0, 1]
    ex_.legend([hnd[k] for k in order], [lab[k] for k in order], fontsize=5.2, loc="upper left", ncol=2, handlelength=1.3,
               columnspacing=0.7, borderaxespad=0.1, labelspacing=0.3)
    ex_.set_title(r"e   Peak: $U=\mathrm{max}_d(\hat L_d+A_{\tau_d})$", fontsize=7.3)

    # f  the peak error decomposition on the same window (P16)
    fx = fig.add_subplot(gs[1, 2])
    dec = op.peak_error_decomposition(A, y, exc)
    terms = [("level", dec["level"], HIST), ("between-day", dec["between_day"], "#E8B67A"),
             ("amplitude", dec["amplitude"], "#807DBA"), ("selection", dec["selection"], "#9E9E9E")]
    run = 0.0
    span = max(abs(t_[1]) for t_ in terms) or 1.0
    for i, (nm, v, c) in enumerate(terms):
        fx.bar(i, v, bottom=run, color=c, width=0.62, lw=0)
        fx.text(i, run + v + (0.05 * span if v >= 0 else -0.05 * span), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.7)
        if i < len(terms) - 1:
            fx.plot([i + 0.31, i + 0.69], [run + v, run + v], color="#7F7F7F", lw=0.5)
        run += v
    fx.bar(4, run, color=LITE, width=0.62, lw=0)
    fx.text(4, run + (-0.05 * span if run < 0 else 0.05 * span), f"{run:+.1f}", ha="center", va="top" if run < 0 else "bottom", fontsize=5.7, color=LITE)
    fx.axhline(0, color="#1A1A1A", lw=0.6)
    fx.set_xticks(range(5)); fx.set_xticklabels([t_[0] for t_ in terms] + ["U − peak"], rotation=35, ha="right", fontsize=5.9)
    cum = np.concatenate([[0.0], np.cumsum([t_[1] for t_ in terms])])
    fx.set_ylim(min(0, cum.min()) - 0.3 * span, max(0, cum.max()) + 0.3 * span); fx.set_ylabel("kW", fontsize=6.5)
    fx.set_title("f   Peak error: four exact terms", fontsize=7.3)

    who = ("ankyra-lite 0.2.0 before the origin-continuity layer (F: carrier recipe Rw, six seeds averaged, anchored; the energy, level "
           "and peak readouts are read from F, readout rule A)" if pre else "ankyra-lite of the released recipe R (three seeds averaged)")
    put_caption(fig, wrap("a, e, f: the Cambridge test window of Figure 6 (University of Cambridge estate archive, CC BY 4.0), " + who
                          + ". b–d: exact properties of the anchoring layer, unchanged from ANKYRA 2.2.1. P-numbers: theory/README.md."), 4.85)
    save(fig, "fig7_operators")


# ============================================================================ Figure 8: loss metrics of every forecaster
LOSS_METRICS = [("RMSE_kW", "a   RMSE (kW, mean over units)", "kw"), ("MAE_kW", "b   MAE (kW, mean over units)", "kw"),
                ("CV_RMSE_pct", "c   CV(RMSE) (%, median unit)", "pct"), ("WAPE_pct", "d   WAPE (%, median unit)", "pct")]


def _loss_text(v, kind):
    if not np.isfinite(v):
        return "–"
    if v >= 1000:
        return f"{v:,.0f}"
    if kind == "pct":
        return f"{v:.1f}"
    return f"{v:#.3g}".rstrip(".")


def fig_loss_metrics():
    cm = rows("conventional_metrics_late.csv")
    rk = [r for r in rows("benchmark_mean_unit_rank.csv") if r["subset"] == "late"]
    flags, note = flag_note(cm)
    sets = present(ALL, cm)
    cls = {r["model"]: r["model_class"] for r in cm}
    lab = {r["model"]: r["model_label"] for r in cm}
    models, _ = ordered_by_rank(rk, present(ALL, rk))
    models = [m for m in models if m in cls]
    n, ns = len(models), len(sets)
    cmap = LinearSegmentedColormap.from_list("rel", [LITE, "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"])
    lim = 30.0
    cap = wrap(f"{n} forecasters on the same test windows of {num(ns)} populations. a, b: means over units, so larger units weigh more; c, d: the "
               "median unit. Bold: lowest in the column. Rows in the order of figure 2. " + notes(models) + note)
    H = 2.0 + 0.29 * n + cap_height(cap)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(2, 2, left=0.175, right=0.975, top=1 - (cap_height(cap) + 0.62) / H, bottom=0.9 / H, wspace=0.05, hspace=0.2)
    ia = models.index(REF)
    cell = {(r["set"], r["model"]): r for r in cm}
    nt = len([s for s in sets if s in TEST])
    for k, (col, title, kind) in enumerate(LOSS_METRICS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        V = np.array([[float(cell[(s, m)][col]) if (s, m) in cell and cell[(s, m)][col] else np.nan for s in sets] for m in models])
        rel = 100.0 * (V / V[ia] - 1.0)
        ax.imshow(np.clip(rel, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        best = np.nanargmin(V, 0)
        for i in range(n):
            for j in range(ns):
                ax.text(j, i, _loss_text(V[i, j], kind), ha="center", va="center", fontsize=4.8,
                        color="white" if abs(rel[i, j]) > 21 else "#1A1A1A", fontweight="bold" if best[j] == i else "normal")
        if nt < ns:
            ax.axvline(nt - 0.5, color="white", lw=2.0)
        ax.add_patch(Rectangle((-0.5, ia - 0.5), ns, 1, fill=False, ec=LITE, lw=1.3, zorder=6))
        ax.set_xticks(range(ns))
        ax.set_xticklabels([pop(s, flags) for s in sets], fontsize=5.5, rotation=35, ha="left", va="bottom", rotation_mode="anchor")
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2.0)
        ax.set_yticks(range(n)); ax.tick_params(axis="y", length=0, pad=8)
        if k % 2 == 0:
            ax.set_yticklabels([label(m, lab) for m in models], fontsize=6.0)
            for t_, m in zip(ax.get_yticklabels(), models):
                if m == REF:
                    t_.set_color(LITE); t_.set_fontweight("bold")
            for i, m in enumerate(models):
                ax.scatter([-0.8], [i], s=10, color=class_color(cls.get(m, ""), m), clip_on=False, marker="s", zorder=5)
        else:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7.3, pad=30)
    cax = fig.add_axes([0.30, 0.56 / H, 0.40, 0.09 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-30, -15, 0, 15, 30]); cb.set_ticklabels(["≤ −30", "−15", "0", "+15", "≥ +30"]); cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Loss relative to ankyra-lite in the same column (%): red lower than ankyra-lite, blue higher", fontsize=6.0, labelpad=2)
    fig.legend(handles=row_major(class_handles(cls, models, size=4.0), 4), loc="lower left", bbox_to_anchor=(0.01, -0.005), ncol=4, fontsize=5.7,
               handletextpad=0.25, columnspacing=0.9)
    put_caption(fig, cap, H)
    save(fig, "fig8_loss_metrics")


# ============================================================================ Figure 9: hourly loss by forecast day; Figure 9b: energy to date
DAY_LINES = [("TimesFM", "#807DBA", "-"), ("Chronos-2", "#807DBA", "--"), ("Chronos-2-X", "#1D91C0", "-"),
             ("TimesFM-X", "#1D91C0", "--"), ("TiDE", "#253494", "-"), ("iTransformer-X", "#253494", "--"),
             ("GBT-T", "#253494", ":"), ("RIDGE-L", "#8C6D31", "-")]


def _day_panels(csv_name, value_col):
    ld = rows(csv_name); lab = {r["model"]: r["model_label"] for r in ld}; out = []
    for s in present(ALL, ld):
        R = [r for r in ld if r["set"] == s]
        curve = {m: np.array([float(r[value_col]) for r in sorted((r for r in R if r["model"] == m), key=lambda r: int(r["day"]))])
                 for m in sorted({r["model"] for r in R})}
        n = R[0].get("n_units_gm") or R[0].get("units_in_U") or R[0].get("units")
        out.append((f"{s.replace(' 2017', '')} ({n} units)", curve, lab, len(curve)))
    return out


def _day_figure(panels, ylabel, energy, stem, caption):
    days = np.arange(1, 32)
    cap = wrap(caption)
    below = len(panels) % 2 == 0                      # legend under the grid; otherwise in the free slot
    nrow = len(panels) // 2 if below else (len(panels) + 1) // 2
    H = 2.15 * nrow + cap_height(cap) + (0.75 if below else 0.35)
    fig, axs = plt.subplots(nrow, 2, figsize=(7.2, H), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=1 - (cap_height(cap) + 0.25) / H, bottom=(0.95 if below else 0.4) / H, hspace=0.45,
                        wspace=0.17)
    flat = axs.ravel()
    named = {REF} | {m for m, *_ in DAY_LINES}
    for ax, (title, curve, lab, nf) in zip(flat, panels):
        top = (2.4 * curve[REF][6:].max()) if energy else 1.6 * curve[REF].max()
        low = 0 if energy else 0.88 * min(v.min() for v in curve.values())
        for m, v in curve.items():
            if m not in named:
                ax.plot(days, np.minimum(v, top * 1.2), color="#D0D0D0", lw=0.55, zorder=1)
        off = []
        for m, c, ls in DAY_LINES:
            if m not in curve:
                continue
            v = curve[m]
            if np.median(v) > top:
                off.append(label(m, lab)); continue
            ax.plot(days, v, color=c, ls=ls, lw=0.95, zorder=2)
        ax.plot(days, curve[REF], color=LITE, lw=1.9, zorder=3)
        for x in (7.5, 14.5, 21.5):
            ax.axvline(x, color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_xlim(1, 31); ax.set_ylim(low, top); ax.set_xticks([1, 7, 14, 21, 28, 31])
        ax.set_title(title, fontsize=7.2)
        if energy:
            lowest = int(sum(all(curve[REF][d] <= v[d] for v in curve.values()) for d in range(31)))
            ax.text(0.985, 0.04, f"ankyra-lite lowest of {nf} on {lowest} of 31 days", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.8, color=LITE)
        if off:
            ax.text(0.985, 0.97, "off scale: " + ", ".join(off), transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=GREY)
        ax.grid(axis="y", color="#EFEFEF", lw=0.4, zorder=0)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel, fontsize=6.6)
    for k, ax in enumerate(flat[:len(panels)]):
        if k >= len(panels) - 2:
            ax.set_xlabel("Forecast day", fontsize=6.8)
    for ax in flat[len(panels):]:
        ax.axis("off")
    lab = panels[0][2]
    handles = [Line2D([], [], color=LITE, lw=1.9, label="ankyra-lite")]
    handles += [Line2D([], [], color=c, ls=ls, lw=0.95, label=label(m, lab)) for m, c, ls in DAY_LINES if m in lab]
    handles += [Line2D([], [], color="#D0D0D0", lw=0.8, label="the other baselines")]
    if below:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.2, handlelength=2.4,
                   columnspacing=1.4, labelspacing=0.5)
    else:
        flat[len(panels)].legend(handles=handles, loc="center", ncol=2, fontsize=6.3, handlelength=2.4, columnspacing=1.2, labelspacing=0.7)
    put_caption(fig, cap, H)
    save(fig, stem)


def fig_lead_days():
    panels = _day_panels("lead_day_metrics.csv", "GM_CV_RMSE_pct")
    nf = max(p[3] for p in panels)
    _day_figure(panels, "CV(RMSE) of the day (%)", False, "fig9_loss_by_day",
                f"Loss by forecast day, {nf} forecasters on the same test windows of the six release test populations. For each unit, the day's "
                "CV(RMSE) is the RMSE of that day's 24 hours over the unit's mean load; curves are geometric means over one fixed set of units "
                "(all daily errors nonzero), the scale of the primary estimand. Vertical lines: week boundaries. "
                + notes(set().union(*(p[1] for p in panels))).rstrip())


def fig_energy_by_day():
    """Figure 9b: error of the energy delivered through each forecast day."""
    panels = _day_panels("lead_day_energy.csv", "GM_CV_cumulative_energy_pct")
    nf = max(p[3] for p in panels)
    _day_figure(panels, "Error of energy to date, CV (%)", True, "fig9b_energy_by_day",
                f"Error of the energy delivered through each forecast day, {nf} forecasters on the same test windows. For each unit and day d, "
                "the RMS over the unit's windows of the error of the mean load over days 1..d, divided by the unit's mean load; curves are "
                "geometric means over one fixed set of units, the aggregation of figure 9 applied to energy. Day 31 is the monthly energy error "
                "of figure 11. Computed after scoring. "
                + ("For ankyra-lite the energy to date is that of its trajectory before the continuation layer (readout rule A). " if release020() else "")
                + notes(set().union(*(p[1] for p in panels))).rstrip())


# ============================================================================ Figure 9c: the continuation layer by lead time
# The segments of results/continuation_layer.csv (evaluation/extra_tables.py, CONT_SEGMENTS), in lead order; "lead hours
# 0-3" is the study's name of hours 0, 1 and 2.
CONT_SEGMENTS = [("lead hour 0", "hour 0"), ("lead hours 0-3", "hours 0–2"), ("forecast day 1", "day 1"),
                 ("forecast days 2-7", "days 2–7"), ("forecast days 8-31", "days 8–31"), ("whole window", "whole window")]
CONT_ENERGY = "energy"                                       # the column key of the delivered trajectory's monthly energy
WORSE = LITE                                                 # resolved against the layer


def _cont_table():
    """{(set, column): row} of continuation_layer.csv (columns: the CONT_SEGMENTS keys of the hourly rows and CONT_ENERGY),
    the populations in the release order, and the pooled set's name (or None)."""
    raw = rows(CONT_FILE)
    need = {"set", "segment", "error", "improvement_pct", "um_low", "um_high", "resolved"}
    if not raw or not need <= set(raw[0]):
        raise Skipped(f"{CONT_FILE} lacks the columns {', '.join(sorted(need - set(raw[0] if raw else ())))}")
    tab = {}
    for r in raw:
        col = r["segment"] if (r.get("error") or "").strip() == "hourly" else (CONT_ENERGY if r["segment"] == "whole window" else None)
        if col is None or (col != CONT_ENERGY and col not in dict(CONT_SEGMENTS)):
            continue
        if (r["set"], col) in tab:
            raise Skipped(f"{CONT_FILE}: two rows for {r['set']}, {col}")
        tab[(r["set"], col)] = r
    sets = {s for s, _ in tab}
    pops = [s for s in ALL if s in sets]
    pooled = next((s for s in sets if s not in ALL and (s.startswith("all ") or "pooled" in s)), None)
    pops += sorted(s for s in sets if s not in ALL and s != pooled)          # a population outside the nine, kept
    if not pops and pooled is None:
        raise Skipped(f"{CONT_FILE} has no row of a known segment")
    return tab, pops, pooled


def _cont_pt(r):
    """(improvement %, interval low %, interval high %, sign of the resolved label) of a row, or None."""
    if r is None:
        return None
    p, lo, hi = _num(r.get("improvement_pct")), _num(r.get("um_low")), _num(r.get("um_high"))
    if p is None:
        return None
    res = (r.get("resolved") or "").strip()
    sign = 1 if res == "first better" else (-1 if res == "second better" else 0)
    return p, (pct(hi) if hi is not None else None), (pct(lo) if lo is not None else None), sign


def _cont_colour(v, sign, vmax):
    """Cell colour: blue for a lower error with the layer, red for a higher one, on a compressed scale (log1p of the
    absolute improvement in %); a cell whose interval includes zero is drawn at a third of the strength."""
    a = min(1.0, math.log1p(abs(v)) / math.log1p(vmax)) if vmax > 0 else 0.0
    a = a if sign else a / 3.0
    end = np.array(to_rgb(BETTER if v >= 0 else WORSE))
    return tuple((1 - 0.88 * a) * np.ones(3) + 0.88 * a * end)


def _layer_record_note():
    """One footer sentence from continuation_record.csv: the layer's rare branches and the rule-A disclosure."""
    try:
        rec = rows("continuation_record.csv")
    except Missing:
        return ""
    tot = next((r for r in rec if r["set"] == "all nine populations"), None)
    if tot is None:
        return ""
    nm = [(r["set"], int(float(r["no_match_reference_day"] or 0))) for r in rec if r is not tot and float(r["no_match_reference_day"] or 0) > 0]
    npos = sum(int(float(tot[k] or 0)) for k in ("nonpositive_fast_span", "nonpositive_day_span"))
    rel = lambda k: 100 * float(tot[k])                                                              # noqa: E731
    return ("Layer record (results/continuation_record.csv): no forecast day 1–7 of the last observed day's workday class, so "
            f"forecast day 7 is the reference day, on {int(float(tot['no_match_reference_day'] or 0))} windows"
            + (" (" + ", ".join(f"{s} {n}" for s, n in nm) + ")" if nm else "")
            + f"; spans whose mean is at or below zero (x = 0, flagged; a release rule): {npos}. Mean of the delivered trajectory "
            f"against the energy readout (rule A): median {rel('delivered_mean_vs_energy_readout_rel_p50'):.2f}%, "
            f"99th percentile {rel('delivered_mean_vs_energy_readout_rel_p99'):.1f}%, "
            f"maximum {rel('delivered_mean_vs_energy_readout_rel_max'):.0f}% (relative to the readout). ")


def fig_continuation():
    """Figure 9c (beside figure 9): the origin-continuity layer on against off on the same carrier and anchoring layer,
    by lead time, from continuation_layer.csv (0.2.0 results only; without the file the figure is skipped)."""
    tab, pops, pooled = _cont_table()
    cols = [k for k, _ in CONT_SEGMENTS if any((s, k) in tab for s in pops + [pooled])]
    has_e = any((s, CONT_ENERGY) in tab for s in pops + [pooled])
    names = dict(CONT_SEGMENTS)
    flags, note = flag_note([r for r in tab.values() if r["set"] in pops])
    cap = wrap("The origin-continuity layer of ankyra-lite 0.2.0 on against off, on the same carrier and anchoring layer (the seasonal naive "
               "of the state rule on state windows in both, where the layer is bypassed), on the test windows of every population, by lead "
               "time. Improvement 100[1 − exp(r)] of the hourly error with the layer, r the unit-equal log RMS ratio over the hours of the "
               "segment (hour 0; hours 0–2; forecast day 1; days 2–7; days 8–31; the whole window); 95% unit-and-month bootstrap "
               "intervals; positive = lower error with the layer. Energy: the monthly energy error of the delivered trajectory, which the "
               "layer changes; the release's energy readout is read before the layer (readout rule A) and does not change. "
               + ("a: the units of all populations pooled (resampled together). " if pooled else "")
               + ("b" if pooled else "a") + ": each population; filled cells and bold numbers: the interval excludes zero; the colour "
               "scale is compressed. Figure 9 shows the hourly error by forecast day against the baselines. " + notes().rstrip())
    foot = wrap(("Windows per population (state windows, bypassed by the layer): " + "; ".join(
        f"{pop(s, flags)} {int(_num(tab[(s, cols[-1])].get('windows')) or 0):,} ({int(_num(tab[(s, cols[-1])].get('state_windows')) or 0):,})"
        for s in pops if (s, cols[-1]) in tab) + ". " if pops else "") + _layer_record_note() + note, fs=5.6, width=165)
    ncell = len(cols) + has_e
    nrow = len(pops) + (1 if pooled else 0)
    row_h = 0.215
    hm_h = max(1.6, row_h * nrow + 0.55)
    H = cap_height(cap) + 0.45 + hm_h + 0.62 + 0.13 * len(foot)
    fig = plt.figure(figsize=(7.2, H))
    top, bot = 1 - (cap_height(cap) + 0.45) / H, (0.62 + 0.13 * len(foot)) / H
    if pooled:
        ga = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.66], wspace=0.04, left=0.095, right=0.49, top=top, bottom=bot)
        gh = fig.add_gridspec(1, 1, left=0.605, right=0.985, top=top, bottom=bot)
        gs = [ga[0], ga[1], gh[0]]
    else:
        gs = fig.add_gridspec(1, 1, left=0.16, right=0.985, top=top, bottom=bot)
    vals = [p[0] for r in tab.values() if (p := _cont_pt(r)) is not None]
    vmax = max([abs(v) for v in vals] + [1.0])

    # ---- a: pooled, by lead segment (forest)
    if pooled:
        ax = fig.add_subplot(gs[0]); tx = fig.add_subplot(gs[1], sharey=ax)
        items = [(names[k], _cont_pt(tab.get((pooled, k)))) for k in cols]
        if has_e:
            items.append(("energy\n(delivered)", _cont_pt(tab.get((pooled, CONT_ENERGY)))))
        ys = -np.arange(len(items), dtype=float)
        if has_e:
            ys[-1] -= 0.5
        pts = [q for _, q in items if q]
        ends = [v for q in pts for v in q[1:3] if v is not None]
        xl = lims([q[0] for q in pts] + ends, (-2, 5), (-60, 80), step=1 if vmax < 10 else 5, margin=0.5)
        ax.axvspan(0, xl[1], color="#F3F6FA", zorder=0)
        ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
        if has_e:
            ax.axhline(ys[-2] - 0.75, color="#E0E0E0", lw=0.5, zorder=0)
        for y0, (lab, q) in zip(ys, items):
            if q is None:
                continue
            p, a, b, sign = q
            col = BETTER if sign > 0 else (WORSE if sign < 0 else GREY)
            ci(ax, y0, p, a, b, xl, col, sign != 0, s=18, marker="o" if lab != "energy\n(delivered)" else "D")
            iv = f" [{a:+.2f}, {b:+.2f}]" if a is not None and b is not None else ""
            tx.text(0.04, y0, f"{p:+.2f}{iv}".replace("-", "−"), va="center", ha="left", fontsize=5.8, color="#1A1A1A",
                    fontweight="bold" if sign else "normal")
        ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in items], fontsize=6.3, linespacing=1.0)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(ys[-1] - 0.6, 0.6); ax.set_xlim(*xl); ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
        ax.set_xlabel("Improvement with the layer (%)", fontsize=6.5)
        n_u = _num(tab[(pooled, cols[-1])].get("units")) if (pooled, cols[-1]) in tab else None
        ax.set_title("a   Pooled" + (f": {int(n_u):,} units" if n_u else ""), fontsize=7.4, pad=24)     # pad of b (top tick labels): titles align
        tx.axis("off"); tx.set_xlim(0, 1)
        tx.text(0.04, 1.0, "% [95% interval]\nbold: resolved", transform=tx.transAxes, fontsize=5.6, color=GREY, va="bottom", ha="left",
                linespacing=1.05)
        hx = fig.add_subplot(gs[2])
    else:
        hx = fig.add_subplot(gs[0])

    # ---- b: by population (heat map)
    rows_ = pops + ([pooled] if pooled else [])
    keys = cols + ([CONT_ENERGY] if has_e else [])
    xs = np.arange(len(keys), dtype=float)
    if has_e:
        xs[-1] += 0.25
    yr = -np.arange(len(rows_), dtype=float)
    if pooled:
        yr[-1] -= 0.3
    for yy, s in zip(yr, rows_):
        for xx, k in zip(xs, keys):
            q = _cont_pt(tab.get((s, k)))
            if q is None:
                hx.add_patch(Rectangle((xx - 0.47, yy - 0.45), 0.94, 0.9, fc="#FAFAFA", ec="none", zorder=1))
                hx.text(xx, yy, "–", ha="center", va="center", fontsize=6.0, color=GREY)
                continue
            p, _, _, sign = q
            fc = _cont_colour(p, sign, vmax)
            hx.add_patch(Rectangle((xx - 0.47, yy - 0.45), 0.94, 0.9, fc=fc, ec="white", lw=0.6, zorder=1))
            dark = sign != 0 and math.log1p(abs(p)) / math.log1p(vmax) > 0.62
            txt = f"{p:+.1f}" if abs(p) >= 1 else f"{p:+.2f}"
            hx.text(xx, yy, txt.replace("-", "−"), ha="center", va="center", fontsize=5.9, zorder=2,
                    color="white" if dark else ("#1A1A1A" if sign else GREY), fontweight="bold" if sign else "normal")
    hx.set_xlim(xs[0] - 0.55, xs[-1] + 0.55); hx.set_ylim(yr[-1] - 0.55, 0.55)
    hx.set_xticks(xs); hx.set_xticklabels([names[k].replace(" ", "\n", 1) if k in names else "energy\n(deliv.)" for k in keys],
                                          fontsize=6.0, linespacing=1.0)
    hx.xaxis.tick_top(); hx.tick_params(axis="both", length=0)
    hx.set_yticks(yr)
    hx.set_yticklabels([pop(s, flags) if s != pooled else "pooled" for s in rows_], fontsize=6.3)
    if pooled:
        hx.get_yticklabels()[-1].set_fontweight("bold")
    for sp in hx.spines.values():
        sp.set_visible(False)
    hx.set_title(("b" if pooled else "a") + "   By population: hourly error by lead time, and energy", fontsize=7.4, pad=24)
    # colour key under the map
    kx = fig.add_axes([0.62, (0.30 + 0.13 * len(foot)) / H, 0.30, 0.06 / H])
    ticks = [t_ for t_ in (-20, -5, -1, 0, 1, 5, 20, 50) if abs(t_) <= vmax * 1.05] or [0]
    grid = np.linspace(-1, 1, 201)
    for i, g in enumerate(grid[:-1]):
        v = math.copysign(math.expm1(abs(g) * math.log1p(vmax)), g)
        kx.add_patch(Rectangle((g, 0), grid[1] - grid[0], 1, fc=_cont_colour(v, 1, vmax), ec="none"))
    pos = [math.copysign(math.log1p(abs(t_)) / math.log1p(vmax), t_) for t_ in ticks]
    kx.set_xlim(-1, 1); kx.set_ylim(0, 1); kx.set_yticks([])
    kx.set_xticks(pos); kx.set_xticklabels([f"{t_:+d}".replace("-", "−") if t_ else "0" for t_ in ticks], fontsize=5.6)
    kx.tick_params(axis="x", length=1.5, pad=1)
    for sp in kx.spines.values():
        sp.set_visible(False)
    fig.text(0.62 - 0.01, (0.30 + 0.13 * len(foot) + 0.03) / H, "improvement with the layer (%), resolved;\npaler: not resolved",
             fontsize=5.6, color=GREY, ha="right", va="center", linespacing=1.05)
    put_caption(fig, cap, H)
    put_footer(fig, foot, H)
    save(fig, "fig9c_continuation_layer")


def fig_lead_days_relative():
    ld = rows("lead_day_metrics.csv")
    lab = {r["model"]: r["model_label"] for r in ld}
    sets = present(ALL, ld)
    days = np.arange(1, 32)
    cap = wrap("Each baseline's loss relative to ankyra-lite's on the same day, 100 (CV_model / CV_ankyra-lite − 1), from the curves of figure "
               "9. ankyra-lite is the zero line by definition; its own loss, rising with the forecast day, is in figure 9. Above zero the "
               "baseline has the larger loss. Each panel has its own vertical scale. Over the whole month the comparison differs, because the "
               "monthly RMS weights days with large errors more. " + notes({r["model"] for r in ld}).rstrip())
    below = len(sets) % 2 == 0
    nrow = len(sets) // 2 if below else (len(sets) + 1) // 2
    H = 2.05 * nrow + cap_height(cap) + (0.75 if below else 0.35)
    fig, axs = plt.subplots(nrow, 2, figsize=(7.2, H), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=1 - (cap_height(cap) + 0.25) / H, bottom=(0.95 if below else 0.4) / H, hspace=0.36,
                        wspace=0.17)
    for ax, s in zip(axs.ravel(), sets):
        R = [r for r in ld if r["set"] == s]
        curve = {m: np.array([float(r["GM_CV_RMSE_pct"]) for r in sorted((r for r in R if r["model"] == m), key=lambda r: int(r["day"]))])
                 for m in sorted({r["model"] for r in R})}
        rels, off = {}, []
        for m, c, ls in DAY_LINES:
            if m not in curve:
                continue
            rel = 100.0 * (curve[m] / curve[REF] - 1.0)
            if np.median(rel) > 60:
                off.append(label(m, lab)); continue
            rels[m] = rel
        vals = np.concatenate(list(rels.values())) if rels else np.array([0.0])
        lo, hi = min(-5.0, 5 * np.floor((vals.min() - 5) / 5)), max(5.0, 5 * np.ceil((vals.max() + 5) / 5))
        span_ = hi - lo                                     # free bands at the top and bottom for the two region labels
        lo, hi = lo - 0.13 * span_, hi + 0.11 * span_
        ax.axhspan(0, hi, color="#F3F6FA", zorder=0)
        ax.axhline(0, color=LITE, lw=1.3, zorder=3)
        for m, c, ls in DAY_LINES:
            if m in rels:
                ax.plot(days, rels[m], color=c, ls=ls, lw=1.0, zorder=2)
        for x in (7.5, 14.5, 21.5):
            ax.axvline(x, color="#E6E6E6", lw=0.5, zorder=1)
        ax.set_xlim(1, 31); ax.set_ylim(lo, hi); ax.set_xticks([1, 7, 14, 21, 28, 31])
        n_ = R[0].get("n_units_gm") or R[0].get("units")
        ax.set_title(f"{s.replace(' 2017', '')} ({n_} units)", fontsize=7.2)
        for yy_, va_, t_ in ((0.965, "top", "baseline worse than ankyra-lite"), (0.035, "bottom", "baseline better than ankyra-lite")):
            ax.text(0.015, yy_, t_, transform=ax.transAxes, fontsize=5.4, color=GREY, va=va_, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75))
        if off:
            ax.text(0.985, 0.965, "off scale: " + ", ".join(off), transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=GREY)
        ax.grid(axis="y", color="#EFEFEF", lw=0.4, zorder=0)
    for ax in axs[:, 0]:
        ax.set_ylabel("Loss relative to ankyra-lite (%)", fontsize=6.5)
    flat = axs.ravel()
    for k, ax in enumerate(flat[:len(sets)]):
        if k >= len(sets) - 2:
            ax.set_xlabel("Forecast day", fontsize=6.8); ax.tick_params(axis="x", labelbottom=True)
    for ax in flat[len(sets):]:
        ax.axis("off")
    handles = [Line2D([], [], color=LITE, lw=1.3, label="ankyra-lite: the reference, 0 by definition")]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.0, label=label(m, lab)) for m, c, ls in DAY_LINES if m in lab]
    if below:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.2, handlelength=2.4,
                   columnspacing=1.4, labelspacing=0.5)
    else:
        flat[len(sets)].legend(handles=handles, loc="center", ncol=2, fontsize=6.2, handlelength=2.4, columnspacing=1.2, labelspacing=0.7)
    put_caption(fig, cap, H)
    save(fig, "fig10_loss_by_day_relative")


# ============================================================================ Figure 11: monthly energy error
def fig_energy():
    en = rows("energy_error.csv")
    rk = [r for r in rows("benchmark_mean_unit_rank.csv") if r["subset"] == "late"]
    flags, note = flag_note(en)
    cols = []
    for s in present(ALL, en):
        for sub in sorted({r["unit_set"] for r in en if r["set"] == s}, key=lambda u: (u != "all units", u)):
            short = head(s, flags)
            if sub != "all units":
                short = SHORT[s] + "\nw/o 3" if "near-zero" in sub else SHORT[s] + "\n" + sub[:10]
            cols.append((s, sub, short))
    order, _ = ordered_by_rank(rk, present(ALL, rk))
    cell = {(r["set"], r["unit_set"], r["model"]): r for r in en}
    models = [m for m in order if m != REF and any((s, sub, m) in cell for s, sub, _ in cols)]
    lab = {r["model"]: r["model_label"] for r in en}
    cls = {r["model"]: r["model_class"] for r in en}
    n, k = len(models), len(cols)
    P = np.array([[float(cell[(s, sub, m)]["improvement_pct"]) if (s, sub, m) in cell else np.nan for s, sub, _ in cols] for m in models])
    cap = wrap("Monthly energy error, 744 × the mean hourly error of the delivered trajectory, on the same test windows"
               + (" (ankyra-lite: its energy readout, the trajectory before the continuation layer, readout rule A)" if release020() else "")
               + ". Unit-equal RMS ratio "
               "with 95% unit-and-month bootstrap intervals; * resolved in ankyra-lite's favour, (−) against it. BDG2 w/o 3: without the three "
               f"near-zero meters, where ankyra-lite's error is exactly zero (the point estimates are equal, only the intervals differ). Rows: the "
               f"{n} baselines ({n + 1} forecasters with ankyra-lite), in the order of figure 2. " + notes(models) + note)
    H = 1.2 + 0.165 * n + cap_height(cap)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 0.7], wspace=0.36, left=0.165, right=0.975, top=1 - (cap_height(cap) + 0.6) / H,
                          bottom=0.75 / H)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("energy", [LITE, "#F4A582", "#F7F7F7", "#92C5DE", "#2166AC"])
    ax.imshow(np.clip(P, -50, 50), cmap=cmap, vmin=-50, vmax=50, aspect="auto", interpolation="nearest")
    for i, m in enumerate(models):
        for j, (s, sub, _) in enumerate(cols):
            r = cell.get((s, sub, m))
            if r is None:
                continue
            tag = "*" if float(r["um_high"]) < 0 else ("(−)" if float(r["um_low"]) > 0 else "")
            v = P[i, j]
            txt_ = "0" if round(v) == 0 else (f"{v:+.0f}" if abs(v) < 999.5 else f"{v:+,.0f}")
            ax.text(j, i, txt_ + tag, ha="center", va="center", fontsize=4.9,
                    color="white" if abs(v) > 35 else "#1A1A1A", fontweight="bold" if tag == "*" else "normal")
    for j in range(1, k):
        if cols[j][0] != cols[j - 1][0] and (cols[j - 1][1] != "all units" or cols[j][0] == "Oslo"):
            ax.axvline(j - 0.5, color="white", lw=2.0)
    ax.set_xticks(range(k)); ax.set_xticklabels([c[2] for c in cols], fontsize=5.5, linespacing=1.0)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(range(n)); ax.set_yticklabels([label(m, lab) for m in models], fontsize=6.1); ax.tick_params(axis="y", length=0, pad=8)
    for i, m in enumerate(models):
        ax.scatter([-0.8], [i], s=10, color=class_color(cls.get(m, ""), m), clip_on=False, marker="s", zorder=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("a   ankyra-lite's improvement in monthly energy error over each baseline (%)", fontsize=7.3, pad=24)
    allc = [c for c in cols if c[1] == "all units"]
    better = np.array([sum(float(cell[(s, sub, m)]["um_high"]) < 0 for m in models if (s, sub, m) in cell) for s, sub, _ in allc])
    worse = np.array([sum(float(cell[(s, sub, m)]["um_low"]) > 0 for m in models if (s, sub, m) in cell) for s, sub, _ in allc])
    tot = np.array([sum((s, sub, m) in cell for m in models) for s, sub, _ in allc])
    tie = tot - better - worse
    bx = fig.add_subplot(gs[1])
    y = np.arange(len(allc))
    bx.barh(y, better, color="#2166AC", height=0.62); bx.barh(y, tie, left=better, color="#D9D9D9", height=0.62)
    bx.barh(y, worse, left=better + tie, color=LITE, height=0.62)
    for i in range(len(allc)):
        bx.text(better[i] - 0.3, i, f"{better[i]}", va="center", ha="right", fontsize=5.9, color="white", fontweight="bold")
    bx.set_yticks(y); bx.set_yticklabels([c[2].replace("\n", " ") for c in allc], fontsize=6.2); bx.invert_yaxis()
    bx.set_xlim(0, max(tot)); bx.set_xticks(range(0, max(tot) + 1, 5)); bx.set_xlabel(f"Number of the {max(tot)} baselines", fontsize=6.6)
    bx.set_title("b   Head-to-head on energy", fontsize=7.3, pad=24)
    bx.legend(handles=[Patch(fc="#2166AC", label="better"), Patch(fc="#D9D9D9", label="not resolved"),
                       Patch(fc=LITE, label="worse")], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
              fontsize=5.6, handlelength=1.0, handleheight=0.8, columnspacing=0.6, handletextpad=0.3)
    fig.legend(handles=row_major(class_handles(cls, models, size=4.0), 4), loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, fontsize=5.7,
               handletextpad=0.25, columnspacing=0.9)
    put_caption(fig, cap, H)
    save(fig, "fig11_energy_error")


# ============================================================================ Figure 12: position on every population
def fig_consistency():
    rk = [r for r in rows("benchmark_mean_unit_rank.csv") if r["subset"] == "late"]
    flags, note = flag_note(rk)
    lab = {r["model"]: r["model_label"] for r in rk}
    try:
        cls = {r["model"]: r["model_class"] for r in rows("benchmark_pairwise.csv")}
    except Missing:
        cls = {}
    sets = present(ALL, rk)
    tsets, nsets = [s for s in sets if s in TEST], [s for s in sets if s in NORWAY]
    pos = {}
    for s in sets:
        order = sorted((r for r in rk if r["set"] == s), key=lambda r: float(r["mean_unit_rank"]))
        for i, r in enumerate(order):
            pos.setdefault(r["model"], {})[s] = i + 1
    nmod = max(len([r for r in rk if r["set"] == s]) for s in sets)
    models = sorted(pos, key=lambda m: (np.mean([pos[m][s] for s in sets if s in pos[m]]), m))
    first = sum(pos[REF].get(s) == 1 for s in sets)
    top2 = {m: sum(pos[m].get(s, 99) <= 2 for s in sets) for m in models if m != REF}
    cap = wrap(f"Where each forecaster lands on each population, test windows (all {nmod} forecasters on the same windows). ankyra-lite is "
               f"first on {first} of {len(sets)} populations; no other forecaster is in the top two on more than "
               f"{max(top2.values()) if top2 else 0} of them. The position uses the mean per-unit rank, a secondary summary; the primary "
               "estimand is in figure 3. Rows: mean position over the populations. " + notes(models) + note)
    leg_h = 0.3 if cls else 0.0
    H = 0.75 + 0.17 * len(models) + cap_height(cap) + leg_h
    fig, ax = plt.subplots(figsize=(7.2, H))
    fig.subplots_adjust(left=0.2, right=0.985, top=1 - (cap_height(cap) + 0.12) / H, bottom=(0.45 + leg_h) / H)
    for i, m in enumerate(models):
        c = class_color(cls.get(m, ""), m) if (cls or m == REF) else "#7F7F7F"
        t = [pos[m][s] for s in tsets if s in pos[m]]; p = [pos[m][s] for s in nsets if s in pos[m]]
        allp = t + p
        ax.plot([min(allp), max(allp)], [i, i], color=c, lw=2.2 if m == REF else 1.2, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.scatter(t, [i - 0.13] * len(t), s=26 if m == REF else 16, color=c, zorder=3, lw=0)
        ax.scatter(p, [i + 0.17] * len(p), s=22 if m == REF else 14, facecolor="white", edgecolor=c, lw=0.9, zorder=3)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([label(m, lab) for m in models], fontsize=6.4)
    for t_, m in zip(ax.get_yticklabels(), models):
        if m == REF:
            t_.set_color(LITE); t_.set_fontweight("bold")
    ax.set_ylim(len(models) - 0.5, -0.6)
    ax.set_xlim(0.5, nmod + 0.5); ax.set_xticks(list(range(1, nmod + 1, 2)))
    ax.set_xlabel(f"Position among the {nmod} forecasters (1 = lowest mean per-unit rank)", fontsize=6.8)
    ax.grid(axis="x", color="#EFEFEF", lw=0.5, zorder=0)
    ax.axvspan(0.5, 2.5, color="#F3F6FA", zorder=0)
    tlab = "six release test populations" if len(tsets) == 6 else "release test populations: " + ", ".join(SHORT[s] for s in tsets)
    ax.legend(handles=([Line2D([], [], marker="o", ls="", color=GREY, markersize=4.5, label=tlab)] if tsets else [])
              + ([Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, markersize=4.5,
                         label=", ".join(nsets))] if nsets else [])
              + [Line2D([], [], color=GREY, lw=1.6, alpha=0.4, label=f"range over the {num(len(sets))} populations")],
              loc="upper right", fontsize=6.1, handlelength=1.6, handletextpad=0.6, frameon=True, facecolor="white", edgecolor="#DDDDDD")
    if cls:
        fig.legend(handles=row_major(class_handles(cls, models, marker="o", size=4.0), 4), loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4,
                   fontsize=5.8, handletextpad=0.25, columnspacing=0.9)
    put_caption(fig, cap, H)
    save(fig, "fig12_consistency")


# ============================================================================ Figures 13, 16: prediction intervals
ARM_LABEL = {"chronos_native": "Chronos-2 native", "timesfm_native": "TimesFM native 0.1–0.9"}
ARM_COLOR = {"chronos_native": "#807DBA", "timesfm_native": FM}
ARM_MARK = {"chronos_native": "s", "timesfm_native": "^"}


def _interval_arms(iv):
    arms = []
    for r in iv:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    ref = next((a for a in arms if "lite" in a.lower()), None)
    if ref is None:
        raise Missing("intervals_by_population.csv: an ankyra-lite arm")
    rest = sorted([a for a in arms if a != ref], key=lambda a: (a not in ARM_LABEL, list(ARM_LABEL).index(a) if a in ARM_LABEL else 0, a))
    return ref, rest


def _f(r, c):
    return float(r[c]) if r.get(c) not in (None, "") else None


def fig_intervals():
    iv = [r for r in rows("intervals_by_population.csv") if r["set"] == "GoiEner households"]
    if not iv:
        raise Missing("intervals_by_population.csv: GoiEner households rows")
    try:
        con = {r["comparator"]: r for r in rows("intervals_winkler_contrasts.csv") if r["set"] == "GoiEner households"}
    except Missing:
        con = {}
    ref, rest = _interval_arms(iv)
    val = {r["arm"]: r for r in iv}
    arms = [(ref, "ankyra-lite\ninterval", LITE)] + [(a, ARM_LABEL.get(a, a).replace(" native", "\nnative"), ARM_COLOR.get(a, "#9E9E9E")) for a in rest]
    u, wn = val[ref].get("units", ""), val[ref].get("windows", "")
    cap = wrap(f"Prediction intervals on GoiEner households ({int(float(u)):,} units, {int(float(wn)):,} windows): the anchoring layer's "
               "interval readout around ankyra-lite's delivered trajectory (after the continuation layer) and the native quantiles of two foundation models. b: ankyra-lite's "
               "improvement in the unit-equal Winkler score over each interval; * 95% unit-and-month interval excludes zero. Computed after "
               "scoring, as a description. " + notes().rstrip())
    H = 2.35 + cap_height(cap)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, H), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.32})
    fig.subplots_adjust(left=0.075, right=0.985, top=1 - (cap_height(cap) + 0.25) / H, bottom=0.5 / H)
    x = np.arange(len(arms)); w = 0.36
    for j, (key, name, c) in enumerate(arms):
        a = val[key]
        c80, c90 = _f(a, "cov80"), _f(a, "cov90")
        ax.bar(j - w / 2, 100 * c80, width=w, color=c, alpha=0.95)
        ax.text(j - w / 2, 2.5, f"{100 * c80:.1f}", ha="center", va="bottom", fontsize=5.6, color="white", fontweight="bold")
        if c90 is not None:
            ax.bar(j + w / 2, 100 * c90, width=w, color=c, alpha=0.45)
            ax.text(j + w / 2, 2.5, f"{100 * c90:.1f}", ha="center", va="bottom", fontsize=5.6, color="#1A1A1A")
        else:
            ax.text(j + w / 2, 3, "n/a", ha="center", fontsize=5.5, color=GREY)
    ax.axhline(80, color="#1A1A1A", lw=0.7, ls="--"); ax.axhline(90, color="#1A1A1A", lw=0.7, ls=":")
    ax.set_xlim(-0.6, len(arms) + 0.3)
    ax.text(len(arms) + 0.27, 80.5, "nominal 80%", fontsize=5.5, ha="right", va="bottom")
    ax.text(len(arms) + 0.27, 90.5, "nominal 90%", fontsize=5.5, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([a[1] for a in arms], fontsize=5.9); ax.set_ylim(0, 100); ax.set_ylabel("Coverage of hours (%)", fontsize=6.5)
    ax.set_title("a   Coverage (dark: 80% band, light: 90% band)", fontsize=7.2)
    wk = [_f(val[k], "winkler80") for k, *_ in arms]
    bx.bar(x, wk, color=[a[2] for a in arms], width=0.6)
    for j, (key, *_r) in enumerate(arms):
        bx.text(j, wk[j] * 1.012, f"{wk[j]:.3g}", ha="center", va="bottom", fontsize=5.7)
        if key in con:
            c = con[key]
            tag = "*" if float(c["um_high"]) < 0 else ("(−)" if float(c["um_low"]) > 0 else "")
            bx.text(j, max(wk) * 0.03, f"ankyra-lite\n{float(c['improvement_pct']):+.1f}%{tag}", ha="center", fontsize=5.3, color="white", fontweight="bold")
    bx.set_xticks(x); bx.set_xticklabels([a[1] for a in arms], fontsize=5.9); bx.set_ylabel("Winkler score, 80% (kW)", fontsize=6.5)
    bx.set_ylim(0, max(wk) * 1.18)
    bx.set_title("b   Winkler score (lower is better)", fontsize=7.2)
    put_caption(fig, cap, H)
    save(fig, "fig13_intervals")


def fig_intervals_all():
    iv = rows("intervals_by_population.csv")
    flags, note = flag_note(iv)
    ref, rest = _interval_arms(iv)
    sets = present(ALL, iv)
    arms = [(ref, "ankyra-lite interval", LITE, "o")] + [(a, ARM_LABEL.get(a, a), ARM_COLOR.get(a, "#9E9E9E"), ARM_MARK.get(a, "D")) for a in rest]
    val = {(r["set"], r["arm"]): r for r in iv}
    cap = wrap(f"Prediction intervals on the {num(len(sets))} populations (bold: the six release test populations). The ankyra-lite interval is "
               "the anchoring layer's interval readout around the delivered trajectory. Each line in b and c is one population; the dashed "
               "line is the nominal level; W4 is days 22–31. Computed after scoring, as a description. " + notes() + note)
    H = 2.95 + cap_height(cap)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.45, left=0.115, right=0.985, top=1 - (cap_height(cap) + 0.42) / H,
                          bottom=0.5 / H)
    ax = fig.add_subplot(gs[0]); y = np.arange(len(sets))[::-1]
    ax.axvline(80, color="#1A1A1A", lw=0.7, ls="--")
    for key, name, col, mk in arms:
        pts = [(100 * _f(val[(s_, key)], "cov80"), yy) for s_, yy in zip(sets, y) if (s_, key) in val and _f(val[(s_, key)], "cov80") is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, color=col, marker=mk, zorder=3, label=name)
    ax.set_yticks(y); ax.set_yticklabels([pop(s_, flags) for s_ in sets], fontsize=6.3)
    for t_, s_ in zip(ax.get_yticklabels(), sets):
        t_.set_fontweight("bold" if s_ in TEST else "normal")
    cov_all = [100 * _f(r, "cov80") for r in iv if r["set"] in sets and _f(r, "cov80") is not None]
    ax.set_xlim(*lims(cov_all, (20, 95), (0, 100), margin=2))                  # widened when a coverage falls outside 20-95%
    ax.set_xlabel("Coverage of the nominal 80% interval (% of hours)", fontsize=6.5); ax.set_title("a   Coverage, all hours", fontsize=7.2)
    ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=3, fontsize=5.7, handletextpad=0.2, columnspacing=0.8, frameon=False)
    wk = np.arange(1, 5)

    def wk_vals(keys):
        return [100 * _f(r, f"cov80_week{w}") for r in iv for w in wk
                if r["set"] in sets and r["arm"] in keys and _f(r, f"cov80_week{w}") is not None]

    def weekly(axx, key, col, mk, alpha):
        for s_ in sets:
            r = val.get((s_, key))
            if r is None or _f(r, "cov80_week1") is None:
                continue
            axx.plot(wk, [100 * _f(r, f"cov80_week{w}") for w in wk], color=col, lw=0.8, alpha=alpha, marker=mk, ms=2.2)

    bx = fig.add_subplot(gs[1]); bx.set_title("b   ankyra-lite by week", fontsize=7.2)
    weekly(bx, ref, LITE, "o", 0.75)
    bx.axhline(80, color="#1A1A1A", lw=0.7, ls="--"); bx.set_ylim(*lims(wk_vals({ref}), (55, 95), (0, 100), margin=2))
    bx.set_xticks(wk); bx.set_xticklabels([f"W{i}" for i in wk])
    bx.set_ylabel("Coverage (%)", fontsize=6.5); bx.set_xlabel("Forecast week", fontsize=6.5)
    cx = fig.add_subplot(gs[2]); cx.set_title("c   Native quantiles by week", fontsize=7.2)
    for key, name, col, mk in arms[1:]:
        weekly(cx, key, col, mk, 0.6)
    cx.axhline(80, color="#1A1A1A", lw=0.7, ls="--"); cx.set_ylim(*lims(wk_vals({a[0] for a in arms[1:]}), (10, 95), (0, 100), margin=2))
    cx.set_xticks(wk); cx.set_xticklabels([f"W{i}" for i in wk])
    cx.set_xlabel("Forecast week", fontsize=6.5)
    put_caption(fig, cap, H)
    save(fig, "fig16_intervals_populations")


# ============================================================================ Figure 14: where the error sits (block attribution)
def fig_block_shares():
    bs = rows("block_shares.csv")
    sets = present(ALL, bs)
    val = {(r["set"], r["model"], r["quantity"], r["block"]): float(r["value"]) for r in bs}
    singles = []
    for r in bs:
        if " vs " not in r["model"] and r["model"] not in singles:
            singles.append(r["model"])
    singles = [REF] + [m for m in singles if m != REF]
    comps = [m.split(" vs ", 1)[1] for m in dict.fromkeys(r["model"] for r in bs) if m.startswith(REF + " vs ")]
    pal = {REF: LITE, "TimesFM": FM, "Chronos-2-X": "#1D91C0", "Chronos-2": "#807DBA"}
    blocks_ = [("level", "level", "#1A1A1A"), ("daily path", "daily path", "#7F7F7F"), ("within-day", "within-day", "#D9D9D9")]
    cap = wrap(f"Block attribution on the test windows of the {num(len(sets))} release test populations. The three blocks are orthogonal (P1), so a "
               "forecast's hourly MSE is the sum of its level, daily-path and within-day MSE. a: the median over units of each block's share "
               "(medians of three shares need not add to 100). " + ", ".join("bcd"[i] for i in range(len(comps))) + ": unit-equal RMS "
               "improvement of ankyra-lite in each block. Descriptive, computed after scoring. " + notes().rstrip())
    H = 3.55 + cap_height(cap)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(1, 1 + len(comps), width_ratios=[1.5] + [0.85] * len(comps), wspace=0.32, left=0.075, right=0.985,
                          top=1 - (cap_height(cap) + 0.3) / H, bottom=0.62 / H)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(sets)); w = 0.8 / len(singles)
    for j, key in enumerate(singles):
        col = pal.get(key, "#9E9E9E")
        left = np.zeros(len(sets))
        for k, (blk, _, _) in enumerate(blocks_):
            v = np.array([100 * val.get((s, key, "median_unit_share", blk), np.nan) for s in sets])
            xx = x + (j - (len(singles) - 1) / 2) * w
            ax.bar(xx, v, bottom=left, width=w * 0.92, color=col, alpha=[1.0, 0.62, 0.3][k], edgecolor="white", lw=0.4)
            if k == 2:
                for i in range(len(sets)):
                    ax.text(xx[i], left[i] + v[i] / 2, f"{v[i]:.0f}", ha="center", va="center", fontsize=4.8, color="#1A1A1A")
            left += np.nan_to_num(v)
    ax.set_xticks(x); ax.set_xticklabels([SHORT.get(s, s) for s in sets], fontsize=6.1, rotation=25, ha="right")
    ax.set_ylim(0, 122); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("Share of the median unit's hourly squared error (%)", fontsize=6.4)
    ax.set_title("a   Which block carries the error", fontsize=7.2)
    ax.legend(handles=[Patch(fc=pal.get(m, "#9E9E9E"), label=m) for m in singles] +
              [Patch(fc="#7F7F7F", alpha=a, label=f"{n} block") for (_, n, _), a in zip(blocks_, [1.0, 0.62, 0.3])],
              loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=5.5, handlelength=1.0, handleheight=0.8, columnspacing=0.8, frameon=False)
    y = np.arange(len(sets))[::-1]; h = 0.2
    first = None
    for ci, comp in enumerate(comps):
        bx = fig.add_subplot(gs[1 + ci], sharey=first) if first is not None else fig.add_subplot(gs[1 + ci])
        first = first or bx
        for k, (blk, name, shade) in enumerate(blocks_ + [("hourly", "hourly total", LITE)]):
            v = np.array([val.get((s, f"{REF} vs {comp}", "improvement_pct", blk), np.nan) for s in sets])
            bx.barh(y + (1.5 - k) * h, np.clip(v, -60, 60), height=h * 0.9, color=shade if blk != "hourly" else LITE, alpha=1.0 if blk == "hourly" else 0.85)
            for i in range(len(sets)):
                if abs(v[i]) > 60:
                    bx.text(np.sign(v[i]) * 61, y[i] + (1.5 - k) * h, f"{v[i]:+.0f}", fontsize=4.8, va="center", ha="left" if v[i] > 0 else "right")
        bx.axvline(0, color="#7F7F7F", lw=0.6)
        bx.set_yticks(y)
        if ci == 0:
            bx.set_yticklabels([SHORT.get(s, s) for s in sets], fontsize=6.1)
        else:
            bx.tick_params(axis="y", labelleft=False)
        bx.tick_params(axis="y", length=0)
        bx.set_xlim(-65, 65); bx.set_xlabel("Improvement by block (%)", fontsize=6.3)
        bx.set_title(f"{'bcd'[ci]}   Gain over {comp}", fontsize=7.2)
        if ci == len(comps) - 1:
            bx.legend(handles=[Patch(fc=c, label=n) for _, n, c in blocks_] + [Patch(fc=LITE, label="hourly total")], loc="upper right", fontsize=5.4,
                      handlelength=1.0, handleheight=0.8, framealpha=0.9)
    put_caption(fig, cap, H)
    save(fig, "fig14_block_attribution")


# ============================================================================ Figure 15: rank significance (Friedman / Nemenyi / Wilcoxon-Holm)
def _pval(v):
    """(value, text) of a p-value cell: a number, or a bound such as '<1e-300' (the tables write p-values below the
    double-precision range that way; the value is then the bound itself, enough for every test against 5%). An empty
    cell is (None, '')."""
    s = (v or "").strip()
    if not s:
        return None, ""
    if s[0] in "<>":
        b = s[1:].strip()
        return float(b), f"p {s[0]} {b}"
    x = float(s)
    return x, ("p < 1e-300" if x == 0 else f"p = {x:.2g}")


def _sig(r):
    """True if the Holm-corrected Wilcoxon test against ankyra-lite is significant at 5%."""
    p_, _ = _pval(r.get("wilcoxon_holm_p_vs_ankyra"))
    return p_ is not None and p_ < 0.05


def fig_rank_tests():
    rt = rows("rank_tests.csv")
    flags, note = flag_note(rt)
    pooled = next((s for s in dict.fromkeys(r["set"] for r in rt) if "pooled" in s.lower()), None)
    R = sorted([r for r in rt if r["set"] == pooled], key=lambda r: float(r["mean_rank"]))
    if not R or not any(r["model"] == REF for r in R):
        raise Skipped("rank_tests.csv has no pooled row with ankyra-lite (the pooled tests need the six release test populations; "
                      "a run on fewer populations writes the per-population rows only)")
    nmod = len(R)
    fps = _pval(R[0].get("friedman_p"))[1] or "p not given"
    worse_names = {}                                   # per population: the baselines significantly better than ankyra-lite
    for s_ in present(ALL, rt):
        Rs = [r for r in rt if r["set"] == s_]
        ar = float(next(x["mean_rank"] for x in Rs if x["model"] == REF))
        worse_names[s_] = [label(r["model"], {r["model"]: r["model_label"]}) for r in Rs if r["model"] != REF
                           and _sig(r) and float(r["mean_rank"]) < ar]
    many = {s_: v for s_, v in worse_names.items() if len(v) > 2}          # b names two at most; the full lists go to the caption
    cap = wrap(f"Rank tests on per-unit RMSE, test windows, {nmod} forecasters. a: mean ranks over the units of the six release test "
               f"populations (Friedman test {fps}); the shaded band is the Nemenyi critical difference from ankyra-lite's rank, and filled "
               "markers differ from ankyra-lite in a Holm-corrected Wilcoxon signed-rank test. b: the same paired test per population (bold: "
               "the six release test populations). Rank tests weight every unit equally and ignore the size of the differences; the primary "
               "estimand is the log RMS ratio of figure 3. " + notes({r["model"] for r in rt})
               + ("Significantly better than ankyra-lite in b: " + "; ".join(f"{pop(s_)}: {', '.join(v)}" for s_, v in many.items()) + ". "
                  if many else "") + note)
    H = 0.95 + 0.235 * nmod + cap_height(cap)
    fig = plt.figure(figsize=(7.2, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.5, left=0.17, right=0.985, top=1 - (cap_height(cap) + 0.45) / H,
                          bottom=0.55 / H)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(R))[::-1]; cd = float(R[0]["nemenyi_cd"]); a_rank = float(next(r["mean_rank"] for r in R if r["model"] == REF))
    ax.axvspan(a_rank, a_rank + cd, color="#F6E3E5", zorder=0)
    for yy, r in zip(y, R):
        me = r["model"] == REF; sig = me or _sig(r)
        ax.scatter([float(r["mean_rank"])], [yy], s=22 if me else 14, color=LITE if me else ("#4D4D4D" if sig else "white"),
                   edgecolor=LITE if me else "#4D4D4D", lw=0.8, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([label(r["model"], {r["model"]: r["model_label"]}) for r in R], fontsize=6.1)
    for t_, r in zip(ax.get_yticklabels(), R):
        if r["model"] == REF:
            t_.set_color(LITE); t_.set_fontweight("bold")
    ax.set_xlabel(f"Mean per-unit rank among {nmod} forecasters (lower is better)", fontsize=6.5); ax.tick_params(axis="y", length=0)
    ax.set_title(f"a   Six test populations pooled ({int(float(R[0]['units'])):,} units)", fontsize=7.2)
    ax.text(a_rank, y[0] + 0.75, f"Nemenyi CD = {cd:.2f}", fontsize=5.7, color=LITE, ha="left", va="bottom")
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.4); ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
    bx = fig.add_subplot(gs[1])
    sets = present(ALL, rt); yy_ = np.arange(len(sets))[::-1]
    nb = 0
    for y_, s_ in zip(yy_, sets):
        Rs = [r for r in rt if r["set"] == s_]; n = len(Rs) - 1; nb = max(nb, n)
        ar = float(next(x["mean_rank"] for x in Rs if x["model"] == REF))
        sig = [r for r in Rs if r["model"] != REF and _sig(r)]
        better = sum(float(r["mean_rank"]) > ar for r in sig); worse_r = [r for r in sig if float(r["mean_rank"]) < ar]
        worse = len(worse_r)
        bx.barh(y_, better, color="#2166AC", height=0.62); bx.barh(y_, n - better - worse, left=better, color="#D9D9D9", height=0.62)
        bx.barh(y_, worse, left=n - worse, color=LITE, height=0.62)
        bx.text(better - 0.3, y_, f"{better}", va="center", ha="right", fontsize=5.9, color="white", fontweight="bold")
        if worse:                                          # one name per line, two at most, so the canvas keeps its width
            nm = worse_names.get(s_, [])
            txt_ = "\n".join(nm[:2]) + (f" +{len(nm) - 2}" if len(nm) > 2 else "")
            bx.text(n + 0.3, y_, txt_, va="center", ha="left", fontsize=5.0 if len(nm) > 1 else 5.4, color=LITE, linespacing=1.0)
    bx.set_yticks(yy_); bx.set_yticklabels([pop(s_, flags) for s_ in sets], fontsize=6.3)
    for t_, s_ in zip(bx.get_yticklabels(), sets):
        t_.set_fontweight("bold" if s_ in TEST else "normal")
    bx.set_xlim(0, nb + 6); bx.set_xticks(range(0, nb + 1, 5)); bx.set_xlabel(f"Number of the {nb} baselines (Wilcoxon, Holm, 5%)", fontsize=6.5)
    bx.set_title("b   Per population, against each baseline", fontsize=7.2); bx.tick_params(axis="y", length=0)
    bx.legend(handles=[Patch(fc="#2166AC", label="ankyra-lite better"), Patch(fc="#D9D9D9", label="not significant"), Patch(fc=LITE, label="ankyra-lite worse")],
              loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=3, fontsize=5.7, handlelength=1.0, handleheight=0.8, columnspacing=0.7, handletextpad=0.3)
    put_caption(fig, cap, H)
    save(fig, "fig15_rank_tests")


# ============================================================================ Figure 17: external tests
# The counterpart of the release's Figure 17 (checks with the forecaster frozen on external data). Test sets in the order
# of results/external_tests.csv; the Helsinki retrospective set is trained and sealed, not scored, and is drawn as an
# empty row of the primary panels only.
EXT_SETS = [("lcl", "LCL"), ("unicon", "UNICON"), ("hkust", "HKUST\n(secondary)"),
            ("goiener_oot", "GoiEner HH,\nout of time\n(supplementary)")]
EXT_SEALED = "Helsinki\n(sealed)"
EXT_PRIMARY = [("TimesFM", "vs TimesFM 2.5"), ("SN-week", "vs seasonal naive (week)"),
               ("Ns", "vs carrier-free control\n      with the state rule")]
EXT_DAYS = [("forecast day 1", "o", "forecast day 1"), ("forecast days 2-7", "s", "days 2–7"), ("forecast days 8-31", "^", "days 8–31")]
EXT_DESC = [("vs history-only configuration", "vs history-only configuration"),
            ("span return vs plain continuation", "span return vs plain continuation\n      (R vs F, three seeds)"),
            ("continuation and span return vs from-scratch stage", "continuation and span return vs\n      from-scratch stage alone")]
# 0.2.0 (evaluation/external_tables.py --version 0.2): the re-run of the 0.2.0 configuration on the same test sets
EXT_H02 = ("forecast hours 0-2 (lead hours 0-3)", "v", "hours 0–2")
EXT_DESC_V2 = [("vs the 0.1.0 configuration (3 seeds)", "0.2.0 vs the 0.1.0 configuration\n      (recipe R, three seeds)"),
               ("layer on / off (recipe Rw, 6 seeds)", "continuation layer on vs off\n      (same Rw carrier, six seeds)"),
               ("weighting (Rw vs R, 6 seeds, no layer)", "loss weighting: Rw vs R\n      (six seeds, no layer)")]
EXT_DESC_READS = [("hourly", "o", "whole window"), EXT_H02, ("forecast day 1", "s", "forecast day 1")]
EXT_V2_FIRST = "ankyra-lite 0.2.0"                       # the first forecaster's label of every 0.2.0 row starts so


def fig_external_tests():
    raw = rows("external_tests.csv")
    R = [r for r in raw if (r.get("improvement_pct") or "").strip()]
    sets = [(k, n) for k, n in EXT_SETS if any(r["test_set_id"] == k for r in R)]
    if not sets:
        raise Skipped("external_tests.csv has no contrast of a known test set")
    v2 = any((r.get("first") or "").startswith(EXT_V2_FIRST) for r in R) or any(r["block"] == b for r in R for b, _ in EXT_DESC_V2)
    rule = next((r for r in raw if r.get("block") == "day-1 rule (design 3.3)"), None)
    f_ = lambda v: float(v) if (v or "").strip() else None                                          # noqa: E731

    def get(sid, block, model=None, readout="hourly"):
        hit = [r for r in R if r["test_set_id"] == sid and r["block"] == block and r["readout"] == readout
               and (model is None or r["model"] == model)]
        if len(hit) > 1:
            raise Skipped(f"external_tests.csv: {len(hit)} rows for {sid}, {block}, {model}, {readout}")
        return hit[0] if hit else None

    def pt(r):
        return None if r is None else (f_(r["improvement_pct"]), f_(r["interval_low_pct"]), f_(r["interval_high_pct"]), r["label"] != "not resolved")

    n = len(sets)
    days = ([EXT_H02] if v2 else []) + EXT_DAYS
    desc = EXT_DESC_V2 if v2 else EXT_DESC
    if v2:
        cap = wrap("External tests of ankyra-lite 0.2.0: the 0.2.0 configuration re-run on the test sets of the frozen external test of 19 "
                   "October 2026 (the same windows, comparators and estimators, read there once with the frozen candidate, recipe R with six "
                   "seeds) and scored once on 20 October 2026. The carrier was trained anew for each test set with recipe Rw from the same "
                   "stage-1 runs (six seeds averaged), under the unchanged anchoring layer, the seasonal-naive state rule and the "
                   "origin-continuity layer (energy and peak read before the layer). LCL, UNICON and HKUST: previously read populations; "
                   "GoiEner households out of time: a same-source out-of-time supplementary check (new households and a later period of study "
                   "population 6), not a new population; Helsinki: trained, anchored and sealed, not scored. Improvement of the first "
                   "forecaster's hourly error over the second (unit-equal RMS ratio; positive = the first lower); 95% intervals combine the "
                   "unit-and-month bootstrap with the spread over seeds (Welch) where the record has a seed layer, else unit-and-month only; "
                   "filled = the interval excludes zero (resolved), hollow = not resolved, which is not a finding of no difference; an "
                   "interval end beyond the axis is drawn as an arrow. a–c: the three pre-specified primary readouts of each test set "
                   f"({3 * n} nominal intervals over the {num(n)} test sets, no multiplicity correction: with no true difference about "
                   f"{0.05 * 3 * n:.2g} of them would be resolved by chance). d–f: the same contrasts by lead time, forecast hours 0–2 "
                   "included. g–i: descriptive contrasts between ankyra-lite's own configurations by lead time (g: 0.2.0 against the "
                   "0.1.0 configuration; h: the continuation layer on against off; i: the loss weighting of recipe Rw against recipe R, "
                   "without the layer). LCL is in the pretraining data of TimesFM 2.5, Chronos and Chronos-2, which favours TimesFM in a "
                   "and d; whether GoiEner is in it is not verified; UNICON and HKUST are not on the published lists, which does not "
                   "establish absence.")
    else:
        cap = None
    cap = cap or wrap("External tests of ankyra-lite, each test set scored once on 19 October 2026, after the run had been frozen. The "
               "carrier was trained anew for each test set with the released recipe R (the test set's own training set and windows "
               "borrowed from the study's populations) under the unchanged anchoring layer and the seasonal-naive state rule; circles: "
               "six seeds averaged, grey diamonds: the three-seed release configuration. LCL, UNICON and HKUST: first read of this "
               "candidate on a previously read population (earlier forecasters had been scored there and the anchoring layer was fixed "
               "after that); GoiEner households out of time: a same-source out-of-time supplementary check (new households and a later "
               "period of study population 6), not a new population; Helsinki: trained, anchored and sealed, not scored. Improvement of "
               "ankyra-lite's hourly error over the second forecaster (unit-equal RMS ratio; positive = ankyra-lite lower); 95% intervals "
               "combine the unit-and-month bootstrap with the spread over seeds (Welch); filled = the interval excludes zero (resolved), "
               "hollow = not resolved, which is not a finding of no difference; an interval end beyond the axis is drawn as an arrow. "
               "a–c: the three pre-specified primary readouts of each "
               "test set (15 nominal intervals over the five test sets, no multiplicity correction: with no true difference about 0.75 "
               "of them would be resolved by chance). d–f: the same contrasts by forecast day (six seeds). g–i: descriptive contrasts "
               "with ankyra-lite's own reduced configurations (g: six seeds; h: three seeds; i: six seeds, unit-and-month interval "
               "only). LCL is in the pretraining data of TimesFM 2.5, Chronos and Chronos-2, which favours TimesFM in a and d; whether "
               "GoiEner is in it is not verified; UNICON and HKUST are not on the published lists, which does not establish absence.")
    prim = [get(k, "primary", "TimesFM") for k, _ in sets]
    foot = wrap("Windows / units / months of the primary readouts: " + "; ".join(
        f"{n_.splitlines()[0].rstrip(',')} {int(p['windows']):,} / {int(p['units']):,} / {p['months']}" for (k, n_), p in zip(sets, prim) if p)
        + ". " + (f"Day-1 rule of the 0.2.0 design, written before the read (does the mean log gap to TimesFM 2.5 on forecast day 1 "
                  f"narrow, from the 0.1.0 configuration to 0.2.0, by at least a third of the development-set narrowing?): {rule['label']}"
                  + (" (" + re.sub(r"-?\d+\.\d{4,}(?:[eE][-+]?\d+)?", lambda m_: f"{float(m_.group()):.3g}",
                                   "; ".join(s_.strip() for s_ in rule['note'].split(';')[:2])) + ")" if (rule.get("note") or "").strip() else "")
                  + ". It allows only the statement that the day-1 gap narrowed, not a lead: on forecast day 1 ankyra-lite 0.2.0 against "
                  "TimesFM 2.5 is " + "; ".join(
                      f"{n_.splitlines()[0].rstrip(',')} {f_(d_['improvement_pct']):+.1f}% ({d_['label']})"
                      for (k, n_), d_ in ((kk, next((r for r in raw if r['test_set_id'] == kk[0] and r['block'] == 'day-1 gap to TimesFM 2.5'
                                                     and r['subgroup'] == 'after'), None)) for kk in sets) if d_) + ". "
                  if v2 and rule else "")
        + "Source: results/external_tests.csv (every row cites the field of the score record it was copied from).", fs=5.6, width=165)
    axes_h = (0.36 * (n + 1) + 0.15, (0.62 if v2 else 0.5) * n + 0.1, (0.5 if v2 else 0.36) * n + 0.1)   # inches: the axes of the rows
    above, below = 0.58, 0.44                                            # inches: titles and legend; ticks and axis label
    H = cap_height(cap) + 0.05 + sum(axes_h) + 3 * (above + below) + 0.14 * len(foot) + 0.1
    fig = plt.figure(figsize=(7.2, H))
    tops, bottoms, y = [], [], cap_height(cap) + 0.05
    for ah in axes_h:
        y += above
        tops.append(1 - y / H)
        y += ah
        bottoms.append(1 - y / H)
        y += below
    grids = [fig.add_gridspec(1, 3, wspace=0.12, left=0.155, right=0.985, top=t, bottom=b) for t, b in zip(tops, bottoms)]
    if y > H - 0.14 * len(foot) - 0.05:
        raise Skipped("figure 17 layout: the footer overlaps the last row")

    def frame(ax, ys, labels, xl, title, first):
        ax.axvspan(0, xl[1], color="#F3F6FA", zorder=0)
        ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
        ax.set_xlim(*xl); ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
        ax.set_yticks(ys); ax.set_yticklabels(labels if first else [""] * len(labels), fontsize=6.0, linespacing=1.0)
        ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=6.0)
        ax.set_title(title, fontsize=6.9, loc="left", pad=4, linespacing=1.05)

    # a-c: primary readouts, six seeds and the release configuration
    ys1 = [-i for i in range(n + 1)]
    for j, (m, title) in enumerate(EXT_PRIMARY):
        ax = fig.add_subplot(grids[0][j])
        six = [pt(get(k, "primary", m)) for k, _ in sets]
        three = [pt(get(k, "release configuration", m)) for k, _ in sets]
        vals = [v for q in six + three if q for v in q[:3] if v is not None]
        xl = lims(vals, (-5, 5), (-45, 50), step=5, margin=3)
        frame(ax, ys1, [nm for _, nm in sets] + [EXT_SEALED], xl, "abc"[j] + "   " + title, j == 0)
        ax.set_ylim(-n - 0.6, 0.75)
        for i, (q6, q3) in enumerate(zip(six, three)):
            if q6:
                ci(ax, -i + 0.13, q6[0], q6[1], q6[2], xl, LITE, q6[3], marker="o", s=16)
                ax.text(min(max(q6[0], xl[0]), xl[1]), -i + 0.42, f"{q6[0]:+.1f}".replace("-", "−"), fontsize=5.3, ha="center", va="bottom", color="#1A1A1A")
            if q3:
                ci(ax, -i - 0.2, q3[0], q3[1], q3[2], xl, GREY, q3[3], marker="D", s=9, lw=0.8)
        ax.axhspan(-n - 0.5, -n + 0.5, color="#FAFAFA", zorder=0)
        ax.text(xl[0] + 0.03 * (xl[1] - xl[0]), -n, "not scored" if j else "trained and sealed; not scored",
                fontsize=5.4, color=GREY, style="italic", va="center", ha="left")
        if j == 0:
            hs = [Line2D([], [], marker="o", ls="", color=LITE, markersize=4, label="0.2.0, six seeds" if v2 else "six seeds")]
            if any(pt(get(k, "release configuration", m_)) for k, _ in sets for m_, _ in EXT_PRIMARY):
                hs.append(Line2D([], [], marker="D", ls="", color=GREY, markersize=3.4, label="three seeds (release configuration)"))
            ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=2, fontsize=5.7, handletextpad=0.2, columnspacing=1.0,
                      borderaxespad=0.1, frameon=False)
    # d-f: by forecast day, six seeds
    ys2 = [-i for i in range(n)]
    for j, (m, title) in enumerate(EXT_PRIMARY):
        ax = fig.add_subplot(grids[1][j])
        pts = {(k, rd): pt(get(k, "primary" if rd == "hourly" else "other readouts", m, rd)) for k, _ in sets for rd, _, _ in days}
        vals = [v for q in pts.values() if q for v in q[:3] if v is not None]
        xl = lims(vals, (-5, 5), (-45, 50) if not v2 else (-80, 60), step=5, margin=3)
        frame(ax, ys2, [nm for _, nm in sets], xl, "def"[j] + "   " + title, j == 0)
        ax.set_ylim(-n + 0.5, 0.6)
        offs = np.linspace(0.3, -0.3, len(days)) if len(days) > 1 else [0.0]
        for i, (k, _) in enumerate(sets):
            for (rd, mk, _), dy in zip(days, offs):
                q = pts[(k, rd)]
                if q:
                    ci(ax, -i + dy, q[0], q[1], q[2], xl, LITE, q[3], marker=mk, s=11, lw=0.8)
        for i in range(1, n):
            ax.axhline(-i + 0.5, color="#E6E6E6", lw=0.5, zorder=0)
        if j == 0:
            ax.legend(handles=[Line2D([], [], marker=mk, ls="", markerfacecolor="white", markeredgecolor=LITE, markersize=3.8, label=lab)
                               for _, mk, lab in days],
                      loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=len(days), fontsize=5.7, handletextpad=0.15, columnspacing=1.0,
                      borderaxespad=0.1, frameon=False)
    # g-i: descriptive contrasts with ankyra-lite's own reduced configurations (0.2.0: by lead time)
    reads = EXT_DESC_READS if v2 else [("hourly", None, "")]
    for j, (blk, title) in enumerate(desc):
        ax = fig.add_subplot(grids[2][j])
        qs = {(k, rd): pt(get(k, blk, None, rd)) for k, _ in sets for rd, _, _ in reads}
        vals = [v for q in qs.values() if q for v in q[:3] if v is not None]
        xl = lims(vals, (-4, 4), (-45, 50), step=2, margin=1.5)
        frame(ax, ys2, [nm for _, nm in sets], xl, "ghi"[j] + "   " + title, j == 0)
        ax.set_ylim(-n + 0.5, 0.6)
        offs = np.linspace(0.28, -0.28, len(reads)) if len(reads) > 1 else [0.0]
        for i, (k, _) in enumerate(sets):
            for (rd, mk, _), dy in zip(reads, offs):
                q = qs[(k, rd)]
                if not q:
                    continue
                ci(ax, -i + dy, q[0], q[1], q[2], xl, LITE, q[3], marker=mk or ("o" if j != 1 else "D"), s=14 if not v2 else 11,
                   lw=1.1 if not v2 else 0.8)
                if not v2 or rd == "hourly":
                    ax.text(min(max(q[0], xl[0]), xl[1]), -i + dy + 0.2 * (1 if not v2 else 0.55), f"{q[0]:+.1f}".replace("-", "−"),
                            fontsize=5.3, ha="center", va="bottom", color="#1A1A1A")
        if v2:
            for i in range(1, n):
                ax.axhline(-i + 0.5, color="#E6E6E6", lw=0.5, zorder=0)
            if j == 0:
                ax.legend(handles=[Line2D([], [], marker=mk, ls="", markerfacecolor="white", markeredgecolor=LITE, markersize=3.8, label=lab)
                                   for _, mk, lab in reads],
                          loc="lower left", bbox_to_anchor=(0.0, 1.17), ncol=len(reads), fontsize=5.7, handletextpad=0.15,
                          columnspacing=1.0, borderaxespad=0.1, frameon=False)
    xlabels = ("Hourly improvement of ankyra-lite (%)",
               "Hourly improvement of ankyra-lite in the lead times named (%)" if v2 else
               "Hourly improvement of ankyra-lite in the forecast days named (%)",
               "Hourly improvement of the first configuration over the second (%)")
    for b, text in zip(bottoms, xlabels):
        fig.text(0.155 + (0.985 - 0.155) / 2, b - 0.24 / H, text, fontsize=6.3, ha="center", va="top")
    put_caption(fig, cap, H)
    put_footer(fig, foot, H)
    save(fig, "fig17_external_tests")


# ============================================================================ Figure 18: the training-recipe study behind the released recipe
RH_CONTRASTS = [("R vs F", LITE, "o"), ("V vs F", "#4A6FA5", "s"), ("V vs R", "#8C6D31", "D")]
RH_GROUPS = [("short", "Three short-record\npopulations"), ("switch", "Switch layer\n(short-record)"),
             ("nonswitch", "Non-switch layer\n(short-record)"), ("nine", "Nine populations\n(equal weight)"),
             ("dev", "Six development\npopulations"), ("long", "Four long-record\npopulations")]
RH_READOUTS = [("hourly", "Hourly"), ("energy", "Monthly energy"), ("energy_traj", "Monthly energy\n(trajectory)"), ("peak", "Peak"),
               ("d1", "Forecast day 1"), ("d2_7", "Days 2–7"), ("d8_31", "Days 8–31")]
FACE_NAME = {"bdg2": "BDG2 2017", "cambridge": "Cambridge", "heew": "HEEW Arizona", "eweld": "EWELD", "goiener_confirm": "GoiEner non-household",
             "households": "GoiEner households", "oslo": "Oslo", "drammen": "Drammen", "cineldi": "CINELDI"}
SHORT_RECORD = ("BDG2 2017", "GoiEner non-household", "GoiEner households")


def _rh_group(g):
    """Key of a recipe_history.csv group: a key of RH_GROUPS, ("pop", population) or ("other", text)."""
    s = " ".join(g.lower().replace("_", " ").split())
    names = {**{k.replace("_", " "): v for k, v in FACE_NAME.items()}, **{v.lower(): v for v in ALL}, **{SHORT[v].lower(): v for v in ALL}}
    if s in names:
        return ("pop", names[s])
    if "switch" in s:
        return "nonswitch" if re.search(r"non[- ]?switch", s) else "switch"
    for key, words in (("short", ("short",)), ("nine", ("nine",)), ("dev", ("development", "six")), ("long", ("long", "four"))):
        if any(w in s for w in words):
            return key
    return ("other", g)


def _rh_readout(r):
    s = r.lower().replace("–", "-")
    if "hour" in s:
        return "hourly"
    if "curve" in s or "trajectory" in s:
        return "energy_traj"
    if "month" in s or "energy" in s:
        return "energy"
    if "peak" in s:
        return "peak"
    if re.search(r"2\s*-\s*7", s):
        return "d2_7"
    if re.search(r"8\s*-\s*31", s):
        return "d8_31"
    if re.search(r"\bday\s*1\b", s):
        return "d1"
    return r


def _rh_contrast(c):
    m = re.match(r"\s*([A-Za-z0-9]+?)(?:_ens)?\s*(?:vs\.?|versus|over)\s*([A-Za-z0-9]+?)(?:_ens)?\s*$", c)
    return f"{m.group(1)} vs {m.group(2)}" if m else c.strip()


def _rh_rows(raw):
    """recipe_history.csv as rows (group, contrast, readout, pct, lo, hi, label). Two layouts are read: these seven columns,
    or the layout of evaluation/recipe_tables.py (comparison, populations, layer, quantity, improvement_pct,
    interval_low_pct, interval_high_pct, label, by_population_pct), whose per-population point estimates (the
    by_population_pct of the nine-population rows) become rows without an interval."""
    cols = set(raw[0]) if raw else set()
    if {"group", "contrast", "readout", "pct", "lo", "hi"} <= cols:
        return raw
    if not {"comparison", "quantity", "improvement_pct", "interval_low_pct", "interval_high_pct"} <= cols:
        raise Skipped(f"recipe_history.csv has neither layout the figure reads (columns: {', '.join(sorted(cols))})")
    out = []
    for r in raw:
        layer = (r.get("layer") or "").strip().lower()
        n = _num(r.get("populations"))
        if layer.startswith("non-switch"):
            g = "non-switch layer"
        elif layer.startswith("switch"):
            g = "switch layer"
        else:
            g = {3: "three short-record", 9: "nine", 6: "six development", 4: "four long-record"}.get(int(n) if n else -1,
                                                                                                  r.get("population_group", ""))
        out.append({"group": g, "contrast": r["comparison"], "readout": r["quantity"], "pct": r["improvement_pct"],
                    "lo": r["interval_low_pct"], "hi": r["interval_high_pct"], "label": r.get("label", "")})
        if g == "nine" and _rh_readout(r["quantity"]) == "hourly" and r.get("by_population_pct"):
            for item in r["by_population_pct"].split(";"):
                m = re.match(r"\s*(.+?)\s+([-+0-9.eE]+)\s*$", item)
                if m:
                    out.append({"group": m.group(1), "contrast": r["comparison"], "readout": "hourly", "pct": m.group(2), "lo": "", "hi": "",
                                "label": "point estimate"})
    return out


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def fig_recipe_history():
    raw = rows("recipe_history.csv")
    if not raw:
        raise Skipped("recipe_history.csv has no rows")
    v2rows = [r for r in raw if (r.get("reading") or "").startswith(V2_TEST_READ)]           # 0.2.0: the study's one test read
    # the recipe study (R, F, V): no reading column (0.1.x) or, in a 0.2.0 file, the reading "0.1.0 study reading of the
    # recipe arms" (evaluation/recipe_tables.py write); the 0.2.0 study's rows (development check, test read) are left out
    raw = [r for r in raw if not (r.get("reading") or "").strip().startswith(V2_STUDY)]
    if not raw:
        raise Skipped("recipe_history.csv has no rows of the recipe study")
    raw = _rh_rows(raw)
    v2 = _v2_study(v2rows)
    R = []
    for r in raw:
        p = _num(r["pct"])
        if p is None:
            continue
        R.append({"g": _rh_group(r["group"]), "graw": r["group"], "c": _rh_contrast(r["contrast"]), "rd": _rh_readout(r["readout"]),
                  "p": p, "lo": _num(r["lo"]), "hi": _num(r["hi"]), "label": r.get("label") or ""})
    is_pop = lambda x: isinstance(x["g"], tuple) and x["g"][0] == "pop"            # noqa: E731
    agg, popr = [x for x in R if not is_pop(x)], [x for x in R if is_pop(x)]
    known = [k for k, _ in RH_GROUPS]
    gkeys = [k for k in known if any(x["g"] == k for x in agg)]
    gkeys += list(dict.fromkeys(x["g"] for x in agg if x["g"] not in known))
    gname = dict(RH_GROUPS)
    gname.update({g: "\n".join(textwrap.wrap(g[1], 22)) for g in gkeys if isinstance(g, tuple)})
    rdname = dict(RH_READOUTS)
    cstyle = {c: (col, mk) for c, col, mk in RH_CONTRASTS}
    agg = [x for x in agg if x["c"] in cstyle]            # the three recipe contrasts (rows against the un-continued stage are not drawn)
    contrasts = [c for c, *_ in RH_CONTRASTS if any(x["c"] == c for x in agg)]
    hourly = {(x["g"], x["c"]): x for x in agg if x["rd"] == "hourly"}
    rows_a = [(g, c) for g in gkeys for c in contrasts if (g, c) in hourly]
    if not rows_a:
        raise Skipped("recipe_history.csv has no hourly rows of a group of populations")
    gkeys = [g for g in gkeys if any(k[0] == g for k in rows_a)]
    rf = "R vs F"
    b_groups = [g for g in ("short", "dev", "nine") if any(x["g"] == g and x["c"] == rf and x["rd"] != "hourly" for x in agg)]
    b_reads = [k for k, _ in RH_READOUTS if any(x["g"] in b_groups and x["c"] == rf and x["rd"] == k for x in agg)]
    b_reads += list(dict.fromkeys(x["rd"] for x in agg if x["g"] in b_groups and x["c"] == rf and x["rd"] not in rdname))
    pop_rf = [x for x in popr if x["c"] == rf and x["rd"] == "hourly"]
    pops = [s for s in ALL if any(x["g"][1] == s for x in pop_rf)]

    def res(x):
        if x["lo"] is not None and x["hi"] is not None:
            return 1 if x["lo"] > 0 else (-1 if x["hi"] < 0 else 0)
        t = x["label"].lower()
        return 0 if "not" in t else (1 if "positive" in t else (-1 if "negative" in t else 0))

    def against_r(x):
        a, b = (x["c"].split(" vs ") + [""])[:2]
        return (a == "R" and res(x) < 0) or (b == "R" and res(x) > 0)

    def fmt(x):
        iv = f" [{x['lo']:+.2f}, {x['hi']:+.2f}]" if x["lo"] is not None and x["hi"] is not None else ""
        return f"{x['p']:+.2f}{iv}"

    one = lambda t: t.replace("\n", " ")                                           # noqa: E731
    neg = [x for x in agg if against_r(x)]
    low = min(pop_rf, key=lambda x: x["p"]) if pop_rf else None
    rel = ("recipe R (the span return; the released recipe Rw adds the inverse-noise loss weights to it)" if v2 else "the released recipe")
    cap = wrap(f"{'a' + chr(8211) + 'c: t' if v2 else 'T'}he training-recipe study behind {rel}, on the test windows (read before: a description, not a confirmation). "
               "All three recipes start from the same checkpoints (three seeds) trained from scratch with early stopping on the 93-day "
               "validation set. F and R then train exactly 8 more epochs: F on the unchanged training set (control); R"
               + (" with" if v2 else ", the released recipe, with") + " the validation span returned to the training set. V instead re-splits with a proportional validation "
               "set (10% of the population's own record, 38–93 days) and stops early on it (5 to 30 epochs; the start weights count as "
               "epoch 0). Improvement of the first recipe "
               "over the second (unit-equal RMS ratio of the three-seed ensembles; positive = the first is better); 95% intervals combine the "
               "unit-and-month bootstrap with the spread over seeds (Welch); filled = interval excludes zero. Switch layer: short-record test "
               "windows whose season had no same-season month in the training set before the return and has one after it. On the four "
               "long-record populations V's validation set is the same 93 days."
               + (" d–e: the 0.2.0 study's test read (once; the test windows had been read before, so directions only), the "
                  "configurations named by recipe (R, Rw), seeds (3: seeds 0–2; 6) and the continuation layer: 0.2.0 (Rw6 with the layer) "
                  "against the 0.1.0 configuration (R3) and the single changes (the layer on the same Rw6 carrier, the loss weighting Rw6 vs "
                  "R6, six vs three seeds R6 vs R3); positive = the first is better; 95% intervals as recorded (z intervals that combine "
                  "the unit-and-month bootstrap with the spread of the two three-seed groups where the record has them, else "
                  "unit-and-month only). Energy and peak are read before the layer (readout rule A), so the layer contrast is zero by construction on "
                  "the energy rows (the readout and the sum of the trajectory before the layer) and on the peak; the energy of the delivered "
                  "trajectory, which the layer does change, is in figure 9c." if v2 else ""))
    disc = []
    if neg:
        disc.append("Resolved against R: " + "; ".join(f"{one(gname.get(x['g'], x['graw']))}, {one(rdname.get(x['rd'], x['rd'])).lower()}, "
                                                        f"{x['c']} {x['p']:+.2f}%" for x in neg) + ".")
    if low is not None and low["p"] < 0:
        disc.append(f"Lowest population estimate of R vs F (hourly): {low['g'][1]} {low['p']:+.1f}%.")
    foot = wrap(" ".join(disc), fs=5.6, width=160) if disc else []
    has_b, has_c = bool(b_reads), bool(pops)
    na = len(rows_a) + 0.6 * (len(gkeys) - 1)
    nb = max(len(b_reads) * (0.5 + 0.3 * len(b_groups)) if has_b else 0, len(pops) * 0.95 if has_c else 0)
    top_h = 0.55 + 0.125 * na
    bot_h = (1.03 + 0.125 * nb) if (has_b or has_c) else 0.0
    v2_axes = (0.2 + 0.165 * max(len(v2["reads"]), 1.3 * len(v2["groups"]))) if v2 else 0.0
    v2_h = (v2_axes + 1.0) if v2 else 0.0                   # the 0.2.0 row: axes, and above them its titles and the legend
    base = 0.42 + 0.12 * len(foot)                          # inches: the footer and the last row's tick labels and axis label
    H = cap_height(cap) + 0.3 + top_h + bot_h + v2_h + 0.12 * len(foot) + 0.12
    fig = plt.figure(figsize=(7.2, H))
    ya_top = 1 - (cap_height(cap) + 0.38) / H
    ya_bot = ya_top - (top_h - 0.3) / H
    ga = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], wspace=0.03, left=0.17, right=0.985, top=ya_top, bottom=ya_bot)
    ax = fig.add_subplot(ga[0]); tx = fig.add_subplot(ga[1], sharey=ax)
    ys, y, ticks = {}, 0.0, []
    for g in gkeys:
        start = y
        for c in contrasts:
            if (g, c) in hourly:
                ys[(g, c)] = -y; y += 1
        ticks.append((-(start + y - 1) / 2, gname.get(g, str(g))))
        y += 0.6
    ends = [v for k in rows_a for v in (hourly[k]["lo"], hourly[k]["hi"]) if v is not None]
    xl = lims([hourly[k]["p"] for k in rows_a] + ends, (-3, 3), (-15, 15), step=1, margin=0.5)
    ax.axvspan(0, xl[1], color="#F3F6FA", zorder=0)
    ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
    for j in range(1, len(gkeys)):
        sep = (min(v for (g, _), v in ys.items() if g == gkeys[j - 1]) + max(v for (g, _), v in ys.items() if g == gkeys[j])) / 2
        ax.axhline(sep, color="#E0E0E0", lw=0.5, zorder=0)
    for k in rows_a:
        x = hourly[k]; col, mk = cstyle[k[1]]
        ci(ax, ys[k], x["p"], x["lo"], x["hi"], xl, col, res(x) != 0, marker=mk, s=16)
        tx.text(0.03, ys[k], f"{k[1]}   {fmt(x)}", va="center", ha="left", fontsize=5.7, color="#1A1A1A",
                fontweight="bold" if res(x) != 0 else "normal")
    ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=6.2, linespacing=1.05)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-(y - 0.6) + 0.4, 0.6)
    ax.set_xlim(*xl); ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
    ax.set_xlabel("Hourly improvement of the first recipe over the second (%)", fontsize=6.5)
    ax.set_title("a   Hourly error: the three recipes compared", fontsize=7.4, pad=14)
    tx.axis("off"); tx.set_xlim(0, 1)
    tx.text(0.03, 1.0, "% [95% interval]; bold: resolved", transform=tx.transAxes, fontsize=5.7, color=GREY, va="bottom", ha="left")
    names = ({"R vs F": "R vs F (control)"} if v2 else {"R vs F": "R (released) vs F (control)", "V vs R": "V vs R (released)"})
    ax.legend(handles=[Line2D([], [], marker=cstyle[c][1], ls="", color=cstyle[c][0], markersize=4.2, label=names.get(c, c)) for c in contrasts],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=len(contrasts), fontsize=5.9, handletextpad=0.25, columnspacing=1.2,
              borderaxespad=0.15)
    if has_b or has_c:
        gb = fig.add_gridspec(1, 2, width_ratios=[1.25 if has_b else 0.01, 1.0 if has_c else 0.01], wspace=0.5, left=0.17, right=0.985,
                              top=ya_bot - 0.86 / H, bottom=(base + v2_h) / H)
        if has_b:
            bx = fig.add_subplot(gb[0])
            marks = {"short": "o", "dev": "s", "nine": "D"}
            off = np.linspace(0.25, -0.25, len(b_groups)) if len(b_groups) > 1 else [0.0]
            yr = np.arange(len(b_reads))[::-1]
            sel = [x for x in agg if x["g"] in b_groups and x["c"] == rf and x["rd"] in b_reads]
            xlb = lims([v for x in sel for v in (x["p"], x["lo"], x["hi"]) if v is not None], (-3, 3), (-12, 12), step=1, margin=0.3)
            bx.axvspan(0, xlb[1], color="#F3F6FA", zorder=0); bx.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
            for g, dy in zip(b_groups, off):
                for k, yy in zip(b_reads, yr):
                    x = next((x for x in sel if x["g"] == g and x["rd"] == k), None)
                    if x is not None:
                        ci(bx, yy + dy, x["p"], x["lo"], x["hi"], xlb, LITE, res(x) != 0, marker=marks.get(g, "^"), s=12, lw=0.9)
            bx.set_yticks(yr); bx.set_yticklabels([rdname.get(k, k) for k in b_reads], fontsize=6.1, linespacing=1.0)
            bx.tick_params(axis="y", length=0); bx.set_xlim(*xlb); bx.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
            bx.set_ylim(-0.6, len(b_reads) - 0.4)
            bx.set_xlabel("Improvement of R over F (%)", fontsize=6.5)
            bx.set_title("b   R vs F by readout", fontsize=7.4, pad=13)
            blab = {"short": "three short-record", "dev": "six development", "nine": "nine (equal weight)"}
            bx.legend(handles=[Line2D([], [], marker=marks.get(g, "^"), ls="", markerfacecolor="white", markeredgecolor=LITE, markersize=4,
                                      label=blab[g]) for g in b_groups],
                      loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=len(b_groups), fontsize=5.6, handletextpad=0.15,
                      columnspacing=0.9, borderaxespad=0.1)
        if has_c:
            cx = fig.add_subplot(gb[1])
            yc = np.arange(len(pops))[::-1]
            xlc = lims([x["p"] for x in pop_rf if x["g"][1] in pops], (-4, 4), (-20, 20), step=1, margin=1.5)
            cx.axvspan(0, xlc[1], color="#F3F6FA", zorder=0); cx.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
            iv_all = True
            for s_, yy in zip(pops, yc):
                x = next(x for x in pop_rf if x["g"][1] == s_)
                has_iv = x["lo"] is not None and x["hi"] is not None
                iv_all &= has_iv
                ci(cx, yy, x["p"], x["lo"], x["hi"], xlc, LITE, res(x) != 0 if has_iv else True, marker="o" if has_iv else "D", s=13)
                cx.text(x["p"] + (0.035 if x["p"] >= 0 else -0.035) * (xlc[1] - xlc[0]), yy, f"{x['p']:+.1f}".replace("-", "−"), fontsize=5.4,
                        va="center", ha="left" if x["p"] >= 0 else "right", color="#1A1A1A")
            cx.set_yticks(yc); cx.set_yticklabels([pop(s_) for s_ in pops], fontsize=6.2); cx.tick_params(axis="y", length=0)
            for t_, s_ in zip(cx.get_yticklabels(), pops):
                t_.set_fontweight("bold" if s_ in SHORT_RECORD else "normal")
            cx.set_xlim(*xlc); cx.set_ylim(-0.6, len(pops) - 0.4); cx.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
            cx.set_xlabel("Hourly improvement of R over F (%)", fontsize=6.5)
            cx.set_title("c   R vs F by population, hourly", fontsize=7.4, pad=13)
            cx.text(0.0, 1.015, "bold: short-record" + ("" if iv_all else "; diamonds: point estimates only"),
                    transform=cx.transAxes, fontsize=5.6, color=GREY, va="bottom", ha="left")
    if v2:
        _fig18_v2(fig, v2, base, v2_axes, H)
    put_caption(fig, cap, H)
    if foot:
        put_footer(fig, foot, H, color=LITE)
    save(fig, "fig18_recipe_history")


# 0.2.0 rows of recipe_history.csv (evaluation/recipe_tables.py, v2_rows): the study's one test read
V2_TEST_READ = "0.2.0 test read"
V2_STUDY = "0.2.0 "                                  # every reading of the 0.2.0 study (test read, development check) starts so
V2_CONTRASTS = [("primary: 0.2.0 vs 0.1.0", "0.2.0 vs 0.1.0 (Rw6 + layer vs R3)", LITE, "o"),
                ("layer on / off (Rw6", "continuation layer on vs off (Rw6)", "#4A6FA5", "s"),
                ("decomposition: weighting", "loss weighting (Rw6 vs R6)", "#8C6D31", "D"),
                ("decomposition: seeds", "six vs three seeds (R6 vs R3)", "#7B5EA7", "^")]
V2_READS = [("hourly", "Whole window"), ("lead hours 0-3", "Hours 0–2"), ("lead day 1", "Forecast day 1"), ("lead days 2-7", "Days 2–7"),
            ("lead days 8-31", "Days 8–31"), ("monthly (independent readout)", "Monthly energy"),
            ("monthly (curve sum)", "Monthly energy\n(trajectory sum)"), ("peak", "Peak")]
V2_GROUP_ROWS = [("all nine", "Nine populations"), ("BDG2 2017, GoiEner", "Three without a\ndevelopment set"),
                 ("GoiEner households", "GoiEner households"), ("Cambridge, HEEW", "Six development\npopulations")]
V2_GROUP_READS = [("hourly", "o", "whole window"), ("lead hours 0-3", "v", "hours 0–2"), ("lead day 1", "s", "forecast day 1")]


def _v2_study(rows_):
    """The 0.2.0 test-read rows drawn in figure 18 d-e: {"pts": {(contrast, group, quantity): (pct, lo, hi, resolved)},
    "contrasts", "reads", "groups"}, or None when the file has none (0.1.x results)."""
    pts = {}
    for r in rows_:
        c = next((k for k, *_ in V2_CONTRASTS if (r.get("comparison") or "").startswith(k)), None)
        g = next((k for k, _ in V2_GROUP_ROWS if (r.get("population_group") or "").startswith(k)), None)
        p = _num(r.get("improvement_pct"))
        if c is None or g is None or p is None:
            continue
        lo, hi = _num(r.get("interval_low_pct")), _num(r.get("interval_high_pct"))
        res = 0 if lo is None or hi is None else (1 if lo > 0 else (-1 if hi < 0 else 0))
        pts[(c, g, r.get("quantity", ""))] = (p, lo, hi, res)
    if not pts:
        return None
    contrasts = [k for k, *_ in V2_CONTRASTS if any(x[0] == k and x[1] == "all nine" for x in pts)]
    reads = [k for k, _ in V2_READS if any(x[0] in contrasts and x[1] == "all nine" and x[2] == k for x in pts)]
    groups = [k for k, _ in V2_GROUP_ROWS if any(x[0] == V2_CONTRASTS[0][0] and x[1] == k for x in pts)]
    return {"pts": pts, "contrasts": contrasts, "reads": reads, "groups": groups}


def _fig18_v2(fig, v2, base, axes_h, H):
    """Figure 18 d-e: the 0.2.0 study's test read. d: the nine populations by readout, 0.2.0 against the 0.1.0
    configuration and the single changes; e: 0.2.0 against 0.1.0 by population group and lead time."""
    pts, contrasts, reads, groups = v2["pts"], v2["contrasts"], v2["reads"], v2["groups"]
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.5, left=0.17, right=0.985, top=(base + axes_h) / H, bottom=base / H)
    sty = {k: (lab, col, mk) for k, lab, col, mk in V2_CONTRASTS}
    rname = dict(V2_READS)

    def frame(ax, vals, title, xlabel, pad=13):
        xl = lims(vals, (-2, 4), (-60, 80), step=1 if max(map(abs, vals or [0])) < 8 else 5, margin=0.5)
        if xl[1] - xl[0] > 20:
            # on the log part a linear margin is too small: widen the ends so the markers at the extremes stay inside the axes
            xl = (xl[0] * 1.5 if xl[0] < -2 else xl[0], xl[1] * 1.5 if xl[1] > 2 else xl[1])
            ax.set_xscale("symlog", linthresh=2.0, linscale=0.8)
            ax.set_xticks([t_ for t_ in (-20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 60) if xl[0] <= t_ <= xl[1]])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}".replace("-", "−") if v else "0"))
        ax.axvspan(0, xl[1], color="#F3F6FA", zorder=0); ax.axvline(0, color="#7F7F7F", lw=0.6, zorder=1)
        ax.set_xlim(*xl); ax.grid(axis="x", color="#EFEFEF", lw=0.4, zorder=0)
        ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=6.0)
        ax.set_xlabel(xlabel, fontsize=6.5)
        ax.set_title(title, fontsize=7.4, pad=pad)
        return xl

    # d: nine populations, by readout
    dx = fig.add_subplot(gs[0])
    yr = np.arange(len(reads))[::-1].astype(float)
    off = np.linspace(0.33, -0.33, len(contrasts)) if len(contrasts) > 1 else [0.0]
    sel = [pts[(c, "all nine", q)] for c in contrasts for q in reads if (c, "all nine", q) in pts]
    dpad = 13 + 9 * ((len(contrasts) + 1) // 2)                     # room for d's legend; e uses the same pad so the titles align
    xl = frame(dx, [v for x in sel for v in x[:3] if v is not None], "d   0.2.0 study, nine populations: by readout",
               "Improvement of the first configuration (%)", pad=dpad)
    for c, dy in zip(contrasts, off):
        lab, col, mk = sty[c]
        for q, yy in zip(reads, yr):
            x = pts.get((c, "all nine", q))
            if x is not None:
                ci(dx, yy + dy, x[0], x[1], x[2], xl, col, x[3] != 0, marker=mk, s=12, lw=0.9)
    dx.set_yticks(yr); dx.set_yticklabels([rname.get(q, q) for q in reads], fontsize=6.1, linespacing=1.0)
    dx.set_ylim(-0.6, len(reads) - 0.4)
    for k in range(1, len(reads)):
        dx.axhline(k - 0.5, color="#E6E6E6", lw=0.5, zorder=0)
    dx.legend(handles=[Line2D([], [], marker=sty[c][2], ls="", color=sty[c][1], markersize=4, label=sty[c][0]) for c in contrasts],
              loc="lower left", bbox_to_anchor=(-0.02, 0.995), ncol=2, fontsize=5.6, handletextpad=0.2, columnspacing=0.9,
              borderaxespad=0.1)
    # e: 0.2.0 vs 0.1.0 by population group and lead time
    if groups:
        ex_ = fig.add_subplot(gs[1])
        c0 = V2_CONTRASTS[0][0]
        rd = [x for x in V2_GROUP_READS if any((c0, g, x[0]) in pts for g in groups)]
        yg = np.arange(len(groups))[::-1].astype(float)
        off = np.linspace(0.25, -0.25, len(rd)) if len(rd) > 1 else [0.0]
        sel = [pts[(c0, g, q)] for g in groups for q, *_ in rd if (c0, g, q) in pts]
        xl = frame(ex_, [v for x in sel for v in x[:3] if v is not None], "e   0.2.0 vs 0.1.0 by population group",
                   "Improvement of 0.2.0 (%)", pad=dpad)
        for (q, mk, _), dy in zip(rd, off):
            for g, yy in zip(groups, yg):
                x = pts.get((c0, g, q))
                if x is not None:
                    ci(ex_, yy + dy, x[0], x[1], x[2], xl, LITE, x[3] != 0, marker=mk, s=12, lw=0.9)
        ex_.set_yticks(yg); ex_.set_yticklabels([dict(V2_GROUP_ROWS)[g] for g in groups], fontsize=6.1, linespacing=1.0)
        ex_.set_ylim(-0.6, len(groups) - 0.4)
        for k in range(1, len(groups)):
            ex_.axhline(k - 0.5, color="#E6E6E6", lw=0.5, zorder=0)
        ex_.legend(handles=[Line2D([], [], marker=mk, ls="", markerfacecolor="white", markeredgecolor=LITE, markersize=4, label=lab)
                            for _, mk, lab in rd],
                   loc="lower left", bbox_to_anchor=(-0.02, 0.995), ncol=len(rd), fontsize=5.6, handletextpad=0.15, columnspacing=0.9,
                   borderaxespad=0.1)


FIGURES = [("fig0_advantage", fig_advantage), ("fig1_architecture", fig_architecture), ("fig1b_data_splits", fig_data_splits),
           ("fig2_test_ranks", fig_test_ranks), ("fig3_test_pairwise", fig_test_forest), ("fig4_ablation", fig_ablation), ("fig5_peak_operator", fig_peak),
           ("fig6_example_window", fig_example), ("fig7_operators", fig_operators), ("fig8_loss_metrics", fig_loss_metrics),
           ("fig9_loss_by_day", fig_lead_days), ("fig9b_energy_by_day", fig_energy_by_day),
           ("fig9c_continuation_layer", fig_continuation),
           ("fig10_loss_by_day_relative", fig_lead_days_relative), ("fig11_energy_error", fig_energy),
           ("fig12_consistency", fig_consistency), ("fig13_intervals", fig_intervals), ("fig14_block_attribution", fig_block_shares),
           ("fig15_rank_tests", fig_rank_tests), ("fig16_intervals_populations", fig_intervals_all),
           ("fig17_external_tests", fig_external_tests), ("fig18_recipe_history", fig_recipe_history)]


def main(argv=None):
    global RES, OUT, STANDINS, STRICT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="comma-separated figure stems or figN prefixes")
    ap.add_argument("--results", default=None, help="results folder (default: ../results)")
    ap.add_argument("--out", default=None, help="output folder (default: this folder)")
    ap.add_argument("--standins", default=os.environ.get("ANKYRA_LITE_FIGURE_STANDINS") or None,
                    help="folder with stand-in result files for missing inputs (watermarked figures)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any figure is skipped or uses stand-in data, if a model class has no colour, or if a result "
                         "file still holds rows of a model outside the comparison set")
    a = ap.parse_args(argv)
    if a.results:
        RES = Path(a.results).resolve()
    if a.out:
        OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    STANDINS = Path(a.standins).resolve() if a.standins else None
    STRICT = bool(a.strict)
    want = [w.strip() for w in a.only.split(",") if w.strip()]
    status = {}
    for stem, fn in FIGURES:
        if want and not any(stem == w or stem.split("_")[0] == w for w in want):
            continue
        CURRENT["stem"], CURRENT["standin"] = stem, set()
        try:
            fn()
            if CURRENT["standin"]:
                status[stem] = "stand-in"
                print(f"{stem}: written from STAND-IN data ({', '.join(sorted(CURRENT['standin']))})")
            else:
                status[stem] = "ok"
                print(f"{stem}: written")
        except Deferred as e:
            plt.close("all"); status[stem] = "deferred"
            print(f"{stem}: skipped, {e}")
        except Missing as e:
            plt.close("all"); status[stem] = "skipped"
            print(f"{stem}: skipped, input not found: {e}")
        except Skipped as e:
            plt.close("all"); status[stem] = "skipped"
            print(f"{stem}: skipped, {e}")
        except (UnknownClass, StaleResults) as e:
            plt.close("all"); status[stem] = "failed"
            print(f"{stem}: FAILED, {e} (--strict)")
        except Exception:                                   # noqa: BLE001  (fail soft: the other figures are still drawn)
            plt.close("all"); status[stem] = "failed"
            print(f"{stem}: FAILED")
            traceback.print_exc()
    counts = {k: sum(v == k for v in status.values()) for k in ("ok", "stand-in", "skipped", "deferred", "failed")}
    print("figures in", OUT, "|", ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    bad = counts["failed"] or (a.strict and (counts["skipped"] or counts["stand-in"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
