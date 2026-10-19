"""Comparison of ankyra-lite (small carrier network + ANKYRA 2.2.1 anchoring layer) with the external baselines, and the
driver that regenerates every file of ../results/.

The tables mirror the results directory of the first release repository (Building_load_ANKYRA/results): "The main
comparison", "Error by forecast day and energy", "Readouts and intervals", "Design choices" and the other sections, on
the test windows (``late`` subset: windows whose origin lies after the training cutoff of the trained baselines, where
every baseline has a forecast).  Nothing is trained: saved forecasts are read and scored with the study's own scoring
functions, imported read-only from the research folder.  Re-anchoring (single seeds, switched-off parts of the anchoring
layer, the anchoring record) is done by ``lite_reanchor.py`` with the unchanged anchoring layer.

Forecasters.  ankyra-lite (the reference) and 21 baselines: the 20 external baselines of the ANKYRA release and
``LSTM-T-KAN-M``, the LSTM-T-KAN model of Shen, Yang, Song and Liang (2026) with its model and training code copied line by
line from the authors' public repository (commit ce24938) and the authors' exp2 feature list (load and calendar).  It is
not a strict reproduction of the paper: the deviations (the panel's data protocol, one 744-step model per population,
per-unit z-scores, at most 30 epochs with patience 8, a learning-rate grid, three seeds, own feature definitions) are
listed in docs/EVALUATION.md.  ANKYRA itself and every ANKYRA version or ablation (F1-off, F1, F0, ANKYRA-1x, -200, -201,
-21) are left out.

Products.  The ankyra-lite forecast of each population is read from the anchored product listed in ``products.json``
(trajectory, level, peak readout, state flags), with the weekly seasonal naive on windows where the anchoring layer's
off-state or micro-load rule fires.  ``products.json`` maps the released recipe of version 0.1.0, the study's span
return (R): the from-scratch stage with early stopping on the 93-day validation set, then the validation span returned
to the training set and 8 more epochs from the selected weights; three seeds averaged.  Its records are the three
span-return runs (``run_dirs``: carrier forecasts, per-epoch record, ``final.pt``), the from-scratch stage they
continued (``start_run_dirs``), the anchored product of each seed alone (``seed_npz``) and the study's other recipe arms
on the same windows (``recipe_variants``).  ``products_p3s.json`` (``--recipe p3s``) maps the from-scratch stage alone
(the study's P3s recipe), not the released recipe.  Drammen and CINELDI were trained with the corrected CINELDI store
(``"store": "corrected"``).  On CINELDI the product's origins are shifted by 1,416 h; its baselines are then read from
the output folder of the study's CINELDI re-scoring on the corrected store (original origin numbers) and its targets,
months and seasonal naive from the corrected store.  LSTM-T-KAN-M on CINELDI is read from its retrain on the corrected store (same code, protocol, learning-rate grid
and seeds; KAN_FIX_DIR, keyed by the original origin numbers like that re-scoring).  If that retrain is missing or its record
does not pass, the original-store forecast is used and every CINELDI row carries CORRECTED_CINELDI_FLAG.

Scoring-code check (step A).  Before the reference is replaced, the late-window rows of ten release tables are
recomputed with the release's own reference forecaster and compared with the published CSV files, read from the
release repository's commit 487e0d2 (ANKYRA_ORIGINAL_REV; ``git show``, read-only) so that later or uncommitted edits
of its working copy do not enter the check.  This validates the
scoring code only; its numbers are not written to any result table.  A table that does not reproduce is not written.

Paths.  ANKYRA_LITE_RESEARCH_ROOT (default <repo>/../HKAD_V3) is the research folder, ANKYRA_ORIGINAL_ROOT (default
<repo>/../Building_load_ANKYRA) the first release repository; both are read-only (an audit hook refuses any write
there).  ANKYRA_LITE_EVAL_CACHE (default: the system temporary directory) holds the re-anchoring cache.

Usage (from the repository root, CPU only):
  python -B evaluation/compare_baselines.py                      every table, nine populations (+ Suzhou park in the check)
  python -B evaluation/compare_baselines.py --reanchor           also run the re-anchoring first (the ensemble with its
                                                                 anchoring record, switched-off parts, a single-seed gate)
  python -B evaluation/compare_baselines.py --faces oslo,bdg2 --out <dir>      a development run into another directory
  python -B evaluation/compare_baselines.py --recipe p3s --out <dir>           the from-scratch stage instead of the release
  python -B evaluation/compare_baselines.py --summary-only       rebuild SUMMARY.md from the JSON record
  python -B evaluation/compare_baselines.py --check-products     list the products and whether their files exist
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # CPU only
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys

sys.dont_write_bytecode = True                                 # never write __pycache__ into the read-only trees
import argparse
import csv
import gc
import io
import json
import math
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"                                     # replaced by --out


def _root(names, default):
    for n in names:
        v = os.environ.get(n)
        if v:
            return Path(v).resolve()
    return Path(default).resolve()


RESEARCH = _root(("ANKYRA_LITE_RESEARCH_ROOT", "ANKYRA_LITE_HKAD"), REPO.parent / "HKAD_V3")
ORIGINAL = _root(("ANKYRA_ORIGINAL_ROOT", "ANKYRA_LITE_RELEASE"), REPO.parent / "Building_load_ANKYRA")
CACHE = _root(("ANKYRA_LITE_EVAL_CACHE",), Path(tempfile.gettempdir()) / "ankyra_lite_eval_cache")
HKAD, REL = RESEARCH, ORIGINAL                                  # short names used by the tests
REL_RES = REL / "results"
AO = HKAD / "analysis_outputs"
P22 = AO / "hcr_baseline_panel_v22"                                   # the release panel (saved baseline forecasts)
SANDBOX = AO / "cineldi_fix_v49"                                      # CINELDI start fix: baselines on the corrected store
SB_PANEL, SB_AUDIT = SANDBOX / "hcr_baseline_panel_v22", SANDBOX / "hcr_f1_path_audit_v1"
FIX_AUDIT_NEW = AO / "cineldi_start_fix_v1" / "audit_new"             # corrected-store window arrays (new origin numbers)
DC_BUILD = AO / "smallnet_sr_formal_v1" / "dc" / "BUILD.json"         # corrected-store record (store path, cuts)
KAN_DIR = AO / "lstm_tkan_baseline_v2"                                # LSTM-T-KAN-M forecasts (authors' public code)
KAN_FIX_DIR = AO / "lstm_tkan_baseline_v2_cineldi_t0fix"              # LSTM-T-KAN-M retrained on the corrected CINELDI store
HOURLY_XCHECK = AO / "smallnet_vs_baselines_v1" / "results.json"      # earlier hourly-only comparison (20 baselines)
NEAR_ZERO = AO / "bdg2_nearzero_sensitivity_v1" / "results.json"      # the three near-zero BDG2 meters (release rule)
SR_SCORE = AO / "smallnet_sr_formal_v1" / "score_all.json"            # the study's one reading of the recipe arms (recipe history)
SPAN_CENSUS = HKAD / "smallnet_overnight_20261017" / "round_b" / "span_coverage_census.json"   # switch-layer window counts
PRODUCTS = HERE / "products.json"                                     # the released recipe (span return)
RECIPE_PRODUCTS = {"R": PRODUCTS, "p3s": HERE / "products_p3s.json"}  # --recipe
ORIGINAL_REV = os.environ.get("ANKYRA_ORIGINAL_REV", "487e0d2")        # the committed release state the anchoring layer was copied from

REF_A, REF_B = "F1-off", "ankyra-lite"                                # panel key of the release reference; key of the product
REF_B_LABEL = "ankyra-lite"
A_FAMILY = ("F1-off", "F1", "F0", "ANKYRA-1x", "ANKYRA-200", "ANKYRA-201", "ANKYRA-21")
A_VERSIONS = ("ANKYRA-1x", "ANKYRA-200", "ANKYRA-201", "ANKYRA-21")
LEFT_OUT = f"ANKYRA itself and its versions ({len(A_FAMILY)} forecaster ids of the first release); no number of them is in any file"
KAN = ("LSTM-T-KAN-M",)                                              # the only LSTM-T-KAN baseline (the load-only version is not used)
KAN_LABEL = {"LSTM-T-KAN-M": "LSTM-T-KAN (authors' public code, exp2 feature list)"}
KAN_CLASS = {"LSTM-T-KAN-M": "load and calendar, trained"}
KAN_NOT_BASELINES = ("LSTM-T-KAN", "LSTM-MLP")                       # name prefixes never read from a panel folder
N_BASELINES = 21
PANEL10 = ("bdg2", "cambridge", "heew", "eweld", "goiener_confirm", "households", "oslo", "drammen", "cineldi", "park")
NINE = PANEL10[:9]
SHOW_NAMES = {"bdg2": "BDG2 2017", "cambridge": "Cambridge", "heew": "HEEW Arizona", "eweld": "EWELD", "goiener_confirm": "GoiEner non-household",
              "households": "GoiEner households", "oslo": "Oslo", "drammen": "Drammen", "cineldi": "CINELDI", "park": "Suzhou park"}   # = the release's set names
LEAD7 = ("bdg2", "cambridge", "heew", "eweld", "goiener_confirm", "households", "park")    # release: six test sets + park
TEST6 = ("bdg2", "cambridge", "heew", "eweld", "goiener_confirm", "households")
SUMMARY_BASELINES = ("TimesFM", "TimesFM-X", "Chronos-2", "Chronos-2-X", "PatchTST", "iTransformer-X", "LSTM", "GBT-T",
                     "LSTM-T-KAN-M", "RIDGE-L", "SN-week")
ORIGINAL_STORE_FLAG = "original CINELDI store (start defect); to be replaced"
CORRECTED_CINELDI_FLAG = ("corrected CINELDI store; LSTM-T-KAN-M: forecast from the original CINELDI store (its corrected-store "
                          "retrain is missing or did not pass its checks; its calendar features differ on the corrected store)")
TIDE_LABEL = "Linear-context (TiDE implementation)"                   # the first release's display label; model id TiDE kept
HORIZON, CONTEXT, DAYS, HR = 744, 1344, 31, 24
MIN_FREE_GB = float(os.environ.get("ANKYRA_LITE_MIN_FREE_GB", "5"))                # wait for this much free memory before loading data
MAX_WAIT_S = float(os.environ.get("ANKYRA_LITE_MAX_WAIT_S", "7200"))
TABLES = ("benchmark_pairwise", "benchmark_mean_unit_rank", "conventional_metrics_late", "energy_error",
          "lead_day_metrics", "lead_day_energy", "block_shares", "rank_tests", "scaled_errors", "peak_readout")


# ----------------------------------------------------------------------------- read-only guard
def _norm(p):
    return os.path.normcase(os.path.abspath(os.fsdecode(p)))


_RO_ROOTS = tuple(_norm(p) for p in (HKAD, REL))
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _in_read_only(path):
    try:
        p = _norm(path)
    except (TypeError, ValueError):
        return False
    return any(p == r or p.startswith(r + os.sep) for r in _RO_ROOTS)


def _audit(event, args):
    """Refuse every write, rename, removal or directory creation under the read-only trees."""
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if path is None or isinstance(path, int):
            return
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if writing and _in_read_only(path):
            raise PermissionError(f"compare_baselines: write refused in a read-only tree: {path}")
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.truncate", "os.chmod", "os.utime", "os.symlink", "os.link",
                   "shutil.rmtree", "shutil.chown"):
        if args and not isinstance(args[0], int) and _in_read_only(args[0]):
            raise PermissionError(f"compare_baselines: {event} refused in a read-only tree: {args[0]}")
    elif event in ("os.rename", "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree", "shutil.move"):
        targets = args[:2] if event in ("os.rename", "shutil.move") else args[1:2]
        if any(not isinstance(t, int) and _in_read_only(t) for t in targets):
            raise PermissionError(f"compare_baselines: {event} refused in a read-only tree: {args[:2]}")


if not getattr(sys, "_ankyra_lite_guard", False):              # one hook per process, whatever the import name
    sys.addaudithook(_audit)
    sys._ankyra_lite_guard = True


# ----------------------------------------------------------------------------- paths in records and logs
def _variants(p):
    s = str(p)
    return {s, s.replace("\\", "/"), s.replace("/", "\\"), s.replace("\\", "\\\\"), s.replace("/", "\\\\")}


def scrub(text):
    """Replace the machine's absolute paths by the names of the roots (no personal path in any result file)."""
    text = str(text)
    subs = [(RESEARCH, "<research root>"), (ORIGINAL, "<original release root>"), (REPO, "<repo>"), (CACHE, "<evaluation cache>"),
            (Path(tempfile.gettempdir()), "<temp>")]
    for root, name in subs:
        for v in sorted(_variants(root), key=len, reverse=True):
            text = re.sub(re.escape(v), name.replace("\\", "\\\\"), text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/'\"\s]+", "<home>", text)
    return text


def show(p):
    """A path as it may appear in a result file: relative to one of the named roots."""
    p = Path(p).resolve()
    for root, name in ((RESEARCH, "research root"), (ORIGINAL, "original release root"), (REPO, "repo"), (CACHE, "evaluation cache")):
        try:
            return f"<{name}>/" + p.relative_to(root).as_posix()
        except ValueError:
            continue
    return scrub(p)


class _Log:
    """Log lines to stdout and to a file in the output directory, with the paths scrubbed."""

    def __init__(self):
        self.fh = None

    def open(self, path):
        self.fh = open(path, "w", encoding="utf-8", newline=chr(10))      # LF line ends on every platform

    def __call__(self, *a):
        line = scrub(" ".join([time.strftime("%H:%M:%S")] + [str(x) for x in a]))
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()


log = _Log()


def wait_for_memory(label, need=None):
    import psutil
    need = MIN_FREE_GB if need is None else need
    t0 = time.time()
    while True:
        free = psutil.virtual_memory().available / 2 ** 30
        if free >= need:
            log(f"[{label}] memory available {free:.2f} GB")
            return free
        if time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"[{label}] less than {need} GB available after {MAX_WAIT_S} s; stopped")
        log(f"[{label}] {free:.2f} GB available, waiting")
        time.sleep(30)


# ----------------------------------------------------------------------------- products
PRODUCT_KINDS = ("faces", "replacements", "superseded")


def load_products(path=PRODUCTS):
    """A products mapping (products.json or products_p3s.json) with resolved paths (relative to the research root).
    Every entry gets npz_path, run_paths, prep_path, checkpoint (weights file name in a run directory) and, where the
    mapping has them, seed_paths (anchored single-seed products), start_paths (the from-scratch stage a continuation
    started from), span_path (the returned validation span) and variant_paths (other recipe arms)."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    rec["path"] = Path(path)

    def res(e):
        e = dict(e)
        e["npz_path"] = AO.parent / e["npz"]
        e["run_paths"] = [AO.parent / r for r in e.get("run_dirs", [])]
        e["prep_path"] = AO.parent / e["prep"] if e.get("prep") else None
        e["seed_paths"] = [AO.parent / r for r in e.get("seed_npz", [])]
        e["start_paths"] = [AO.parent / r for r in e.get("start_run_dirs", [])]
        e["span_path"] = AO.parent / e["span_plan"] if e.get("span_plan") else None
        e["variant_paths"] = {k: AO.parent / v for k, v in e.get("recipe_variants", {}).items()}
        e["checkpoint"] = e.get("checkpoint") or rec.get("checkpoint") or "best.pt"
        e["start_checkpoint"] = e.get("start_checkpoint") or rec.get("start_checkpoint") or "best.pt"
        if e.get("store") not in ("original", "corrected"):
            raise SystemExit(f"products: store must be 'original' or 'corrected', not {e.get('store')!r}")
        for k in ("run_paths", "seed_paths", "start_paths"):
            if e[k] and len(e[k]) != len(rec.get("seeds", (0, 1, 2))):
                raise SystemExit(f"products: {k} must list one entry per seed")
        return e

    for kind in PRODUCT_KINDS:
        rec[kind] = {f: res(e) for f, e in rec.get(kind, {}).items()}
    return rec


def check_products(prods):
    out = {"mapping": show(prods["path"]), "label": prods.get("label"), "recipe": prods.get("recipe")}
    for kind in PRODUCT_KINDS:
        for f, e in prods[kind].items():
            ck = e["checkpoint"]
            out[f"{kind}:{f}"] = {"store": e["store"], "npz": show(e["npz_path"]), "exists": e["npz_path"].exists(),
                                  "run_dirs_exist": all(p.exists() for p in e["run_paths"]),
                                  "checkpoints_exist": all((p / ck).exists() for p in e["run_paths"]),
                                  "seed_products_exist": all(p.exists() for p in e["seed_paths"]) if e["seed_paths"] else None,
                                  "start_runs_exist": all(p.exists() for p in e["start_paths"]) if e["start_paths"] else None,
                                  "variants_exist": {k: p.exists() for k, p in e["variant_paths"].items()} or None}
    return out


def on_corrected_cineldi(face, entry):
    """True when the population's own data are the corrected CINELDI store (origins shifted by 1,416 h)."""
    return face == "cineldi" and entry["store"] == "corrected"


def corrected_store_info():
    """Corrected CINELDI store: its path and the hour shift against the original store (from the study's records)."""
    b = json.loads(DC_BUILD.read_text(encoding="utf-8"))
    return Path(b["store"]), int(b["cuts"]["C"])


def kan_fixed_record():
    """The record of the LSTM-T-KAN-M retrain on the corrected CINELDI store, or None when the retrain is missing or
    did not pass (its original-store gate reproduced the stored forecast; its corrected-store checks passed)."""
    try:
        gate = json.loads((KAN_FIX_DIR / "gate.json").read_text(encoding="utf-8"))
        rec = json.loads((KAN_FIX_DIR / "record.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (gate.get("pass") and rec.get("checks", {}).get("pass") and (KAN_FIX_DIR / "cineldi" / f"pred_{KAN[0]}.npz").exists()):
        return None
    return {"gate": gate, "record": rec}


def kan_dir(face, entry):
    """(folder of the LSTM-T-KAN-M forecast of one population, whether it is the corrected-store retrain)."""
    if on_corrected_cineldi(face, entry) and kan_fixed_record() is not None:
        return KAN_FIX_DIR / face, True
    return KAN_DIR / face, False


def face_flag(face, entry):
    """The flag of every row of one population: the original CINELDI store (Drammen, CINELDI), or on the corrected
    CINELDI store a baseline kept from the original store (LSTM-T-KAN-M when its corrected-store retrain is missing; it
    enters every CINELDI rank and test).  Empty when every forecast of the population is on its own correct store."""
    if face in ("drammen", "cineldi") and entry["store"] == "original":
        return ORIGINAL_STORE_FLAG
    if on_corrected_cineldi(face, entry) and not kan_dir(face, entry)[1]:
        return CORRECTED_CINELDI_FLAG
    return ""


# ----------------------------------------------------------------------------- the study's scoring modules
class Tools:
    """The study's scoring modules, imported read-only (their main() functions are never called)."""

    def __init__(self):
        for p in (str(REL), str(HKAD)):
            if p not in sys.path:
                sys.path.insert(0, p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tools_hcr_baseline_panel_v1 as B
            import tools_hcr_ext_drammen_score_v1 as DS
            import tools_hcr_lead_day_metrics_v1 as L
            import tools_hcr_energy_error_v1 as E
            import tools_hcr_block_shares_v1 as S
            import tools_lead_day_energy_v6_v1 as LE            # sets B.OUT / B.FAMILY at import; reset below
            import tools_presub_stats_v1 as ST
            import tools_release_results_export_v1 as X1
            import tools_ankyra_v22_release_v1 as R22
            from recovery import eo_peak_readout_scalar_v1 as PK
        self.B, self.DS, self.L, self.E, self.S, self.LE, self.ST, self.X1, self.R22, self.PK = B, DS, L, E, S, LE, ST, X1, R22, PK
        self.PWF = None
        self.V = None
        B.OUT, B.FAMILY, B.FLAGSHIP = P22, (), REF_A
        assert set(R22.FAMILY21) == set(A_FAMILY[1:]), R22.FAMILY21          # the release's own list of ANKYRA versions
        self.names = {k: n for k, n, *_ in X1.SETS}
        self.tiers = {k: t for k, _, t, *_ in X1.SETS}
        self.near_zero = json.loads(NEAR_ZERO.read_text(encoding="utf-8"))["near_zero_units"]

    def pwf(self):
        if self.PWF is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import tools_smallnet_pwformal_v1 as PWF
            self.PWF = PWF
            self.V = PWF.V
            self.B.OUT = P22
        return self.PWF

    def label(self, m, ref, ref_label):
        if m == ref:
            return ref_label
        if m in KAN:
            return KAN_LABEL[m]
        if m in self.R22.LABELS:
            return self.R22.LABELS[m][0]
        return self.X1.DISPLAY.get(m, m)

    def mclass(self, m, ref):
        if m == ref:
            return ""
        if m in KAN:
            return KAN_CLASS[m]
        if m in self.R22.LABELS:
            return self.R22.LABELS[m][1]
        return self.X1.MODEL_CLASS.get(m, "")


def contrast_um(T, a, b, unit, month, rng):
    """The study's unit-equal log ratio with its unit x month bootstrap (the same estimator and random stream as
    tools_hcr_ext_drammen_score_v1.contrast, without the month-equal summary, which draws no random numbers)."""
    pt = T.DS.log_ratio(a, b, unit)
    bt = T.DS.boot(a, b, unit, month, rng)
    return {"log_ratio": pt, "UM": bt["UM"], "pooled_ratio": float(a.mean() / b.mean())}


def ape_mum(err, ypk, unit):
    """Median over units of the unit's median absolute percentage peak error (tools_hcr_f1_readouts_v1.ape_mum)."""
    ok = ypk > 1e-6
    a = err[ok] / ypk[ok]
    u = unit[ok]
    return float(100 * np.median([np.median(a[u == x]) for x in np.unique(u)]))


def unit_rank(preds, y, unit, ref):
    """Mean per-unit rank of ``ref`` among ``preds`` (tools_hcr_baseline_panel_v1.block: unit RMSE rounded to 12
    decimals, ties averaged) and the rank position (1 + forecasters with a lower mean rank)."""
    from scipy.stats import rankdata
    uu, inv = np.unique(unit, return_inverse=True)
    names = list(preds)
    R = np.stack([np.sqrt(np.bincount(inv, ((preds[k] - y) ** 2).mean(1), len(uu)) / np.bincount(inv, minlength=len(uu))) for k in names], 1)
    rk = np.apply_along_axis(lambda r: rankdata(np.round(r, 12), method="average"), 1, R).mean(0)
    mr = dict(zip(names, (float(v) for v in rk)))
    return mr[ref], 1 + sum(v < mr[ref] - 1e-9 for v in mr.values()), len(names), mr


# ----------------------------------------------------------------------------- data
def load_panel(T, face, full_peak=False, panel=P22, audit=None):
    """Panel windows of one population: the release loader (tools_hcr_baseline_panel_v1.load_set; its cache write is
    verified instead of written), every saved forecast of ``panel`` aligned on (building, origin), the late subset
    exactly as tools_hcr_baseline_panel_v1.score defines it, and the late subset of tools_presub_stats_v1.late
    (recorded).  ``panel``/``audit``: the output folders of the study's CINELDI re-scoring on the corrected store, for
    corrected-store products."""
    B, L = T.B, T.L
    B.OUT = panel
    old_audit = B.AUDIT
    if audit is not None:
        B.AUDIT = audit
    real = np.savez_compressed
    np.savez_compressed = L._verify_instead_of_write
    try:
        d = B.load_set(face)
    finally:
        np.savez_compressed = real
        B.AUDIT = old_audit
    wb, wo = d["wb"], d["wo"]
    fo = panel / face
    full, partial, incomplete, refused = dict(d["base"]), {}, {}, []
    for f in sorted(fo.glob("pred_*.npz")):
        name = f.stem[5:]
        if name.endswith("_SMOKE"):
            continue
        if name.startswith(KAN_NOT_BASELINES):                       # LSTM-T-KAN-M is read from its own folder in step B only
            refused.append(name)
            continue
        a = np.load(f, allow_pickle=False)
        key = {(int(b), int(o)): i for i, (b, o) in enumerate(zip(a["building"], a["origin"]))}
        idx = [key.get((int(b), int(o))) for b, o in zip(wb, wo)]
        have = np.array([i is not None for i in idx])
        if "subset" in a.files:
            partial[name] = (have, a, key)
            continue
        if not have.all():
            incomplete[name] = (have, a, key)
            continue
        full[name] = a["pred"][idx].astype(np.float64)
    if not partial:
        raise SystemExit(f"{face}: no trained-baseline forecasts, no late subset")
    m = np.logical_and.reduce([v[0] for v in partial.values()])
    m_st = np.logical_and.reduce([v[0] for v in list(partial.values()) + list(incomplete.values())])
    il = np.flatnonzero(m)
    late = {k: v[il] for k, v in full.items()}
    for k, (_, a, key) in partial.items():
        late[k] = a["pred"][[key[(int(wb[i]), int(wo[i]))] for i in il]].astype(np.float64)
    ct = np.stack([B.types_at(d, b, o - CONTEXT + 12 + 24 * np.arange(56)) for b, o in zip(wb, wo)])
    tt = np.stack([B.types_at(d, b, o + 12 + 24 * np.arange(DAYS)) for b, o in zip(wb, wo)])
    first = np.array([int(np.argmax(np.isfinite(d["load"][b]))) for b in range(len(d["load"]))])
    tyh = np.stack([B.types_at(d, int(wb[i]), int(wo[i]) + np.arange(HORIZON)) for i in il]).astype(np.int8)   # hourly day types, test windows
    out = {"face": face, "name": T.names[face], "tier": T.tiers[face], "late": late, "il": il, "wbL": wb[il], "woL": wo[il],
           "yL": d["y"][il], "uL": d["unit"][il], "mL": d["month"][il], "ctxL": d["ctx"][il], "ctL": ct[il], "ttL": tt[il], "tyL": tyh,
           "histL": (wo - first[wb])[il].astype(int), "start_iso": d["start_iso"], "groups": list(d["groups"]),
           "late_equals_rank_tool_subset": bool(np.array_equal(m, m_st)), "incomplete_files": sorted(incomplete),
           "refused_files": refused, "windows_full": int(len(wb)), "units_full": int(len(np.unique(d["unit"])))}
    if full_peak:
        out["full_peak"] = {"y": d["y"], "ctx": d["ctx"], "unit": d["unit"], "month": d["month"], "ct": ct, "tt": tt,
                            "A": full["F1-off"], "F0": full["F0"], "TimesFM": full["TimesFM"]}
    del d, full
    return out


def kan_baselines(face, P, entry):
    """LSTM-T-KAN-M forecasts on the late windows (original origin numbers), aligned on (building, origin); on the
    corrected CINELDI store the corrected-store retrain (kan_dir)."""
    out, info = {}, {}
    folder, fixed = kan_dir(face, entry)
    for name in KAN:
        f = folder / f"pred_{name}.npz"
        a = np.load(f, allow_pickle=False)
        key = {(int(b), int(o)): i for i, (b, o) in enumerate(zip(a["building"], a["origin"]))}
        idx = [key.get((int(b), int(o))) for b, o in zip(P["wbL"], P["woL"])]
        if any(i is None for i in idx):
            raise SystemExit(f"{face}: {name} has no forecast on {sum(i is None for i in idx)} test windows")
        out[name] = a["pred"][idx].astype(np.float64)
        if not np.isfinite(out[name]).all():
            raise SystemExit(f"{face}: {name} has non-finite forecasts")
        info[name] = {"file": show(f), "windows": len(idx), "extra_windows_in_file": int(len(key) - len(set(idx))),
                      "store": ("corrected CINELDI store (retrained: same code, protocol, learning-rate grid and seeds)" if fixed else
                                "original CINELDI store" if on_corrected_cineldi(face, entry) else "the population's store")}
        if fixed:
            k = kan_fixed_record()
            info[name]["retrain"] = {"gate_original_store_bitwise": k["gate"]["bitwise"], "gate_pass": k["gate"]["pass"],
                                     "checks_pass": k["record"]["checks"]["pass"], "cutoff_label": k["record"]["checks"]["cutoff_label_new"],
                                     "lr": k["record"]["training"]["lr"], "best_epochs": k["record"]["training"]["best_epochs"],
                                     "vs_original_store_forecast": k["record"]["vs_original_store_forecast"]}
    return out, info


def corrected_cineldi_late(T):
    """CINELDI test data on the corrected store, as tools_smallnet_sr_formal_score_v1.late_cineldi: targets, units and
    months through the panel loader pointed at the corrected store, state flags = the study's state flags at the same
    windows, seasonal naive from the corrected store.  Origins are the corrected-store origins (original + shift)."""
    import tools_lstm_tkan_baseline_v1 as V1
    from ankyra import readouts as RO
    B, V = T.B, T.pwf().V
    store, c_new = corrected_store_info()
    pad = c_new - int(V.cuts("cineldi")["C"])
    old_a, old_f = B.AUDIT, B.AUDIT_FACES["cineldi"]
    B.AUDIT, B.AUDIT_FACES["cineldi"] = FIX_AUDIT_NEW, str(store)
    try:
        with V1.no_panel_writes():
            d = B.load_set("cineldi")
    finally:
        B.AUDIT, B.AUDIT_FACES["cineldi"] = old_a, old_f
    wb, wo = V.windows_late("cineldi")
    z = np.load(V.OUT / "cineldi" / "arms.npz", allow_pickle=False)
    if not (np.array_equal(z["building"], wb) and np.array_equal(z["origin"], wo)):
        raise SystemExit("cineldi: state-flag windows differ from the test windows")
    wo_new = wo + pad
    key = {(int(b), int(o)): i for i, (b, o) in enumerate(zip(d["wb"], d["wo"]))}
    ix = np.array([key[(int(b), int(o))] for b, o in zip(wb, wo_new)])
    state = z["off_S"] | z["micro_S"]
    SN = np.zeros((len(wb), HORIZON))
    snp = np.full(len(wb), np.nan)
    ty = d["types"]
    for i in np.flatnonzero(state):
        b_, o_ = int(wb[i]), int(wo_new[i])
        SN[i] = np.nan_to_num(d["load"][b_, o_ - 168:o_], nan=0.0)[np.arange(HORIZON) % 168]
        ty_ = ty if ty.ndim == 1 else ty[b_]
        cx = np.nan_to_num(d["load"][b_, o_ - CONTEXT:o_], nan=0.0)
        snp[i] = float(RO.peak_readout(SN[i], cx[None], ty_[o_ - CONTEXT:o_:HR].astype(np.int64)[None], ty_[o_:o_ + HORIZON:HR].astype(np.int64)[None])[0])
    return {"y": d["y"][ix], "unit": d["unit"][ix], "month": d["month"][ix], "state": state, "SN": SN, "snp": snp,
            "wb": wb, "wo": wo_new, "pad": pad, "store": store}


def product(T, face, P, entry):
    """ankyra-lite on the panel's late windows (panel order), from the product named in products.json, with the
    weekly seasonal naive on off-state / micro-load windows (tools_smallnet_pwformal_v1._with_state, re-implemented
    to keep the anchoring layer's own output)."""
    PWF = T.pwf()
    if on_corrected_cineldi(face, entry):
        D = corrected_cineldi_late(T)
        pad = D["pad"]
    else:
        PWF.NINE = (face,)
        D = PWF._late_data()[face]
        pad = 0
    zx = np.load(entry["npz_path"], allow_pickle=False)
    if not (np.array_equal(zx["building"], D["wb"]) and np.array_equal(zx["origin"], D["wo"])):
        raise SystemExit(f"{face}: product windows differ from the study's test windows")
    st = D["state"]
    if not np.array_equal(zx["off"] | zx["micro"], st):
        raise SystemExit(f"{face}: product state flags differ from the study's state flags")
    kp = {(int(b), int(o) - pad): i for i, (b, o) in enumerate(zip(D["wb"], D["wo"]))}
    mine = [(int(b), int(o)) for b, o in zip(P["wbL"], P["woL"])]
    if sorted(kp) != sorted(mine):
        raise SystemExit(f"{face}: product windows differ from the panel's late windows")
    ix = np.array([kp[k] for k in mine])
    if not np.array_equal(D["y"][ix], P["yL"]):
        raise SystemExit(f"{face}: product targets differ from the panel's targets")
    raw = zx["traj"][ix].astype(np.float64)
    state = st[ix]
    SN, snp = D["SN"][ix], D["snp"][ix]
    traj = np.where(state[:, None], SN, raw)
    level = np.where(state, SN.mean(1), zx["level"][ix].astype(np.float64))
    peak = np.where(state, snp, zx["peak"][ix].astype(np.float64))
    month_same = bool(np.array_equal(D["month"][ix], P["mL"]))
    pk = T.PK.peak_readout_scalar(traj, P["ctxL"], P["ctL"], P["ttL"], 1.0)        # release readout on the stored trajectory
    rel = np.abs(pk - peak) / np.maximum(np.abs(peak), 1e-9)
    info = {"file": show(entry["npz_path"]), "store": entry["store"], "origin_shift_h": int(pad), "windows": int(len(ix)),
            "state_windows_seasonal_naive": int(state.sum()), "months_equal_panel_months": month_same,
            "peak_readout_recomputed_vs_stored_max_rel": float(rel.max()),
            "level_vs_trajectory_mean_max_rel": float((np.abs(level - traj.mean(1)) / np.maximum(np.abs(traj.mean(1)), 1e-9)).max()),
            "level_vs_trajectory_mean_windows_over_1e-6": int((np.abs(level - traj.mean(1)) > 1e-6 * np.maximum(1.0, np.abs(traj.mean(1)))).sum())}
    out = {"traj": traj, "level": level, "peak": peak, "raw": raw, "raw_level": zx["level"][ix].astype(np.float64),
           "raw_peak": zx["peak"][ix].astype(np.float64), "state": state, "SN": SN, "snp": snp, "pad": pad,
           "origins_product": D["wo"][ix], "month": D["month"][ix], "info": info,
           "C0": zx["C0"][ix].astype(np.float64) if "C0" in zx.files else None,
           "S3s": {k: np.asarray(v)[ix].astype(np.float64) for k, v in D["S3s"].items()} if "S3s" in D else None}
    del D, zx
    return out


def load_carriers(entry, origins_product, wb):
    """Per-seed carrier forecasts at the origin (k = 0) on the product's windows and the three-seed ensemble (float64
    sum in seed order, as the study's anchoring did)."""
    seeds, ens = [], None
    for rd in entry["run_paths"]:
        q = np.load(rd / "carrier.npz", allow_pickle=False)
        key = {(int(b), int(o)): i for i, (b, o) in enumerate(zip(q["building"], q["origin"]))}
        ix = np.array([key[(int(b), int(o))] for b, o in zip(wb, origins_product)])
        if not q["have"][ix, 0].all():
            raise SystemExit(f"{rd}: a carrier forecast at the origin is missing")
        p0 = q["pred"][ix, 0]
        seeds.append(p0.astype(np.float64))
        ens = p0.astype(np.float64) if ens is None else ens + p0
        del q
    return seeds, ens / len(seeds)


# ----------------------------------------------------------------------------- tables (one population)
def pairwise_and_ranks(T, P, preds, ref, ref_name, ref_label):
    B = T.B
    B.FLAGSHIP, B.FAMILY = ref, ()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = B.block(preds, P["yL"], P["uL"], P["mL"])
    B.FLAGSHIP = REF_A
    bench, ranks, conv = [], [], []
    for m, v in r["vs"].items():
        bench.append({"set": P["name"], "tier": P["tier"], "subset": "late", "windows": r["windows"], "units": r["units"], "model": m,
                      "model_label": T.label(m, ref, ref_label), "model_class": T.mclass(m, ref),
                      "log_ratio": round(v["log_ratio"], 6), "um_low": round(v["UM"][0], 6), "um_high": round(v["UM"][1], 6),
                      "improvement_pct": round(v["pct"], 3), "pooled_mse_ratio": round(v["pooled_ratio"], 6)})
    mr = r["mean_unit_rank_external"]
    for m, v in mr.items():
        ranks.append({"set": P["name"], "tier": P["tier"], "subset": "late", "model": ref_name if m == ref else m,
                      "model_label": ref_label if m == ref else T.label(m, ref, ref_label), "mean_unit_rank": round(v, 4), "n_models": len(mr)})
    for m, c in r["conventional"].items():
        conv.append(conv_row(T, P, r, m, c, ref, ref_name, ref_label))
    return r, bench, ranks, conv


def conv_row(T, P, r, m, c, ref, ref_name, ref_label):
    return {"set": P["name"], "tier": P["tier"], "windows": r["windows"], "units": r["units"], "model": ref_name if m == ref else m,
            "model_label": ref_label if m == ref else T.label(m, ref, ref_label), "model_class": T.mclass(m, ref),
            "RMSE_kW": round(c["RMSE_kW_unit_mean"], 4), "MAE_kW": round(c["MAE_kW_unit_mean"], 4),
            "CV_RMSE_pct": round(c["CV_RMSE_pct_unit_median"], 2), "NMBE_pct": round(c["NMBE_pct_unit_median"], 2),
            "WAPE_pct": round(c["WAPE_pct_unit_median"], 2), "WAPE_pooled_pct": round(c["WAPE_pct_pooled"], 2),
            "units_excluded": c.get("units_excluded_from_ratios", 0)}


def energy_rows(T, P, preds, ref, ref_name, ref_label):
    """tools_hcr_energy_error_v1.rows_for (one random stream per population and unit set, models in sorted order)."""
    B, E = T.B, T.E
    B.FLAGSHIP = ref
    out = []
    sets = [("all units", np.ones(len(P["yL"]), bool))]
    if P["face"] == "bdg2":
        sets.append(("without the three near-zero meters", ~np.isin(P["uL"], T.near_zero)))
    try:
        for subset, keep in sets:
            rows = E.rows_for(P["name"], P["tier"], subset, {k: v[keep] for k, v in preds.items()}, P["yL"][keep], P["uL"][keep], P["mL"][keep])
            for r_ in rows:
                r_ = {("unit_set" if k == "subset" else k): v for k, v in r_.items()}
                if r_["model"] == "ANKYRA":                              # rows_for names the reference "ANKYRA"
                    r_["model"], r_["model_label"] = ref_name, ref_label
                else:
                    r_["model_label"], r_["model_class"] = T.label(r_["model"], ref, ref_label), T.mclass(r_["model"], ref)
                out.append(r_)
    finally:
        B.FLAGSHIP = REF_A
    return out


def energy_level_readout(T, P, preds, level, ref):
    """Sensitivity: the energy contrasts of energy_rows with the product's independent level readout (744 x level) in
    place of its delivered trajectory's mean (same estimator, same random stream and model order as rows_for)."""
    y, unit, month = P["yL"], P["uL"], P["mL"]
    err = y.mean(1) - level
    a = err ** 2
    rng = np.random.default_rng(T.DS.SEED + 11)
    out = {"median_energy_ape_pct": T.E.energy_ape(err, y, unit), "vs": {}}
    for m in sorted(k for k in preds if k != ref):
        c = contrast_um(T, a, (y - preds[m]).mean(1) ** 2, unit, month, rng)
        out["vs"][m] = {"log_ratio": c["log_ratio"], "UM": c["UM"], "pct": 100 * (1 - math.exp(c["log_ratio"]))}
    return out


def lead_day_rows(T, P, preds, ref, ref_name, ref_label):
    """tools_hcr_lead_day_metrics_v1.main for one population (same unit set U over all forecasters, release rounding)."""
    L = T.L
    y, unit = P["yL"], P["uL"]
    days = {m: L.by_day(p, y, unit) for m, p in preds.items()}
    ybar = days[ref][2]
    ok = np.abs(ybar) >= 1e-6
    U = ok & np.logical_and.reduce([(v[0] > 0).all(1) for v in days.values()])
    rows = []
    for m, (rmse, mae, _) in days.items():
        gm = np.exp(np.log(100 * rmse[U] / ybar[U, None]).mean(0))
        for dd in range(DAYS):
            rows.append({"set": P["name"], "tier": P["tier"], "windows": len(y), "units": int(len(np.unique(unit))),
                         "model": ref_name if m == ref else m, "model_label": ref_label if m == ref else T.label(m, ref, ref_label),
                         "model_class": T.mclass(m, ref), "day": dd + 1,
                         "RMSE_kW": round(float(rmse[:, dd].mean()), 4), "MAE_kW": round(float(mae[:, dd].mean()), 4),
                         "CV_RMSE_pct": round(float(np.median(100 * rmse[ok, dd] / ybar[ok])), 2),
                         "nMAE_pct": round(float(np.median(100 * mae[ok, dd] / ybar[ok])), 2),
                         "GM_CV_RMSE_pct": round(float(gm[dd]), 2), "n_units_gm": int(U.sum())})
    return rows


def lead_day_energy_rows(T, P, preds, ref, ref_name, ref_label):
    """tools_lead_day_energy_v6_v1.main for one population."""
    y, unit = P["yL"], P["uL"]
    stats = {m: T.LE.per_unit(p, y, unit) for m, p in preds.items()}
    ybar = stats[ref][2]
    U = (np.abs(ybar) >= 1e-6) & np.logical_and.reduce([(v[0] > 0).all(1) & (v[1] > 0).all(1) for v in stats.values()])
    rows = []
    for m, v in stats.items():
        gd = np.exp(np.log(100 * v[0][U] / ybar[U, None]).mean(0))
        gc_ = np.exp(np.log(100 * v[1][U] / ybar[U, None]).mean(0))
        for dd in range(DAYS):
            rows.append({"set": P["name"], "tier": P["tier"], "windows": len(y), "units_in_U": int(U.sum()),
                         "model": ref_name if m == ref else m, "model_label": ref_label if m == ref else T.label(m, ref, ref_label),
                         "model_class": T.mclass(m, ref), "day": dd + 1,
                         "GM_CV_daily_energy_pct": round(float(gd[dd]), 4), "GM_CV_cumulative_energy_pct": round(float(gc_[dd]), 4)})
    return rows


def block_share_rows(T, P, preds, models, ref_label):
    """tools_hcr_block_shares_v1.main and tools_ankyra_v22_release_v1.block_shares for one population.
    models: {label: key}, the reference label first."""
    S = T.S
    y, unit = P["yL"], P["uL"]
    uu, inv = np.unique(unit, return_inverse=True)
    per = {}
    for label, key in models.items():
        e = y - preds[key]
        b = S.block_mse(e)
        hourly = (e ** 2).mean(1)
        assert np.allclose(b["level"] + b["daily path"] + b["within-day"], hourly, rtol=1e-9, atol=1e-12)
        ub = {k: np.bincount(inv, v, len(uu)) / np.bincount(inv, minlength=len(uu)) for k, v in b.items()}
        tot = sum(ub.values())
        pos = tot > 0
        per[label] = {"unit_blocks": ub, "unit_total": tot, "pooled": {k: float(v.sum() / hourly.sum()) for k, v in b.items()},
                      "median_unit": {k: float(np.median(ub[k][pos] / tot[pos])) for k in b}}
    rows = []
    base = {"set": P["name"], "windows": int(len(y)), "units": int(len(uu))}
    for label, p in per.items():
        for kind in ("pooled", "median_unit"):
            for k in S.BLOCKS:
                rows.append(base | {"model": label, "quantity": f"{kind}_share", "block": k, "value": round(p[kind][k], 4)})
    for other in ("TimesFM", "Chronos-2-X"):
        for k in S.BLOCKS + ("hourly",):
            a = per[ref_label]["unit_total"] if k == "hourly" else per[ref_label]["unit_blocks"][k]
            o = per[other]["unit_total"] if k == "hourly" else per[other]["unit_blocks"][k]
            ok = (a > 0) & (o > 0)
            r = float(np.mean(0.5 * np.log(a[ok] / o[ok])))
            for q, val in (("improvement_pct", round(100 * (1 - math.exp(r)), 3)), ("log_ratio", round(r, 6)), ("units_in_ratio", int(ok.sum()))):
                rows.append(base | {"model": f"{ref_label} vs {other}", "quantity": q, "block": k, "value": val})
    return rows


def rank_scaled(T, P, preds, ref):
    """tools_presub_stats_v1 for one population: per-unit RMSE, Friedman / Nemenyi / Holm-Wilcoxon (the reference is
    keyed "ANKYRA" inside rank_block) and RMSSE / MASE."""
    ST = T.ST
    R_in = {("ANKYRA" if k == ref else k): v for k, v in preds.items()}
    uu, R = ST.unit_rmse(R_in, P["yL"], P["uL"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rb = ST.rank_block(R)
        sc = ST.scaled(R_in, P["yL"], P["ctxL"], P["uL"])
    return R, rb, sc


def fmt_p(p, release=False):
    """p-values as text; the release format writes 0 where the value is below the smallest double (the release's own
    convention), ankyra-lite's files write "<1e-300"."""
    if not release and p == 0:
        return "<1e-300"
    return f"{p:.3g}"


def rank_rows(T, name, tier, rb, ref_name, ref_label, release=False):
    rows = []
    for m in sorted(rb["mean_rank"], key=lambda m: rb["mean_rank"][m]):
        rows.append({"set": name, "tier": tier, "units": rb["units"], "model": ref_name if m == "ANKYRA" else m,
                     "model_label": ref_label if m == "ANKYRA" else T.label(m, None, ref_label), "mean_rank": round(rb["mean_rank"][m], 3),
                     "friedman_p": fmt_p(rb["friedman_p"], release), "nemenyi_cd": round(rb["nemenyi_cd"], 3),
                     "wilcoxon_holm_p_vs_ankyra": "" if m == "ANKYRA" else fmt_p(rb["wilcoxon_holm_p"][m], release)})
    return rows


def scaled_rows(T, P, rb, sc, ref_name, ref_label):
    rows = []
    for m in sorted(rb["mean_rank"], key=lambda m: rb["mean_rank"][m]):
        rows.append({"set": P["name"], "tier": P["tier"], "model": ref_name if m == "ANKYRA" else m,
                     "model_label": ref_label if m == "ANKYRA" else T.label(m, None, ref_label),
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc[m].items()}})
    return rows


def peak_rows_release(T, P):
    """peak_readout.csv of the release on ALL windows of one population, recomputed (tools_hcr_f1_readouts_v1.peak:
    one random stream in the tool's contrast order; tools_hcr_ankyra_readout_check_v1.peak: the trajectory maximum)."""
    PK, F = T.PK, P["full_peak"]
    y, ctx, unit, month, ct, tt = F["y"], F["ctx"], F["unit"], F["month"], F["ct"], F["tt"]
    ypk, prev = y.max(1), ctx[:, -HORIZON:].max(1)
    R = {k: PK.peak_readout_scalar(F[k], ctx, ct, tt, 1.0) for k in ("A", "F0", "TimesFM")}
    E = {f"readout_{k}": np.abs(v - ypk) for k, v in R.items()}
    E["prev_window_peak"] = np.abs(prev - ypk)
    E["A_max"] = np.abs(F["A"].max(1) - ypk)
    rng = np.random.default_rng(T.DS.SEED + 13)
    got = {}
    with np.errstate(all="ignore"):
        for lab, a_, b_ in (("same readout on fixed division (F0)", "readout_A", "readout_F0"), ("previous-window peak", "readout_A", "prev_window_peak"),
                            ("same readout on TimesFM", "readout_A", "readout_TimesFM")):
            got[lab] = contrast_um(T, E[a_] ** 2, E[b_] ** 2, unit, month, rng)
        got["trajectory maximum"] = contrast_um(T, E["readout_A"] ** 2, E["A_max"] ** 2, unit, month, np.random.default_rng(T.DS.SEED + 13))
    ape_a, ape_p = ape_mum(E["readout_A"], ypk, unit), ape_mum(E["prev_window_peak"], ypk, unit)
    rows = []
    for lab in ("trajectory maximum", "same readout on TimesFM", "previous-window peak", "same readout on fixed division (F0)"):
        c = got[lab]
        if not np.isfinite(c["log_ratio"]):
            continue
        rows.append({"set": P["name"], "tier": P["tier"], "comparator": lab, "log_ratio": round(c["log_ratio"], 6), "um_low": round(c["UM"][0], 6),
                     "um_high": round(c["UM"][1], 6), "improvement_pct": round(100 * (1 - math.exp(c["log_ratio"])), 3),
                     "median_peak_ape_ankyra_pct": round(ape_a, 2), "median_peak_ape_previous_peak_pct": round(ape_p, 2)})
    return rows


def peak_rows_product(T, P, preds, peak_ref, ref):
    """ankyra-lite's peak readout against its own trajectory maximum, the previous window's peak and, for every
    baseline, the same readout on the baseline's trajectory (the release rule) and the baseline's trajectory maximum.
    Late windows; one fresh random stream (seed + 13) per comparator."""
    PK = T.PK
    y, ctx, unit, month, ct, tt = P["yL"], P["ctxL"], P["uL"], P["mL"], P["ctL"], P["ttL"]
    ypk, prev = y.max(1), ctx[:, -HORIZON:].max(1)
    e_ref = np.abs(peak_ref - ypk)
    comps = {"trajectory maximum": np.abs(preds[ref].max(1) - ypk), "previous-window peak": np.abs(prev - ypk)}
    ape = {"reference": ape_mum(e_ref, ypk, unit), "previous-window peak": ape_mum(comps["previous-window peak"], ypk, unit),
           "reference trajectory maximum": ape_mum(comps["trajectory maximum"], ypk, unit)}
    for m, p in preds.items():
        if m == ref:
            continue
        comps[f"same readout on {m}"] = np.abs(PK.peak_readout_scalar(p, ctx, ct, tt, 1.0) - ypk)
        comps[f"trajectory maximum of {m}"] = np.abs(p.max(1) - ypk)
        ape[f"same readout on {m}"] = ape_mum(comps[f"same readout on {m}"], ypk, unit)
        ape[f"trajectory maximum of {m}"] = ape_mum(comps[f"trajectory maximum of {m}"], ypk, unit)
    rows, vs = [], {}
    with np.errstate(all="ignore"):
        for lab, e in comps.items():
            c = contrast_um(T, e_ref ** 2, e ** 2, unit, month, np.random.default_rng(T.DS.SEED + 13))
            vs[lab] = c
            if not np.isfinite(c["log_ratio"]):
                continue
            rows.append({"set": P["name"], "tier": P["tier"], "comparator": lab, "log_ratio": round(c["log_ratio"], 6),
                         "um_low": round(c["UM"][0], 6), "um_high": round(c["UM"][1], 6), "improvement_pct": round(100 * (1 - math.exp(c["log_ratio"])), 3),
                         "median_peak_ape_ankyra_pct": round(ape["reference"], 2), "median_peak_ape_previous_peak_pct": round(ape["previous-window peak"], 2)})
    return rows, vs, ape


# ----------------------------------------------------------------------------- comparison with the published files
_PUB = {}


def published_text(name):
    """A results/ file of the first release repository as committed (``git show <rev>:results/<name>``, read-only, no
    optional locks), so that uncommitted edits of its working copy do not enter the check; the working-copy file when
    git or the revision is not available."""
    if name not in _PUB:
        txt, src = None, None
        try:
            r = subprocess.run(["git", "-C", str(REL), "--no-optional-locks", "show", f"{ORIGINAL_REV}:results/{name}"],
                               capture_output=True, timeout=120)
            if r.returncode == 0:
                txt, src = r.stdout.decode("utf-8").replace("\r\n", "\n"), f"git:{ORIGINAL_REV}"
        except (OSError, subprocess.SubprocessError):
            pass
        if txt is None:
            txt, src = (REL_RES / name).read_text(encoding="utf-8"), "working copy"
        _PUB[name] = (txt, src)
    return _PUB[name][0]


def published_source():
    """Which state of the first release repository the check read (revision and commit, or the working copy)."""
    srcs = sorted({v[1] for v in _PUB.values()})
    commit = None
    try:
        r = subprocess.run(["git", "-C", str(REL), "--no-optional-locks", "rev-parse", ORIGINAL_REV], capture_output=True, timeout=60)
        commit = r.stdout.decode().strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"sources": srcs, "revision": ORIGINAL_REV, "commit": commit}


DEC = {
    "benchmark_pairwise": {"log_ratio": 6, "um_low": 6, "um_high": 6, "improvement_pct": 3, "pooled_mse_ratio": 6},
    "benchmark_mean_unit_rank": {"mean_unit_rank": 4},
    "conventional_metrics_late": {"RMSE_kW": 4, "MAE_kW": 4, "CV_RMSE_pct": 2, "NMBE_pct": 2, "WAPE_pct": 2, "WAPE_pooled_pct": 2},
    "energy_error": {"log_ratio": 6, "um_low": 6, "um_high": 6, "improvement_pct": 3, "median_energy_ape_pct": 3},
    "lead_day_metrics": {"RMSE_kW": 4, "MAE_kW": 4, "CV_RMSE_pct": 2, "nMAE_pct": 2, "GM_CV_RMSE_pct": 2},
    "lead_day_energy": {"GM_CV_daily_energy_pct": 4, "GM_CV_cumulative_energy_pct": 4},
    "block_shares": {"value": None},
    "rank_tests": {"mean_rank": 3, "nemenyi_cd": 3, "friedman_p": "g", "wilcoxon_holm_p_vs_ankyra": "g"},
    "scaled_errors": {"RMSSE_median_unit": 4, "RMSSE_geo_mean": 4, "MASE_median_unit": 4, "MASE_geo_mean": 4, "units_below_1_share": 4},
    "peak_readout": {"log_ratio": 6, "um_low": 6, "um_high": 6, "improvement_pct": 3, "median_peak_ape_ankyra_pct": 2, "median_peak_ape_previous_peak_pct": 2},
}
KEYS = {"benchmark_pairwise": ("set", "subset", "model"), "benchmark_mean_unit_rank": ("set", "subset", "model"),
        "conventional_metrics_late": ("set", "model"), "energy_error": ("set", "unit_set", "model"), "lead_day_metrics": ("set", "model", "day"),
        "lead_day_energy": ("set", "model", "day"), "block_shares": ("set", "model", "quantity", "block"), "rank_tests": ("set", "model"),
        "scaled_errors": ("set", "model"), "peak_readout": ("set", "comparator")}
INTS = {"windows", "units", "n_models", "units_excluded", "day", "n_units_gm", "units_in_U"}


def compare(table, mine, scope):
    """Published rows in scope against the recomputed rows: numbers to the published rounding (a difference of at most
    one unit in the last published decimal counts as rounding), integers and strings exactly."""
    pub = [r for r in csv.DictReader(io.StringIO(published_text(f"{table}.csv"))) if scope(r)]
    key = KEYS[table]
    idx = {tuple(str(r[k]) for k in key): r for r in mine}
    res = {"published_rows_in_scope": len(pub), "recomputed_rows": len(mine), "fields_compared": 0, "exact": 0,
           "within_last_digit": 0, "max_abs_diff": 0.0, "mismatches": [], "missing": [], "extra": []}
    seen = set()
    for r in pub:
        k = tuple(str(r[x]) for x in key)
        q = idx.get(k)
        if q is None:
            res["missing"].append(k)
            continue
        seen.add(k)
        for f, pv in r.items():
            if f in key:
                continue
            mv = q.get(f)
            res["fields_compared"] += 1
            dec = DEC[table].get(f, "absent")
            if table == "block_shares" and f == "value":
                dec = {"improvement_pct": 3, "log_ratio": 6, "units_in_ratio": 0}.get(r["quantity"], 4)
            if f in INTS or dec == 0:
                ok_exact = int(float(pv)) == int(float(mv))
                ok = ok_exact
                diff = abs(float(pv) - float(mv))
            elif dec == "g":
                if pv == "" or mv == "":
                    ok_exact = ok = (pv == mv)
                    diff = 0.0 if ok else float("inf")
                else:
                    a, b = float(pv), float(mv)
                    ok_exact = pv == str(mv)
                    ok = ok_exact or abs(a - b) <= 1e-2 * max(abs(a), abs(b))
                    diff = abs(a - b)
            elif dec == "absent":
                ok_exact = ok = str(pv) == str(mv)
                diff = 0.0 if ok else float("inf")
            else:
                a, b = float(pv), float(mv)
                diff = abs(a - b)
                ok_exact = diff == 0.0
                ok = diff <= 10.0 ** (-dec) * (1 + 1e-9) + 1e-12
            if math.isfinite(diff) and f not in INTS and dec not in ("absent",):
                res["max_abs_diff"] = max(res["max_abs_diff"], diff if dec != "g" else 0.0)
            if ok_exact:
                res["exact"] += 1
            elif ok:
                res["within_last_digit"] += 1
            else:
                res["mismatches"].append({"key": k, "field": f, "published": pv, "recomputed": mv})
    res["extra"] = [k for k in idx if k not in seen]
    if res["published_rows_in_scope"] == 0:
        res["status"] = "NOT CHECKED (no published row for these populations)"
    else:
        res["status"] = "REPRODUCED" if not res["mismatches"] and not res["missing"] else "MISMATCH"
    res["mismatches"] = res["mismatches"][:25]
    res["missing"] = [list(k) for k in res["missing"][:25]]
    res["extra"] = [list(k) for k in res["extra"][:25]]
    return res


# ----------------------------------------------------------------------------- step A: scoring-code check
def step_a(T, faces):
    """Recompute the release rows with the release's reference forecaster and compare them with the published files.
    Only the match counts are kept; the recomputed numbers are not written anywhere."""
    mine = {t: [] for t in TABLES}
    pooled_R, notes = {}, {}
    for face in faces:
        if face not in PANEL10:
            continue
        wait_for_memory(f"check {face}")
        t0 = time.time()
        P = load_panel(T, face, full_peak=True)
        late = P["late"]
        p21 = {k: v for k, v in late.items() if k not in A_FAMILY[1:]}           # the release reference and the 20 baselines
        assert REF_A in p21 and len(p21) == 21, (face, sorted(p21))
        r, bench, ranks, conv = pairwise_and_ranks(T, P, p21, REF_A, "ANKYRA", "ANKYRA")
        for m in A_VERSIONS:                                                     # extra rows of conventional_metrics_late
            if m in late:
                conv.append(conv_row(T, P, r, m, T.B.conventional(late[m], P["yL"], P["uL"]), REF_A, "ANKYRA", "ANKYRA"))
        mine["benchmark_pairwise"] += bench
        mine["benchmark_mean_unit_rank"] += ranks
        mine["conventional_metrics_late"] += conv
        if face in LEAD7:
            mine["energy_error"] += energy_rows(T, P, p21, REF_A, "ANKYRA", "ANKYRA")
            mine["lead_day_metrics"] += lead_day_rows(T, P, p21, REF_A, "ANKYRA", "ANKYRA")
            mine["lead_day_energy"] += lead_day_energy_rows(T, P, p21, REF_A, "ANKYRA", "ANKYRA")
            mine["block_shares"] += block_share_rows(T, P, late, {"ANKYRA": REF_A, "ANKYRA 1.x": "ANKYRA-1x", "TimesFM": "TimesFM", "Chronos-2-X": "Chronos-2-X"}, "ANKYRA")
        R, rb, sc = rank_scaled(T, P, p21, REF_A)
        if P["tier"] == "test":
            pooled_R[face] = R
        mine["rank_tests"] += rank_rows(T, P["name"], P["tier"], rb, "ANKYRA", "ANKYRA", release=True)
        mine["scaled_errors"] += scaled_rows(T, P, rb, sc, "ANKYRA", "ANKYRA")
        mine["peak_readout"] += peak_rows_release(T, P)
        notes[face] = {"windows_full": P["windows_full"], "windows_late": int(len(P["yL"])), "units_late": int(len(np.unique(P["uL"]))),
                       "late_equals_rank_tool_subset": P["late_equals_rank_tool_subset"], "incomplete_files": P["incomplete_files"],
                       "files_left_to_step_b": P["refused_files"], "seconds": round(time.time() - t0, 1)}
        log(f"check {face}: {notes[face]}")
        del P, late, p21, r
        gc.collect()
    if all(f in pooled_R for f in TEST6):
        mine["rank_tests"] += pooled_rank_rows(T, pooled_R, "ANKYRA", "ANKYRA", release=True)[0]
    names = {T.names[f] for f in faces if f in PANEL10}
    pooled_ok = "six test populations pooled" if any(r["set"] == "six test populations pooled" for r in mine["rank_tests"]) else None
    checks = {}
    for t in TABLES:
        if t in ("benchmark_pairwise", "benchmark_mean_unit_rank"):
            scope = lambda r, n=names: r["set"] in n and r["subset"] == "late"           # noqa: E731
        elif t == "rank_tests":
            scope = lambda r, n=names | ({pooled_ok} if pooled_ok else set()): r["set"] in n   # noqa: E731
        else:
            scope = lambda r, n=names: r["set"] in n                                          # noqa: E731
        checks[t] = compare(t, mine[t], scope)
        c = checks[t]
        log(f"check {t}: {c['status']} rows {c['published_rows_in_scope']} fields {c['fields_compared']} exact {c['exact']} "
            f"last-digit {c['within_last_digit']} mismatches {len(c['mismatches'])} missing {len(c['missing'])} max|diff| {c['max_abs_diff']:.3g}")
    return checks, notes


def pooled_rank_rows(T, pooled_R, ref_name, ref_label, release=False):
    """tools_presub_stats_v1.main, pooled block: per-unit RMSE of the six test populations concatenated (forecasters
    present on all six).  Returns the rows and the test record."""
    common = set.intersection(*[set(pooled_R[f]) for f in TEST6])
    P = {m: np.concatenate([pooled_R[f][m] for f in TEST6]) for m in sorted(common)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rb = T.ST.rank_block(P)
    return rank_rows(T, "six test populations pooled", "test", rb, ref_name, ref_label, release=release), rb


def _rename_ref(x, ref_name):
    """The reference forecaster is keyed "ANKYRA" inside the study's rank tool; rename it in the records."""
    if isinstance(x, dict):
        return {(ref_name if k == "ANKYRA" else k): _rename_ref(v, ref_name) for k, v in x.items()}
    if isinstance(x, list):
        return [_rename_ref(v, ref_name) for v in x]
    return ref_name if x == "ANKYRA" else x


# ----------------------------------------------------------------------------- step B
def step_b(T, faces, proceed, prods, extra=True):
    import extra_tables as XT
    out = {t: [] for t in TABLES}
    rec = {"populations": {}}
    pooled_R = {}
    xt = XT.Collector(T, prods) if extra else None
    xc = json.loads(HOURLY_XCHECK.read_text(encoding="utf-8"))["faces"] if HOURLY_XCHECK.exists() else {}
    for face in faces:
        if face not in NINE:
            continue
        entry = prods["faces"].get(face)
        if entry is None or not entry["npz_path"].exists():
            log(f"tables {face}: no ankyra-lite product; skipped")
            continue
        wait_for_memory(f"tables {face}")
        t0 = time.time()
        corrected = on_corrected_cineldi(face, entry)
        P = load_panel(T, face, panel=SB_PANEL if corrected else P22, audit=SB_AUDIT if corrected else None)
        baselines = {k: v for k, v in P["late"].items() if k not in A_FAMILY}
        assert len(baselines) == 20, (face, sorted(baselines))
        kan, kan_info = kan_baselines(face, P, entry)
        baselines |= kan
        assert len(baselines) == N_BASELINES, (face, sorted(baselines))
        P["late"] = None
        gc.collect()
        wait_for_memory(f"tables {face} product")
        prod = product(T, face, P, entry)
        if not prod["info"]["months_equal_panel_months"]:
            P["mL"] = prod["month"]                                  # corrected store: months of the corrected calendar
        preds = {REF_B: prod["traj"]} | baselines
        flag = face_flag(face, entry)
        fr = {"name": P["name"], "tier": P["tier"], "windows": int(len(P["yL"])), "units": int(len(np.unique(P["uL"]))), "flag": flag or None,
              "product": prod["info"], "baselines": sorted(baselines), "extra_baselines": kan_info,
              "panel": show(SB_PANEL if corrected else P22)}
        r, bench, ranks, conv = pairwise_and_ranks(T, P, preds, REF_B, REF_B, REF_B_LABEL)
        mr = r["mean_unit_rank_external"]
        fr["hourly"] = {"vs": {k: {"log_ratio": v["log_ratio"], "UM": v["UM"], "pct": v["pct"], "pooled_ratio": v["pooled_ratio"]} for k, v in r["vs"].items()},
                        "mean_unit_rank": mr[REF_B], "rank": 1 + sum(v < mr[REF_B] - 1e-9 for v in mr.values()), "of": len(mr),
                        "mean_unit_rank_all": mr}
        if face in xc and entry.get("crosscheck"):                         # the earlier hourly-only version (20 baselines, same product)
            x = xc[face]
            p21 = {k: v for k, v in preds.items() if k not in KAN}
            mr21, rk21, of21, _ = unit_rank(p21, P["yL"], P["uL"], REF_B)
            dmax = max([abs(x["vs"][k]["pct"] - fr["hourly"]["vs"][k]["pct"]) for k in x["vs"]] +
                       [abs(a - b) for k in x["vs"] for a, b in zip(x["vs"][k]["UM"], fr["hourly"]["vs"][k]["UM"])] +
                       [abs(x["mean_unit_rank"] - mr21)])
            fr["hourly_crosscheck"] = {"same_windows_units": x["windows"] == fr["windows"] and x["units"] == fr["units"],
                                       "same_rank_among_21": x["rank"] == rk21 and x["of"] == of21,
                                       "same_baselines": sorted(x["vs"]) == sorted(k for k in fr["hourly"]["vs"] if k not in KAN),
                                       "max_abs_diff": dmax, "mean_unit_rank_among_21": mr21, "rank_among_21": rk21}
            log(f"tables {face}: hourly cross-check {fr['hourly_crosscheck']}")
        if "benchmark_pairwise" in proceed:
            out["benchmark_pairwise"] += bench
        if "benchmark_mean_unit_rank" in proceed:
            out["benchmark_mean_unit_rank"] += ranks
        if "conventional_metrics_late" in proceed:
            out["conventional_metrics_late"] += conv
        if "energy_error" in proceed:                                       # all nine populations (release: six test + park)
            rows = energy_rows(T, P, preds, REF_B, REF_B, REF_B_LABEL)
            out["energy_error"] += rows
            fr["energy"] = {r_["unit_set"] + "|" + r_["model"]: {k: r_[k] for k in ("log_ratio", "um_low", "um_high", "improvement_pct", "median_energy_ape_pct")} for r_ in rows}
            fr["energy_level_readout"] = energy_level_readout(T, P, preds, prod["level"], REF_B)
        if face in LEAD7:
            if "lead_day_metrics" in proceed:
                rows = lead_day_rows(T, P, preds, REF_B, REF_B, REF_B_LABEL)
                out["lead_day_metrics"] += rows
                fr["lead_day_gm_cv"] = {f"{r_['model']}|{r_['day']}": r_["GM_CV_RMSE_pct"] for r_ in rows}
            if "lead_day_energy" in proceed:
                rows = lead_day_energy_rows(T, P, preds, REF_B, REF_B, REF_B_LABEL)
                out["lead_day_energy"] += rows
                fr["month_energy_gm_cv_day31"] = {r_["model"]: r_["GM_CV_cumulative_energy_pct"] for r_ in rows if r_["day"] == DAYS}
            if "block_shares" in proceed:
                rows = block_share_rows(T, P, preds, {REF_B_LABEL: REF_B, "TimesFM": "TimesFM", "Chronos-2-X": "Chronos-2-X"}, REF_B_LABEL)
                out["block_shares"] += rows
                fr["block_shares"] = {f"{r_['model']}|{r_['quantity']}|{r_['block']}": r_["value"] for r_ in rows}
        fr["conventional"] = {r_["model"]: {k: r_[k] for k in ("RMSE_kW", "MAE_kW", "CV_RMSE_pct", "WAPE_pct", "WAPE_pooled_pct")} for r_ in conv}
        R, rb, sc = rank_scaled(T, P, preds, REF_B)
        if P["tier"] == "test":
            pooled_R[face] = R
        if "rank_tests" in proceed:
            out["rank_tests"] += rank_rows(T, P["name"], P["tier"], rb, REF_B, REF_B_LABEL)
            fr["rank_tests"] = {"friedman_p": rb["friedman_p"], "nemenyi_cd": rb["nemenyi_cd"], "position": rb["ankyra_position"],
                                "significantly_better_than": rb["ankyra_significantly_better_than"],
                                "significantly_worse_than": rb["ankyra_significantly_worse_than"]}
        if "scaled_errors" in proceed:
            rows = scaled_rows(T, P, rb, sc, REF_B, REF_B_LABEL)
            out["scaled_errors"] += rows
            fr["rmsse_median_unit"] = {r_["model"]: r_.get("RMSSE_median_unit") for r_ in rows}
        if "peak_readout" in proceed:
            rows, vs, ape = peak_rows_product(T, P, preds, prod["peak"], REF_B)
            out["peak_readout"] += rows
            fr["peak"] = {"vs": {k: {"log_ratio": v["log_ratio"], "UM": v["UM"]} for k, v in vs.items()}, "median_peak_ape_pct": ape}
        if xt is not None:
            wait_for_memory(f"extra {face}")
            fr["extra"] = xt.face(face, entry, P, prod, baselines, flag)
        fr["seconds"] = round(time.time() - t0, 1)
        rec["populations"][face] = fr
        log(f"tables {face}: rank {fr['hourly']['rank']}/{fr['hourly']['of']} ({fr['hourly']['mean_unit_rank']:.3f}); product {prod['info']}; {fr['seconds']} s")
        del P, preds, baselines, prod, r
        gc.collect()
    if "rank_tests" in proceed and all(f in pooled_R for f in TEST6):
        rows, rb = pooled_rank_rows(T, pooled_R, REF_B, REF_B_LABEL)
        out["rank_tests"] += rows
        rec["rank_tests_pooled"] = {"units": rb["units"], "friedman_p": rb["friedman_p"], "nemenyi_cd": rb["nemenyi_cd"], "position": rb["ankyra_position"],
                                    "mean_rank": rb["mean_rank"], "significantly_better_than": rb["ankyra_significantly_better_than"],
                                    "significantly_worse_than": rb["ankyra_significantly_worse_than"], "forecasters": len(rb["mean_rank"])}
    return out, _rename_ref(rec, REF_B), xt


# ----------------------------------------------------------------------------- output
FIELDS = {
    "benchmark_pairwise": ["set", "tier", "subset", "windows", "units", "model", "model_label", "model_class", "log_ratio", "um_low", "um_high", "improvement_pct", "pooled_mse_ratio"],
    "benchmark_mean_unit_rank": ["set", "tier", "subset", "model", "model_label", "mean_unit_rank", "n_models"],
    "conventional_metrics_late": ["set", "tier", "windows", "units", "model", "model_label", "model_class", "RMSE_kW", "MAE_kW", "CV_RMSE_pct", "NMBE_pct", "WAPE_pct", "WAPE_pooled_pct", "units_excluded"],
    "energy_error": ["set", "tier", "unit_set", "windows", "units", "model", "model_label", "model_class", "log_ratio", "um_low", "um_high", "improvement_pct", "median_energy_ape_pct"],
    "lead_day_metrics": ["set", "tier", "windows", "units", "model", "model_label", "model_class", "day", "RMSE_kW", "MAE_kW", "CV_RMSE_pct", "nMAE_pct", "GM_CV_RMSE_pct", "n_units_gm"],
    "lead_day_energy": ["set", "tier", "windows", "units_in_U", "model", "model_label", "model_class", "day", "GM_CV_daily_energy_pct", "GM_CV_cumulative_energy_pct"],
    "block_shares": ["set", "windows", "units", "model", "quantity", "block", "value"],
    "rank_tests": ["set", "tier", "units", "model", "model_label", "mean_rank", "friedman_p", "nemenyi_cd", "wilcoxon_holm_p_vs_ankyra"],
    "scaled_errors": ["set", "tier", "model", "model_label", "RMSSE_median_unit", "RMSSE_geo_mean", "MASE_median_unit", "MASE_geo_mean", "units", "units_below_1_share"],
    "peak_readout": ["set", "tier", "comparator", "log_ratio", "um_low", "um_high", "improvement_pct", "median_peak_ape_ankyra_pct", "median_peak_ape_previous_peak_pct"],
}
FLAG_BY_NAME = {}                                                      # set name -> flag of the run (filled in main)


def write_rows(name, rows, fields):
    """Write one CSV of the output directory (LF line ends, atomic replace)."""
    path = RESULTS / name
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r_.get(k, "") for k in fields} for r_ in rows)
    os.replace(tmp, path)
    return path


def write_csv(table, rows):
    fields = FIELDS[table] + ["flag"]
    for r_ in rows:
        if r_.get("model") == "TiDE" and "model_label" in r_:
            r_["model_label"] = TIDE_LABEL
        r_["flag"] = FLAG_BY_NAME.get(r_["set"], "")
    return write_rows(f"{table}.csv", rows, fields)


def _json_default(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, Path):
        return show(x)
    return str(x)


def dump_json(path, obj):
    tmp = path.with_name(path.name + ".part")
    text = json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default)
    tmp.write_bytes(scrub(text).encode("utf-8"))                       # LF line ends on every platform
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- summary
def _pct(v):
    return 100 * (1 - math.exp(v))


def _mark(lo, hi):
    return " *" if hi < 0 else (" (-)" if lo > 0 else "")


def _cell(lr, lo, hi):
    if lr is None or not math.isfinite(lr):
        return "n/a"
    return f"{_pct(lr):+.1f}{_mark(lo, hi)}"


def _p_text(p):
    return "p < 1e-300" if p == 0 else f"p = {p:.3g}"


def _recipe_sentence(rec):
    """How the evaluated carrier was trained, from the products mapping's recipe."""
    r = str(rec.get("recipe", ""))
    if r.startswith("span return"):
        return ("For each population the network was trained from scratch on a pooled training set (windows of several populations "
                "whose labels end before that population's validation set) with early stopping on the population's last 93 days "
                "before the training cut; then that validation span was returned to the training set and training continued for 8 "
                "epochs (the study's R recipe, the released recipe); three seeds averaged before anchoring.")
    return ("For each population the network was trained on a pooled training set (windows of several populations whose labels "
            "end before that population's validation set), early-stopped on that population's validation set, with three seeds "
            f"averaged before anchoring (recipe: {r or 'unknown'}).")


def _kan_kept_on_cineldi(pops):
    """True when the CINELDI rows of this run use the original-store LSTM-T-KAN-M forecast (flagged)."""
    return (pops.get("cineldi", {}).get("flag") or "").startswith("corrected CINELDI store")


def _kan_gate_bitwise(pops):
    return bool(pops.get("cineldi", {}).get("extra_baselines", {}).get(KAN[0], {}).get("retrain", {}).get("gate_original_store_bitwise"))


def summary_md(checks, rec, written):
    import extra_tables as XT                                 # the sections of the files beyond the ten tables
    pops = rec["populations"]
    order = [f for f in ("bdg2", "goiener_confirm", "households", "cambridge", "heew", "eweld", "oslo", "drammen", "cineldi") if f in pops]
    nb = N_BASELINES
    L = ["# ankyra-lite against the baselines", "",
         f"Test windows (`late` subset: origins after the training cutoff of the trained baselines) of {len(order)} population"
         f"{'s' if len(order) != 1 else ''}. The reference is **ankyra-lite** (version {rec.get('product_version', '?')}): the small carrier network",
         "(copy-pointer structured load network, 61,613 parameters) feeding the ANKYRA 2.2.1 anchoring layer, month-ahead (744 h)",
         "hourly forecasts. " + (_recipe_sentence(rec)),
         f"It is compared with **{nb} baselines** ({nb + 1} forecasters in all): the 20 external baselines of the ANKYRA release",
         "and LSTM-T-KAN-M, the LSTM-T-KAN model with its model and training code copied line by line from the authors' public",
         "repository (commit ce24938) and the authors' exp2 feature list (load and calendar). ANKYRA itself and its versions",
         "are not among the forecasters.", ""]
    if rec.get("partial_run"):
        L += [f"> **Development run** on {', '.join(pops[f]['name'] for f in order)} only; the full run covers the nine populations.", ""]
    L += ["## Read this first", "",
          "- **Re-evaluation.** The test windows have been read before (the release scoring, the scoring of the small carrier",
          "  network and an earlier hourly comparison), and parts of the recipe were chosen after looking at them (the copy",
          "  pointer's direction, the seasonal-naive state rule). Every number here is a retrospective description, not a first test.",
          "- **Drammen and CINELDI.** " + ("Their products are on the original CINELDI store, whose first hour is labelled 59 days"
                                           " too early (this shifts the calendar the anchoring layer uses on CINELDI; the Drammen carrier's"
                                           " pooled training set includes CINELDI windows from that store). Their rows are flagged `"
                                           + ORIGINAL_STORE_FLAG + "`."
                                           if any(rec.get("products", {}).get(f, {}).get("store") == "original" for f in ("drammen", "cineldi")) else
                                           "Both carriers were retrained with the corrected CINELDI store (the original store labels CINELDI's"
                                           " first hour 59 days too early): CINELDI on its own corrected data, Drammen with corrected CINELDI"
                                           " windows in its pooled training set. CINELDI is scored on the corrected store, with the baselines"
                                           " recomputed there: re-forecast where the start date enters, and the calendar-using trained baselines"
                                           + (" (GBT-T, TiDE and iTransformer-X) retrained with the same code and seeds."
                                              " **LSTM-T-KAN-M has no corrected-store CINELDI forecast in this run**: its original-store"
                                              " forecast is kept (not dropped), its CINELDI cells are marked `+` below, and every CINELDI row of"
                                              " the result files carries the flag `" + CORRECTED_CINELDI_FLAG + "`, because that forecast also"
                                              " enters CINELDI's ranks and tests."
                                              if _kan_kept_on_cineldi(pops) else
                                              " (GBT-T, TiDE, iTransformer-X and LSTM-T-KAN-M) retrained with the same code and seeds (LSTM-T-KAN-M"
                                              " also with the same protocol and learning-rate grid"
                                              + ("; its retrain first reproduced its stored original-store forecast bit for bit" if _kan_gate_bitwise(pops) else "")
                                              + ").")
                                           + " Drammen's baselines are unchanged (its own data are not affected)."),
          "- **TiDE** has a known implementation flaw: a decoder layer normalises a one-dimensional output to a constant, so",
          "  the model reduces to a linear map of the 1,344-hour context plus a constant, and its covariates are inert. It is",
          f"  kept under its release name, labelled `{TIDE_LABEL}`.",
          "- **LSTM-T-KAN-M** was trained per population with the model and training code copied line by line from the",
          "  authors' public repository (commit ce24938, MIT licence) and the authors' exp2 feature list. It is not a strict",
          "  reproduction of the paper. Deviations: one 744-step model per population instead of one model per horizon and",
          "  building; per-unit z-scores from the hours before the cut instead of Min-Max scaling; the panel's split (training",
          "  cut, last 20% of training origins for selection, origins every 24 h, at most 16,000 training pairs) instead of a",
          "  70/10/20 stride-1 split; no 3-point smoothing; at most 30 epochs with early-stopping patience 8 (authors: 100 and",
          "  10); learning rate chosen from 1e-4, 3e-4 and 1e-3 on the selection pairs; mean of seeds 0, 1 and 2 instead of one",
          "  run with seed 42; one fixed feature list with this repository's definitions and no weather. Every deviation is",
          "  listed in docs/EVALUATION.md. It is an ordinary baseline here, in every table and rank.",
          "- TimesFM 2.5's pretraining collection includes seven BDG2 sites.",
          "- On the three near-zero BDG2 meters (10 test windows) ankyra-lite's error is exactly zero: the micro-load rule",
          "  fires and the seasonal naive repeats the meter's constant reading. The unit-equal contrasts need a nonzero error",
          "  on both sides, so these meters drop out of them, and they count with ankyra-lite ranked first in the mean unit rank",
          "  (`bdg2_near_zero_sensitivity.csv`).",
          "- The Suzhou park (four series) has no ankyra-lite forecast and is not in the tables.",
          "- **The recipe was chosen on these test windows.** The span return was picked in the study's one reading of four recipe",
          "  arms on these windows (`recipe_history.csv`). Its gain is concentrated on the short-record populations' windows without",
          "  same-season training labels; it is about zero elsewhere, and it has two disclosed negatives: Drammen (hourly, against",
          "  the same continuation without the span return) and forecast days 2-7 of the six development populations (resolved).",
          "- No model was trained or tuned for this comparison; saved forecasts were scored with the release's own scoring",
          "  functions. The rows of switched-off parts of the anchoring layer re-run the unchanged anchoring layer on the saved",
          "  carrier forecasts (`evaluation/lite_reanchor.py`); the single-seed rows read the study's anchored single-seed products",
          "  (one checked bit for bit by re-anchoring); the cost and robustness tables run this repository's package on the CPU.", "",
          "## How to read the numbers", "",
          "- `improvement_pct` = 100[1 - exp(r)], r = the unit-equal log RMS ratio of ankyra-lite against the baseline",
          "  (positive favours ankyra-lite). `*` = resolved better (the 95% unit x target-month bootstrap interval of r lies",
          "  below zero), `(-)` = resolved worse (the interval lies above zero), no mark = not resolved.",
          "- **Hourly**: hourly errors over the 744 hours. **Energy**: monthly energy error, 744 x the error of the window's",
          "  mean load of the delivered trajectory (the release definition, applied to every forecaster).",
          "- **Peak**: ankyra-lite's peak readout (the envelope operator on the daily means of its delivered trajectory) against",
          "  the same readout applied to the baseline's trajectory, which is the release rule (`same readout on X`); the files",
          "  also hold the baseline's own trajectory maximum (`trajectory maximum of X`).",
          f"- **Rank**: position among the {nb + 1} forecasters by mean per-unit rank (forecasters ranked within each unit by RMSE).",
          "- On windows where the anchoring layer's off-state or micro-load rule fires, ankyra-lite returns the weekly",
          "  seasonal naive forecast (the last 168 hours repeated).",
          "- In the result files the reference forecaster is ankyra-lite: `model` = `ankyra-lite` on its rows, and column names",
          "  that say `ankyra` (`median_peak_ape_ankyra_pct`, `wilcoxon_holm_p_vs_ankyra`, `ankyra_mean_unit_rank`,",
          "  `ankyra_design_status`, `ankyra_kw`) refer to ankyra-lite.", "",
          "## Overview", "",
          f"Counts are over the {nb} baselines: resolved better / resolved worse.", "",
          f"| Population | Windows / units | Seasonal-naive windows | Rank of {nb + 1} (mean unit rank) | Hourly | Energy | Energy (level readout) | Peak (same readout) | Peak (trajectory max) | Flag |",
          "|---|---|---:|---|---|---|---|---|---|---|"]

    def counts(d, lo, hi):
        return f"{sum(v[hi] < 0 for v in d.values())} / {sum(v[lo] > 0 for v in d.values())}" if d else "-"

    for f in order:
        p = pops[f]
        h = {k: {"lo": v["UM"][0], "hi": v["UM"][1]} for k, v in p["hourly"]["vs"].items()}
        e = {k.split("|", 1)[1]: v for k, v in p.get("energy", {}).items() if k.startswith("all units|") and not k.endswith("|" + REF_B)}
        el = {k: {"lo": v["UM"][0], "hi": v["UM"][1]} for k, v in p.get("energy_level_readout", {}).get("vs", {}).items()}
        pvs = p.get("peak", {}).get("vs", {})
        pk = {k: {"lo": v["UM"][0], "hi": v["UM"][1]} for k, v in pvs.items() if k.startswith("same readout on ")}
        px = {k: {"lo": v["UM"][0], "hi": v["UM"][1]} for k, v in pvs.items() if k.startswith("trajectory maximum of ")}
        L.append(f"| {p['name']} | {p['windows']} / {p['units']} | {p['product']['state_windows_seasonal_naive']} | {p['hourly']['rank']} ({p['hourly']['mean_unit_rank']:.2f}) | "
                 f"{counts(h, 'lo', 'hi')} | {counts(e, 'um_low', 'um_high')} | {counts(el, 'lo', 'hi')} | {counts(pk, 'lo', 'hi')} | {counts(px, 'lo', 'hi')} | {p['flag'] or ''} |")
    nbind = {f: pops[f]["product"]["level_vs_trajectory_mean_windows_over_1e-6"] for f in order}
    losses = []
    for f in order:
        for k, v in pops[f]["hourly"]["vs"].items():
            if v["UM"][0] > 0:
                losses.append(f"{pops[f]['name']}: {k} ({_cell(v['log_ratio'], *v['UM'])})")
    L += ["", "- **Hourly.** " + ("Resolved losses: " + "; ".join(losses) + "." if losses else "No resolved loss."),
          "- **Energy** is the release definition (delivered trajectory). The release's energy table covers the six test",
          "  populations and the Suzhou park; the Oslo, Drammen and CINELDI rows are additional, with the same estimator.",
          "- **Energy (level readout)** is a sensitivity: ankyra-lite's independent energy readout (744 x its level) in place",
          "  of the delivered trajectory's mean. The two differ only on windows where the projection onto nonnegative load",
          "  binds: " + ", ".join(f"{pops[f]['name']} {n}" for f, n in nbind.items()) + " windows.",
          "- **Peak (trajectory max)** counts the contrasts against each baseline's own trajectory maximum.", "",
          "## By population", "",
          "Improvement of ankyra-lite in % (positive = ankyra-lite better). Energy (level) = with ankyra-lite's level",
          "readout; Peak (readout) = against the same readout on the baseline; Peak (max) = against the baseline's",
          f"trajectory maximum. The CSV files hold all {nb} baselines.", ""]
    for f in order:
        p = pops[f]
        h = p["hourly"]["vs"]
        e = {k.split("|", 1)[1]: v for k, v in p.get("energy", {}).items() if k.startswith("all units|")}
        el = p.get("energy_level_readout", {}).get("vs", {})
        pv = p.get("peak", {}).get("vs", {})
        L += [f"### {p['name']} ({p['windows']} windows, {p['units']} units; rank {p['hourly']['rank']} of {p['hourly']['of']}, mean unit rank {p['hourly']['mean_unit_rank']:.2f})", ""]
        if p["flag"]:
            L += [f"Flag: {p['flag']}." + (" `+`: forecast from the original CINELDI store." if f == "cineldi" and p["flag"].startswith("corrected CINELDI store") else ""), ""]
        L += ["| Baseline | Hourly | Energy | Energy (level) | Peak (readout) | Peak (max) |", "|---|---|---|---|---|---|"]
        for m in SUMMARY_BASELINES:
            hc = _cell(h[m]["log_ratio"], *h[m]["UM"]) if m in h else "-"
            ec = _cell(e[m]["log_ratio"], e[m]["um_low"], e[m]["um_high"]) if m in e else "-"
            lc = _cell(el[m]["log_ratio"], *el[m]["UM"]) if m in el else "-"
            a_, b_ = pv.get(f"same readout on {m}"), pv.get(f"trajectory maximum of {m}")
            pc = _cell(a_["log_ratio"], *a_["UM"]) if a_ else "-"
            qc = _cell(b_["log_ratio"], *b_["UM"]) if b_ else "-"
            lab = "Seasonal naive (week)" if m == "SN-week" else m
            if f == "cineldi" and m == "LSTM-T-KAN-M" and (p["flag"] or "").startswith("corrected CINELDI store"):
                lab += " +"
            L.append(f"| {lab} | {hc} | {ec} | {lc} | {pc} | {qc} |")
        own, prev = pv.get("trajectory maximum"), pv.get("previous-window peak")
        extra = []
        if own:
            extra.append(f"peak readout vs its own trajectory maximum {_cell(own['log_ratio'], *own['UM'])}")
        if prev:
            extra.append(f"vs the previous window's peak {_cell(prev['log_ratio'], *prev['UM'])}")
        if f == "bdg2" and "energy" in p:
            ez = {k.split("|", 1)[1]: v for k, v in p["energy"].items() if k.startswith("without the three near-zero meters|")}
            if ez:
                extra.append("energy without the three near-zero meters: " + ", ".join(
                    f"{m} {_cell(ez[m]['log_ratio'], ez[m]['um_low'], ez[m]['um_high'])}" for m in SUMMARY_BASELINES if m in ez))
        if extra:
            L += ["", "Also: " + "; ".join(extra) + "."]
        L += [""]
    if "rank_tests_pooled" in rec:
        q = rec["rank_tests_pooled"]
        L += ["## Rank tests, six test populations pooled", "",
              f"{q['units']} units, Friedman {_p_text(q['friedman_p'])}, Nemenyi critical difference {q['nemenyi_cd']:.3f}; ankyra-lite's mean rank",
              f"{q['mean_rank'][REF_B]:.3f} (position {q['position']} of {q.get('forecasters', nb + 1)}). Holm-corrected Wilcoxon: significantly better than "
              f"{len(q['significantly_better_than'])} of {nb} baselines, significantly worse than {len(q['significantly_worse_than'])}"
              + (f" ({', '.join(q['significantly_worse_than'])})" if q["significantly_worse_than"] else "") + ". Descriptive.", ""]
    if rec.get("extra"):                                      # rebuilt from the saved record with the current code (no data read)
        L += XT.summary_from_record(rec)
    elif rec.get("extra_summary"):
        L += rec["extra_summary"]
    L += ["## Scoring-code check", "",
          "Before the reference was replaced, the scoring code was checked: the late-window rows of ten release tables were",
          "recomputed with the release's own reference forecaster and compared with the published files of the first release",
          "repository (the peak table on all windows, as published). A table counts as reproduced when every compared field",
          "agrees to the published rounding; only reproduced tables are written. The recomputed release numbers are not part",
          "of any result file of this repository.", "",
          "| Table | Published rows | Fields compared | Exact | Last digit | Mismatches | Result |", "|---|---:|---:|---:|---:|---:|---|"]
    for t in TABLES:
        c = checks[t]
        L.append(f"| `{t}` | {c['published_rows_in_scope']} | {c['fields_compared']} | {c['exact']} | {c['within_last_digit']} | "
                 f"{len(c['mismatches']) + len(c['missing'])} | {c['status']}{'' if t in written else ' (not written)'} |")
    L += ["", f"Hourly cross-check against the earlier hourly-only comparison (20 baselines): {XT.plain(rec.get('hourly_crosscheck', 'not run'))}.", ""]
    sc = [p.get("extra", {}).get("study_scorer_crosscheck") for p in pops.values()]
    sc = [x for x in sc if x]
    if sc:
        L += [f"Product check against the study's own scorer of the recipe arms: {sum(x['pass'] for x in sc)} of {len(sc)} populations reproduce "
              f"its per-population contrasts (largest difference {max(x['max_abs_diff_pct_points'] for x in sc):.1e} percentage points).", ""]
    L += ["## Files", "",
          "All in `results/`. The files of the main comparison have the columns of the release files of the same name plus a",
          "trailing `flag` column; the other files are described with their columns in `results/README.md`.", "",
          "| File | Content |", "|---|---|"]
    desc = {"benchmark_pairwise": f"hourly contrasts against each of the {nb} baselines (late windows)",
            "benchmark_mean_unit_rank": f"mean per-unit rank of the {nb + 1} forecasters",
            "conventional_metrics_late": f"RMSE, MAE, CV(RMSE), NMBE, WAPE of the {nb + 1} forecasters",
            "energy_error": "monthly energy contrasts and median energy APE (BDG2 also without the three near-zero meters)",
            "lead_day_metrics": "hourly error by forecast day 1-31 (six test populations)",
            "lead_day_energy": "daily and cumulative energy error by forecast day (six test populations)",
            "block_shares": "where the hourly squared error sits (level, daily path, within-day) for ankyra-lite, TimesFM and Chronos-2-X",
            "rank_tests": "Friedman, Nemenyi and Holm-corrected Wilcoxon tests on per-unit RMSE (descriptive)",
            "scaled_errors": "RMSSE and MASE with the weekly seasonal-naive in-sample scale (descriptive)",
            "peak_readout": "peak readout contrasts (late windows): own trajectory maximum, previous-window peak, and for each baseline the same readout and its trajectory maximum"}
    for t in TABLES:
        if t in written:
            L.append(f"| `{t}.csv` | {desc[t]} |")
    for name, d in rec.get("extra_files", {}).items():
        L.append(f"| `{name}` | {d} |")
    L += ["| `REPRODUCTION_CHECK.json` | the scoring-code check: every table's comparison with the published release files |",
          "| `compare_baselines.json` | all numbers behind this summary, per population, with the product checks |",
          "| `compare_baselines.log` | the log of the run that wrote these files |", "",
          "Produced by `evaluation/compare_baselines.py` (products: `evaluation/products.json`).", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default=",".join(PANEL10), help="populations (the check also reads the Suzhou park)")
    ap.add_argument("--out", default=None, help="output directory (default: results/)")
    ap.add_argument("--recipe", default="R", choices=sorted(RECIPE_PRODUCTS),
                    help="products mapping: R = the released recipe (products.json), p3s = its from-scratch stage alone (products_p3s.json)")
    ap.add_argument("--products", default=None, help="another products mapping (overrides --recipe)")
    ap.add_argument("--reanchor", action="store_true", help="run the re-anchoring (lite_reanchor.py) for the faces first")
    ap.add_argument("--reanchor-modes", default=None, help="comma list of re-anchoring modes (default: lite_reanchor.default_modes)")
    ap.add_argument("--seed-gate", type=int, default=8, help="windows of the single-seed re-anchoring gate (0: none)")
    ap.add_argument("--workers", type=int, default=2, help="processes for the re-anchoring")
    ap.add_argument("--no-extra", action="store_true", help="only the ten tables of the main comparison")
    ap.add_argument("--no-cost", action="store_true", help="skip the timing of cost_per_window.csv")
    ap.add_argument("--no-robustness", action="store_true", help="skip robustness.csv (CPU re-forecast on corrupted contexts)")
    ap.add_argument("--robustness-face", default=None, help="population of robustness.csv (default: drammen, else oslo)")
    ap.add_argument("--robustness-windows", type=int, default=None, help="windows of robustness.csv (default 64)")
    ap.add_argument("--only-check", action="store_true", help="only the scoring-code check")
    ap.add_argument("--summary-only", action="store_true", help="rebuild SUMMARY.md from the saved JSON records (no data read)")
    ap.add_argument("--check-products", action="store_true")
    a = ap.parse_args()
    if a.out:
        RESULTS = Path(a.out).resolve()
    if _in_read_only(RESULTS):
        raise SystemExit("the output directory lies in a read-only tree")
    RESULTS.mkdir(parents=True, exist_ok=True)
    sys.modules.setdefault("compare_baselines", sys.modules[__name__])      # one module object for extra_tables / lite_reanchor
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    prods = load_products(a.products or RECIPE_PRODUCTS[a.recipe])
    if a.check_products:
        print(json.dumps(check_products(prods), indent=1))
        return
    if a.summary_only:
        chk = json.loads((RESULTS / "REPRODUCTION_CHECK.json").read_text(encoding="utf-8"))
        rec = json.loads((RESULTS / "compare_baselines.json").read_text(encoding="utf-8"))
        (RESULTS / "SUMMARY.md").write_bytes(scrub(summary_md(chk["tables"], rec, rec["tables_written"])).encode("utf-8"))
        print("SUMMARY.md rebuilt from the saved records")
        return
    log.open(RESULTS / "compare_baselines.log")
    faces = [f for f in a.faces.split(",") if f]
    t0 = time.time()
    if a.reanchor:
        import lite_reanchor as LR
        LR.run(faces=[f for f in faces if f in NINE], modes=a.reanchor_modes.split(",") if a.reanchor_modes else None, workers=a.workers, prods=prods,
               seed_gate=a.seed_gate)
    T = Tools()
    log("tools imported; read-only roots: research root, original release root")
    checks, notes = step_a(T, faces)
    proceed = [t for t in TABLES if checks[t]["status"] == "REPRODUCED"]
    stopped = [t for t in TABLES if t not in proceed]
    dump_json(RESULTS / "REPRODUCTION_CHECK.json", {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what": "scoring-code check: the late-window rows of ten release tables recomputed with the scoring functions used for ankyra-lite and the release's own reference forecaster, compared field by field with the published files; only the match counts are kept",
        "reference_of_the_check": "the release's reference forecaster (saved forecasts of the release panel)",
        "published": "<original release root>/results", "published_state": published_source(), "faces": faces, "tables": checks, "populations": notes,
        "tables_reproduced": proceed, "tables_not_reproduced": stopped})
    log("check done; reproduced:", proceed, "| not reproduced:", stopped)
    if a.only_check:
        return
    nine_run = [f for f in faces if f in NINE]
    for f in nine_run:
        e = prods["faces"].get(f)
        if e is not None:
            FLAG_BY_NAME[T.names[f]] = face_flag(f, e)
    out, rec, xt = step_b(T, faces, proceed, prods, extra=not a.no_extra)
    written = []
    for t in proceed:
        if out[t]:
            write_csv(t, out[t])
            written.append(t)
    xc = [p.get("hourly_crosscheck") for p in rec["populations"].values()]
    if any(xc):
        xok = (f"{sum(1 for x in xc if x)} populations compared, maximum absolute difference "
               f"{max((x['max_abs_diff'] for x in xc if x), default=float('nan')):.3g} (percentage points / log-ratio units), "
               f"same ranks among the 21: {all(x['same_rank_among_21'] for x in xc if x)}, same windows and units: {all(x['same_windows_units'] for x in xc if x)}")
    else:
        xok = ("not applicable: the earlier hourly-only comparison scored the from-scratch stage alone, not this product; "
               "the product reading is checked against the study's own scorer instead (`study_scorer_crosscheck`)")
    rec.update({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "product_label": prods["label"], "product_version": prods.get("version"), "recipe": prods["recipe"],
                "recipe_description": prods.get("recipe_description"), "products_mapping": show(prods["path"]),
                "products": {f: {"store": e["store"], "npz": show(e["npz_path"]), "checkpoint": e["checkpoint"], "crosscheck": bool(e.get("crosscheck")),
                                 "seed_products": [show(x) for x in e["seed_paths"]], "start_runs": [show(x) for x in e["start_paths"]]}
                             for f, e in prods["faces"].items()},
                "reference": f"ankyra-lite ({prods['path'].name}; seasonal naive on state windows)",
                "forecasters_left_out": LEFT_OUT, "extra_baselines": list(KAN), "baselines": N_BASELINES,
                "tables_written": written, "tables_not_written": [t for t in TABLES if t not in written],
                "hourly_crosscheck": xok, "faces_run": [f for f in faces if f in rec["populations"]],
                "partial_run": sorted(rec["populations"]) != sorted(NINE)})
    if xt is not None:
        extra_written, extra_rec, extra_summary, extra_files = xt.finish(RESULTS, rec, cost=not a.no_cost, robustness=not a.no_robustness,
                                                                         robustness_face=a.robustness_face, robustness_windows=a.robustness_windows)
        rec["extra"] = extra_rec
        rec["extra_written"] = extra_written
        rec["extra_summary"] = extra_summary
        rec["extra_files"] = extra_files
    rec["seconds"] = round(time.time() - t0, 1)
    dump_json(RESULTS / "compare_baselines.json", rec)
    (RESULTS / "SUMMARY.md").write_bytes(scrub(summary_md(checks, rec, written)).encode("utf-8"))
    log("written:", written + rec.get("extra_written", []), "| total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
