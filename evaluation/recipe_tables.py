"""recipe_history.csv and constants.csv: ankyra-lite's own version history and every constant of the forecaster
(called by compare_baselines.py; neither file reads a data store).

recipe_history.csv  The study's one reading of the recipe arms on the test windows (the counterpart of the first release's
                    version archives): the from-scratch stage (the study's P3s recipe) and three ways to continue it,
                    F (8 more epochs, validation set held out), R (the 93-day validation span returned to the training
                    set, 8 more epochs; the released recipe) and V (a proportional validation set with early stopping).
                    Every number is copied unrounded from the study's score record (analysis_outputs/smallnet_sr_formal_v1/
                    score_all.json); nothing is recomputed.  Contrasts R vs F, V vs F, V vs R: three seeds per arm, the
                    95% interval combines the unit x target-month bootstrap of the three-seed ensembles with the spread
                    of the three single-seed contrasts (Welch degrees of freedom).  F, R, V against the un-continued
                    from-scratch stage: unit x month bootstrap only (no seed layer; descriptive).
constants.csv       The carrier's constants (the defaults of ankyra_lite.carrier.train.TrainConfig and of the span-return
                    recipe, the network configuration of ankyra_lite.carrier.model) and the anchoring layer's constants,
                    which are those of the first release's constants.csv (the layer is a byte-identical copy of ANKYRA
                    2.2.1); the anchoring rows keep the first release's group, constant, value and where_fixed.
"""
from __future__ import annotations

import csv
import io
import json
import sys

import compare_baselines as CB

ARMS = {"P3s": "from-scratch stage alone (93-day validation set held out, early stopping; no continuation)",
        "F": "from-scratch stage + 8 continuation epochs, validation set still held out",
        "R": "from-scratch stage + validation span returned to the training set + 8 continuation epochs (released recipe)",
        "V": "from-scratch stage + continuation with a proportional validation set (10% of the record, 38-93 days) and early stopping"}
GROUPS = {"three short-record populations": ("BDG2 2017, GoiEner non-household, GoiEner households", 3),
          "nine": ("all nine populations, each with equal weight", 9),
          "six development populations": ("Cambridge, HEEW Arizona, EWELD, Oslo, Drammen, CINELDI", 6),
          "four long-record populations (V keeps the 93-day validation)": ("Cambridge, HEEW Arizona, EWELD, Oslo (V keeps the 93-day validation set)", 4)}
LAYERS = {"switch layer | three short-record populations": "switch layer",
          "non-switch layer | three short-record populations": "non-switch layer"}
SHORT3 = ("bdg2", "goiener_confirm", "households")
FIELDS = ["comparison", "first_arm", "second_arm", "population_group", "populations", "layer", "quantity", "improvement_pct",
          "interval_low_pct", "interval_high_pct", "interval", "label", "seed0_pct", "seed1_pct", "seed2_pct", "unit_month_low_pct",
          "unit_month_high_pct", "welch_df", "by_population_pct", "first_arm_recipe", "second_arm_recipe"]
WELCH = "Welch 95%: unit x target-month bootstrap of the three-seed ensembles and the spread of the three single-seed contrasts"
UM_ONLY = "unit x target-month bootstrap 95% of the three-seed ensembles only (no seed layer; descriptive)"


def _label(lo, hi):
    return "resolved positive" if lo > 0 else ("resolved negative" if hi < 0 else "not resolved")


def _bypop(d):
    return "; ".join(f"{CB.SHOW_NAMES.get(f, f)} {v!r}" for f, v in d.items())


def recipe_history(score=None, census=None):
    """Rows of recipe_history.csv and a short record (the readouts of the summary)."""
    S = json.loads((score or CB.SR_SCORE).read_text(encoding="utf-8"))
    C = json.loads((census or CB.SPAN_CENSUS).read_text(encoding="utf-8"))["faces"] if (census or CB.SPAN_CENSUS).exists() else {}
    sw = {f: int(C[f]["late_switch_uncovered_to_covered"]) for f in SHORT3 if f in C}
    rows = []

    def add(comp, group, layer, q, v):
        a, b = comp.split(" vs ")
        rows.append({"comparison": comp, "first_arm": a, "second_arm": b, "population_group": GROUPS[group][0], "populations": GROUPS[group][1],
                     "layer": layer, "quantity": q, "improvement_pct": v["pct"], "interval_low_pct": v["welch_95_pct"][0],
                     "interval_high_pct": v["welch_95_pct"][1], "interval": WELCH, "label": v["label"],
                     "seed0_pct": v["seed_pct"][0], "seed1_pct": v["seed_pct"][1], "seed2_pct": v["seed_pct"][2],
                     "unit_month_low_pct": v["unit_month_interval_pct"][0], "unit_month_high_pct": v["unit_month_interval_pct"][1],
                     "welch_df": v["welch_df"], "by_population_pct": _bypop(v["by_population_pct"]), "first_arm_recipe": ARMS[a], "second_arm_recipe": ARMS[b]})

    for group in GROUPS:
        for comp, qs in S[group].items():
            for q, v in qs.items():
                add(comp, group, "all test windows", q, v)
    for key, layer in LAYERS.items():
        desc = (f"{layer}: test windows whose target month had no same-season training labels before the span return and has them after it"
                if layer == "switch layer" else f"{layer}: the other test windows of the three populations")
        if sw:
            n = {"switch layer": sw, "non-switch layer": {f: int(C[f]["late_windows"]) - sw[f] for f in sw}}[layer]
            desc += " (" + " / ".join(f"{n[f]}" for f in SHORT3) + " windows of BDG2 2017 / GoiEner non-household / GoiEner households)"
        for comp, qs in S[key].items():
            for q, v in qs.items():
                add(comp, "three short-record populations", desc, q, v)
    for key, v in S["descriptive vs un-continued P3s (unit x month only)"].items():
        arm, group = key.split("_ens vs P3s | ")
        g = {"short3": "three short-record populations", "nine": "nine"}[group]
        lo, hi = v["pct_interval"]
        rows.append({"comparison": f"{arm} vs P3s", "first_arm": arm, "second_arm": "P3s", "population_group": GROUPS[g][0], "populations": GROUPS[g][1],
                     "layer": "all test windows", "quantity": "hourly", "improvement_pct": v["improvement_pct"], "interval_low_pct": lo,
                     "interval_high_pct": hi, "interval": UM_ONLY, "label": _label(lo, hi), "seed0_pct": "", "seed1_pct": "", "seed2_pct": "",
                     "unit_month_low_pct": lo, "unit_month_high_pct": hi, "welch_df": "", "by_population_pct": _bypop(v["by_population_pct"]),
                     "first_arm_recipe": ARMS[arm], "second_arm_recipe": ARMS["P3s"]})
    guards = {f"{k} | {q}": bool(g["credible_harm"]) for k, qs in S["guards (credible harm = Welch upper below the line) | nine"].items() for q, g in qs.items()}
    rec = {"source": CB.show(score or CB.SR_SCORE), "identity": S.get("identity"), "arms": S.get("arms"), "rows": len(rows),
           "credible_harm_guards_nine": guards, "V_selected_epochs": S.get("V picks (best epoch per population and seed)"),
           "switch_layer_windows": sw,
           "headline": {k: S[g][c][q] for k, (g, c, q) in {
               "short3 R vs F hourly": ("three short-record populations", "R vs F", "hourly"),
               "switch layer R vs F hourly": ("switch layer | three short-record populations", "R vs F", "hourly"),
               "non-switch layer R vs F hourly": ("non-switch layer | three short-record populations", "R vs F", "hourly"),
               "nine R vs F hourly": ("nine", "R vs F", "hourly"),
               "six development R vs F hourly": ("six development populations", "R vs F", "hourly"),
               "six development R vs F lead days 2-7": ("six development populations", "R vs F", "lead days 2-7"),
               "short3 R vs F monthly energy": ("three short-record populations", "R vs F", "monthly (curve sum)"),
               "short3 R vs F peak": ("three short-record populations", "R vs F", "peak"),
               "switch layer V vs R hourly": ("switch layer | three short-record populations", "V vs R", "hourly"),
               "four long-record V vs F hourly": ("four long-record populations (V keeps the 93-day validation)", "V vs F", "hourly")}.items()},
           "continuation_vs_from_scratch": {k: {"pct": v["improvement_pct"], "unit_month_interval_pct": v["pct_interval"]}
                                            for k, v in S["descriptive vs un-continued P3s (unit x month only)"].items()}}
    return rows, rec


# ----------------------------------------------------------------------------- constants
def _fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (tuple, list)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def carrier_constants(score_rec=None):
    """The carrier's rows, read from the package (so that they follow its defaults)."""
    if str(CB.REPO) not in sys.path:
        sys.path.append(str(CB.REPO))
    from ankyra_lite.carrier import train as TR
    from ankyra_lite.carrier import model as MD
    cfg, rcfg = TR.TrainConfig(), TR.recipe("span_return")
    mc = MD.build_carrier()
    m = mc.config
    head = score_rec.get("headline", {}) if score_rec else {}
    cont = score_rec.get("continuation_vs_from_scratch", {}) if score_rec else {}

    def sens(key, what):
        v = head.get(key)
        return "" if v is None else f"{what}: {v['pct']:+.2f}% hourly, Welch 95% {v['welch_95_pct'][0]:+.2f} to {v['welch_95_pct'][1]:+.2f} ({v['label']}; recipe_history.csv)"

    def sens_um(key, what):
        v = cont.get(key)
        return "" if v is None else f"{what}: {v['pct']:+.2f}% hourly, unit x month 95% {v['unit_month_interval_pct'][0]:+.2f} to {v['unit_month_interval_pct'][1]:+.2f} (descriptive; recipe_history.csv)"

    DEV = "the study's carrier recipe (from-scratch stage); design choices checked on the development windows of six populations (not BDG2 or GoiEner)"
    NET = "network configuration of the study's carrier, unchanged since the from-scratch recipe was fixed"
    PLAN = "the study's span-return plan, written before its one reading of the recipe arms on the test windows"
    v = cfg.validation
    rs = rcfg.validation
    rows = [
        ("carrier: windows", "context / horizon", f"{cfg.context_hours:,} h / {cfg.horizon_hours} h", "problem definition (as the anchoring layer)", ""),
        ("carrier: windows", "eligibility: observed share of the context and of the label hours", f">= {cfg.min_observed_fraction:g}", DEV, ""),
        ("carrier: windows", "origin grids: training / validation / per-population window count", f"{cfg.train_origin_grid_hours} h / {cfg.val_origin_grid_hours} h / {cfg.pool_count_grid_hours} h", DEV, ""),
        ("carrier: split", "validation set before the training cut (from-scratch stage)", f"{v.days:g} days (3 x 744 h)", DEV,
         sens("four long-record V vs F hourly", "proportional validation set with early stopping against the held-out 93 days, four long-record populations (where the proportional rule also gives 93 days)")),
        ("carrier: split", "caps per population: training / validation windows (unit-balanced)", f"{cfg.train_cap:,} / {cfg.val_cap:,}", DEV, ""),
        ("carrier: pooled training set", "borrowed windows: labels end before the target's validation start (absolute time); per-population count matched to the weekly grid", "yes", DEV, ""),
        ("carrier: sampler", "draws per epoch / share from the target population", f"{cfg.draws_per_epoch:,} / {cfg.own_fraction:g}", DEV, ""),
        ("carrier: optimisation", "AdamW learning rate / weight decay / batch", f"{cfg.lr:g} / {cfg.weight_decay:g} / {cfg.batch_size}", DEV, ""),
        ("carrier: optimisation", "loss: scale-normalised trajectory MSE + KL weight, log1p compression per window", f"KL weight {cfg.kl_weight:g}; compression {_fmt(cfg.log1p_compression)}", DEV, ""),
        ("carrier: optimisation", "gradient-norm clipping (separately for the base and the copy-pointer parameters)", f"{cfg.grad_clip:g}", DEV, ""),
        ("carrier: early stopping", "from-scratch stage: at most epochs / patience", f"{cfg.max_epochs} / {cfg.patience}", DEV, ""),
        ("carrier: span return", "validation span returned to the training set after the from-scratch stage", f"{rs.days:g} days, at the population's own sampling density", PLAN,
         sens("nine R vs F hourly", "span returned against held out (both with 8 continuation epochs), nine populations")),
        ("carrier: span return", "returned labels end at least this long before the training cut", f"{rs.boundary_hours} h", PLAN, ""),
        ("carrier: span return", "continuation: fixed epochs, final weights kept (no selection), fresh AdamW", f"{rcfg.fixed_epochs}", PLAN,
         sens_um("R_ens vs P3s | nine", "released recipe against the un-continued from-scratch stage, nine populations")),
        ("carrier: span return", "continuation sampler stream", f"default_rng({rcfg.sampler_seed_offset} + seed)", PLAN, ""),
        ("carrier: seeds", "seeds (forecast = plain average of the seeds' forecasts)", _fmt(cfg.seeds), DEV, ""),
        ("carrier network", "parameters", f"{MD.parameter_count(mc):,}", NET, ""),
        ("carrier network", "daily / monthly / future-day tokens", f"{366} x {m.day_features} / 24 x {m.month_features} / {m.days} x {m.future_features}", NET, ""),
        ("carrier network", "width / attention heads / dropout", f"{m.width} / {m.heads} / {m.dropout:g}", NET, ""),
        ("carrier network", "dilations: daily tokens / monthly tokens (two shared gated causal blocks, kernel 3)", f"{_fmt(m.day_dilations)} / {_fmt(m.month_dilations)}", NET, ""),
        ("carrier network", "hourly dictionary rank / bound of the origin-conditioned change", f"{m.rank} / {m.dictionary_delta:g}", NET, ""),
        ("carrier network", "copy pointer: complete days attended / key dimension", f"{mc.copy_days} / {mc.key_dim}", NET, ""),
        ("carrier network", "scale: max(mean of the last 744 observed hours, mean of the last 8,760 observed hours)", "1 if not positive", NET, ""),
    ]
    return [{"group": g, "constant": c, "value": val, "where_fixed": w, "sensitivity": s, "component": "carrier"} for g, c, val, w, s in rows]


def anchoring_constants():
    """The first release's constants.csv as committed (the anchoring layer is a byte-identical copy of that state)."""
    txt = CB.published_text("constants.csv")
    out = []
    for r in csv.DictReader(io.StringIO(txt)):
        out.append({"group": r["group"], "constant": r["constant"], "value": r["value"], "where_fixed": r["where_fixed"],
                    "sensitivity": "", "component": "anchoring layer (unchanged copy; values and where_fixed as in the first release)"})
    return out


CONSTANT_FIELDS = ["group", "constant", "value", "where_fixed", "sensitivity", "component"]


def write(out_dir, log=print):
    """Write both files; returns (written names, record, descriptions)."""
    written, files, rec = [], {}, {}
    if CB.SR_SCORE.exists():
        rows, rrec = recipe_history()
        CB.write_rows("recipe_history.csv", [{k: ("" if r[k] is None else r[k]) for k in FIELDS} for r in rows], FIELDS)
        written.append("recipe_history.csv")
        files["recipe_history.csv"] = "the study's one reading of the recipe arms (from-scratch stage, F, R, V): ankyra-lite's version history"
        rec["recipe_history"] = rrec
    else:
        rrec = None
        log("recipe history: the study's score record is missing; recipe_history.csv not written")
    crow = carrier_constants(rrec)
    try:
        arow = anchoring_constants()
    except (OSError, KeyError) as e:                                                  # no release state to read
        arow = []
        log(f"constants: the first release's constants.csv could not be read ({type(e).__name__}); anchoring rows left out")
    CB.write_rows("constants.csv", crow + arow, CONSTANT_FIELDS)
    written.append("constants.csv")
    files["constants.csv"] = "every constant: the carrier's (training recipe, span return, network) and the anchoring layer's (as in the first release)"
    rec["constants"] = {"carrier_rows": len(crow), "anchoring_rows": len(arow)}
    return written, rec, files


def summary(rec):
    """SUMMARY.md lines of the recipe history."""
    r = (rec or {}).get("recipe_history")
    if not r:
        return []
    h = r["headline"]

    def c(k):
        v = h[k]
        return f"{v['pct']:+.2f}% ({v['welch_95_pct'][0]:+.2f} to {v['welch_95_pct'][1]:+.2f}; {v['label']})"
    sw = r.get("switch_layer_windows") or {}
    dram = h["nine R vs F hourly"]["by_population_pct"].get("drammen")
    cont = r["continuation_vs_from_scratch"]
    L = ["## Recipe history (`recipe_history.csv`)", "",
         "The released recipe was chosen in the study's one reading of four recipe arms on these test windows: the from-scratch",
         "stage alone, and three continuations of it for 8 epochs from the same weights, with the validation set held out (plain",
         "continuation), with the validation span returned to the training set (span return, the released recipe) and with a",
         "proportional validation set and early stopping (the study's P3s, F, R and V arms; the codes are the `first_arm` and",
         "`second_arm` values of the file). Positive favours the first-named arm; Welch 95% intervals include the seed layer.", "",
         "| Readout | Span return against the plain continuation |", "|---|---|",
         f"| three short-record populations, hourly | {c('short3 R vs F hourly')} |",
         f"| switch layer{(' (' + ', '.join(f'{CB.SHOW_NAMES[f]} {sw[f]}' for f in SHORT3 if f in sw) + ' windows)') if sw else ''}, hourly | {c('switch layer R vs F hourly')} |",
         f"| non-switch layer, hourly | {c('non-switch layer R vs F hourly')} |",
         f"| nine populations, hourly | {c('nine R vs F hourly')} |",
         f"| six development populations, hourly | {c('six development R vs F hourly')} |",
         f"| six development populations, forecast days 2-7 | {c('six development R vs F lead days 2-7')} |",
         f"| three short-record populations, monthly energy (delivered trajectory) | {c('short3 R vs F monthly energy')} |",
         f"| three short-record populations, peak | {c('short3 R vs F peak')} |", "",
         "- The switch layer is the short-record populations' test windows whose target month had no same-season training labels",
         "  before the span return and has them after it. The gain sits there, as the prediction written before the reading put",
         "  it; elsewhere it is about zero, and over the nine populations it is not resolved.",
         f"- **Negatives, disclosed:** Drammen {dram:+.2f}% hourly (span return against the plain continuation), and forecast days",
         f"  2-7 of the six development populations {h['six development R vs F lead days 2-7']['pct']:+.2f}% (resolved negative).",
         f"- The proportional validation set with early stopping gained nothing (four long-record populations, against the plain",
         f"  continuation: {c('four long-record V vs F hourly')}); on the switch layer it trails the span return",
         f"  ({c('switch layer V vs R hourly')}).",
         f"- Against the un-continued from-scratch stage (unit x month intervals only, nine populations): span return",
         f"  {cont['R_ens vs P3s | nine']['pct']:+.2f}%, plain continuation {cont['F_ens vs P3s | nine']['pct']:+.2f}%, proportional validation "
         f"{cont['V_ens vs P3s | nine']['pct']:+.2f}%.",
         "- These test windows had been read before; the reading gives directions, not a confirmation.", ""]
    return L
