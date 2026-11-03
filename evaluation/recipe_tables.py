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

Version 0.2.0 (a products mapping with a configuration).  recipe_history.csv also holds the 0.2.0 study's two readings,
copied unrounded from their records: the development check (analysis_outputs/lite_upgrade_v2/dev_check/dev_check_v1.json,
development windows of six populations) and the one test read (analysis_outputs/lite_upgrade_v2/test_score/score.json,
the test windows of the nine populations; read many times before, so it sets directions only).  Only contrasts between
ankyra-lite's own configurations are copied (a record key or value naming another forecaster stops the copy); the
release-table comparison with the baselines inside score.json is not copied (the main tables hold it).  Two columns are
added: ``reading`` and ``seed_group_pcts``.  constants.csv gets the rows of the origin-continuity layer
(ankyra_lite.continuation) and of the inverse-noise loss weighting (ankyra_lite.carrier.train.NoiseWeights), and the seeds
of the configuration.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

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
V2 = CB.AO / "lite_upgrade_v2"
V2_SCORE = V2 / "test_score" / "score.json"                      # the 0.2.0 study's one test read
V2_DEV = V2 / "dev_check" / "dev_check_v1.json"                  # the 0.2.0 study's development check
V2_FIELDS = FIELDS + ["reading", "seed_group_pcts"]
V2_CODES = {"R3": "0.1.0 configuration: recipe R, seeds 0-2", "R3b": "recipe R, seeds 3-5", "R6": "recipe R, six seeds",
            "Rw3a": "recipe Rw, seeds 0-2", "Rw3b": "recipe Rw, seeds 3-5", "Rw6": "recipe Rw, six seeds"}
V2_DEV_CODES = {"R_g0": "R3", "R_g1": "R3b", "R_ens6": "R6", "Rw_g0": "Rw3a", "Rw_g1": "Rw3b", "Rw_ens6": "Rw6"}
V2_GROUPS = {"nine": "all nine populations, each with equal weight", "three without a development set": "BDG2 2017, GoiEner non-household, GoiEner households",
             "households": "GoiEner households", "six development populations": "Cambridge, HEEW Arizona, EWELD, Oslo, Drammen, CINELDI"}
V2_SKIP = ("decision", "implementation checks", "per population vs", "release table reproduction", "descriptive: the layer conditions")
READ_TEST = "0.2.0 test read (once; the test windows were read before: directions only)"
READ_DEV = "0.2.0 development check (development set two of six populations; one reading)"


def v2_code_words(code):
    """Plain words of a 0.2.0 study product code (R3, Rw6_c, ...)."""
    base, layer = (code[:-2], True) if code.endswith("_c") else (code, False)
    w = V2_CODES.get(base)
    if w is None:
        raise ValueError(f"unknown product code {code!r}")
    return w + (", with the continuation layer" if layer else "")


def _named_other(x):
    """True when a record node names a forecaster that is not one of ankyra-lite's configurations (the copy stops)."""
    t = json.dumps(x, ensure_ascii=False)
    return "ANKYRA" in t.upper().replace("ANKYRA-LITE", "").replace("ANKYRA 2.2.1", "")


def v2_rows(score=None, dev=None):
    """Rows of the 0.2.0 study's readings (development check, test read), copied unrounded, and a short record."""
    rows, rec = [], {}
    sp = Path(score) if score else V2_SCORE
    dp = Path(dev) if dev else V2_DEV
    if dp.exists():
        D = json.loads(dp.read_text(encoding="utf-8"))
        for key, segs in (D.get("B3d") or {}).items():
            if _named_other(key):
                continue
            if key.startswith("layer on "):
                c0 = V2_DEV_CODES.get(key[len("layer on "):].strip())
                key_a, key_b = (c0 + "_c", c0) if c0 else (None, None)
            elif " vs " in key:
                def code(x):
                    c1 = V2_DEV_CODES.get(x.replace(" + layer", "").replace(" (config)", "").strip())
                    return (c1 + "_c" if " + layer" in x else c1) if c1 else None
                a, b = key.split(" vs ", 1)
                key_a, key_b = code(a), code(b)
            else:
                continue
            if key_a is None or key_b is None:
                continue
            comp = f"{key_a} vs {key_b}" + (" (config)" if "(config)" in key else "")
            for q, v in segs.items():
                if not isinstance(v, dict) or "pct" not in v or _named_other(v):
                    continue
                lo, hi = v.get("lo_pct"), v.get("hi_pct")
                rows.append({"comparison": comp, "first_arm": key_a, "second_arm": key_b, "population_group": V2_GROUPS["six development populations"],
                             "populations": len(v.get("by_population_pct") or {}) or 6, "layer": "development windows", "quantity": q, "improvement_pct": v["pct"],
                             "interval_low_pct": lo, "interval_high_pct": hi, "interval": "unit x target-month bootstrap 95% (development check)",
                             "label": _label(lo, hi) if lo is not None else "", "seed0_pct": "", "seed1_pct": "", "seed2_pct": "", "unit_month_low_pct": lo,
                             "unit_month_high_pct": hi, "welch_df": "", "by_population_pct": _bypop(v.get("by_population_pct") or {}),
                             "first_arm_recipe": v2_code_words(key_a), "second_arm_recipe": v2_code_words(key_b), "reading": READ_DEV, "seed_group_pcts": ""})
        rec["development_check"] = {"source": CB.show(dp), "status": D.get("status"), "adopted": (D.get("decision") or {}).get("adopted")}
    if sp.exists():
        S = json.loads(sp.read_text(encoding="utf-8"))
        if S.get("status") != "complete":
            raise SystemExit(f"{sp.name}: the test read is not complete ({S.get('status')})")
        for block, groups in S.items():
            if not isinstance(groups, dict) or block.startswith(V2_SKIP) or _named_other(block) or not (
                    block.startswith(("primary", "decomposition", "layer on / off", "descriptive", "final configuration"))):
                continue
            for g, qs in groups.items():
                if not isinstance(qs, dict):
                    continue
                for q, v in qs.items():
                    if not isinstance(v, dict) or "a" not in v or _named_other(v):
                        continue
                    seeded = "z_95_pct" in v
                    lo, hi = (v["z_95_pct"] if seeded else v["unit_month_interval_pct"])
                    um = v.get("unit_month_interval_pct") or [None, None]
                    rows.append({"comparison": block, "first_arm": v["a"], "second_arm": v["b"], "population_group": V2_GROUPS.get(g, g),
                                 "populations": len(v.get("populations") or []), "layer": "all test windows", "quantity": q, "improvement_pct": v["pct"],
                                 "interval_low_pct": lo, "interval_high_pct": hi,
                                 "interval": ("z 95%: unit x target-month bootstrap and the spread of the two three-seed groups (decision interval of the reading)"
                                              if seeded else "unit x target-month bootstrap 95% (no seed layer)"),
                                 "label": v.get("label", ""), "seed0_pct": "", "seed1_pct": "", "seed2_pct": "", "unit_month_low_pct": um[0],
                                 "unit_month_high_pct": um[1], "welch_df": v.get("welch_df", ""), "by_population_pct": _bypop(v.get("by_population_pct") or {}),
                                 "first_arm_recipe": v2_code_words(v["a"]), "second_arm_recipe": v2_code_words(v["b"]), "reading": READ_TEST,
                                 "seed_group_pcts": "; ".join(repr(x) for x in v.get("group_pct", []))})
        dec = S.get("decision") or {}
        rec["test_read"] = {"source": CB.show(sp), "identity": S.get("identity"), "final": dec.get("final"),
                            "layer_rule_triggered": (dec.get("layer") or {}).get("triggered"),
                            "headline": {k: S[b_][g_][q_] for k, (b_, g_, q_) in {
                                "0.2.0 vs 0.1.0, nine, hourly": ("primary: 0.2.0 vs 0.1.0", "nine", "hourly"),
                                "0.2.0 vs 0.1.0, nine, lead day 1": ("primary: 0.2.0 vs 0.1.0", "nine", "lead day 1"),
                                "layer on / off, nine, hourly": ("layer on / off (Rw6 + layer vs Rw6; unit layer only)", "nine", "hourly"),
                                "weighting Rw6 vs R6, nine, hourly": ("decomposition: weighting (Rw6 vs R6)", "nine", "hourly"),
                                "seeds R6 vs R3, nine, hourly": ("decomposition: seeds (R6 vs R3)", "nine", "hourly")}.items()
                                if b_ in S and g_ in S[b_] and q_ in S[b_][g_]}}
    rec["rows"] = len(rows)
    return rows, rec
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


def carrier_constants(score_rec=None, prods=None):
    """The carrier's rows, read from the package (so that they follow its defaults); with a 0.2.0 mapping also the
    loss-weighting rows of recipe Rw and the configuration's seeds."""
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
        ("carrier: seeds", "seeds (forecast = plain average of the seeds' forecasts)",
         _fmt(tuple((prods or {}).get("seeds") or cfg.seeds)) if (prods or {}).get("configuration") else _fmt(cfg.seeds),
         ("the 0.2.0 study (six seeds averaged at every issue time; fixed before its development check)" if len((prods or {}).get("seeds") or ()) == 6 else DEV)
         if (prods or {}).get("configuration") else DEV, ""),
        ("carrier network", "parameters", f"{MD.parameter_count(mc):,}", NET, ""),
        ("carrier network", "daily / monthly / future-day tokens", f"{366} x {m.day_features} / 24 x {m.month_features} / {m.days} x {m.future_features}", NET, ""),
        ("carrier network", "width / attention heads / dropout", f"{m.width} / {m.heads} / {m.dropout:g}", NET, ""),
        ("carrier network", "dilations: daily tokens / monthly tokens (two shared gated causal blocks, kernel 3)", f"{_fmt(m.day_dilations)} / {_fmt(m.month_dilations)}", NET, ""),
        ("carrier network", "hourly dictionary rank / bound of the origin-conditioned change", f"{m.rank} / {m.dictionary_delta:g}", NET, ""),
        ("carrier network", "copy pointer: complete days attended / key dimension", f"{mc.copy_days} / {mc.key_dim}", NET, ""),
        ("carrier network", "scale: max(mean of the last 744 observed hours, mean of the last 8,760 observed hours)", "1 if not positive", NET, ""),
    ]
    if ((prods or {}).get("configuration_parts") or {}).get("recipe") == "Rw":
        nw = TR.NoiseWeights()
        RW = "the frozen inverse-noise rule of recipe Rw (the 0.2.0 study's record, fixed before its development check)"
        rows += [("carrier: loss weighting (recipe Rw)", "stage-2 loss weight per training window: max(sigma^2, floor)^-gamma, normalised to a draw-weighted mean of 1 within each source",
                  f"gamma {nw.gamma:g}", RW, ""),
                 ("carrier: loss weighting (recipe Rw)", "floor: draw-probability-weighted quantile of sigma^2 over the stage-2 pool", f"{nw.floor_quantile:g}", RW, ""),
                 ("carrier: loss weighting (recipe Rw)", "sigma^2: the window's own context day tokens (days, lag against the same hours earlier)",
                  f"days {nw.context_days[0]}..{nw.context_days[1] - 1}, lag {nw.lag_days} days", RW, ""),
                 ("carrier: loss weighting (recipe Rw)", "minimum observed pairs for sigma^2 (else the draw-weighted median raw weight)", f"{nw.min_pairs}", RW, ""),
                 ("carrier: loss weighting (recipe Rw)", "observation flag threshold", f"{nw.observed_threshold:g}", RW, "")]
    return [{"group": g, "constant": c, "value": val, "where_fixed": w, "sensitivity": s, "component": "carrier"} for g, c, val, w, s in rows]


def continuation_constants(v2rec=None):
    """The origin-continuity layer's rows (ankyra_lite.continuation), with the test read's layer on / off as sensitivity."""
    if str(CB.REPO) not in sys.path:
        sys.path.append(str(CB.REPO))
    from ankyra_lite import continuation as CT
    c = CT.CONSTANTS
    FIX = ("the 0.2.0 study's design on development windows (rho from the grids 2..12 h and 72, 120, 168 h; kappa fitted and rounded), "
           "frozen before its development check and test read")
    RULE = "the layer's rule (ankyra_lite.continuation.RULE, digest " + CT.DIGEST[:12] + "...), frozen with the constants"
    h = ((v2rec or {}).get("test_read") or {}).get("headline", {}).get("layer on / off, nine, hourly")
    sens = (f"layer on / off on the same carrier (test read, nine populations): {h['pct']:+.2f}% hourly, unit x month 95% "
            f"{h['unit_month_interval_pct'][0]:+.2f} to {h['unit_month_interval_pct'][1]:+.2f} ({h['label']}; recipe_history.csv)") if h else ""
    rows = [("continuation layer", "Delta(h) = kappa1 x_fast exp(-h / rho1) + kappa2 x_day exp(-h / rho2); F' = F exp(Delta)",
             f"kappa1 {c.kappa1:g}, rho1 {c.rho1:g} h, kappa2 {c.kappa2:g}, rho2 {c.rho2:g} h", FIX, sens),
            ("continuation layer", "x_fast: log ratio of the last observed hours to the same hours of the reference forecast day", f"{CT.FAST_HOURS} h", RULE, ""),
            ("continuation layer", "x_day: log ratio of the last observed day to the reference forecast day", f"{CT.DAY_HOURS} h", RULE, ""),
            ("continuation layer", "scale s (mean of the last hours, at least the floor) and e = fraction x s added to both sides of each ratio",
             f"{CT.SCALE_HOURS} h, floor {CT.SCALE_FLOOR:g} kW, fraction {CT.EPS_FRACTION:g}", RULE, ""),
            ("continuation layer", "clip of x_fast and x_day", f"+/- {CT.CLIP:g}", RULE, ""),
            ("continuation layer", "reference day: the first of forecast days 1-7 in the last observed day's workday class (day type < 5), else day",
             f"{CT.REFERENCE_DAYS} days, else day {CT.NO_MATCH_DAY + 1}; day types read at hour {CT.TYPE_HOUR}", RULE, ""),
            ("continuation layer", "bypass: off-state and micro-load windows (unchanged); an unobserved span, or a span whose mean + e is at or below 0, gets x = 0",
             "yes", RULE + "; the nonpositive-load rule is a release rule (no study window reached it)", ""),
            ("continuation layer", "readouts: energy, level and peak read before the layer (readout rule A)", "yes", RULE, "")]
    return [{"group": g, "constant": cc, "value": val, "where_fixed": w, "sensitivity": s_, "component": "continuation layer"} for g, cc, val, w, s_ in rows]


def anchoring_constants():
    """The first release's constants.csv as committed (the anchoring layer is a byte-identical copy of that state)."""
    txt = CB.published_text("constants.csv")
    out = []
    for r in csv.DictReader(io.StringIO(txt)):
        out.append({"group": r["group"], "constant": r["constant"], "value": r["value"], "where_fixed": r["where_fixed"],
                    "sensitivity": "", "component": "anchoring layer (unchanged copy; values and where_fixed as in the first release)"})
    return out


CONSTANT_FIELDS = ["group", "constant", "value", "where_fixed", "sensitivity", "component"]


def write(out_dir, log=print, prods=None):
    """Write both files; returns (written names, record, descriptions).  ``prods``: the products mapping (a 0.2.0 mapping
    adds the 0.2.0 study's readings and the layer's and the weighting's constants)."""
    written, files, rec = [], {}, {}
    v2 = bool((prods or {}).get("configuration"))
    v2rows, v2rec = (v2_rows() if v2 else ([], None))
    if CB.SR_SCORE.exists():
        rows, rrec = recipe_history()
        fields = V2_FIELDS if v2rows else FIELDS
        rows = [r | {"reading": "0.1.0 study reading of the recipe arms", "seed_group_pcts": ""} for r in rows] if v2rows else rows
        CB.write_rows("recipe_history.csv", [{k: ("" if r[k] is None else r[k]) for k in fields} for r in rows + v2rows], fields)
        written.append("recipe_history.csv")
        files["recipe_history.csv"] = ("the study's one reading of the recipe arms (from-scratch stage, F, R, V): ankyra-lite's version history"
                                       + ("; and the 0.2.0 study's development check and test read" if v2rows else ""))
        rec["recipe_history"] = rrec
        if v2rec is not None:
            rec["recipe_history_v020"] = v2rec
    else:
        rrec = None
        log("recipe history: the study's score record is missing; recipe_history.csv not written")
    crow = carrier_constants(rrec, prods)
    if ((prods or {}).get("continuation") or {}).get("enabled"):
        crow += continuation_constants(v2rec)
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
    v2 = ((rec or {}).get("recipe_history_v020") or {}).get("test_read")
    if v2:
        h = v2.get("headline", {})

        def c2(k):
            v = h.get(k)
            if not v:
                return "-"
            lo, hi = v.get("z_95_pct") or v["unit_month_interval_pct"]
            return f"{v['pct']:+.2f}% ({lo:+.2f} to {hi:+.2f}; {v['label']})"
        fin = v2.get("final") or {}
        L += ["### Version 0.2.0 (the 0.2.0 study's development check and one test read)", "",
              f"Configuration after the test read: `{fin.get('product')}` (weighting {fin.get('weighting')}, six seeds {fin.get('six_seeds')}, continuation "
              f"layer {fin.get('layer')}; after selection: {fin.get('after_selection')}). Positive favours the first-named configuration; the 0.2.0 against",
              "0.1.0 and the weighting rows have the seed layer of the two three-seed groups.", "",
              "| Readout (nine populations) | Result |", "|---|---|"]
        for k in ("0.2.0 vs 0.1.0, nine, hourly", "0.2.0 vs 0.1.0, nine, lead day 1", "layer on / off, nine, hourly", "weighting Rw6 vs R6, nine, hourly",
                  "seeds R6 vs R3, nine, hourly"):
            L.append(f"| {k.replace(', nine', '')} | {c2(k)} |")
        L += ["", "- The test windows had been read many times before; this reading sets directions only.", ""]
    return L
